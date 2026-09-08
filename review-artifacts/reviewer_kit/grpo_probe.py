"""Read-only GRPO observations; preserve all original training calculations."""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def install():
    if not os.environ.get('GRPO_AUDIT_DIR'):
        return
    import torch
    import psutil
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    if getattr(TrainingWorker, '_grpo_observed', False):
        return
    TrainingWorker._grpo_observed = True
    root = Path(os.environ['GRPO_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)

    def write(event, **values):
        row = dict(event=event, pid=os.getpid(), time=time.time(), rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else None, world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else None, **values)
        with (root/f'worker_{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(row, default=str)+'\n')

    def local(t):
        return t.to_local() if hasattr(t,'to_local') else t

    def state(engine):
        seen = set()
        result = {}
        if engine.optimizer is None:
            return result
        for values in engine.optimizer.state.values():
            for v in values.values():
                if not torch.is_tensor(v):
                    continue
                t = local(v)
                storage = t.untyped_storage()
                key = (str(t.device),storage.data_ptr(),storage.nbytes())
                if key not in seen:
                    result[t.device.type] = result.get(t.device.type,0)+storage.nbytes()
                    seen.add(key)
        return result

    def mem():
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                    host_rss=psutil.Process().memory_info().rss)

    def sample(engine):
        t = next(p for n,p in engine.module.named_parameters() if 'q_proj.weight' in n)
        return local(t).detach().flatten()[:4096].cpu().clone()

    def tensor_info(t):
        t = local(t)
        offsets = t.offsets().cpu().tolist() if getattr(t,'is_nested',False) else None
        t = t.values() if getattr(t,'is_nested',False) else t
        t = t.detach().cpu().contiguous()
        assert torch.isfinite(t).all()
        return dict(sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                    numel=t.numel(), nonzero=int(torch.count_nonzero(t)), shape=list(t.shape), offsets=offsets)

    original = TrainingWorker.train_mini_batch
    @functools.wraps(original)
    def train_mini(self, data):
        inputs = {k:tensor_info(data[k]) for k in ['input_ids','advantages','old_log_probs','ref_log_prob'] if k in data.keys()}
        before_param = sample(self.engine)
        torch.cuda.synchronize()
        before_state = state(self.engine)
        torch.cuda.reset_peak_memory_stats()
        before = mem()
        start = time.perf_counter()
        try:
            output = original(self,data)
        except BaseException as error:
            torch.cuda.synchronize()
            write('actor_update_failed',error=repr(error),seconds=time.perf_counter()-start,after=mem())
            raise
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        after = mem()
        after_state = state(self.engine)
        delta = float((sample(self.engine)-before_param).abs().sum())
        write('actor_update',lazy=bool(self.engine_config.optimizer_offload_step),seconds=seconds,
              before=before,after=after,state_before=before_state,state_after=after_state,
              probe_delta_l1=delta,inputs=inputs)
        return output
    TrainingWorker.train_mini_batch = train_mini

    old_step = FSDPEngine.optimizer_step
    @functools.wraps(old_step)
    def optimizer_step(self):
        before = sample(self)
        result = old_step(self)
        torch.cuda.synchronize()
        grad = float(local(result))
        delta = float((sample(self)-before).abs().sum())
        write('optimizer_step',grad_norm=grad,probe_delta_l1=delta,state_after=state(self))
        assert torch.isfinite(torch.tensor(grad)), grad
        return result
    FSDPEngine.optimizer_step = optimizer_step

    old_fb = FSDPEngine.forward_backward_batch
    @functools.wraps(old_fb)
    def forward_backward(self,*args,**kwargs):
        if self.optimizer is not None:
            devices = state(self)
            lazy = bool(self.engine_config.optimizer_offload_step)
            if lazy:
                assert devices.get('cuda',0)==0, devices
            write('forward_backward_state',lazy=lazy,bytes_by_device=devices)
        return old_fb(self,*args,**kwargs)
    FSDPEngine.forward_backward_batch = forward_backward
    if os.environ.get('GRPO_CHECKPOINT_AUDIT') == '1':
        import pickle
        import random
        import numpy as np

        def digest(t):
            t = local(t).detach().cpu().contiguous()
            return dict(shape=list(t.shape), dtype=str(t.dtype),
                        sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

        def full_fingerprint(engine):
            torch.cuda.synchronize()
            names = {id(p): name for name, p in engine.module.named_parameters()}
            result = dict(
                parameters={name: digest(p) for name,p in engine.module.named_parameters()},
                buffers={name: digest(p) for name,p in engine.module.named_buffers()},
                optimizer={names[id(p)]: {k: digest(v) if torch.is_tensor(v) else v for k,v in st.items()}
                           for p,st in engine.optimizer.state.items()},
                scheduler=engine.lr_scheduler.state_dict(),
                param_groups=[{k: [names[id(p)] for p in v] if k=='params' else v for k,v in group.items()}
                              for group in engine.optimizer.param_groups],
                rng=dict(torch=digest(torch.get_rng_state()),cuda=digest(torch.cuda.get_rng_state()),
                         python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                         numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()))
            return json.loads(json.dumps(result, default=str))

        old_save = FSDPEngine.save_checkpoint
        @functools.wraps(old_save)
        def save_checkpoint(self, local_path, *args, **kwargs):
            before = full_fingerprint(self)
            value = old_save(self, local_path, *args, **kwargs)
            after = full_fingerprint(self)
            assert before == after, 'checkpoint save changed complete engine state'
            (Path(local_path)/f'engine_fingerprint_rank_{torch.distributed.get_rank()}.json').write_text(json.dumps(after))
            write('checkpoint_save_verified',path=str(local_path),parameters=len(after['parameters']),
                  optimizer_parameters=len(after['optimizer']))
            return value
        FSDPEngine.save_checkpoint = save_checkpoint

        old_load = FSDPEngine.load_checkpoint
        @functools.wraps(old_load)
        def load_checkpoint(self, local_path, *args, **kwargs):
            expected = json.loads((Path(local_path)/f'engine_fingerprint_rank_{torch.distributed.get_rank()}.json').read_text())
            value = old_load(self, local_path, *args, **kwargs)
            actual = full_fingerprint(self)
            assert actual == expected, 'checkpoint restore differs in complete engine state'
            write('checkpoint_load_verified',path=str(local_path),parameters=len(actual['parameters']),
                  optimizer_parameters=len(actual['optimizer']))
            return value
        FSDPEngine.load_checkpoint = load_checkpoint

    write('observer_installed',engine_file=__import__('inspect').getfile(FSDPEngine))


install()
