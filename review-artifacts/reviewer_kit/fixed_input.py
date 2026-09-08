"""Bounded, real VERL FSDP2 Engine baseline; no lifecycle changes."""
import argparse
import math
import hashlib
import random
import pickle
import contextlib
import numpy as np
import dataclasses
import functools
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault('VERL_USE_EXTERNAL_PLUGINS', 'none')
os.environ.setdefault('VERL_DISABLE_FLASH_ATTN_CE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import psutil
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import sft_loss

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seq', type=int, default=128)
p.add_argument('--lazy', action='store_true')
p.add_argument('--steps', type=int, default=7)
p.add_argument('--snapshot', action='store_true')
p.add_argument('--check-lifecycle', action='store_true')
p.add_argument('--deterministic', action='store_true')
p.add_argument('--resume')
p.add_argument('--start-step', type=int, default=0)
p.add_argument('--save-at', type=int, default=0)
p.add_argument('--checkpoint')
p.add_argument('--outer-context', action='store_true')
a = p.parse_args()
rank=int(os.environ.get('RANK','0'))
world_size=int(os.environ.get('WORLD_SIZE','1'))
out = Path(a.out)/f'rank_{rank}'
out.mkdir(parents=True, exist_ok=True)
assert platform.system() == 'Linux' and torch.cuda.device_count() >= world_size
if a.deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
torch.manual_seed(5282)
torch.cuda.manual_seed_all(5282)
torch.set_num_threads(8)
dist.init_process_group('nccl')

mc = HFModelConfig(path=a.model, load_tokenizer=False, use_remove_padding=False,
                   enable_gradient_checkpointing=True, override_config={'attn_implementation': 'sdpa'},
                   use_fused_kernels=False, use_liger=False, trust_remote_code=False)
ec = FSDPEngineConfig(strategy='fsdp2', fsdp_size=world_size, param_offload=False,
                     optimizer_offload=True, offload_policy=False, use_torch_compile=False,
                     model_dtype='fp32', use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                     mixed_precision={'param_dtype': 'bf16', 'reduce_dtype': 'fp32', 'buffer_dtype': 'fp32'})
oc = FSDPOptimizerConfig(lr=1e-5, total_training_steps=64, lr_scheduler_type='cosine', clip_grad=1.0,
                       override_optimizer_config={'foreach': False, 'fused': False})
if a.lazy:
    ec = dataclasses.replace(ec, optimizer_offload_step=True)
engine = FSDPEngineWithLMHead(mc, ec, oc, CheckpointConfig())
engine.initialize()
assert type(engine.optimizer) is torch.optim.AdamW

def local(t):
    return t.to_local() if isinstance(t, DTensor) else t

def states():
    seen = set()
    sizes = {}
    dtypes = set()
    for state in engine.optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            t = local(value)
            storage = t.untyped_storage()
            key = (str(t.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                sizes[t.device.type] = sizes.get(t.device.type, 0) + storage.nbytes()
                seen.add(key)
            dtypes.add(str(t.dtype))
    return {'bytes_by_device': sizes, 'dtypes': sorted(dtypes)}

def mem():
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
            'host_rss': psutil.Process().memory_info().rss}

params = [x for x in engine.module.parameters() if x.requires_grad]
parameter_count = sum(local(x).numel() for x in params)
expected_state_bytes = parameter_count*8 + len(params)*4
# Copy only a small, non-embedding parameter to CPU outside measurement intervals.
probe = next(x for n, x in engine.module.named_parameters() if 'q_proj.weight' in n)
def sample():
    return local(probe).detach().flatten()[:4096].cpu().clone()

ids = torch.randint(10, mc.hf_config.vocab_size - 1, (2, a.seq))
mask = torch.ones_like(ids, dtype=torch.float32)
mask[:, 0] = 0
positions = torch.arange(a.seq).repeat(2, 1)
def nested(t):
    return torch.nested.as_nested_tensor(list(t.unbind()), layout=torch.jagged)
data = TensorDict({'input_ids': nested(ids), 'position_ids': nested(positions), 'loss_mask': nested(mask)}, batch_size=[2])
tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                     use_remove_padding=False, use_fused_kernels=False, temperature=1.0,
                     calculate_entropy=False, global_batch_size=2*world_size, pad_mode='no_padding')
torch.save({'input_ids': ids, 'position_ids': positions, 'loss_mask': mask}, out / 'inputs.pt')
loss_fn = functools.partial(sft_loss, config=None)
# Correctness tests use a changing batch and exercise all four RNG sources.
random.seed(5282)
np.random.seed(5282)

def tensor_fingerprint(t):
    t = local(t).detach().cpu().contiguous()
    assert torch.isfinite(t).all(), 'nonfinite full tensor'
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'numel': t.numel(),
            'sha256': hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

def fingerprint():
    result = {'parameters': {}, 'optimizer': {}, 'buffers': {}}
    for name, param in engine.module.named_parameters():
        result['parameters'][name] = tensor_fingerprint(param)
        result['optimizer'][name] = {
            key: tensor_fingerprint(value) if torch.is_tensor(value) else value
            for key, value in engine.optimizer.state.get(param, {}).items()}
    for name, buffer in engine.module.named_buffers():
        result['buffers'][name] = tensor_fingerprint(buffer)
    result['scheduler'] = engine.lr_scheduler.state_dict()
    names = {id(param): name for name, param in engine.module.named_parameters()}
    result['param_groups'] = [{k: [names[id(x)] for x in v] if k == 'params' else v
                              for k, v in g.items()} for g in engine.optimizer.param_groups]
    result['rng'] = {
        'torch': tensor_fingerprint(torch.get_rng_state()),
        'cuda': [tensor_fingerprint(x) for x in torch.cuda.get_rng_state_all()],
        'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}
    return json.loads(json.dumps(result))

def next_batch():
    draws = [random.random(), float(np.random.random()), float(torch.rand((), device='cuda'))]
    ids = torch.randint(10, mc.hf_config.vocab_size - 1, (2, a.seq))
    mask = torch.ones_like(ids, dtype=torch.float32)
    mask[:, 0] = 0
    positions = torch.arange(a.seq).repeat(2, 1)
    batch = TensorDict({'input_ids': nested(ids), 'position_ids': nested(positions),
                        'loss_mask': nested(mask)}, batch_size=[2])
    tu.assign_non_tensor(batch, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                         use_remove_padding=False, use_fused_kernels=False, temperature=1.0,
                         calculate_entropy=False, global_batch_size=2*world_size, pad_mode='no_padding')
    return batch, tensor_fingerprint(ids)['sha256'], draws

if a.resume:
    # Perturb RNG first: equality must come from checkpoint restore, not reseeding.
    random.seed(9876)
    np.random.seed(9876)
    torch.manual_seed(9876)
    torch.cuda.manual_seed_all(9876)
    engine.load_checkpoint(a.resume, del_local_after_load=False)
    torch.cuda.synchronize()
    assert states()['bytes_by_device'] == {'cpu': expected_state_bytes}
    restored = fingerprint()
    saved = json.loads((Path(a.resume) / f'state_fingerprint_rank_{rank}.json').read_text())
    assert restored == saved, 'checkpoint state does not match saved state'
    (out / 'restore.json').write_text(json.dumps({'status': 'PASS', 'start_step': a.start_step,
                                                 'all_tensors_scheduler_groups_rng_match': True}))

records = []
try:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    context_start = time.perf_counter()
    initial = mem()
    outer = engine.train_mode() if a.outer_context else contextlib.nullcontext()
    with outer:
        for step in range(a.start_step, a.steps):
            batch, input_sha, draws = next_batch()
            before_sample = sample()
            before_state = states()
            torch.cuda.synchronize()
            before = mem()
            t0 = time.perf_counter()
            with (contextlib.nullcontext() if a.outer_context else engine.train_mode()):
                outputs = engine.train_batch(batch, loss_function=loss_fn)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            after = mem()
            engine.lr_scheduler_step()
            delta = float((sample() - before_sample).abs().sum())
            grad = float(outputs['metrics']['grad_norm'])
            assert math.isfinite(grad) and grad > 0 and delta > 0, (step, grad, delta)
            after_state = states()
            expected = 'cuda' if a.outer_context and not a.lazy else 'cpu'
            assert sum(after_state['bytes_by_device'].values()) == expected_state_bytes, after_state
            for optimizer_state in engine.optimizer.state.values():
                for moment in ('exp_avg', 'exp_avg_sq'):
                    assert local(optimizer_state[moment]).device.type == expected
            if expected == 'cpu':
                assert after_state['bytes_by_device'] == {'cpu': expected_state_bytes}, after_state
            records.append({'step': step + 1, 'seconds': elapsed, 'before': before, 'after': after,
                'loss': outputs['loss'], 'grad_norm': grad, 'probe_delta_l1': delta,
                'input_sha256': input_sha, 'rng_draws': draws, 'lr': engine.optimizer.param_groups[0]['lr'],
                'state_before': before_state, 'state_after': after_state})
            (out / 'steps.json').write_text(json.dumps(records, indent=2))
            print('RELIABILITY_STEP', step + 1, grad, elapsed, flush=True)
            if a.save_at == step + 1:
                assert a.checkpoint and not a.outer_context
                saved = fingerprint()
                engine.save_checkpoint(a.checkpoint, global_step=step + 1)
                torch.cuda.synchronize()
                # Saving must not mutate state or consume any RNG stream.
                assert fingerprint() == saved, 'save_checkpoint changed training state'
                assert states()['bytes_by_device'] == {'cpu': expected_state_bytes}
                (Path(a.checkpoint) / f'state_fingerprint_rank_{rank}.json').write_text(json.dumps(saved, indent=2))
                (out / 'save.json').write_text(json.dumps({'status': 'PASS', 'step': step + 1,
                                                           'state_unchanged_after_save': True}))
    torch.cuda.synchronize()
    context_seconds = time.perf_counter() - context_start
    final_memory = mem()
    assert states()['bytes_by_device'] == {'cpu': expected_state_bytes}
    (out / 'final_fingerprint.json').write_text(json.dumps(fingerprint(), indent=2))
    summary = {'engine_source': __import__('inspect').getfile(type(engine)), 'status': 'PASS', 'lazy': a.lazy, 'outer_context': a.outer_context,
               'sequence_length': a.seq, 'start_step': a.start_step, 'end_step': a.steps,
               'world_size': world_size, 'rank': rank, 'updates': len(records), 'full_window_seconds_including_monitoring': context_seconds,
               'full_window_peak_allocated': final_memory['peak_allocated'],
               'full_window_peak_reserved': final_memory['peak_reserved'],
               'initial_memory': initial, 'final_memory': final_memory,
               'note': 'No phase peak resets; full outer window includes monitoring/input generation and any checkpoint work.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    import yaml
    (out / 'config.yaml').write_text(yaml.safe_dump({'engine': dataclasses.asdict(ec),
        'optimizer': dataclasses.asdict(oc), 'args': vars(a)}))
    print('RELIABILITY_PASS', json.dumps(summary), flush=True)
finally:
    dist.destroy_process_group()
