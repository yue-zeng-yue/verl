"""Official single-turn loop with distinct, reproducible per-session seeds.

V1 TQ supplies session_id but does not supply a distinct priority. The stock
full_determinism path otherwise uses both request_id='det-0' and one shared
sampling seed for the n responses. Keep all tokenization and generation logic.
"""
import hashlib
import json
import os
from pathlib import Path

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class SeededSingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params, priority=0, **kwargs):
        step = kwargs.get('global_steps', 0)
        if hasattr(step, 'item'):
            step = step.item()
        session = int(kwargs.get('session_id', 0))
        messages = list(kwargs['raw_prompt'])
        payload = json.dumps([int(self.rollout_config.seed), int(step), session, messages],
                             ensure_ascii=False, sort_keys=True)
        identity = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') & ((1 << 63) - 1)
        params = dict(sampling_params, seed=identity)
        audit = os.environ.get('GRPO_AUDIT_DIR')
        if audit:
            root = Path(audit)
            root.mkdir(parents=True, exist_ok=True)
            with (root / f'sampling_{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='sampling_request', step=int(step), session=session,
                                       seed=identity, request_id=f'det-{identity}',
                                       prompt_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest())) + '\n')
        return await super().run(params, priority=identity, **kwargs)
