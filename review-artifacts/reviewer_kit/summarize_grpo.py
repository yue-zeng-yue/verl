"""Summarize exported observations without loading model checkpoints."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(path):
    exit_info = json.loads((path / 'exit.json').read_text())
    rows = [r for p in sorted((path / 'audit').glob('*.jsonl')) for r in read_rows(p)]
    updates = sorted((r for r in rows if r['event'] == 'actor_update'), key=lambda r: r['time'])
    steps = [r for r in rows if r['event'] == 'optimizer_step']
    forwards = [r for r in rows if r['event'] == 'forward_backward_state']
    metrics = read_rows(path / 'metrics.jsonl')
    with (path / 'gpu_samples.csv').open() as f:
        gpu = list(csv.DictReader(f))
    validation = {}
    for p in sorted((path / 'validation').glob('*.jsonl')):
        records = read_rows(p)
        validation[p.stem] = {
            'count': len(records),
            'mean_score': statistics.mean(r['score'] for r in records),
        }
    rollout_groups = []
    for p in sorted((path / 'rollouts').glob('*.jsonl')):
        groups = defaultdict(list)
        for row in read_rows(p):
            groups[row['input']].append(row)
        rollout_groups.extend(groups.values())
    by_rank = {rank: sorted((r for r in updates if r['rank']==rank),key=lambda r:r['time']) for rank in (0,1)}
    assert all(r['world_size']==2 for r in updates+steps), 'expected true dual-rank training'
    assert len(by_rank[0])==len(by_rank[1])
    actor_seconds=[max(a['seconds'],b['seconds']) for a,b in zip(by_rank[0],by_rank[1])]
    result = {
        'exit_code': exit_info['exit_code'],
        'wall_seconds': exit_info['end_unix'] - exit_info['start_unix'],
        'actor_updates': len(updates)//2, 'optimizer_steps': len(steps)//2,
        'actor_rank_records':len(updates), 'optimizer_rank_records':len(steps),
        'per_rank':{str(rank):{'actor_updates':len(by_rank[rank]),'optimizer_steps':sum(r['rank']==rank for r in steps),'peak_allocated_bytes':max((r['after']['peak_allocated'] for r in by_rank[rank]),default=0)} for rank in (0,1)},
        'nonzero_grad_steps': sum(r['grad_norm'] > 0 for r in steps),
        'changed_probe_steps': sum(r['probe_delta_l1'] > 0 for r in steps),
        'finite_grad_steps': all(math.isfinite(r['grad_norm']) for r in steps),
        'updates_with_nonzero_advantages': sum(r['inputs'].get('advantages', {}).get('nonzero', 0) > 0 for r in updates),
        'actor_window_peak_allocated_bytes': max((r['after']['peak_allocated'] for r in updates), default=0),
        'actor_window_peak_reserved_bytes': max((r['after']['peak_reserved'] for r in updates), default=0),
        'actor_window_seconds_sum': sum(actor_seconds),
        'actor_window_seconds_median': statistics.median(actor_seconds) if actor_seconds else None,
        'device_memory_sampled_peak_bytes': max((int(r['memory_used_bytes']) for r in gpu), default=0),
        'gpu_sample_count': len(gpu),
        'nvml_peak_by_uuid':{uuid:max(int(r['memory_used_bytes']) for r in gpu if r['gpu_uuid']==uuid) for uuid in {r['gpu_uuid'] for r in gpu}},
        'actor_seconds_steady_median':statistics.median(actor_seconds[5:]) if len(actor_seconds)>5 else None,
        'lazy_forward_cuda_state_violations': sum(r['lazy'] and r['bytes_by_device'].get('cuda', 0) > 0 for r in forwards),
        'max_optimizer_cuda_state_during_forward_bytes': max((r['bytes_by_device'].get('cuda', 0) for r in forwards), default=0),
        'max_optimizer_cpu_state_during_forward_bytes': max((r['bytes_by_device'].get('cpu', 0) for r in forwards), default=0),
        'cpu_optimizer_state_bytes_after_updates': sorted({r['state_after'].get('cpu', 0) for r in updates}),
        'cuda_optimizer_state_bytes_after_updates': sorted({r['state_after'].get('cuda', 0) for r in updates}),
        'validation': validation,
        'last_logged_step': max((r['step'] for r in metrics), default=None),
        'rollout_prompt_groups': len(rollout_groups),
        'groups_with_reward_variation': sum(len({r['score'] for r in group}) > 1 for group in rollout_groups),
        'groups_with_output_variation': sum(len({r['output'] for r in group}) > 1 for group in rollout_groups),
        'rollout_group_sizes': sorted({len(group) for group in rollout_groups}),
    }
    return result, updates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--names', nargs='+', default=['grpo_off', 'grpo_on'])
    args = parser.parse_args()
    summary, all_updates = {}, {}
    for name in args.names:
        summary[name], all_updates[name] = summarize(args.root / 'results' / name)
    if len(args.names) == 2:
        a, b = args.names
        # Arrival time can interleave ranks differently between runs. Pair by
        # rank and the rank-local update index instead of global timestamp.
        ua = sorted(all_updates[a], key=lambda r: (r['rank'], r['time']))
        ub = sorted(all_updates[b], key=lambda r: (r['rank'], r['time']))
        assert len(ua) == len(ub)
        assert all(x['rank'] == y['rank'] for x, y in zip(ua, ub))
        summary['comparison'] = {
            'paired_actor_update_count': min(len(ua), len(ub)),
            'exact_input_hash_matches': [x['inputs'] == y['inputs'] for x, y in zip(ua, ub)],
            'actor_peak_allocated_saved_bytes': summary[a]['actor_window_peak_allocated_bytes'] - summary[b]['actor_window_peak_allocated_bytes'],
            'actor_seconds_ratio': summary[b]['actor_window_seconds_sum'] / summary[a]['actor_window_seconds_sum'] if summary[a]['actor_window_seconds_sum'] else None,
            'note': 'Parameter probes are samples; matching probes do not prove full-state equality. NVML peaks are sampled, not continuous maxima.',
        }
    output = args.root / 'summary.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
