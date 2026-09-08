"""Use VERL's existing flexible GSM8K numeric answer scorer; retain strict score."""
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    assert data_source == 'openai/gsm8k', data_source
    flexible = gsm8k_score(solution_str, ground_truth, method='flexible')
    strict = gsm8k_score(solution_str, ground_truth, method='strict')
    return {'score': float(flexible), 'acc': float(flexible), 'strict_format_acc': float(strict)}
