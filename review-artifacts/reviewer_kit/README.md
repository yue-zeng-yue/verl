# Reviewer quick start: FSDP2 optimizer step offload

This companion kit runs the PR's small distributed regression, a controlled
full-model memory A/B, and an optional GRPO A/B. The production feature and small
regression tests are in `implementation.patch`; the measurement observers in
this kit are external to VERL. Original experiment results are in the separate
`VERL_A800_REVIEW_BUNDLE.zip` evidence archive.

## Downloads and verification

Download the [reviewer kit ZIP](../VERL_FSDP2_REVIEWER_KIT.zip?raw=1) (1.18 MB)
for the launchers, prepared data and implementation patch. The separate
[original experiment evidence ZIP](../VERL_A800_REVIEW_BUNDLE.zip?raw=1)
(15.98 MB) contains the recorded 100-step results and audits.
Verify the downloads against the [published SHA-256 checksums](../README.md#sha-256),
then unzip the reviewer kit before following the setup below.

## Environment and source

The validated environment is Linux x86_64, Python 3.12, PyTorch 2.11.0+cu130,
Transformers 5.9.0, vLLM 0.24.0 and FlashAttention 2.8.3. Use a CUDA 13 compatible
NVIDIA driver. Full-model experiments were tested with two A800 80GB GPUs,
28 CPU cores and 240GB host RAM; these are tested resources, not minimums.
The small `check` uses a tiny MLP and needs neither model weights nor a dataset.

Unzip this kit, then prepare a checkout and environment. The pinned source makes
the instructions independent of later main-branch changes. If reviewing an
actual PR checkout, use that checkout and skip applying the patch a second time.

```bash
export REVIEW_KIT="$(pwd)/reviewer_kit"
git clone https://github.com/verl-project/verl.git verl-review
git -C verl-review checkout 7cb65014d3a6c84f59458367df768999e4f36c67
git -C verl-review apply --check "$REVIEW_KIT/implementation.patch"
git -C verl-review apply "$REVIEW_KIT/implementation.patch"
export VERL_SOURCE="$(cd verl-review && pwd)"
cd "$VERL_SOURCE"
uv sync --frozen --extra fsdp --extra vllm --python 3.12
source .venv/bin/activate
# Recorded environment-only compatibility adjustment; does not change the PR.
uv pip install numpy==2.3.5 psutil nvidia-ml-py
uv pip check
```

The frozen lock's NumPy 2.4.6 conflicts with mistral-common's declared range;
the original experiments used 2.3.5. Use the activated `python` below, so a later
automatic `uv run` sync does not silently undo this adjustment. Dependency
downloads require Internet access. A successful `--dry-run` only constructs
commands; it is not a dependency or GPU check.

## 1. Small distributed correctness test

Choose exactly two idle CUDA devices. On a two-GPU container the following is
appropriate; on a larger machine select the desired indices or UUIDs.

```bash
export CUDA_VISIBLE_DEVICES=0,1
python "$REVIEW_KIT/review.py" check \
  --source "$VERL_SOURCE" --out "$PWD/review-check"
```

Expected: six `PASS world_size=2 ...` cases and `summary.json` with `status=PASS`.
This exercises ordinary/foreach/fused AdamW with parameter offload on/off, true
DTensor shards, gradient accumulation, skipped nonfinite updates, nested training
contexts and cross-option optimizer reload. It checks all local parameter and
optimizer tensors on both ranks. It is not a full-model memory benchmark.

The equivalent direct command from the patched checkout is:

```bash
torchrun --standalone --nproc-per-node=2 \
  tests/special_distributed/test_fsdp2_optimizer_offload_step.py
```

## 2. Controlled full-model memory A/B

Download the pinned model once (or reuse the same local snapshot). The kit does
not contain model weights. This recipe and its parameter probe target Qwen2.5;
arbitrary architectures are not claimed to work.

```bash
export REVIEW_MODEL="$PWD/review-model"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-1.5B-Instruct',
    revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    local_dir=os.environ['REVIEW_MODEL'],
)
PY
python "$REVIEW_KIT/review.py" fixed \
  --source "$VERL_SOURCE" --model "$REVIEW_MODEL" \
  --out "$PWD/review-fixed" --seq 128 --steps 6
```

Each arm runs the same seeded inputs through the real FSDP Engine, keeping one
outer training context across all updates. It uses the SFT loss to isolate
training memory and numerical equivalence; it is not an SFT quality experiment
or a GRPO simulation. Only step offload and the output directory differ.

The command compares full parameter, Adam, buffer, scheduler, parameter-group
and RNG fingerprints on both ranks and fails on any difference. It reports
peak allocated actor memory in bytes and the slower rank's complete window time
in seconds, including transfers and observer work. A smaller peak is an observed
result, not a hardcoded pass condition. The first update initializes Adam states;
at least two updates are required.

Original final-code A800 results at 128 tokens: 15.703→12.472 GiB with offload,
with a longer actor window. To repeat the bounded long-input case:

```bash
python "$REVIEW_KIT/review.py" fixed \
  --source "$VERL_SOURCE" --model "$REVIEW_MODEL" \
  --out "$PWD/review-long" --seq 8192 --steps 2 --order on-off
```

The original 8192-token check used only two updates. It does not measure steady
throughput, the OOM boundary, or convergence. Memory and timing vary by hardware.

## 3. Optional actual GRPO

`data/` contains the original prepared GSM8K train/test parquet subsets (1024/128
rows) and a manifest with their SHA256, source revision and selected row indices.
No dataset download is required. The upstream dataset is `openai/gsm8k` revision
`740312add88f781978c0658806c59bc2815b9866`. To audit preparation, the original
`prepare_data.py` and manifest are in the evidence archive.

```bash
# Short integration check: three GRPO steps per arm, twelve global Adam updates.
python "$REVIEW_KIT/review.py" grpo \
  --source "$VERL_SOURCE" --model "$REVIEW_MODEL" \
  --data "$REVIEW_KIT/data" --out "$PWD/review-grpo-smoke" --steps 3

# Optional reproduction of the original 100-step workload.
python "$REVIEW_KIT/review.py" grpo \
  --source "$VERL_SOURCE" --model "$REVIEW_MODEL" \
  --data "$REVIEW_KIT/data" --out "$PWD/review-grpo-100" --steps 100 \
  --save-freq 50 --test-freq 50 --audit-checkpoints
```

The recipe uses two-rank FSDP2 actor/reference, two TP=1 vLLM replicas, eight
prompts per step, four responses per prompt and two actor epochs. Each 100-step
arm consumes 800 prompts and performs 400 global Adam updates. It selects GRPO
with no critic training; `verl.trainer.main_ppo` is VERL's shared entry point.

Checkpoints are disabled by default, to keep the smoke small. Full checkpoint
runs need substantial disk space: allow at least 80GiB free for both arms and
temporary checkpoint rotation, in addition to environment and model storage.
The kit does not relocate or delete checkpoints to manage disk space. The
original 100-step arms took about 50–52 minutes each on the tested host, excluding
environment setup. Start with the short check if only assessing integration.

The original measurement and request-seeding observers are retained unchanged.
Generated outputs can nevertheless differ between arms. The summary records
actor memory, sampled whole-GPU memory, timings and actual update counts; neither
time differences nor validation-score differences prove speedup or quality gains.
The fixed-input check supplies the strict numerical comparison.

## Outputs, isolation and failures

- Every output directory must be new; existing results are never overwritten.
- `plan.json`, `preflight.json`, per-arm `command.json`, `train.log`, `exit.json`
  and `summary.json` retain commands, versions, selected GPUs, source identity,
  measurements and pass/fail evidence. No credentials are recorded.
- `--dry-run` prints commands without starting CUDA, training or creating output.
- Each GRPO arm starts a private local Ray session and calls `ray.shutdown()` in
  `finally`. It neither connects to `RAY_ADDRESS` nor runs machine-wide `ray stop`.
- `--timeout-seconds` bounds each arm. Failed logs remain and the pair stops on
  failure. Do not interpret an incomplete directory as a successful result.
- These launchers parameterize paths and preserve the original recipe. They do
  not install software, rent GPUs, upload results, or shut down the machine.

See `VALIDATION.md` for which portable entry points have actually been rerun.
The local CPU-only launcher checks can be run with:

```bash
python -m unittest discover -s "$REVIEW_KIT" -p 'test_launcher.py' -v
```
