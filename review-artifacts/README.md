# FSDP2 optimizer step offload: reviewer material

Companion material for the opt-in `optimizer_offload_step` change, related to
[verl-project/verl#5282](https://github.com/verl-project/verl/issues/5282).
The feature commit is
[`4d7e9a3`](https://github.com/yue-zeng-yue/verl/commit/4d7e9a3a265ba652c5ba095f98cdf54b142887d8),
based on upstream main `7cb65014d3a6c84f59458367df768999e4f36c67`.
This separate branch holds benchmark material; it is not part of the upstream
feature diff.

## Start here

Read the [PR preview with inline experiment figures](PR_PREVIEW.md), or open the
[figure gallery, plotted values and regeneration script](figures/README.md).
The figures replot the existing experiments; no new GPU runs are claimed.

The [single-GPU supplement](single_gpu/README.md) adds the historical RTX 5090 D
prototype's 20-step GRPO result and a separate single-A800 case with little
memory benefit. It includes an additional figure, source data and a compact
evidence archive, with explicit version/workload labels and runtime costs.

1. Download [VERL_FSDP2_REVIEWER_KIT.zip](VERL_FSDP2_REVIEWER_KIT.zip?raw=1)
   (1,179,268 bytes), verify its checksum below, and unzip it.
2. Follow the [reviewer quick start](reviewer_kit/README.md) to prepare the pinned
   environment and source. Use the actual feature checkout instead of applying
   the included patch twice.
3. Run the small two-GPU regression first. The fixed-input A/B and optional GRPO
   A/B commands, pinned model revision, bundled dataset, expected output, and
   resource requirements are in that README.

The kit contains source scripts, prepared GSM8K subsets with their license and
provenance, the implementation patch, and current-main acceptance logs. Its
[validation record](reviewer_kit/VALIDATION.md) states exactly what was run:
six distributed cases, a six-update fixed-input pair, and a three-step GRPO pair
on two A800 80GB GPUs. The portable 100-step recipe was not rerun. The acceptance
reused a validated environment and model; it does not attest to a fresh install.

## Original benchmark evidence

[VERL_A800_REVIEW_BUNDLE.zip](VERL_A800_REVIEW_BUNDLE.zip?raw=1)
(15,982,151 bytes) is the immutable original archive: scripts, configuration,
100-step GRPO logs, fixed-input checks, checkpoint-resume audit records, and
offline analysis. It contains neither model weights nor full checkpoint shards.
It predates the portable kit, and its archived PR draft is historical.

Both original GRPO arms used Qwen2.5-1.5B-Instruct and GSM8K, with two A800 80GB
PCIe GPUs. Each completed 100 steps, 400 global Adam updates, and 800 prompts
with four responses per prompt. The measured actor peak was 14.693/12.372 GiB
off/on; sampled whole-GPU peak was 21.621/21.623 GiB. Generated trajectories
differed, so this is not a controlled throughput or quality comparison. The
fixed-input checks provide the strict state-equivalence comparison. See the
archive reports for limitations and the added transfer cost.

## SHA-256

```text
ca139eb822ece6bffe61c9fb280635ab1be1b12b63981906a1c142d3df410944  VERL_FSDP2_REVIEWER_KIT.zip
ccd2002d5554b11fe37c6f157da5820d796cc1d35b5505e0b13a9cc327d91190  VERL_A800_REVIEW_BUNDLE.zip
```

The source files under `reviewer_kit/` are identical to their copies in the kit
archive and are included here for browser review. Dataset and raw acceptance
records are in the downloadable archives. The kit's internal `SHA256SUMS` checks
individual files after extraction.

AI assistance: Codex was used for implementation, tests, experiment automation,
and writing. Recorded commands were executed by Codex; these records do not
represent an attestation that the submitting human personally ran them.
