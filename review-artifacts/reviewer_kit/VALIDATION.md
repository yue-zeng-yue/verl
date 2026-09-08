# Acceptance of the portable reviewer entry points

Date: 2026-09-08. Source: upstream `7cb65014d3a6c84f59458367df768999e4f36c67`
plus the unchanged feature patch. All nine patched files match the local review
checkout by SHA256. The original 100-step benchmark remains pinned to `c80729f`;
this acceptance is a separate run on another 2×A800 80GB PCIe host.

| Check | Result |
|---|---|
| Selected CPU configuration/Engine regressions | 58 passed |
| Repository pre-commit hooks on changed files | All passed |
| Portable launcher failure/path checks | 4 passed |
| Real two-rank small regression | All six AdamW/parameter-offload combinations passed |
| Fixed input, 128 tokens, six updates per arm | Full state fingerprints and input streams equal on both ranks |
| Actual GRPO, three steps per arm | Both passed; twelve global Adam updates per arm |
| GPU/Ray cleanup | No compute processes or Ray head processes remained after completion |

## Controlled fixed-input result

| Metric | Off | On |
|---|---:|---:|
| Actor peak allocated, max rank | 15.703 GiB | 12.472 GiB |
| Full actor window, slower rank, includes transfers/monitoring | 10.771 s | 13.910 s |

The memory reduction is accompanied by increased time in this controlled case.
This is a bounded correctness/memory check, not a steady throughput estimate.

## Short GRPO result

Both arms used the kit's default three-step recipe, two TP=1 rollout replicas,
GSM8K and the pinned Qwen2.5-1.5B-Instruct. Each arm consumed 24 training prompts,
generated 96 responses and completed twelve global Adam updates. All 24 rank-local
Adam records per arm had finite, nonzero gradients and changed parameter probes.
Initial and final greedy validation each used the 128 held-out examples. The
shared `main_ppo` entry point selected GRPO; no critic training was run.

The enabled arm's maximum post-step optimizer CUDA storage was **0 bytes**; the disabled arm retained 6174858568 bytes per observed rank inside the training context.

## Reproduction boundaries

- GPU acceptance used a clean source directory extracted from the current-main
  Git archive, applied the patch, and used an independent reviewer-kit folder.
  It reused the validated Python environment and downloaded model; a completely
  fresh dependency installation was not repeated in this acceptance.
- Fixed-input worker, GRPO observers, reward function and request-seeding loop
  are byte-identical to the original experiment scripts; see observer_provenance.json.
- The optional portable 100-step command has the same training hyperparameters
  as the original formal run. Only paths/interpreter and private Ray session
  setup differ. The portable 100-step command has not been rerun; the original
  two 100-step trials and both checkpoint-resume directions remain in the
  separate original evidence archive.
- After the GPU run, formatting, explicit check=False declarations and a stronger
  post-step residency summary were checked locally. No training kernel, recipe,
  observer or launch arguments changed. The final summarizer successfully
  re-analyzed the exported current-main results, including every optimizer step.
- Exported raw evidence was verified against remote SHA256. The final GRPO
  summary adds the stronger residency assertion to the original raw records.
- Each successful arm released its private Ray session. The dedicated GPU
  instance was shut down after all evidence was copied and checked.

Raw current-main logs, commands, summaries and source hashes are in `acceptance/`
inside the zip. `SHA256SUMS` covers every packaged file except the hash list
itself; run `sha256sum -c reviewer_kit/SHA256SUMS` from the extraction directory.
