# Figures for the FSDP2 step-offload PR

These figures replot existing experiments. They do not represent additional GPU
runs. Each experiment has one trial per arm; no confidence intervals are shown.
Off is the default; On enables `optimizer_offload_step`.

## 1. Controlled fixed-input Engine A/B

![Controlled memory and runtime comparison](fixed_input_ab.png)

This uses the current-main reviewer acceptance, based on `7cb6501`, on two A800
80GB PCIe GPUs. The model is Qwen2.5-1.5B-Instruct, sequence length 128, six
updates per arm. Identical inputs and full state fingerprints match on both
ranks. The SFT loss isolates the Engine; this is not an end-to-end GRPO workload.
Peak allocated memory is the maximum rank; full-window time is the slower rank
and includes transfers and observer work. The measured memory change is
−20.58% and time change is +29.15%. Six updates do not establish steady throughput.

## 2. Original 100-step GRPO memory

![GRPO actor and whole-GPU memory](grpo_memory.png)

The original experiment uses base `c80729f` and a different two-A800 host from
the current-main acceptance. Each arm completed 100 GRPO steps / 400 global
Adam updates, consuming 800 GSM8K prompts with four responses each. Curves show
all 100 raw actor peaks, taking the maximum rank at each step. The line plot
starts at 12 GiB to show variation; the bar chart uses a zero baseline. Sampled
whole-GPU NVML peak measures a different scope from actor allocated memory.

## 3. Original GRPO runtime and held-out validation

![GRPO runtime and validation observations](grpo_runtime_validation.png)

Actor-window times include transfers and observer work. The displayed median
uses steps 6–100. Generated response lengths differ: 872,551 tokens off versus
693,670 on, so timing does not isolate transfer cost or prove a speedup.

Validation uses 128 held-out GSM8K examples. Only steps 0, 50 and 100 were
evaluated; dashed lines only connect those measured points. Accuracy is correct
count divided by 128, with a 0–100% axis. The initial scores already differ
because the generated outputs were not identical. These data do not establish
quality improvement, statistical equivalence, or convergence.

## Numerical provenance and regeneration

`plot_data.json` contains every plotted value and the SHA-256 of 24 source files.
The extraction checks actor curves against archived summary maxima/medians,
fixed-input values against both rank summaries, and validation counts against
the raw per-example score files. Original records are in the two companion ZIPs
linked from the [artifact index](../README.md).

To replot the published values (Python 3.12; generated with Matplotlib 3.11.1):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python matplotlib==3.11.1
.venv/bin/python plot_pr_figures.py --data plot_data.json --out regenerated
```

The output includes PNGs for the PR, SVGs for export, numerical data and a hash
manifest. To repeat extraction in the original evidence directory layout:

```bash
python plot_pr_figures.py --experiment-root /path/to/experiment_20260907_a800 --out regenerated
```

No smoothing, fabricated observations, error bars or inferred intermediate
validation measurements are added. Source code and generated figures were
prepared with Codex assistance.
