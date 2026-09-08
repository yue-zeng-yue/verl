"""Derive and plot PR evidence without smoothing or synthesizing measurements.

Extract from local evidence:
  python plot_pr_figures.py --experiment-root ../.. --out .
Replot the published numerical data:
  python plot_pr_figures.py --data plot_data.json --out .
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GIB = 2**30
COLORS = {"off": "#526783", "on": "#008575"}
LABELS = {"off": "Off (default)", "on": "On (opt-in)"}


def extract(root):
    sources = {}

    def read(path, jsonl=False):
        content = (root / path).read_bytes()
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        if jsonl:
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        return json.loads(content)

    analysis = read(Path("analysis.json"))
    fixed = read(Path("reviewer_validation/fixed/summary.json"))
    assert fixed["status"] == "PASS" and fixed["full_state_equal_on_both_ranks"]
    assert fixed["updates_per_arm"] == 6 and fixed["sequence_length"] == 128
    for arm in COLORS:
        ranks = [read(Path(f"reviewer_validation/fixed/{arm}/rank_{r}/summary.json")) for r in (0, 1)]
        assert fixed["arms"][arm]["peak_allocated_bytes"] == max(r["full_window_peak_allocated"] for r in ranks)
        assert fixed["arms"][arm]["actor_window_seconds"] == max(r["full_window_seconds_including_monitoring"] for r in ranks)

    grpo = {}
    for arm in COLORS:
        name = "grpo_" + arm
        path = Path("results") / name
        ranks = defaultdict(list)
        for file in sorted((root / path / "audit").glob("worker_*.jsonl")):
            for row in read(file.relative_to(root), jsonl=True):
                if row["event"] == "actor_update":
                    ranks[row["rank"]].append(row)
        assert set(ranks) == {0, 1}
        for rows in ranks.values():
            rows.sort(key=lambda row: row["time"])
            assert len(rows) == 100
        memory = [max(a["after"]["peak_allocated"], b["after"]["peak_allocated"]) for a, b in zip(ranks[0], ranks[1])]
        seconds = [max(a["seconds"], b["seconds"]) for a, b in zip(ranks[0], ranks[1])]
        ref = analysis[name]
        assert max(memory) == ref["actor_window_peak_allocated_bytes"]
        assert statistics.median(seconds[5:]) == ref["actor_seconds_steady_median"]
        validation = []
        for step in (0, 50, 100):
            rows = read(path / "validation" / f"{step}.jsonl", jsonl=True)
            assert len(rows) == 128 and all(row["score"] in (0, 1) for row in rows)
            correct = sum(int(row["score"]) for row in rows)
            assert correct / 128 == ref["validation"][str(step)]["mean_score"]
            validation.append({"step": step, "correct": correct, "count": 128})
        grpo[arm] = {
            "steps": list(range(1, 101)),
            "actor_peak_allocated_bytes": memory,
            "actor_window_seconds": seconds,
            "actor_steady_median_seconds": statistics.median(seconds[5:]),
            "whole_run_sampled_nvml_peak_bytes": ref["device_memory_sampled_peak_bytes"],
            "response_tokens": ref["training_response_tokens_from_logged_mean"],
            "validation": validation,
        }
    return {
        "fixed": fixed, "grpo": grpo, "source_sha256": sources,
        "fixed_base": "7cb65014d3a6c84f59458367df768999e4f36c67",
        "original_grpo_base": "c80729f",
        "feature_commit": "4d7e9a3a265ba652c5ba095f98cdf54b142887d8",
        "note": "Separate experiments on two A800 80GB PCIe hosts. One trial per arm in each experiment. No confidence intervals or quality/speedup claims.",
    }


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A6AFBA", "text.color": "#172A3A",
        "axes.labelcolor": "#253B4D", "xtick.color": "#425466", "ytick.color": "#425466",
        "svg.hashsalt": "verl-5282-pr-figures",
    })


def canvas(title, subtitle, footer):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    fig.subplots_adjust(top=0.72, bottom=0.23, left=0.075, right=0.98, wspace=0.32)
    fig.text(0.075, 0.94, title, fontsize=17, weight="bold", va="top")
    fig.text(0.075, 0.86, subtitle, fontsize=10, va="top", color="#425466")
    fig.text(0.075, 0.095, footer, fontsize=9, va="top", linespacing=1.5, color="#425466")
    for ax in axes:
        ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{stem}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def plot(data, out):
    style()
    fixed = data["fixed"]["arms"]
    memory_change = 100 * (fixed["on"]["peak_allocated_bytes"] / fixed["off"]["peak_allocated_bytes"] - 1)
    time_change = 100 * (fixed["on"]["actor_window_seconds"] / fixed["off"]["actor_window_seconds"] - 1)
    fig, axes = canvas(
        "Fixed-input Engine A/B: lower memory, added transfer time",
        "2 × A800 80GB PCIe · Qwen2.5-1.5B-Instruct · 128 tokens · 6 updates per arm · base 7cb6501",
        "Complete state fingerprints equal on both ranks. Identical input streams; SFT loss isolates the Engine.\nOne bounded trial per arm; times include transfers and monitoring. This does not measure GRPO throughput.",
    )
    for ax, field, unit, title, change in [
        (axes[0], "peak_allocated_bytes", GIB, "Actor peak allocated memory", memory_change),
        (axes[1], "actor_window_seconds", 1, "Full six-update actor window", time_change),
    ]:
        vals = [fixed[arm][field] / unit for arm in COLORS]
        bars = ax.bar([0, 1], vals, width=0.52, color=list(COLORS.values()))
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=11)
        ax.set(xticks=[0, 1], xticklabels=list(LABELS.values()), ylim=(0, max(vals) * 1.28),
               ylabel="GiB / GPU (maximum rank)" if unit == GIB else "Seconds (slower rank)")
        ax.set_title(f"{title}\n{change:+.2f}% on vs off", pad=13)
    save(fig, out, "fixed_input_ab")

    grpo = data["grpo"]
    fig, axes = canvas(
        "100-step GRPO: actor memory and whole-GPU memory",
        "2 × A800 80GB PCIe · Qwen2.5-1.5B-Instruct · GSM8K · 100 steps / 400 Adam updates per arm · base c80729f",
        "Raw per-step curves, maximum of two ranks; line-chart memory axis starts at 12 GiB. Bars start at zero.\nOne trial per arm with different generated trajectories. Whole-GPU peaks are sampled and can be dominated by rollout.",
    )
    for i, arm in enumerate(COLORS):
        row = grpo[arm]
        axes[0].plot(row["steps"], [n / GIB for n in row["actor_peak_allocated_bytes"]], lw=1.7, color=COLORS[arm], label=LABELS[arm])
        vals = [max(row["actor_peak_allocated_bytes"]) / GIB, row["whole_run_sampled_nvml_peak_bytes"] / GIB]
        bars = axes[1].bar([i * 0.32, 1 + i * 0.32], vals, width=0.29, color=COLORS[arm])
        axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=10)
    reduction = 100 * (1 - max(grpo["on"]["actor_peak_allocated_bytes"]) / max(grpo["off"]["actor_peak_allocated_bytes"]))
    axes[0].set(title="Actor peak at every GRPO step", xlabel="GRPO step", ylabel="Allocated GiB / GPU", xlim=(0, 101), ylim=(12, 15.2))
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[1].set(title=f"Actor −{reduction:.2f}%; whole GPU essentially unchanged", ylabel="GiB / GPU", ylim=(0, 27), xticks=[0.16, 1.16], xticklabels=["Actor allocated peak", "Whole-run NVML peak"])
    save(fig, out, "grpo_memory")

    fig, axes = canvas(
        "100-step GRPO: runtime observations and held-out scores",
        "Same original GRPO pair · 800 prompts and 3,200 responses per arm · validation: 128 examples at steps 0 / 50 / 100",
        f"Different trajectories: {grpo['off']['response_tokens']:,} / {grpo['on']['response_tokens']:,} response tokens off/on. Timing does not isolate transfer overhead.\nOne trial per arm; validation differs already at step 0. Three evaluation points do not establish quality equivalence or improvement.",
    )
    for arm in COLORS:
        row = grpo[arm]
        axes[0].plot(row["steps"], row["actor_window_seconds"], color=COLORS[arm], lw=1.3,
                     label=f"{LABELS[arm]} · median {row['actor_steady_median_seconds']:.3f} s")
        val = row["validation"]
        xs = [v["step"] for v in val]
        ys = [100 * v["correct"] / v["count"] for v in val]
        axes[1].plot(xs, ys, color=COLORS[arm], lw=1.3, marker="o", ls="--", label=LABELS[arm])
        for x, y, v in zip(xs, ys, val):
            other = next(p for p in grpo["on" if arm == "off" else "off"]["validation"] if p["step"] == x)
            offset = 11 if v["correct"] > other["correct"] else -18
            axes[1].annotate(f"{v['correct']}/128", (x, y), xytext=(0, offset),
                             textcoords="offset points", ha="center", fontsize=9, color=COLORS[arm])
    axes[0].set(title="Actor window incl. transfers; median = steps 6–100", xlabel="GRPO step", ylabel="Seconds / actor window", xlim=(0, 101), ylim=(0, 17.5))
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].set(title="Greedy GSM8K validation (three measured points)", xlabel="GRPO step", ylabel="Accuracy (%)", xlim=(-8, 108), ylim=(0, 100), xticks=[0, 50, 100])
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, out, "grpo_runtime_validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-root", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = extract(args.experiment_root.resolve()) if args.experiment_root else json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plot_data.json").write_text(json.dumps(data, indent=2) + "\n")
    plot(data, args.out)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.suffix in {".png", ".svg", ".py", ".json"} and p.name != "figure_manifest.json"}
    (args.out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": 3, "source_files": len(data["source_sha256"]), "validation": "raw actor and validation records agree with recorded summaries"}))


if __name__ == "__main__":
    main()
