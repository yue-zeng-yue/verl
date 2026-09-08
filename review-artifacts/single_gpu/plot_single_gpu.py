"""Plot single-GPU GRPO evidence; no new training runs or interpolation.

python plot_single_gpu.py --evidence extracted-evidence --out regenerated
python plot_single_gpu.py --data plot_data.json --out regenerated
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GIB = 2**30
COLORS = {"off": "#526783", "on": "#008575"}
LABELS = {"off": "Off", "on": "On"}


def extract(root):
    hashes = {}

    def read(path, kind="json"):
        b = (root / path).read_bytes()
        hashes[path] = hashlib.sha256(b).hexdigest()
        if kind == "jsonl":
            return [json.loads(line) for line in b.splitlines() if line.strip()]
        if kind == "csv":
            return list(csv.DictReader(b.decode().splitlines()))
        return json.loads(b)

    summary = read("grpo_5090/summary.json")
    metadata = read("METADATA.json")
    grpo = {}
    for arm in COLORS:
        folder = f"grpo_5090/results/grpo_{arm}"
        observations = [row for p in sorted((root / folder / "audit").glob("worker_*.jsonl"))
                        for row in read(p.relative_to(root).as_posix(), "jsonl")]
        actor = sorted((row for row in observations if row["event"] == "actor_update"), key=lambda row: row["time"])
        optimizer = [row for row in observations if row["event"] == "optimizer_step"]
        assert len(actor) == 20 and len(optimizer) == 80
        ref = summary["grpo_" + arm]
        memory = [row["after"]["peak_allocated"] for row in actor]
        seconds = [row["seconds"] for row in actor]
        samples = read(folder + "/gpu_samples.csv", "csv")
        nvml = max(int(row["memory_used_bytes"]) for row in samples)
        assert ref["exit_code"] == 0 and ref["finite_grad_steps"]
        assert ref["nonzero_grad_steps"] == ref["changed_probe_steps"] == 80
        assert max(memory) == ref["actor_window_peak_allocated_bytes"]
        assert statistics.median(seconds) == ref["actor_window_seconds_median"]
        assert sum(seconds) == ref["actor_window_seconds_sum"]
        assert nvml == ref["device_memory_sampled_peak_bytes"]
        grpo[arm] = {"steps": list(range(1, 21)), "actor_peak_bytes": memory,
                     "actor_seconds": seconds, "actor_median_seconds": statistics.median(seconds),
                     "nvml_peak_bytes": nvml, "nvml_samples": len(samples)}
    return {"metadata": metadata, "grpo_5090": grpo, "source_sha256": hashes}


def plot(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#A6AFBA", "text.color": "#172A3A",
                         "axes.labelcolor": "#253B4D", "xtick.color": "#425466",
                         "ytick.color": "#425466", "svg.hashsalt": "verl-5282-single-gpu"})
    rows = data["grpo_5090"]
    actor_saved = 100 * (1 - max(rows["on"]["actor_peak_bytes"]) / max(rows["off"]["actor_peak_bytes"]))
    nvml_saved = 100 * (1 - rows["on"]["nvml_peak_bytes"] / rows["off"]["nvml_peak_bytes"])
    time_change = 100 * (rows["on"]["actor_median_seconds"] / rows["off"]["actor_median_seconds"] - 1)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))
    fig.subplots_adjust(left=0.055, right=0.98, top=0.71, bottom=0.26, wspace=0.37)
    fig.text(0.055, 0.94, "Single-GPU GRPO: RTX 5090 D", fontsize=18, weight="bold", va="top")
    fig.text(0.055, 0.86, "Qwen2.5-0.5B-Instruct · GSM8K · 20 steps / 80 Adam updates per arm · base 23af6a7 · 2026-09-05", fontsize=11, color="#425466", va="top")
    for ax in axes:
        ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
        ax.set_axisbelow(True)
    for i, arm in enumerate(COLORS):
        row = rows[arm]
        axes[0].plot(row["steps"], [n / GIB for n in row["actor_peak_bytes"]], color=COLORS[arm], lw=1.8, label=LABELS[arm])
        values = [max(row["actor_peak_bytes"]) / GIB, row["nvml_peak_bytes"] / GIB]
        bars = axes[1].bar([i * 0.32, 1 + i * 0.32], values, width=0.29, color=COLORS[arm])
        axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
        bars = axes[2].bar([i], [row["actor_median_seconds"]], width=0.55, color=COLORS[arm])
        axes[2].bar_label(bars, labels=[f"{row['actor_median_seconds']:.3f}"], padding=4, fontsize=10)
    axes[0].set(title="Actor peak at every GRPO step", xlabel="GRPO step", ylabel="Allocated GiB", xlim=(0, 21), ylim=(0, 14), xticks=[0, 5, 10, 15, 20])
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[1].set(title="Actor and sampled whole-GPU peaks", ylabel="GiB", xticks=[0.16, 1.16], xticklabels=["Actor allocated", "Whole GPU"], ylim=(0, 18.5))
    axes[2].set(title=f"Actor median time: +{time_change:.2f}%", ylabel="Seconds / actor window", xticks=[0, 1], xticklabels=["Off", "On"], ylim=(0, 18.5))
    fig.text(0.055, 0.135, f"Own-baseline changes: actor peak −{actor_saved:.2f}%; whole-GPU peak −{nvml_saved:.2f}%. Actor median uses all 20 steps, including warmup.", fontsize=10, color="#425466", va="top")
    fig.text(0.055, 0.08, "Tested code: 23af6a7 + experiment patch; differs from final PR. One trial per arm; generated trajectories differ.", fontsize=10, color="#425466", va="top")
    fig.savefig(out / "grpo_5090_historical.png", dpi=180, facecolor="white")
    fig.savefig(out / "grpo_5090_historical.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evidence", type=Path)
    group.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = extract(args.evidence) if args.evidence else json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plot_data.json").write_text(json.dumps(data, indent=2) + "\n")
    plot(data, args.out)
    print(json.dumps({"figure": "grpo_5090_historical", "source_files_checked": len(data["source_sha256"]), "actor_windows": "20 per arm", "optimizer_updates": "80 per arm", "raw_record_checks": "PASS"}))


if __name__ == "__main__":
    main()
