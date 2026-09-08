"""Portable entry points for the FSDP2 step-offload review; see README.md."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import child_environment, preflight, run_logged

HERE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "fixed", "grpo"])
    p.add_argument("--source", required=True, type=Path, help="Patched VERL checkout")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="New output directory; never overwritten",
    )
    p.add_argument("--model", type=Path, help="Local Qwen2.5-1.5B-Instruct snapshot")
    p.add_argument(
        "--data",
        type=Path,
        default=HERE / "data",
        help="Prepared GSM8K parquet directory",
    )
    p.add_argument("--steps", type=int, help="Defaults: fixed=6, grpo=3; full GRPO=100")
    p.add_argument("--seq", type=int, default=128, help="Fixed-input sequence length")
    p.add_argument("--order", choices=["off-on", "on-off"], default="off-on")
    p.add_argument("--timeout-seconds", type=int, default=10800, help="Timeout per arm")
    p.add_argument(
        "--save-freq",
        type=int,
        default=-1,
        help="GRPO checkpoint interval; -1 disables",
    )
    p.add_argument("--test-freq", type=int, default=50, help="GRPO validation interval")
    p.add_argument(
        "--val-before-train", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--audit-checkpoints", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands; no CUDA imports or output writes",
    )
    return p


def fixed_summary(root, steps, seq):
    arms = {}
    for name in ("off", "on"):
        summaries = []
        for rank in (0, 1):
            directory = root / name / f"rank_{rank}"
            result = json.loads((directory / "summary.json").read_text())
            if (
                result["status"],
                result["updates"],
                result["world_size"],
                result["sequence_length"],
            ) != ("PASS", steps, 2, seq):
                raise RuntimeError(f"Incomplete fixed-input result: {directory}")
            if result["lazy"] != (name == "on") or not result["outer_context"]:
                raise RuntimeError(f"Wrong offload configuration: {directory}")
            summaries.append(result)
        arms[name] = {
            "peak_allocated_bytes": max(
                s["full_window_peak_allocated"] for s in summaries
            ),
            "actor_window_seconds": max(
                s["full_window_seconds_including_monitoring"] for s in summaries
            ),
        }
    for rank in (0, 1):

        def read(name, filename, rank=rank):
            return json.loads((root / name / f"rank_{rank}" / filename).read_text())

        if read("off", "final_fingerprint.json") != read(
            "on", "final_fingerprint.json"
        ):
            raise RuntimeError(f"Full state differs on rank {rank}")
        off, on = read("off", "steps.json"), read("on", "steps.json")
        if len(off) != steps or len(on) != steps:
            raise RuntimeError(f"Missing updates on rank {rank}")
        if any(
            (a["input_sha256"], a["rng_draws"]) != (b["input_sha256"], b["rng_draws"])
            for a, b in zip(off, on, strict=True)
        ):
            raise RuntimeError(f"Fixed inputs differ on rank {rank}")
    return {
        "status": "PASS",
        "mode": "fixed",
        "updates_per_arm": steps,
        "sequence_length": seq,
        "full_state_equal_on_both_ranks": True,
        "arms": arms,
        "peak_saved_bytes": arms["off"]["peak_allocated_bytes"]
        - arms["on"]["peak_allocated_bytes"],
        "note": "Actor window includes state transfers and observer work. Not end-to-end GRPO throughput.",
    }


def grpo_summary(root, steps):
    from summarize_grpo import summarize

    arms = {}
    for name in ("off", "on"):
        result, updates = summarize(root / name)
        if (
            result["exit_code"] != 0
            or result["actor_updates"] != steps
            or result["optimizer_steps"] != steps * 4
        ):
            raise RuntimeError(f"Incomplete GRPO run: {name}")
        if (
            not result["finite_grad_steps"]
            or not result["changed_probe_steps"]
            or not result["nonzero_grad_steps"]
        ):
            raise RuntimeError(f"Invalid GRPO parameter updates: {name}")
        if result["lazy_forward_cuda_state_violations"] or any(
            u["lazy"] != (name == "on") for u in updates
        ):
            raise RuntimeError(
                f"Incorrect optimizer residency or configuration: {name}"
            )
        for rank in ("0", "1"):
            if result["per_rank"][rank]["optimizer_steps"] != steps * 4:
                raise RuntimeError(f"Missing optimizer updates: {name}, rank {rank}")
        optimizer_rows = [
            json.loads(line)
            for file in (root / name / "audit").glob("*.jsonl")
            for line in file.read_text().splitlines()
            if line.strip() and json.loads(line).get("event") == "optimizer_step"
        ]
        if name == "on" and any(
            row["state_after"].get("cuda", 0) for row in optimizer_rows
        ):
            raise RuntimeError(
                "Optimizer state remained on CUDA after an enabled update"
            )
        result["optimizer_post_step_cuda_peak_bytes"] = max(
            row["state_after"].get("cuda", 0) for row in optimizer_rows
        )
        arms[name] = result
    return {
        "status": "PASS",
        "mode": "grpo",
        "arms": arms,
        "note": "Generated trajectories can differ. These timings do not isolate transfer overhead; "
        "no quality or speedup claim. Fixed-input tests check strict numerical equivalence.",
    }


def main():
    p = parser()
    a = p.parse_args()
    a.source, a.out, a.data = a.source.resolve(), a.out.resolve(), a.data.resolve()
    if a.model:
        a.model = a.model.resolve()
    a.steps = a.steps if a.steps is not None else (6 if a.mode == "fixed" else 3)
    if a.steps < 1 or a.seq < 2 or a.timeout_seconds < 1:
        p.error("steps/timeout must be positive and seq must be at least 2")
    if a.mode != "check" and a.model is None:
        p.error("--model is required for fixed and grpo")
    if a.mode == "fixed" and a.steps < 2:
        p.error("Use at least two updates to measure initialized Adam states")
    if a.mode == "grpo" and a.steps > 256:
        p.error(
            "This bounded recipe has 1024 train rows and two epochs (256 steps maximum)"
        )
    commands = {}
    if a.mode == "check":
        commands["check"] = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(
                a.source
                / "tests/special_distributed/test_fsdp2_optimizer_offload_step.py"
            ),
        ]
    for name in [] if a.mode == "check" else a.order.split("-"):
        if a.mode == "fixed":
            commands[name] = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(HERE / "fixed_input.py"),
                "--model",
                str(a.model),
                "--out",
                str(a.out / name),
                "--seq",
                str(a.seq),
                "--steps",
                str(a.steps),
                "--outer-context",
                "--deterministic",
            ] + (["--lazy"] if name == "on" else [])
        else:
            commands[name] = [
                sys.executable,
                str(HERE / "run_grpo.py"),
                "--source",
                str(a.source),
                "--model",
                str(a.model),
                "--data",
                str(a.data),
                "--out",
                str(a.out / name),
                "--name",
                f"grpo_{name}",
                "--steps",
                str(a.steps),
                "--save-freq",
                str(a.save_freq),
                "--test-freq",
                str(a.test_freq),
                "--timeout-seconds",
                str(a.timeout_seconds),
                "--val-before-train" if a.val_before_train else "--no-val-before-train",
            ]
            commands[name] += ["--lazy"] if name == "on" else []
            commands[name] += ["--audit-checkpoints"] if a.audit_checkpoints else []
    if a.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if a.out.exists():
        p.error("--out already exists; choose a new directory")
    if a.mode != "check" and not (a.model / "config.json").is_file():
        p.error(
            "--model must contain a downloaded model snapshot including config.json"
        )
    if a.mode == "grpo" and not all(
        (a.data / f"{s}.parquet").is_file() for s in ("train", "test")
    ):
        p.error("--data must contain train.parquet and test.parquet")
    info = preflight(a.source)
    a.out.mkdir(parents=True)
    (a.out / "preflight.json").write_text(json.dumps(info, indent=2))
    (a.out / "plan.json").write_text(json.dumps(commands, indent=2))
    for name, command in commands.items():
        print(f"START {a.mode}/{name}: {a.out / name}", flush=True)
        if a.mode == "grpo":
            # The launcher owns its timeout, logs, monitor and private Ray session.
            code = subprocess.call(command)
        else:
            (a.out / name).mkdir()
            code = run_logged(
                command,
                a.source,
                child_environment(a.source),
                a.out / name,
                a.timeout_seconds,
            )
        if code:
            raise RuntimeError(f"{name} exited {code}; see {a.out / name}/train.log")
    if a.mode == "check":
        text = (a.out / "check/train.log").read_text()
        if (
            text.count("PASS world_size=2 ") != 6
            or "test_fsdp2_optimizer_offload_step passed" not in text
        ):
            raise RuntimeError("Distributed test did not report all six cases")
        result = {"status": "PASS", "mode": "check", "cases": 6}
    elif a.mode == "fixed":
        result = fixed_summary(a.out, a.steps, a.seq)
    else:
        result = grpo_summary(a.out, a.steps)
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
