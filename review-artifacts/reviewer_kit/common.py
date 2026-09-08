"""Launch only this review's processes and record enough context to audit a run."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def child_environment(source):
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    env.pop("RAY_NAMESPACE", None)
    env.update(
        PYTHONPATH=str(source),
        VERL_USE_EXTERNAL_PLUGINS="none",
        VERL_DISABLE_FLASH_ATTN_CE="1",
        TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="8",
    )
    return env


def preflight(source):
    source = Path(source).resolve()
    if not (source / "verl/workers/engine/fsdp/transformer_impl.py").is_file():
        raise ValueError("--source must point to the patched VERL checkout")
    code = """
import dataclasses, importlib.metadata, json, platform
import torch
from verl.workers.config import FSDPEngineConfig
assert platform.system() == 'Linux', 'GPU checks require Linux'
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expose exactly two CUDA GPUs'
assert 'optimizer_offload_step' in {f.name for f in dataclasses.fields(FSDPEngineConfig)}, 'Apply implementation.patch first'
gpus = []
for i in range(2):
    p = torch.cuda.get_device_properties(i)
    gpus.append(dict(index=i, name=p.name, total_memory=p.total_memory, uuid=str(p.uuid)))
versions = {}
for package in ['torch', 'transformers', 'vllm', 'flash-attn', 'ray', 'numpy', 'pyarrow']:
    try: versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: versions[package] = None
print('REVIEW_PREFLIGHT_JSON=' + json.dumps(dict(gpus=gpus, versions=versions, cuda=torch.version.cuda)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=source,
        env=child_environment(source),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    info = json.loads(
        next(
            line.split("=", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("REVIEW_PREFLIGHT_JSON=")
        )
    )
    for gpu in info["gpus"]:
        if not gpu["uuid"].startswith("GPU-"):
            gpu["uuid"] = "GPU-" + gpu["uuid"]
    info["python"] = sys.executable
    info["source"] = str(source)
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["engine_sha256"] = hashlib.sha256(
        (source / "verl/workers/engine/fsdp/transformer_impl.py").read_bytes()
    ).hexdigest()
    rev = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    info["git_head"] = rev.stdout.strip() if rev.returncode == 0 else None
    return info


def run_logged(command, cwd, env, out, timeout):
    """Preserve failed logs; on timeout terminate this child process group only."""
    out = Path(out)
    safe_keys = [
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
        "VERL_USE_EXTERNAL_PLUGINS",
        "VERL_DISABLE_FLASH_ATTN_CE",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "WANDB_MODE",
        "GRPO_AUDIT_DIR",
        "GRPO_CHECKPOINT_AUDIT",
        "VERL_FILE_LOGGER_PATH",
        "VERL_REVIEW_RAY_TMP",
        "RAY_DEDUP_LOGS",
        "HYDRA_FULL_ERROR",
    ]
    (out / "command.json").write_text(
        json.dumps(
            {
                "command": list(map(str, command)),
                "cwd": str(cwd),
                "env": {k: env[k] for k in safe_keys if k in env},
            },
            indent=2,
        )
    )
    started = time.time()
    code = 1
    reason = None
    with (out / "train.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            reason = type(error).__name__
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        finally:
            (out / "exit.json").write_text(
                json.dumps(
                    {
                        "exit_code": code,
                        "reason": reason,
                        "start_unix": started,
                        "end_unix": time.time(),
                    },
                    indent=2,
                )
            )
    return code
