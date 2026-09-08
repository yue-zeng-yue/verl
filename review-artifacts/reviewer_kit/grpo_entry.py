"""Run the upstream Hydra entry point and clean up its private local Ray session."""

import json
import os
import runpy
import signal
import sys

import ray


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
sys.argv.append(
    "+ray_kwargs.ray_init._temp_dir=" + json.dumps(os.environ["VERL_REVIEW_RAY_TMP"])
)
try:
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
finally:
    ray.shutdown()
