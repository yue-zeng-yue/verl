"""CPU-only checks for launcher failure handling; no CUDA packages required."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from common import child_environment, run_logged

HERE = Path(__file__).resolve().parent


class LauncherTests(unittest.TestCase):
    def test_dry_run_preserves_paths_without_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "new output"
            model = Path(folder) / "model with spaces"
            result = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "review.py"),
                    "fixed",
                    "--source",
                    folder,
                    "--model",
                    str(model),
                    "--out",
                    str(out),
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            command = json.loads(result.stdout)["off"]
            self.assertEqual(
                command[command.index("--model") + 1], str(model.resolve())
            )
            self.assertFalse(out.exists())

    def test_grpo_dry_run_quotes_hydra_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "new output"
            model = Path(folder) / "model with spaces,commas"
            result = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_grpo.py"),
                    "--source",
                    folder,
                    "--model",
                    str(model),
                    "--data",
                    folder,
                    "--out",
                    str(out),
                    "--name",
                    "off",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            command = json.loads(result.stdout)["command"]
            model_arg = next(
                c for c in command if c.startswith("actor_rollout_ref.model.path=")
            )
            self.assertEqual(
                json.loads(model_arg.split("=", 1)[1]), str(model.resolve())
            )
            self.assertIn("+ray_kwargs.ray_init.address=local", command)
            self.assertFalse(out.exists())

    def test_child_failure_retains_exit_and_log(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            code = run_logged(
                [
                    sys.executable,
                    "-c",
                    'print("expected failure"); raise SystemExit(7)',
                ],
                folder,
                child_environment(folder),
                out,
                10,
            )
            self.assertEqual(code, 7)
            self.assertEqual(
                json.loads((out / "exit.json").read_text())["exit_code"], 7
            )
            self.assertIn("expected failure", (out / "train.log").read_text())

    @unittest.skipUnless(os.name == "posix", "process group timeouts require POSIX")
    def test_timeout_does_not_stop_unrelated_process(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            with tempfile.TemporaryDirectory() as folder:
                code = run_logged(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    folder,
                    child_environment(folder),
                    Path(folder),
                    0.2,
                )
                self.assertEqual(code, 124)
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait()


if __name__ == "__main__":
    unittest.main()
