import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliImportTests(unittest.TestCase):
    def test_stats_runs_when_torch_is_unavailable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = """
import builtins
import sys

orig_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
sys.modules.pop("llm.cli", None)
sys.modules.pop("llm.model", None)
sys.argv = ["llm.cli", "stats", "--input", "README.md"]
import llm.cli as cli
raise SystemExit(cli.main())
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("file=README.md", proc.stdout)


if __name__ == "__main__":
    unittest.main()

