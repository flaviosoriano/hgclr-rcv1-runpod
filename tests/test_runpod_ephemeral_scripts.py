"""Behavior tests for the no-Network-Volume, one-Pod-per-fold runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy/runpod-hgclr/scripts/run_ephemeral_fold.sh"


class EphemeralFoldScriptTests(unittest.TestCase):
    def _make_fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, str]]:
        self.assertTrue(SCRIPT.is_file(), f"missing ephemeral runner: {SCRIPT}")
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "source"
        deploy = source / "deploy/runpod-hgclr/scripts"
        deploy.mkdir(parents=True)
        copied_script = deploy / SCRIPT.name
        shutil.copy2(SCRIPT, copied_script)

        (deploy / "prepare_standard_rcv1.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ -n ${HF_TOKEN:-} ]] || exit 61\n"
            "printf 'prepare:%s\n' \"$FOLD\" >> \"$TRACE_FILE\"\n"
        )
        (deploy / "run_standard_fold.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ -z ${HF_TOKEN:-} ]] || exit 62\n"
            "printf 'run:%s\n' \"$FOLD\" >> \"$TRACE_FILE\"\n"
        )
        for path in deploy.iterdir():
            path.chmod(0o755)

        workspace = root / "workspace"
        workspace.mkdir()
        runtime = workspace / "hgclr-runtime.env"
        runtime.write_text(
            f"export WORKSPACE_ROOT={workspace}\n"
            f"export IMAGE_SOURCE={source}\n"
            "export IMAGE_HGCLR_REVISION=fixture\n"
        )
        trace = root / "trace.txt"
        env = os.environ | {
            "WORKSPACE_ROOT": str(workspace),
            "RUNTIME_ENV": str(runtime),
            "TRACE_FILE": str(trace),
        }
        return temporary, trace, env

    def test_runs_local_staging_then_selected_fold_without_passing_hf_token_to_fold(self) -> None:
        temporary, trace, env = self._make_fixture()
        with temporary:
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env | {"FOLD": "3", "HF_TOKEN": "fixture-secret"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(trace.read_text().splitlines(), ["prepare:3", "run:3"])

    def test_resumes_ready_local_staging_without_hf_token(self) -> None:
        temporary, trace, env = self._make_fixture()
        with temporary:
            data_dir = Path(env["WORKSPACE_ROOT"]) / "hgclr-shared" / "RCV1-103-H3"
            data_dir.mkdir(parents=True)
            (data_dir / "READY").write_text(
                "dataset=RCV1-103-H3\nimage_revision=fixture\nprepared_utc=fixture\n"
            )
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env | {"FOLD": "2"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(trace.read_text().splitlines(), ["run:2"])
            self.assertIn("reusing local RCV1", result.stdout)

    def test_requires_hf_token_before_starting_local_staging(self) -> None:
        temporary, trace, env = self._make_fixture()
        with temporary:
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env | {"FOLD": "0"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("HF_TOKEN", result.stderr)
            self.assertFalse(trace.exists())

    def test_rejects_fold_outside_zero_through_four_before_staging(self) -> None:
        temporary, trace, env = self._make_fixture()
        with temporary:
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env | {"FOLD": "5", "HF_TOKEN": "fixture-secret"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Invalid FOLD", result.stderr)
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
