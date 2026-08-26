"""Runnable downstream adapter examples (spec sections 24/25/27).

Each example is deterministic and asserts its own sanity gate:
  slam_adapter.py    raw feature pixels -> bearings -> triangulation   (§24)
  render_for_3dgs.py gaussians -> raw fisheye pixels + ray map         (§25)
  handpose_3d.py     virtual-view detections -> 3D keypoints           (§27)
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script", ["slam_adapter.py", "render_for_3dgs.py", "handpose_3d.py"])
def test_example_runs_and_passes(script):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout
