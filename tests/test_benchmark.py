"""Benchmark regression gates (loose, CI-safe): catches gross perf regressions."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark  # noqa: E402


def test_unproject_throughput():
    r = benchmark.bench_unproject(n=500_000, repeats=2)
    # LUT warm-started Newton: >2 M pts/s on any modern CPU
    assert r["M_pts_per_s"] > 2.0, r


def test_raymap_latency():
    r = benchmark.bench_raymap(repeats=1)
    assert r["raymap_1600x1300_s"] < 5.0, r


def test_remap_fps():
    r = benchmark.bench_remap(n_frames=20, repeats=1)
    assert r["remap_fps"] > 100.0, r
