# Auto Camera Geometry

Pixel ↔ Ray ↔ 3D camera infrastructure for ego / first-person dual-fisheye rigs.
Not an "undistortion" tool: undistortion is only one output (Virtual Pinhole View).
Raw images are never modified; geometry is derived and cached.

Implements the "Ego Camera Auto Geometry & Undistortion" technical spec (MVP scope, §48):

```
Raw Stereo Fisheye
      │
      ├────→ Native Rays        (pixel_to_ray for SLAM/3DGS/Depth/Hand/Object 3D)
      ├────→ Virtual Pinhole    (90°/100°/110° presets for VLM/VLA/CV)
      └────→ Rectified Stereo   (horizontal epipolar, Z = f·B/d)
```

## Status: MVP acceptance (spec §49) — verified on synthetic ground truth

| Criterion | Target | Measured |
|---|---|---|
| KB convention vs OpenCV fisheye | match | P99 projection diff < 1e-6 px (`tests/test_kb4.py`) |
| Projection round-trip | P99 < 0.05 px | ~2e-13 px |
| Resolution | 1600×1300 | ✓ |
| Baseline / transform verified | 0.12 m ± tolerance | ✓ (unit + integration tests) |
| Rectification epipolar residual | median < 0.5 px, P95 < 1 px | median 0.05 px, P95 0.22 px |
| 1000-frame-class sequence, fully automatic | no human input | ✓ (`process-dataset`) |

Run it yourself: `python scripts/synthetic_demo.py`

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # core (numpy/opencv/pydantic/yaml)
.venv/bin/pip install -e ".[torch]"                          # optional GPU batch backend
pytest                                                       # 41 tests
```

Python ≥ 3.11.

## Layout

```
src/camera_geometry/
├── schema.py               canonical calibration schema (camera-geometry/v1)
├── transforms.py           SE3 with explicit T_A_B convention
├── models/                 CameraModel API: base, kb4, pinhole, eucm,
│                           double_sphere, factory
├── calibration/            parser (units/vendor normalization), resolver,
│                           store (registry), validator (static QA),
│                           fitting (LM engine for AnyCalib/model selection)
├── geometry/               ray map + valid mask, midpoint triangulation
├── virtual_camera/         presets (ego_90/100/110), remap LUT construction
├── stereo/                 rectifier, native ray triangulation, SGBM depth
├── qa/                     schema/rotation/baseline/roundtrip/image-quality/
│                           epipolar/line checks + health score report
├── registry/               LUT cache keyed by hash(calibration+output)
├── runtime/                cv2.remap backend, torch grid_sample batch backend
├── fallback/               WORKING: stereo prior refinement (fixed-E
│                           signed-sampson + robust coordinate descent) and
│                           AnyCalib-style dense-ray model fitting/selection
├── testing/synthetic.py    exact ray-cast fisheye scene renderer
└── cli.py                  camera-geometry <command>
configs/device_orbbec_example.yaml          IR_L factory values from the device doc
scripts/synthetic_demo.py                   end-to-end demo without hardware
tests/                                      unit + property + golden-style + integration
```

## Conventions (non-negotiable)

* Units: **meter / nanosecond / radian / pixel**. Parser auto-converts `translation_mm`.
* Extrinsics named `T_A_B` ("B into A"): `P_A = R @ P_B + t`. The right-camera
  center in the left frame is exactly `t` (verified by unit test).
* Downstream code only calls `camera.project/unproject/pixel_to_ray/ray_to_pixel`
  — never touches fx/fy/k1..k4 directly.
* Canonical representation: raw image + CameraModel + extrinsics + timestamp.
* Calibration ID: `{serial}:{camera}:{WxH}:{version}`, e.g. `CH5LB5400J5:IR_L:1600x1300:v1`.

## CLI

```bash
camera-geometry inspect-camera      --config configs/device_orbbec_example.yaml
camera-geometry validate-calibration --config ... [--samples 100000] [--json-out qa.json]
camera-geometry generate-raymap     --config ... --out dir/     # fp16 ≈ 14 MB/camera
camera-geometry virtual             --config ... --image raw.png --out v.png --preset ego_100
camera-geometry rectify             --config ... --left l.png --right r.png --out dir/ [--depth]
camera-geometry process-dataset     --input episode/ --output episode/ --config ...
```

Exit codes: `0` PASS/DEGRADED, `1` FAIL/QUARANTINE, `2` usage error — automation-friendly.

## Python API

```python
from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera
from camera_geometry.pipeline import AutoGeometryPipeline
from camera_geometry.virtual_camera.presets import create_virtual_camera
from camera_geometry.virtual_camera.remap import create_remap_grid
from camera_geometry.runtime.torch_backend import make_sampling_grid, remap_batch

calib = load_device_calibration("device.yaml")
cam   = build_camera(calib.camera("IR_L"))
rays  = cam.pixel_to_ray(pixels)            # [...,2] -> [...,3] unit vectors
px    = cam.ray_to_pixel(rays)

vcam = create_virtual_camera("ego_100")     # 640x640, fx≈268.51
grid, valid = create_remap_grid(cam, vcam)  # cached per calibration
# runtime (GPU): images [B,C,1600,1300] -> [B,C,640,640]
sampling = make_sampling_grid(grid, cam.width, cam.height, device="cuda")
out = remap_batch(images_gpu, sampling)
```

Episode processing is fully automatic (spec §41–42):

```python
pipe = AutoGeometryPipeline(config_path="device.yaml", home="~/.cache/camera_geometry")
report = pipe.process_episode(input_dir=..., output_dir=...)
report.status  # PASS | DEGRADED | QUARANTINE (+ quarantine.json reason/stage)
```

Outputs follow spec §38: `metadata/`, `geometry/calibration_refs.json`,
`qa/report.json`, `derived/virtual_ego_100/`, `derived/stereo_rectified/`.
Ray maps / virtual grids / stereo grids are content-hash cached and reused across episodes.

## QA & health score

Static: schema rules, RᵀR≈I & det(R)=1, baseline range (default 0.10–0.14 m),
projection round-trip on 100k valid pixels (mean/median/P95/P99/max).
Per-frame sampling: blur/exposure/black/keypoints, row-guided epipolar vertical
error (median/P95/P99 + inlier ratio), line-straightness metric (LSD/Hough, non-gating).

`S = 0.15·schema + 0.20·projection + 0.25·stereo + 0.25·multiview + 0.15·temporal`;
unevaluated categories are excluded with renormalized weights and flagged.
PASS ≥ 0.90 · DEGRADED ≥ 0.75 · else FAIL → automatic FALLBACK → QUARANTINE
(`quarantine.json`: status/reason/failure_stage/metrics). No stage ever waits for a human.

## Engineering notes & deviations from the draft spec

1. **Package layout**: the spec's flat folders (`calibration/`, `registry/`, …)
   would pollute `sys.path`; everything lives under one installable package
   `camera_geometry.*` while preserving the module names.
2. **NumPy core + torch interop**: model math is NumPy; passing a `torch.Tensor`
   returns a `torch.Tensor`. Batched GPU resampling uses precomputed grids +
   `F.grid_sample(align_corners=True)` (validated to match cv2.remap).
3. **KB4 project at r→0**: spec's `clamp(r, 1e-8)` breaks exactness on the
   optical axis; we use `theta_d / max(r, eps)` with X·scale→0 handling — round-trip
   at image center is machine-precision instead of ~µpx-level biased.
4. **Extrinsics sign convention**: `-Rᵀt` (Basalt/Kalibr habit) contradicts this
   spec's `P_ref = R@P_self + t`; center-in-reference is exactly `t`. Caught by tests,
   fixed everywhere (rectifier, triangulation, fixtures).
5. **Epipolar QA matcher**: naive SIFT+kNN+F-RANSAC degenerates on repetitive
   texture (fake consensus sets). Final design: global ratio-test → rank by
   distinctiveness → MAD trim; validated by control experiments
   (aligned pair ⇒ 0.05 px, injected 4 px shift ⇒ detected 4 px ⇒ FAIL).
6. **Synthetic renderer**: planar homography warps are invalid under fisheye;
   the demo/test renderer ray-casts every pixel against wall planes using the
   cached ray map (exact w.r.t. the KB model, ~0.7 s/frame at 1600×1300).
7. **Health-score hard gate**: schema failures force FAIL regardless of average
   (a wrong fx must not be averaged away).

## Phase 0 checklist (real hardware)

1. Fill `configs/device_orbbec_example.yaml` IR_R intrinsics + real `R_left_right`,
   `t_left_right` from the vendor SDK (parser accepts mm).
2. Confirm Orbbec KB == OpenCV fisheye with `validate-calibration` plus the SDK
   cross-check pattern in `tests/test_kb4.py::test_matches_opencv_fisheye_convention`
   (store SDK outputs as `tests/golden/*.npz`).
3. Run `process-dataset` on one real episode; review `qa/report.json`.

## Roadmap (post-MVP)

Phase 4+: dataset DAG hardening, Prometheus metrics export.
## Service API, metrics, rolling shutter, golden tests

```bash
uvicorn camera_geometry.service:app --port 8300   # spec section 40
#  POST /camera/resolve | validate | ray-map | virtual-camera
#  POST /camera/stereo-rectify | reprocess      GET /camera/{id} | /metrics
```

Every calibration-bearing response contains
`calibration_id / camera_model / quality_status / artifact_uri`.
`GET /metrics` exposes the section-43 metrics in Prometheus text format
(also available programmatically via `camera_geometry.metrics`).

Rolling shutter (spec section 28) is implemented as a strict separation:
`RollingShutterModel` (t(y) = t0 + y/H·T_readout) + `RowAwareCamera`
(row-aware `pixel_to_ray_rs(uv, pose_provider)` and a row-aware undistortion
map given a pose provider from the VIO stack). KB parameters never fit
motion distortion.

Golden regression tests (spec section 44): `scripts/export_golden.py`
writes reference pixel<->ray sets; `tests/test_golden.py` gates every code
change against them. Synthetic golden ships in `tests/golden/synth_default`;
Phase 0 vendor truth = implement `orbbec_pixel_to_ray()` in the export
script and regenerate with `--source orbbec_sdk`.

## Fallback chain (implemented, automatic, no human input)

```
Factory QA fail (static or stereo-geometry)
   → Stage 1: stereo refinement with FIXED factory E
       L = Σ huber(signed sampson)² + λ·Σ((θ−θ₀)/scale)²
       (reference camera frozen; only the identifiable vertical-channel
        parameters fy/cy/k1 are refined — fx/cx are unobservable through a
        horizontal-baseline parity constraint and stay anchored)
   → Stage 2: AnyCalib-style model fitting from stereo-derived dense rays
       (KB4 / EUCM / Double Sphere / pinhole candidates, per-frame robust
        fits + median consensus + pooled polish, automatic model selection)
   → still below gate ⇒ QUARANTINE with reason (spec §42)
```

Verified on synthetic ground truth: corrupting IR_R (fy×1.03, cy+6, k1+0.03)
is detected via stereo geometry QA and recovered to sub-pixel parameters
(fy err 15.1 → 0.75, k1 err 0.03 → 0.0005); parity improves 6.3 px → ~1.5 px
on the synthetic-texture noise floor.

AnyCalib network integration: wrap any dense-ray predictor as
`DenseRayProvider = Callable[image] -> (pixels, rays)` and call
`anycalib_fit(provider, images=...)` — the fitting/consensus/selection
machinery is model-agnostic.

Identifiability findings (validated experimentally, encoded in the design):
* Monocular 2-view reprojection with free depths cannot identify intrinsics
  (depths absorb everything) — mono fallback needs a dense-ray provider.
* A free essential matrix absorbs smooth distortion errors — epipolar
  self-consistency with FREE E is degenerate.
* Fixed-E stereo parity constrains only the vertical channel for
  horizontal baselines; "fix R" vs "equivalently warp L" are
  indistinguishable → the reference camera is frozen by convention.

Rolling-shutter compensator (`t(y)=t₀+y/H·T_readout`) remains planned;
it needs IMU/readout metadata.
