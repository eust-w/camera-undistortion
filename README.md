# Auto Camera Geometry

<p align="center">
  <b>Pixel ↔ Ray ↔ 3D camera infrastructure for ego fisheye stereo rigs</b>
</p>

<p align="center">
  <a href="https://github.com/eust-w/camera-undistortion/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/eust-w/camera-undistortion/ci.yml?branch=main&label=CI&labelColor=333b45&color=2ea44f"></a>
  <a href="#validation-on-real-data-tum-vi"><img alt="Tests" src="https://img.shields.io/badge/tests-79%20passed-2ea44f?labelColor=333b45"></a>
  <a href="https://github.com/eust-w/camera-undistortion/actions/workflows/ci.yml"><img alt="mypy" src="https://img.shields.io/badge/mypy-clean-2ea44f?labelColor=333b45"></a>
  <a href="https://github.com/eust-w/camera-undistortion"><img alt="models" src="https://img.shields.io/badge/models-KB4%C2%B7DS%C2%B7EUCM%C2%B7BC-blue?labelColor=333b45"></a>
  <img alt="python" src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab?labelColor=333b45">
  <a href="LICENSE"><img alt="Code license" src="https://img.shields.io/badge/license-Apache--2.0-orange?labelColor=333b45"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> |
  <a href="#downstream-adapters">Examples</a> |
  <a href="#validation-on-real-data-tum-vi">TUM-VI validation</a> |
  <a href="#service-api">Service API</a> |
  <a href="scripts/benchmark.py">Benchmarks</a>
</p>

<p align="center">
  English |
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

This is the end-to-end camera geometry infrastructure for dual-fisheye ego rigs
(Orbbec-style IR stereo): factory calibration is parsed, validated and health-scored
automatically; ray maps, virtual pinhole views, rectified stereo pairs and depth
are derived and cached with **zero human interaction** — no manual model choice,
no parameter tuning, and a full fallback chain (stereo refinement → AnyCalib →
quarantine) when the factory data fails QA.

**It is not an "undistortion" tool**: undistortion is only one output of the
canonical abstraction `pixel ↔ ray ↔ 3D`.

## Demo — real TUM-VI frames through the full pipeline

Raw fisheye pair (TUM-VI, official DoubleSphere calibration) → automatic QA →
virtual pinhole view + rectified stereo pair. All outputs produced by the
pipeline without any human input.

![demo](docs/demo_tumvi.png)

## Why

VLM/VLA pipelines want standard pinhole images; SLAM / 3DGS / depth want native
rays at full FOV; stereo depth wants rectified pairs. This project serves all
three from a single canonical representation:

```
Raw fisheye (never modified)
      │
      ├─→ Native rays        pixel_to_ray()  → SLAM / 3DGS / Hand-3D / Object-3D
      ├─→ Virtual pinhole    90° / 100° / 110° presets → VLM / VLA / 2D detectors
      └─→ Rectified stereo   horizontal epipolar, Z = f·B/d → depth
```

## Capability boundaries

Read this before integrating. The rules below are mathematical limits or
verified experiment results, not missing features.

| | Capability | Boundary |
|---|---|---|
| ✅ | Dual-fisheye stereo, factory-calibrated (KB4 / DS) | full pipeline: QA → rays → virtual views → rectified stereo → depth |
| ✅ | Mono video/image undistortion | requires that camera's factory calibration (a video file alone carries none) |
| ✅ | Virtual pinhole views | up to **110°** — rectilinear projection beyond ~120° is mathematically impossible (infinite edge stretch) |
| ✅ | Auto QA + fallback chain | vertical-channel intrinsics (fy / cy / k1) recoverable from fixed-E stereo parity |
| ⚠️ | Refinement of `fx` / `cx` | **not observable** through horizontal-baseline parity — stays anchored at factory values (needs plumb-line / multi-view additions) |
| ⚠️ | Common-mode bias on both cameras | indistinguishable from a single-camera error by stereo parity alone — resolved by convention (reference camera frozen), residual possible |
| ❌ | Mono self-calibration without any calibration | theoretically impossible in our experiments (free depths + free E absorb everything); a `DenseRayProvider` (e.g. AnyCalib network) is required |
| ❌ | Fisheye → rectilinear at fisheye's native FOV (>150°) | projection-impossible; use stereographic if you must keep full FOV (not yet in scope) |
| ⚠️ | Fallback precision floor | on synthetic low-texture data parity converges to ~1.5 px, not the 0.5 px PASS gate → QUARANTINE by design; richer real texture expected to do better |
| ⚠️ | Rolling shutter | model + row-aware API ready, but needs IMU/readout-time metadata from your VIO stack |
| ❌ | Feature frontend / SLAM backend / GS renderer themselves | this project provides the geometry layer only; examples show the adapter contracts |

Performance envelope (reference machine, M-series Mac): unproject ~8 M pts/s,
ray map 1600×1300 ≈ 0.7 s, remap ~390 fps CPU, 12-frame stereo episode cold
~12 s. Gates: 79 tests, golden regression at 1e-6°, mypy clean.

## Quick start

```bash
git clone https://github.com/eust-w/camera-undistortion.git
cd camera-undistortion
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest          # 79 tests: unit / property / integration / golden / examples
```

Undistort a video with a known factory calibration:

```bash
python scripts/undistort_video.py input.mp4 --preset ego_100 --size 1024
```

Run the full automatic episode DAG on a stereo image folder:

```bash
python -m camera_geometry process-dataset --input episode/ --output episode/ --config device.yaml
camera-geometry validate-calibration --config device.yaml     # QA + health score
camera-geometry inspect-camera       --config device.yaml     # parsed canonical view
```

Python API:

```python
from camera_geometry.calibration.parser import load_device_calibration
from camera_geometry.models.factory import build_camera

cam = build_camera(load_device_calibration("device.yaml").camera("IR_L"))
rays = cam.pixel_to_ray(pixels)   # [...,2] -> [...,3] unit bearings (full FOV)
px   = cam.ray_to_pixel(rays)     # back-projection
mask = cam.valid_mask()           # numerically valid region
```

## Validation on real data (TUM-VI)

Validated end-to-end on the public [TUM VI Benchmark](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
(RealSense-style global-shutter fisheye stereo) with the official basalt
DoubleSphere calibration:

| Check | Result |
|---|---|
| Projection round-trip (DS model) | machine precision |
| Automatic QA + health score | PASS (schema / rotation / baseline / round-trip) |
| Fallback chain | triggered on injected corruption & recovered automatically |
| Selective dataset fetch | 1.4 GB archive → ~16 MB transferred via range-request tar streaming |

Reproduce:

```bash
python scripts/fetch_tumvi_subset.py \
    --url https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar \
    --out third_party/tumvi/room1 --per-cam 40
python scripts/import_tumvi.py --root third_party/tumvi/room1 \
    --calib third_party/tumvi/tumvi_512_ds_calib.json --out outputs/tumvi_episode
python -m camera_geometry process-dataset --input outputs/tumvi_episode \
    --output outputs/tumvi_episode --config outputs/tumvi_episode/device.yaml
```

## Camera models

| Model | Status | Notes |
|---|---|---|
| Kannala-Brandt 4 | ✅ | theta-LUT Newton unprojection, ~8 M pts/s, machine-precision round-trip |
| Double Sphere | ✅ | closed-form unprojection incl. negative xi (official TUM-VI calib) |
| EUCM | ✅ | alpha ∈ [0,1), pinhole at 0 |
| Brown-Conrady | ✅ | plumb_bob, normal lenses |
| Pinhole | ✅ | virtual views / rectified cameras |

## Automatic calibration pipeline

```
factory calibration
        │ QA: schema · RᵀR≈I · baseline range · round-trip P99 · stereo parity
      PASS ────────────────► cache LUTs ► rays/views/stereo/depth
        │ FAIL
        ▼
Stage 1  stereo refinement (FIXED factory essential matrix, signed-sampson,
         robust coordinate descent + LM; identifiable-subset only;
         reference camera frozen — see identifiability notes in code)
        ▼ FAIL
Stage 1.5 external COLMAP SfM hook (optional binary)
        ▼ FAIL
Stage 2  AnyCalib-style fitting: dense-ray provider → per-frame robust fits,
         median consensus, model selection (KB4/EUCM/DS/pinhole)
        ▼ FAIL
QUARANTINE (reason + stage recorded; re-runnable, never silent)
```

Key identifiability findings encoded in the design (verified experimentally):
mono 2-view reprojection with free depths cannot identify intrinsics; a free
essential matrix absorbs smooth distortion; fixed-E stereo parity only sees the
vertical channel on horizontal baselines — so `fx/cx` stay anchored and the
reference camera is frozen by convention.

## Downstream adapters

```bash
python examples/slam_adapter.py     # raw feature pixels -> bearings -> triangulation
python examples/render_for_3dgs.py  # gaussians -> RAW fisheye pixels + ray map
python examples/handpose_3d.py      # virtual-view detections -> 3D keypoints
```

## Service API

```bash
uvicorn camera_geometry.service:app --port 8300
# POST /camera/resolve | validate | ray-map | virtual-camera
# POST /camera/stereo-rectify | stereo-rectify-upload | reprocess (+ GET /jobs/{id})
# GET  /camera/{calibration_id} | /metrics (Prometheus) | /healthz
```

Every calibration-bearing response carries `calibration_id`, `camera_model`,
`quality_status`, `artifact_uri`. `GET /metrics` exposes spec-section-43
metrics in Prometheus format.

## Storage, Docker, CI

* LUT cache on local FS or any S3-compatible bucket (`CAMERA_GEOMETRY_LUT_BACKEND=s3`)
* Timestamps as parquet (pyarrow) with CSV fallback
* `docker build -t camera-geometry . && docker run -p 8300:8300 camera-geometry`
* CI runs ruff + mypy (clean over 48 files) + pytest on Python 3.11/3.12

## Roadmap (external inputs needed)

* Phase-0 vendor golden for Orbbec (`orbbec_pixel_to_ray()` slot in `scripts/export_golden.py`)
* Real-scene validation of the 0.5 px stereo gate after refinement
* Rolling-shutter closure needs IMU/readout metadata (model + row-aware API ready)
* AnyCalib network weights behind the existing `DenseRayProvider` interface

## License

Apache-2.0. See [LICENSE](LICENSE).
