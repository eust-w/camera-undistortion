"""Data-platform service API (spec section 40).

Path-based MVP (the service reads configs/images from local paths the
platform already controls). Every response that references a calibration
contains: calibration_id, camera_model, quality_status, artifact_uri.

Run:  uvicorn camera_geometry.service:app --port 8300
Auth/TLS/rate-limiting belong to the platform gateway, not this module.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .calibration.parser import load_device_calibration
from .calibration.resolver import CalibrationNotFound, CalibrationResolver
from .calibration.store import CalibrationStore
from .calibration.validator import validate_calibration
from .geometry.ray_map import generate_ray_map
from .metrics import to_prometheus
from .models.factory import build_camera
from .pipeline import AutoGeometryPipeline
from .registry.lut_cache import LUTCache, make_key
from .schema import CODE_VERSION
from .stereo.rectify import build_stereo_rectifier
from .transforms import SE3
from .virtual_camera.presets import create_virtual_camera
from .virtual_camera.remap import create_remap_grid, remap_coverage


def _home() -> Path:
    return Path(os.environ.get("CAMERA_GEOMETRY_HOME", Path.home() / ".cache" / "camera_geometry"))


app = FastAPI(title="Auto Camera Geometry", version=CODE_VERSION)


class ResolveRequest(BaseModel):
    config_path: str = Field(description="device metadata YAML (canonical schema)")
    camera_id: str


class ValidateRequest(BaseModel):
    config_path: str
    roundtrip_samples: int = 20_000
    baseline_range: tuple[float, float] = (0.10, 0.14)


class RayMapRequest(BaseModel):
    config_path: str
    camera_id: str


class VirtualCameraRequest(BaseModel):
    config_path: str
    camera_id: str
    preset: str = "ego_100"


class StereoRectifyRequest(BaseModel):
    config_path: str
    left_image: str
    right_image: str
    out_dir: str
    width: int = 1024
    height: int = 1024
    fov_x_deg: float = 100.0
    depth: bool = False


class ReprocessRequest(BaseModel):
    input_dir: str
    output_dir: str
    config_path: str
    preset: str = "ego_100"
    max_frames: int | None = None
    enable_fallback: bool = True


def _load(path: str):
    try:
        return load_device_calibration(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid calibration config: {exc}") from exc


@app.post("/camera/resolve")
def camera_resolve(req: ResolveRequest):
    calib = _load(req.config_path)
    store = CalibrationStore(_home())
    resolver = CalibrationResolver(calib, store)
    try:
        resolved = resolver.resolve(req.camera_id, None)
    except CalibrationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entry = resolved.calibration.camera(resolved.camera_id)
    return {
        "calibration_id": resolved.calibration_id,
        "camera_model": entry.model.type,
        "quality_status": resolved.quality_status,
        "artifact_uri": str(_home() / "calibrations"),
        "source": resolved.source,
        "resolution": list(entry.resolution.wh),
    }


@app.post("/camera/validate")
def camera_validate(req: ValidateRequest):
    calib = _load(req.config_path)
    results = validate_calibration(calib, baseline_range=req.baseline_range,
                                   roundtrip_samples=req.roundtrip_samples)
    store = CalibrationStore(_home())
    from .calibration.store import CalibrationRecord

    for cid, qa in results.items():
        if store.exists(qa.calibration_id):
            store.update_qa(qa.calibration_id, qa.status, qa.score, qa.report.to_json())
        else:
            store.add(
                CalibrationRecord(
                    calibration_id=qa.calibration_id,
                    calibration=calib,
                    quality_status=qa.status,
                    health_score=qa.score,
                    qa_report=qa.report.to_json(),
                )
            )
    payload = {cid: qa.report.to_json() for cid, qa in results.items()}
    status = "PASS" if all(qa.status == "PASS" for qa in results.values()) else \
             ("FAIL" if any(qa.status == "FAIL" for qa in results.values()) else "DEGRADED")
    return {
        "status": status,
        "calibration_ids": {cid: qa.calibration_id for cid, qa in results.items()},
        "report": payload,
    }


@app.get("/camera/{calibration_id}")
def camera_get(calibration_id: str):
    store = CalibrationStore(_home())
    try:
        record = store.get(calibration_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "calibration_id": record.calibration_id,
        "camera_model": [c.model.type for c in record.calibration.cameras],
        "quality_status": record.quality_status,
        "health_score": record.health_score,
        "artifact_uri": str(_home() / "calibrations"),
        "calibration": record.calibration.model_dump(),
    }


@app.post("/camera/ray-map")
def camera_ray_map(req: RayMapRequest):
    calib = _load(req.config_path)
    entry = calib.camera(req.camera_id)
    cam = build_camera(entry)
    cache = LUTCache(_home() / "luts")
    key = make_key({"kind": "raymap", "calibration_id": calib.calibration_id(req.camera_id),
                    "resolution": list(entry.resolution.wh), "code": CODE_VERSION})
    cached = cache.get(key, "raymap")
    if cached is None:
        rays, valid = generate_ray_map(cam)
        cache.put(key, "raymap", {"rays": rays.astype(np.float16), "valid": valid})
        valid_ratio = float(valid.mean())
    else:
        valid_ratio = float(cached["valid"].mean())
    return {
        "calibration_id": calib.calibration_id(req.camera_id),
        "camera_model": entry.model.type,
        "quality_status": calib.quality.status,
        "artifact_uri": str(cache.path(key, "raymap")),
        "valid_pixel_ratio": round(valid_ratio, 4),
    }


@app.post("/camera/virtual-camera")
def camera_virtual(req: VirtualCameraRequest):
    calib = _load(req.config_path)
    entry = calib.camera(req.camera_id)
    cam = build_camera(entry)
    vcam = create_virtual_camera(req.preset)
    cache = LUTCache(_home() / "luts")
    key = make_key({"kind": f"virtual_{req.preset}", "calibration_id": calib.calibration_id(req.camera_id),
                    "dst": [vcam.width, vcam.height], "code": CODE_VERSION})
    cached = cache.get(key, "grid")
    if cached is None:
        grid, valid = create_remap_grid(cam, vcam)
        cache.put(key, "grid", {"grid": grid, "valid": valid})
    else:
        valid = cached["valid"]
    return {
        "calibration_id": calib.calibration_id(req.camera_id),
        "camera_model": entry.model.type,
        "quality_status": calib.quality.status,
        "artifact_uri": str(cache.path(key, "grid")),
        "preset": req.preset,
        "coverage": round(remap_coverage(valid), 4),
    }


@app.post("/camera/stereo-rectify")
def camera_stereo_rectify(req: StereoRectifyRequest):
    import cv2

    calib = _load(req.config_path)
    el, er = calib.camera("IR_L"), calib.camera("IR_R")
    ext = er.extrinsics or el.extrinsics
    if ext is None:
        raise HTTPException(status_code=400, detail="no extrinsics in config")
    cam_l, cam_r = build_camera(el), build_camera(er)
    rig = build_stereo_rectifier(cam_l, cam_r, SE3.from_Rt(ext.rotation, ext.translation_m),
                                 req.width, req.height, req.fov_x_deg)
    img_l, img_r = cv2.imread(req.left_image), cv2.imread(req.right_image)
    if img_l is None or img_r is None:
        raise HTTPException(status_code=400, detail="cannot read input images")
    out = Path(req.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from .runtime.opencv_backend import remap_image as _remap

    rl, rr = _remap(img_l, rig.grid_left), _remap(img_r, rig.grid_right)
    cv2.imwrite(str(out / "rectified_left.png"), rl)
    cv2.imwrite(str(out / "rectified_right.png"), rr)
    result = {
        "calibration_id": f"{calib.calibration_id('IR_L')};{calib.calibration_id('IR_R')}",
        "camera_model": el.model.type,
        "quality_status": calib.quality.status,
        "artifact_uri": str(out),
        "baseline_m": rig.baseline_m,
        "fx_rectified": rig.virtual_camera.fx,
    }
    if req.depth:
        from .stereo.depth import stereo_depth

        disp, depth = stereo_depth(rl, rr, rig.virtual_camera.fx, rig.baseline_m)
        np.savez_compressed(out / "depth.npz", disparity=disp, depth=depth,
                            fx=rig.virtual_camera.fx, baseline_m=rig.baseline_m)
        result["artifact_uri"] = str(out / "depth.npz")
    return result


@app.post("/camera/reprocess")
def camera_reprocess(req: ReprocessRequest):
    pipe = AutoGeometryPipeline(config_path=req.config_path, home=_home())
    report = pipe.process_episode(
        input_dir=req.input_dir, output_dir=req.output_dir, preset=req.preset,
        max_frames=req.max_frames, enable_fallback=req.enable_fallback,
    )
    payload = report.to_json()
    payload["artifact_uri"] = req.output_dir
    return payload


@app.get("/metrics")
def prometheus_metrics():
    latest = None
    store = CalibrationStore(_home())
    for record in store.list_records():
        if record.qa_report:
            latest = record
    metrics = {"calibration_records": len(store.list_records())}
    if latest is not None:
        metrics["latest_calibration_id"] = latest.calibration_id
        metrics["latest_health_score"] = latest.health_score if latest.health_score is not None else -1
        metrics["latest_quality_status"] = latest.quality_status
    return __import__("fastapi").Response(
        content=to_prometheus(metrics), media_type="text/plain; version=0.0.4"
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "code_version": CODE_VERSION}
