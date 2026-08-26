"""End-to-end processing DAG with automatic failure state machine.

INGEST -> RESOLVE -> STATIC QA -> [PASS | FALLBACK -> QUARANTINE]
       -> GENERATE/CACHE LUTS (raymap, virtual, stereo)
       -> SAMPLED FRAME QA (image quality, epipolar, line straightness)
       -> HEALTH SCORE + METRICS + EPISODE OUTPUTS

States: PASS / RETRY / FALLBACK / QUARANTINE. No human interaction.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration.parser import load_device_calibration
from .calibration.resolver import (
    CalibrationNotFound,
    CalibrationResolver,
    FallbackRequired,
    ResolvedCalibration,
)

from .calibration.store import CalibrationStore
from .calibration.validator import CalibrationQA, validate_calibration
from .geometry.ray_map import generate_ray_map
from .models.base import CameraModel
from .models.factory import build_camera
from .qa.checks import (
    CheckResult,
    check_epipolar,
    check_image_quality,
)
from .qa.report import aggregate_health, build_metrics_section
from .registry.lut_cache import LUTCache, make_key
from .runtime.opencv_backend import remap_image
from .schema import CODE_VERSION, DeviceCalibration
from .stereo.rectify import StereoRectifier, build_stereo_rectifier
from .stereo.depth import stereo_depth
from .transforms import SE3
from .virtual_camera.presets import DEFAULT_PRESET, create_virtual_camera
from .virtual_camera.remap import create_remap_grid, remap_coverage

logger = logging.getLogger("camera_geometry")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class EpisodeReport:
    status: str
    outputs: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    quarantine: dict | None = None

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "quarantine": self.quarantine,
        }


def _timed():
    t0 = time.perf_counter()
    return lambda: int((time.perf_counter() - t0) * 1000)


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def find_stereo_pairs(input_dir: Path, reference_camera: str = "IR_L",
                      other_camera: str = "IR_R") -> list[tuple[Path, Path]]:
    """Frame pairs (reference, other) matched by basename."""
    left_dir = input_dir / "raw" / reference_camera
    right_dir = input_dir / "raw" / other_camera
    right_by_name = {p.name: p for p in _list_images(right_dir)}
    pairs = []
    for lp in _list_images(left_dir):
        rp = right_by_name.get(lp.name)
        if rp is not None:
            pairs.append((lp, rp))
    return pairs


class AutoGeometryPipeline:
    def __init__(
        self,
        config_path: str | Path | None = None,
        device_calibration: DeviceCalibration | None = None,
        home: str | Path | None = None,
        baseline_range: tuple[float, float] = (0.10, 0.14),
        roundtrip_samples: int = 100_000,
        enable_fallback: bool = True,
        max_fallback_frames: int = 8,
        geometry_fail_median_px: float = 2.0,
    ):
        if (config_path is None) == (device_calibration is None):
            raise ValueError("provide exactly one of config_path / device_calibration")
        if device_calibration is not None:
            self.device_calibration = device_calibration
        else:
            self.device_calibration = load_device_calibration(config_path)  # type: ignore[arg-type]
        self.home = Path(
            home
            or __import__("os").environ.get("CAMERA_GEOMETRY_HOME")
            or Path.home() / ".cache" / "camera_geometry"
        )
        self.store = CalibrationStore(self.home)
        self.cache = LUTCache(self.home / "luts")
        self.baseline_range = baseline_range
        self.roundtrip_samples = roundtrip_samples
        self.enable_fallback = enable_fallback
        self.max_fallback_frames = max_fallback_frames
        self.geometry_fail_median_px = geometry_fail_median_px
        self.resolver = CalibrationResolver(self.device_calibration, self.store)

    # ------------------------------------------------------------ cameras
    def get_camera(self, calibration_id: str) -> CameraModel:
        record = self.store.get(calibration_id)
        calib = record.calibration
        for entry in calib.cameras:
            if calib.calibration_id(entry.id) == calibration_id:
                return build_camera(entry)
        raise KeyError(f"calibration_id '{calibration_id}' does not match any camera")

    # -------------------------------------------------------------- stage 1
    def resolve_all(self) -> dict[str, ResolvedCalibration]:
        out = {}
        for entry in self.device_calibration.cameras:
            out[entry.id] = self.resolver.resolve(entry.id, entry.resolution.wh)
        return out

    # -------------------------------------------------------------- stage 2
    def run_static_qa(self) -> dict[str, CalibrationQA]:
        return validate_calibration(
            self.device_calibration,
            baseline_range=self.baseline_range,
            roundtrip_samples=self.roundtrip_samples,
        )

    # -------------------------------------------------------------- luts
    def ray_map_artifacts(self, camera_id: str, resolved: ResolvedCalibration):
        entry = resolved.calibration.camera(camera_id)
        cam = build_camera(entry)
        key = make_key(
            {
                "kind": "raymap",
                "calibration_id": resolved.calibration_id,
                "resolution": list(entry.resolution.wh),
                "code": CODE_VERSION,
            }
        )
        cached = self.cache.get(key, "raymap")
        if cached is not None:
            return cached["rays"], cached["valid"], key
        rays, valid = generate_ray_map(cam)
        self.cache.put(
            key,
            "raymap",
            {"rays": rays.astype(np.float16), "valid": valid},
            meta={"calibration_id": resolved.calibration_id},
        )
        return rays.astype(np.float16), valid, key

    def virtual_grid(self, camera_id: str, resolved: ResolvedCalibration, preset: str = DEFAULT_PRESET):
        entry = resolved.calibration.camera(camera_id)
        cam = build_camera(entry)
        vcam = create_virtual_camera(preset)
        key = make_key(
            {
                "kind": f"virtual_{preset}",
                "calibration_id": resolved.calibration_id,
                "src_resolution": list(entry.resolution.wh),
                "dst_resolution": [vcam.width, vcam.height],
                "code": CODE_VERSION,
            }
        )
        cached = self.cache.get(key, "grid")
        if cached is not None:
            return cached["grid"], cached["valid"], key
        grid, valid = create_remap_grid(cam, vcam)
        self.cache.put(key, "grid", {"grid": grid, "valid": valid})
        return grid, valid, key

    def stereo_grids(
        self,
        resolved_left: ResolvedCalibration,
        resolved_right: ResolvedCalibration,
        width: int = 640,
        height: int = 640,
        fov_x_deg: float = 100.0,
    ) -> tuple[StereoRectifier, str]:
        cam_l = build_camera(resolved_left.calibration.camera(resolved_left.camera_id))
        cam_r = build_camera(resolved_right.calibration.camera(resolved_right.camera_id))
        ext_rr = resolved_right.calibration.camera(resolved_right.camera_id).extrinsics
        if ext_rr is None:  # pragma: no cover - guarded by callers
            raise ValueError("stereo rectification requires extrinsics on the right camera")
        T = SE3.from_Rt(ext_rr.rotation, ext_rr.translation_m)
        key = make_key(
            {
                "kind": "stereo_rectify",
                "left": resolved_left.calibration_id,
                "right": resolved_right.calibration_id,
                "T": {"R": np.asarray(T.R).round(9).tolist(), "t": np.asarray(T.t).round(9).tolist()},
                "size": [width, height],
                "fov": fov_x_deg,
                "code": CODE_VERSION,
            }
        )
        rig = build_stereo_rectifier(cam_l, cam_r, T, width, height, fov_x_deg)
        cached = self.cache.get(key, "grids")
        if cached is not None:
            rig.grid_left = cached["grid_left"]
            rig.grid_right = cached["grid_right"]
            rig.valid_left = cached["valid_left"]
            rig.valid_right = cached["valid_right"]
        else:
            self.cache.put(
                key,
                "grids",
                {
                    "grid_left": rig.grid_left,
                    "grid_right": rig.grid_right,
                    "valid_left": rig.valid_left,
                    "valid_right": rig.valid_right,
                },
            )
        return rig, key

    # -------------------------------------------------------------- episode
    def process_episode(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        preset: str = DEFAULT_PRESET,
        rectified_size: tuple[int, int] = (640, 640),
        rectified_fov_deg: float = 100.0,
        qa_pairs: int = 16,
        max_frames: int | None = None,
        write_virtual: bool = True,
        write_rectified: bool = True,
        write_depth: bool = False,
        enable_fallback: bool | None = None,
    ) -> EpisodeReport:
        # per-call override only: never mutate instance state (a pipeline
        # reused across episodes must keep its configured fallback policy)
        effective_fallback = self.enable_fallback if enable_fallback is None else enable_fallback
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        timings: dict[str, int] = {}
        metrics = build_metrics_section()

        t = _timed()
        cam_ids = [c.id for c in self.device_calibration.cameras]
        ref_id = self._reference_camera()
        other_id = next((c for c in cam_ids if c != ref_id), cam_ids[0])
        pairs = find_stereo_pairs(input_dir, ref_id, other_id)
        timings["ingest_ms"] = t()
        if not pairs:
            return self._quarantine(output_dir, "no_stereo_pairs_found", "ingest", {})

        t = _timed()
        try:
            resolved = self.resolve_all()
        except (CalibrationNotFound, FallbackRequired) as exc:
            return self._quarantine(output_dir, "calibration_unresolvable", "resolve", {"error": str(exc)})
        timings["resolve_ms"] = t()

        for cid_, res_ in resolved.items():
            if not self.store.exists(res_.calibration_id):
                from .calibration.store import CalibrationRecord

                self.store.add(
                    CalibrationRecord(
                        calibration_id=res_.calibration_id,
                        calibration=res_.calibration,
                        quality_status=res_.quality_status,
                    )
                )

        t = _timed()
        static_qa = self.run_static_qa()
        timings["static_qa_ms"] = t()

        failing = {cid: qa for cid, qa in static_qa.items() if qa.status == "FAIL"}
        fallback_history: list[dict] = []
        if failing:
            adopted = None
            if effective_fallback:
                t = _timed()
                adopted = self._attempt_fallback(pairs, "static_qa", fallback_history)
                timings["fallback_attempt_ms"] = t()
            if adopted is not None:
                self.device_calibration = adopted
                self.resolver = CalibrationResolver(self.device_calibration, self.store)
                self._register_current()
                report = self.process_episode(
                    input_dir, output_dir, preset=preset, rectified_size=rectified_size,
                    rectified_fov_deg=rectified_fov_deg, qa_pairs=qa_pairs, max_frames=max_frames,
                    write_virtual=write_virtual, write_rectified=write_rectified,
                    write_depth=write_depth, enable_fallback=False,
                )
                report.metrics["fallback"] = {"attempted": True, "adopted": True, "stage": "static_qa", "history": fallback_history}
                return report
            return self._quarantine(
                output_dir,
                "camera_geometry_unstable",
                "static_qa",
                {
                    "cameras": {cid: qa.report.to_json() for cid, qa in failing.items()},
                    "fallback_history": fallback_history,
                },
            )
        for cid, qa in static_qa.items():
            self.store.update_qa(qa.calibration_id, qa.status, qa.score, qa.report.to_json())

        t = _timed()
        artifacts = {}
        for cid, res in resolved.items():
            rays, valid, rkey = self.ray_map_artifacts(cid, res)
            grid, vvalid, vkey = self.virtual_grid(cid, res, preset)
            artifacts[cid] = {
                "rays": rays,
                "ray_valid": valid,
                "virtual_grid": grid,
                "virtual_valid": vvalid,
                "keys": {"raymap": rkey, "virtual": vkey},
            }
        left_id, right_id = ref_id, other_id
        rig, skey = self.stereo_grids(resolved[left_id], resolved[right_id], *rectified_size, rectified_fov_deg)
        timings["lut_generation_ms"] = t()

        t = _timed()
        sampled = pairs if len(pairs) <= qa_pairs else [
            pairs[i] for i in np.linspace(0, len(pairs) - 1, qa_pairs).astype(int)
        ]
        epipolar_checks: list[CheckResult] = []
        quality_scores: list[float] = []
        retention_values: list[float] = []
        for lp, rp in sampled:
            img_l = _imread(lp)
            img_r = _imread(rp)
            if img_l is None or img_r is None:
                continue
            q = check_image_quality(img_l)
            quality_scores.append(q.score)
            rect_l = remap_image(img_l, rig.grid_left)
            rect_r = remap_image(img_r, rig.grid_right)
            epi = check_epipolar(rect_l, rect_r)
            n_kp = epi.metrics.get("n_kp_left", 0) or 0
            if n_kp > 0:
                retention_values.append(epi.metrics.get("n_matches", 0) / n_kp)
            epipolar_checks.append(epi)
        timings["frame_qa_ms"] = t()

        def _stat(key, default=np.inf):
            vals = [c.metrics.get(key, default) for c in epipolar_checks]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.median(vals)) if vals else float(default)

        epipolar_median = _stat("median")
        epipolar_p95 = _stat("p95")
        epi_ok_ratio = float(np.mean([c.passed for c in epipolar_checks])) if epipolar_checks else 0.0
        stereo_check = CheckResult(
            "stereo_epipolar",
            epi_ok_ratio >= 0.75 and epipolar_median < 0.5,
            float(np.clip(1.0 - epipolar_median, 0, 1)) if np.isfinite(epipolar_median) else 0.0,
            {
                "median": epipolar_median,
                "p95": epipolar_p95,
                "pass_ratio": epi_ok_ratio,
                "n_pairs": len(epipolar_checks),
                "feature_retention": float(np.mean(retention_values)) if retention_values else 0.0,
            },
        )
        iq_check = CheckResult(
            "image_quality_dataset",
            bool(np.mean(quality_scores) > 0.66) if quality_scores else False,
            float(np.mean(quality_scores)) if quality_scores else 0.0,
            {"n_sampled": len(quality_scores)},
        )

        geometry_bad = (
            not np.isfinite(epipolar_median) or epipolar_median > self.geometry_fail_median_px
        )
        if geometry_bad and effective_fallback:
            t = _timed()
            adopted = self._attempt_fallback(pairs, "geometry_qa", fallback_history)
            timings["fallback_attempt_ms"] = t()
            if adopted is not None:
                self.device_calibration = adopted
                self.resolver = CalibrationResolver(self.device_calibration, self.store)
                self._register_current()
                report = self.process_episode(
                    input_dir, output_dir, preset=preset, rectified_size=rectified_size,
                    rectified_fov_deg=rectified_fov_deg, qa_pairs=qa_pairs, max_frames=max_frames,
                    write_virtual=write_virtual, write_rectified=write_rectified,
                    write_depth=write_depth, enable_fallback=False,
                )
                report.metrics["fallback"] = {
                    "attempted": True, "adopted": True, "stage": "geometry_qa",
                    "history": fallback_history,
                }
                (output_dir / "status.json").write_text(json.dumps(report.to_json(), indent=2))
                return report

        static_checks = [c for qa in static_qa.values() for c in qa.report.checks]
        health = aggregate_health(static_checks + [stereo_check, iq_check])
        overall_status = health.status
        if overall_status == "FAIL" and not effective_fallback and fallback_history:
            # fallback chain exhausted, still below gate -> QUARANTINE (spec 42);
            # derived outputs are still written for later diagnosis/reprocessing
            pass

        t = _timed()
        out_paths = self._write_outputs(
            output_dir,
            resolved,
            artifacts,
            rig,
            health,
            static_qa,
            pairs,
            left_id=left_id,
            right_id=right_id,
            preset=preset,
            max_frames=max_frames,
            write_virtual=write_virtual,
            write_rectified=write_rectified,
            write_depth=write_depth,
        )
        timings["derive_write_ms"] = t()

        metrics.update(
            {
                "camera_model": "kannala_brandt_4",
                "calibration_source": ",".join(sorted({r.source for r in resolved.values()})),
                "calibration_id": ";".join(r.calibration_id for r in resolved.values()),
                "projection_roundtrip_p99": max(
                    c.metrics.get("p99") for qa in static_qa.values() for c in qa.report.checks if c.name == "projection_roundtrip"
                ),
                "valid_pixel_ratio": min(
                    c.metrics.get("valid_pixel_ratio") for qa in static_qa.values() for c in qa.report.checks if c.name == "projection_roundtrip"
                ),
                "stereo_epipolar_median": epipolar_median,
                "stereo_epipolar_p95": epipolar_p95,
                "virtual_camera_coverage": float(np.mean([
                    remap_coverage(a["virtual_valid"]) for a in artifacts.values()
                ])),
                "feature_retention": stereo_check.metrics.get("feature_retention"),
                "calibration_health_score": round(health.score, 4),
                "runtime_ms": {**timings, **out_paths.pop("_runtime_extra", {})},
            }
        )
        report = EpisodeReport(status=overall_status, outputs=out_paths, metrics=metrics)
        (output_dir / "status.json").write_text(json.dumps(report.to_json(), indent=2))
        logger.info("episode %s -> %s", input_dir, overall_status)
        return report

    # -------------------------------------------------------------- writers
    def _write_outputs(
        self,
        output_dir: Path,
        resolved: dict[str, ResolvedCalibration],
        artifacts: dict,
        rig: StereoRectifier,
        health,
        static_qa: dict[str, CalibrationQA],
        pairs: list[tuple[Path, Path]],
        left_id: str,
        right_id: str,
        preset: str,
        max_frames,
        write_virtual: bool,
        write_rectified: bool,
        write_depth: bool,
    ) -> dict:
        meta_dir = output_dir / "metadata"
        geom_dir = output_dir / "geometry"
        qa_dir = output_dir / "qa"
        for d in (meta_dir, geom_dir, qa_dir):
            d.mkdir(parents=True, exist_ok=True)

        (meta_dir / "cameras.json").write_text(
            json.dumps(self.device_calibration.model_dump(), indent=2)
        )
        refs = {cid: res.calibration_id for cid, res in resolved.items()}
        refs["reference_camera"] = self._reference_camera()
        (geom_dir / "calibration_refs.json").write_text(json.dumps(refs, indent=2))
        (geom_dir / "geometry_version.json").write_text(
            json.dumps({"schema_version": "camera-geometry/v1", "code_version": CODE_VERSION}, indent=2)
        )
        (qa_dir / "report.json").write_text(json.dumps(health.to_json(), indent=2))
        rows = [(i, lp.name, rp.name) for i, (lp, rp) in enumerate(pairs)]
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({
                "index": [r[0] for r in rows],
                "left_file": [r[1] for r in rows],
                "right_file": [r[2] for r in rows],
            })
            pq.write_table(table, meta_dir / "timestamps.parquet")
        except ImportError:
            with (meta_dir / "timestamps.csv").open("w") as f:
                f.write("index,left_file,right_file\n")
                for i, lp_name, rp_name in rows:
                    f.write(f"{i},{lp_name},{rp_name}\n")

        selected = pairs if max_frames is None else pairs[:max_frames]
        virt_root = output_dir / "derived" / f"virtual_{preset}"
        rect_root = output_dir / "derived" / f"stereo_rectified_{left_id}-{right_id}"
        for cid in (left_id, right_id):
            (virt_root / cid).mkdir(parents=True, exist_ok=True)
        for sub in ("left", "right"):
            (rect_root / sub).mkdir(parents=True, exist_ok=True)

        for lp, rp in selected:
            stem = lp.stem
            img_l = _imread(lp)
            img_r = _imread(rp)
            if img_l is None or img_r is None:
                continue
            if write_virtual:
                for cid, img in ((left_id, img_l), (right_id, img_r)):
                    _imwrite(virt_root / cid / f"{stem}.png", remap_image(img, artifacts[cid]["virtual_grid"]))
                raw_root = lp.parent.parent
                for cid in resolved:
                    if cid in (left_id, right_id):
                        continue
                    extra = _imread(raw_root / cid / f"{stem}{lp.suffix}")
                    if extra is not None:
                        _imwrite(virt_root / cid / f"{stem}.png",
                                 remap_image(extra, artifacts[cid]["virtual_grid"]))
            if write_rectified:
                rl = remap_image(img_l, rig.grid_left)
                rr = remap_image(img_r, rig.grid_right)
                _imwrite(rect_root / "left" / f"{stem}.png", rl)
                _imwrite(rect_root / "right" / f"{stem}.png", rr)
                if write_depth:
                    disp, depth = stereo_depth(rl, rr, rig.virtual_camera.fx, rig.baseline_m)
                    np.savez_compressed(
                        rect_root / "left" / f"{stem}_depth.npz",
                        disparity=disp,
                        depth=depth,
                        fx=rig.virtual_camera.fx,
                        baseline_m=rig.baseline_m,
                    )
        return {
            "metadata": str(meta_dir),
            "geometry": str(geom_dir),
            "qa_report": str(qa_dir / "report.json"),
            "virtual_view": str(virt_root) if write_virtual else None,
            "stereo_rectified": str(rect_root) if write_rectified else None,
            "frames_processed": len(selected),
        }

    def _reference_camera(self) -> str:
        exts = [c.extrinsics for c in self.device_calibration.cameras if c.extrinsics]
        return exts[0].reference_camera if exts else self.device_calibration.cameras[0].id

    def _register_current(self) -> None:
        from .calibration.store import CalibrationRecord

        for entry in self.device_calibration.cameras:
            cid = self.device_calibration.calibration_id(entry.id)
            if not self.store.exists(cid):
                self.store.add(
                    CalibrationRecord(
                        calibration_id=cid,
                        calibration=self.device_calibration,
                        quality_status=self.device_calibration.quality.status,
                    )
                )

    def _attempt_fallback(self, pairs: list[tuple[Path, Path]], stage: str, history: list[dict]):
        """Stage 1: stereo refinement with fixed factory E (strong prior).
        Stage 2: AnyCalib model fit from stereo-derived dense rays.

        Returns an adopted DeviceCalibration or None (caller quarantines).
        """
        from .calibration.validator import validate_calibration as _validate
        from .fallback.anycalib import build_camera_entry, fit_from_observations
        from .fallback.colmap_refine import refine_from_stereo_frames
        from .models.factory import build_camera as _build
        from .transforms import SE3

        sample = pairs if len(pairs) <= self.max_fallback_frames else [
            pairs[i] for i in np.linspace(0, len(pairs) - 1, self.max_fallback_frames).astype(int)
        ]
        stereo_imgs = []
        for lp, rp in sample:
            il, ir = _imread(lp), _imread(rp)
            if il is not None and ir is not None:
                stereo_imgs.append((il, ir))
        if not stereo_imgs:
            return None

        cam_ids = [c.id for c in self.device_calibration.cameras]
        left_id, right_id = (cam_ids + ["IR_L", "IR_R"])[:2] if len(cam_ids) < 2 else cam_ids[:2]
        entries = {c.id: c for c in self.device_calibration.cameras}
        el, er = entries[left_id], entries[right_id]

        def _validate_candidate(cand: DeviceCalibration) -> dict:
            return _validate(cand, baseline_range=self.baseline_range,
                             roundtrip_samples=min(self.roundtrip_samples, 20_000))

        # ---- stage 1: stereo refinement with fixed factory E ------------
        # progressive: two passes (r1 -> r2); the second pass re-anchors the
        # prior at r1 so the remaining error can keep shrinking.
        cand = self.device_calibration.model_copy(deep=True)
        stage1: dict = {}
        ext = er.extrinsics or el.extrinsics
        if ext is not None:
            rot = np.asarray(ext.rotation, dtype=np.float64)
            tra = np.asarray(ext.translation_m, dtype=np.float64)
            T = SE3.from_Rt(rot, tra)
            prior_l, prior_r = _build(el), _build(er)
            cur_l, cur_r = el, er
            rnd = 0
            for _attempt in range(4):
                rep = refine_from_stereo_frames(
                    stereo_imgs, cur_l, cur_r, prior_l, prior_r, T,
                    max_pairs=len(stereo_imgs),
                    reference_camera_id=self._reference_camera(),
                )
                if rep is None:
                    continue
                rnd += 1
                stage1[f"round{rnd}"] = rep.to_json()
                cand.calibration.source = "multiview_refinement"
                cand.calibration.version = f"{self.device_calibration.calibration.version}-r{rnd}"
                cand.cameras = [rep.entries.get(c.id, c) for c in cand.cameras]
                new_l, new_r = rep.entries[cur_l.id], rep.entries[cur_r.id]
                if new_l == cur_l and new_r == cur_r:
                    break
                cur_l, cur_r = new_l, new_r
                prior_l, prior_r = _build(cur_l), _build(cur_r)
            if rnd == 0:
                cand = None  # type: ignore[assignment]
        history.append({"stage": "stereo_refinement", "trigger": stage, "stats": stage1})
        if stage1:
            qa = _validate_candidate(cand)
            if all(q.status != "FAIL" for q in qa.values()):
                return cand

        # ---- stage 1.5: external COLMAP SfM (optional, binary on PATH) ----
        # COLMAP self-calibrates each camera independently on the episode's
        # frames; per-camera vertical-channel params are mapped back with the
        # same identifiable-subset rule. Skipped silently when no binary.
        if ext is not None:
            try:
                from .fallback.colmap_external import colmap_available, refine_with_colmap_cli

                if colmap_available():
                    raw_root = sample[0][0].parent.parent
                    cand_ext = self.device_calibration.model_copy(deep=True)
                    cand_ext.calibration.source = "colmap_external"
                    cand_ext.calibration.version = f"{self.device_calibration.calibration.version}-c1"
                    stage15: dict = {}
                    for entry in cand_ext.cameras:
                        try:
                            refined = refine_with_colmap_cli(
                                images_dir=raw_root / entry.id,
                                workdir=self.home / "colmap" / f"{entry.id}_{stage}",
                                entry=entry,
                            )
                            idx = cand_ext.cameras.index(entry)
                            cand_ext.cameras[idx] = refined
                            stage15[entry.id] = "refined"
                        except Exception as exc:
                            stage15[entry.id] = {"error": str(exc)}
                    history.append({"stage": "colmap_external", "trigger": stage, "cameras": stage15})
                    if stage15 and all(not isinstance(v, dict) for v in stage15.values()):
                        qa_ext = _validate_candidate(cand_ext)
                        if all(q.status != "FAIL" for q in qa_ext.values()):
                            return cand_ext
            except FallbackRequired:
                pass

        # ---- stage 2: AnyCalib (model may change) ------------------------
        cand2 = self.device_calibration.model_copy(deep=True)
        cand2.calibration.source = "anycalib"
        cand2.calibration.version = f"{self.device_calibration.calibration.version}-a1"
        stage2: dict = {}
        obs_by_cam: dict[str, list] = {}
        if ext is not None:
            T = SE3.from_Rt(ext.rotation, ext.translation_m)
            rep2 = refine_from_stereo_frames(
                stereo_imgs, el, er, _build(el), _build(er), T,
                max_pairs=len(stereo_imgs), collect_obs=True,
                reference_camera_id=self._reference_camera(),
            )
            if rep2 is not None:
                for o in rep2.observations:
                    obs_by_cam.setdefault(o.camera_id, []).append((o.pixels, o.rays))
        for entry in self.device_calibration.cameras:
            obs = obs_by_cam.get(entry.id) or []
            if not obs:
                continue
            w, h = entry.resolution.wh
            try:
                result = fit_from_observations(obs, w, h)
            except Exception as exc:
                stage2[entry.id] = {"error": str(exc)}
                continue
            new_entry = build_camera_entry(result, entry.id, w, h)
            new_entry.extrinsics = entry.extrinsics
            idx = cand2.cameras.index(entry)
            cand2.cameras[idx] = new_entry
            stage2[entry.id] = result.to_json()
        history.append({"stage": "anycalib", "trigger": stage, "cameras": stage2})
        if stage2 and all("error" not in v for v in stage2.values()):
            qa2 = _validate_candidate(cand2)
            if all(q.status != "FAIL" for q in qa2.values()):
                return cand2
        return None

    def _quarantine(self, output_dir: Path, reason: str, stage: str, details: dict) -> EpisodeReport:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "QUARANTINE",
            "reason": reason,
            "failure_stage": stage,
            "metrics": details,
        }
        (output_dir / "quarantine.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "status.json").write_text(json.dumps(payload, indent=2))
        logger.error("QUARANTINE (%s/%s)", stage, reason)
        return EpisodeReport(status="QUARANTINE", quarantine=payload)


def _imread(path: Path):
    import cv2

    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _imwrite(path: Path, img) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
