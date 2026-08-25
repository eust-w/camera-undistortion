"""Device metadata parsing: vendor files -> canonical DeviceCalibration schema.

Accepts YAML or JSON. Normalizes units: translation_mm -> translation_m.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..schema import DeviceCalibration


def _normalize_units(node):
    """Recursively convert legacy 'translation_mm' keys to canonical meters."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "translation_mm" and "translation_m" not in node:
                out["translation_m"] = [float(x) / 1000.0 for x in v]
                continue
            if k in ("baseline_mm",) and "baseline_m" not in node:
                out["baseline_m"] = float(v) / 1000.0
                continue
            out[k] = _normalize_units(v)
        return out
    if isinstance(node, list):
        return [_normalize_units(x) for x in node]
    return node


def load_device_calibration(path: str | Path) -> DeviceCalibration:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = yaml.safe_load(text)
    data = _normalize_units(raw)

    cameras = data.get("cameras")
    if isinstance(cameras, dict):
        data["cameras"] = [{"id": k, **v} for k, v in cameras.items()]

    for cam in data.get("cameras", []):
        extr = cam.get("extrinsics")
        if isinstance(extr, dict) and "rotation" not in extr and "R_left_right" in extr:
            cam["extrinsics"] = {
                "reference_camera": extr.get("reference_camera", "IR_L"),
                "rotation": extr["R_left_right"],
                "translation_m": extr.get("translation_m", extr.get("t_left_right")),
            }
    model_type_aliases = {
        "kb": "kannala_brandt_4",
        "kb4": "kannala_brandt_4",
        "opencv_fisheye": "kannala_brandt_4",
        "eucm": "eucm",
        "double_sphere": "double_sphere",
        "ds": "double_sphere",
    }
    for cam in data.get("cameras", []):
        model = cam.get("model")
        if isinstance(model, str):
            cam["model"] = {"type": model_type_aliases.get(model.lower(), model)}
        elif isinstance(model, dict) and model.get("type") in model_type_aliases:
            model["type"] = model_type_aliases[model["type"].lower()]
    return DeviceCalibration.model_validate(data)


def dump_device_calibration(calib: DeviceCalibration, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = calib.model_dump()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
