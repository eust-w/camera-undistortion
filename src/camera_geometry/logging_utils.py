"""Structured logging helpers (JSON mode for platform ingestion)."""

from __future__ import annotations

import json
import logging
import os
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, json_mode: bool | None = None) -> None:
    json_mode = (
        os.environ.get("CAMERA_GEOMETRY_LOG_JSON", "0") == "1"
        if json_mode is None
        else json_mode
    )
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_mode else logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
