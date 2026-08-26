"""LUT cache: all fixed geometry artifacts cached by content key.

cache_key = hash(calibration_id + output_projection + output_resolution + output_fov)
Artifacts: ray_map / valid_mask / virtual grids / stereo grids.

Storage backends (spec section 37 recommends object storage for the registry):
  * local  (default): filesystem directory
  * s3:              any S3-compatible bucket via boto3 (optional dependency),
                     configured with CAMERA_GEOMETRY_S3_BUCKET / CAMERA_GEOMETRY_S3_PREFIX
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Protocol

import numpy as np


def make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


class StorageBackend(Protocol):
    def exists(self, key: str) -> bool: ...
    def load(self, key: str) -> bytes | None: ...
    def store(self, key: str, blob: bytes) -> str: ...
    def uri(self, key: str) -> str: ...


class LocalBackend:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def store(self, key: str, blob: bytes) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(p)
        return str(p)

    def uri(self, key: str) -> str:
        return str(self._path(key))


class S3Backend:
    def __init__(self, bucket: str | None = None, prefix: str = "luts"):
        import boto3  # optional dependency

        self.bucket = bucket or os.environ["CAMERA_GEOMETRY_S3_BUCKET"]
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}.npz" if self.prefix else f"{key}.npz"

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except self.client.exceptions.ClientError as exc:
            if "404" in str(exc) or "Not Found" in str(exc):
                return False
            raise

    def load(self, key: str) -> bytes | None:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
            return obj["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def store(self, key: str, blob: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=blob)
        return f"s3://{self.bucket}/{self._key(key)}"

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"


def _backend_from_env(root: str | Path) -> StorageBackend:
    if os.environ.get("CAMERA_GEOMETRY_LUT_BACKEND", "local") == "s3":
        return S3Backend(prefix=os.environ.get("CAMERA_GEOMETRY_S3_PREFIX", "luts"))
    return LocalBackend(root)


class LUTCache:
    def __init__(self, root: str | Path, backend: StorageBackend | None = None):
        self.backend = backend or _backend_from_env(root)
        if isinstance(self.backend, LocalBackend):
            self.root = self.backend.root
        else:
            self.root = Path(root)

    def _storage_key(self, key: str, name: str) -> str:
        return f"{key}__{name}"

    def path(self, key: str, name: str) -> Path:
        """Local path (local backend) or stable pseudo-uri (s3)."""
        return self.root / f"{self._storage_key(key, name)}.npz"

    def has(self, key: str, name: str) -> bool:
        return self.backend.exists(self._storage_key(key, name))

    def get(self, key: str, name: str) -> dict[str, np.ndarray] | None:
        blob = self.backend.load(self._storage_key(key, name))
        if blob is None:
            return None
        with np.load(io.BytesIO(blob)) as z:
            return {k: z[k] for k in z.files if not k.startswith("_")}

    def put(self, key: str, name: str, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
        buf = io.BytesIO()
        np.savez(buf, **arrays, _meta=json.dumps(meta or {}))  # type: ignore[arg-type]
        uri = self.backend.store(self._storage_key(key, name), buf.getvalue())
        return Path(uri)
