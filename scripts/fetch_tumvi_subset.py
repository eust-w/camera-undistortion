"""Selective extractor for TUM-VI Euroc-format tar over HTTP Range requests.

Reads tar headers sequentially via 512-byte range requests; downloads only
the members matching --include patterns; stops when --limit files per pattern
group are collected or EOF. Keeps transfer to a few tens of MB instead of 1.4 GB.
"""

from __future__ import annotations

import argparse
import http.client
import sys
import time
import urllib.parse
from pathlib import Path

BLOCK = 512


class RangeReader:
    """Keep-alive HTTPS connection with Range reads (deterministic, no cv2 RNG)."""

    def __init__(self, url: str):
        p = urllib.parse.urlparse(url)
        self.host = p.netloc
        self.path = p.path
        self.conn = http.client.HTTPSConnection(self.host, timeout=60)
        self.n_requests = 0

    def fetch(self, start: int, length: int, retries: int = 4) -> bytes:
        want = f"bytes={start}-{start + length - 1}"
        for attempt in range(retries):
            try:
                self.conn.request("GET", self.path, headers={"Range": want})
                resp = self.conn.getresponse()
                data = resp.read()
                if resp.status in (200, 206) and len(data) == length:
                    self.n_requests += 1
                    return data
                if resp.status == 416:
                    return b""
                self.conn.close()
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(1.0 * (attempt + 1))
            self.conn = http.client.HTTPSConnection(self.host, timeout=60)
        raise IOError(f"range read failed at {want}")


def parse_header(hdr: bytes):
    name = hdr[0:100].rstrip(b"\0").decode("utf-8", "replace")
    size = int(hdr[124:136].rstrip(b"\0 ").decode() or "0", 8)
    ftype = hdr[156:157].decode("utf-8", "replace") or "0"
    prefix = hdr[345:500].rstrip(b"\0").decode("utf-8", "replace")
    if prefix:
        name = prefix + "/" + name
    return name, size, ftype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cam", type=int, default=40)
    ap.add_argument("--cams", nargs="+", default=["cam0", "cam1"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = RangeReader(args.url)
    counts: dict[str, int] = {c: 0 for c in args.cams}
    csv_done: set[str] = set()
    t0 = time.time()
    downloaded = 0
    headers = 0

    def save(name: str, data_start: int, size: int):
        nonlocal downloaded
        blob = reader.fetch(data_start, size)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        downloaded += size
        for cam in args.cams:
            if f"mav0/{cam}/data/" in name and name.endswith(".png"):
                counts[cam] += 1
            if name == f"mav0/{cam}/data.csv":
                csv_done.add(cam)

    def walk(pos: int, want_cam: str | None, stop_before: str | None, budget_s: float):
        """Sequential header walk; download members of want_cam (respecting
        limits); stop before stop_before prefix or when budget expires.
        Returns next position and whether stop prefix was reached."""
        nonlocal headers, downloaded
        t_walk = time.time()
        while True:
            if time.time() - t_walk > budget_s:
                return pos, False
            hdr = reader.fetch(pos, BLOCK)
            headers += 1
            name, size, ftype = parse_header(hdr)
            if not name:
                return pos, True
            data_start = pos + BLOCK
            data_pad = (size + BLOCK - 1) // BLOCK * BLOCK
            if stop_before and name >= stop_before:
                return pos, True
            take = False
            if want_cam is None or name.startswith(f"mav0/{want_cam}/"):
                if name.endswith(".png") and want_cam is not None and counts[want_cam] < args.per_cam:
                    take = True
                if name.endswith("data.csv") and want_cam is not None and want_cam not in csv_done:
                    take = True
            if take and ftype == "0":
                save(name, data_start, size)
                print(f"[tar] +{name} ({size // 1024}KB) counts={counts} dl={downloaded/1e6:.1f}MB t={time.time()-t0:.0f}s", flush=True)
            pos = data_start + data_pad

    # single sequential pass: archive order is NOT sorted (cam1 first,
    # hashed file order), so just walk everything until both cameras have
    # enough frames or the archive ends.
    pos = 0
    top_prefix = None
    while True:
        hdr = reader.fetch(pos, BLOCK)
        headers += 1
        name, size, ftype = parse_header(hdr)
        if not name:
            print("[tar] end of archive")
            break
        if top_prefix is None and name.endswith("/"):
            top_prefix = name  # e.g. dataset-room1_512_16/
        rel = name[len(top_prefix):] if top_prefix and name.startswith(top_prefix) else name
        data_start = pos + BLOCK
        data_pad = (size + BLOCK - 1) // BLOCK * BLOCK

        take = False
        for cam in args.cams:
            if rel == f"mav0/{cam}/data.csv":
                take = cam not in csv_done
            elif rel.startswith(f"mav0/{cam}/data/") and rel.endswith(".png"):
                take = counts[cam] < args.per_cam

        if take and ftype == "0":
            blob = reader.fetch(data_start, size)
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            downloaded += size
            for cam in args.cams:
                if rel == f"mav0/{cam}/data.csv":
                    csv_done.add(cam)
                elif rel.startswith(f"mav0/{cam}/data/") and rel.endswith(".png"):
                    counts[cam] += 1
            print(f"[tar] +{rel} ({size//1024}KB) counts={counts} dl={downloaded/1e6:.1f}MB t={time.time()-t0:.0f}s", flush=True)

        pos = data_start + data_pad
        if all(counts[c] >= args.per_cam for c in args.cams):
            print("[tar] collected enough frames")
            break
        if time.time() - t0 > 1500:
            print("[tar] time budget exceeded")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
