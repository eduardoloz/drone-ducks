"""Ingest: record HLS clips politely and read frames with their real timestamps.

Why timestamps matter: speed = distance / time. We never assume "frame i is at
i / fps"; we read each frame's presentation timestamp (PTS, 90 kHz clock) from
the stream with ffprobe and carry it with the frame.

Etiquette (see docs/SOURCES.md): one stream at a time per camera, short clips,
an honest user agent, no retries in tight loops.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

USER_AGENT = "Mozilla/5.0 (compatible; ProjectQuack-CCTV/0.1; student traffic research)"


# --------------------------------------------------------------------------- HTTP
def fetch_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https URLs from manifest)
        return resp.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


# ------------------------------------------------------------------ HLS playlists
def parse_master_playlist(text: str) -> list[dict]:
    """Return variants: [{bandwidth, codecs, width, height, uri}]."""
    variants, lines = [], [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.startswith("#EXT-X-STREAM-INF:"):
            attrs = ln.split(":", 1)[1]
            bw = re.search(r"BANDWIDTH=(\d+)", attrs)
            res = re.search(r"RESOLUTION=(\d+)x(\d+)", attrs)
            codecs = re.search(r'CODECS="([^"]+)"', attrs)
            uri = lines[i + 1] if i + 1 < len(lines) else ""
            variants.append({
                "bandwidth": int(bw.group(1)) if bw else None,
                "codecs": codecs.group(1) if codecs else None,
                "width": int(res.group(1)) if res else None,
                "height": int(res.group(2)) if res else None,
                "uri": uri,
            })
    return variants


def parse_media_playlist(text: str) -> dict:
    """Return {target_duration, media_sequence, segments: [(duration_s, uri)]}."""
    out = {"target_duration": None, "media_sequence": None, "segments": []}
    dur = None
    for ln in (x.strip() for x in text.splitlines()):
        if ln.startswith("#EXT-X-TARGETDURATION:"):
            out["target_duration"] = float(ln.split(":", 1)[1])
        elif ln.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            out["media_sequence"] = int(ln.split(":", 1)[1])
        elif ln.startswith("#EXTINF:"):
            dur = float(ln.split(":", 1)[1].split(",")[0])
        elif ln and not ln.startswith("#") and dur is not None:
            out["segments"].append((dur, ln))
            dur = None
    return out


# ---------------------------------------------------------------------- recording
@dataclass
class Recording:
    camera_id: str
    url: str
    path: str
    requested_seconds: float
    started_utc: str
    start_wall: float
    end_wall: float
    segment_arrivals: list[tuple[float, int]] = field(default_factory=list)  # (wall time, media seq)
    returncode: int = 0
    stderr_tail: str = ""

    def save_sidecar(self) -> Path:
        side = Path(self.path).with_suffix(".json")
        side.write_text(json.dumps(asdict(self), indent=1))
        return side


SEG_RE = re.compile(r"Opening '([^']+?_(\d+)\.ts)' for reading")


def record_hls(url: str, out_path: Path | str, seconds: float, camera_id: str = "",
               timeout_pad: float = 30.0) -> Recording:
    """Record ``seconds`` of an HLS stream to ``out_path`` (.ts) without re-encoding.

    Segment arrival times are taken from ffmpeg's verbose log as it runs, which
    tells us how often the server publishes new segments (no extra requests).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "verbose", "-y",
        "-user_agent", USER_AGENT, "-rw_timeout", "15000000",
        "-i", url, "-t", f"{seconds}", "-map", "0:v:0", "-c", "copy", str(out_path),
    ]
    rec = Recording(camera_id=camera_id, url=url, path=str(out_path), requested_seconds=seconds,
                    started_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    start_wall=time.time(), end_wall=0.0)
    tail: list[str] = []
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
                            bufsize=1)
    deadline = time.time() + seconds + timeout_pad
    assert proc.stderr is not None
    for line in proc.stderr:
        now = time.time()
        m = SEG_RE.search(line)
        if m:
            rec.segment_arrivals.append((round(now, 3), int(m.group(2))))
        tail = (tail + [line.rstrip()])[-15:]
        if now > deadline:
            proc.kill()
            break
    rec.returncode = proc.wait()
    rec.end_wall = time.time()
    rec.stderr_tail = "\n".join(tail)
    return rec


# ------------------------------------------------------------------- frame access
def probe_video(path: Path | str) -> dict:
    """Basic stream info via ffprobe (width, height, nominal fps, codec)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {"codec": s.get("codec_name"), "profile": s.get("profile"), "width": s["width"],
            "height": s["height"], "nominal_fps": num / den if den else None}


def frame_times(path: Path | str) -> np.ndarray:
    """Presentation timestamps (seconds) of every video frame, in display order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    pts = [float(x.split(",")[0]) for x in out.split() if x and x.split(",")[0] not in ("", "N/A")]
    return np.sort(np.asarray(pts, dtype=float))


def iter_frames(path: Path | str, max_frames: int | None = None
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(index, t_seconds, frame_bgr)`` with t relative to the first frame."""
    import cv2  # local import so pure modules don't need OpenCV

    pts = frame_times(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    t0 = pts[0] if len(pts) else 0.0
    i = 0
    try:
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # If decode count and PTS count disagree, fall back to nominal spacing for the tail.
            t = float(pts[i] - t0) if i < len(pts) else float(i / 15.0)
            yield i, t, frame
            i += 1
    finally:
        cap.release()


def decode_gray_small(path: Path | str, width: int = 160, height: int = 120) -> np.ndarray:
    """Decode the whole clip to small grayscale frames (N, H, W) uint8 via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:{height},format=gray",
         "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (width * height)
    return np.frombuffer(raw, np.uint8)[: n * width * height].reshape(n, height, width)


def main(argv: list[str] | None = None) -> None:
    """Record one clip from a manifest camera into data/clips/ (with a .json sidecar)."""
    import argparse

    from cctv.registry import DATA_DIR, get_camera

    ap = argparse.ArgumentParser(description="Record a short HLS clip from an LIE camera.")
    ap.add_argument("--camera", required=True, help="camera id, e.g. R10_041")
    ap.add_argument("--seconds", type=float, default=60.0, help="keep clips short (<= 60 s)")
    args = ap.parse_args(argv)
    if args.seconds > 120:
        ap.error("please keep research clips short (<= 120 s)")
    cam = get_camera(args.camera)
    if cam["feed"]["type"] != "hls":
        ap.error(f"{args.camera} is a still-image camera; only HLS video cameras can be recorded")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = DATA_DIR / "clips" / f"{cam['id']}_{stamp}.ts"
    rec = record_hls(cam["feed"]["url"], out, args.seconds, camera_id=cam["id"])
    side = rec.save_sidecar()
    print(f"rc={rec.returncode} clip={out} ({out.stat().st_size / 1e6:.1f} MB) sidecar={side}"
          if out.exists() else f"recording failed rc={rec.returncode}\n{rec.stderr_tail}")


if __name__ == "__main__":
    main()
