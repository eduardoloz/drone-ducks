"""Camera views (PTZ presets): several calibrations per camera, and picking the right one.

NYSDOT LIE cameras are pan-tilt-zoom. Operators re-aim them at incidents and
then (usually) send them back to a preset. A homography is only valid for the
view it was fitted on, so a camera holds one calibration *per view*, and every
clip is matched against all known views before any speed is computed.

On-disk layout (``data/calibration/``)::

    <CAM>/<VIEW>.json            calibration (optional: a view can be known but not calibrated yet)
    <CAM>/<VIEW>.points.json     clicked points the calibration was fitted from (optional)
    <CAM>/<VIEW>.ref.jpg         reference frame for matching (at least one needed to match)
    <CAM>/<VIEW>.ref-<tag>.jpg   extra reference frames for the same view, e.g. ``ref-night``
    <CAM>/views.json             optional labels/notes per view: {"views": {"<VIEW>": {"label": ...}}}

The round-1 flat layout still works and is read as the view ``default``::

    <CAM>.json  <CAM>.points.json  <CAM>.ref.jpg

``python -m cctv.calibrate migrate --camera <CAM> --view <VIEW>`` moves a flat
calibration into the per-view layout.

Matching: ORB + RANSAC against every reference frame of every view (see
``camera_motion.py``). Pairs whose shift/zoom/rotation are inside the
thresholds are candidates; the one with the most RANSAC inliers wins. If no
view matches, the clip is from an unknown view and gets no speeds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cctv.geometry import Calibration
from cctv.registry import CALIBRATION_DIR

LEGACY_VIEW = "default"
VIEW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


@dataclass
class CameraView:
    camera_id: str
    view_id: str
    calibration_path: Path | None = None
    points_path: Path | None = None
    reference_paths: list[Path] = field(default_factory=list)
    label: str = ""
    legacy: bool = False            # True for the round-1 flat files <CAM>.json / <CAM>.ref.jpg

    @property
    def calibrated(self) -> bool:
        return self.calibration_path is not None

    def load_calibration(self) -> Calibration | None:
        return Calibration.load(self.calibration_path) if self.calibration_path else None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"view_id": self.view_id, "label": self.label, "calibrated": self.calibrated,
                               "references": [p.name for p in self.reference_paths], "legacy": self.legacy}
        if self.calibration_path:
            d = json.loads(self.calibration_path.read_text())
            out.update({k: d.get(k) for k in ("created_utc", "rmse_m", "scale_uncertainty", "direction_labels")})
        return out


def validate_view_id(view_id: str) -> str:
    if not VIEW_ID_RE.match(view_id) or view_id == "views":
        raise ValueError(f"bad view id {view_id!r}: use letters, digits, '_' or '-', up to 40 chars")
    return view_id


def camera_dir(camera_id: str, cal_dir: Path = CALIBRATION_DIR) -> Path:
    return Path(cal_dir) / camera_id


def _ref_sort_key(p: Path) -> tuple[int, str]:
    return (0 if p.name.endswith(".ref.jpg") else 1, p.name)


def list_views(camera_id: str, cal_dir: Path | str = CALIBRATION_DIR) -> list[CameraView]:
    """All known views of a camera: per-view directory first (sorted by id), then the legacy flat files."""
    cal_dir = Path(cal_dir)
    views: dict[str, CameraView] = {}
    d = camera_dir(camera_id, cal_dir)
    labels: dict[str, Any] = {}
    if d.is_dir():
        meta = d / "views.json"
        if meta.exists():
            labels = json.loads(meta.read_text()).get("views", {})
        for p in sorted(d.iterdir()):
            name = p.name
            if name == "views.json":
                continue
            m = re.match(r"^(.+?)\.(points\.json|json|ref\.jpg|ref-[A-Za-z0-9_-]+\.jpg)$", name)
            if not m:
                continue
            vid, kind = m.group(1), m.group(2)
            v = views.setdefault(vid, CameraView(camera_id, vid))
            if kind == "json":
                v.calibration_path = p
            elif kind == "points.json":
                v.points_path = p
            else:
                v.reference_paths.append(p)
        for vid, info in labels.items():
            views.setdefault(vid, CameraView(camera_id, vid)).label = (info or {}).get("label", "")
    out = sorted(views.values(), key=lambda v: v.view_id)
    for v in out:
        v.reference_paths.sort(key=_ref_sort_key)
    # Round-1 flat layout -> view "default" (unless the directory already defines one).
    flat_cal, flat_pts, flat_ref = (cal_dir / f"{camera_id}{s}" for s in (".json", ".points.json", ".ref.jpg"))
    if LEGACY_VIEW not in views and (flat_cal.exists() or flat_ref.exists()):
        out.append(CameraView(camera_id, LEGACY_VIEW,
                              calibration_path=flat_cal if flat_cal.exists() else None,
                              points_path=flat_pts if flat_pts.exists() else None,
                              reference_paths=[flat_ref] if flat_ref.exists() else [],
                              label="round-1 calibration (flat layout)", legacy=True))
    return out


def get_view(camera_id: str, view_id: str, cal_dir: Path | str = CALIBRATION_DIR) -> CameraView:
    for v in list_views(camera_id, cal_dir):
        if v.view_id == view_id:
            return v
    raise KeyError(f"camera {camera_id} has no view {view_id!r}")


def primary_view(camera_id: str, cal_dir: Path | str = CALIBRATION_DIR) -> CameraView | None:
    """The view shown by default: the first calibrated one, else the first with a reference frame."""
    views = list_views(camera_id, cal_dir)
    return next((v for v in views if v.calibrated), None) or next((v for v in views if v.reference_paths), None)


def set_view_label(camera_id: str, view_id: str, label: str, cal_dir: Path | str = CALIBRATION_DIR,
                   **extra: Any) -> Path:
    d = camera_dir(camera_id, Path(cal_dir))
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "views.json"
    data = json.loads(meta.read_text()) if meta.exists() else {"camera_id": camera_id, "views": {}}
    data.setdefault("views", {}).setdefault(view_id, {}).update({"label": label, **extra})
    meta.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return meta


# ---------------------------------------------------------------------- matching
@dataclass
class ViewMatch:
    status: str                       # "matched" | "no_match" | "no_views" | "unreadable"
    view: CameraView | None = None    # matched view (may be uncalibrated)
    reference: str | None = None      # file name of the reference frame that matched
    result: dict[str, Any] | None = None  # MotionResult of the winning (or best failing) comparison
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def calibration(self) -> Calibration | None:
        return self.view.load_calibration() if (self.status == "matched" and self.view) else None

    @property
    def usable(self) -> bool:
        """True when speeds can be computed: a view matched and it has a calibration."""
        return self.status == "matched" and self.view is not None and self.view.calibrated

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "view_id": self.view.view_id if self.view else None,
                "calibrated": bool(self.view and self.view.calibrated), "reference": self.reference,
                "reason": self.reason, "result": self.result, "candidates": self.candidates}


def _score(r: dict[str, Any]) -> float:
    return r["inlier_ratio"] * r["n_matches"]


def match_view(camera_id: str, frame: np.ndarray, views: list[CameraView] | None = None,
               cal_dir: Path | str = CALIBRATION_DIR, **detector_kw: Any) -> ViewMatch:
    """Compare ``frame`` with every reference of every view and pick the matching view."""
    import cv2

    from cctv.camera_motion import DEFAULT_OSD_BOXES, CameraMotionDetector

    views = list_views(camera_id, cal_dir) if views is None else views
    views = [v for v in views if v.reference_paths]
    if not views:
        return ViewMatch("no_views", reason="no reference frames for this camera: view not checked")
    luma_cur = float(frame.mean())
    cands: list[dict[str, Any]] = []
    for v in views:
        cal = v.load_calibration()
        boxes = [tuple(b) for b in cal.osd_boxes] if cal and cal.osd_boxes else DEFAULT_OSD_BOXES
        for ref_path in v.reference_paths:
            ref = cv2.imread(str(ref_path))
            if ref is None:
                continue
            r = CameraMotionDetector(ref, osd_boxes=boxes, **detector_kw).check(frame).to_dict()
            r.update(view_id=v.view_id, reference=ref_path.name, luma_ref=round(float(ref.mean()), 1))
            cands.append(r)
    if not cands:
        return ViewMatch("unreadable", reason="could not read any reference frame")
    ok = [c for c in cands if not c["moved"]]
    by_id = {v.view_id: v for v in views}
    if ok:
        best = max(ok, key=_score)
        view = by_id[best["view_id"]]
        why = f"matches view {view.view_id} ({best['reference']}): shift {best['shift_px']:.1f}px, " \
              f"zoom x{best['scale']:.3f}, {best['n_matches']} matches"
        if not view.calibrated:
            why += "; this view is not calibrated, so no speeds"
        return ViewMatch("matched", view, best["reference"], best, cands, why)
    best = max(cands, key=lambda c: (_score(c), -np.nan_to_num(c["shift_px"], nan=1e9)))
    why = f"matches none of {len(views)} known view(s); closest {best['view_id']}: {best['reason']}"
    if luma_cur < 0.6 * best["luma_ref"]:
        why += " (clip much darker than the reference: verify by eye, or add a night reference for this view)"
    return ViewMatch("no_match", None, None, best, cands, why)
