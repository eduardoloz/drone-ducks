"""Ground-plane geometry: image pixels <-> road-plane metres.

A camera looking at a flat road is a *homography*: a 3x3 matrix H that maps an
image point (u, v) to a road point (X, Y) in metres:

    [X*w, Y*w, w]^T = H @ [u, v, 1]^T

We fit H from >= 4 point pairs whose road positions we know from standard road
geometry (see ``road_points_from_dashes``), e.g. the corners of dashed lane
lines: on US interstates each dash is 10 ft long followed by a 30 ft gap, and
lanes are 12 ft wide (citations in docs/SOURCES.md).

Road frame convention used everywhere in this package:
    X = metres *along* the road (direction of the lane lines)
    Y = metres *across* the road (lateral; lane k centre at (k + 0.5) * lane_width)

Only numpy is required. The vehicle point we map is the bottom-centre of its
bounding box, which is close to where the vehicle touches the road.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FT = 0.3048                  # metres per foot (exact)
MPH_PER_MPS = 2.2369362920544  # 1 m/s in mph
DASH_FT = 10.0               # broken lane line segment (MUTCD)
GAP_FT = 30.0                # gap between segments (MUTCD)
LANE_FT = 12.0               # interstate lane width


# --------------------------------------------------------------------- core math
def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return ph[:, :2], T


def fit_homography(img_pts, road_pts) -> np.ndarray:
    """Least-squares homography (normalised DLT) mapping image -> road.

    ``img_pts`` and ``road_pts`` are (N, 2) with N >= 4 and not all collinear.
    """
    src = np.asarray(img_pts, dtype=float)
    dst = np.asarray(road_pts, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("img_pts and road_pts must both be (N, 2)")
    if len(src) < 4:
        raise ValueError("need at least 4 point pairs")
    s, Ts = _normalise(src)
    d, Td = _normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows)
    _, sv, vt = np.linalg.svd(A)
    if sv[-2] < 1e-10:
        raise ValueError("degenerate point configuration (collinear points?)")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """Map (N, 2) points through H. Returns (N, 2)."""
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
    w = ph[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point maps to infinity (above the horizon?)")
    return ph[:, :2] / w


def homography_jacobian(H: np.ndarray, pts) -> np.ndarray:
    """Jacobian d(X, Y)/d(u, v) of the image->road map at each point: (N, 2, 2).

    Row 0 is how X changes per pixel in u and v, row 1 the same for Y. Used to
    convert pixel noise into metres of road (perspective makes far pixels big).
    """
    H = np.asarray(H, dtype=float)
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    ph = np.c_[p, np.ones(len(p))]
    num = ph @ H[:2].T                      # (N, 2): h1.p, h2.p
    w = ph @ H[2]                           # (N,)
    J = (H[None, :2, :2] * w[:, None, None] - num[:, :, None] * H[2, :2][None, None, :])
    return J / (w ** 2)[:, None, None]


def reprojection_errors(H: np.ndarray, img_pts, road_pts) -> np.ndarray:
    """Per-point distance (metres) between mapped image points and true road points."""
    return np.linalg.norm(apply_homography(H, img_pts) - np.asarray(road_pts, float), axis=1)


def local_scale(H: np.ndarray, pt, eps: float = 0.5) -> tuple[float, float]:
    """Metres per pixel at image point ``pt`` for a 1-px step in u and in v.

    Used in the error budget: a 1-px error in the bbox bottom far from the
    camera can be a metre or more on the road.
    """
    u, v = float(pt[0]), float(pt[1])
    p = apply_homography(H, [[u - eps, v], [u + eps, v], [u, v - eps], [u, v + eps]])
    du = np.linalg.norm(p[1] - p[0]) / (2 * eps)
    dv = np.linalg.norm(p[3] - p[2]) / (2 * eps)
    return float(du), float(dv)


def road_points_from_dashes(line_index: int, dash_index: int, end: str,
                            lane_width_m: float = LANE_FT * FT,
                            dash_m: float = DASH_FT * FT, gap_m: float = GAP_FT * FT) -> tuple[float, float]:
    """Road coordinates of a dash corner.

    ``line_index``: 0 for the first lane line you use, 1 for the next one across, ...
    ``dash_index``: 0 for the first dash you use along that line, 1 for the next, ...
    ``end``: "start" or "end" of the dash (along the direction of increasing X).

    Assumes dashes on neighbouring lines are aligned (true on most freeways, but
    check! staggered dashes break this; then use one line + lane width only).
    """
    x = dash_index * (dash_m + gap_m) + (dash_m if end == "end" else 0.0)
    return x, line_index * lane_width_m


# ------------------------------------------------------------------ calibration
@dataclass
class Calibration:
    camera_id: str
    image_size: tuple[int, int]               # (width, height)
    img_pts: list[list[float]]
    road_pts: list[list[float]]
    H: list[list[float]] = field(default_factory=list)
    rmse_m: float = float("nan")
    max_err_m: float = float("nan")
    scale_uncertainty: float = 0.03            # relative 1-sigma scale error used for speed sigma
    reference_frame: str | None = None         # path (gitignored) of the frame the points came from
    osd_boxes: list[list[float]] | None = None  # overlay boxes (fractions) for motion checks
    roi_polygon: list[list[float]] | None = None  # image polygon where speeds are trusted
    created_utc: str = ""
    method: str = ""
    notes: str = ""

    @classmethod
    def fit(cls, camera_id: str, image_size, img_pts, road_pts, **kw) -> "Calibration":
        H = fit_homography(img_pts, road_pts)
        err = reprojection_errors(H, img_pts, road_pts)
        return cls(camera_id=camera_id, image_size=tuple(image_size),
                   img_pts=np.asarray(img_pts, float).tolist(),
                   road_pts=np.asarray(road_pts, float).tolist(), H=H.tolist(),
                   rmse_m=float(np.sqrt(np.mean(err ** 2))), max_err_m=float(err.max()),
                   created_utc=kw.pop("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                   **kw)

    @property
    def H_np(self) -> np.ndarray:
        return np.asarray(self.H, dtype=float)

    def image_to_road(self, pts) -> np.ndarray:
        return apply_homography(self.H_np, pts)

    def road_to_image(self, pts) -> np.ndarray:
        return apply_homography(np.linalg.inv(self.H_np), pts)

    def in_roi(self, pts) -> np.ndarray:
        """Boolean mask: which image points fall inside the trusted ROI polygon."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if not self.roi_polygon:
            return np.ones(len(pts), dtype=bool)
        return points_in_polygon(pts, np.asarray(self.roi_polygon, float))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "Calibration":
        d = json.loads(Path(path).read_text())
        d["image_size"] = tuple(d["image_size"])
        return cls(**d)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule point-in-polygon for (N, 2) points and an (M, 2) polygon."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return (cross.sum(axis=1) % 2) == 1
