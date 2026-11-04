"""
Generalized winding number and a point-sampling IoU arbiter.

The winding number w(p) of a closed, oriented triangle mesh counts how many
times the surface wraps around p: 1 inside a single outward solid, 0 outside
and inside a cavity, 2 where two bodies (or two passes of one self-overlapping
shell) overlap.  The metric's definition of a solid is the point set w >= 1
(docs/DEFINITION.md), and this module evaluates that definition directly --
no boolean engine, no ray parity, no `split`.

w(p) = (1 / 4 pi) * sum of the signed solid angles of the triangles seen from
p (Van Oosterom & Strackee, 1983).  Cost is O(points x faces), so this is a
checking tool, not the hot path: 30k points on a 100k-face mesh take minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

# points x faces handled per block; bounds peak memory to a few hundred MB
BLOCK = 4_000_000
# below this many point x face pairs (~seconds serially) a pool costs more than it saves
PARALLEL_MIN = 50_000_000


def winding_number(mesh: trimesh.Trimesh, points: np.ndarray,
                   workers: int = 1) -> np.ndarray:
    """
    Generalized winding number of `mesh` at each of `points` (float, ~integer).

    workers > 1 splits the points over a forkserver process pool (the sum per
    point is unchanged, so the answer is identical to the serial one); small
    jobs stay in-process.  forkserver re-imports the caller's main module, so
    a calling script needs the `if __name__ == "__main__":` guard.
    """
    tri = np.asarray(mesh.triangles, dtype=np.float64)          # (F, 3, 3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if workers > 1 and len(tri) * len(pts) >= PARALLEL_MIN:
        return _winding_parallel(tri, pts, workers)
    return _winding_serial(tri, pts)


def _winding_serial(tri: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(pts))
    if len(tri) == 0 or len(pts) == 0:
        return out
    step = max(1, BLOCK // len(tri))
    for s in range(0, len(pts), step):
        p = pts[s:s + step]
        a = tri[None, :, 0, :] - p[:, None, :]                  # (P, F, 3)
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        det = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = (la * lb * lc
               + np.einsum("pfi,pfi->pf", a, b) * lc
               + np.einsum("pfi,pfi->pf", b, c) * la
               + np.einsum("pfi,pfi->pf", c, a) * lb)
        out[s:s + step] = np.arctan2(det, den).sum(axis=1) / (2.0 * np.pi)
    return out


_worker_tri: Optional[np.ndarray] = None


def _init_worker(tri: np.ndarray) -> None:
    global _worker_tri
    _worker_tri = tri


def _winding_chunk(pts: np.ndarray) -> np.ndarray:
    return _winding_serial(_worker_tri, pts)


def _winding_parallel(tri: np.ndarray, pts: np.ndarray, workers: int) -> np.ndarray:
    # forkserver, not fork: callers may already hold an OCCT thread pool;
    # the triangles go to each worker once.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    workers = min(workers, len(pts))
    chunks = np.array_split(pts, workers * 4)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"),
                             initializer=_init_worker, initargs=(tri,)) as ex:
        return np.concatenate(list(ex.map(_winding_chunk, chunks)))


def inside(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Membership under the metric's definition: w >= 1 (threshold 0.5)."""
    return winding_number(mesh, points) >= 0.5


@dataclass
class ArbiterResult:
    iou: Optional[float]
    iog: Optional[float]
    iop: Optional[float]
    se: Optional[float]          # binomial standard error of the IoU estimate
    n_points: int
    n_union: int
    max_winding: dict            # {"gt": max |w| seen, "pred": ...}


def arbiter_iou(gt: trimesh.Trimesh, pred: trimesh.Trimesh, n: int = 30_000,
                seed: int = 1, workers: int = 1) -> ArbiterResult:
    """
    IoU of the w >= 1 solids of `gt` and `pred`, by uniform sampling of their
    joint bounding box.  Independent of any boolean engine; the answer carries
    a standard error, so a disagreement with the exact IoU is significant only
    beyond ~3 se.  `workers` -- see winding_number.
    """
    lo = np.minimum(gt.bounds[0], pred.bounds[0])
    hi = np.maximum(gt.bounds[1], pred.bounds[1])
    rng = np.random.default_rng(seed)
    pts = lo + rng.random((n, 3)) * (hi - lo)
    wg = winding_number(gt, pts, workers)
    wp = winding_number(pred, pts, workers)
    ig, ip = wg >= 0.5, wp >= 0.5
    n_int = int(np.count_nonzero(ig & ip))
    n_uni = int(np.count_nonzero(ig | ip))
    ng, npred = int(ig.sum()), int(ip.sum())
    mx = {"gt": float(np.abs(wg).max(initial=0)), "pred": float(np.abs(wp).max(initial=0))}
    if n_uni == 0:
        return ArbiterResult(None, None, None, None, n, 0, mx)
    iou = n_int / n_uni
    se = float(np.sqrt(iou * (1 - iou) / n_uni))
    return ArbiterResult(iou, n_int / ng if ng else None, n_int / npred if npred else None,
                         se, n, n_uni, mx)
