"""
Point-in-mesh containment by ray parity, in numpy -- no `rtree`.

`trimesh.Trimesh.contains` builds its ray accelerator through
`mesh.triangles_tree`, which imports `rtree`.  The training image has
manifold3d, trimesh and networkx but NOT rtree, so every containment test in
`mesh_iou` -- cavity orientation, interior probe points -- raised
ModuleNotFoundError there and the module degraded to a
refusal.  This is a drop-in replacement built on the same broad-phase idea the
self-intersection detector already uses.

    from volumetric_iou.inside import contains, PointInMesh

    mask = contains(mesh, points)                  # bool (n,)
    inside, resolved = PointInMesh(mesh).query(points)

METHOD.  All rays share one direction `d`.  Vertices are projected onto an
orthonormal frame (u, v, d), so a ray becomes a single 2D point (p, q) and a
triangle becomes a 2D triangle plus a height along `d`.  Counting forward hits
is then a 2D point-in-triangle test, and the parity of that count answers
containment.  Candidates come from a one-dimensional uniform grid over the
`u` axis, so the test never expands the full points x faces product.

WHY A GENERIC DIRECTION.  CAD geometry is axis-aligned, and an axis-aligned ray
hits shared edges and vertices constantly -- each one counted twice or not at
all.  The default directions are irrational-looking unit vectors with no
special relationship to the axes, which pushes those coincidences to measure
zero.

UNRESOLVED POINTS.  A ray that grazes an edge, or a point that lies exactly ON
the surface, has no defined parity.  Rather than return a coin flip, such a
point is retried with the next direction and, if it stays ambiguous, reported
as unresolved.  `contains()` maps unresolved to False -- "not confirmed
inside", the reading every caller in `mesh_iou` already treats conservatively.
That is a deliberate difference from `trimesh.contains`, which always answers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

# Ray directions tried in order.  Generic, mutually unrelated, unit length --
# see WHY A GENERIC DIRECTION above.  A point still ambiguous after all of them
# is reported unresolved rather than guessed.
_DIRECTIONS = np.array(
    [
        [0.31215, 0.57113, 0.75907],
        [-0.68301, 0.42262, 0.59563],
        [0.49321, -0.79117, 0.36097],
        [0.13793, 0.29231, -0.94631],
        [-0.55719, -0.61847, 0.55396],
    ]
)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True)

# Relative tolerances.  `_TOL_BARY` is on normalised barycentric coordinates,
# so it is dimensionless; `_TOL_T` is relative to mesh scale.
_TOL_BARY = 1e-9
_TOL_T = 1e-9

# Cap on (points x candidate triangles) handled in one vectorised block.
_BLOCK = 4_000_000


def _frame(direction: np.ndarray) -> np.ndarray:
    """Orthonormal (u, v, d) as rows, with d the ray direction."""
    d = direction / np.linalg.norm(direction)
    # any axis not nearly parallel to d gives a stable first cross product
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return np.stack([u, v, d])


class PointInMesh:
    """
    Reusable containment tester for one mesh.

    Building the per-direction index costs a sort over the faces, so callers
    that probe the same mesh repeatedly (nesting parity walks every candidate
    shell pair) should keep the object rather than calling `contains` again.
    """

    def __init__(self, mesh: trimesh.Trimesh, max_directions: int = 3):
        self.mesh = mesh
        self.max_directions = max(1, min(max_directions, len(_DIRECTIONS)))
        self.scale = float(mesh.scale) if len(mesh.vertices) else 1.0
        if not np.isfinite(self.scale) or self.scale <= 0:
            self.scale = 1.0
        self._cache: dict[int, tuple] = {}

    # ------------------------------------------------------------------ #
    # index
    # ------------------------------------------------------------------ #

    def _projection(self, k: int):
        """Triangles projected into frame k, plus a grid over the u axis."""
        if k in self._cache:
            return self._cache[k]

        frame = _frame(_DIRECTIONS[k])
        tri = self.mesh.triangles @ frame.T          # (m, 3, 3) -> u, v, d
        a, b, t = tri[..., 0], tri[..., 1], tri[..., 2]

        a_lo, a_hi = a.min(axis=1), a.max(axis=1)
        b_lo, b_hi = b.min(axis=1), b.max(axis=1)

        m = len(a_lo)
        # One bucket per ~8 triangles, bounded: fewer buckets means longer
        # candidate lists, more buckets means a longer build for no gain.
        n_cells = int(np.clip(m // 8 + 1, 1, 8192))
        lo, hi = float(a_lo.min()), float(a_hi.max())
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        step = span / n_cells

        c_lo = np.clip(((a_lo - lo) / step).astype(np.int64), 0, n_cells - 1)
        c_hi = np.clip(((a_hi - lo) / step).astype(np.int64), 0, n_cells - 1)
        widths = c_hi - c_lo + 1

        # CSR: cell -> triangle ids, built by expanding each triangle over the
        # cells its u-interval spans, then counting-sorting on the cell id.
        tri_ids = np.repeat(np.arange(m, dtype=np.int64), widths)
        offs = np.arange(widths.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(widths) - widths, widths
        )
        cell_ids = np.repeat(c_lo, widths) + offs
        order = np.argsort(cell_ids, kind="stable")
        starts = np.zeros(n_cells + 1, dtype=np.int64)
        starts[1:] = np.cumsum(np.bincount(cell_ids, minlength=n_cells))

        out = (frame, a, b, t, a_lo, a_hi, b_lo, b_hi,
               tri_ids[order], starts, lo, step, n_cells)
        self._cache[k] = out
        return out

    # ------------------------------------------------------------------ #
    # query
    # ------------------------------------------------------------------ #

    def _query_one(self, points: np.ndarray, k: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        """(inside, resolved) for one ray direction."""
        (frame, a, b, t, a_lo, a_hi, b_lo, b_hi,
         cell_tris, starts, lo, step, n_cells) = self._projection(k)

        pr = points @ frame.T
        p, q, pt = pr[:, 0], pr[:, 1], pr[:, 2]

        crossings = np.zeros(len(points), dtype=np.int64)
        ambiguous = np.zeros(len(points), dtype=bool)

        cells = np.clip(((p - lo) / step).astype(np.int64), 0, n_cells - 1)
        # points outside the mesh's u range have no candidates at all
        outside_u = (p < a_lo.min()) | (p > a_hi.max())

        order = np.argsort(cells, kind="stable")
        order = order[~outside_u[order]]
        if len(order) == 0:
            return np.zeros(len(points), dtype=bool), np.ones(len(points), dtype=bool)

        cell_sorted = cells[order]
        bounds = np.searchsorted(cell_sorted, np.arange(n_cells + 1), side="left")

        for cell in np.unique(cell_sorted):
            pi = order[bounds[cell]:bounds[cell + 1]]
            ti = cell_tris[starts[cell]:starts[cell + 1]]
            if len(pi) == 0 or len(ti) == 0:
                continue
            # block over points so (points x candidates) stays bounded
            chunk = max(1, _BLOCK // len(ti))
            for s in range(0, len(pi), chunk):
                sub = pi[s:s + chunk]
                self._accumulate(sub, ti, p, q, pt, a, b, t,
                                 a_lo, a_hi, b_lo, b_hi, crossings, ambiguous)

        inside = (crossings % 2) == 1
        resolved = ~ambiguous
        return inside & resolved, resolved

    def _accumulate(self, pi, ti, p, q, pt, a, b, t,
                    a_lo, a_hi, b_lo, b_hi, crossings, ambiguous) -> None:
        """Count forward hits for points `pi` against candidate triangles `ti`."""
        pp = p[pi][:, None]
        qq = q[pi][:, None]

        box = ((pp >= a_lo[ti]) & (pp <= a_hi[ti])
               & (qq >= b_lo[ti]) & (qq <= b_hi[ti]))
        rows, cols = np.nonzero(box)
        if len(rows) == 0:
            return

        idx_pt = pi[rows]
        idx_tri = ti[cols]

        a0, a1, a2 = a[idx_tri, 0], a[idx_tri, 1], a[idx_tri, 2]
        b0, b1, b2 = b[idx_tri, 0], b[idx_tri, 1], b[idx_tri, 2]
        px, py = p[idx_pt], q[idx_pt]

        # twice the signed 2D areas of the three sub-triangles
        w0 = (a1 - px) * (b2 - py) - (a2 - px) * (b1 - py)
        w1 = (a2 - px) * (b0 - py) - (a0 - px) * (b2 - py)
        w2 = (a0 - px) * (b1 - py) - (a1 - px) * (b0 - py)
        area = w0 + w1 + w2

        # A triangle parallel to the ray projects to a segment: it cannot be
        # pierced, but a point landing on that segment has no defined parity.
        scale2 = self.scale ** 2
        degenerate = np.abs(area) <= _TOL_BARY * scale2
        if degenerate.any():
            np.logical_or.at(ambiguous, idx_pt[degenerate], True)

        live = ~degenerate
        if not live.any():
            return
        idx_pt, idx_tri = idx_pt[live], idx_tri[live]
        u0 = w0[live] / area[live]
        u1 = w1[live] / area[live]
        u2 = w2[live] / area[live]

        near_edge = ((np.abs(u0) <= _TOL_BARY) | (np.abs(u1) <= _TOL_BARY)
                     | (np.abs(u2) <= _TOL_BARY))
        if near_edge.any():
            np.logical_or.at(ambiguous, idx_pt[near_edge], True)

        strict = (u0 > _TOL_BARY) & (u1 > _TOL_BARY) & (u2 > _TOL_BARY)
        if not strict.any():
            return
        idx_pt, idx_tri = idx_pt[strict], idx_tri[strict]
        u0, u1, u2 = u0[strict], u1[strict], u2[strict]

        t_hit = (u0 * t[idx_tri, 0] + u1 * t[idx_tri, 1] + u2 * t[idx_tri, 2])
        delta = t_hit - pt[idx_pt]

        # |delta| ~ 0 means the point sits on this very face: containment is
        # undefined there, and saying "outside" would be a coin flip.
        on_face = np.abs(delta) <= _TOL_T * self.scale
        if on_face.any():
            np.logical_or.at(ambiguous, idx_pt[on_face], True)

        forward = delta > _TOL_T * self.scale
        if forward.any():
            np.add.at(crossings, idx_pt[forward], 1)

    def query(self, points) -> tuple[np.ndarray, np.ndarray]:
        """
        (inside, resolved) for `points`, retrying ambiguous ones.

        A point is retried with the next ray direction only while its parity
        stays undefined; a point answered on the first direction is never
        recomputed.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        n = len(pts)
        inside = np.zeros(n, dtype=bool)
        resolved = np.zeros(n, dtype=bool)
        if n == 0 or len(self.mesh.faces) == 0:
            return inside, np.ones(n, dtype=bool)

        todo = np.arange(n)
        for k in range(self.max_directions):
            got_in, got_ok = self._query_one(pts[todo], k)
            done = got_ok
            inside[todo[done]] = got_in[done]
            resolved[todo[done]] = True
            todo = todo[~done]
            if len(todo) == 0:
                break
        return inside, resolved


def contains(mesh: trimesh.Trimesh, points, *, max_directions: int = 3
             ) -> np.ndarray:
    """
    Boolean mask: which `points` lie strictly inside `mesh`.

    Drop-in for `trimesh.Trimesh.contains` minus the `rtree` dependency, with
    one deliberate difference: a point whose parity stays undefined after every
    retry is reported False ("not confirmed inside") instead of a guess.
    """
    inside, _ = PointInMesh(mesh, max_directions=max_directions).query(points)
    return inside


def contains_resolved(mesh: trimesh.Trimesh, points, *, max_directions: int = 3
                      ) -> tuple[np.ndarray, np.ndarray]:
    """`(inside, resolved)`; see `PointInMesh.query`."""
    return PointInMesh(mesh, max_directions=max_directions).query(points)
