"""
Cells of space cut by a set of triangles -- closed or open, written once,
twice, both ways or with coincident and crossing faces.

A side the regular preparation cannot read (a shell written on both sides, a
one-sided wall inside a closed surface, docs/DEFINITION.md 4.8, 4.9)
still cuts space into **cells**: connected regions left when every triangle
is cut out; one cell is unbounded (the outer cell, index 0).  What such a
record says about the solid is a statement about its cells, so this module
builds them and the numbers the reading needs:

- every **triangle** t of the cut (the input resolved in exact arithmetic:
  crossings and coplanar overlaps remeshed) separates the cell `below` it
  from the cell `above` it (the side its normal points to); faces lying on
  each other are one triangle with `up` copies along its orientation and
  `down` against it -- its signed multiplicity `m = up - down`;
- the **winding number** of each cell: 0 in the outer cell, and crossing t
  from below to above it drops by m.  Defined (and equal to the generalized
  winding number of the input anywhere in the cell) only when that walk is
  consistent -- every edge of the input balanced; else `winding` is None;
- for a record written on both sides (`up == down == k` on every triangle,
  DEFINITION 4.8 "Is a two-sided record unambiguous"): `k_distance` -- the
  least sum of k along a path of cells from the outer cell, `parity` -- that
  sum mod 2 along any path, and `filled` -- the cells full under every set of
  bodies the record may come from (fact 4: odd parity, or a triangle to a
  neighbour cell a with k above the k-distance of a).

Vertices are merged as cleaning merges them (DEFINITION 4.2: the 1e-8 grid,
the first occurrence keeps its exact coordinates), so a copy of a face shifted
by noise below the grid is the same face; the cut itself is exact
(`_cgal.cells`: libigl's remesh_self_intersections and extract_cells, the faces
around an edge in exact angular order).  Faces collapsed by the merge or with
exactly collinear points bound nothing and are dropped.

Not on the metric's path: a mechanism for rare sides, correctness first.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

GRID = 1e8          # cleaning's vertex grid: positions rounded to 1e-8


@dataclass
class Cells:
    """The cells of a set of triangles; see the module docstring.

    Per merged triangle t (arrays of length T): `triangles` (T, 3) into
    `vertices`, in the orientation of its first copy; `up`, `down` -- copies
    along / against it; `below`, `above` -- the cells on its two sides (the
    same cell for a sheet with that cell on both sides).  Per cell (length
    `num_cells`, cell 0 the outer one): `volume`, `point` (a point inside the
    cell; for the outer cell one outside the bounding box), `winding`,
    `k_distance`, `parity`, `filled`.  `face_triangle`, `face_input` -- per
    face of the exact cut: its merged triangle and the input face it was born on
    (the first of the input faces on the same three vertices: one is cut).
    """
    vertices: np.ndarray
    triangles: np.ndarray
    up: np.ndarray
    down: np.ndarray
    below: np.ndarray
    above: np.ndarray
    num_cells: int
    volume: np.ndarray
    point: np.ndarray
    winding: Optional[np.ndarray]
    k_distance: np.ndarray
    parity: Optional[np.ndarray]
    filled: Optional[np.ndarray]
    face_triangle: np.ndarray
    face_input: np.ndarray

    @property
    def multiplicity(self) -> np.ndarray:
        """Signed multiplicity m = up - down of each triangle."""
        return self.up - self.down

    @property
    def k(self) -> np.ndarray:
        """max(up, down): k of a two-sided record, where up == down."""
        return np.maximum(self.up, self.down)

    @property
    def two_sided(self) -> bool:
        """Every triangle written as often one way as the other."""
        return bool(np.all(self.up == self.down))

    @property
    def balanced(self) -> bool:
        """The winding number is defined cell by cell."""
        return self.winding is not None


def merge_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices merged as cleaning merges them: positions rounded to the
    1e-8 grid (half to even), referenced vertices only, the first occurrence
    in index order keeps its exact coordinates.  Faces re-indexed, none dropped."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(f) == 0:
        return np.zeros((0, 3)), f.copy()
    used = np.unique(f.ravel())                       # sorted: index order
    key = np.round(v[used] * GRID)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    # new index of each class: by its first occurrence, in index order
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    remap = np.full(len(v), -1, np.int64)
    remap[used] = rank[inv]
    return v[used[first[order]]].copy(), remap[f]


def _signed_volumes(tri: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0


def _first_hit(origin: np.ndarray, direction: np.ndarray, tri: np.ndarray,
               skip: int) -> float:
    """Distance along the ray to the nearest triangle other than `skip`
    (Moller-Trumbore, float64); inf if none."""
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-300
    ok[skip] = False
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    w = (q @ direction) * inv
    d = np.einsum("ij,ij->i", e2, q) * inv
    eps = 1e-12
    hit = ok & (u >= -eps) & (w >= -eps) & (u + w <= 1 + eps) & (d > 0)
    return float(d[hit].min()) if hit.any() else np.inf


def cells(vertices, faces) -> Cells:
    """The cells of space cut by the triangles (vertices, faces); see the
    module docstring.  Needs the package module `_cgal`."""
    from . import _cgal
    v_in = np.ascontiguousarray(vertices, np.float64).reshape(-1, 3)
    f_in = np.ascontiguousarray(faces, np.int64).reshape(-1, 3)
    vm, fm = merge_vertices(v_in, f_in)
    live = np.flatnonzero((fm[:, 0] != fm[:, 1]) & (fm[:, 1] != fm[:, 2])
                          & (fm[:, 0] != fm[:, 2]))
    # input faces on the same three vertices: one is cut, carrying the count
    # of copies along it and against it -- cut separately, copies crossed by
    # another face may be triangulated differently and leave cells of zero
    # thickness that no longer cancel.  A face written both ways with an edge
    # of no other face (an open sheet, a T-junction) keeps one copy each way:
    # the two close the sheet into a cell of zero thickness, as before
    fl = fm[live]
    if len(fl):
        in_group, in_first = _groups(fl)
        rep = fl[in_first]
        along = _same_cycle(fl, rep[in_group])
        in_up = np.bincount(in_group, weights=along, minlength=len(rep)).astype(np.int64)
        in_down = np.bincount(in_group, weights=~along, minlength=len(rep)).astype(np.int64)
        e = np.sort(np.concatenate([rep[:, [0, 1]], rep[:, [1, 2]], rep[:, [2, 0]]]), axis=1)
        _, e_inv, e_cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        free = (e_cnt[e_inv.reshape(-1)] == 1).reshape(3, -1).any(axis=0)
        two = np.flatnonzero(free & (in_up > 0) & (in_down > 0))
        if len(two):
            rep = np.concatenate([rep, rep[two][:, ::-1]])
            in_first = np.concatenate([in_first, in_first[two]])
            in_up = np.concatenate([in_up, in_down[two]])
            in_down = np.concatenate([in_down, np.zeros(len(two), np.int64)])
            in_down[two] = 0
    else:
        in_first, rep = np.zeros(0, np.int64), fl
        in_up = in_down = np.zeros(0, np.int64)
    V, F, birth, side, n_raw = _cgal.cells(np.ascontiguousarray(vm), np.ascontiguousarray(rep))
    V = np.asarray(V)
    F = np.asarray(F, np.int64)
    side = np.asarray(side, np.int64)
    birth = np.asarray(birth, np.int64)
    face_input = live[in_first[birth]] if len(F) else np.zeros(0, np.int64)
    nf = len(F)

    # coincident faces: the same three vertices of the exact cut
    if nf:
        group, first = _groups(F)
    else:
        group = np.zeros(0, np.int64)
        first = np.zeros(0, np.int64)
    # orientation of each face against its group's first face: +1 / -1
    ref = F[first[group]] if nf else F
    sign = np.where(_same_cycle(F, ref), 1, -1) if nf else np.zeros(0, np.int64)

    # zero-thickness cells: the outward boundary chain of the cell cancels
    # (a cell of positive volume has a non-zero boundary); never the outer cell
    pos, neg = side[:, 0], side[:, 1]
    chain: dict = {}
    for f in range(nf):
        g, s = int(group[f]), int(sign[f])
        chain[(int(neg[f]), g)] = chain.get((int(neg[f]), g), 0) + s   # normal leaves neg
        chain[(int(pos[f]), g)] = chain.get((int(pos[f]), g), 0) - s
    nonzero = np.zeros(n_raw, bool)
    nonzero[0] = True
    for (c, _), val in chain.items():
        if val:
            nonzero[c] = True

    # one triangle per group, between the non-zero cells at the two ends of its stack
    n_tri = int(group.max()) + 1 if nf else 0
    above_of = np.where(sign > 0, pos, neg)
    below_of = np.where(sign > 0, neg, pos)
    tri_below = np.full(n_tri, -1, np.int64)
    tri_above = np.full(n_tri, -1, np.int64)
    for f in range(nf):
        g = group[f]
        for cell, slot in ((below_of[f], tri_below), (above_of[f], tri_above)):
            if nonzero[cell]:
                if slot[g] != -1:
                    raise RuntimeError("cells: a stack of coincident faces with a cell of "
                                       "volume inside it")
                slot[g] = cell
    if np.any(tri_below < 0) or np.any(tri_above < 0):
        raise RuntimeError("cells: a stack of coincident faces without a cell at an end")
    # copies of each cut face: those of its input face, turned to the group's first
    f_up = in_up[birth] if nf else np.zeros(0, np.int64)
    f_down = in_down[birth] if nf else np.zeros(0, np.int64)
    up = np.bincount(group, weights=np.where(sign > 0, f_up, f_down),
                     minlength=n_tri).astype(np.int64)
    down = np.bincount(group, weights=np.where(sign > 0, f_down, f_up),
                       minlength=n_tri).astype(np.int64)
    triangles = F[first] if nf else np.zeros((0, 3), np.int64)

    # renumber the non-zero cells, the outer one stays 0
    new = np.cumsum(nonzero) - 1
    num_cells = int(nonzero.sum())
    below, above = new[tri_below], new[tri_above]

    tri = V[triangles] if n_tri else np.zeros((0, 3, 3))
    tv = _signed_volumes(tri) if n_tri else np.zeros(0)
    volume = np.zeros(num_cells)
    np.add.at(volume, below, tv)                # the normal leaves the cell below
    np.add.at(volume, above, -tv)

    m = up - down
    k = np.maximum(up, down)
    adj: list[list] = [[] for _ in range(num_cells)]
    for t in range(n_tri):
        a, b = int(below[t]), int(above[t])
        adj[a].append((b, t, -1))               # crossing below -> above: w drops by m
        adj[b].append((a, t, +1))
    winding = _walk(adj, num_cells, lambda t, d: d * int(m[t]))
    two = bool(np.all(up == down))
    parity = _walk(adj, num_cells, lambda t, d: int(k[t]), mod=2)

    # k-distance: least sum of k from the outer cell (Dijkstra)
    k_distance = np.full(num_cells, np.iinfo(np.int64).max, np.int64)
    k_distance[0] = 0
    heap = [(0, 0)]
    while heap:
        d, c = heapq.heappop(heap)
        if d > k_distance[c]:
            continue
        for nb, t, _ in adj[c]:
            nd = d + int(k[t])
            if nd < k_distance[nb]:
                k_distance[nb] = nd
                heapq.heappush(heap, (nd, nb))

    filled = None
    if two and parity is not None:
        filled = parity.astype(bool).copy()
        filled[0] = False
        for t in range(n_tri):
            a, b = int(below[t]), int(above[t])
            if a == b:
                continue
            if k[t] > k_distance[a]:
                filled[b] = filled[b] or b != 0
            if k[t] > k_distance[b]:
                filled[a] = filled[a] or a != 0

    point = _points(V, tri, below, above, num_cells)
    return Cells(vertices=V, triangles=triangles, up=up, down=down, below=below,
                 above=above, num_cells=num_cells, volume=volume, point=point,
                 winding=winding, k_distance=k_distance, parity=parity, filled=filled,
                 face_triangle=group.astype(np.int64), face_input=face_input)


def _groups(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group of each face (same vertex set), numbered by first face; the
    first face of each group."""
    _, idx, inv = np.unique(np.sort(F, axis=1), axis=0, return_index=True,
                            return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(idx, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inv], idx[order]


def _same_cycle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows of a run the same cyclic order as the rows of b (same vertex sets)."""
    out = np.zeros(len(a), bool)
    for r in range(3):
        out |= np.all(np.roll(b, r, axis=1) == a, axis=1)
    return out


def _walk(adj, n: int, step, mod: Optional[int] = None) -> Optional[np.ndarray]:
    """Values over cells from 0 at the outer cell, value(nb) = value(c) +
    step(t, d) across each triangle; None if some cycle disagrees.  Cells
    unreachable from the outer cell cannot exist (every cell borders one)."""
    val = np.zeros(n, np.int64)
    seen = np.zeros(n, bool)
    seen[0] = True
    queue = deque([0])
    while queue:
        c = queue.popleft()
        for nb, t, d in adj[c]:
            x = val[c] + step(t, d)
            if mod:
                x %= mod
            if not seen[nb]:
                seen[nb] = True
                val[nb] = x
                queue.append(nb)
            elif val[nb] != x:
                return None
    return val


def _points(V: np.ndarray, tri: np.ndarray, below: np.ndarray, above: np.ndarray,
            n: int) -> np.ndarray:
    """A point inside each cell: off the centroid of its largest boundary
    triangle, half way to the next triangle along the normal into the cell.
    The outer cell: a point beyond the bounding box."""
    pts = np.full((n, 3), np.nan)
    if len(V) == 0:
        pts[0] = 0.0
        return pts
    lo, hi = V.min(axis=0), V.max(axis=0)
    pts[0] = hi + (hi - lo).max() + 1.0
    if len(tri) == 0:
        return pts
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    centre = tri.mean(axis=1)
    for c in range(1, n):
        cand = [(area[t], t, -1.0) for t in np.flatnonzero(below == c) if above[t] != c]
        cand += [(area[t], t, 1.0) for t in np.flatnonzero(above == c) if below[t] != c]
        for _, t, s in sorted(cand, reverse=True):
            d = s * nrm[t] / area[t]
            hit = _first_hit(centre[t], d, tri, t)
            if np.isfinite(hit) and hit > 1e-12:
                pts[c] = centre[t] + 0.5 * hit * d
                break
    return pts
