"""
Exact split of T-junctions, before the crossing detector decides the path.

A T-junction is a vertex of a shell lying exactly strictly inside an edge of
the same shell: a tessellation that splits an edge on one side only, or the
hole of a sliver face that cleaning dropped, closed by a fan over points on
one line (docs/DEFINITION.md 4.2, 4.3).  The detector reports the vertex
against the faces of that edge as a touch, and a touch inside one body sends
the body to the exact CGAL arrangement (4.7) -- for time only, the number is
the same.

The split: each such edge is cut at the vertices inside it, every face
holding the edge is re-triangulated over its corners and the points on its
edges (no new vertex, no new coordinate), and a face whose three corners lie
exactly on one line is dropped -- its middle corner is put on its long edge,
so its other two edges are matched by the pieces of the neighbour across the
long edge.  "Exactly" is rational arithmetic on the float64 coordinates
(`fractions.Fraction`): a float filter far looser than its rounding only picks
the candidates.

The surface as a sum of oriented triangles is the same: the pieces of a face
cover exactly that face, with its orientation, and a face on one line has no
area.  So the winding number is the same at every point off the surface, and
so are the solid, its volume and the verdict of either mode; no warning is
due.  What changes is only the tessellation the detector and the engines see.
A shell whose split leaves more unbalanced edges, or more edges not shared by
exactly two faces, than it had is left as it was: e.g. a vertex on an edge
that already has an edge of two faces to an end of it -- two sheets touching
along that segment -- would make an edge of four faces, which manifold must
not get; such a shell keeps its pairs and goes to CGAL as before.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

# float filter: the rounding of a cross product of float64 differences is
# below ~1e-15 of |d| |w|, so every exactly collinear triple passes it
_NEAR = 1e-9


def _fr(p) -> list:
    return [Fraction(float(x)) for x in p]       # a float64 is a rational, exactly


def _cross_zero(d: list, w: list) -> bool:
    return not (d[1] * w[2] - d[2] * w[1] or d[2] * w[0] - d[0] * w[2]
                or d[0] * w[1] - d[1] * w[0])


def inside(p, a, b) -> bool:
    """p strictly inside segment ab, exactly."""
    P, A, B = _fr(p), _fr(a), _fr(b)
    d = [B[i] - A[i] for i in range(3)]
    w = [P[i] - A[i] for i in range(3)]
    if not _cross_zero(d, w):
        return False
    t = sum(w[i] * d[i] for i in range(3))
    return 0 < t < sum(d[i] * d[i] for i in range(3))


def collinear(a, b, c) -> bool:
    """a, b, c on one line (or coincident), exactly."""
    A, B, C = _fr(a), _fr(b), _fr(c)
    return _cross_zero([B[i] - A[i] for i in range(3)], [C[i] - A[i] for i in range(3)])


def _param(p, a, b) -> Fraction:
    P, A, B = _fr(p), _fr(a), _fr(b)
    d = [B[i] - A[i] for i in range(3)]
    return (sum((P[i] - A[i]) * d[i] for i in range(3))
            / sum(d[i] * d[i] for i in range(3)))


def _near_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rows where p may lie on segment ab: a superset of the exact answer."""
    d, w = b - a, p - a
    cr = np.linalg.norm(np.cross(d, w), axis=1)
    dn, wn = np.linalg.norm(d, axis=1), np.linalg.norm(w, axis=1)
    t = np.einsum("ij,ij->i", w, d) / np.maximum(dn * dn, 1e-300)
    return (cr <= _NEAR * dn * wn + 1e-300) & (t > -_NEAR) & (t < 1 + _NEAR)


def from_pairs(V: np.ndarray, F: np.ndarray, pairs: np.ndarray) -> list:
    """(v, a, b), a < b: vertex v exactly strictly inside edge {a, b}, from
    the corners of one face of each detector pair against the edges of the
    other.  A T-junction no pair reports does not change the detector's
    answer, so these are all the path needs."""
    if not len(pairs):
        return []
    fa, fb = F[pairs[:, 0]], F[pairs[:, 1]]
    cand = []
    for X, Y in ((fa, fb), (fb, fa)):
        for k in range(3):
            a, b = X[:, k], X[:, (k + 1) % 3]
            for j in range(3):
                v = Y[:, j]
                keep = (v != a) & (v != b)
                cand.append(np.stack([v[keep], np.minimum(a, b)[keep],
                                      np.maximum(a, b)[keep]], 1))
    cand = np.unique(np.concatenate(cand), axis=0)
    if not len(cand):
        return []
    near = _near_line(V[cand[:, 0]], V[cand[:, 1]], V[cand[:, 2]])
    return [(int(v), int(a), int(b)) for v, a, b in cand[near]
            if inside(V[v], V[a], V[b])]


def at_junction(V: np.ndarray, F: np.ndarray, pairs: np.ndarray) -> bool:
    """Some face of `pairs` is exactly on one line, or holds a vertex exactly
    inside an edge of the other face of its pair: the pairs may be those of
    a T-junction.  A side in "fail" asks the detector for one pair; it asks
    for all (the candidates of the split) only when this one may be such."""
    pairs = np.asarray(pairs, np.int64)
    if not len(pairs):
        return False
    if from_pairs(V, F, pairs):
        return True
    return any(collinear(*V[F[f]]) for f in np.unique(pairs).tolist())


def from_collinear_faces(V: np.ndarray, F: np.ndarray) -> list:
    """(v, a, b) for every face exactly on one line: its middle corner v
    inside its long edge {a, b} (a < b) -- the split puts it there and drops
    the face."""
    tri = V[F]
    d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cr = np.linalg.norm(np.cross(d1, d2), axis=1)
    near = np.flatnonzero(cr <= _NEAR * np.linalg.norm(d1, axis=1)
                          * np.linalg.norm(d2, axis=1) + 1e-300)
    out = []
    for f in near.tolist():
        c = [int(x) for x in F[f]]
        if not collinear(V[c[0]], V[c[1]], V[c[2]]):
            continue
        for k in range(3):
            v, a, b = c[k], c[(k + 1) % 3], c[(k + 2) % 3]
            if inside(V[v], V[a], V[b]):
                out.append((v, min(a, b), max(a, b)))
    return out


def _edge_faults(F: np.ndarray) -> tuple[int, int]:
    """(edges whose face orientations do not cancel, edges not shared by
    exactly two faces), by vertex index (a shell's vertices are merged)."""
    if not len(F):
        return 0, 0
    a = F.ravel()
    b = F[:, [1, 2, 0]].ravel()
    key = np.minimum(a, b) * (int(F.max()) + 1) + np.maximum(a, b)
    _, k = np.unique(key, return_inverse=True)
    k = k.reshape(-1)
    s = np.bincount(k, weights=np.where(a < b, 1.0, -1.0))
    n = np.bincount(k)
    return int(np.count_nonzero(s)), int(np.count_nonzero(n != 2))


def _clip(poly: list) -> list:
    """Ear clipping of a polygon whose points lie on the edges of one
    triangle, each point with the set of triangle edges it lies on: the
    polygon is convex, so an ear whose three points are not on one edge is a
    triangle of positive area inside it; the rest is never left on one edge."""
    tris = []
    while len(poly) > 3:
        n = len(poly)
        for i in range(n):
            pr, cu, nx = poly[i - 1], poly[i], poly[(i + 1) % n]
            if pr[1] & cu[1] & nx[1]:
                continue
            rest = poly[:i] + poly[i + 1:]
            if set.intersection(*[q[1] for q in rest]):
                continue
            tris.append((pr[0], cu[0], nx[0]))
            poly = rest
            break
        else:
            raise RuntimeError("no ear")
    if poly[0][1] & poly[1][1] & poly[2][1]:
        raise RuntimeError("last triangle on one line")
    tris.append((poly[0][0], poly[1][0], poly[2][0]))
    return tris


def split_faces(V: np.ndarray, F: np.ndarray, tjs: list) -> tuple[np.ndarray, int, int]:
    """Faces of one shell with its edges cut at the T-junctions `tjs` (local
    indices): (faces, junctions cut, faces on one line dropped).  A face the
    clipping cannot split is kept whole (then the edge check undoes the
    shell)."""
    on_edge: dict = {}
    for v, a, b in tjs:
        on_edge.setdefault((a, b), set()).add(v)
    order = {k: sorted(vs, key=lambda v: _param(V[v], V[k[0]], V[k[1]]))
             for k, vs in on_edge.items()}
    lo = np.minimum(F, np.roll(F, -1, axis=1))
    hi = np.maximum(F, np.roll(F, -1, axis=1))
    keys = set(order)
    touched = [f for f in range(len(F))
               if any((int(lo[f, k]), int(hi[f, k])) in keys for k in range(3))]
    keep = np.ones(len(F), bool)
    new, dropped = [], 0
    for f in touched:
        c = [int(x) for x in F[f]]
        keep[f] = False
        if collinear(V[c[0]], V[c[1]], V[c[2]]):
            dropped += 1
            continue
        poly = []
        for k in range(3):
            a, b = c[k], c[(k + 1) % 3]
            poly.append((a, {k, (k + 2) % 3}))
            pts = order.get((min(a, b), max(a, b)), [])
            poly.extend((p, {k}) for p in (pts if a < b else pts[::-1]))
        try:
            new.extend(_clip(poly))
        except RuntimeError:
            keep[f] = True
    F2 = np.concatenate([F[keep], np.asarray(new, np.int64).reshape(-1, 3)])
    return F2, sum(len(v) for v in order.values()), dropped


def split_t_junctions(own: list, pairs: np.ndarray, record: dict) -> bool:
    """
    Split, in place, the T-junctions of the shells `own` of one solid that the
    detector `pairs` (faces of `own` concatenated in order) report, and the
    faces on one line of those shells (called only for a solid with pairs).  A vertex on an edge of another shell
    of the solid is not split: shells are concatenated without merging, the
    touch stays.  Counts go to `record` ("t_junctions_split",
    "collinear_faces_removed", "t_junction_splits_undone").  True if a shell
    changed.
    """
    V = np.concatenate([np.asarray(s.vertices, np.float64) for s in own])
    F = np.concatenate([np.asarray(s.faces, np.int64) + o for s, o in
                        zip(own, np.cumsum([0] + [len(s.vertices) for s in own[:-1]]))])
    voff = np.cumsum([0] + [len(s.vertices) for s in own])
    shell_of = lambda x: int(np.searchsorted(voff, x, side="right") - 1)   # noqa: E731
    per: dict = {}
    for v, a, b in from_pairs(V, F, np.asarray(pairs, np.int64)):
        s = shell_of(v)
        if s == shell_of(a) == shell_of(b):
            o = int(voff[s])
            per.setdefault(s, set()).add((v - o, a - o, b - o))
    changed = False
    for s, sh in enumerate(own):
        Vs = np.asarray(sh.vertices, np.float64)
        Fs = np.asarray(sh.faces, np.int64)
        tjs = per.get(s, set()) | set(from_collinear_faces(Vs, Fs))
        if not tjs:
            continue
        F2, cut, dropped = split_faces(Vs, Fs, sorted(tjs))
        before, after = _edge_faults(Fs), _edge_faults(F2)
        if after[0] > before[0] or after[1] > before[1]:
            record["t_junction_splits_undone"] = record.get("t_junction_splits_undone", 0) + 1
            continue
        sh.faces = F2
        record["t_junctions_split"] = record.get("t_junctions_split", 0) + cut
        record["collinear_faces_removed"] = record.get("collinear_faces_removed", 0) + dropped
        changed = True
    return changed

