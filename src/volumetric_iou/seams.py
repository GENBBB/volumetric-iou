"""
Closing the holes of a side with a proven volume bound (docs/DEFINITION.md 4.3).

For a side S and a patch Z with dZ = -dS the closed surface T = S + Z has an
integer winding number, and its body B(T) = {w_T >= 1} differs from the body
the side's own winding number gives, {round(w_S) >= 1}, only where
|w_Z| >= 1/2.  That set holds at most pi/4 * I(Z) of volume,

    I(Z) = int_Z int_Z dA dA' / |y - y'|

(|w_Z| <= 1/(4 pi) int_Z dA / r^2; Chebyshev with the square; the convolution
int_R3 dx / (|x - y|^2 |x - y'|^2) = pi^3 / |y - y'|).  `energy_bound` bounds
I(Z) from above, so the bound holds for whatever Z is built: a patch that
closes badly only gives a large bound -- a refusal, not a wrong number.

The loops of a side are grouped by the boundary facing them (a seam of two
parts, a slit run out and back); every group -- a crack as much as a large
hole -- is closed by the least-bound candidate: the band between two loops,
the least-area triangulation of a loop or of the group's loops bridged into
one, or the fan.  The patch is accepted when, for every group and for all
together, the bound is at most SEAM_ETA of the volume of the parts it
closes -- not of the whole side, or a small part without its cap beside a
large one would pass.  Facing cracks are closed as one group for the same
reason as a seam: two fans of a seam cancel each other, their bound does not.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

SEAM_ETA = 1e-3          # proven volume error allowed, of the parts the patch closes
PARTNER_RADIUS = 1e-2    # of the side's scale: how near a facing edge may lie
PARTNER_SHARE = 0.2      # of a loop's samples near another loop: the loops are one group
POLY_MAX = 400           # least-area triangulation: O(n^3) in the loop's points
STITCH_MAX = 4_000_000   # band between two loops: O(n m)


def tri_area(V: np.ndarray, T) -> np.ndarray:
    t = V[np.asarray(T, np.int64).reshape(-1, 3)]
    return np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2.0


# ------------------------------------------------------------------ closings
# loops here run as their faces run them (face direction); every closing Z
# has dZ = -(the loops)

def poly_close(V: np.ndarray, seq: list) -> Optional[list]:
    """Least-area triangulation of the loop `seq`, reversed against it."""
    n = len(seq)
    if n < 3:
        return []
    if n > POLY_MAX:
        return None
    P = V[np.asarray(seq)]
    C = np.zeros((n, n))
    K = -np.ones((n, n), np.int64)
    for gap in range(2, n):
        for i in range(0, n - gap):
            j = i + gap
            ks = np.arange(i + 1, j)
            a = np.linalg.norm(np.cross(P[ks] - P[i], P[j] - P[i]), axis=1) / 2.0
            c = C[i, ks] + C[ks, j] + a
            t = int(np.argmin(c))
            C[i, j], K[i, j] = c[t], ks[t]
    out, stack = [], [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = int(K[i, j])
        if len({seq[i], seq[j], seq[k]}) == 3:     # a bridge repeats a vertex: (v, v, w) has no boundary
            out.append((seq[i], seq[j], seq[k]))
        stack += [(i, k), (k, j)]
    return out


def stitch_close(V: np.ndarray, seqA: list, seqB: list) -> Optional[list]:
    """The least-area band between loops A and B facing each other: A from its
    first vertex, B backwards from its vertex nearest to that one."""
    n, m = len(seqA), len(seqB)
    if n * m > STITCH_MAX:
        return None
    A, B = V[np.asarray(seqA)], V[np.asarray(seqB)]
    s = int(np.argmin(np.linalg.norm(B - A[0], axis=1)))
    qi = [(s - j) % m for j in range(m + 1)]
    pi = [i % n for i in range(n + 1)]
    Pc, Qc = A[pi], B[qi]
    D = np.full((n + 1, m + 1), np.inf)
    W = np.zeros((n + 1, m + 1), np.int8)         # 1: came along A, 2: along B
    D[0, 0] = 0.0
    for i in range(n + 1):
        if i > 0:
            a = np.linalg.norm(np.cross(Pc[i] - Pc[i - 1], Qc - Pc[i - 1]), axis=1) / 2.0
            cand = D[i - 1] + a
            better = cand < D[i]
            D[i, better], W[i, better] = cand[better], 1
        row = D[i]
        a = np.linalg.norm(np.cross(Qc[1:] - Pc[i], Qc[:-1] - Pc[i]), axis=1) / 2.0
        for j in range(1, m + 1):
            c = row[j - 1] + a[j - 1]
            if c < row[j]:
                row[j], W[i, j] = c, 2
    if not np.isfinite(D[n, m]):
        return None
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if W[i, j] == 1:
            out.append((seqA[pi[i]], seqA[pi[i - 1]], seqB[qi[j]]))
            i -= 1
        else:
            out.append((seqB[qi[j - 1]], seqB[qi[j]], seqA[pi[i]]))
            j -= 1
    return out


def bridged(V: np.ndarray, loops: list) -> list:
    """One polygon from several loops: each next loop joined by a bridge between
    the nearest vertices, run there and back (the bridge cancels)."""
    poly = list(loops[0])
    for L in loops[1:]:
        d = np.linalg.norm(V[np.asarray(poly)][:, None, :] - V[np.asarray(L)][None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        rot = list(L[j:]) + list(L[:j])
        poly = poly[:i + 1] + rot + [rot[0]] + poly[i:]
    return poly


def chain_closed(F) -> bool:
    """dF = 0: every edge has net directed multiplicity 0."""
    F = np.asarray(F, np.int64).reshape(-1, 3)
    if not len(F):
        return True
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    sgn = np.where(e[:, 0] < e[:, 1], 1, -1)
    k = np.sort(e, axis=1)
    _, inv = np.unique(k[:, 0] * (int(k.max()) + 1) + k[:, 1], return_inverse=True)
    return bool(np.all(np.bincount(inv.reshape(-1), weights=sgn) == 0))


def signed_volume(V: np.ndarray, F) -> float:
    t = V[np.asarray(F, np.int64).reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def unsigned_volume(V: np.ndarray, F) -> float:
    """Sum of |signed volume| over the parts of F connected through vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f = np.asarray(F, np.int64).reshape(-1, 3)
    if not len(f):
        return 0.0
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]]])
    n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                                             shape=(len(V), len(V))), directed=False)
    fl = lab[f[:, 0]]
    if len(np.unique(fl)) == 1:
        return abs(signed_volume(V, f))
    t = V[f]
    per = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0
    return float(np.abs(np.bincount(fl, weights=per)).sum())


# ------------------------------------------------------------------ the bound

def energy_bound(V: np.ndarray, Z, chunk: int = 2048) -> float:
    """
    Upper bound of I(Z): I(T_k, T_l) <= A_k * min(Phi_l, A_l / d_kl), with d_kl
    a lower bound of the distance (bounding spheres) and Phi_l the largest
    potential of the triangle T_l -- reached on it, and there at most that of
    the rectangle 2L x 2h about the point (L the longest edge, h the height on
    it; the angles at the longest edge are acute, so the triangle lies in it):
    4 (L asinh(h/L) + h asinh(L/h)).
    """
    Z = np.asarray(Z, np.int64).reshape(-1, 3)
    if not len(Z):
        return 0.0
    t = V[Z]
    A = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2.0
    keep = A > 0
    t, A = t[keep], A[keep]
    if not len(A):
        return 0.0
    L = np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2).max(axis=1)
    h = 2.0 * A / L
    phi = 4.0 * (L * np.arcsinh(h / L) + h * np.arcsinh(L / h))
    c = t.mean(axis=1)
    rho = np.linalg.norm(t - c[:, None, :], axis=2).max(axis=1)
    total = 0.0
    for s in range(0, len(A), chunk):
        d = (np.linalg.norm(c[s:s + chunk, None, :] - c[None, :, :], axis=2)
             - rho[s:s + chunk, None] - rho[None, :])
        with np.errstate(divide="ignore"):
            far = np.where(d > 0, A[None, :] / np.maximum(d, 1e-300), np.inf)
        total += float((A[s:s + chunk, None] * np.minimum(phi[None, :], far)).sum())
    return total


def volume_bound(V: np.ndarray, Z) -> float:
    """vol(B(S + Z) xor {round(w_S) >= 1}) <= pi/4 * I(Z)."""
    return math.pi / 4.0 * energy_bound(V, Z)


# ------------------------------------------------------------------ groups

def _partners(V: np.ndarray, seqs: list, scale: float, k: int = 32) -> list:
    """For each loop, how many of its edge samples have the nearest edge running
    the other way (tangents cos < -0.5, not the edge itself) on each loop,
    within PARTNER_RADIUS of the scale; and its sample count."""
    from scipy.spatial import cKDTree
    E = np.array([(s[i], s[(i + 1) % len(s)]) for s in seqs for i in range(len(s))], np.int64)
    loop_of = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
    ts = np.array([1, 2, 3, 4, 5]) / 6.0
    a, b = V[E[:, 0]], V[E[:, 1]]
    tan = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-300)
    P = (a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)
    Pe = np.repeat(np.arange(len(E)), len(ts))
    k = min(k, len(P))
    _, nb = cKDTree(P).query(P, k=k)
    nb = np.asarray(nb).reshape(len(P), k)
    best = np.full(len(P), np.inf)
    best_e = np.full(len(P), -1, np.int64)
    for j in range(k):
        ce = Pe[nb[:, j]]
        ok = (ce != Pe) & (np.einsum("ij,ij->i", tan[Pe], tan[ce]) < -0.5)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        aa, ab = a[ce[ok]], b[ce[ok]] - a[ce[ok]]
        tt = np.clip(np.einsum("ij,ij->i", P[ok] - aa, ab)
                     / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
        d = np.linalg.norm(P[ok] - (aa + tt[:, None] * ab), axis=1)
        better = d < best[idx]
        best[idx[better]] = d[better]
        best_e[idx[better]] = ce[ok][better]
    near = best <= PARTNER_RADIUS * scale
    out = []
    for li in range(len(seqs)):
        mine = loop_of[Pe] == li
        cnt: dict = {}
        for p in loop_of[best_e[mine & near]].tolist():
            cnt[p] = cnt.get(p, 0) + 1
        out.append((int(mine.sum()), cnt))
    return out


def _groups(V: np.ndarray, seqs: list, scale: float) -> list:
    pr = _partners(V, seqs, scale)
    parent = list(range(len(seqs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, (n, cnt) in enumerate(pr):
        for j, c in cnt.items():
            if j != i and (c >= PARTNER_SHARE * n or pr[j][1].get(i, 0) >= PARTNER_SHARE * pr[j][0]):
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(seqs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def close_loops(V: np.ndarray, F: np.ndarray, label: np.ndarray, loops: list, comps: list,
                fans: list, scale: float, cutoff: float) -> tuple[list, dict]:
    """
    Close the loops of a side (`loops` as the patch runs them, `fans` -- their
    fans, `comps` -- the components each loop bounds).  Returns (faces to add
    when accepted, the record).  Loops facing each other are closed as one
    group, each group by its candidate of least bound.  A group whose closed
    parts hold at most `cutoff` of volume is a sheet: its patch adds nothing
    (the zero-volume shell is dropped later, 4.5), so it is closed and left out
    of the bound.  The volume of the parts is taken unsigned, part by part
    (parts joined through vertices, `unsigned_volume`):
    the shells are not oriented yet (4.6), an inside-out part is no less
    defined, and two parts facing opposite ways must not cancel into a sheet.
    """
    seqs = [list(L[::-1]) for L in loops]                    # face direction
    groups = _groups(V, seqs, scale)
    chosen = []
    for g in groups:
        cands = [("fan" if len(g) == 1 else "fans", [t for i in g for t in fans[i]])]
        if len(g) == 1:
            if len(seqs[g[0]]) > 3:
                p = poly_close(V, seqs[g[0]])
                if p is not None:
                    cands.append(("triangulation", p))
        else:
            if len(g) == 2:
                s = stitch_close(V, seqs[g[0]], seqs[g[1]])
                if s is not None:
                    cands.append(("band", s))
            if sum(len(seqs[i]) + 2 for i in g) <= POLY_MAX:
                p = poly_close(V, bridged(V, [seqs[i] for i in g]))
                if p is not None:
                    cands.append(("bridged triangulation", p))
        scored = [(name, z, volume_bound(V, z)) for name, z in cands]
        chosen.append(min(scored, key=lambda x: x[2]))
    # the parts: components joined by any loop or group, closed with all their patches
    root = {}

    def find(x):
        root.setdefault(x, x)
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for g in groups:
        cs = [c for i in g for c in comps[i]]
        for c in cs[1:]:
            root[find(cs[0])] = find(c)
    cluster_of = [find(comps[g[0]][0]) for g in groups]
    clusters = {}
    for k in sorted(set(cluster_of)):
        mine = [j for j, c in enumerate(cluster_of) if c == k]
        cc = sorted({c for c in np.unique(label).tolist() if find(c) == k})
        parts = _parts(F, label, cc, [t for j in mine for t in chosen[j][1]], [])
        vol = unsigned_volume(V, parts)
        clusters[k] = {"volume": vol, "closed": chain_closed(parts), "sheet": vol <= cutoff,
                       "groups": mine}
    rec_groups, Z, S = [], [], []                            # S: patches of sheets
    for g, (name, z, bound), k in zip(groups, chosen, cluster_of):
        c = clusters[k]
        rec_groups.append({"loops": len(g), "patch": name, "faces": len(z), "bound": bound,
                           "parts_volume": c["volume"], "closed": c["closed"], "sheet": c["sheet"]})
        (S if c["sheet"] else Z).extend(z)
    rs = [r for r in rec_groups if not r["sheet"]]
    solid = [c for c in clusters.values() if not c["sheet"]]
    vol = sum(c["volume"] for c in solid)
    bound = volume_bound(V, Z) if len(rs) > 1 else (rs[0]["bound"] if rs else 0.0)
    closed = all(c["closed"] for c in clusters.values())
    ok_groups = all(r["parts_volume"] > 0 and r["bound"] <= SEAM_ETA * r["parts_volume"] for r in rs)
    accepted = closed and ok_groups and (not rs or (vol > 0 and bound <= SEAM_ETA * vol))
    ratios = [r["bound"] / r["parts_volume"] if r["parts_volume"] > 0 else math.inf for r in rs]
    ratios.append(bound / vol if vol > 0 else (math.inf if rs else 0.0))
    rec = {"groups": rec_groups, "loops": len(loops), "faces": len(Z) + len(S),
           "worst_ratio": max(ratios),
           "sheets": len(rec_groups) - len(rs),
           "area": float(tri_area(V, Z + S).sum()) if Z or S else 0.0, "bound": bound, "parts_volume": vol,
           "ratio": bound / vol if vol > 0 else None, "closed": closed, "accepted": accepted}
    return Z + S, rec


def _parts(F: np.ndarray, label: np.ndarray, comps: list, cracks: list, Z: list) -> np.ndarray:
    return np.vstack([F[np.isin(label, comps)], np.array(cracks, np.int64).reshape(-1, 3),
                      np.array(Z, np.int64).reshape(-1, 3)])
