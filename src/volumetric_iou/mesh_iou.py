"""
Volume IoU between two meshes -- a checked result or an explicit failure status.

The metric is defined in docs/DEFINITION.md.  In short: the solid of a mesh is
the set of points its surface winds around at least once (w >= 1) -- bodies
that overlap count once, cavities (shells facing inwards) are empty -- and the
IoU of two such solids comes from one float64 boolean in manifold3d, or
in CGAL where a side stays there or manifold fails the intersection.
There is no approximate fallback and no clipping beyond rounding (an
intersection over the smaller volume by at most 1e-6 is cut to it and marked
in diagnostics["volume_inter_raw"]): a number either passed every
check or the result carries a status, a cause and the side to blame.

    from volumetric_iou import volume_iou

    res = volume_iou("gt.stl", "pred.stl")
    if res:
        print(res.iou, res.iog, res.iop, res.warnings)
    else:
        print(res.status, res.cause, res.blame, res.detail)

PIPELINE per side: load -> check of the unit-cube contract -> clean (merge
vertices, drop degenerate/duplicate faces, close holes by a patch of proven
volume error at most 1e-3 of the parts it closes, reported) -> split into
shells and orient them (DEFINITION 4.6) -> crossing detector: crossings
between bodies are resolved by a union (5.4), crossings inside one body (or
one the detector failed on) go to an exact CGAL arrangement (4.7, 5.3), where
"fail" refuses a body winding 2 or more around a volume; a side winding -1 or less
somewhere, or one body of it alone (an inverted pocket), is refused -> one manifold per body, union
where extents overlap.  Then the intersection, challenged when empty and
bounded by the smaller operand; if manifold fails it, CGAL recomputes it.

NORMALISATION is the caller's.  The metric never moves the pair: the GT must
already lie in the unit cube [-0.5, 0.5]^3 (NOT_NORMALIZED otherwise), and the
prediction in the same frame -- one far outside the cube gets a warning, not a
refusal.  `normalize_pair` puts the GT into the cube by its own extent and the
prediction by the frame the caller names: the GT's transform, its own extent
(erases errors of scale and placement), or /200 about the origin.

Dependencies: trimesh, manifold3d, numpy, networkx, and the package's own CGAL module
`_cgal` (the crossing detector on every body, the exact arrangement where a
body needs it, the winding number of the probes).  libigl is not needed: its
mesh_boolean (`_boolean_module`, `_cgal_boolean`) is only a reference.
networkx is needed by trimesh itself (`split()` repairs submeshes with
`fill_holes`, `fix_normals` walks the face graph), though holes of the input
are closed by our own filler.  No rtree (containment goes through
`inside.PointInMesh`).
"""

from __future__ import annotations

import copy
import enum
import os
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Union

import numpy as np
import trimesh
from trimesh.grouping import group_rows

from . import seams
from .inside import PointInMesh
from .self_intersect import self_intersecting_pairs
from .tjunction import at_junction, split_t_junctions

try:
    import manifold3d

    _HAS_ENGINE = True
except Exception:
    manifold3d = None
    _HAS_ENGINE = False


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #


class Status(str, enum.Enum):
    OK = "ok"

    # input is not a mesh at all
    LOAD_FAILED = "load_failed"          # unreadable / not a mesh
    EMPTY_MESH = "empty_mesh"            # no faces left after cleanup
    INVALID_COORDS = "invalid_coords"    # NaN / inf in vertices

    # input is a mesh, but has no well-defined volume
    OPEN_SURFACE = "open_surface"        # boundary edges remain after hole filling
    DEGENERATE_VOLUME = "degenerate_volume"  # flat / zero-thickness / null volume

    SELF_INTERSECTING = "self_intersecting"  # surface passes through itself
    INVERTED_VOLUME = "inverted_volume"  # side (or one body alone) winds -1 or less (inverted pocket)
    NOT_MANIFOLD = "not_manifold"        # closed, but not a valid 2-manifold (3+ faces per edge)
    NOT_NORMALIZED = "not_normalized"         # precondition violated

    # geometry is fine, the computation is not
    NO_ENGINE = "no_engine"              # manifold3d unavailable
    BOOLEAN_FAILED = "boolean_failed"    # engine crashed or produced garbage
    TIMEOUT = "timeout"                  # isolated run exceeded its time limit
    CRASHED = "crashed"                  # isolated run died (segfault, OOM) or raised


class Cause(str, enum.Enum):
    """Coarse bucket a failure belongs to -- what would have to change to fix it."""

    OK = "ok"
    INPUT = "input"                        # not a mesh / empty / NaN coordinates
    INVALID_GEOMETRY = "invalid_geometry"  # a closed surface that lies about its volume
    TESSELLATION = "tessellation"          # cracks and holes in the triangulation
    ENGINE = "engine"                      # the boolean library failed on valid input
    ENVIRONMENT = "environment"            # a dependency is missing
    PRECONDITION = "precondition"          # the caller broke the contract


_CAUSE_OF = {
    Status.OK: Cause.OK,
    Status.LOAD_FAILED: Cause.INPUT,
    Status.EMPTY_MESH: Cause.INPUT,
    Status.INVALID_COORDS: Cause.INPUT,
    Status.OPEN_SURFACE: Cause.TESSELLATION,
    Status.DEGENERATE_VOLUME: Cause.INVALID_GEOMETRY,
    Status.SELF_INTERSECTING: Cause.INVALID_GEOMETRY,
    Status.INVERTED_VOLUME: Cause.INVALID_GEOMETRY,
    Status.NOT_MANIFOLD: Cause.TESSELLATION,
    Status.NOT_NORMALIZED: Cause.PRECONDITION,
    Status.NO_ENGINE: Cause.ENVIRONMENT,
    Status.BOOLEAN_FAILED: Cause.ENGINE,
    Status.TIMEOUT: Cause.ENGINE,
    Status.CRASHED: Cause.ENGINE,
}


@dataclass
class IouResult:
    status: Status
    iou: Optional[float] = None
    method: Optional[str] = None
    detail: str = ""
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blame: str = ""          # "gt" / "pred" / "engine" / "" when not attributable
    iog: Optional[float] = None   # V(G and P) / V(G)
    iop: Optional[float] = None   # V(G and P) / V(P)

    @property
    def cause(self) -> Cause:
        """Which kind of problem this is -- derived from the status, never set by hand."""
        return _CAUSE_OF[self.status]

    def __bool__(self) -> bool:
        return self.status is Status.OK

    def __repr__(self) -> str:
        if self:
            return f"<IoU {self.iou:.6f} via {self.method}>"
        who = f" [{self.blame}]" if self.blame else ""
        return f"<{self.status.value}{who} ({self.cause.value}): {self.detail}>"


# --------------------------------------------------------------------------- #
# loading & repair
# --------------------------------------------------------------------------- #


def _as_mesh(obj) -> Optional[trimesh.Trimesh]:
    """Accept a Trimesh, a Scene, a path, or a raw (vertices, faces) pair."""
    if isinstance(obj, trimesh.Trimesh):
        return obj.copy()
    if isinstance(obj, trimesh.Scene):
        try:
            return obj.to_mesh()
        except Exception:
            return None
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        try:
            return trimesh.Trimesh(np.asarray(obj[0]), np.asarray(obj[1]), process=False)
        except Exception:
            return None
    try:
        loaded = trimesh.load(obj, force="mesh", process=False)
    except Exception:
        return None
    return loaded if isinstance(loaded, trimesh.Trimesh) else None


def _boundary_edges(m: trimesh.Trimesh) -> int:
    """Edges used by exactly one face -> the surface is genuinely open there."""
    if len(m.faces) == 0:
        return 0
    return len(group_rows(m.edges_sorted, require_count=1))


def _nonmanifold_edges(m: trimesh.Trimesh) -> int:
    """Edges shared by three or more faces -- a tessellation defect, not a hole."""
    if len(m.faces) == 0:
        return 0
    groups = group_rows(m.edges_sorted, require_count=None)
    return int(sum(1 for g in groups if len(g) > 2))


def _unbalanced_edges(m: trimesh.Trimesh) -> int:
    """
    Edges (after merging exactly equal vertices) whose faces' orientations do
    not cancel: the winding number is then not piecewise constant, and
    libigl/CGAL mesh_boolean refuses the input -- after resolving all its
    self-intersections, which on a large side took up to 30 min (on 1386 dataset files this count was > 0 exactly where CGAL failed).
    """
    return _raw_edge_counts(m)[0]


def _raw_edge_counts(m: trimesh.Trimesh) -> tuple[int, int]:
    """Counts of unbalanced and crowded edges after exact vertex merging."""
    if len(m.faces) == 0:
        return 0, 0
    v = np.ascontiguousarray(np.asarray(m.vertices, np.float64) + 0.0)     # -0.0 -> 0.0
    # rows compared as bytes: ~6x faster than np.unique(axis=0) on 1.6M faces
    _, inv = np.unique(v.view(np.dtype((np.void, 24))).ravel(), return_inverse=True)
    f = inv.reshape(-1)[np.asarray(m.faces, np.int64)]
    a, b = f.ravel(), f[:, [1, 2, 0]].ravel()            # edges a -> b of every face
    keep = a != b
    a, b = a[keep], b[keep]
    key = np.minimum(a, b) * (int(inv.max()) + 1) + np.maximum(a, b)
    _, k = np.unique(key, return_inverse=True)
    return (int(np.count_nonzero(np.bincount(k, weights=np.where(a < b, 1.0, -1.0)))),
            int(np.count_nonzero(np.bincount(k) > 2)))


def _row_hash(v: np.ndarray) -> np.ndarray:
    """A 64-bit hash of each row of a contiguous (n, 3) float64 array, by its
    bytes: rows with equal bytes hash equal (unequal rows may collide)."""
    u = v.view(np.uint64)
    h = ((u[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (u[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F))
         ^ (u[:, 2] * np.uint64(0x165667B19E3779F9)))
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h


def _maybe_in(x: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """x in keys (uint64 hashes) -- or not, where low bits collide: a superset,
    for a filter; one look-up in a table, not a search."""
    bits = max(10, (8 * len(keys)).bit_length())
    mask = np.uint64((1 << bits) - 1)
    t = np.zeros(1 << bits, bool)
    t[keys & mask] = True
    return t[x & mask]


def _merged(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The faces with their vertices merged exactly (equal bytes), among
    these faces' vertices only: labels compare as the side's merged indices."""
    _, inv = np.unique(v[faces.ravel()].view(np.dtype((np.void, 24))).ravel(),
                       return_inverse=True)
    return inv.reshape(faces.shape)


def _maybe_doubled(v: np.ndarray, F: np.ndarray) -> bool:
    """
    False only where `_doubled_shells` finds no shell written twice; its
    test, without merging every vertex of the side and sorting every face
    (on a raw STL of 10 million faces 15 of 16 s).  True: maybe -- the full
    test decides (at once where most faces repeat).  A paired face has an
    equal face: the hash of its set of vertex coordinates repeats.  Only those
    faces are merged exactly to find the pairs, and of the other faces only
    those with two vertices hashing as a paired face's, to find the edges the
    pairs share with a face that is not paired.  Same decision as the full
    test: every set is taken by exact equality, every hash only widens it.
    `v`: the vertices as the full test merges them (keys on cleaning's grid).
    """
    hv = _row_hash(v)
    hf = hv[F[:, 0]] + hv[F[:, 1]] + hv[F[:, 2]]        # of the set: order-free
    hs = np.sort(hf)
    same = hs[1:] == hs[:-1]
    n_same = int(np.count_nonzero(same))
    if not n_same:
        return False
    if 8 * n_same < len(F):                             # few: by a table of the repeated
        o = np.flatnonzero(_maybe_in(hf, hs[1:][same]))
        o = o[np.argsort(hf[o])]
    else:
        o = np.argsort(hf)
    k = hf[o]
    same = k[1:] == k[:-1]
    rep = np.r_[same, False] | np.r_[False, same]
    o, k = o[rep], k[rep]                               # faces whose hash repeats, by hash
    # of those, sets with faces running both ways (a set repeated one way only
    # is not paired): the order of the vertex hashes of a face gives its way,
    # the same for equal faces, where the three differ -- else kept
    hc = hv[F[o]]
    r = np.argmin(hc, axis=1)
    ar = np.arange(len(o))
    way = hc[ar, (r + 1) % 3] < hc[ar, (r + 2) % 3]
    tie = (hc[:, 0] == hc[:, 1]) | (hc[:, 1] == hc[:, 2]) | (hc[:, 0] == hc[:, 2])
    gid = np.cumsum(np.r_[True, k[1:] != k[:-1]]) - 1
    n = np.bincount(gid)
    n_way = np.bincount(gid, weights=way)
    both = ((n_way > 0) & (n_way < n)) | (np.bincount(gid, weights=tie) > 0)
    cand = np.sort(o[both[gid]])
    if not len(cand):
        return False
    if 4 * len(cand) > len(F):          # mostly repeated faces: the full test, not twice
        return True
    f = _merged(v, F[cand])
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    s = np.ascontiguousarray(np.sort(f, axis=1))
    _, g, cnt = np.unique(s.view(np.dtype((np.void, 24))).ravel(),
                          return_inverse=True, return_counts=True)
    g = g.reshape(-1)
    r = np.argmin(f, axis=1)
    even = f[np.arange(len(f)), (r + 1) % 3] < f[np.arange(len(f)), (r + 2) % 3]
    n_even = np.bincount(g, weights=even, minlength=len(cnt)).astype(np.int64)
    pair = ok & (2 * n_even[g] == cnt[g])
    if not pair.any():
        return False
    idx = cand[pair]                                    # ascending, as in the full test
    hp = np.unique(hv[F[idx].ravel()])
    q = _maybe_in(hv, hp)[F]
    two = (q[:, 0] & q[:, 1]) | (q[:, 1] & q[:, 2]) | (q[:, 2] & q[:, 0])
    two[idx] = False
    other = np.flatnonzero(two)
    f = _merged(v, F[np.concatenate([idx, other])])
    fo, fn = f[:len(idx)], f[len(idx):]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = int(f.max()) + 1
    e = np.sort(np.concatenate([fo[:, [0, 1]], fo[:, [1, 2]], fo[:, [2, 0]]]), axis=1)
    keys, ei = np.unique(e[:, 0] * nv + e[:, 1], return_inverse=True)
    ei = ei.reshape(-1)
    rows = np.tile(np.arange(len(idx)), 3)
    a = coo_matrix((np.ones(len(rows)), (rows, ei)), shape=(len(idx), len(keys))).tocsr()
    n_comp, lab = connected_components(a @ a.T, directed=False)
    vp = np.zeros(nv, bool)
    vp[fo.ravel()] = True
    qn = vp[fn]
    ok = (fn[:, 0] != fn[:, 1]) & (fn[:, 1] != fn[:, 2]) & (fn[:, 0] != fn[:, 2])
    near = ok & ((qn[:, 0] & qn[:, 1]) | (qn[:, 1] & qn[:, 2]) | (qn[:, 2] & qn[:, 0]))
    fs = fn[near]
    es = np.sort(np.concatenate([fs[:, [0, 1]], fs[:, [1, 2]], fs[:, [2, 0]]]), axis=1)
    ks = es[:, 0] * nv + es[:, 1]
    pos = np.minimum(np.searchsorted(keys, ks), len(keys) - 1)
    shared = np.zeros(len(keys), bool)
    shared[pos[keys[pos] == ks]] = True
    touched = shared[ei].reshape(3, -1).any(axis=0)
    return bool((np.bincount(lab, weights=touched, minlength=n_comp) == 0).any())


def _doubled_shells(m: trimesh.Trimesh, keep_cut: bool = False):
    """
    Shells written twice (docs/DEFINITION.md 4.8): a component (faces joined
    by edges, vertices merged on cleaning's 1e-8 grid) in which every set of coincident faces
    runs as many times one way as the other -- k each.  Its winding is 0
    everywhere, though one copy may bound a solid: a two-sided STL, of one
    body (k = 1) or of bodies touching with the same triangulation (a contact
    face, one copy per body: k = 2), of a body written twice (k = 2 on all
    of it) and of any mix.  A component is closed when one copy of it -- a
    face where k is odd, or failing that one face of every set -- is closed
    after closing its holes by the side's rule (4.3: its windings agreed, a
    patch of proven volume error within 1e-3 of its parts; `holes` in the
    returned dict, a record per patched copy); the closed components
    together are replaced by every cell of space they enclose
    (`_doubled_cells`; `ambiguous_cells` counts the cells that may be a
    cavity, which `_prepare` refuses unless the side is filled or they are
    too small to matter).  Open, but enclosing no volume
    even with every hole capped, it is a flat sheet written on both sides and
    adds nothing under any reading: both copies are dropped (`sheets`).  Open
    otherwise, it is left as it is: a sheet and a solid with holes look the
    same, and on the datasets it was the second (a body dropped as a sheet
    left a stray shell for its solid) -- `_prepare` refuses the side as open
    (`open` in the returned dict), even where cleaning the whole side, with
    the volume of its parts, would close it.  Components with other faces (a
    single-sided body touching another) are not touched.  Returns the side
    unchanged and None when there is nothing to unfold.  `keep_cut`: a
    surface of enclosed cells with vertices of the exact cut is not joined to
    the side but returned third (`cut`; `_prepare` builds its solid in CGAL
    without cleaning it again); the third item is there only when unfolded.
    """
    if len(m.faces) == 0:
        return m, None
    v = np.ascontiguousarray(np.asarray(m.vertices, np.float64) + 0.0)     # -0.0 -> 0.0
    # vertices merged as cleaning merges them (the 1e-8 grid, 4.2): a copy
    # written back with noise below the grid is the same copy -- merged
    # exactly, it was not unfolded, and cleaning then kept one copy of the
    # two-sided record (a hollow box read solid, without a word)
    from .cells import GRID
    key = np.ascontiguousarray(np.round(v * GRID) + 0.0)
    if not _maybe_doubled(key, np.asarray(m.faces, np.int64)):
        return m, None
    _, first, inv = np.unique(key.view(np.dtype((np.void, 24))).ravel(),
                              return_index=True, return_inverse=True)
    f = inv.reshape(-1)[np.asarray(m.faces, np.int64)]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    # 1-d keys: np.unique(axis=0) costs several times more on large sides; the
    # groups are only labels, their order does not matter
    s = np.ascontiguousarray(np.sort(f, axis=1))
    _, g, cnt = np.unique(s.view(np.dtype((np.void, 24))).ravel(),
                          return_inverse=True, return_counts=True)
    g = g.reshape(-1)
    # the cyclic order of a face is one of two: (lo, mid, hi) rotated or reversed
    r = np.argmin(f, axis=1)
    even = f[np.arange(len(f)), (r + 1) % 3] < f[np.arange(len(f)), (r + 2) % 3]
    n_even = np.bincount(g, weights=even, minlength=len(cnt)).astype(np.int64)
    pair = ok & (2 * n_even[g] == cnt[g])                               # k each way
    if not pair.any():
        return m, None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    # A component of all faces is doubled when all its faces are paired: it is
    # then a component of the paired faces alone that shares no edge with any
    # other face.  Only the paired faces are joined -- on large sides they are
    # few; labels come in order of the first face, as over all faces.
    nv = len(first)
    idx = np.flatnonzero(pair)
    fo = f[idx]
    e = np.sort(np.concatenate([fo[:, [0, 1]], fo[:, [1, 2]], fo[:, [2, 0]]]), axis=1)
    keys, ei = np.unique(e[:, 0] * nv + e[:, 1], return_inverse=True)
    ei = ei.reshape(-1)
    rows = np.tile(np.arange(len(idx)), 3)
    a = coo_matrix((np.ones(len(rows)), (rows, ei)), shape=(len(idx), len(keys))).tocsr()
    n_comp, lab = connected_components(a @ a.T, directed=False)
    # edges of the paired faces used by an unpaired face: only faces with two
    # vertices of paired faces can have one
    vp = np.zeros(nv, bool)
    vp[fo.ravel()] = True
    q = vp[f]
    near = ok & ~pair & ((q[:, 0] & q[:, 1]) | (q[:, 1] & q[:, 2]) | (q[:, 2] & q[:, 0]))
    fs = f[near]
    es = np.sort(np.concatenate([fs[:, [0, 1]], fs[:, [1, 2]], fs[:, [2, 0]]]), axis=1)
    ks = es[:, 0] * nv + es[:, 1]
    pos = np.minimum(np.searchsorted(keys, ks), len(keys) - 1)
    shared = np.zeros(len(keys), bool)
    shared[pos[keys[pos] == ks]] = True
    touched = shared[ei].reshape(3, -1).any(axis=0)
    doubled = np.bincount(lab, weights=touched, minlength=n_comp) == 0
    if not doubled.any():
        return m, None
    verts = v[first]
    # one face of every set with k odd: its first face running the even way
    lead = np.zeros(len(f), bool)
    ev = np.flatnonzero(ok & even)
    lead[ev[np.unique(g[ev], return_index=True)[1]]] = True
    lead &= n_even[g] % 2 == 1
    drop = np.zeros(len(f), bool)
    record, patches = [], []         # faces of the closed components, their crack patches
    info = {"closed": 0, "open": 0, "sheets": 0, "contact_faces": 0, "loop_faces_added": 0,
            "patched_area_frac": 0.0}
    area = patched = 0.0             # of the copies, for the share of invented surface
    cutoff = (m.scale ** 3) * 1e-9   # as for zero-volume shells

    def closed_after_patches(copy: trimesh.Trimesh) -> Optional[tuple]:
        """(patch faces, their count, share of the copy's area, the holes'
        record or None) if `copy` is closed after closing its holes as the
        side's are (4.3: `_patch_loops`, a proven volume bound within
        `seams.SEAM_ETA` of the copy's parts), else None.  A copy whose patch
        closes a sheet (parts of no volume) is not closed by it: whether it is
        a sheet is decided as for an open copy (`_flat_sheet`)."""
        if not len(copy.faces):
            return None
        if not _boundary_edges(copy):
            return copy.faces[:0], 0, 0.0, None
        c = copy.copy()
        # one face of every set: their windings are not agreed, and the loops
        # and the patch's chain run by the faces' winding
        _fix_winding(c, any_start=True)
        n0 = len(c.faces)
        added, frac, rec = _patch_loops(c)
        if not added or rec is None or not rec["accepted"] or rec["sheets"]:
            return None
        return c.faces[n0:], added, frac, rec

    for c in np.flatnonzero(doubled):
        faces = idx[lab == c]
        one = trimesh.Trimesh(verts, f[faces[lead[faces]]], process=False)
        gc = g[faces]
        info["contact_faces"] += len(np.unique(gc[n_even[gc] % 2 == 0]))
        surface = one
        got = closed_after_patches(one)
        if got is None:
            # a body written an even number of times has no face of odd k: the
            # copy is empty or open, but the component -- one face of every
            # set -- may still be closed
            support = trimesh.Trimesh(
                verts, f[faces[np.unique(gc, return_index=True)[1]]], process=False)
            if len(support.faces) > len(one.faces):
                got = closed_after_patches(support)
                if got is not None:
                    surface = support
        area_one = float(surface.area if got is not None else one.area)
        area += area_one
        if got is not None:
            record.append(faces)
            # the cells read the record by parity: around a body written an
            # even number of times the patch goes in twice each way, as its faces
            even_only = not (n_even[gc] % 2).any()
            patches.append(np.concatenate([got[0], got[0]]) if even_only else got[0])
            info["loop_faces_added"] += got[1]
            patched += got[2] * area_one
            if got[3] is not None:
                info.setdefault("holes", []).append(got[3])
            info["closed"] += 1
            drop[faces] = True
        elif _flat_sheet(one if len(one.faces) else support, cutoff):
            info["sheets"] += 1
            drop[faces] = True
        else:
            info["open"] += 1
    if area > 0:
        info["patched_area_frac"] = patched / area
    if not drop.any():
        return m, info
    parts = [trimesh.Trimesh(verts, f[~drop], process=False)] if (~drop).any() else []
    cut = None
    if record:
        hull = _doubled_cells(verts, f[np.concatenate(record)], np.concatenate(patches), info)
        if hull is not None and len(hull.faces):
            # a surface of the exact cut with new vertices (crossings, coplanar
            # overlaps) is rounded: cleaning it again would drop its slivers
            # and open it -- its solid is built in CGAL as it is
            ci = _clean(hull, fill=False)[1]
            if keep_cut and (ci["degenerate_removed"] or ci["duplicate_removed"]
                             or ci["boundary_edges_before_fill"]):
                cut = hull
                info["cut_vertices"] = True
            else:
                parts.append(hull)
    if not parts:
        out = trimesh.Trimesh(verts, np.empty((0, 3), np.int64), process=False)
    else:
        out = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return (out, info, cut) if keep_cut else (out, info)


# cells of shells written twice that may be cavities are read as solid, without
# filling, while their volume is at most this share of the side's solid
AMBIGUOUS_VOLUME_FRAC = 1e-6


def _doubled_cells(verts: np.ndarray, faces: np.ndarray, patches: np.ndarray, info: dict
                   ) -> Optional[trimesh.Trimesh]:
    """
    The solid of the closed components written twice (`faces`, with their
    crack `patches` taken as written both ways), by the cells of space they
    cut (docs/DEFINITION.md 4.8 "Is a two-sided record unambiguous"): the
    union of their bodies with its cavities filled is every bounded cell
    under any set of bodies the record may come from (fact 3); without
    filling it is that only where no bounded cell may be empty (fact 4) --
    `info["ambiguous_cells"]` counts the cells that may, `["ambiguous_volume"]`
    their volume (the caller refuses them unless the side's cavities are
    filled or their volume is within AMBIGUOUS_VOLUME_FRAC of the side's
    solid -- then `["ambiguous_accepted"]`).  Returns the boundary of the
    outer cell, facing it -- every bounded cell inside -- or None (with
    `info["cells_error"]`) if the cut failed.  All components together: a
    component nested in another is a cell of both.
    """
    from .cells import cells
    rec = np.concatenate([faces, patches, patches[:, ::-1]]) if len(patches) else faces
    try:
        c = cells(verts, rec)
    except Exception as exc:                            # noqa: BLE001
        info["cells_error"] = f"{type(exc).__name__}: {exc}"
        return None
    info["cells"] = c.num_cells - 1
    filled = c.filled if c.filled is not None else np.zeros(c.num_cells, bool)
    info["ambiguous_cells"] = int((~filled[1:]).sum())
    info["ambiguous_volume"] = float(c.volume[1:][~filled[1:]].sum())
    return _outer_boundary(c)


def _outer_boundary(c) -> trimesh.Trimesh:
    """The triangles between the outer cell and a bounded one, each facing the
    outer cell: a closed surface winding 1 around every bounded cell."""
    out = (c.below == 0) ^ (c.above == 0)
    tri = c.triangles[out].copy()
    flip = c.below[out] == 0
    tri[flip] = tri[flip][:, ::-1]
    return trimesh.Trimesh(c.vertices, tri, process=False)


def _flat_sheet(copy: trimesh.Trimesh, cutoff: float) -> bool:
    """
    An open copy that encloses no volume even with every hole capped: a sheet
    of zero thickness written on both sides (a stray face of a part), not a
    solid with holes -- under any reading it adds nothing to the volume.
    """
    if len(copy.faces) == 0:
        return True
    cap = copy.copy()
    _fill_boundary_loops(cap, max_patch_frac=np.inf)
    if _boundary_edges(cap):
        return False
    _fix_winding(cap)               # one copy mixes both windings: volume needs one
    return abs(float(cap.volume)) <= cutoff


def _fix_winding(m: trimesh.Trimesh, any_start: bool = False) -> None:
    """
    `trimesh.repair.fix_winding(m)`, the same faces, without its breadth-first
    walk in Python (~65 us a face).  trimesh joins faces at edges of exactly
    two faces and, from a start face of each component, flips a face whose
    shared edge runs the same way as its neighbour's: the component takes the
    winding of its start face.  Here the flips are connected components of the
    double cover of that graph (a face and its flip; an edge run the same way
    by both faces joins a face to the other's flip), and the start faces are
    trimesh's own (networkx, as it picks them).  Where every edge has two
    faces the start faces are not looked up: a component is then a closed
    shell of its own, and the caller (`_flat_sheet`) takes the volume's
    magnitude, so which of its two windings it gets here does not matter.
    `any_start`: neither are they where the caller does not care which of its
    two windings a component gets (the copy of a shell written twice before
    its holes are patched: the patch runs against its loops, the parts'
    volume is taken unsigned).  A component that cannot be wound consistently
    (its result depends on the walk) goes to trimesh.
    """
    f = np.asarray(m.faces, np.int64)
    n = len(f)
    if n == 0 or m.is_winding_consistent:
        return
    adj = np.asarray(m.face_adjacency, np.int64)
    if len(adj) == 0:
        return
    e = np.asarray(m.face_adjacency_edges, np.int64)
    nxt = f[:, [1, 2, 0]]

    def runs(face):                                     # does the face run the edge a -> b
        return ((f[face] == e[:, :1]) & (nxt[face] == e[:, 1:])).any(axis=1)

    s = (runs(adj[:, 0]) == runs(adj[:, 1])).astype(np.int64)   # same way: one flips
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r = np.concatenate([adj[:, 0], adj[:, 0] + n])
    c = np.concatenate([adj[:, 1] + s * n, adj[:, 1] + (1 - s) * n])
    k, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(2 * n, 2 * n)),
                                  directed=False)
    if (lab[:n] == lab[n:]).any():                      # a face is its own flip
        trimesh.repair.fix_winding(m)
        return
    if any_start or 2 * len(adj) == 3 * n:              # every edge has two faces
        flip = lab[:n] > lab[n:]
    else:
        import networkx as nx
        g = nx.from_edgelist(m.face_adjacency)
        keep = np.zeros(k, bool)
        keep[[lab[next(iter(g.subgraph(cc).nodes()))] for cc in nx.connected_components(g)]] = True
        flip = ~keep[lab[:n]]
        flip[np.bincount(adj.ravel(), minlength=n) == 0] = False    # faces trimesh never visits
    if flip.any():
        faces = f.copy()
        faces[flip] = faces[flip][:, ::-1]
        m.faces = faces


def _fill_boundary_loops(m: trimesh.Trimesh, max_patch_frac: float = 0.05
                         ) -> tuple[int, float]:
    """
    Fan-triangulate each boundary loop whose fan covers no more than
    `max_patch_frac` of the mesh's surface area.

    Only for capping an open copy of a shell written twice to tell whether it
    encloses any volume (`_flat_sheet`, every loop: `max_patch_frac=inf`) --
    not a repair: holes that become part of a solid, of a side or of a copy,
    are closed with a proven volume bound (`_patch_loops`, docs/DEFINITION.md
    4.3).

    Returns (triangles added, patched area as a fraction of the surface).
    """
    loops = _boundary_loops(m)
    if not loops:
        return 0, 0.0
    budget = float(m.area) * max_patch_frac
    total_area = float(m.area)
    patched = 0.0
    new_faces: list[tuple[int, int, int]] = []
    for loop in loops:
        fan = _fan(loop, m.vertices)
        area = _fan_area(m.vertices, fan)
        if area > budget:
            continue  # a hole this big is missing geometry, not a crack
        patched += area
        new_faces.extend(fan)

    if not new_faces:
        return 0, 0.0
    m.faces = np.vstack([m.faces, np.array(new_faces, dtype=m.faces.dtype)])
    return len(new_faces), (patched / total_area if total_area else 0.0)


def _boundary_loops(m: trimesh.Trimesh) -> list[list[int]]:
    """
    The boundary loops of `m` (edges of one face), each as its vertices in
    the order a patch runs them: the other way round from the loop's faces,
    or the patch's winding contradicts the mesh.  Empty where the boundary
    branches (a vertex on more than two boundary edges): not simple loops.
    """
    edges = m.edges_sorted
    groups = group_rows(edges, require_count=1)
    if len(groups) == 0:
        return []
    boundary = edges[groups]
    # the same edges as their one face traverses them
    directed = {(int(a), int(b)) for a, b in m.edges[groups]}

    # adjacency along the boundary; a well-formed loop has degree 2 everywhere
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adj.values()):
        return []  # branching boundary: not simple loops, leave it alone

    seen: set[int] = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev), None)
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) < 3:
            continue
        if (loop[0], loop[1]) in directed:
            loop = loop[::-1]
        loops.append(loop)
    return loops


FAN_LEAST_AREA_MAX = 2000     # points: the least-area fan start is O(n^2)


def _fan(loop: list[int], verts: Optional[np.ndarray] = None) -> list[tuple[int, int, int]]:
    """
    The fan of a boundary loop.  With `verts`, from the start of least fan
    area (`_fan_start`), so that neither the patch nor its bound (4.3) -- nor
    whether a capped open copy (4.8) encloses volume -- depends on where the vertex
    numbering starts the loop.  A slit run out and back, or a loop off its plane, fans into large
    triangles folding over each other from one start and into slivers from
    another (mcb/00045610: a fan of 1.1e-3 against 1.1e-5 for one loop); a
    fold can wind -1 around a sliver of volume -- an inverted pocket the
    patch itself made (mcb/00028104: 5e-9, and 13 more sides in a check of
    the sets; none with the least-area start).
    """
    if verts is not None and len(loop) > 3:
        s = _fan_start(np.asarray(verts, np.float64)[np.asarray(loop)])
        loop = loop[s:] + loop[:s]
    return [(loop[0], loop[k], loop[k + 1]) for k in range(1, len(loop) - 1)]


def _fan_start(p: np.ndarray, chunk: int = 256) -> int:
    """
    Index into the loop's points `p` of the fan's start: the least fan area,
    near ties (1e-9 relative: a planar convex loop has the same area from
    every start) and loops of more than FAN_LEAST_AREA_MAX points (a hole, not
    a crack, as a rule) by the lexicographically least point.  Twice the area
    of the fan's triangle (a, p_j, p_j+1) is |p_j x p_j+1 - a x (p_j+1 - p_j)|,
    and the two edges at a itself give 0, so all starts broadcast (centred on
    the loop, against cancellation).
    """
    n = len(p)
    if n > FAN_LEAST_AREA_MAX:
        return int(np.lexsort(p.T[::-1])[0])
    c0 = p - p.mean(axis=0)
    q = np.roll(c0, -1, axis=0)
    c, d = np.cross(c0, q), q - c0
    area = np.empty(n)
    for s0 in range(0, n, chunk):
        a = c0[s0:s0 + chunk]
        area[s0:s0 + chunk] = np.linalg.norm(c[None] - np.cross(a[:, None, :], d[None]), axis=2).sum(axis=1)
    best = np.flatnonzero(area <= area.min() * (1 + 1e-9))
    return int(best[np.lexsort(p[best].T[::-1])[0]])


def _fan_area(verts: np.ndarray, fan: list) -> float:
    tri = np.asarray(verts)[np.array(fan)]
    return float(np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2.0)


def _face_components(m: trimesh.Trimesh) -> np.ndarray:
    """Label of each face's component: faces joined by shared edges, any
    number of faces at an edge.  Vertices are merged exactly by their
    coordinates first: a surface read without merging (an STL file: three
    vertices per face) is joined as the same surface merged."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(m.faces)
    if not n:
        return np.zeros(0, np.int64)
    v = np.ascontiguousarray(np.asarray(m.vertices, np.float64) + 0.0)     # -0.0 -> 0.0
    _, inv = np.unique(v.view(np.dtype((np.void, 24))).ravel(), return_inverse=True)
    f = inv.reshape(-1)[np.asarray(m.faces, np.int64)]
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)      # as trimesh's edges_sorted
    # 1-d keys: np.unique(axis=0) costs several times more on large sides
    _, ei = np.unique(e[:, 0] * (int(e.max()) + 1) + e[:, 1], return_inverse=True)
    ei = ei.reshape(-1)
    rows = np.repeat(np.arange(n), 3)              # edge k belongs to face k // 3
    a = coo_matrix((np.ones(len(rows)), (rows, ei)), shape=(n, int(ei.max()) + 1)).tocsr()
    return connected_components(a @ a.T, directed=False)[1]


def _patch_loops(m: trimesh.Trimesh) -> tuple[int, float, Optional[dict]]:
    """
    Close the boundary loops of a cleaned surface (docs/DEFINITION.md 4.3), in
    place.  Every loop -- a crack as much as missing geometry or a hole another
    part closes (a seam of two parts, a slit, a junk sheet) -- is closed with
    the loops facing it by the patch of least proven volume error
    (`seams.close_loops`), and only if that error is at most `seams.SEAM_ETA`
    of the volume of the parts it closes: an open box -- whose cap the surface
    does not define -- stays open, whatever the size of the side around it.
    Refused, no loop is closed.
    Returns (triangles added, patched area as a fraction of the surface, the
    holes' record or None where there are none).
    """
    loops = _boundary_loops(m)
    if not loops:
        return 0, 0.0, None
    verts = np.asarray(m.vertices, np.float64)
    total_area = float(m.area)
    label = _face_components(m)
    # the face on each boundary edge, by its directed edge as the loop runs it
    groups = group_rows(m.edges_sorted, require_count=1)
    face_of = {(int(a), int(b)): int(g // 3) for (a, b), g in zip(m.edges_sorted[groups], groups)}
    fans, comps = [], []
    for loop in loops:
        c = {int(label[face_of[(min(a, b), max(a, b))]]) for a, b in zip(loop, loop[1:] + loop[:1])}
        fans.append(_fan(loop, verts))
        comps.append(sorted(c))
    accept, record = seams.close_loops(verts, np.asarray(m.faces, np.int64), label, loops, comps, fans,
                                       float(m.scale), (m.scale ** 3) * 1e-9)
    record["area_frac"] = record["area"] / total_area if total_area else 0.0
    new_faces = accept if record["accepted"] else []
    if not new_faces:
        return 0, 0.0, record
    patched = float(seams.tri_area(verts, new_faces).sum())
    m.faces = np.vstack([m.faces, np.array(new_faces, dtype=m.faces.dtype)])
    return len(new_faces), (patched / total_area if total_area else 0.0), record


def _interior_points(shell: trimesh.Trimesh, n: int = 8) -> Optional[np.ndarray]:
    """
    A few points strictly inside a closed, outward-oriented shell.

    Vertices themselves are useless as probes: they sit exactly on a surface,
    and ray-parity containment is undefined there -- two shells that touch then
    vote at random and the nesting parity flips between runs.  Stepping inwards
    along the vertex normal moves the probe off every surface; the offset is
    tried at three magnitudes because a thin shell would be pierced by the
    largest one.  Returns None if no probe could be confirmed inside.
    """
    if len(shell.vertices) == 0:
        return None
    step = max(1, len(shell.vertices) // n)
    base = shell.vertices[::step][:n]
    normals = shell.vertex_normals[::step][:n]
    tester = PointInMesh(shell)
    for factor in (1e-3, 1e-4, 1e-5):
        pts = base - normals * (factor * shell.scale)
        try:
            keep, _ = tester.query(pts)
        except Exception:
            return None
        if keep.sum() >= max(2, len(pts) // 2):
            return pts[keep]
    return None


def _points_inside(host: trimesh.Trimesh, probe: trimesh.Trimesh,
                   n: int = 128) -> Optional[int]:
    """
    How many sampled vertices of `probe` fall strictly inside `host`.

    A sampled check of an empty boolean result, kept for the measuring tools:
    volume_iou proves an empty intersection exactly (_disjoint_proof), since
    128 points miss a thin overlap.  Returns None if containment could not be
    evaluated (no ray engine, degenerate host).
    """
    if len(probe.vertices) == 0 or len(host.faces) == 0:
        return None

    # Points must be strictly INSIDE the probe, not on its surface.  Two
    # near-identical solids share their surfaces, ray parity on a shared face
    # is undefined, and a vertex-based probe finds no witness at all -- which
    # is exactly the case where manifold returned an empty intersection for
    # two boxes differing by 0.08% in volume.
    sample = _interior_points(probe, n)
    if sample is None:
        step = max(1, len(probe.vertices) // n)
        sample = probe.vertices[::step][:n]

    try:
        hits, _ = PointInMesh(host).query(sample)
    except Exception:
        return None
    return int(np.count_nonzero(hits))


def _shell_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """One vertex of each connected shell of `mesh` (connected through edges)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f = np.asarray(mesh.faces, np.int64)
    n = len(mesh.vertices)
    e = f[:, [0, 1, 1, 2]].reshape(-1, 2)
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                                             shape=(n, n)), directed=False)
    used = np.unique(f)
    _, first = np.unique(lab[used], return_index=True)
    return np.asarray(mesh.vertices, np.float64)[used[first]]


def _shared_positions(a: trimesh.Trimesh, b: trimesh.Trimesh) -> int:
    """How many distinct positions are vertices of faces of both `a` and `b`."""
    def rows(m):
        # + 0.0: -0.0 and 0.0 are one position for the detector, not for the bytes
        v = np.asarray(m.vertices, np.float64)[np.unique(np.asarray(m.faces))] + 0.0
        return np.unique(np.ascontiguousarray(v).view(np.dtype((np.void, 24))).ravel())
    return int(len(np.intersect1d(rows(a), rows(b), assume_unique=True)))


def _disjoint_proof(a: trimesh.Trimesh, b: trimesh.Trimesh) -> Optional[str]:
    """
    None if the solids bounded by the closed surfaces `a` and `b` are proven
    to share no volume; otherwise why not (a reason for a message).

    1. No vertex of `a` has the position of a vertex of `b`, and no face of
       `a` meets a face of `b` -- CGAL's exact detector on both surfaces as one
       mesh, pairs within one surface ignored; a contact counts as meeting.
       The detector takes equal positions for a shared vertex, so a contact
       through them is no pair for it: such sides are not proven apart (CGAL
       decides if a side is in "exact", else the pair is refused).
    2. Then every shell of one surface lies wholly inside or wholly outside
       the other solid, and one vertex decides it: its winding number about
       the other surface, an integer off that surface, must be 0 -- counted
       exactly (_cgal.winding_exact: signed ray crossings, exact predicates).
    Both hold => the interiors are disjoint: a point inside both would see a
    ray leave A through some point of the surface of A, which is in B or the
    ray would have left B first through the surface of B, which is then in A.
    """
    from . import _cgal
    shared = _shared_positions(a, b)
    if shared:
        return f"the surfaces meet at {shared} shared vertex positions"
    n_a = len(a.faces)
    both = trimesh.Trimesh(np.vstack([a.vertices, b.vertices]),
                           np.vstack([a.faces, np.asarray(b.faces) + len(a.vertices)]),
                           process=False)
    try:
        hits = self_intersecting_pairs(both)
        cross = int(np.count_nonzero((hits[:, 0] < n_a) & (hits[:, 1] >= n_a)))
        ws = [] if cross else [
            _cgal.winding_exact(np.ascontiguousarray(h.vertices, np.float64),
                                np.ascontiguousarray(h.faces, np.int64),
                                np.ascontiguousarray(_shell_vertices(p), np.float64))
            for h, p in ((a, b), (b, a))]
    except Exception as exc:
        return f"the disjointness check raised {type(exc).__name__}: {exc}"
    if cross:
        return f"the surfaces meet ({cross} face pairs)"
    for w in ws:
        if np.any(w == WINDING_UNDECIDED):
            return "the winding number of a shell vertex is undecided"
        if np.any(w != 0):
            return "a shell of one operand lies inside the other"
    return None


# _cgal.winding_exact for a point it could not decide
WINDING_UNDECIDED = np.iinfo(np.int64).min


UNIT_TOL = 1e-6          # GT may leave the unit cube by rounding of its normalisation
INTER_TOL = 1e-6         # relative excess of the intersection over an operand taken as rounding
PRED_OUT_WARN = 0.1      # pred leaving it by more than this is worth a warning


PRED_FRAMES = ("gt", "bbox", "div200")


def _unit_box(m: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Centre and scale that put `m`'s own extent into [-0.5, 0.5]^3."""
    lo, hi = np.asarray(m.bounds, np.float64)
    span = float(np.max(hi - lo))
    return (lo + hi) / 2.0, (1.0 / span if span > 0 else 1.0)


def _moved(m: trimesh.Trimesh, centre, scale: float) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_translation(-np.asarray(centre, np.float64))
    m.apply_scale(scale)
    return m


def normalize_pair(gt: trimesh.Trimesh, pred: trimesh.Trimesh, *, pred_frame: str
                   ) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    The GT into [-0.5, 0.5]^3 by its own extent (centre to the origin, longest
    side to 1); the prediction by `pred_frame`, which the caller must name:

      "gt"      the GT's transform -- errors of scale and placement are kept;
      "bbox"    its own extent, like the GT -- they are erased;
      "div200"  divided by 200 about the origin: the prediction frame
                (pred / 200 + 0.5 against a GT in [0, 1]^3), shifted by -0.5.

    Boolean engines work in floating point, so absolute coordinate magnitude
    costs precision: at an offset of ~1e7 manifold silently returns an EMPTY
    intersection for two clearly overlapping cubes, which would surface as a
    confident IoU of 0.

    Either side may be a Body (`to_body`), which is already in its frame and
    is returned as is: a GT body places the prediction by its recorded frame;
    a prediction body must have been built in `pred_frame` of this GT
    (ValueError otherwise).  A GT PreparedSide (`prepare_side`) made from a
    body is returned as is and places the prediction by its body's frame.
    """
    if pred_frame not in PRED_FRAMES:
        raise ValueError(f"pred_frame must be one of {PRED_FRAMES}, got {pred_frame!r}")
    if isinstance(gt, PreparedSide):
        if gt.body is None:
            raise ValueError("a prepared GT keeps the frame of its source only if made "
                             "from a body (to_body); place the pred with the raw GT")
        return gt, normalize_pair(gt.body, pred, pred_frame=pred_frame)[1]
    from .body import Body, frame_for
    if isinstance(gt, Body) or isinstance(pred, Body):
        centre, scale = frame_for("gt", None, gt)     # the GT's own-extent frame
        g = gt if isinstance(gt, Body) else _moved(gt, centre, scale)
        if isinstance(pred, Body):
            f = pred.frame
            if f["kind"] != pred_frame or (pred_frame == "gt" and (
                    f["centre"] != [float(c) for c in centre] or f["scale"] != scale)):
                raise ValueError(f"pred body was built in frame {f['kind']!r}"
                                 + (" of another GT" if f["kind"] == pred_frame else "")
                                 + f", not {pred_frame!r} of this GT")
            return g, pred
        c, s = frame_for(pred_frame, pred, gt if pred_frame == "gt" else None)
        return g, _moved(pred, c, s)
    centre, scale = _unit_box(gt)
    if pred_frame == "bbox":
        return _moved(gt, centre, scale), _moved(pred, *_unit_box(pred))
    if pred_frame == "div200":
        return _moved(gt, centre, scale), _moved(pred, (0.0, 0.0, 0.0), 1.0 / 200)
    return _moved(gt, centre, scale), _moved(pred, centre, scale)


def _out_of_cube(m: trimesh.Trimesh) -> float:
    lo, hi = m.bounds
    return float(max(np.max(hi - 0.5), np.max(-0.5 - lo)))


def _check_normalized(gt: trimesh.Trimesh, pred: trimesh.Trimesh
                      ) -> tuple[Optional[IouResult], list[str]]:
    """
    The caller's contract: GT in the unit cube [-0.5, 0.5]^3 up to rounding,
    pred in the same frame.  Returns (error or None, warnings).
    """
    warnings: list[str] = []
    out = _out_of_cube(gt)
    if out > UNIT_TOL:
        return IouResult(
            Status.NOT_NORMALIZED,
            detail=f"gt leaves the unit cube [-0.5, 0.5]^3 by {out:.3g}; move the pair "
                   f"into it, e.g. normalize_pair(gt, pred, pred_frame=...)",
            diagnostics={"gt_min": gt.bounds[0].tolist(), "gt_max": gt.bounds[1].tolist()},
        ), warnings

    lo, hi = gt.bounds
    centre = (lo + hi) / 2.0
    if float(np.max(np.abs(centre))) > 1e-3:
        warnings.append(f"gt bounding box is off-centre by {np.max(np.abs(centre)):.3g}")
    extent = float(np.max(hi - lo))
    if extent < 1.0 - 1e-3:
        warnings.append(f"gt bounding box fills only {extent:.3g} of the unit cube")
    out = _out_of_cube(pred)
    if out > PRED_OUT_WARN:
        warnings.append(f"pred leaves the unit cube by {out:.3g} -- check that it is "
                        "in the frame of the gt")
    return None, warnings


def _load_and_check(obj, label: str) -> tuple[Optional[trimesh.Trimesh], Optional[IouResult]]:
    """Cheap checks that must run before any scale-relative operation."""
    m = _as_mesh(obj)
    if m is None:
        return None, IouResult(Status.LOAD_FAILED,
                               detail=f"{label}: not loadable as a mesh", blame=label)
    if len(m.faces) == 0 or len(m.vertices) == 0:
        return None, IouResult(Status.EMPTY_MESH,
                               detail=f"{label}: no geometry", blame=label)
    if not np.isfinite(m.vertices).all():
        return None, IouResult(Status.INVALID_COORDS,
                               detail=f"{label}: NaN/inf in vertices", blame=label)
    return m, None


# --------------------------------------------------------------------------- #
# cleaning, orientation, solids
# --------------------------------------------------------------------------- #


def _clean(m: trimesh.Trimesh, fill: bool = True
           ) -> tuple[trimesh.Trimesh, dict]:
    """
    Merge split vertices, drop degenerate/duplicate faces, close small cracks.

    The first four steps are trimesh's merge_vertices, nondegenerate_faces,
    unique_faces and remove_unreferenced_vertices, done in the package module
    (`_cgal.clean`) with the same result -- except that vertices beyond ~9.2e10
    are not merged into one (the key is a float64, not trimesh's int64).  Holes are closed ONLY by
    `_patch_loops`: a crack and a larger hole alike, only with the patch of
    proven volume error at most 1e-3 of the parts it closes
    (`seams.close_loops`; `info["holes"]`); the patched
    area is reported.  trimesh's `fill_holes` is
    deliberately not used: it closes any 3- or 4-edge hole whatever its size --
    the missing side of a box included -- and says nothing, which invented
    volume silently.
    """
    from . import _cgal
    info: dict = {"faces_raw": len(m.faces), "vertices_raw": len(m.vertices)}
    V, F, degenerate, duplicate, boundary, _, _ = _cgal.clean(
        np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64))
    m = trimesh.Trimesh(V, F, process=False)
    info["degenerate_removed"] = degenerate
    info["duplicate_removed"] = duplicate
    info["boundary_edges_before_fill"] = boundary
    info["loop_faces_added"] = 0
    info["patched_area_frac"] = 0.0
    if fill and len(m.faces) and boundary:
        info["loop_faces_added"], info["patched_area_frac"], record = _patch_loops(m)
        if record is not None:
            info["holes"] = record
    return m, info


def _signed_volume(m: trimesh.Trimesh) -> float:
    return float(m.volume)


def _probe_inside(outer: trimesh.Trimesh, inner_outward: trimesh.Trimesh,
                  testers: dict, key: int, points: Optional[dict] = None,
                  inner_key: int = -1) -> Optional[bool]:
    """
    Is the (outward-oriented) shell `inner_outward` inside `outer`?
    True / False, or None when the vote is split (the shells touch) or no
    probe could be placed.  `points` keeps the probes of each inner shell
    (`inner_key`) for its next host.
    """
    if points is not None and inner_key in points:
        probe = points[inner_key]
    else:
        probe = _interior_points(inner_outward)
        if points is not None:
            points[inner_key] = probe
    if probe is None:
        return None
    if key not in testers:
        testers[key] = PointInMesh(outer)
    hits, _ = testers[key].query(probe)
    if hits.all():
        return True
    if not hits.any():
        return False
    return None


def _box_inside(inner: trimesh.Trimesh, outer: trimesh.Trimesh) -> bool:
    return bool((inner.bounds[0] >= outer.bounds[0] - 1e-12).all()
                and (inner.bounds[1] <= outer.bounds[1] + 1e-12).all())


def _split(m: trimesh.Trimesh, repair: bool = True) -> tuple[list, list, list]:
    """
    The shells of `m` with their signed volumes and winding consistency, split
    in the package module (`_cgal.shells`).  Where every edge has two faces
    they are trimesh's split(only_watertight=False) (faces inside a shell in
    index order, so a volume may differ from trimesh's in the last bits).  At
    an edge of 4, 6, ... faces -- bodies touching along it -- the faces are
    ordered around the edge by exact predicates and paired across wedges of
    material, so each shell is closed without new faces.  A surface the module
    does not split (an edge of an odd number of faces, faces at an edge that do
    not alternate in direction, a shell pinched at an edge) goes through
    trimesh, which cuts it at edges of 3+ faces and, with `repair`, fills the
    holes this leaves with its `fill_holes` (any hole of 3-4 edges, whatever
    its size).  The side's preparation passes repair=False and refuses or
    rebuilds a side whose pieces are open (`_prepare`, `_split_open_side`):
    then the count of open pieces comes back in place of the shells, as an int.
    """
    from . import _cgal
    got = _cgal.shells(np.ascontiguousarray(m.vertices, np.float64),
                       np.ascontiguousarray(m.faces, np.int64)) if len(m.faces) else None
    if got is None:
        shells = list(m.split(only_watertight=False, repair=repair))
        if not repair:
            n_open = sum(1 for s in shells if not s.is_watertight)
            if n_open:
                return n_open, [], []
        return (shells, [_signed_volume(s) for s in shells],
                [s.is_winding_consistent for s in shells])
    return ([trimesh.Trimesh(v, f, process=False) for v, f, _, _ in got],
            [g[2] for g in got], [g[3] for g in got])


def _carries_volume(shell: trimesh.Trimesh, cutoff: float) -> bool:
    """
    Does a closed, consistently wound shell whose signed volume is ~ 0 wind
    round any volume?  The signed volume is the integral of w, and +V and -V
    cancel: a shell through itself (a prism over a figure eight) winds +1 round
    one lobe and -1 round the other.  Exactly, cheapest first: every vertex on
    one axis plane -- w is 0 off the plane, empty; no pair of the detector --
    the shell is embedded, w is one sign inside it and |V| is all it holds,
    empty; else the exact CGAL arrangement of the shell alone (`_cgal_solid`):
    its w >= 1 solid or its w <= -1 pockets above `cutoff` -- it carries
    volume.  A shell CGAL does not take is not proven empty: it carries volume.
    """
    v = np.asarray(shell.vertices)
    if len(v) and (v == v[0]).all(axis=0).any():
        return False
    try:
        if not len(self_intersecting_pairs(shell, first=True)):
            return False
    except Exception:                   # noqa: BLE001 -- left to CGAL
        pass
    try:
        _, solid, _, pocket = _cgal_solid(shell)
    except Exception:                   # noqa: BLE001 -- not proven empty: kept
        return True
    return abs(solid) > cutoff or abs(pocket) > cutoff


def _orient_shells(m: trimesh.Trimesh, parity: bool = False, repair: bool = True):
    """
    Split into shells and decide which ones bound material and which bound voids.

    DEFINITION (docs/DEFINITION.md 4.5, 4.6).  A shell is dropped as carrying
    no volume only once its winding is agreed (a shell with faces against the
    rest has a meaningless signed volume) and only if it is empty: |V| at most
    1e-9 d^3 and, as +V and -V cancel, neither its w >= 1 nor its w <= -1 part
    above that (`_carries_volume`).  A shell that cannot be agreed (not
    orientable) is kept.

    Orientation is decided per nesting group -- an outer shell and every shell
    nested in it (`_containment`: connected by containment, a touching pair
    included).  In a group whose shells are all consistently wound the given
    orientation is trusted: the group is flipped when its outer shell faces
    inwards (inside out as a whole), then a shell facing inwards is a cavity,
    provided it sits inside a shell of the group facing outwards; one that sits
    inside none is an inverted body and is flipped -- except a shell kept for
    the volume it carries (+V and -V cancelled): the sign of its ~0 says
    nothing, it stays a body as given (its pocket is found where CGAL builds
    it, 5.3).  Only in a group with a
    shell that is not consistently wound is the orientation meaningless; there
    every shell is made outward and cavities are decided by nesting parity.
    Another body of the side does not change how a group is read.

    `parity`: the orientation carries no meaning anywhere -- every group by
    nesting parity.

    `repair`: as in `_split`; without it, a split into open pieces returns no
    shells and their count in `info["split_open"]`.

    Returns (shells, roles, parent, info) where roles[i] is "solid" or
    "cavity" and parent[i] is the index of the solid a cavity belongs to.
    """
    info = {"orientation": "given", "shells_dropped": 0, "flipped_inverted_mesh": False,
            "flipped_shells": 0, "ambiguous_containment": 0}
    shells, volumes, consistent = _split(m, repair=repair)
    if isinstance(shells, int):
        info["split_open"] = shells
        return [], [], [], info
    if not shells:
        return [], [], [], info

    # empty shells: nothing to the volume, but they poison the boolean
    cutoff = (m.scale ** 3) * 1e-9
    kept, carrying = [], []
    for k in range(len(shells)):
        agreed = shells[k]
        if not consistent[k]:
            agreed = shells[k] = shells[k].copy()     # the copy is what parity reads
            agreed.fix_normals()
            if not agreed.is_winding_consistent:
                kept.append(k)          # not orientable: not proven empty
                continue
        if abs(volumes[k] if consistent[k] else _signed_volume(agreed)) > cutoff:
            kept.append(k)
        elif _carries_volume(agreed, cutoff):
            kept.append(k)              # +V and -V cancelled: left to the detector
            carrying.append(k)
    info["shells_dropped"] = len(shells) - len(kept)
    if carrying:
        info["zero_volume_kept"] = len(carrying)
    shells = [shells[k] for k in kept]
    if not shells:
        return [], [], [], info

    n = len(shells)
    roles = ["solid"] * n
    parent = [-1] * n
    vols = np.array([volumes[k] for k in kept])
    cons = [bool(consistent[k]) for k in kept]
    # +V and -V cancelled: the sign of ~0 says nothing of where it faces
    unsigned = [k in carrying for k in kept]
    if not parity and all(cons) and ((vols > 0).all() or (vols < 0).all()):
        # no shell faces against the rest: every group reads alike -- all
        # bodies, or the mesh inside out as a whole; nothing to probe
        if vols[0] < 0:
            for s in shells:
                s.invert()
            info["flipped_inverted_mesh"] = True
        return shells, roles, parent, info

    # outward copies: probes go inside a shell along its vertex normals
    outward = []
    for i, s in enumerate(shells):
        if parity or not cons[i]:
            if cons[i]:                 # agreed already when it was not
                s.fix_normals()
            if _signed_volume(s) < 0:
                s.invert()
            outward.append(s)
        elif vols[i] <= 0:              # inwards, as in the roles below
            o = s.copy()
            o.invert()
            outward.append(o)
        else:
            outward.append(s)
    # consistently wound only: a pair matters when one of it faces inwards
    any_pair = parity or not all(cons)
    verdict = _containment(outward, None if any_pair else vols <= 0)

    # nesting groups: components of the containment, touching pairs included
    group = list(range(n))

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
    hosts_of: list[list] = [[] for _ in range(n)]       # (j, True | None) for i inside j
    for (i, j), v in verdict.items():
        group[root(i)] = root(j)
        hosts_of[i].append((j, v))
    members: dict = {}
    for i in range(n):
        members.setdefault(root(i), []).append(i)

    flipped_groups = n_parity = 0
    for g in members.values():
        if parity or not all(cons[i] for i in g):
            n_parity += 1
            _parity_roles(g, outward, hosts_of, shells, roles, parent, info)
            continue
        # the outer shells: in no other shell (touching counts); a group is
        # connected by containment, so every host of a member is in it
        outer = [i for i in g if not hosts_of[i]] or \
                [i for i in g if not any(v for _, v in hosts_of[i])] or \
                [max(g, key=lambda i: abs(vols[i]))]
        if sum(vols[i] for i in outer) < 0:          # the group is inside out
            for i in g:
                shells[i].invert()
                vols[i] = -vols[i]
            flipped_groups += 1
        for i in g:
            if vols[i] > 0 or unsigned[i]:   # a body as given: its pocket is CGAL's (5.3)
                continue
            owners = []
            for j, v in hosts_of[i]:
                if vols[j] <= 0:
                    continue
                if v is None:
                    info["ambiguous_containment"] += 1   # touching its host: trust the orientation
                owners.append(j)
            if owners:
                roles[i] = "cavity"
                parent[i] = min(owners, key=lambda j: vols[j])   # innermost host
            else:
                shells[i].invert()      # an inside-out body standing on its own
                vols[i] = -vols[i]
                info["flipped_shells"] += 1
    if n_parity:
        info["orientation"] = "parity"
        info["parity_groups"] = n_parity
    if flipped_groups == len(members):
        info["flipped_inverted_mesh"] = True
    else:
        info["flipped_shells"] += flipped_groups
    return shells, roles, parent, info


def _containment(shells: list, inward: Optional[np.ndarray]) -> dict:
    """
    Which outward-oriented shell lies inside which: {(i, j): True | None} for
    shell i inside shell j -- None when the probes of i vote both ways (the
    shells touch) or none could be placed in it; pairs not inside are left out.
    A pair is probed only when the box of i is inside the box of j and, given
    `inward` (a mask), when i or j faces inwards.  The box test of a shell
    against all at once, and its probes found once for all its hosts: a side
    of ~200 shells spent 0.4 of 0.5 s on them.
    """
    testers: dict = {}
    points: dict = {}
    out: dict = {}
    lo = np.array([s.bounds[0] for s in shells])
    hi = np.array([s.bounds[1] for s in shells])
    lo_out, hi_out = lo - 1e-12, hi + 1e-12
    for i in range(len(shells)):
        box = (lo[i] >= lo_out).all(axis=1) & (hi[i] <= hi_out).all(axis=1)
        if inward is not None and not inward[i]:
            box &= inward
        for j in np.flatnonzero(box).tolist():
            if j == i:
                continue
            v = _probe_inside(shells[j], shells[i], testers, j, points, i)
            if v is None or v:
                out[(i, j)] = v
    return out


def _parity_roles(g: list, outward: list, hosts_of: list, shells: list, roles: list,
                  parent: list, info: dict) -> None:
    """A nesting group read by parity: every shell outward (`outward` holds
    them), a shell inside an odd number of others is a cavity of a host one
    level up; a touching pair does not count."""
    for i in g:
        shells[i] = outward[i]
    vols = {i: _signed_volume(shells[i]) for i in g}
    depth = {i: sum(1 for _, v in hosts_of[i] if v) for i in g}
    info["ambiguous_containment"] += sum(1 for i in g for _, v in hosts_of[i] if v is None)
    for i in g:
        if depth[i] % 2:
            roles[i] = "cavity"
            hosts = [j for j, v in hosts_of[i] if v and depth[j] == depth[i] - 1]
            parent[i] = min(hosts, key=lambda j: vols[j]) if hosts else -1
            shells[i].invert()


def _to_manifold(meshes: list[trimesh.Trimesh]):
    m = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    return manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.ascontiguousarray(m.vertices, dtype=np.float64),
        tri_verts=np.ascontiguousarray(m.faces, dtype=np.uint64)))


def _intersect(a, b):
    """The one boolean the IoU rests on; a seam for tests to fail it on purpose."""
    return a ^ b


def _from_manifold(man) -> trimesh.Trimesh:
    out = man.to_mesh64()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3],
                           np.asarray(out.tri_verts), process=False)


@dataclass
class _Prepared:
    mesh: trimesh.Trimesh          # the side's solid, w >= 1, after any union
    solid: object                  # the same as a manifold3d.Manifold; None if CGAL-resolved
    volume: float
    info: dict
    shells: Optional[trimesh.Trimesh] = None   # oriented shells: the raw surface, w >= 1 of it
    hosts: Optional[list] = None               # outward host shell of each body, for probes
    probes: Optional[np.ndarray] = None        # points inside the bodies; None: from hosts

    def probe_points(self) -> np.ndarray:
        return self.probes if self.probes is not None else _probes(self.hosts)


# "SITE[:LIB]": mesh_boolean from another build of the libigl bindings, e.g. one
# with CGAL on GMP (tools/build_igl_gmp.sh; the PyPI wheel has boost
# cpp_rational) -- SITE holds igl/copyleft/cgal/pyigl_copyleft_cgal*.so, LIB the
# libgmp/libmpfr it links.  A reference for tests and tools/: the metric itself
# goes through the package module (_cgal_solid, _cgal_intersection).
BOOLEAN_ENV = "VOLUMETRIC_IOU_CGAL_BOOLEAN"
_boolean_mod = None


def _boolean_module():
    """libigl's mesh_boolean, the reference of tests and tools/ (the metric does
    not call it), loaded once per process."""
    global _boolean_mod
    if _boolean_mod is None:
        spec = os.environ.get(BOOLEAN_ENV)
        if spec:
            _boolean_mod = _load_boolean_module(*spec.split(":", 1))
        else:
            import igl.copyleft.cgal as cgal     # lazy: only the reference needs it
            _boolean_mod = cgal
    return _boolean_mod


def _load_boolean_module(site: str, lib: str = ""):
    # beside the installed igl, under the extension's own name; it has no RPATH,
    # so its libraries are loaded first.  A bad setting raises, never falls back.
    import ctypes
    import glob
    import importlib.util
    for name in ("libgmp.so.10", "libmpfr.so.6") if lib else ():
        ctypes.CDLL(os.path.join(lib, name), mode=ctypes.RTLD_GLOBAL)
    found = glob.glob(os.path.join(site, "igl", "copyleft", "cgal", "pyigl_copyleft_cgal*.so"))
    if not found:
        raise ImportError(f"{BOOLEAN_ENV}: no igl/copyleft/cgal/pyigl_copyleft_cgal*.so in {site}")
    spec = importlib.util.spec_from_file_location("pyigl_copyleft_cgal", found[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cgal_boolean(a: trimesh.Trimesh, b: Optional[trimesh.Trimesh], op: str
                  ) -> tuple[trimesh.Trimesh, float]:
    """
    libigl/CGAL mesh arrangement boolean on the w >= 1 solids of the inputs
    (unary "union" with b=None resolves one self-intersecting surface).  Exact
    rational arithmetic inside; the output is only rounded to float64, which
    may merge vertices and flatten slivers -- so its volume is taken by the
    divergence theorem as is, never after cleaning (cleaning would open it).
    """
    cgal = _boolean_module()
    args = [np.asarray(a.vertices, np.float64), np.asarray(a.faces, np.int64)]
    if b is not None:
        args += [np.asarray(b.vertices, np.float64), np.asarray(b.faces, np.int64)]
    V, F, _ = cgal.mesh_boolean(*args, type_str=op)
    out = trimesh.Trimesh(V, F, process=False)
    return out, float(out.volume) if len(F) else 0.0


def _cgal_solid(m: trimesh.Trimesh, cavity: bool = False) -> tuple:
    """
    The w >= 1 solid of one surface and its inverted pockets (w <= -1), from
    one exact CGAL arrangement (`_cgal.solid_and_pocket`): the solid is what
    the unary libigl union gives, the pockets what the same union of the
    surface with its faces flipped gives -- flipping negates w.  Returns the
    solid, its volume (as `_cgal_boolean`: rounded output, not cleaned), and
    the pockets' faces and volume -- summed exactly on the arrangement, so a
    sliver of a rounding step is not below 0 as the float64 sum of the
    rounded output may be; a pocket of any size is a face.
    `cavity`: one more item, whether the pockets lie in cavities of the
    solid, as `_fill_cavities` fills them -- the void around them (cells of
    w <= 0 joined by faces) holds no unbounded cell and shares no vertex with
    another void; False without a pocket -- decided exactly on the same
    arrangement (docs/DEFINITION.md 5.5).
    """
    from . import _cgal
    args = (np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64))
    got = (_cgal.solid_and_pocket(*args, cavity=True, volume=True) if cavity
           else _cgal.solid_and_pocket(*args, volume=True))
    V, F, Vp, Fp = got[:4]
    out = trimesh.Trimesh(V, F, process=False)
    res = (out, float(out.volume) if len(F) else 0.0, len(Fp), float(got[-1]))
    return res + (bool(got[4]),) if cavity else res


def _cgal_solid_bodies(m: trimesh.Trimesh, sizes: list[int], check: list[int],
                       cavity: bool = False, pockets: tuple = ()) -> tuple:
    """
    `_cgal_solid` of bodies stacked in `m` (sizes[k] faces of body k in
    turn; the winding summed over them), and from the same arrangement, per
    body k in `check`, the faces and volume of the boundary of {w_k >= 2}:
    where that body alone winds twice or more -- passes through itself
    (`_cgal.solid_pocket_overlap`); cells of zero volume are left out.  An
    overlap of two bodies is not there.
    The pockets are {w <= -1} of the sum and, per body k in `pockets`, of
    w_k alone: inside another body the sum hides a body's pocket.
    `cavity`: whether the pockets lie in cavities appended, as in `_cgal_solid`.
    """
    from . import _cgal
    args = (np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64),
            [int(s) for s in sizes], [int(k) for k in check])
    kw = {"pockets": [int(k) for k in pockets]} if pockets else {}
    got = (_cgal.solid_pocket_overlap(*args, cavity=True, volume=True, **kw) if cavity
           else _cgal.solid_pocket_overlap(*args, volume=True, **kw))
    V, F, Vp, Fp, ofaces, ovol = got[:6]
    out = trimesh.Trimesh(V, F, process=False)
    res = (out, float(out.volume) if len(F) else 0.0, len(Fp), float(got[-1]),
           list(ofaces), list(ovol))
    return res + (bool(got[7]),) if cavity else res


def _cgal_solid_parts(m: trimesh.Trimesh, comp: np.ndarray, cavity: bool = False) -> tuple:
    """
    `_cgal_solid` of one surface and, from the same arrangement, the largest
    winding of the sum over its cells of non-zero volume and the number of
    components (comp[f]: the component of face f) that each wind 2 or more
    by themselves -- {w_k >= 2} not empty, cells of zero volume left out
    (`_cgal.solid_pocket_overlap` with a label per face: the faces keep
    their order, and the solid is the one `_cgal_solid` builds).  Returns
    `_cgal_solid`'s items, the largest winding and that number.
    """
    from . import _cgal
    comp = np.asarray(comp, np.int64)
    sizes = np.bincount(comp, minlength=1).tolist() if len(comp) else [0]
    args = (np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64),
            [int(s) for s in sizes], list(range(len(sizes))))
    got = _cgal.solid_pocket_overlap(*args, cavity=cavity, labels=comp.tolist(), volume=True)
    V, F, Vp, Fp, ofaces = got[:5]
    out = trimesh.Trimesh(V, F, process=False)
    res = (out, float(out.volume) if len(F) else 0.0, len(Fp), float(got[-1]))
    if cavity:
        res += (bool(got[7]),)
    return res, int(got[6]), sum(1 for f in ofaces if f)


def _cgal_intersection(a: trimesh.Trimesh, b: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float]:
    """
    The w >= 1 intersection of two surfaces from one exact CGAL arrangement
    (`_cgal.intersection`, what libigl's mesh_boolean "intersect" gives);
    the volume as `_cgal_boolean`: rounded output, not cleaned.
    """
    from . import _cgal
    V, F = _cgal.intersection(np.ascontiguousarray(a.vertices, np.float64),
                              np.ascontiguousarray(a.faces, np.int64),
                              np.ascontiguousarray(b.vertices, np.float64),
                              np.ascontiguousarray(b.faces, np.int64))
    out = trimesh.Trimesh(V, F, process=False)
    return out, float(out.volume) if len(F) else 0.0


def _embedded(mesh: trimesh.Trimesh) -> bool:
    """
    No face pair of the exact detector in `mesh`.  A CGAL output rounded to
    float64 may cross itself where its faces were closer than a rounding step
    (or touch itself, as two bodies of the solid meeting at a point do):
    manifold takes only a surface without pairs (DEFINITION 7).  A detector
    failure counts as pairs.
    """
    try:
        return not len(self_intersecting_pairs(mesh, first=True))
    except Exception:                   # noqa: BLE001 -- unproven: stays in CGAL
        return False


def _resolved_to_manifold(solid_mesh: trimesh.Trimesh, volume: float):
    """
    The CGAL-resolved solid as a manifold, for the pair intersection; None
    when float64 rounding left face pairs in it (`_embedded`) or broke its
    topology or volume -- the intersection then stays in CGAL, on this solid
    rather than on the raw shells.
    """
    if not _HAS_ENGINE or not _embedded(solid_mesh):
        return None
    man = _to_manifold([solid_mesh])
    if (man.status() == manifold3d.Error.NoError
            and abs(float(man.volume()) - volume) <= 1e-9 * volume):
        return man
    return None


def crossing_detail(label: str) -> str:
    """The refusal of a side in "fail" mode with a crossing inside one body
    not shown to be a touch: a body file built in "exact", or written before
    `resolved.self_overlap_bodies` was recorded."""
    return (f"{label}: intersecting or touching triangles inside one body; its "
            "surface is not proven free of self-intersection")


def through_detail(label: str, volume: float) -> str:
    """The refusal of a side in "fail" mode with a body passing through itself."""
    return (f"{label}: the surface of one body winds 2 or more around a volume of "
            f"{volume:.3g} (it passes through itself); the mesh does not define its solid")


def _crossings(shells: list[trimesh.Trimesh], group: list[int], first: bool = False,
               split: Optional[dict] = None):
    """
    Intersecting face pairs inside each solid (a shell with its cavities).

    A crossing inside one solid -- a shell through itself, or a cavity through
    its host -- is a surface that double-covers space and no union removes
    (DEFINITION 4.7).  A crossing between two different solids is an overlap
    of bodies, which the union resolves (5.4), so it is not looked for: checked on the
    whole side, overlapping bodies cost millions of pairs for nothing
    (mcb/00027755: 29 overlapping copies of one body, 12.8 million crossings,
    all between bodies).  Returns the internal count, the set of solids with
    internal crossings, the set of solids the detector failed on, and the
    first detector error ("" if none).  first: stop each solid at its first
    crossing -- a side in "fail" needs the solids with one, not the count
    (then the count is that of such solids, not of pairs).
    split: a dict -- a solid with pairs has its T-junctions split in place
    (`tjunction.split_t_junctions`, DEFINITION 4.7; the counts go to the
    dict), and its pairs are those of the split shells.  A solid without
    pairs costs nothing more; with `first`, all pairs are asked for only
    when the first one may be a T-junction's (`tjunction.at_junction`) --
    then a body whose first pair is a T-junction's but which has many real
    crossings too pays for listing them all (time only, not the answer).
    A failure of the split stage leaves the status as without it
    (`_split_t_junctions`).
    """
    internal, bad, unchecked, error = 0, set(), set(), ""
    for g in sorted(set(group)):
        own = [s for s, k in zip(shells, group) if k == g]
        try:
            whole = trimesh.util.concatenate(own) if len(own) > 1 else own[0]
            pairs = self_intersecting_pairs(whole, first=first)
        except Exception as exc:        # noqa: BLE001 -- unproven, not fatal
            unchecked.add(g)
            error = error or f"{type(exc).__name__}: {exc}"
            continue
        if len(pairs) and split is not None:
            pairs = _split_t_junctions(own, whole, pairs, first, split)
        if len(pairs):
            internal += len(pairs)
            bad.add(g)
    return internal, bad, unchecked, error


def _split_t_junctions(own: list, whole: trimesh.Trimesh, pairs: np.ndarray, first: bool,
                       record: dict) -> np.ndarray:
    """
    The T-junction split of one solid with detector pairs `pairs` (one pair
    if `first`), and the pairs that then decide its path.  With `first`, all
    pairs (the split's candidates) are asked for only if that one may be a
    T-junction's; else the body goes to CGAL as without the split -- time,
    not number.  Should anything raise -- the full detector call, the split,
    the recount on the split shells -- the shells are put back, the counts
    of this solid are dropped, the error goes to `record["t_junction_error"]`
    and the original pairs decide: the status is that without the split.
    """
    faces = [s.faces.copy() for s in own]
    local: dict = {}
    try:
        full = pairs
        if first:
            full = (self_intersecting_pairs(whole) if at_junction(
                np.asarray(whole.vertices, np.float64),
                np.asarray(whole.faces, np.int64), pairs) else None)
        if full is None or not split_t_junctions(own, full, local):
            for k, v in local.items():          # only undone shells, if any
                record[k] = record.get(k, 0) + v
            return pairs
        new = self_intersecting_pairs(
            trimesh.util.concatenate(own) if len(own) > 1 else own[0], first=first)
    except Exception as exc:            # noqa: BLE001 -- the split only saves time
        for s, f in zip(own, faces):
            s.faces = f
        record.setdefault("t_junction_error", f"{type(exc).__name__}: {exc}")
        return pairs
    for k, v in local.items():
        record[k] = record.get(k, 0) + v
    return new


def _winding(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    # igl, not the numpy sum of solid angles (winding.winding_number): that one
    # is O(points x faces) and was 18-27x slower on bodies of 10^5..10^6 faces.
    # _cgal.winding_number is igl::winding_number, as igl.winding_number runs it
    from . import _cgal
    return _cgal.winding_number(np.ascontiguousarray(mesh.vertices, np.float64),
                                np.ascontiguousarray(mesh.faces, np.int64),
                                np.ascontiguousarray(pts, np.float64))


def _union(mans: list):
    """
    Union of manifolds, pairwise as a balanced tree: on mcb/00027755 (30
    overlapping bodies) 66 s against 93 s of batch_boolean(Add).  Each level
    is evaluated (num_tri): manifold3d is lazy and flattens a tree of pending
    Adds back into one batch, which takes the batch's 93 s.
    """
    while len(mans) > 1:
        mans = [mans[i] + mans[i + 1] if i + 1 < len(mans) else mans[i]
                for i in range(0, len(mans), 2)]
        for m in mans:
            m.num_tri()
    return mans[0]


def _probes(hosts: list[trimesh.Trimesh]) -> np.ndarray:
    """Points just inside each outward host shell (a few per body)."""
    pts = [q for h in hosts if (q := _interior_points(h)) is not None]
    return np.concatenate(pts) if pts else np.empty((0, 3))


def _lost_probes(pts: np.ndarray, raws: list[trimesh.Trimesh], out: trimesh.Trimesh
                 ) -> tuple[int, int]:
    """
    Check of a CGAL result against the definition, both ways: a probe point
    where every raw input winds at least once (w >= 1) must lie inside `out`,
    any other probe outside it.  libigl's mesh_boolean was seen to drop a
    whole body of a multi-body input with no error (mcb/00015262).
    Returns (probes used, probes on the wrong side).
    """
    if not len(pts):
        return 0, 0
    inside = np.ones(len(pts), bool)
    for raw in raws:
        inside &= _winding(raw, pts) >= 0.5
    got = (_winding(out, pts) >= 0.5) if len(out.faces) else np.zeros(len(pts), bool)
    return len(pts), int((inside != got).sum())


def _resolve_bodies(bodies: list[list[trimesh.Trimesh]], label: str,
                    check: tuple = (), cavity: bool = False, exact: bool = False,
                    pockets: tuple = ()) -> tuple[Optional[tuple], Optional[str]]:
    """
    The w >= 1 solid of a cluster of bodies with overlapping extents (each a
    list of shells, host first).  One unary CGAL arrangement union of them
    all -- the winding is summed over the bodies, so a body inside another's
    inverted pocket stays empty.  If its result fails the probe check (CGAL
    dropped a body), each body is resolved alone and the solids united; that
    is accepted only if it passes the same check, which a body in a pocket
    would fail.  Returns ((mesh, volume, manifold or None, probes used,
    pocket faces, pocket volume, overlap, in_cavity), None) or (None,
    reason); the pockets (w <= -1) are those of the whole cluster and, for
    the bodies in `pockets` (indices into `bodies`: those not proven
    embedded, whose own w may be -1), of each alone -- a body's pocket
    inside another body is hidden by the sum -- from the same arrangement.
    `check`: bodies (indices into `bodies`) to look for a
    pass through itself in the same arrangement -- `overlap` is [(index,
    faces, volume)] of their {w_body >= 2} (`_cgal_solid_bodies`), [] without
    them.  `cavity`: with a pocket, `in_cavity` tells whether it lies in
    cavities of the cluster's solid (`_cgal_solid`); False otherwise.
    `exact` (`exact_booleans`): no manifold -- the solid is never handed to
    it, and the per-body fallback unites the bodies' solids in CGAL.
    """
    shells = [s for b in bodies for s in b]
    body = trimesh.util.concatenate(shells) if len(shells) > 1 else shells[0]
    unbalanced = _unbalanced_edges(body)
    if unbalanced:
        return None, (f"{label}: {unbalanced} edges with unbalanced face orientations "
                      "(winding number not piecewise constant); CGAL refuses such input")
    pts = _probes([b[0] for b in bodies])
    overlap: list = []
    in_cavity = False
    # the extra argument only when asked: callers wrapping these see the old call
    extra = {"cavity": True} if cavity else {}
    pockets = pockets if len(bodies) > 1 else ()     # one body: its w is the sum
    try:
        if check or pockets:
            sizes = [sum(len(s.faces) for s in b) for b in bodies]
            got = _cgal_solid_bodies(body, sizes, list(check), **extra,
                                     **({"pockets": pockets} if pockets else {}))
            solid_mesh, volume, pk_faces, pk_volume, of, ov = got[:6]
            overlap = [(k, f, v) for k, f, v in zip(check, of, ov)]
        else:
            got = _cgal_solid(body, **extra)
            solid_mesh, volume, pk_faces, pk_volume = got[:4]
        if cavity:
            in_cavity = got[-1]
    except Exception as exc:
        return None, f"{label}: CGAL union raised {type(exc).__name__}: {exc}"
    used, wrong = _lost_probes(pts, [body], solid_mesh)
    if not wrong:
        man = None if exact else _resolved_to_manifold(solid_mesh, volume)
        return (solid_mesh, volume, man, used, pk_faces, pk_volume, overlap, in_cavity), None
    why = f"{label}: CGAL union put {wrong} of {used} probe points on the wrong side"
    if len(bodies) == 1 or not _HAS_ENGINE:
        return None, why
    mans, solids = [], []
    for b in bodies:
        one = trimesh.util.concatenate(b) if len(b) > 1 else b[0]
        try:
            m1, v1 = _cgal_solid(one)[:2]
        except Exception as exc:
            return None, f"{why}; per body: CGAL raised {type(exc).__name__}: {exc}"
        solids.append(m1)
        mans.append(None if exact else _resolved_to_manifold(m1, v1))
    if all(man is not None for man in mans):
        united = _union(mans)
        solid_mesh, volume = _from_manifold(united), float(united.volume())
    else:
        # a rounded solid manifold does not take (face pairs, DEFINITION 7):
        # the bodies' solids united in CGAL, the result checked as above
        united = None
        try:
            solid_mesh, volume = _cgal_solid(trimesh.util.concatenate(solids))[:2]
        except Exception as exc:
            return None, f"{why}; per body: CGAL union raised {type(exc).__name__}: {exc}"
    used, wrong2 = _lost_probes(pts, [body], solid_mesh)
    if wrong2:
        return None, f"{why}; per body: {wrong2} wrong"
    # in_cavity came from the arrangement that failed the probes: not trusted,
    # a pocket stays a refusal here
    return (solid_mesh, volume, united, used, pk_faces, pk_volume, overlap, False), None


def _clusters(boxes: list) -> list[list[int]]:
    """Indices grouped by transitively overlapping (or touching) boxes."""
    root = list(range(len(boxes)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if (boxes[a][0] <= boxes[b][1]).all() and (boxes[b][0] <= boxes[a][1]).all():
                root[find(a)] = find(b)
    out: dict = {}
    for i in range(len(boxes)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _prepare(m: trimesh.Trimesh, label: str, *, check_self_intersection: bool,
             on_self_intersection: str, fill_cavities: bool = False,
             exact_booleans: bool = False
             ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    Repair, orient and solidify one already-normalised side.  Shells written
    twice are unfolded first (`_doubled_shells`): their solid is every cell
    of space they enclose, refused when cells that may be a cavity hold more
    than AMBIGUOUS_VOLUME_FRAC of the side's solid -- unless the side's
    cavities are filled (`fill_cavities`).  A side whose
    split leaves open pieces is decided by its raw surface
    (`_split_open_side`).  A side refused as open or not manifold
    after cleaning is built from its raw surface when that surface defines a
    solid (`_raw_fallback`; in "fail" with the self-intersection check
    refused where one of its parts winds 2 or more by itself)
    -- unless a shell written twice is open or refused:
    its raw winding is 0, and the raw surface would drop it.  With
    `fill_cavities`, an inverted pocket in a cavity of the solid is not
    refused (`_prepare_shells`, `_raw_solid`): the filling takes it in.
    `exact_booleans`: the solid is CGAL's on every path, never a manifold
    (`info["exact_booleans"]`), so the pair intersection is CGAL's as well.
    """
    exact = {"exact": True} if exact_booleans else {}
    prep, err = _prepare_shells(m, label, check_self_intersection=check_self_intersection,
                                on_self_intersection=on_self_intersection,
                                fill_cavities=fill_cavities, **exact)
    m0, doubled = m, None
    # no repeated face in cleaning: no shell written twice, nothing to redo
    if (prep.info if prep is not None else err.diagnostics or {}).get("duplicate_removed") != 0:
        got = _doubled_shells(m, keep_cut=True)
        m, doubled, cut = got if len(got) == 3 else (*got, None)
        if doubled is not None and (doubled["closed"] or doubled["sheets"]):
            prep, err = _prepare_shells(m, label,
                                        check_self_intersection=check_self_intersection,
                                        on_self_intersection=on_self_intersection, cut=cut,
                                        fill_cavities=fill_cavities, **exact)
            if cut is not None:
                m = trimesh.util.concatenate([m, cut]) if len(m.faces) else cut
    ambiguous = (doubled or {}).get("ambiguous_cells", 0) if not fill_cavities else 0
    if (ambiguous and err is None
            and doubled["ambiguous_volume"] <= AMBIGUOUS_VOLUME_FRAC * prep.volume):
        # cells that may be cavities, too small to matter: read as solid, the
        # volume off by at most AMBIGUOUS_VOLUME_FRAC of the solid
        doubled["ambiguous_accepted"] = True
        ambiguous = 0
    if doubled is not None and doubled.get("cells_error"):
        prep, err = None, IouResult(
            Status.BOOLEAN_FAILED,
            detail=f"{label}: cells of shells written twice failed: {doubled['cells_error']}",
            diagnostics=dict(prep.info if prep is not None else err.diagnostics or {}),
            blame="engine")
    elif err is None and doubled is not None and doubled["open"]:
        # its copy stayed open: no patch within the bound of the copy's own
        # parts; cleaning the whole side may close it with the volume of others
        prep, err = None, IouResult(
            Status.OPEN_SURFACE,
            detail=f"{label}: {doubled['open']} shell(s) written twice stay open: their holes "
                   f"are not closed by a patch of proven volume error within 1e-3 of the parts",
            diagnostics=dict(prep.info), blame=label)
    elif err is None and ambiguous:
        # a two-sided record keeps only how often each triangle is written:
        # a hollow body and a body with another one inside read the same
        prep, err = None, IouResult(
            Status.DEGENERATE_VOLUME,
            detail=f"{label}: shells written twice (winding 0) enclose {ambiguous} cell(s), "
                   f"volume {doubled['ambiguous_volume']:.3g} (of the solid's "
                   f"{prep.volume:.3g}), that may be a cavity or a body "
                   "inside a body (or the overlap of two bodies) -- the two-sided record does "
                   "not tell them apart; with fill_cavities every enclosed cell is solid",
            diagnostics=dict(prep.info), blame=label)
    kept = not (doubled is not None and (doubled["open"] or ambiguous
                                         or doubled.get("cells_error")))
    # holes whose patch closes no chain (edges of 3+ faces: cleaning kept one
    # copy of a contact face) -- the raw surface decides, as for open pieces
    holes = (err.diagnostics or {}).get("holes") if err is not None else None
    unclosed = err is not None and err.status is Status.OPEN_SURFACE and bool(holes) and not holes["closed"]
    raw_attempted = False
    if err is not None and kept and ((err.diagnostics or {}).get("split_open") or unclosed):
        raw_attempted = True
        prep, err = _split_open_side(m, label, err, on_self_intersection, fill_cavities,
                                     check_self_intersection)
    elif (err is not None
            and err.status in (Status.OPEN_SURFACE, Status.NOT_MANIFOLD)
            and kept):
        raw_attempted = True
        prep, err = _raw_fallback(m, label, err,
                                  on_self_intersection != "exact" and check_self_intersection,
                                  fill_cavities)
    reading_info = prep.info if prep is not None else (err.diagnostics or {})
    needs_roles = bool(reading_info.get("cavities"))
    needs_pocket = err is not None and err.status is Status.INVERTED_VOLUME
    unbalanced, crowded = _raw_edge_counts(m) if kept and not raw_attempted else (1, 0)
    if (kept and not raw_attempted and unbalanced == 0
            and (needs_roles or needs_pocket or crowded or reading_info.get("flipped_shells"))):
        # An inward shell with no single host may belong to the union of
        # several bodies. Resolve that ownership before flipping it into an
        # unrelated outward body. A genuinely unhosted inverted body keeps
        # the ordinary orientation rule.
        reason = err if err is not None else IouResult(
            Status.NOT_MANIFOLD, detail=f"{label}: raw-surface interpretations require an exact check",
            diagnostics=reading_info, blame=label)
        audited = m
        if reading_info.get("flipped_inverted_mesh"):
            # Repairing the entire orientation must not bypass ambiguity
            # checks. Audit the same globally corrected raw surface.
            audited = m.copy()
            audited.invert()
        candidate, refused = _raw_solid(audited, label, reason, fill=fill_cavities,
                                        check=on_self_intersection != "exact" and check_self_intersection)
        if candidate is not None and (needs_roles or needs_pocket or crowded
                                      or candidate.info["raw_solid"]["assembly"]["assembly_owners"]):
            source_info = {k: v for k, v in reading_info.items() if not k.startswith("inverted_")}
            raw_info = candidate.info["raw_solid"]
            if "shells_dropped" in source_info:
                raw_info["cleaned_shells_dropped"] = source_info.pop("shells_dropped")
            if prep is not None:
                raw_info["audited"] = True
            if candidate.info["raw_solid"]["assembly"]["assembly_owners"]:
                source_info.pop("flipped_shells", None)
            candidate.info = {**source_info, **candidate.info}
            prep, err = candidate, None
        elif (refused is not None and prep is not None and reading_info.get("flipped_shells")
              and refused.status is Status.INVERTED_VOLUME
              and not (refused.diagnostics or {}).get("raw_assembly_candidates")):
            pass   # the ordinary path has already oriented an inside-out group
        elif refused is not None and (needs_roles or needs_pocket or crowded or refused.status is Status.BOOLEAN_FAILED
                                     or (refused.diagnostics or {}).get("raw_assembly_candidates")):
            prep, err = None, refused
    if doubled is not None:
        # faces_raw / vertices_raw count the side as given, not as unfolded
        raw = {"faces_raw": len(m0.faces), "vertices_raw": len(m0.vertices)}
        if prep is not None:
            prep.info.update(raw, doubled_shells=doubled)
        elif err.diagnostics is not None:
            err.diagnostics.update(raw, doubled_shells=doubled)
    if exact_booleans and prep is not None:
        # the other paths (_raw_solid, cells) build in CGAL too; their manifold
        # copy is dropped, so the intersection and a filling stay in CGAL
        prep.solid = None
        prep.info["exact_booleans"] = True
    return prep, err


def _raw_fallback(m: trimesh.Trimesh, label: str, err: IouResult, check: bool, fill: bool
                  ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    The raw-surface fallback (docs/DEFINITION.md 4.9) for a side refused as
    open or not manifold after cleaning: close the raw surface's holes by
    the same proven-error rule, then require every edge to be balanced for
    its solid under the raw readings (`_raw_solid`). A rejected patch or a remaining wall
    keeps the original refusal, including with `fill`.  `check` (a side in "fail" with the
    self-intersection check): each body in every admissible reading must
    avoid winding 2 or more after its cavities are subtracted. Both valid
    and defective readings make the input ambiguous. With `check`, the solid taken
    carries `winding_max` (the largest winding of the sum) in its `raw_solid`
    diagnostics, and `self_overlap_parts` 0 where that is 2 or more.
    """
    raw, patch = _raw_patched(m)
    if ((patch.get("holes") and not patch["holes"]["accepted"])
            or _unbalanced_edges(raw)):
        return None, err
    prep, e = _raw_solid(raw, label, err, **({"fill": True} if fill else {}),
                        **({"check": True} if check else {}))
    if prep is not None:
        prep.info.update(faces_raw=len(m.faces), vertices_raw=len(m.vertices))
        prep.info["raw_solid"].update(patch)
    return prep, e


PARTS_LABELLED_MAX = 256    # components labelled in the solid's own arrangement


def _raw_parts(raw: trimesh.Trimesh, fill: bool) -> tuple:
    """
    Reference for the former component-based reading, not a preparation path.
    The solid of a balanced raw surface (`_cgal_solid`), the largest winding
    of the sum over its cells and the number of components (faces joined by
    edges, `_face_components`) that each wind 2 or more by themselves --
    from one arrangement, each component labelled (`_cgal_solid_parts`).
    Over PARTS_LABELLED_MAX components (W holds two numbers per face and
    component) the sum alone, and the components labelled in a second
    arrangement only where the sum winds 2 or more.
    """
    comp = _face_components(raw)
    if len(comp) and int(comp.max()) + 1 > PARTS_LABELLED_MAX:
        got, w_max, _ = _cgal_solid_parts(raw, np.zeros(len(comp), np.int64), fill)
        through = _cgal_solid_parts(raw, comp)[2] if w_max >= 2 else 0
        return got, w_max, through
    return _cgal_solid_parts(raw, comp, fill)


def _raw_solid(m: trimesh.Trimesh, label: str, err: IouResult, fill: bool = False,
               check: bool = False) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """Read all valid oriented raw decompositions and assembly cavity owners.

    Ordinary edges stay paired before the geometric cut. Alternative readings
    must agree on exact nonzero-volume cells, after filling when requested.
    A budget refusal is an engine failure; ambiguity is invalid geometry.
    """
    from .raw_solid import resolve, InvalidReading

    t0 = time.perf_counter()
    info = {"faces_raw": len(m.faces), "vertices_raw": len(m.vertices),
            "self_intersection_checked": False}
    try:
        solid_mesh, volume, reading = resolve(m, check=check, fill=fill)
    except InvalidReading as exc:
        if exc.kind == "empty":
            return None, err
        status = {"pocket": Status.INVERTED_VOLUME, "overlap": Status.SELF_INTERSECTING,
                  "ambiguous": Status.DEGENERATE_VOLUME}[exc.kind]
        return None, IouResult(status, detail=f"{label}: {exc}",
                               diagnostics={**(err.diagnostics or {}), **info, **exc.diagnostics}, blame=label)
    except Exception as exc:
        return None, IouResult(
            Status.BOOLEAN_FAILED,
            detail=f"{err.detail}; raw assembly resolution raised {type(exc).__name__}: {exc}",
            diagnostics=info, blame="engine")
    if volume <= (m.scale ** 3) * 1e-9:
        return None, err
    inside = _interior_points(solid_mesh)
    inside = inside if inside is not None else np.empty((0, 3))
    man = _resolved_to_manifold(solid_mesh, volume)
    info["raw_solid"] = {"engine": "libigl-cgal", "refused_as": err.status.value,
                         "refused_detail": err.detail, "faces_out": len(solid_mesh.faces),
                         "probes": len(inside), "manifold": man is not None,
                         "seconds": round(time.perf_counter() - t0, 3),
                         "assembly": reading}
    if check:
        info["raw_solid"]["winding_max"] = reading["winding_max"]
        if reading["winding_max"] >= 2:
            info["raw_solid"]["self_overlap_parts"] = 0
    for key in ("inverted_cavity_volume", "inverted_cavity_faces",
                "cavities_filled", "volume_cavities", "flipped_inverted_mesh"):
        if key in reading:
            info[key] = reading[key]
    info["cavities_solid"] = (reading["cavities_filled_range"][1] if fill
                               else _solid_cavities(solid_mesh))
    if fill:
        info["raw_filled"] = True
    # The chosen material can differ from the summed raw winding for a
    # cavity belonging to an assembly. Body probes must describe this solid.
    return _Prepared(solid_mesh, man, volume, info, shells=solid_mesh, probes=inside), None


def _split_open_side(m: trimesh.Trimesh, label: str, err: IouResult, mode: str,
                     fill: bool, check: bool = True
                     ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    A side whose split leaves open pieces (`err`: edges of 3+ faces the
    package module cannot pair; docs/DEFINITION.md 4.4), decided by its raw
    surface (vertices merged exactly, repeated faces kept), in either mode:
    - every edge balanced (bodies touching with the same triangulation: the
      contact written once each way) -- the winding number is an integer;
      `_raw_solid` checks every shell and cavity-owner reading. Mixed
      validity or different material is ambiguous, and in "fail" every
      reading must avoid a body winding 2 or more after cavity subtraction;
    - unbalanced (a one-sided wall inside a closed surface: the winding is
      fractional on its two sides) -- a partition and the boundary of a void
      read the same, so the refusal stays; with the side's cavities filled
      every bounded cell is solid under any reading (`_filled_cells`).  A
      side in "fail" is then refused on any crossing or touching pair inside
      one component of its cleaned surface: with no winding number, a pass
      through itself is not found otherwise; a pair between two components
      is an overlap of parts, and every bounded cell is solid anyway.
    Balance is taken after the holes of the raw surface are closed as
    cleaning closes them (`_patch_loops`): the edges of a
    crack are a hole, not a wall; a wall (edges of 3+ faces) stays unbalanced.
    Holes with no patch of proven error -- beside a wall, always: the patch
    closes no chain -- leave the side open (`open_surface`).
    In "fail" both checks are the self-intersection check (`check`).
    trimesh's fill_holes, which closed such pieces before (any hole of 3-4
    edges, uncounted), is not used.
    """
    raw, patch = _raw_patched(m)
    holes = patch.get("holes")
    if holes and not holes["accepted"]:
        # no patch of proven error (by the cells either: they need the holes closed)
        return None, IouResult(
            Status.OPEN_SURFACE, detail=f"{err.detail}; the raw surface: {_holes_left_open(holes)}",
            diagnostics={**(err.diagnostics or {}), "raw_holes": holes}, blame=label)
    if _unbalanced_edges(raw) == 0:
        # Exact construction also records the fail classification for Body.
        prep, e = _raw_solid(raw, label, err, **({"fill": True} if fill else {}),
                             **({"check": True} if mode != "exact" and check else {}))
        if prep is not None:
            prep.info.update(faces_raw=len(m.faces), vertices_raw=len(m.vertices))
            prep.info["raw_solid"]["rule"] = "split_open"
            if patch:
                prep.info["raw_solid"].update(patch)
        return prep, e
    detail = (f"{err.detail}; the raw surface has unbalanced edges (winding number not "
              "defined" + (" after closing its cracks" if patch.get("loop_faces_added") else "")
              + "): a wall inside a body and the boundary of a void read the same")
    if not fill:
        return None, replace(err, detail=detail + "; with fill_cavities every enclosed "
                                                   "cell is solid")
    if mode != "exact" and check:
        # no winding to find a pass through itself: any crossing or touching
        # pair inside one component of the cleaned surface (faces joined by
        # edges), as "fail" refuses inside a body; a pair between two
        # components is an overlap of parts, whose cells are all solid anyway
        cleaned = _clean(m)[0]
        comp = _face_components(cleaned)
        found = 0
        try:
            for k in range(int(comp.max()) + 1 if len(comp) else 0):
                part = trimesh.Trimesh(cleaned.vertices, np.asarray(cleaned.faces)[comp == k],
                                       process=False)
                if len(self_intersecting_pairs(part, first=True)):
                    found += 1
                    break
        except Exception as exc:                        # noqa: BLE001
            return None, IouResult(
                Status.BOOLEAN_FAILED,
                detail=f"{label}: self-intersection check failed ({type(exc).__name__}: {exc})",
                diagnostics=err.diagnostics, blame="engine")
        if found:
            return None, IouResult(
                Status.SELF_INTERSECTING,
                detail=f"{label}: intersecting or touching triangles inside one part (faces "
                       "joined by edges) of a surface cut into open pieces, whose winding "
                       "number is not defined; its surface is not proven free of "
                       "self-intersection",
                diagnostics=err.diagnostics, blame=label)
    return _filled_cells(m, raw, patch.get("loop_faces_added", 0), label,
                         replace(err, detail=detail))


def _holes_left_open(pc: dict) -> str:
    """Why the holes of a `_patch_loops` record were left open."""
    return (f"{pc['loops']} holes left open: "
            + ("the parts they bound are not closed by the patch (edges of 3+ faces "
               "or run twice one way)" if not pc["closed"] else
               f"the patch's proven volume error is {pc['worst_ratio']:.3g} of the parts "
               f"it closes, above {seams.SEAM_ETA:g}"))


def _raw_patched(m: trimesh.Trimesh
                 ) -> tuple[trimesh.Trimesh, dict]:
    """
    The raw surface (vertices merged exactly, faces on fewer than three
    vertices dropped, repeated faces kept) with its boundary loops closed as
    cleaning closes them (`_patch_loops`), and a record of the patches ({}
    without any).  Without a boundary edge it is `m` itself: the solid is
    built from the surface as given.
    """
    v = np.ascontiguousarray(np.asarray(m.vertices, np.float64) + 0.0)     # -0.0 -> 0.0
    _, first, inv = np.unique(v.view(np.dtype((np.void, 24))).ravel(),
                              return_index=True, return_inverse=True)
    f = inv.reshape(-1)[np.asarray(m.faces, np.int64)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    raw = trimesh.Trimesh(v[first], f, process=False)
    if not len(f) or not _boundary_edges(raw):
        return m, {}
    added, frac, record = _patch_loops(raw)
    if not added:
        return m, {"loop_faces_added": 0, **({"holes": record} if record else {})}
    out = {"loop_faces_added": added, "patched_area_frac": frac}
    if record is not None:
        out["holes"] = record
    return raw, out


def _filled_cells(m: trimesh.Trimesh, raw: trimesh.Trimesh, n: int, label: str,
                  err: IouResult) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    Every bounded cell of space cut by the side's raw surface (with the
    patches of its cracks, written both ways), for a side whose cavities are
    filled: that is the side's solid under any reading of a one-sided wall.
    `raw`, `n`: the raw surface with its `n` patch faces last (`_raw_patched`)
    -- the patches its balance was taken with; cleaning's may be refused
    where the raw surface's are not (a contact face kept once), and a crack
    left open joins a cell to the outer one.
    Refused as inverted where a triangle on the outer cell faces inward
    (material outside).  Built in CGAL from the outer cell's boundary.
    """
    from .cells import cells
    F = np.asarray(raw.faces, np.int64)
    V, F = np.asarray(raw.vertices, np.float64), np.concatenate([F, F[len(F) - n:, ::-1]])
    try:
        c = cells(V, F)
    except Exception as exc:                            # noqa: BLE001
        return None, IouResult(
            Status.BOOLEAN_FAILED,
            detail=f"{err.detail}; cells of the raw surface failed: {type(exc).__name__}: {exc}",
            diagnostics=err.diagnostics, blame="engine")
    outer = (c.below == 0) ^ (c.above == 0)
    # w just inside an outer triangle: its multiplicity toward the outer cell
    inside = np.where(c.above == 0, c.multiplicity, -c.multiplicity)[outer]
    info = {"faces_raw": len(m.faces), "vertices_raw": len(m.vertices),
            "self_intersection_checked": False}
    if (inside < 0).any():
        return None, IouResult(
            Status.INVERTED_VOLUME,
            detail=f"{label}: {int((inside < 0).sum())} triangles on the outside of the "
                   "surface face inward (material outside); the mesh does not define its solid",
            diagnostics={**info, "inverted_faces": int((inside < 0).sum())}, blame=label)
    prep, e = _raw_solid(_outer_boundary(c), label, err)
    if prep is None:
        return None, e
    raw = prep.info.pop("raw_solid")
    prep.info.update(info, cells_filled={"cells": c.num_cells - 1, "faces_out": raw["faces_out"],
                                         "patch_faces": n, "manifold": raw["manifold"],
                                         "seconds": raw["seconds"],
                                         "refused_detail": err.detail})
    return prep, None


def _prepare_shells(m: trimesh.Trimesh, label: str, *, check_self_intersection: bool,
                    on_self_intersection: str, cut: Optional[trimesh.Trimesh] = None,
                    fill_cavities: bool = False, exact: bool = False
                    ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """Repair, orient and solidify one already-normalised side, shell by shell.
    `exact` (`exact_booleans`): every body is built in CGAL, the side's solid
    stays there (no manifold boolean, `_cgal_side`).
    `cut`: a closed surface of the exact cut (the cells enclosed by shells
    written twice, `_doubled_shells`) -- one more body, neither cleaned nor
    checked by the detector, built in CGAL with the bodies around it.
    `fill_cavities`: the side's cavities will be filled -- an inverted pocket
    lying in cavities of its cluster's solid is not refused; the solid is
    built without it (w >= 1), and the filling takes the pocket in."""
    if len(m.faces) or cut is None:
        m, info = _clean(m)
    else:
        info = {"faces_raw": 0, "vertices_raw": len(m.vertices), "degenerate_removed": 0,
                "duplicate_removed": 0, "boundary_edges_before_fill": 0,
                "loop_faces_added": 0, "patched_area_frac": 0.0}
    if len(m.faces) == 0 and cut is None:
        return None, IouResult(Status.EMPTY_MESH, detail=f"{label}: all faces degenerate",
                               diagnostics=info, blame=label)

    open_edges = (info["boundary_edges_before_fill"] if not info["loop_faces_added"]
                  else _boundary_edges(m))
    if open_edges:
        pc = info.get("holes")
        why = ""
        if pc is not None and not pc["accepted"]:
            why = "; " + _holes_left_open(pc)
        return None, IouResult(
            Status.OPEN_SURFACE,
            detail=f"{label}: {open_edges} boundary edges remain open{why}",
            diagnostics={**info, "open_edges": open_edges},
            blame=label,
        )

    shells, roles, parent, oinfo = (_orient_shells(m, repair=False) if len(m.faces)
                                    else ([], [], [], {}))
    info.update(oinfo)
    if oinfo.get("split_open"):
        # edges of 3+ faces the module cannot pair cut the surface into open
        # pieces: a one-sided wall inside a closed surface, or bodies touching
        # with the same triangulation (one copy of the contact left by
        # cleaning); `_prepare` decides by the raw surface
        nonmanifold = _nonmanifold_edges(m)
        return None, IouResult(
            Status.NOT_MANIFOLD,
            detail=f"{label}: {nonmanifold} edges shared by 3+ faces cut the surface into "
                   f"{oinfo['split_open']} open pieces",
            diagnostics={**info, "nonmanifold_edges": nonmanifold}, blame=label)
    n_own = len(shells)              # shells of the cleaned side; the cut comes last
    if cut is not None:
        shells, roles, parent = shells + [cut], roles + ["solid"], parent + [-1]
    if not shells:
        return None, IouResult(
            Status.DEGENERATE_VOLUME,
            detail=f"{label}: volume ~ 0 (flat or self-cancelling surface)",
            diagnostics=info, blame=label)

    # group = the solid each shell belongs to (a cavity joins its host)
    group = [i if roles[i] == "solid" else parent[i] for i in range(len(shells))]
    orphan = [i for i in range(len(shells)) if group[i] < 0]
    if orphan:
        return None, IouResult(
            Status.SELF_INTERSECTING,
            detail=f"{label}: {len(orphan)} cavity shells have no enclosing body",
            diagnostics=info, blame=label)
    solids = sorted(set(group))
    info["solids"] = len(solids)
    info["cavities"] = roles.count("cavity")

    info["self_intersection_checked"] = False
    to_cgal: set = set()             # bodies whose w >= 1 solid CGAL builds (5.3)
    suspect: set = set()             # of them, looked at for a pass through itself
    # bodies not proven embedded: their own w may be -1 (a pocket the sum of
    # a cluster hides inside another body); an embedded body winds 0 or 1
    loose: set = set(solids)
    crossing_pairs: Optional[int] = None
    if check_self_intersection:
        # "fail" needs the bodies with a crossing (built in CGAL, refused only
        # where one passes through itself), not the count of their pairs.
        # T-junctions of a body with pairs are split first (same surface, same
        # w: a touch at a vertex on an edge sends a body to CGAL for nothing);
        # the pairs that decide are those of the split shells, and the bodies
        # left with pairs go to CGAL split as well
        split = {"t_junctions_split": 0, "collinear_faces_removed": 0}
        n_cross, bad, unchecked, error = _crossings(shells[:n_own], group[:n_own],
                                                    first=on_self_intersection != "exact",
                                                    split=split)
        info.update(split)
        if unchecked:
            info["self_intersection_checked"] = None     # the detector failed
            info["unchecked_bodies"] = len(unchecked)
            info["detector_error"] = error
        else:
            info["self_intersection_checked"] = True
        info["crossing_pairs_internal"] = n_cross
        crossing_pairs = None if unchecked and not n_cross else n_cross
        if unchecked and on_self_intersection != "exact":
            # unproven clean: no number without the check
            return None, IouResult(
                Status.BOOLEAN_FAILED,
                detail=f"{label}: self-intersection check failed on {len(unchecked)} "
                       f"bodies ({error})",
                diagnostics=info, blame="engine")
        # a crossing inside a body: its w >= 1 solid from the exact arrangement;
        # in "fail" the same arrangement tells a touch (a number) from a pass
        # through itself (w >= 2: refused).  Unproven clean ("exact" only): a
        # hidden crossing would make manifold wrong
        to_cgal = bad | unchecked
        loose = bad | unchecked
        if on_self_intersection != "exact":
            suspect = set(bad)
    if cut is not None:
        to_cgal = to_cgal | {len(shells) - 1}
    if exact:
        to_cgal = set(solids)

    if not _HAS_ENGINE:
        return None, IouResult(Status.NO_ENGINE,
                               detail="no boolean engine available (pip install manifold3d)",
                               diagnostics=info, blame="engine")

    whole = trimesh.util.concatenate(shells) if len(shells) > 1 else shells[0]
    hosts = [shells[g] for g in solids]
    boxes = [shells[g].bounds for g in solids]
    # A body to resolve goes to CGAL together with every body whose extent
    # reaches it: the solid is w >= 1 of the SUM of their windings (a body in
    # another's inverted pocket is not material).  Bodies apart from it are
    # resolved or built on their own -- a whole multi-body side in one CGAL
    # union was seen to drop a separate body (mcb/00015262).
    units = []                     # (body indices into solids, resolve in CGAL?)
    for cl in _clusters(boxes):
        if any(solids[k] in to_cgal for k in cl):
            units.append((cl, True))
        else:
            units.extend(([k], False) for k in cl)
    parts, meshes, part_boxes = [], [], []   # manifold (None: CGAL only), CGAL mesh, extent
    t0, faces_in, faces_out, probes, n_cgal = time.perf_counter(), 0, 0, 0, 0
    self_through = [0, 0.0]          # bodies winding 2+ around a volume, that volume
    in_cavities = [0, 0.0]           # pockets in cavities (fill_cavities): faces, volume
    for ks, in_cgal in units:
        members = [shells[i] for i in range(len(shells)) if group[i] in {solids[k] for k in ks}]
        if in_cgal:
            cluster = [[hosts[k]] + [shells[i] for i in range(len(shells))
                                     if group[i] == solids[k] and shells[i] is not hosts[k]]
                       for k in ks]
            res, why = _resolve_bodies(
                cluster, label, check=tuple(j for j, k in enumerate(ks) if solids[k] in suspect),
                pockets=tuple(j for j, k in enumerate(ks) if solids[k] in loose),
                **({"cavity": True} if fill_cavities else {}), **({"exact": True} if exact else {}))
            if res is None:
                nonmanifold = _nonmanifold_edges(m)
                unbalanced = _unbalanced_edges(trimesh.util.concatenate(members))
                if nonmanifold or unbalanced:
                    # An unbalanced cluster cannot define a piecewise-constant
                    # winding number, even with only two faces at every edge.
                    # This is a defect of the side, not a failed boolean.
                    detail = (f"{label}: {nonmanifold} edges shared by 3+ faces; {why}"
                              if nonmanifold else why)
                    return None, IouResult(
                        Status.NOT_MANIFOLD,
                        detail=detail,
                        diagnostics={**info, "nonmanifold_edges": nonmanifold,
                                     "unbalanced_edges": unbalanced,
                                     "crossing_pairs": crossing_pairs},
                        blame=label)
                return None, IouResult(Status.BOOLEAN_FAILED, detail=why,
                                       diagnostics={**info, "crossing_pairs": crossing_pairs},
                                       blame="engine")
            solid_mesh, _, man, used, inv_faces, inv_volume, self_over, in_cavity = res
            # w <= -1 anywhere: no defined solid -- an inverted feature added
            # to a body (meant H u F) and a cut that leaks through the wall
            # (meant H \ F) give the same oriented surface, w = 1_H - 1_F --
            # unless the pocket lies in a cavity of the solid and the side's
            # cavities are filled: then both fill to one solid
            if inv_faces and in_cavity:
                in_cavities[0] += inv_faces
                in_cavities[1] += inv_volume
            elif inv_faces:
                return None, IouResult(
                    Status.INVERTED_VOLUME,
                    detail=f"{label}: the surface winds -1 or less around a volume of "
                           f"{inv_volume:.3g} in one cluster of bodies (an inverted "
                           "pocket); the mesh does not define its solid",
                    diagnostics={**info, "crossing_pairs": crossing_pairs,
                                 "inverted_volume": inv_volume, "inverted_faces": inv_faces},
                    blame=label)
            through = [(f, v) for _, f, v in self_over if f]
            self_through[0] += len(through)
            self_through[1] += sum(v for _, v in through)
            if through and on_self_intersection != "exact":
                # a body winding twice around a volume passes through itself;
                # a touch inside a body (crossing pairs, w <= 1) is a number
                return None, IouResult(
                    Status.SELF_INTERSECTING,
                    detail=through_detail(label, sum(v for _, v in through)),
                    diagnostics={**info, "crossing_pairs": crossing_pairs,
                                 "self_overlap_bodies": len(through),
                                 "self_overlap_volume": sum(v for _, v in through)},
                    blame=label)
            faces_in +=sum(len(sh.faces) for sh in members)
            faces_out += len(solid_mesh.faces)
            probes += used
            n_cgal += len(ks)
            parts.append(man)
            meshes.append(solid_mesh)
        else:
            man = _to_manifold(members)
            if man.status() != manifold3d.Error.NoError:
                return None, IouResult(
                    Status.NOT_MANIFOLD,
                    detail=f"{label}: a body is not a valid manifold ({man.status()})",
                    diagnostics={**info, "nonmanifold_edges": _nonmanifold_edges(m)},
                    blame=label)
            parts.append(man)
            meshes.append(None)
        part_boxes.append(np.array([np.min([boxes[k][0] for k in ks], axis=0),
                                    np.max([boxes[k][1] for k in ks], axis=0)]))
    # set before the CGAL-only return: its side reports the union as well
    if in_cavities[0]:
        info.update(inverted_cavity_volume=in_cavities[1], inverted_cavity_faces=in_cavities[0])
    overlap = len(_clusters(part_boxes)) < len(part_boxes)
    info["union"] = bool(overlap) or any(in_cgal and len(ks) > 1 for ks, in_cgal in units)
    if to_cgal:
        info["resolved"] = {"engine": "libigl-cgal", "crossing_pairs": crossing_pairs,
                            "bodies": n_cgal, "faces_in": faces_in,
                            "faces_out": faces_out, "probes": probes,
                            "manifold": all(p is not None for p in parts),
                            "seconds": round(time.perf_counter() - t0, 3)}
        if on_self_intersection != "exact" and check_self_intersection:
            # looked for in "fail" only (0 here: a pass through itself is
            # refused); a body file without them came from "exact"
            info["resolved"].update(self_overlap_bodies=self_through[0],
                                    self_overlap_volume=self_through[1])
        if not info["resolved"]["manifold"]:
            return _cgal_side(parts, meshes, whole, hosts, info, label, m)

    if len(parts) == 1:
        solid = parts[0]
    elif overlap:
        solid = _union(parts)
    else:
        solid = manifold3d.Manifold.compose(parts)   # disjoint: no boolean needed

    volume = float(solid.volume())
    if volume <= (m.scale ** 3) * 1e-9:
        return None, IouResult(
            Status.DEGENERATE_VOLUME,
            detail=f"{label}: volume ~ 0 (flat or self-cancelling surface)",
            diagnostics={**info, "volume": volume}, blame=label)
    info["volume_sum_of_solids"] = float(sum(p.volume() for p in parts))
    mesh = _from_manifold(solid)
    info["cavities_solid"] = _solid_cavities(mesh)
    return _Prepared(mesh, solid, volume, info, whole, hosts), None


def _solid_cavities(mesh: trimesh.Trimesh) -> int:
    """
    Closed cavities of a built solid: its inward shells -- those
    `fill_cavities` fills.  Unlike `cavities` (inward shells of the raw
    surface), it sees a void enclosed by several overlapping bodies or made
    by CGAL.  Shells are the vertex-connected components (cheaper than
    `split`, which builds a submesh per shell); signed volume per component.
    A shell counts only if its volume is negative beyond the rounding bound
    of its sum: a flat sliver, whose sign is rounding, is not a cavity.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(mesh.faces)
    if not len(f):
        return 0
    n = len(mesh.vertices)
    rows = np.concatenate([f[:, 0], f[:, 1]])
    cols = np.concatenate([f[:, 1], f[:, 2]])
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    k, labels = connected_components(graph, directed=False)
    v = np.asarray(mesh.vertices, np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    comp = labels[f[:, 0]]
    vol = np.bincount(comp, np.einsum("ij,ij->i", a, np.cross(b, c)), minlength=k)
    # each monomial of a.(b x c) takes 5 roundings, a component's sum of n
    # terms n - 1 more: within gamma_{n+4} of the sum of the monomials'
    # magnitudes (Higham), and (n + 5) eps covers it with a factor 2 to spare
    a, b, c = np.abs(a), np.abs(b), np.abs(c)
    mono = np.einsum("ij,ij->i", a, b[:, [1, 2, 0]] * c[:, [2, 0, 1]]
                     + b[:, [2, 0, 1]] * c[:, [1, 2, 0]])
    bound = ((np.bincount(comp, minlength=k) + 5) * np.finfo(np.float64).eps
             * np.bincount(comp, mono, minlength=k))
    return int((vol < -bound).sum())


def _cgal_side(parts, meshes, whole, hosts, info: dict, label: str, m: trimesh.Trimesh
               ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    A side with a CGAL-resolved body that manifold would not take: its solid
    stays a mesh, the union of the bodies' solids in CGAL (checked by probes),
    and the pair intersection runs in CGAL.
    """
    bodies = [mesh if mesh is not None else _from_manifold(man)
              for man, mesh in zip(parts, meshes)]
    joined = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
    try:
        solid_mesh, volume = (_cgal_solid(joined)[:2] if len(bodies) > 1
                              else (joined, float(joined.volume)))
    except Exception as exc:
        return None, IouResult(Status.BOOLEAN_FAILED,
                               detail=f"{label}: CGAL union raised {type(exc).__name__}: {exc}",
                               diagnostics=info, blame="engine")
    used, lost = _lost_probes(_probes(hosts), [whole], solid_mesh)
    if lost:
        return None, IouResult(Status.BOOLEAN_FAILED,
                               detail=f"{label}: CGAL union of the bodies put {lost} of "
                                      f"{used} probe points on the wrong side", diagnostics=info, blame="engine")
    if volume <= (m.scale ** 3) * 1e-9:
        return None, IouResult(
            Status.DEGENERATE_VOLUME,
            detail=f"{label}: volume ~ 0 (flat or self-cancelling surface)",
            diagnostics={**info, "volume": volume}, blame=label)
    info["cavities_solid"] = _solid_cavities(solid_mesh)
    return _Prepared(solid_mesh, None, volume, info, whole, hosts), None


# --------------------------------------------------------------------------- #
# per-mesh triage
# --------------------------------------------------------------------------- #


def _count_pairs(mesh: trimesh.Trimesh) -> Optional[int]:
    """Intersecting face pairs of `mesh`; None if the detector failed."""
    try:
        return int(len(self_intersecting_pairs(mesh)))
    except Exception:                   # noqa: BLE001
        return None


def diagnose_mesh(mesh, *, check_self_intersection: bool = True) -> dict:
    """
    Describe what is wrong with a single mesh, without computing any IoU.

      * `boundary_edges` > 0 after closing the holes whose patch has a proven
        volume error within 1e-3 (docs/DEFINITION.md 4.3) -- the triangulation
        is open (TESSELLATION);
      * `nonmanifold_edges` > 0 -- edges shared by 3+ faces;
      * `self_intersections` > 0 -- face pairs that meet beyond a shared
        vertex or edge (crossings, and contacts such as coplanar faces on
        each other), whether between bodies (resolved by the union) or inside
        one (see `crossing_pairs_internal`);
      * `solids`, `cavities` -- how the shells were read (docs/DEFINITION.md 4.6);
        a void enclosed by overlapping bodies is not among them -- the built
        solid's `cavities_solid` (`volume_iou`, `to_body`) is;
      * `volume` -- signed divergence volume of the cleaned mesh.

    `self_intersections` is None when the detector (CGAL) failed on the mesh
    or was not run (`check_self_intersection=False`).
    """
    out: dict = {"loaded": False}
    m = _as_mesh(mesh)
    if m is None:
        return out
    out["loaded"] = True
    out["faces_raw"] = len(m.faces)
    out["vertices_raw"] = len(m.vertices)
    out["finite_coords"] = bool(len(m.vertices)) and bool(np.isfinite(m.vertices).all())
    if not out["faces_raw"] or not out["finite_coords"]:
        return out

    m, info = _clean(m)
    out.update({k: v for k, v in info.items() if k not in ("faces_raw", "vertices_raw")})
    out["faces"] = len(m.faces)
    out["vertices"] = len(m.vertices)
    out["boundary_edges"] = _boundary_edges(m)
    out["nonmanifold_edges"] = _nonmanifold_edges(m)
    out["watertight"] = bool(m.is_watertight)
    out["volume"] = float(m.volume) if len(m.faces) else 0.0
    # the count of split(only_watertight=False), without building a submesh per part
    out["components"] = len(trimesh.graph.connected_components(
        m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)) if len(m.faces) else 0
    out["self_intersections"] = None
    if out["boundary_edges"] == 0 and len(m.faces):
        shells, roles, parent, oinfo = _orient_shells(m)
        out.update(oinfo)
        out["solids"] = roles.count("solid")
        out["cavities"] = roles.count("cavity")
        if check_self_intersection and shells:
            group = [i if roles[i] == "solid" else parent[i] for i in range(len(shells))]
            n_cross, _, unchecked, _ = _crossings(shells, group)
            out["crossing_pairs_internal"] = None if unchecked else n_cross
            out["self_intersections"] = _count_pairs(
                trimesh.util.concatenate(shells) if len(shells) > 1 else shells[0])
    elif check_self_intersection:
        out["self_intersections"] = _count_pairs(m)
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


SELF_INTERSECTION_MODES = ("exact", "fail")


def _side_modes(on_self_intersection) -> tuple[str, str]:
    """(gt mode, pred mode) from one mode for both sides or a pair."""
    modes = ((on_self_intersection,) * 2 if isinstance(on_self_intersection, str)
             else tuple(on_self_intersection))
    if len(modes) != 2 or any(m not in SELF_INTERSECTION_MODES for m in modes):
        raise ValueError("on_self_intersection must be 'exact' or 'fail', or a "
                         f"(gt, pred) pair of them; got {on_self_intersection!r}")
    return modes


def _side_fill(fill_cavities) -> tuple[bool, bool]:
    """(gt, pred) from one flag for both sides or a pair."""
    fill = ((fill_cavities,) * 2 if isinstance(fill_cavities, bool)
            else tuple(fill_cavities) if isinstance(fill_cavities, (tuple, list)) else ())
    if len(fill) != 2 or any(not isinstance(f, bool) for f in fill):
        raise ValueError("fill_cavities must be a bool or a (gt, pred) pair of bools; "
                         f"got {fill_cavities!r}")
    return fill


def _fill_cavities(prep: _Prepared, label: str, exact: bool = False
                   ) -> tuple[Optional[_Prepared], Optional[IouResult]]:
    """
    The side's solid with its closed cavities filled: the union of the outward
    shells of the built solid (a body inside a cavity is absorbed).  Taken
    from the solid, not the raw shells, so a void enclosed by several
    overlapping bodies is a cavity too.  The union is manifold's only for
    shells without face pairs; otherwise, or with `exact` (`exact_booleans`),
    it is CGAL's, checked by probes.
    """
    if prep.info.get("raw_filled"):
        return prep, None
    shells, vols, _ = _split(prep.mesh)
    n_inward = _solid_cavities(prep.mesh)       # the count reported as cavities_solid
    info = {**prep.info, "cavities_filled": n_inward}
    if not n_inward:
        return _Prepared(prep.mesh, prep.solid, prep.volume, info, prep.shells,
                         prep.hosts, prep.probes), None
    outward = [s for s, v in zip(shells, vols) if v > 0]
    joined = trimesh.util.concatenate(outward) if len(outward) > 1 else outward[0]
    if exact or not _embedded(joined):
        # a rounded CGAL solid whose shells cross or touch (DEFINITION 7), or
        # the exact mode: the union of the outward shells in CGAL
        try:
            mesh, volume = _cgal_solid(joined)[:2]
        except Exception as exc:
            return None, IouResult(Status.BOOLEAN_FAILED,
                                   detail=f"{label}: CGAL union of the outward shells raised "
                                          f"{type(exc).__name__}: {exc}",
                                   diagnostics=info, blame="engine")
        used, lost = _lost_probes(prep.probe_points(), [joined], mesh)
        if lost:
            return None, IouResult(Status.BOOLEAN_FAILED,
                                   detail=f"{label}: CGAL union of the outward shells put {lost} "
                                          f"of {used} probe points on the wrong side",
                                   diagnostics=info, blame="engine")
        info["volume_cavities"] = volume - prep.volume
        return _Prepared(mesh, None, volume, info, shells=mesh, probes=prep.probe_points()), None
    parts = [_to_manifold([s]) for s in outward]
    if any(p.status() != manifold3d.Error.NoError for p in parts):
        # a CGAL-resolved solid may glue a cavity to its host at an edge that
        # _split leaves to trimesh: the outward shells cannot be told apart
        return None, IouResult(
            Status.BOOLEAN_FAILED,
            detail=f"{label}: cavities of a CGAL-resolved solid could not be filled "
                   "(its shells are not separate manifolds)",
            diagnostics=info, blame="engine")
    solid = parts[0] if len(parts) == 1 else _union(parts)
    volume = float(solid.volume())
    info["volume_cavities"] = volume - prep.volume
    mesh = _from_manifold(solid)
    # the filled solid is its own w >= 1 surface; the probes lie in the
    # unfilled solid, which the filled one contains
    return _Prepared(mesh, solid, volume, info, shells=mesh, probes=prep.probe_points()), None


DEFAULT_ON_SELF_INTERSECTION = ("exact", "fail")
SIDES = ("gt", "pred")


class PreparedSide:
    """
    One side of the pair, loaded and built once (`prepare_side`), for many
    `volume_iou` calls: the side's solid, or the refusal it would get.  The
    result of a call is the same as with the mesh or body it was made from.
    A call does not change it; it holds the solid in memory as long as it lives.
    """

    def __init__(self, obj, side: str, check_self_intersection: bool,
                 on_self_intersection: str, fill_cavities: bool, exact_booleans: bool = False):
        self.side = side
        self.check_self_intersection = check_self_intersection
        self.on_self_intersection = on_self_intersection
        self.fill_cavities = fill_cavities
        self.exact_booleans = exact_booleans
        self._shared = False         # True: handed to the caller, reused across calls
        self.raw, self.body, self._load_err = _load_side(obj, side)
        self._prep: Optional[_Prepared] = None
        self._prep_err: Optional[IouResult] = None
        self._fill_err: Optional[IouResult] = None
        self._built = self._filled = False

    # The built solid (a manifold3d object) does not pickle: a pickled side
    # (`volume_iou(..., timeout=)` sends it to a child process) carries its
    # loaded source and is built again on arrival, as `prepare_side` built it --
    # the same result, at the price of the build.
    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.update(_prep=None, _prep_err=None, _fill_err=None, _built=False, _filled=False)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._shared:
            self._prepare()

    def _prepare(self) -> None:
        # a gt out of the unit cube is refused before it is built (volume_iou)
        if self._load_err is None and (self.side == "pred" or _out_of_cube(self.raw) <= UNIT_TOL):
            if self._build() is None:
                self._fill()

    def __repr__(self) -> str:
        state = (f"refused: {self.error.status.value}" if self.error is not None
                 else f"volume {self._prep.volume:.6g}" if self._prep is not None
                 else "not built")
        exact = ", exact_booleans=True" if getattr(self, "exact_booleans", False) else ""
        return (f"PreparedSide({self.side}, {self.on_self_intersection}, "
                f"fill_cavities={self.fill_cavities}{exact}, {state})")

    @property
    def error(self) -> Optional[IouResult]:
        """The refusal of this side alone (load, unit cube for a gt, build, cavity
        filling), or None."""
        # not `or`: a refusal is falsy (IouResult.__bool__)
        err = next((e for e in (self._load_err, self._prep_err, self._fill_err)
                    if e is not None), None)
        if err is None and self.side == "gt" and self._prep is None:
            err = _check_normalized(self.raw, self.raw)[0]
        return None if err is None else self._out(err)

    def _build(self) -> Optional[IouResult]:
        if not self._built:
            self._built = True
            from . import body as B
            self._prep, self._prep_err = (
                B.prepared(self.body, self.side, self.on_self_intersection, self.exact_booleans)
                if self.body is not None
                else _prepare(self.raw, self.side,
                              check_self_intersection=self.check_self_intersection,
                              on_self_intersection=self.on_self_intersection,
                              fill_cavities=self.fill_cavities,
                              exact_booleans=self.exact_booleans))
        return None if self._prep_err is None else self._out(self._prep_err)

    def _fill(self) -> Optional[IouResult]:
        if self.fill_cavities and not self._filled:
            self._filled = True
            prep, self._fill_err = _fill_cavities(self._prep, self.side, self.exact_booleans)
            self._prep = prep if self._fill_err is None else None
        return None if self._fill_err is None else self._out(self._fill_err)

    def _solid(self) -> _Prepared:
        p = self._prep
        return replace(p, info=copy.deepcopy(p.info)) if self._shared else p

    def _out(self, err: IouResult) -> IouResult:
        return copy.deepcopy(err) if self._shared else err

    def _expect(self, side: str, check: bool, mode: str, fill: bool, exact: bool) -> None:
        got = (self.side, self.check_self_intersection, self.on_self_intersection,
               self.fill_cavities, self.exact_booleans)
        if got != (side, check, mode, fill, exact):
            raise ValueError(
                f"{side}: prepared side has (side, check_self_intersection, "
                f"on_self_intersection, fill_cavities, exact_booleans) = {got}, this call "
                f"needs {(side, check, mode, fill, exact)}")


def _load_side(obj, label: str):
    """(raw mesh, Body or None, error or None) of a mesh, Body or body path."""
    from . import body as B
    if B.is_body_path(obj):
        try:
            obj = B.load_body(obj)
        except (OSError, ValueError) as exc:
            return None, None, IouResult(Status.LOAD_FAILED, detail=f"{label}: {exc}",
                                         blame=label)
    if isinstance(obj, B.Body):
        return obj.mesh, obj, None
    raw, err = _load_and_check(obj, label)
    return raw, None, err


def prepare_side(obj, side: str, *, check_self_intersection: bool = True,
                 on_self_intersection: Optional[str] = None,
                 fill_cavities: bool = False, exact_booleans: bool = False) -> PreparedSide:
    """
    Load and build one side of the pair once, for many `volume_iou` calls --
    e.g. one GT scored against many predictions.  `obj` is what `volume_iou`
    takes for that side: a mesh, a path, a Body or a body path.

    side
        "gt" or "pred": the side the result is passed as.
    check_self_intersection, on_self_intersection, fill_cavities, exact_booleans
        This side's options, as in `volume_iou`; on_self_intersection defaults
        to that side's default ("exact" for the gt, "fail" for the pred).  A
        call with other options for this side raises ValueError.

    Never raises on bad geometry: a side that would be refused keeps its
    refusal (`PreparedSide.error`), and every call with it returns that
    refusal, as it would with the mesh.  Keeping it between calls is the
    caller's choice: the package caches nothing.
    """
    if side not in SIDES:
        raise ValueError(f"side must be 'gt' or 'pred'; got {side!r}")
    mode = (DEFAULT_ON_SELF_INTERSECTION[SIDES.index(side)]
            if on_self_intersection is None else on_self_intersection)
    if mode not in SELF_INTERSECTION_MODES:
        raise ValueError(f"on_self_intersection must be 'exact' or 'fail'; got {mode!r}")
    if not isinstance(fill_cavities, bool):
        raise ValueError(f"fill_cavities must be a bool; got {fill_cavities!r}")
    if not isinstance(exact_booleans, bool):
        raise ValueError(f"exact_booleans must be a bool; got {exact_booleans!r}")
    s = PreparedSide(obj, side, check_self_intersection, mode, fill_cavities, exact_booleans)
    s._prepare()
    s._shared = True
    return s


def volume_iou(gt, pred, *, check_self_intersection: bool = True,
               on_self_intersection: Union[str, tuple[str, str]] = DEFAULT_ON_SELF_INTERSECTION,
               fill_cavities: Union[bool, tuple[bool, bool]] = False,
               exact_booleans: bool = False,
               timeout: Optional[float] = None,
               isolation: str = "server") -> IouResult:
    """
    Volume IoU of the solids of `gt` and `pred`, or a status saying why not.

    The solid of a mesh is the set of points its surface winds around at least
    once (docs/DEFINITION.md): overlapping bodies count once, cavities are
    empty.  IoU = V(G and P) / V(G or P), iog = V(G and P) / V(G),
    iop = V(G and P) / V(P), from a float64 boolean in manifold3d, or
    in CGAL (method "cgal") where a side's solid stays there or manifold fails
    the intersection.

    The pair is taken as given: the gt must lie in the unit cube
    [-0.5, 0.5]^3 (NOT_NORMALIZED otherwise), the pred in the same frame
    (a warning if it leaves the cube by more than PRED_OUT_WARN); see
    `normalize_pair`.

    check_self_intersection
        Run the crossing detector.  Crossings between bodies are resolved by
        the union; crossings inside one body are handled per the next option.
    on_self_intersection
        A (gt, pred) pair of modes, or one mode for both sides; default
        ("exact", "fail") -- a GT is repaired, a prediction must be a valid
        solid.  "exact" -- the w >= 1 solid of a body whose surface passes
        through itself, or of a body the detector failed on, is built by
        libigl/CGAL mesh arrangement in exact arithmetic
        (`diagnostics[...]["resolved"]`, method "cgal:manifold64", or "cgal"
        if manifold does not take that solid); "fail" -- a body with
        intersecting or touching triangle pairs is built the same way, and
        refused (SELF_INTERSECTING, blamed on the side) only where it winds
        2 or more around a volume (passes through itself; a touch inside a
        body is a number; w <= -1 is INVERTED_VOLUME, as in "exact"); a body
        the detector failed on is refused (BOOLEAN_FAILED, blamed on the
        engine).  docs/DEFINITION.md 4.7.
    fill_cavities
        A (gt, pred) pair of flags, or one flag for both sides; default False.
        True -- closed cavities of that side's solid are filled (the solid is
        the union of its outward shells; `diagnostics[...]["cavities_filled"]`,
        `["volume_cavities"]`): for inputs that show a model only the outside
        of the part.  docs/DEFINITION.md 5.5.  Filling also decides two
        readings the surface leaves open, which are refused without it: an
        enclosed cell of shells written twice that may be a cavity (without
        filling taken as solid while such cells hold at most 1e-6 of the
        side's solid, `["doubled_shells"]["ambiguous_accepted"]`), and a
        one-sided wall inside a closed surface -- every enclosed cell is solid.
    exact_booleans
        False (default) -- bodies without face pairs, their union and the
        pair intersection are manifold3d's (float64; no proven error bound).
        True -- every boolean is CGAL's in exact arithmetic: each body, the
        union of a side's bodies, a filling, and the intersection (method
        "cgal"); the volumes then carry only the rounding of the output to
        float64 and the summation, with a proven bound (docs/DEFINITION.md
        7).  Slower; a body file must be built with it too (`to_body`),
        LOAD_FAILED otherwise.
    timeout
        Seconds; None runs in-process.  Otherwise the computation runs in a
        child process, which is killed on expiry (status TIMEOUT; CRASHED if
        it dies or raises).  The child is forked from a server process that
        only imports the package (`isolate`; started at the first such call
        in a process), so the inputs travel to it as pickles; a PreparedSide
        is built again there from its source.
    isolation
        How the child of `timeout` is made.  "server" (default) -- forked
        from the server above: safe in any caller.  "fork" -- forked from the
        caller itself: no server and no pickling of the inputs, but only for
        a caller that has computed no geometry
        in-process and built no PreparedSide (a library thread pool, such as
        oneTBB of the PyPI manifold3d, copied by the fork may hang the child:
        a false TIMEOUT).  Nothing checks this.

    Volumes in `diagnostics` are in the caller's (unit-cube) units.

    Either side may be a Body (`to_body`) or a path to a `*.body.ply` file:
    its solid is taken as built, after cheap checks, with no repair and no
    detector run; its diagnostics and warnings are those of the source mesh.
    In "fail" mode a body whose source had crossings inside a body (unless it
    was built in "fail", which found no pass through itself), or was not
    checked, or was built from the raw surface (`raw_solid`; unless built in
    "fail", which found it winding below 2, or 2 or more only where different
    parts overlap),
    is refused.

    Either side may also be a PreparedSide (`prepare_side`): loaded and built
    once, with this call's options for that side (ValueError otherwise); the
    result is the one its source would give.
    """
    mode_g, mode_p = _side_modes(on_self_intersection)
    fill_g, fill_p = _side_fill(fill_cavities)
    # options of a prepared side are checked here, so a mismatch raises
    # ValueError also with a timeout (not a crash of the isolated child)
    if not isinstance(exact_booleans, bool):
        raise ValueError(f"exact_booleans must be a bool; got {exact_booleans!r}")
    for obj, label, mode, fill in ((gt, "gt", mode_g, fill_g), (pred, "pred", mode_p, fill_p)):
        if isinstance(obj, PreparedSide):
            obj._expect(label, check_self_intersection, mode, fill, exact_booleans)
    if isolation not in ("server", "fork"):
        raise ValueError(f"isolation must be 'server' or 'fork'; got {isolation!r}")
    if timeout is not None:
        from .isolate import run_forked, run_isolated
        out = (run_isolated if isolation == "server" else run_forked)(volume_iou, (gt, pred),
                           dict(check_self_intersection=check_self_intersection,
                                on_self_intersection=on_self_intersection,
                                fill_cavities=fill_cavities, exact_booleans=exact_booleans),
                           timeout)
        if out["status"] == "ok":
            return out["value"]
        return IouResult(Status.TIMEOUT if out["status"] == "timeout" else Status.CRASHED,
                         detail=f"isolated run: {out}", blame="engine")

    sides: list[PreparedSide] = []
    for obj, label, mode, fill in ((gt, "gt", mode_g, fill_g), (pred, "pred", mode_p, fill_p)):
        if not isinstance(obj, PreparedSide):
            obj = PreparedSide(obj, label, check_self_intersection, mode, fill, exact_booleans)
        if obj._load_err is not None:
            return obj._out(obj._load_err)
        sides.append(obj)
    side_g, side_p = sides
    body_g, body_p = side_g.body, side_p.body
    if body_g is not None and body_p is not None and body_p.frame["kind"] == "gt" and (
            body_p.frame["centre"] != body_g.frame["centre"]
            or body_p.frame["scale"] != body_g.frame["scale"]):
        return IouResult(Status.NOT_NORMALIZED,
                         detail="pred body was built in the frame of another GT")

    err, warnings = _check_normalized(side_g.raw, side_p.raw)
    if err is not None:
        return err

    # both sides built before either is filled: the order of refusals
    for step in (PreparedSide._build, PreparedSide._fill):
        for s in sides:
            err = step(s)
            if err is not None:
                err.warnings = warnings + err.warnings
                return err
    prep_g, prep_p = side_g._solid(), side_p._solid()

    diagnostics = {"volume_gt": prep_g.volume, "volume_pred": prep_p.volume,
                   "gt": prep_g.info, "pred": prep_p.info}
    if exact_booleans:
        diagnostics["exact_booleans"] = True
    for prep, label in ((prep_g, "gt"), (prep_p, "pred")):
        i = prep.info
        if i.get("loop_faces_added"):
            warnings.append(
                f"{label}: {i['loop_faces_added']} triangles added to close cracks, "
                f"covering {i['patched_area_frac'] * 100:.2f}% of its surface; the IoU "
                "includes that invented geometry")
        pc = i.get("holes")
        if pc and pc.get("accepted") and pc["loops"]:
            warnings.append(
                f"{label}: {pc['loops']} holes patched: proven volume error at most {pc['bound']:.3g}, "
                f"{pc['worst_ratio']:.2g} of the parts they close")
        d = i.get("doubled_shells") or {}
        if d.get("loop_faces_added"):
            # patched in the copy before the side is prepared again, so not counted above;
            # a body file written before the share was recorded has none
            share = (f", covering {d['patched_area_frac'] * 100:.2f}% of their surface"
                     if "patched_area_frac" in d else "")
            warnings.append(
                f"{label}: {d['loop_faces_added']} triangles added to close cracks in "
                f"shells written twice{share}; the IoU includes that invented geometry")
        dh = d.get("holes") or []
        if dh:
            # each copy patched by itself: their errors add up
            warnings.append(
                f"{label}: {sum(h['loops'] for h in dh)} holes patched in shells written twice: "
                f"proven volume error at most {sum(h['bound'] for h in dh):.3g}, "
                f"{max(h['worst_ratio'] for h in dh):.2g} of the parts they close")
        if i.get("shells_dropped"):
            warnings.append(f"{label}: {i['shells_dropped']} zero-volume shells dropped")
        if i.get("flipped_inverted_mesh"):
            warnings.append(f"{label}: mesh was inside out; flipped")
        if i.get("flipped_shells"):
            warnings.append(f"{label}: {i['flipped_shells']} inside-out bodies flipped")
        if i.get("orientation") == "parity":
            warnings.append(f"{label}: inconsistent winding; cavities decided by nesting parity")
        if i.get("ambiguous_containment"):
            warnings.append(f"{label}: {i['ambiguous_containment']} touching shells, "
                            "containment ambiguous")
        if i.get("cavities_filled"):
            warnings.append(f"{label}: {i['cavities_filled']} cavities filled, volume "
                            f"{i['volume_cavities']:.6g}")
        if i.get("inverted_cavity_faces"):
            warnings.append(f"{label}: the surface winds -1 or less around a volume of "
                            f"{i['inverted_cavity_volume']:.3g} (an inverted pocket) lying in a "
                            "cavity of the solid; filled with it (fill_cavities)")
        if i.get("union"):
            warnings.append(f"{label}: {i['solids']} bodies with overlapping extents merged "
                            "(union)")
        elif i.get("solids", 1) > 1:
            warnings.append(f"{label}: {i['solids']} separate bodies")
        if i.get("self_intersection_checked") is None:
            warnings.append(f"{label}: self-intersection check failed ("
                            f"{i.get('detector_error', '')}); resolved by CGAL")
        if i.get("raw_solid"):
            if i["raw_solid"].get("audited"):
                warnings.append(f"{label}: raw-surface readings checked by CGAL")
            else:
                warnings.append(f"{label}: {i['raw_solid']['refused_as']} after cleaning; "
                                "solid built from the raw surface by CGAL")
        if i.get("cells_filled"):
            warnings.append(f"{label}: edges of 3+ faces with an unbalanced raw surface (a wall "
                            "inside a body or the boundary of a void); every enclosed cell taken "
                            "as solid (fill_cavities), built by CGAL")
        if d.get("ambiguous_accepted"):         # without filling: within the threshold
            warnings.append(f"{label}: {d['ambiguous_cells']} cells enclosed by shells written "
                            f"twice may be cavities; taken as solid: their volume "
                            f"{d['ambiguous_volume']:.3g} is within {AMBIGUOUS_VOLUME_FRAC:g} "
                            "of the solid")
        elif d.get("ambiguous_cells"):          # taken only where the side is filled
            warnings.append(f"{label}: {d['ambiguous_cells']} cells enclosed by shells written "
                            "twice may be cavities; taken as solid (fill_cavities)")
        if i.get("resolved") and not (i.get("exact_booleans")
                                      and not i["resolved"]["crossing_pairs"]):
            # with exact_booleans every body is CGAL's: said only for crossings
            n = i["resolved"]["crossing_pairs"]
            # "fail" stops each body at its first pair: the count is of bodies
            what = "bodies with crossing or touching faces" if "self_overlap_bodies" in i["resolved"] \
                else "crossing pairs"
            # n is None also with the detector off: a carved body written twice (4.8)
            off = i.get("self_intersection_checked") is False
            warnings.append(f"{label}: surface resolved by CGAL arrangement ("
                            + (f"{n} {what}" if n is not None
                               else "detector off" if off else "detector failed")
                            + ")")

    ratio = max(prep_g.mesh.scale, prep_p.mesh.scale) / max(
        min(prep_g.mesh.scale, prep_p.mesh.scale), 1e-30)
    if ratio > 1e3:
        warnings.append(f"bounding boxes differ by {ratio:.3g}x -- check units/alignment")

    def over_operand(v_inter: float) -> bool:
        # both sides are clean solids here, so an intersection larger than an
        # operand beyond rounding is the engine's
        return v_inter > min(prep_g.volume, prep_p.volume) * (1.0 + INTER_TOL)

    def ok(v_inter: float, method: str, detail: str = "") -> IouResult:
        vg, vp = prep_g.volume, prep_p.volume
        # I <= min(V) by definition; an excess within INTER_TOL is rounding,
        # clipped to min(V) with the engine's volume kept as volume_inter_raw
        if v_inter > min(vg, vp):
            diagnostics["volume_inter_raw"] = v_inter
            v_inter = min(vg, vp)
        diagnostics["volume_inter"] = v_inter
        if v_inter == 0.0 and method in ("bounds", "cgal"):
            diagnostics["zero_proof"] = method      # disjoint_surfaces: set by its caller
        iou, iog, iop = v_inter / (vg + vp - v_inter), v_inter / vg, v_inter / vp
        if not all(0.0 <= x <= 1.0 for x in (iou, iog, iop)):
            return IouResult(Status.BOOLEAN_FAILED,
                             detail=f"{method}: iou/iog/iop = {iou!r}/{iog!r}/{iop!r} out of "
                                    f"[0, 1] (volumes gt {vg!r}, pred {vp!r}, "
                                    f"intersection {v_inter!r})",
                             diagnostics=diagnostics, warnings=warnings, blame="engine")
        return IouResult(Status.OK, iou, method, detail,
                         diagnostics, warnings, iog=iog, iop=iop)

    if (prep_g.mesh.bounds[0] > prep_p.mesh.bounds[1]).any() or (
            prep_p.mesh.bounds[0] > prep_g.mesh.bounds[1]).any():
        return ok(0.0, "bounds", "disjoint bounding boxes")

    def cgal_inter(why: str = "") -> IouResult:
        # each side goes in as its solid: the manifold one (body overlaps
        # already united) or the CGAL-resolved w >= 1 one -- not raw shells,
        # which would make CGAL resolve the same self-crossings a second time
        try:
            inter_mesh, v = _cgal_intersection(prep_g.mesh, prep_p.mesh)
        except Exception as exc:
            return IouResult(Status.BOOLEAN_FAILED,
                             detail=f"{why}CGAL intersect raised {type(exc).__name__}: {exc}",
                             diagnostics=diagnostics, warnings=warnings, blame="engine")
        # probes of both sides: in both raw solids <=> in the result
        used, lost = _lost_probes(np.concatenate([prep_g.probe_points(), prep_p.probe_points()]),
                                  [prep_g.shells, prep_p.shells], inter_mesh)
        diagnostics["intersection_probes"] = used
        if lost:
            return IouResult(Status.BOOLEAN_FAILED,
                             detail=f"{why}CGAL intersection put {lost} of {used} probe points "
                                    "on the wrong side",
                             diagnostics=diagnostics, warnings=warnings, blame="engine")
        if over_operand(v):
            return IouResult(Status.BOOLEAN_FAILED,
                             detail=f"{why}CGAL intersection volume exceeds an operand; the "
                                    "boolean result is invalid",
                             diagnostics={**diagnostics, "volume_inter": v},
                             warnings=warnings, blame="engine")
        if why:
            warnings.append(f"{why}intersection recomputed by CGAL")
        return ok(max(v, 0.0), "cgal")

    if prep_g.solid is None or prep_p.solid is None:
        return cgal_inter()
    # CGAL recomputes a failed manifold intersection unless neither side is "exact"
    exact = "exact" in (mode_g, mode_p)
    # a side resolved by CGAL, intersected in manifold
    manifold_method = ("cgal:manifold64" if any(p.info.get(k) for p in (prep_g, prep_p)
                                                for k in ("resolved", "raw_solid",
                                                          "cells_filled"))
                       else "boolean:manifold64")

    try:
        inter = _intersect(prep_g.solid, prep_p.solid)
        v_inter = float(inter.volume())
    except Exception as exc:
        if exact:
            return cgal_inter(f"manifold raised {type(exc).__name__}; ")
        return IouResult(Status.BOOLEAN_FAILED,
                         detail=f"manifold raised {type(exc).__name__}: {exc}",
                         diagnostics=diagnostics, warnings=warnings, blame="engine")

    if inter.is_empty() or v_inter <= 0:
        # an empty result claims IoU exactly 0: kept only if proven exactly
        # (surfaces apart, no shell inside the other solid); else CGAL decides
        why = _disjoint_proof(prep_g.mesh, prep_p.mesh)
        if why is None:
            diagnostics["zero_proof"] = "disjoint_surfaces"
            return ok(0.0, manifold_method)
        if exact:
            return cgal_inter(f"manifold returned an empty intersection, but {why}; ")
        return IouResult(Status.BOOLEAN_FAILED,
                         detail=f"engine returned an empty intersection, but {why}",
                         diagnostics=diagnostics, warnings=warnings, blame="engine")

    if over_operand(v_inter):
        if exact:
            return cgal_inter("manifold intersection exceeded an operand; ")
        return IouResult(Status.BOOLEAN_FAILED,
                         detail="intersection volume exceeds an operand; the boolean "
                                "result is invalid",
                         diagnostics={**diagnostics, "volume_inter": v_inter},
                         warnings=warnings, blame="engine")
    return ok(v_inter, manifold_method)
