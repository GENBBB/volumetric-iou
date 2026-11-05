"""
Self-intersection detection for triangle meshes -- CGAL, exact.

`volumetric_iou._cgal.self_intersecting_pairs`, the detect-only path of
libigl's `remesh_self_intersections` with the same kernel and per-pair tests:
CGAL's box intersection finds the candidate pairs, exact predicates decide
them.  A pair is reported when two triangles meet in more than what they share
by construction (a common vertex or edge): a crossing, and also a contact such as
two coplanar faces lying on each other.  The metric reads any reported pair as
"not proven embedded" and sends the body to the exact CGAL arrangement, so a
contact costs time, never a wrong number (docs/DEFINITION.md 4.7).

There is no tolerance.  The numpy detector this replaces (up to v0.2.0) had
one, relative to the mesh scale, and it raised false alarms on pairs that are
exactly disjoint, skipped every pair sharing a vertex although such pairs do
cross, and gave up on sides of 270-460 thousand faces, which CGAL checks in a
few seconds.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

from . import _cgal


def detector_threads() -> int:
    """Positive IGL_NUM_THREADS (also used by libigl), or one thread."""
    n = os.environ.get("IGL_NUM_THREADS", "")
    return int(n) if n.isdigit() and int(n) > 0 else 1


def self_intersecting_pairs(mesh: trimesh.Trimesh, threads: int | None = None,
                            first: bool = False) -> np.ndarray:
    """
    Pairs of faces (i < j, sorted) that intersect beyond their shared vertices
    or edge.  An empty array means the surface is embedded.  CGAL errors are
    raised, not hidden: the caller decides what an unchecked mesh means.
    first: stop at the first pair and return it alone -- for a caller that
    only needs to know whether the surface is embedded.
    threads: explicit per-call budget; None uses IGL_NUM_THREADS, defaulting
    to one. Unlike libigl's budget, the detector reads the environment per call.
    """
    if len(mesh.faces) < 2:
        return np.empty((0, 2), dtype=np.int64)
    return _cgal.self_intersecting_pairs(
        np.ascontiguousarray(mesh.vertices, np.float64),
        np.ascontiguousarray(mesh.faces, np.int64),
        threads or detector_threads(), first)


def is_self_intersecting(mesh: trimesh.Trimesh) -> tuple[bool, int]:
    """(any pair reported, number of pairs)."""
    hits = self_intersecting_pairs(mesh)
    return len(hits) > 0, int(len(hits))
