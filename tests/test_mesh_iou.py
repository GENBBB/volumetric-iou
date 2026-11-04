"""
Tests for mesh_iou.volume_iou.

Grouped as:
  1. cases with an exact analytic answer
  2. invariances the metric must satisfy
  3. dirty-but-recoverable inputs
  4. inputs that must fail, with a specific status
"""

import numpy as np
import pytest
import trimesh

from volumetric_iou import _cgal, mesh_iou
from volumetric_iou.body import Body, to_body
from volumetric_iou.inside import contains, contains_resolved
from volumetric_iou.mesh_iou import Cause, Status, diagnose_mesh, normalize_pair, volume_iou
from volumetric_iou.self_intersect import is_self_intersecting, self_intersecting_pairs
from volumetric_iou.winding import arbiter_iou

from shapes import coiled_bar, extruded, klein_bottle, via_stl

try:            # libigl: only the reference the package module is checked against
    import igl  # noqa: F401
    _HAS_IGL = True
except ImportError:
    _HAS_IGL = False
needs_igl = pytest.mark.skipif(not _HAS_IGL, reason="libigl (reference) not installed")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


# manifold3d is exact in its predicates but not to the last bit of a double:
# geometric results agree to ~1e-7 relative, so that is the tolerance used for
# any comparison against an analytic value.
TOL = 1e-6


def box(extents=(1, 1, 1), translate=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    return m


def sphere(radius=1.0, translate=(0, 0, 0), subdivisions=4):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.apply_translation(translate)
    return m


def iou_res(a, b, **kw):
    """Same, but returns the IouResult without asserting success."""
    la, lb = _load(a), _load(b)
    if la is not None and lb is not None:
        a, b = normalize_pair(la, lb, pred_frame="gt")
    return volume_iou(a, b, **kw)


def _load(x):
    """Best-effort load so the harness can normalise paths and raw arrays too."""
    if isinstance(x, trimesh.Trimesh):
        return x
    try:
        m = trimesh.load(x, force="mesh", process=False)
        return m if isinstance(m, trimesh.Trimesh) and len(m.faces) else None
    except Exception:
        return None


def iou_of(a, b, **kw):
    """Normalise the pair (the module's precondition), assert success, return the number."""
    res = iou_res(a, b, **kw)
    assert res, f"expected success, got {res.status.value}: {res.detail}"
    return res.iou


def sphere_sphere_iou(r, d):
    """Analytic IoU of two equal spheres of radius r whose centres are d apart."""
    if d >= 2 * r:
        return 0.0
    v_sphere = 4 / 3 * np.pi * r**3
    v_lens = np.pi * (4 * r + d) * (2 * r - d) ** 2 / 12
    return v_lens / (2 * v_sphere - v_lens)


# --------------------------------------------------------------------------- #
# 1. analytic ground truth
# --------------------------------------------------------------------------- #


def test_identical_cubes():
    assert iou_of(box(), box()) == pytest.approx(1.0, rel=TOL)


def test_half_overlapping_cubes():
    # overlap 0.5, union 1.5
    assert iou_of(box(), box(translate=(0.5, 0, 0))) == pytest.approx(1 / 3, rel=TOL)


@pytest.mark.parametrize("shift", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_cube_shift_sweep(shift):
    v_inter = 1 - shift
    expected = v_inter / (2 - v_inter)
    assert iou_of(box(), box(translate=(shift, 0, 0))) == pytest.approx(expected, rel=TOL)


def test_nested_cubes():
    # small cube fully inside the big one -> IoU = V_small / V_big
    assert iou_of(box((2, 2, 2)), box((1, 1, 1))) == pytest.approx(1 / 8, rel=TOL)


def test_corner_overlap_3d():
    # offset along all three axes at once
    v_inter = 0.5**3
    expected = v_inter / (2 - v_inter)
    assert iou_of(box(), box(translate=(0.5, 0.5, 0.5))) == pytest.approx(expected, rel=TOL)


def test_disjoint_returns_zero():
    res = iou_res(box(), box(translate=(5, 0, 0)))
    assert res and res.iou == 0.0
    assert res.method == "bounds"


def test_face_touching_is_zero():
    # coplanar contact: the classic boolean edge case, must be 0 and not a crash
    res = iou_res(box(), box(translate=(1.0, 0, 0)))
    assert res and res.iou == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("d", [0.0, 0.5, 1.0, 1.5])
def test_spheres_against_analytic(d):
    # tessellated icospheres, so allow a small discretisation error
    got = iou_of(sphere(), sphere(translate=(d, 0, 0)))
    assert got == pytest.approx(sphere_sphere_iou(1.0, d), rel=0.02)


# --------------------------------------------------------------------------- #
# 2. invariances
# --------------------------------------------------------------------------- #


def test_symmetric():
    a, b = box(), box(translate=(0.3, 0.2, 0))
    assert iou_of(a, b) == pytest.approx(iou_of(b, a), rel=TOL)


def test_translation_invariant():
    base = iou_of(box(), box(translate=(0.5, 0, 0)))
    shifted = iou_of(box(translate=(100, -7, 3)), box(translate=(100.5, -7, 3)))
    assert shifted == pytest.approx(base, rel=TOL)


def test_scale_invariant():
    base = iou_of(box(), box(translate=(0.5, 0, 0)))
    k = 1000.0
    big = iou_of(box((k, k, k)), box((k, k, k), translate=(0.5 * k, 0, 0)))
    assert big == pytest.approx(base, rel=TOL)


def test_rotation_invariant():
    a, b = box(), box(translate=(0.5, 0, 0))
    R = trimesh.transformations.random_rotation_matrix()
    a2, b2 = a.copy(), b.copy()
    a2.apply_transform(R)
    b2.apply_transform(R)
    assert iou_of(a2, b2) == pytest.approx(iou_of(a, b), rel=TOL)


@pytest.mark.parametrize("shift", [0.0, 0.4, 0.8, 1.2, 5.0])
def test_always_in_unit_range(shift):
    assert 0.0 <= iou_of(box(), box(translate=(shift, 0, 0))) <= 1.0


def test_monotonic_in_separation():
    vals = [iou_of(box(), box(translate=(s, 0, 0))) for s in np.linspace(0, 1, 6)]
    assert all(x >= y - TOL for x, y in zip(vals, vals[1:]))


# --------------------------------------------------------------------------- #
# 3. dirty but recoverable input
# --------------------------------------------------------------------------- #


def test_unmerged_duplicate_vertices():
    # every triangle carries its own copy of its vertices (typical of STL)
    m = box()
    soup = trimesh.Trimesh(m.triangles.reshape(-1, 3),
                           np.arange(len(m.faces) * 3).reshape(-1, 3),
                           process=False)
    assert iou_of(soup, box()) == pytest.approx(1.0, rel=TOL)


def test_inverted_winding_is_repaired():
    flipped = box()
    flipped.faces = np.fliplr(flipped.faces)  # volume would come out negative
    assert iou_of(flipped, box(translate=(0.5, 0, 0))) == pytest.approx(1 / 3, rel=TOL)


def fine_box():
    """The unit box subdivided 5 times: one triangle is 1/2048 of a face."""
    m = box()
    for _ in range(5):
        m = m.subdivide()
    return m


def test_hole_is_filled():
    # one triangle of a finely tessellated cube: a crack whose patch is a
    # proven guess of under 1e-3 of the cube (1/2048 of a face)
    holed = fine_box()
    holed.update_faces(np.arange(len(holed.faces)) != 0)  # drop one triangle
    holed.remove_unreferenced_vertices()
    res = iou_res(holed, box())
    assert res and res.iou == pytest.approx(1.0, rel=TOL)
    assert any("to close cracks" in w for w in res.warnings)


def test_large_hole_is_not_filled():
    """
    One triangle of a 12-face cube is 8% of its surface: over the patch budget,
    so the mesh is open -- the same rule as any other hole (docs/DEFINITION.md).
    trimesh's fill_holes used to close it silently.
    """
    holed = box()
    holed.update_faces(np.arange(len(holed.faces)) != 0)
    holed.remove_unreferenced_vertices()
    assert iou_res(holed, box()).status is Status.OPEN_SURFACE


def test_missing_side_is_not_invented():
    """A box missing two sides used to be closed by fill_holes -> IoU 0.545."""
    lid = box(translate=(3, 0, 0))
    lid.update_faces(np.arange(12) >= 4)
    res = iou_res(trimesh.util.concatenate([box(), lid]), box())
    assert res.status is Status.OPEN_SURFACE


def test_duplicate_faces_removed():
    m = box()
    doubled = trimesh.Trimesh(m.vertices, np.vstack([m.faces, m.faces]), process=False)
    res = iou_res(doubled, box())
    assert res and res.iou == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["duplicate_removed"] == 12
    assert res.diagnostics["gt"]["raw_solid"]["audited"]
    assert any("raw-surface readings checked" in warning for warning in res.warnings)


@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("first", [False, True])
def test_reversed_duplicate_face_removed(first, mode):
    """
    One face written again the other way (not a shell written twice: the
    other faces are written once): one copy is kept whichever way it runs.
    Kept against its neighbours (`first`), the winding is agreed before the
    shells are read -- the box either way, by nesting parity then.
    """
    m = box()
    extra = m.faces[[0], ::-1]
    faces = np.vstack([extra, m.faces] if first else [m.faces, extra])
    side = trimesh.Trimesh(m.vertices, faces, process=False)
    res = iou_res(side, box(), on_self_intersection=mode)
    assert res and res.iou == pytest.approx(1.0, rel=TOL)
    g = res.diagnostics["gt"]
    assert g["duplicate_removed"] == 1 and g["degenerate_removed"] == 0
    assert "doubled_shells" not in g
    assert g["orientation"] == ("parity" if first else "given")


def test_degenerate_faces_removed():
    """A face with a repeated vertex and a needle 1e-9 high over a box edge:
    both dropped by cleaning, counted, no warning -- the solid is the box."""
    m = box()
    a, b = m.vertices[m.faces[0][0]], m.vertices[m.faces[0][1]]
    v = np.vstack([m.vertices, (a + b) / 2 + [1e-9, 0, 0]])
    needle = [m.faces[0][0], m.faces[0][1], len(m.vertices)]
    side = trimesh.Trimesh(v, np.vstack([m.faces, [[0, 0, 1], needle]]), process=False)
    res = iou_res(side, box())
    assert res and res.iou == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["degenerate_removed"] == 2
    assert res.diagnostics["gt"]["duplicate_removed"] == 0
    assert not res.warnings


def test_accepts_vertices_faces_tuple():
    m = box()
    assert iou_of((m.vertices, m.faces), box()) == pytest.approx(1.0, rel=TOL)


def test_accepts_file_path(tmp_path):
    p = tmp_path / "cube.stl"
    box().export(p)
    assert iou_of(str(p), box(translate=(0.5, 0, 0))) == pytest.approx(1 / 3, rel=TOL)


def test_multibody_warns_and_computes():
    # two separate cubes as one mesh; only the first overlaps the probe
    pair = trimesh.util.concatenate([box(), box(translate=(10, 0, 0))])
    res = iou_res(pair, box(translate=(0.5, 0, 0)))
    assert res
    # V_inter = 0.5, V_a = 2, V_b = 1  ->  0.5 / 2.5
    assert res.iou == pytest.approx(0.2, rel=TOL)
    assert any("separate bodies" in w for w in res.warnings)
    assert res.diagnostics["gt"]["solids"] == 2
    assert res.diagnostics["gt"]["cavities"] == 0
    assert not res.diagnostics["gt"]["union"]


def test_unit_mismatch_warns():
    res = iou_res(box(), box((1e5, 1e5, 1e5)))
    assert res
    assert any("units" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# 4. must fail, with the right status
# --------------------------------------------------------------------------- #


def test_empty_mesh():
    res = volume_iou(trimesh.Trimesh(), box())
    assert not res and res.status is Status.EMPTY_MESH


def test_all_faces_degenerate():
    """Faces on one line and a face with a repeated vertex: the cheap checks
    pass (there are faces), cleaning leaves none."""
    flat = trimesh.Trimesh([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0, 0.1, 0]],
                           [[0, 1, 2], [0, 0, 3]], process=False)
    res = volume_iou(box(), flat)
    assert not res and res.status is Status.EMPTY_MESH and res.blame == "pred"
    assert "all faces degenerate" in res.detail
    assert res.diagnostics["degenerate_removed"] == 2


def _two_holes(touching):
    """A fine box with two triangles of its top missing: sharing one vertex
    (the boundary branches there) or apart (two simple loops)."""
    m = fine_box()
    top = np.flatnonzero(m.triangles_center[:, 2] > 0.499)
    shared = 1 if touching else 0
    i, j = next((i, j) for i in top for j in top
                if i < j and len(set(m.faces[i]) & set(m.faces[j])) == shared)
    keep = np.ones(len(m.faces), bool)
    keep[[i, j]] = False
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
def test_branching_boundary_is_not_patched(mode, fill):
    """
    Two missing triangles meeting at a vertex: four boundary edges there, no
    simple loops to fan, nothing is patched -- though the same two holes
    apart are closed as cracks.  Open edges are unbalanced, so an "exact"
    side has no raw-surface solid either.
    """
    res = iou_res(_two_holes(touching=True), box(), on_self_intersection=mode, fill_cavities=fill)
    assert not res and res.status is Status.OPEN_SURFACE
    assert res.diagnostics["loop_faces_added"] == 0 and res.diagnostics["open_edges"] == 6
    assert "raw_solid" not in res.diagnostics
    apart = iou_res(_two_holes(touching=False), box(), on_self_intersection=mode, fill_cavities=fill)
    assert apart and apart.iou == pytest.approx(1.0, rel=TOL)
    assert apart.diagnostics["gt"]["loop_faces_added"] == 2


def test_nan_vertices():
    bad = box()
    bad.vertices[0] = np.nan
    res = iou_res(bad, box())
    assert not res and res.status is Status.INVALID_COORDS


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("side", ["gt", "pred"])
def test_inf_vertices(side, value):
    # one coordinate of one vertex: refused before anything that depends on
    # the scale (not normalised here: inf has no frame), the side to blame
    bad = box()
    bad.vertices[3, 1] = value
    res = volume_iou(bad, box()) if side == "gt" else volume_iou(box(), bad)
    assert not res and res.status is Status.INVALID_COORDS and res.blame == side
    assert res.detail == f"{side}: NaN/inf in vertices"


def test_unloadable_path():
    res = volume_iou("/definitely/not/here.obj", box())
    assert not res and res.status is Status.LOAD_FAILED


def test_flat_sheet_has_no_volume():
    res = iou_res(trimesh.creation.box(extents=(1, 1, 0)), box())
    assert not res and res.status is Status.DEGENERATE_VOLUME


def test_open_surface_rejected():
    # upper half of a subdivided cube: a large boundary loop that survives filling
    g = trimesh.creation.box(extents=(2, 2, 2)).subdivide().subdivide()
    g.update_faces(g.triangles_center[:, 2] > 0)
    g.remove_unreferenced_vertices()
    res = iou_res(g, box())
    assert not res and res.status is Status.OPEN_SURFACE
    assert res.diagnostics["open_edges"] > 0


def test_failure_carries_no_number():
    res = volume_iou(trimesh.Trimesh(), box())
    assert res.iou is None and res.detail


def test_second_argument_is_validated_too():
    res = volume_iou(box(), trimesh.Trimesh())
    assert not res and res.status is Status.EMPTY_MESH
    assert "pred" in res.detail and res.blame == "pred"


def test_no_engine_reported(monkeypatch):
    from volumetric_iou import mesh_iou

    monkeypatch.setattr(mesh_iou, "_HAS_ENGINE", False)
    res = mesh_iou.volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"))
    assert not res and res.status is Status.NO_ENGINE


def test_oversized_intersection_rejected(monkeypatch):
    """An intersection bigger than both operands is impossible -> reject."""
    from volumetric_iou import mesh_iou

    monkeypatch.setattr(mesh_iou, "_intersect",
                        lambda a, b: mesh_iou._to_manifold([box((5, 5, 5))]))
    res = mesh_iou.volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"), on_self_intersection="fail")
    assert not res and res.status is Status.BOOLEAN_FAILED
    # the default recomputes the intersection exactly instead of refusing
    res = mesh_iou.volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"))
    assert res and res.method == "cgal" and res.iou == pytest.approx(1 / 3, rel=1e-9)


def test_engine_exception_is_caught(monkeypatch):
    from volumetric_iou import mesh_iou

    def boom(a, b):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(mesh_iou, "_intersect", boom)
    res = mesh_iou.volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"), on_self_intersection="fail")
    assert not res and res.status is Status.BOOLEAN_FAILED
    assert "engine exploded" in res.detail
    res = mesh_iou.volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"))
    assert res and res.method == "cgal" and res.iou == pytest.approx(1 / 3, rel=1e-9)
    assert any("recomputed by CGAL" in w for w in res.warnings)


class _FakeInter:
    """A manifold intersection result of a given volume."""

    def __init__(self, v):
        self.v = v

    def volume(self):
        return self.v

    def is_empty(self):
        return False


def test_intersection_over_an_operand_by_rounding_is_capped(monkeypatch):
    """
    I above min(V) within 1e-6 relative is rounding: I = min(V), ratios <= 1;
    the engine's volume is kept as volume_inter_raw, only when clipped.
    """
    from volumetric_iou import mesh_iou

    g, p = normalize_pair(box(), box(), pred_frame="gt")
    first = volume_iou(g, p)
    v = first.diagnostics["volume_gt"]
    assert "volume_inter_raw" not in first.diagnostics
    monkeypatch.setattr(mesh_iou, "_intersect", lambda a, b: _FakeInter(v * (1 + 1e-9)))
    res = volume_iou(g, p)
    assert res and res.method == "boolean:manifold64"
    assert res.iou == res.iog == res.iop == 1.0 and res.diagnostics["volume_inter"] == v
    assert res.diagnostics["volume_inter_raw"] == v * (1 + 1e-9)
    monkeypatch.setattr(mesh_iou, "_intersect", lambda a, b: _FakeInter(v * 0.5))
    res = volume_iou(g, p)
    assert res and "volume_inter_raw" not in res.diagnostics


def test_ratio_out_of_range_is_refused(monkeypatch):
    from volumetric_iou import mesh_iou

    monkeypatch.setattr(mesh_iou, "_intersect", lambda a, b: _FakeInter(float("nan")))
    res = volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"))
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "out of [0, 1]" in res.detail


def test_cgal_intersection_over_an_operand(monkeypatch):
    """The CGAL path takes the same I <= min(V) (1 + 1e-6) check as manifold."""
    from volumetric_iou import mesh_iou

    def boom(a, b):
        raise RuntimeError("engine exploded")

    real = mesh_iou._cgal_intersection
    monkeypatch.setattr(mesh_iou, "_intersect", boom)
    g, p = normalize_pair(box(), box(), pred_frame="gt")
    monkeypatch.setattr(mesh_iou, "_cgal_intersection",
                        lambda a, b: (real(a, b)[0], real(a, b)[1] * 2))
    res = volume_iou(g, p)
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "exceeds an operand" in res.detail
    monkeypatch.setattr(mesh_iou, "_cgal_intersection",
                        lambda a, b: (real(a, b)[0], real(a, b)[1] * (1 + 1e-9)))
    res = volume_iou(g, p)
    assert res and res.method == "cgal" and res.iou == res.iog == res.iop == 1.0
    assert res.diagnostics["volume_inter_raw"] > res.diagnostics["volume_inter"]


# --------------------------------------------------------------------------- #
# an empty intersection is a proven zero or recomputed exactly
# --------------------------------------------------------------------------- #


def _empty_manifold(monkeypatch):
    import manifold3d
    monkeypatch.setattr(mesh_iou, "_intersect", lambda a, b: manifold3d.Manifold())


def _thin_overlap():
    """[0, 1]^3 and a slab crossing its x = 1 face 0.01 deep: overlap 0.005,
    no vertex of either box inside the other."""
    return box(translate=(0.5, 0.5, 0.5)), box((1.01, 2.0, 0.5), translate=(1.495, 0.5, 0.5))


def test_thin_overlap_missed_by_sampling_is_recomputed(monkeypatch):
    """
    An empty manifold result over a thin overlap: no sampled interior point
    of either side lies in the other, the exact surface check finds the
    crossing and CGAL gives the overlap.
    """
    g, p = normalize_pair(*_thin_overlap(), pred_frame="gt")
    ref = volume_iou(g, p)
    assert ref and ref.method == "boolean:manifold64" and ref.iou > 0
    sg, sp = (mesh_iou.PreparedSide(m, s, True, "exact", False) for m, s in ((g, "gt"), (p, "pred")))
    assert sg._build() is None and sp._build() is None
    assert mesh_iou._points_inside(sp._solid().mesh, sg._solid().mesh) == 0
    assert mesh_iou._points_inside(sg._solid().mesh, sp._solid().mesh) == 0
    _empty_manifold(monkeypatch)
    res = volume_iou(g, p)
    assert res and res.method == "cgal" and res.iou == pytest.approx(ref.iou, rel=1e-9)
    assert "zero_proof" not in res.diagnostics
    assert any("surfaces meet" in w and "recomputed by CGAL" in w for w in res.warnings)
    # neither side "exact": no CGAL fallback, and no unproven zero either
    res = volume_iou(g, p, on_self_intersection="fail")
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "surfaces meet" in res.detail


def test_nested_solid_under_an_empty_manifold_result(monkeypatch):
    """Surfaces apart but one solid inside the other: the winding number of a
    shell vertex catches it, CGAL gives 1/8."""
    _empty_manifold(monkeypatch)
    res = iou_res(box((2, 2, 2)), box((1, 1, 1)))
    assert res and res.method == "cgal" and res.iou == pytest.approx(1 / 8, rel=1e-9)
    assert any("inside the other" in w for w in res.warnings)
    res = iou_res(box((2, 2, 2)), box((1, 1, 1)), on_self_intersection="fail")
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"


def test_contact_through_shared_vertices_is_not_proven_zero():
    """
    The detector takes equal positions for a shared vertex: a tetrahedron on
    three corners of a cube, whose faces carry the diagonals between them,
    meets the cube only through shared vertices and edges -- no pair for the
    detector, and the vertex of each shell checked lies outside the other
    solid.  The overlap is 1/6; the proof must leave it to CGAL.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float)
    f = [[0, 2, 1], [1, 2, 3], [0, 1, 4], [1, 5, 4], [0, 4, 2], [2, 4, 6],
         [3, 7, 5], [3, 5, 1], [2, 6, 3], [3, 6, 7], [4, 5, 6], [5, 7, 6]]
    a = trimesh.Trimesh(v, f, process=False)
    a.fix_normals()
    b = trimesh.Trimesh([[1.5, 1.5, 1.5], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
                        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], process=False)
    b.fix_normals()
    assert a.is_volume and b.is_volume
    inter = trimesh.boolean.intersection([a, b], engine="manifold")
    assert inter.volume == pytest.approx(1 / 6, rel=1e-9)
    assert "meet at 3 shared vertex positions" in mesh_iou._disjoint_proof(a, b)
    assert "meet at 3 shared vertex positions" in mesh_iou._disjoint_proof(b, a)


def _l_shape():
    return extruded([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], centre=(0.45, 0.6))


def test_face_contact_with_both_sides_in_fail_is_refused():
    """
    Faces in contact: manifold's empty result is not proven zero (the
    surfaces meet), and with neither side in "exact" there is no CGAL to
    decide it -- a refusal of the engine, not an unproven 0.
    """
    res = iou_res(box(), box(translate=(1.0, 0, 0)), on_self_intersection="fail")
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "empty intersection" in res.detail and "surfaces meet" in res.detail
    res = iou_res(box(), box(translate=(1.0, 0, 0)), on_self_intersection=("fail", "exact"))
    assert res and res.iou == 0.0 and res.diagnostics["zero_proof"] == "cgal"


@pytest.mark.parametrize("pair, method, proof", [
    (lambda: (box(), box(translate=(5, 0, 0))), "bounds", "bounds"),
    # a cube in the notch of an L: extents overlap, surfaces apart
    (lambda: (_l_shape(), box((0.6, 0.6, 0.6), translate=(1.5, 1.5, 0.5))),
     "boolean:manifold64", "disjoint_surfaces"),
    # a cube in the void of a hollow box
    (lambda: (hollow_box(), box((0.5, 0.5, 0.5))), "cgal:manifold64", "disjoint_surfaces"),
    # faces in contact: the surfaces meet, so CGAL decides the zero
    (lambda: (box(), box(translate=(1.0, 0, 0))), "cgal", "cgal"),
])
def test_true_zero_is_proven(pair, method, proof):
    res = iou_res(*pair())
    assert res and res.iou == 0.0 and res.method == method
    assert res.diagnostics["zero_proof"] == proof


# --------------------------------------------------------------------------- #
# 5. regressions: engine precision hazards found by probing manifold3d
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("d", [1e3, 1e6, 1e7, 1e9, 1e12])
def test_far_from_origin_still_correct(d):
    """
    Raw manifold returns an EMPTY intersection past ~1e7 absolute coordinates,
    which would surface as a confident IoU of 0.  normalize_pair() must fix it.
    """
    got = iou_of(box(translate=(d, d, d)), box(translate=(d + 0.5, d, d)))
    assert got == pytest.approx(1 / 3, rel=1e-4)


@pytest.mark.parametrize("d", [9.3e10, 1e11])
def test_far_pred_in_the_metric_frame_is_zero_by_bounds(d):
    """
    Cleaning keyed rounded coordinates by int64: past |x| ~ 9.2e10 they all
    became one key and the far pred's vertices merged into one point -- a
    refusal blamed on pred instead of IoU 0.  No normalisation here.
    """
    gt = box()
    for t in ((d, d, d), (d, 0, 0)):
        res = volume_iou(gt, box(translate=t))
        assert res and res.iou == 0.0 and res.method == "bounds"
    # a pred covering the gt with corners past that range: a number, not a refusal
    huge = box((2 * d, 2 * d, 2 * d))
    res = volume_iou(gt, huge)
    assert res and res.iou == pytest.approx(1 / (2 * d) ** 3, rel=1e-6)


@pytest.mark.parametrize("k", [1e-6, 1e-3, 1e3, 1e6, 1e9])
def test_extreme_absolute_scale(k):
    """The same pair at wildly different absolute sizes must give one answer."""
    got = iou_of(box((k, k, k)), box((k, k, k), translate=(0.5 * k, 0, 0)))
    assert got == pytest.approx(1 / 3, rel=1e-4)


def test_dense_noisy_intersection_not_falsely_torn():
    """
    A dense boolean result must not be reported as BOOLEAN_FAILED just because
    a cleanup pass would collapse its close-together vertices.
    """
    rng = np.random.default_rng(0)
    a = sphere(subdivisions=5)
    b = sphere(subdivisions=5, translate=(0.3, 0, 0))
    b.vertices += rng.normal(0, 1e-7, b.vertices.shape)
    res = iou_res(a, b)
    assert res, f"falsely rejected: {res.status.value}"
    assert res.iou == pytest.approx(sphere_sphere_iou(1.0, 0.3), rel=0.02)


def test_overlapping_bodies_are_merged():
    """
    Two overlapping cubes as ONE mesh: the divergence volume counts the overlap
    twice (2.0 instead of 1.5).  By definition the solid is w >= 1, so the
    bodies are merged and the IoU against the first cube is 1 / 1.5.
    """
    si = trimesh.util.concatenate([box(), box(translate=(0.5, 0, 0))])
    res = iou_res(si, box())
    assert res and res.iou == pytest.approx(1 / 1.5, rel=TOL)
    assert res.iog == pytest.approx(1 / 1.5, rel=TOL) and res.iop == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["union"]
    assert res.diagnostics["gt"]["crossing_pairs_internal"] == 0
    assert res.diagnostics["gt"]["self_intersection_checked"] is True
    d = diagnose_mesh(si)
    assert d["self_intersections"] > 0 and d["crossing_pairs_internal"] == 0


def test_union_is_warned_on_the_manifold_path():
    """
    Bodies whose extents overlap are united in manifold and the warning
    counts every body of the side; the choice is by extents, so a cube in
    the notch of an L (extents overlap, bodies apart) is merged too.
    """
    side = trimesh.util.concatenate([box(), box(translate=(0.5, 0, 0)),
                                     box(translate=(5, 0, 0))])
    res = iou_res(side, box())
    assert res and res.method == "boolean:manifold64"
    assert res.iou == pytest.approx(1 / 2.5, rel=TOL)
    g = res.diagnostics["gt"]
    assert g["union"] and g["solids"] == 3 and "resolved" not in g
    assert "gt: 3 bodies with overlapping extents merged (union)" in res.warnings
    assert not any("separate bodies" in w for w in res.warnings)
    notch = trimesh.util.concatenate([_l_shape(), box((0.6, 0.6, 0.6), translate=(1.5, 1.5, 0.5))])
    res = iou_res(notch, _l_shape())
    assert res and res.method == "boolean:manifold64"
    assert "gt: 2 bodies with overlapping extents merged (union)" in res.warnings


def test_self_overlapping_shell_is_caught():
    """
    One closed, consistently wound shell that runs through itself: every cheap
    check passes, the union cannot help, so under "fail" it is refused -- it
    winds twice around the overlap of its ends.
    """
    res = iou_res(coiled_bar(), box((2.4, 2.4, 0.6)), on_self_intersection="fail")
    assert not res and res.status is Status.SELF_INTERSECTING
    assert res.blame == "gt" and "winds 2 or more" in res.detail
    assert res.diagnostics["self_overlap_bodies"] == 1


def test_non_manifold_edge_contact_still_works():
    """Two cubes meeting at a single edge: an edge with 4 faces, yet computable."""
    nm = trimesh.util.concatenate([box(), box(translate=(1, 1, 0))])
    assert iou_of(nm, box(translate=(0.5, 0.5, 0))) == pytest.approx(0.2, rel=TOL)


def test_diagnostics_are_self_consistent():
    """Volumes are in the normalised frame; the reported IoU must follow from them."""
    res = iou_res(box((10, 10, 10)), box((10, 10, 10), translate=(5, 0, 0)))
    assert res
    va, vb = res.diagnostics["volume_gt"], res.diagnostics["volume_pred"]
    vi = res.diagnostics["volume_inter"]
    assert res.iou == pytest.approx(vi / (va + vb - vi), rel=TOL)
    assert vi == pytest.approx(va / 2, rel=1e-4)


# --------------------------------------------------------------------------- #
# 6. the normalisation contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("frame", ["gt", "bbox", "div200"])
def test_normalize_pair_puts_the_gt_into_the_cube_by_its_own_extent(frame):
    gt = box((3, 2, 1), translate=(1e6, 0, 0))
    a, _ = normalize_pair(gt, box(translate=(1e6 + 5, 0, 0)), pred_frame=frame)
    assert np.allclose(a.bounds.mean(axis=0), 0, atol=1e-9)
    assert np.ptp(a.bounds, axis=0).max() == pytest.approx(1.0, rel=1e-9)


def test_pred_frame_is_required_and_checked():
    with pytest.raises(TypeError):
        normalize_pair(box(), box())
    with pytest.raises(ValueError):
        normalize_pair(box(), box(), pred_frame="joint")


def test_pred_frame_bbox_erases_scale_and_placement():
    """Each side by its own extent: a pred 3x larger and far away scores 1."""
    res = volume_iou(*normalize_pair(box(), box((3, 3, 3), translate=(50, 0, 0)),
                                     pred_frame="bbox"))
    assert res and res.iou == pytest.approx(1.0, rel=TOL)


def test_pred_frame_div200_is_the_pipeline_frame():
    """
    Pipeline convention: GT into [0, 1]^3 by its own extent, pred / 200 + 0.5. In the
    metric's frame that is pred / 200 about the origin.
    """
    gt = box((200, 100, 50))                       # own extent 200, centred: same scale as /200
    pred = box((200, 100, 50), translate=(100, 0, 0))
    g, p = normalize_pair(gt, pred, pred_frame="div200")
    assert np.allclose(p.vertices, pred.vertices / 200)
    res = volume_iou(g, p)
    assert res and res.iou == pytest.approx(1 / 3, rel=TOL)
    off = volume_iou(*normalize_pair(box((20, 10, 5)), box((20, 10, 5)), pred_frame="div200"))
    assert off and off.iou == pytest.approx(0.1 ** 3, rel=TOL)   # GT fills the cube, pred 1/10 of it


def test_gt_outside_the_unit_cube_is_rejected():
    """The metric never moves the pair: a GT out of the cube is the caller's error."""
    res = volume_iou(box(translate=(0.5, 0, 0)), box())
    assert not res and res.status is Status.NOT_NORMALIZED
    assert res.cause is Cause.PRECONDITION
    assert "normalize_pair" in res.detail


def test_gt_normalisation_rounding_is_tolerated():
    gt = box().apply_scale(1 + 1e-7)
    res = volume_iou(gt, box())
    assert res and res.iou == pytest.approx(1.0, rel=1e-6)


def test_pred_outside_the_cube_is_computed_with_a_warning():
    """A prediction is not bound by the cube; far out of it, it gets a warning."""
    res = volume_iou(box(), box(translate=(0.5, 0, 0)))
    assert res and res.iou == pytest.approx(1 / 3, rel=TOL)
    assert any("pred leaves the unit cube" in w for w in res.warnings)
    near = volume_iou(box(), box(translate=(0.003, 0, 0)))
    assert near and not any("unit cube" in w for w in near.warnings)


def test_production_frame_needs_normalize_pair():
    """
    The training pipeline hands over GT scaled into [0,1] and the prediction
    scaled by 1/200 and shifted by +0.5 -- a shared frame centred on
    (0.5, 0.5, 0.5).  It is refused as is; normalize_pair makes it valid.
    """
    gt = box(extents=(1, 1, 1), translate=(0.5, 0.5, 0.5))
    pred = box(extents=(1, 1, 1), translate=(1.0, 0.5, 0.5))  # sticks out past 1.0
    assert volume_iou(gt, pred).status is Status.NOT_NORMALIZED
    res = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1 / 3, rel=TOL)


def test_pred_frame_gt_preserves_relative_placement():
    """Per-mesh normalisation would collapse the offset; the GT's transform must not."""
    assert iou_of(box(), box(translate=(0.5, 0, 0))) == pytest.approx(1 / 3, rel=TOL)


def test_underfilled_box_warns():
    """
    A GT occupying a fraction of the cube hints at a wrong normalisation."""
    a, b = normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt")
    a.apply_scale(0.1)
    b.apply_scale(0.1)
    res = volume_iou(a, b)
    assert res and any("fills only" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# 7. self-intersection detector
# --------------------------------------------------------------------------- #


def unit(m):
    m = m.copy()
    m.apply_translation(-m.bounds.mean(axis=0))
    m.apply_scale(1.0 / np.ptp(m.bounds, axis=0).max())
    return m


def glued(*meshes):
    return unit(trimesh.util.concatenate(list(meshes)))


@pytest.mark.parametrize("mesh", [
    pytest.param(unit(box()), id="cube"),
    pytest.param(unit(sphere(subdivisions=3)), id="sphere"),
    pytest.param(unit(trimesh.creation.torus(1.0, 0.3)), id="torus"),
    pytest.param(unit(trimesh.creation.annulus(0.4, 1.0, 1.0)), id="annulus"),
    pytest.param(glued(box(), box(translate=(3, 0, 0))), id="disjoint_bodies"),
])
def test_clean_meshes_are_not_flagged(mesh):
    assert is_self_intersecting(mesh) == (False, 0)


@pytest.mark.parametrize("offset,label", [
    ((1, 1, 0), "edge_contact"),
    ((1, 1, 1), "vertex_contact"),
])
def test_edge_and_vertex_contact_is_not_reported(offset, label):
    assert is_self_intersecting(glued(box(), box(translate=offset)))[0] is False


def test_face_contact_is_reported_but_costs_no_accuracy():
    """
    Coplanar faces lying on each other are a contact, not a crossing, yet the
    exact detector reports them (it has no tolerance to tell the two apart).
    Between two bodies the metric takes it as a contact for the union, not a
    crossing inside a body: touching bodies still get the exact answer.
    """
    assert is_self_intersecting(glued(box(), box(translate=(1, 0, 0))))[0] is True
    res = iou_res(trimesh.util.concatenate([box(), box(translate=(1, 0, 0))]),
                  box((2, 1, 1), translate=(0.5, 0, 0)))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_face_contact_inside_one_body(mode):
    """
    A cavity touching the wall of its body: its face lies on the wall's face
    (coplanar, another triangulation) inside one body.  The detector cannot
    tell this contact from a crossing: both modes build the solid in CGAL
    (the box with an open pocket, 8 - 1); "fail" takes it, as the surface
    winds at most once.
    """
    void = box(translate=(0.5, 0, 0))
    void.invert()
    side = trimesh.util.concatenate([box((2, 2, 2)), void])
    pairs = self_intersecting_pairs(side)
    n = side.face_normals
    on_wall = np.isclose(side.triangles_center[:, 0], 1.0) & np.isclose(np.abs(n[:, 0]), 1.0)
    assert (on_wall[pairs[:, 0]] & on_wall[pairs[:, 1]]).any()   # face on face
    res = iou_res(side, box((2, 2, 2)), on_self_intersection=mode)
    assert res and res.method == "cgal:manifold64"
    assert res.diagnostics["gt"]["cavities"] == 1
    assert res.diagnostics["gt"]["resolved"]["crossing_pairs"] > 0
    assert res.iou == pytest.approx(7 / 8, rel=1e-9)


def _prism(z0, z1, fan, n=6):
    """An n-gonal prism whose caps are fanned from vertex `fan`."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    v = np.r_[np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i]]
    for k in range(1, n - 1):
        i, j = (fan + k) % n, (fan + k + 1) % n
        f += [[fan, j, i], [n + fan, n + i, n + j]]
    m = trimesh.Trimesh(v, np.array(f), process=False)
    if m.volume < 0:
        m.invert()
    return m


HEX = _prism(0, 1, 0).volume


@pytest.mark.parametrize("parts, volume, shells", [
    ([box(), box(translate=(1, 0, 0))], 2.0, 2),                        # face, diagonals cross
    ([box(), box(translate=(1, 1, 0))], 2.0, 2),                        # one edge
    ([box(), box(translate=(1, 1, 0)), box(translate=(1, 0, 0))], 3.0, 3),
    ([_prism(0, 1, 0), _prism(1, 2, 3)], 2 * HEX, 2),                   # 6-edge contact
])
def test_bodies_touching_split_into_closed_shells(parts, volume, shells):
    """
    Edges of 4 or 6 faces where bodies touch: the faces are paired around the
    edge across wedges of material, so each shell is closed without inventing
    faces.  trimesh's split cut such shells open and its fill_holes closed
    only holes of 3-4 edges: the 6-edge contact was refused as not_manifold.
    """
    m, _ = mesh_iou._clean(trimesh.util.concatenate(parts))
    got = _cgal.shells(np.ascontiguousarray(m.vertices), np.ascontiguousarray(m.faces, np.int64))
    assert got is not None and len(got) == shells and all(g[3] for g in got)
    assert sum(g[2] for g in got) == pytest.approx(volume, rel=1e-12)
    prep, err = mesh_iou._prepare(m, "gt", check_self_intersection=True,
                                  on_self_intersection="exact")
    assert err is None and prep.volume == pytest.approx(volume, rel=1e-9)


def test_contact_tessellated_alike_is_built_from_the_raw_surface():
    """
    A contact face tessellated alike on both bodies: cleaning keeps one copy
    of each coincident triangle (as trimesh does, so a double-sided STL keeps
    its surface), its edges get 3 faces, the module leaves the split to
    trimesh, and the pieces are open (trimesh's fill_holes is not used): the
    cleaned side alone is not_manifold.  The raw surface has every edge
    balanced and defines the solid, so the side is built from it in CGAL --
    in "fail" too, where it winds at most 1 (no overlap, no pass through
    itself to tell apart).
    """
    raw = trimesh.util.concatenate([_prism(0, 1, 0), _prism(1, 2, 0)])
    m, _ = mesh_iou._clean(raw)
    assert _cgal.shells(np.ascontiguousarray(m.vertices),
                        np.ascontiguousarray(m.faces, np.int64)) is None
    _, err = mesh_iou._prepare(m, "gt", check_self_intersection=True,
                               on_self_intersection="exact")
    assert err is not None and err.status is Status.NOT_MANIFOLD     # cleaned: unbalanced
    prep, err = mesh_iou._prepare(raw, "gt", check_self_intersection=True,
                                  on_self_intersection="exact")
    assert err is None and prep.volume == pytest.approx(2 * HEX, rel=1e-9)
    assert prep.info["raw_solid"]["refused_as"] == "not_manifold"
    assert prep.info["raw_solid"]["rule"] == "split_open"
    prep, err = mesh_iou._prepare(raw, "pred", check_self_intersection=True,
                                  on_self_intersection="fail")
    assert err is None and prep.volume == pytest.approx(2 * HEX, rel=1e-9)
    assert prep.info["raw_solid"]["winding_max"] == 1


def test_bodies_touching_alike_in_l_shape_match_separate_boxes():
    """
    Three unit cubes in an L, each contact face tessellated alike (the second
    and third cube mirror the first): the solid from the raw surface, through
    volume_iou and to_body, against the same L built with separate boxes.
    """
    a = box()

    def mirror(axis):
        v = a.vertices.copy()
        v[:, axis] = 1 - v[:, axis]
        return trimesh.Trimesh(v, a.faces[:, ::-1], process=False)
    side = trimesh.util.concatenate([a, mirror(0), mirror(1)])
    ref = trimesh.util.concatenate([box(), box(translate=(1, 0, 0)), box(translate=(0, 1, 0))])
    res = volume_iou(*normalize_pair(side, ref, pred_frame="gt"))
    assert res.status is Status.OK and res.iou == pytest.approx(1.0, abs=1e-9)
    assert any("solid built from the raw surface" in w for w in res.warnings)
    assert not any("check failed" in w for w in res.warnings)
    half = trimesh.util.concatenate([box(), box(translate=(1, 0, 0))])
    res = volume_iou(*normalize_pair(half, side, pred_frame="gt"),
                     on_self_intersection=("exact", "exact"))
    assert res.status is Status.OK and res.iou == pytest.approx(2 / 3, abs=1e-9)
    b = to_body(side, frame="bbox", label="gt", on_self_intersection="exact")
    assert isinstance(b, Body) and b.volume == pytest.approx(3 / 8, rel=1e-9)


def _twice(m):
    """Every face and its reversed copy: a two-sided STL, winding 0 everywhere."""
    return trimesh.util.concatenate([m, trimesh.Trimesh(m.vertices, m.faces[:, ::-1],
                                                        process=False)])


def test_shell_written_twice_is_unfolded_before_the_raw_surface():
    """
    A box written twice next to a contact tessellated alike: the raw surface
    is built in CGAL (the contact), where the twice-written box winds 0 and
    was lost.  Unfolded first -- the cell it encloses -- the box is a body in
    both paths and both modes; the arbiter over the unfolded side agrees.
    """
    contact = trimesh.util.concatenate([_prism(0, 1, 0), _prism(1, 2, 0)])
    side = trimesh.util.concatenate([_twice(box(translate=(-5, 0, 0))), contact])
    prep, err = mesh_iou._prepare(side, "gt", check_self_intersection=True,
                                  on_self_intersection="exact")
    assert err is None and prep.volume == pytest.approx(1 + 2 * HEX, rel=1e-9)
    assert prep.info["doubled_shells"]["closed"] == 1
    assert "raw_solid" in prep.info
    assert prep.info["faces_raw"] == len(side.faces)                    # the side as given
    unfolded, _ = mesh_iou._doubled_shells(side)
    assert len(unfolded.faces) < len(side.faces)
    assert arbiter_iou(unfolded, prep.mesh, n=4000).iou == pytest.approx(1.0, abs=1e-12)
    prep, err = mesh_iou._prepare(side, "pred", check_self_intersection=True,
                                  on_self_intersection="fail")
    assert err is None and prep.volume == pytest.approx(1 + 2 * HEX, rel=1e-9)  # the contact
    plain, _ = mesh_iou._prepare(contact, "gt", check_self_intersection=True,
                                 on_self_intersection="exact")
    assert "doubled_shells" not in plain.info                           # pairs joined to others


def test_raw_surface_body_uses_the_stored_fail_proof():
    """
    A body built in "exact" from the raw surface after cleaning opened it (a
    fin written both ways on a box's edge: one copy left, its free edges
    open; the raw surface winds at most 1, a number in "fail" as well) is
    taken in "fail" when its stored exact readings prove that mode valid.
    A legacy record without that proof keeps the original source refusal.
    Bodies touching with the same triangulation use the same stored proof.
    """
    gt = box((2.0, 2.0, 2.0), translate=(0, 0, 1))
    cube = box()
    fin = trimesh.Trimesh(np.r_[cube.vertices, [[1.5, 0.0, 0.5]]],
                          np.r_[cube.faces, [[1, 3, 8], [8, 3, 1]]], process=False)
    a, b = cube.vertices[1], cube.vertices[3]
    assert np.count_nonzero(np.abs(a - b) > 0) == 1                      # a box edge
    ref = volume_iou(*normalize_pair(gt, fin, pred_frame="gt"))
    assert ref                          # "fail" too: the raw surface winds at most 1
    pb = to_body(fin, frame="gt", gt=gt, label="pred", on_self_intersection="exact")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["refused_as"] == "open_surface"
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)
    import copy
    legacy = Body(pb.vertices, pb.faces, copy.deepcopy(pb.meta))
    legacy.meta["info"]["raw_solid"]["assembly"].pop("fail_validity")
    refused = volume_iou(*normalize_pair(gt, legacy, pred_frame="gt"))
    assert refused.status is Status.OPEN_SURFACE and refused.blame == "pred"
    assert refused.detail == pb.meta["info"]["raw_solid"]["refused_detail"]
    assert volume_iou(*normalize_pair(gt, pb, pred_frame="gt"), on_self_intersection="exact")
    pb = to_body(fin, frame="gt", gt=gt, label="pred", on_self_intersection="fail")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["winding_max"] == 1
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)
    contact = trimesh.util.concatenate([_prism(0, 1, 0), _prism(1, 2, 0)])
    ref = volume_iou(*normalize_pair(gt, contact, pred_frame="gt"))
    assert ref
    pb = to_body(contact, frame="gt", gt=gt, label="pred", on_self_intersection="fail")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["winding_max"] == 1
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)
    pb = to_body(contact, frame="gt", gt=gt, label="pred", on_self_intersection="exact")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["rule"] == "split_open"
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)
    over = trimesh.util.concatenate([contact, box((0.5, 0.5, 0.5), translate=(0, 0, 0.5))])
    ref = volume_iou(*normalize_pair(gt, over, pred_frame="gt"))
    assert ref                          # winds 2 by two parts: their overlap, a union
    pb = to_body(over, frame="gt", gt=gt, label="pred", on_self_intersection="fail")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["winding_max"] == 2
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)


@pytest.mark.parametrize("stl", [False, True])
def test_raw_surface_fallback_in_fail_refuses_a_part_through_itself(stl, tmp_path, monkeypatch):
    """
    Two boxes touching with a face tessellated alike, a fin written both ways
    on an outer edge: cleaning leaves the fin's edges open (`open_surface`,
    no split into open pieces), and the side goes to the raw-surface fallback
    (DEFINITION 4.9) in either mode.  In "fail" (the pred) the raw surface
    winding at most 1 gives the two boxes; a third box overlapping them
    (w = 2 of the sum, another component) is their union, as on 4.9; a
    coiled bar with a fin -- one component winding 2 by itself -- is refused
    as self-intersecting.
    In "exact" (the GT) all are built from the raw surface, as before.
    From an STL file (three vertices per face) the same: components are
    joined at equal coordinates, and the cells of zero volume between the
    two copies of the contact, cut apart by the third box, are not a pass
    through itself.
    """
    a = box()
    v = a.vertices.copy()
    v[:, 0] = 1 - v[:, 0]
    b = trimesh.Trimesh(v, a.faces[:, ::-1], process=False)              # touches x = 0.5
    assert np.allclose(a.vertices[[1, 3], 0], -0.5)                     # an edge of x = -0.5
    fin = trimesh.Trimesh(np.r_[a.vertices, [[-1.0, -0.5, 0.0]]],
                          np.r_[a.faces, [[1, 3, 8], [8, 3, 1]]], process=False)
    pair = trimesh.util.concatenate([fin, b])
    over = trimesh.util.concatenate([pair, box((0.5, 0.5, 0.5), translate=(0.5, 0, 0))])
    coil = coiled_bar()
    k = len(coil.vertices)
    assert coil.faces[0].tolist() == [5, 1, 0]                          # 0-1: an edge
    coil = trimesh.Trimesh(np.r_[coil.vertices, [[1.0, 0.0, -0.7]]],
                           np.r_[coil.faces, [[0, 1, k], [k, 1, 0]]], process=False)
    if stl:
        pair, over, coil = (via_stl(m, tmp_path / f"{i}.stl")
                            for i, m in enumerate((pair, over, coil)))
    from test_side_rules import spy_parts as spy_raw_readings
    calls, parts = [], spy_raw_readings(monkeypatch)
    orig = mesh_iou._raw_solid

    def spy(m, label, err, **kw):
        calls.append(err.status)
        return orig(m, label, err, **kw)

    monkeypatch.setattr(mesh_iou, "_raw_solid", spy)

    def fail(side):
        calls.clear()
        parts.clear()
        return mesh_iou._prepare(side, "pred", check_self_intersection=True,
                                 on_self_intersection="fail")
    prep, err = fail(pair)
    assert calls == [Status.OPEN_SURFACE] and parts == [0]              # the 4.9 path
    assert err is None and prep.volume == pytest.approx(2.0, rel=1e-9)
    raw = prep.info["raw_solid"]
    assert raw["winding_max"] == 1 and raw["refused_as"] == "open_surface" and "rule" not in raw
    assert "self_overlap_parts" not in raw
    prep, err = fail(over)
    assert calls == [Status.OPEN_SURFACE] and parts == [0]              # parts overlapping only
    assert err is None and prep.volume == pytest.approx(2.0, rel=1e-9)
    raw = prep.info["raw_solid"]
    assert raw["winding_max"] == 2 and raw["self_overlap_parts"] == 0 and "rule" not in raw
    prep, err = fail(coil)
    assert parts == [1] and calls == [Status.OPEN_SURFACE]              # one part through itself
    assert prep is None and err.status is Status.SELF_INTERSECTING and err.blame == "pred"
    assert err.diagnostics["raw_winding_max"] == 2
    assert err.diagnostics["self_overlap_parts"] == 1
    for side in (pair, over, coil):
        calls.clear()
        parts.clear()
        prep, err = mesh_iou._prepare(side, "gt", check_self_intersection=True,
                                      on_self_intersection="exact")
        assert calls == [Status.OPEN_SURFACE] and parts == []
        assert err is None
        assert "winding_max" not in prep.info["raw_solid"]
        assert "self_overlap_parts" not in prep.info["raw_solid"]
        if side is not coil:
            assert prep.volume == pytest.approx(2.0, rel=1e-9)
    gt = box((2.0, 1.0, 1.0), translate=(0.5, 0, 0))
    res = volume_iou(*normalize_pair(gt, pair, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    res = volume_iou(*normalize_pair(gt, over, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)             # the union
    res = volume_iou(*normalize_pair(over, gt, pred_frame="gt"))       # over as the GT
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    res = volume_iou(*normalize_pair(gt, coil, pred_frame="gt"))
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"
    # a body file of the side: built in "fail" (parts overlapping only) it is
    # taken in "fail"; exact construction also stores the proof for that mode
    pb = to_body(over, frame="gt", gt=gt, label="pred", on_self_intersection="fail")
    assert isinstance(pb, Body) and pb.meta["info"]["raw_solid"]["self_overlap_parts"] == 0
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    pb = to_body(over, frame="gt", gt=gt, label="pred", on_self_intersection="exact")
    assert isinstance(pb, Body)
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_hollow_box_written_twice_is_refused_unless_filled(mode):
    """
    A hollow box written twice reads the same as a box with another box
    inside it (DEFINITION 4.8, fact 2): the inner cell may be a cavity or a
    body.  Refused without filling; with the cavities filled it is the outer
    box under either reading.
    """
    hollow = trimesh.util.concatenate([box(extents=(2, 2, 2)), box()])
    hollow.faces[12:] = hollow.faces[12:, ::-1]
    _, err = mesh_iou._prepare(_twice(hollow), "gt", check_self_intersection=True,
                               on_self_intersection=mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    assert err.diagnostics["doubled_shells"]["ambiguous_cells"] == 1
    prep, err = mesh_iou._prepare(_twice(hollow), "gt", check_self_intersection=True,
                                  on_self_intersection=mode, fill_cavities=True)
    assert err is None and prep.volume == pytest.approx(8.0, rel=1e-9)
    assert prep.info["doubled_shells"]["closed"] == 2


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_flat_sheet_written_twice_is_dropped(mode):
    """
    A square written twice: its copy is open, but encloses nothing even when
    capped -- a stray sheet, no volume under any reading.  Both copies are
    dropped, the side is the box.
    """
    sq = trimesh.Trimesh([[0, 0, 3], [1, 0, 3], [1, 1, 3], [0, 1, 3]], [[0, 1, 2], [0, 2, 3]],
                         process=False)
    prep, err = mesh_iou._prepare(trimesh.util.concatenate([box(), _twice(sq)]), "gt",
                                  check_self_intersection=True, on_self_intersection=mode)
    assert err is None and prep.volume == pytest.approx(1.0, rel=1e-12)
    d = prep.info["doubled_shells"]
    assert d["sheets"] == 1 and d["open"] == 0 and d["closed"] == 0


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_open_shell_written_twice_refuses_the_side(mode):
    """
    A box without its top written twice: one copy is open and capped would
    enclose a volume.  A solid with a hole and a sheet then look the same, so
    the side is refused as open in both modes -- in "exact" too, where the raw
    surface (winding 0 on it) would drop it.
    """
    lid = box()
    lid = trimesh.Trimesh(lid.vertices, lid.faces[lid.face_normals[:, 2] < 0.5], process=False)
    _, err = mesh_iou._prepare(trimesh.util.concatenate([box(), _twice(lid.apply_translation((3, 0, 0)))]),
                               "gt", check_self_intersection=True, on_self_intersection=mode)
    assert err is not None and err.status is Status.OPEN_SURFACE
    assert err.diagnostics["doubled_shells"]["open"] == 1
    assert "raw_solid" not in err.diagnostics


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_open_copy_is_refused_though_small_for_the_side(mode):
    """
    A box without its top written twice beside a large box: the hole is 20%
    of the copy -- left open -- but under 5% of the whole side.  The side is
    refused as open, not measured: the hole is measured against its own
    component in cleaning too, and its patch would add volume where the
    surface's winding number is fractional (cleaning keeps one copy of the
    open box: it winds 0.5..1 inside).
    """
    lid = box()
    lid = trimesh.Trimesh(lid.vertices, lid.faces[lid.face_normals[:, 2] < 0.5], process=False)
    side = trimesh.util.concatenate([box(extents=(10, 10, 10)),
                                     _twice(lid.apply_translation((20, 0, 0)))])
    _, err = mesh_iou._prepare(side, "gt", check_self_intersection=True,
                               on_self_intersection=mode)
    assert err is not None and err.status is Status.OPEN_SURFACE
    assert "holes left open" in err.detail
    assert err.diagnostics["doubled_shells"]["open"] == 1


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_housing_written_twice_with_a_single_sided_pin(mode):
    """
    A housing written twice and a pin written once, sticking out of it: the
    housing is unfolded, the pin taken as is, the two united.  Read by the
    winding number of the raw surface the housing winds 0 and only the pin
    would be left.
    """
    pin = box((0.5, 0.5, 2), translate=(0, 0, 1.5))
    side = trimesh.util.concatenate([_twice(box((2, 2, 2))), pin])
    prep, err = mesh_iou._prepare(side, "gt", check_self_intersection=True,
                                  on_self_intersection=mode)
    assert err is None and prep.volume == pytest.approx(8 + 0.5 - 0.125, rel=1e-9)
    assert prep.info["doubled_shells"]["closed"] == 1
    assert prep.info["solids"] == 2 and prep.info["union"] and "raw_solid" not in prep.info
    assert mesh_iou._cgal_solid(side)[1] == pytest.approx(0.5, rel=1e-9)
    ref = trimesh.util.concatenate([box((2, 2, 2)), pin])
    res = iou_res(side, ref, on_self_intersection=mode)
    assert res and res.iou == pytest.approx(1.0, rel=TOL)


def _cracked_box(n_subdivide):
    """A unit box subdivided `n_subdivide` times, its first triangle missing."""
    fine = box()
    for _ in range(n_subdivide):
        fine = fine.subdivide()
    return trimesh.Trimesh(fine.vertices, fine.faces[1:], process=False)


def test_crack_closed_in_a_copy_written_twice_is_warned():
    """
    A fine box with one triangle missing, written twice: the crack is closed in
    the copy by the side's rule (DEFINITION 4.3: the patch's proven volume
    error within 1e-3 of the copy's volume), before the side is prepared again
    -- the warnings about invented geometry and the patch's bound come from
    the copy's own record.
    """
    cracked = _cracked_box(4)
    res = iou_res(_twice(cracked), box())
    assert res and res.iou == pytest.approx(1.0, rel=TOL)
    d = res.diagnostics["gt"]["doubled_shells"]
    assert d["closed"] == 1 and d["loop_faces_added"] == 1
    # as for the side: the patch against the surface before it
    assert d["patched_area_frac"] == pytest.approx(1 / len(cracked.faces), rel=1e-6)
    (h,) = d["holes"]
    assert h["accepted"] and h["loops"] == 1 and h["parts_volume"] == pytest.approx(1.0, rel=1e-9)
    assert 0 < h["bound"] <= 1e-3 * h["parts_volume"]
    assert any("shells written twice, covering 0.03%" in w for w in res.warnings)
    assert any(f"1 holes patched in shells written twice: proven volume error at most "
               f"{h['bound']:.3g}" in w for w in res.warnings)


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_crack_in_a_copy_above_the_bound_leaves_it_open(mode):
    """
    The same crack on a coarser box: its patch is 0.5% of the copy's area,
    but its proven volume error is 4% of the box -- the copy stays open and
    the side is refused, as the side's own crack would be (no budget of area).
    """
    _, err = mesh_iou._prepare(_twice(_cracked_box(2)), "gt", check_self_intersection=True,
                               on_self_intersection=mode)
    assert err is not None and err.status is Status.OPEN_SURFACE
    d = err.diagnostics["doubled_shells"]
    assert d["open"] == 1 and d["closed"] == 0 and d["loop_faces_added"] == 0 and "holes" not in d


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_branching_hole_in_a_copy_leaves_it_open(mode):
    """
    Two missing triangles of a fine box meeting at a vertex, the box written
    twice: the copy's hole has no simple loops, nothing patches it, the copy
    stays open and the side is refused as open in both modes.  The same two
    holes apart are cracks the copy's patch closes.
    """
    for touching in (True, False):
        side = _twice(_two_holes(touching))
        prep, err = mesh_iou._prepare(side, "gt", check_self_intersection=True,
                                      on_self_intersection=mode)
        if not touching:
            assert err is None and prep.volume == pytest.approx(1.0, rel=TOL)
            assert prep.info["doubled_shells"]["closed"] == 1
            continue
        assert err is not None and err.status is Status.OPEN_SURFACE
        d = err.diagnostics["doubled_shells"]
        assert d["open"] == 1 and d["closed"] == 0 and d["loop_faces_added"] == 0
        assert "raw_solid" not in err.diagnostics


def test_copy_written_twice_is_wound_before_its_crack_is_patched(monkeypatch):
    """
    The copy takes one face of every pair, whichever way it runs: its windings
    are not agreed, and the patch's chain does not close over it.  Wound
    first, the copy is closed; left as it is, it would stay open.
    """
    cracked = _twice(_cracked_box(4))
    _, d = mesh_iou._doubled_shells(cracked)
    assert d["closed"] == 1 and d["open"] == 0 and d["holes"][0]["closed"]
    with monkeypatch.context() as mp:
        mp.setattr(mesh_iou, "_fix_winding", lambda m, any_start=False: None)
        _, d = mesh_iou._doubled_shells(cracked)
    assert d["closed"] == 0 and d["open"] == 1


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_bodies_touching_alike_written_twice_are_their_union(mode):
    """
    Two boxes touching along a face tessellated alike, the whole written
    twice: the contact face comes four times (two each way).  One copy of the
    component drops it -- it is inside the union -- and keeps every other face
    once; next to a single-sided contact (the raw-surface path) the union is
    not lost either.  Requiring exactly two copies per face missed such a
    side, and its raw surface kept only a small separate shell.
    """
    a = box()
    v = a.vertices.copy()
    v[:, 0] = 1 - v[:, 0]
    pair = trimesh.util.concatenate([a, trimesh.Trimesh(v, a.faces[:, ::-1], process=False)])
    prep, err = mesh_iou._prepare(_twice(pair), "gt", check_self_intersection=True,
                                  on_self_intersection=mode)
    assert err is None and prep.volume == pytest.approx(2.0, rel=1e-9)
    d = prep.info["doubled_shells"]
    assert d["closed"] == 1 and d["contact_faces"] == 2 and d["open"] == 0
    if mode == "exact":
        contact = trimesh.util.concatenate([_prism(5, 6, 0), _prism(6, 7, 0)])
        prep, err = mesh_iou._prepare(trimesh.util.concatenate([_twice(pair), contact]), "gt",
                                      check_self_intersection=True, on_self_intersection=mode)
        assert err is None and prep.volume == pytest.approx(2.0 + 2 * HEX, rel=1e-9)


def _scrambled(m, rng, flip=True):
    """`m` with its faces shuffled and (`flip`) about half of them reversed."""
    f = m.faces.copy()
    flip = (rng.random(len(f)) < 0.5) & flip
    f[flip] = f[flip][:, ::-1]
    return trimesh.Trimesh(m.vertices, f[rng.permutation(len(f))], process=False)


def _unfold_both(side, monkeypatch):
    """`_doubled_shells(side)` as it is and with trimesh's own fix_winding."""
    fast = mesh_iou._doubled_shells(side)
    with monkeypatch.context() as mp:
        mp.setattr(mesh_iou, "_fix_winding", trimesh.repair.fix_winding)
        slow = mesh_iou._doubled_shells(side)
    return fast, slow


def test_fix_winding_matches_trimesh(monkeypatch):
    """
    The one copy of a shell written twice mixes windings; it is made
    consistent without trimesh's walk.  Where an edge has other than two
    faces (boxes touching along an edge, a stray triangle) the faces are
    trimesh's to the bit, a surface that cannot be oriented goes to trimesh
    itself; where every edge has two faces a shell may get the other winding,
    but the unfolded side -- shells oriented outward -- is the same to the bit.
    """
    rng = np.random.default_rng(1)
    edge = trimesh.util.concatenate([box(), box(translate=(1, 1, 0))])   # touch along an edge
    edge = trimesh.Trimesh(edge.vertices, edge.faces, process=True)
    stray = trimesh.Trimesh([[5, 5, 5], [6, 5, 5], [5, 6, 5]], [[0, 1, 2]], process=False)
    for m in [trimesh.util.concatenate([edge, stray])] * 10 + [klein_bottle()]:
        m = _scrambled(m, rng)
        assert not m.is_winding_consistent
        ours, ref = m.copy(), m.copy()
        mesh_iou._fix_winding(ours)
        trimesh.repair.fix_winding(ref)
        assert np.array_equal(ours.faces, ref.faces)
    hollow = trimesh.util.concatenate([box(extents=(2, 2, 2)), box(), box(translate=(4, 0, 0))])
    hollow.faces[12:24] = hollow.faces[12:24, ::-1]
    mixed = _scrambled(hollow, rng)
    mesh_iou._fix_winding(mixed)
    assert mixed.is_winding_consistent
    for side, closed in ((_twice(hollow), 3), (_twice(edge), 1)):
        side = _scrambled(side, rng, flip=False)
        (fast, info), (slow, info_slow) = _unfold_both(side, monkeypatch)
        assert info == info_slow and info["closed"] == closed
        assert np.array_equal(fast.vertices, slow.vertices)
        assert np.array_equal(fast.faces, slow.faces)


def test_crossing_pair_with_a_shared_vertex_is_detected():
    """
    The former numpy detector skipped every pair sharing a vertex, but a
    triangle can leave a shared vertex straight through its neighbour.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.6, 0.2, 0.5], [0.2, 0.6, -0.5]], float)
    m = trimesh.Trimesh(v, np.array([[0, 1, 2], [0, 3, 4]]), process=False)
    assert is_self_intersecting(m) == (True, 1)


def test_near_contact_is_not_a_crossing():
    """
    An edge 1e-12 above a face is exactly disjoint from it; the
    former numpy detector's tolerance called it a crossing and sent the body to CGAL.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0.2, 0.2, 1e-12], [0.6, 0.2, 1e-12], [0.4, 0.2, 0.5]], float)
    m = trimesh.Trimesh(v, np.array([[0, 1, 2], [3, 4, 5]]), process=False)
    assert is_self_intersecting(m) == (False, 0)


@pytest.mark.parametrize("offset", [(0.5, 0, 0), (0.4, 0.3, 0.2), (0.9, 0.9, 0.9)])
def test_real_penetration_is_detected(offset):
    hit, count = is_self_intersecting(glued(box(), box(translate=offset)))
    assert hit is True and count > 0


def test_penetrating_spheres_detected():
    hit, count = is_self_intersecting(
        glued(sphere(subdivisions=2), sphere(subdivisions=2, translate=(0.5, 0, 0)))
    )
    assert hit is True and count > 0


def test_self_intersection_check_can_be_disabled():
    c = coiled_bar()
    assert iou_res(c, box((2.4, 2.4, 0.6)), on_self_intersection="fail").status is Status.SELF_INTERSECTING
    res = iou_res(c, box((2.4, 2.4, 0.6)), check_self_intersection=False)
    # unchecked, the double-covered part counts twice -- a number, but not the
    # metric's; the result says it was not checked
    assert res.diagnostics["gt"]["self_intersection_checked"] is False


def test_flat_plate_is_not_flagged():
    """A thin plate: every face's box spans the plate -- the worst broad phase."""
    plate = trimesh.creation.box(extents=(1, 1, 0.002)).subdivide().subdivide()
    assert is_self_intersecting(plate) == (False, 0)


def test_detector_pairs_do_not_depend_on_threads():
    m = glued(sphere(subdivisions=4), sphere(subdivisions=4, translate=(0.5, 0.1, 0)))
    one = self_intersecting_pairs(m, threads=1)
    assert len(one) > 0
    for t in (2, 5):
        assert np.array_equal(self_intersecting_pairs(m, threads=t), one)


@pytest.mark.parametrize("threads", [1, 4])
def test_detector_first_stops_at_one_pair(threads):
    """first=True returns one of the pairs a full scan finds, or none on an
    embedded surface."""
    m = glued(sphere(subdivisions=4), sphere(subdivisions=4, translate=(0.5, 0.1, 0)))
    full = {tuple(p) for p in self_intersecting_pairs(m, threads=1)}
    got = self_intersecting_pairs(m, threads=threads, first=True)
    assert len(got) == 1 and tuple(got[0]) in full
    clean = sphere(subdivisions=4)
    assert len(self_intersecting_pairs(clean, threads=threads, first=True)) == 0


def test_detector_profile_counts_as_the_scan():
    from volumetric_iou import _cgal
    m = glued(sphere(subdivisions=3), sphere(subdivisions=3, translate=(0.5, 0.1, 0)))
    p = _cgal.detector_profile(np.ascontiguousarray(m.vertices, np.float64),
                               np.ascontiguousarray(m.faces, np.int64))
    assert p["pairs"] == len(self_intersecting_pairs(m, threads=1))
    assert 0 <= p["first_hit"] < p["candidates"]
    p = _cgal.detector_profile(np.ascontiguousarray(sphere().vertices, np.float64),
                               np.ascontiguousarray(sphere().faces, np.int64))
    assert p["pairs"] == 0 and p["first_hit"] == -1


def test_fail_side_stops_at_the_first_crossing(monkeypatch):
    """A side refused on any crossing asks the detector for one; an "exact"
    side and diagnose_mesh still count them all."""
    from volumetric_iou import mesh_iou
    seen = []
    orig = mesh_iou.self_intersecting_pairs

    def spy(mesh, threads=None, first=False):
        seen.append(first)
        return orig(mesh, threads, first)
    monkeypatch.setattr(mesh_iou, "self_intersecting_pairs", spy)
    res = iou_res(coiled_bar(), box((2.4, 2.4, 0.6)), on_self_intersection="fail")
    assert res.status is Status.SELF_INTERSECTING and seen and all(seen)
    seen.clear()
    iou_res(coiled_bar(), box((2.4, 2.4, 0.6)), on_self_intersection="exact")
    # the counts of pairs: all of them; the one call for the first pair is the
    # check of the rounded CGAL solid (one pair keeps it in CGAL)
    assert len(seen) >= 2 and not seen[0] and seen.count(True) == 1
    seen.clear()
    assert diagnose_mesh(coiled_bar())["crossing_pairs_internal"] > 1
    assert not any(seen)


def test_detector_coplanar_fold_on_tilted_plane():
    """
    Two faces sharing an edge and folded onto each other in a tilted plane:
    coordinates of 30 bits make e = 2a - b exactly coplanar, while a plane
    through the face rounded to double does not hold it.  libigl's test (and
    the module before 0.9.0) missed every such pair, and with a rounded plane
    the answer could depend on which face comes first.
    """
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b, c = rng.integers(-2**29, 2**29, size=(3, 3)) / 2**30
        v = np.array([a, b, c, 2 * a - b])
        for f in ([[0, 1, 2], [1, 2, 3]], [[1, 2, 3], [0, 1, 2]]):
            m = trimesh.Trimesh(v, np.array(f), process=False)
            assert len(self_intersecting_pairs(m, threads=1)) == 1


def test_detector_rejects_bad_input():
    from volumetric_iou import _cgal
    v = np.zeros((3, 3))
    with pytest.raises(IndexError):
        _cgal.self_intersecting_pairs(v, np.array([[0, 1, 3]], np.int64))
    with pytest.raises(ValueError):
        _cgal.self_intersecting_pairs(v, np.array([[0, 1, 2]], np.int64), 0)


# --------------------------------------------------------------------------- #
# 8. cavities: a hollow part must not be counted as solid
# --------------------------------------------------------------------------- #


def hollow_box(outer=2.0, inner=1.0):
    """A closed shell with a fully enclosed cubic void: volume = outer^3 - inner^3."""
    shell = trimesh.creation.box(extents=(outer,) * 3)
    void = trimesh.creation.box(extents=(inner,) * 3)
    void.invert()  # normals point into the void, as a cavity's must
    return trimesh.util.concatenate([shell, void])


def test_cavity_is_subtracted_from_volume():
    """
    fix_normals(multibody=True) orients every shell outwards, which turns a
    2x2x2 box with a 1x1x1 void from volume 7 into volume 9.  The DSL has a
    `shell` operation, so hollow parts are ordinary input here.
    """
    res = iou_res(hollow_box(), hollow_box())
    assert res
    # diagnostics are in normalised units; the joint span here is 2
    assert res.diagnostics["volume_gt"] * 2**3 == pytest.approx(7.0, rel=1e-3)
    assert res.iou == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["cavities"] == 1
    assert res.diagnostics["gt"]["cavities_solid"] == 1


def test_cavity_changes_the_iou():
    """A solid cube against the same cube hollowed out: IoU is 7/8, not 1."""
    res = iou_res(hollow_box(), box(extents=(2, 2, 2)))
    assert res
    assert res.iou == pytest.approx(7 / 8, rel=1e-3)


def test_nested_solid_inside_a_cavity_is_added_back():
    """Depth-2 nesting: a solid core inside the void counts as material again."""
    core = trimesh.creation.box(extents=(0.5,) * 3)
    m = trimesh.util.concatenate([hollow_box(), core])
    res = iou_res(m, m)
    assert res
    expected = 8.0 - 1.0 + 0.125
    assert res.diagnostics["volume_gt"] * 2**3 == pytest.approx(expected, rel=1e-3)
    assert res.diagnostics["gt"]["solids"] == 2 and res.diagnostics["gt"]["cavities"] == 1


@pytest.mark.parametrize("timeout", [None, 60])
def test_filled_gt_cavity_against_a_solid_pred(timeout):
    """fill_cavities: the GT's void is material, a solid prediction scores 1."""
    res = iou_res(hollow_box(), box(extents=(2, 2, 2)), fill_cavities=(True, False),
                  timeout=timeout)
    assert res
    assert res.iou == pytest.approx(1.0, rel=TOL)
    g = res.diagnostics["gt"]
    assert g["cavities_filled"] == 1
    assert g["volume_cavities"] * 2**3 == pytest.approx(1.0, rel=1e-3)
    assert res.diagnostics["volume_gt"] * 2**3 == pytest.approx(8.0, rel=1e-3)
    assert any("1 cavities filled" in w for w in res.warnings)


@pytest.mark.parametrize("fill, expected", [(False, 1.0), ((True, False), 7 / 8),
                                            ((False, True), 7 / 8), (True, 1.0)])
def test_fill_cavities_per_side(fill, expected):
    res = iou_res(hollow_box(), hollow_box(), fill_cavities=fill)
    assert res
    assert res.iou == pytest.approx(expected, rel=1e-3)


def test_filling_absorbs_a_body_inside_the_cavity():
    core = trimesh.creation.box(extents=(0.5,) * 3)
    m = trimesh.util.concatenate([hollow_box(), core])
    res = iou_res(m, box(extents=(2, 2, 2)), fill_cavities=True)
    assert res
    assert res.diagnostics["volume_gt"] * 2**3 == pytest.approx(8.0, rel=1e-3)
    assert res.iou == pytest.approx(1.0, rel=TOL)


def test_void_enclosed_by_overlapping_bodies_is_filled():
    """Six overlapping walls: no cavity shell in the input, a cavity in the solid."""
    # extents differ per axis so that no two walls share a face plane
    walls = []
    for axis, ext in enumerate([(0.5, 1.8, 1.6), (1.6, 0.5, 1.6), (2.0, 2.0, 0.5)]):
        for s in (-0.75, 0.75):
            t = [0.0, 0.0, 0.0]
            t[axis] = s
            walls.append(box(ext, t))
    m = trimesh.util.concatenate(walls)
    res = iou_res(m, box())
    assert res and res.diagnostics["gt"]["cavities"] == 0 and res.iou == 0.0
    # the raw surface shows no cavity; the built solid does
    assert res.diagnostics["gt"]["cavities_solid"] == 1
    assert res.diagnostics["pred"]["cavities_solid"] == 0
    res = iou_res(m, box(), fill_cavities=(True, False))
    assert res
    g = res.diagnostics["gt"]
    assert g["cavities_filled"] == 1
    # the void is the unit cube; the GT spans 2, so 1/8 in normalised units
    assert g["volume_cavities"] == pytest.approx(1 / 8, rel=1e-6)
    assert res.iog == pytest.approx(1 / 8 / res.diagnostics["volume_gt"], rel=1e-6)


def test_flat_inward_shell_is_not_a_cavity():
    """
    A flat shell, its sign set by rounding, is not a cavity; a thin shell of
    provable volume is one, as is a plain cavity.
    """
    from volumetric_iou.mesh_iou import _solid_cavities
    void = box((1, 1, 1))
    void.invert()
    flat = box((0.5, 0.5, 0.0))
    flat.apply_transform(trimesh.transformations.rotation_matrix(0.7, [1, 2, 3]))
    flat.invert()
    thin = box((0.5, 0.5, 1e-16))       # at 0: the thickness is exact, volume 2.5e-17
    thin.invert()
    solid = box((2, 2, 2))
    assert _solid_cavities(trimesh.util.concatenate([solid, void])) == 1
    assert _solid_cavities(trimesh.util.concatenate([solid, void, flat])) == 1
    assert _solid_cavities(trimesh.util.concatenate([solid, void, thin])) == 2


def test_filling_a_side_without_cavities_changes_nothing():
    a, b = box(), box(translate=(0.5, 0, 0))
    ref, res = iou_res(a, b), iou_res(a, b, fill_cavities=True)
    assert res.iou == ref.iou
    assert res.diagnostics["gt"]["cavities_filled"] == 0
    assert not any("cavities filled" in w for w in res.warnings)


@pytest.mark.parametrize("bad", ["yes", (True,), (True, 1), None])
def test_fill_cavities_is_validated(bad):
    with pytest.raises(ValueError):
        iou_res(box(), box(), fill_cavities=bad)


# --------------------------------------------------------------------------- #
# 9. failure attribution
# --------------------------------------------------------------------------- #


def test_cause_and_blame_are_reported():
    g = trimesh.creation.box(extents=(2, 2, 2)).subdivide().subdivide()
    g.update_faces(g.triangles_center[:, 2] > 0)
    g.remove_unreferenced_vertices()

    res = iou_res(g, box())
    assert res.status is Status.OPEN_SURFACE
    assert res.cause is Cause.TESSELLATION
    assert res.blame == "gt"

    res = iou_res(box(), g)
    assert res.blame == "pred"


def test_self_intersection_is_invalid_geometry_not_engine():
    res = iou_res(box(), coiled_bar(), on_self_intersection="fail")
    assert res.cause is Cause.INVALID_GEOMETRY and res.blame == "pred"


def test_engine_failure_is_blamed_on_the_engine(monkeypatch):
    from volumetric_iou import mesh_iou

    monkeypatch.setattr(mesh_iou, "_intersect",
                        lambda a, b: mesh_iou._to_manifold([box((5, 5, 5))]))
    res = mesh_iou.volume_iou(box(), box(translate=(0.5, 0, 0)), on_self_intersection="fail")
    assert res.cause is Cause.ENGINE and res.blame == "engine"


def test_exact_mode_self_overlap_matches_arbiter():
    """The coiled bar (w = 2 cell) through CGAL: same answer as the arbiter."""
    c, b = coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    res = volume_iou(*normalize_pair(c, b, pred_frame="gt"), on_self_intersection="exact")
    # its solid rounded to float64 crosses itself near x = 0: kept in CGAL
    assert res and res.method == "cgal", f"{res.status.value}: {res.detail}"
    assert res.diagnostics["gt"]["resolved"]["manifold"] is False
    assert res.diagnostics["gt"]["resolved"]["crossing_pairs"] > 0
    arb = arbiter_iou(c, b, n=20_000)
    assert abs(res.iou - arb.iou) < 4 * arb.se
    assert volume_iou(*normalize_pair(c, c, pred_frame="gt"), on_self_intersection="exact").iou == pytest.approx(1.0, abs=1e-9)


def test_cgal_path_goes_through_the_package_module(monkeypatch):
    """The solid and its pockets (one arrangement) and the pair intersection in
    CGAL come from the package module; the libigl build of _boolean_module
    (BOOLEAN_ENV) is not called -- it is only a reference for tools."""
    from volumetric_iou import mesh_iou
    calls, solids = [], []

    class Spy:
        def mesh_boolean(self, *args, **kw):
            calls.append(kw["type_str"])
            raise AssertionError("libigl called")
    real_solid, real_inter = mesh_iou._cgal_solid, mesh_iou._cgal_intersection

    def spy_solid(m):
        solids.append(len(m.faces))
        return real_solid(m)

    def spy_inter(a, b):
        calls.append("module intersection")
        return real_inter(a, b)
    c, b = coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    monkeypatch.setattr(mesh_iou, "_resolved_to_manifold", lambda *a: None)
    base = volume_iou(*normalize_pair(c, b, pred_frame="gt"))
    monkeypatch.setattr(mesh_iou, "_boolean_mod", Spy())
    monkeypatch.setattr(mesh_iou, "_cgal_solid", spy_solid)
    monkeypatch.setattr(mesh_iou, "_cgal_intersection", spy_inter)
    res = volume_iou(*normalize_pair(c, b, pred_frame="gt"))
    assert res and res.method == "cgal"
    assert calls == ["module intersection"] and len(solids) == 1
    assert res.iou == base.iou


def test_bad_boolean_setting_raises(monkeypatch, tmp_path):
    """A wrong BOOLEAN_ENV is an error, not a silent fall back to the installed igl."""
    from volumetric_iou import mesh_iou
    monkeypatch.setattr(mesh_iou, "_boolean_mod", None)
    monkeypatch.setenv(mesh_iou.BOOLEAN_ENV, str(tmp_path))
    with pytest.raises(ImportError, match="pyigl_copyleft_cgal"):
        mesh_iou._boolean_module()


def test_body_the_detector_failed_on_goes_to_cgal(monkeypatch):
    """A body the detector could not check is not trusted to manifold."""
    from volumetric_iou import mesh_iou

    def fail(mesh, **k):
        raise RuntimeError("detector down")
    monkeypatch.setattr(mesh_iou, "self_intersecting_pairs", fail)
    c, b = coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    res = volume_iou(*normalize_pair(c, b, pred_frame="gt"), on_self_intersection="exact")
    # the rounded solid is not checked either: it stays in CGAL
    assert res and res.method == "cgal"
    assert res.diagnostics["gt"]["resolved"]["crossing_pairs"] is None
    assert "detector down" in res.diagnostics["gt"]["detector_error"]
    assert any("check failed" in w for w in res.warnings)
    arb = arbiter_iou(c, b, n=20_000)
    assert abs(res.iou - arb.iou) < 4 * arb.se


def test_cgal_refusing_a_nonmanifold_side_blames_the_side(monkeypatch):
    """
    deepcad/00273379: a prediction with edges of 3+ faces had crossings inside
    a body, CGAL refused it, and the refusal went to the engine.  The fault is
    the side's tessellation -- NOT_MANIFOLD, as manifold would have said.
    """
    from volumetric_iou import mesh_iou

    def fail(*a, **k):
        raise RuntimeError("Mesh boolean failed")
    monkeypatch.setattr(mesh_iou, "_cgal_solid", fail)
    side = trimesh.util.concatenate([coiled_bar(), box(translate=(3, 3, 0)),
                                     box(translate=(4, 4, 0))])     # edge contact: 4 faces
    # a repeated face: cleaning drops it, but the raw surface is unbalanced,
    # so it is not built from the raw surface either
    side = trimesh.Trimesh(side.vertices, np.vstack([side.faces, side.faces[:1]]), process=False)
    res = volume_iou(*normalize_pair(side, box((2.4, 2.4, 0.6)), pred_frame="gt"))
    assert res.status is Status.NOT_MANIFOLD and res.blame == "gt"
    assert "Mesh boolean failed" in res.detail


@needs_igl
@pytest.mark.parametrize("exact_booleans", [False, True])
def test_unbalanced_side_refused_before_cgal(monkeypatch, exact_booleans):
    """
    CGAL resolves every self-intersection of a side before it
    finds the winding number is not piecewise constant and refuses -- up to
    30 min on dataset files.  The edge count says so beforehand.
    """
    from volumetric_iou import mesh_iou

    k = klein_bottle()
    assert mesh_iou._boundary_edges(k) == 0 and mesh_iou._unbalanced_edges(k) > 0
    with pytest.raises(RuntimeError, match="Mesh boolean failed"):
        mesh_iou._cgal_boolean(k, None, "union")               # CGAL does refuse it
    with pytest.raises(RuntimeError, match="Mesh boolean failed"):
        mesh_iou._cgal_solid(k)                                # and so does the module

    real_boolean, real_solid = mesh_iou._cgal_boolean, mesh_iou._cgal_solid

    def guarded(fn):
        def call(mesh, *args, **kwargs):
            if mesh_iou._unbalanced_edges(mesh):
                raise AssertionError("CGAL called on a non-PWN side")
            return fn(mesh, *args, **kwargs)
        return call

    monkeypatch.setattr(mesh_iou, "_cgal_boolean", guarded(real_boolean))
    monkeypatch.setattr(mesh_iou, "_cgal_solid", guarded(real_solid))
    monkeypatch.setattr(mesh_iou, "_cgal_solid_bodies", guarded(mesh_iou._cgal_solid_bodies))
    for label in ("gt", "pred"):
        pair = (k, box((4, 4, 1))) if label == "gt" else (box((4, 4, 1)), k)
        res = volume_iou(*normalize_pair(*pair, pred_frame="gt"),
                         on_self_intersection=("exact", "exact"),
                         exact_booleans=exact_booleans)
        assert res.status is Status.NOT_MANIFOLD and "unbalanced" in res.detail
        assert res.blame == label and res.cause.value == "tessellation"
        assert res.diagnostics["unbalanced_edges"] > 0
        assert res.diagnostics["nonmanifold_edges"] == 0
        assert res.diagnostics["crossing_pairs_internal"] > 0
        if label == "pred":
            refused = volume_iou(*normalize_pair(*pair, pred_frame="gt"),
                                 exact_booleans=exact_booleans)
            assert refused.status is Status.NOT_MANIFOLD and refused.blame == "pred"
            assert refused.cause.value == "tessellation"


@needs_igl
def test_resolved_side_same_answer_in_manifold_and_cgal(monkeypatch):
    """The CGAL-resolved solid intersected in manifold, the same solid in CGAL
    (manifold refused it), and the raw shells in CGAL (the path before) agree."""
    from volumetric_iou import mesh_iou
    # a box with a void through its wall: crossings inside one body, CGAL's
    # solid exact in float64 (axis-aligned, dyadic), so manifold takes it
    void = box(translate=(0.5, 0, 0))
    void.invert()
    c, b = trimesh.util.concatenate([box((2, 2, 2)), void]), box((2.4, 2.4, 0.6), (0, 0, 0.1))
    g, p = normalize_pair(c, b, pred_frame="gt")
    fast = volume_iou(g, p)
    assert fast and fast.method == "cgal:manifold64"

    kw = dict(check_self_intersection=True, on_self_intersection="exact")
    prep_g, _ = mesh_iou._prepare(g, "gt", **kw)
    prep_p, _ = mesh_iou._prepare(p, "pred", **kw)
    _, v = mesh_iou._cgal_boolean(prep_g.shells, prep_p.mesh, "intersect")
    raw = v / (prep_g.volume + prep_p.volume - v)
    assert fast.iou == pytest.approx(raw, rel=1e-9)

    monkeypatch.setattr(mesh_iou, "_resolved_to_manifold", lambda *a: None)
    slow = volume_iou(g, p)
    assert slow and slow.method == "cgal"
    assert slow.diagnostics["gt"]["resolved"]["manifold"] is False
    assert slow.iou == pytest.approx(raw, rel=1e-9)


def test_cgal_only_side_reports_its_union(monkeypatch):
    """Bodies merged in one CGAL cluster are reported as a union also when the
    side stays in CGAL (manifold refused its solid)."""
    side = trimesh.util.concatenate([coiled_bar(), box((0.6, 0.6, 0.6), translate=(1.0, 0, 0))])
    monkeypatch.setattr(mesh_iou, "_resolved_to_manifold", lambda *a: None)
    res = volume_iou(*normalize_pair(side, box((2.4, 2.4, 0.6)), pred_frame="gt"))
    assert res and res.method == "cgal"
    assert res.diagnostics["gt"]["union"]
    assert any("bodies with overlapping extents merged" in w for w in res.warnings)


def _coil_and_island():
    """A self-overlapping body plus a separate clean one, as one side."""
    island = box((0.4, 0.4, 0.4), translate=(3.0, 0, 0))
    return trimesh.util.concatenate([coiled_bar(), island]), island


def test_only_the_crossing_body_goes_to_cgal():
    """
    CGAL resolves each crossing body on its own: the whole-side union once
    dropped a separate body (mcb/00015262).  The island keeps its volume.
    """
    from volumetric_iou import mesh_iou
    side, island = _coil_and_island()
    g, p = normalize_pair(side, side, pred_frame="gt")
    kw = dict(check_self_intersection=True, on_self_intersection="exact")
    prep, err = mesh_iou._prepare(g, "gt", **kw)
    assert err is None and prep.info["resolved"]["bodies"] == 1
    coil, _ = mesh_iou._prepare(normalize_pair(side, coiled_bar(), pred_frame="gt")[1], "gt", **kw)
    s = 1.0 / np.max(side.extents)
    assert prep.volume == pytest.approx(coil.volume + island.volume * s ** 3, rel=1e-9)
    res = volume_iou(g, p, on_self_intersection="exact")
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)


def test_body_reaching_a_crossing_body_is_resolved_with_it():
    """
    Bodies whose extents reach a crossing body go to CGAL together with it:
    the solid is w >= 1 of the summed winding (mcb/00048450); the coiled bar
    winds 2, no pocket -- a number.
    """
    lump = box((0.8, 0.8, 0.8), translate=(0.9, 0, 0.1))
    side = trimesh.util.concatenate([coiled_bar(), lump])
    other = box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    g, p = normalize_pair(side, other, pred_frame="gt")
    res = volume_iou(g, p)
    assert res, f"{res.status.value}: {res.detail}"
    assert res.diagnostics["gt"]["resolved"]["bodies"] == 2
    arb = arbiter_iou(g, p, n=20_000)
    assert abs(res.iou - arb.iou) < 4 * arb.se


def test_cluster_union_failing_the_probes_falls_back_to_bodies(monkeypatch):
    """A cluster union that loses volume is redone body by body, and checked."""
    from volumetric_iou import mesh_iou
    lump = box((0.8, 0.8, 0.8), translate=(0.9, 0, 0.1))
    g, p = normalize_pair(trimesh.util.concatenate([coiled_bar(), lump]),
                          box((2.4, 2.4, 0.6), translate=(0, 0, 0.1)), pred_frame="gt")
    want = volume_iou(g, p)
    real, real_bodies, calls = mesh_iou._cgal_solid, mesh_iou._cgal_solid_bodies, []

    def once_lossy(m):
        calls.append(len(m.faces))
        if len(calls) == 1:
            return real(box((0.01, 0.01, 0.01), translate=(9, 9, 9)))
        return real(m)

    def once_lossy_bodies(m, sizes, check, **kw):
        # the cluster: the coil, not proven embedded, has its own pockets looked for
        calls.append(len(m.faces))
        if len(calls) == 1:
            got = real(box((0.01, 0.01, 0.01), translate=(9, 9, 9)))
            return (*got, [0] * len(check), [0.0] * len(check))
        return real_bodies(m, sizes, check, **kw)
    monkeypatch.setattr(mesh_iou, "_cgal_solid", once_lossy)
    monkeypatch.setattr(mesh_iou, "_cgal_solid_bodies", once_lossy_bodies)
    res = volume_iou(g, p)
    assert res and res.iou == pytest.approx(want.iou, rel=1e-9)
    # the cluster, each of its 2 bodies, then their union in CGAL: the coiled
    # bar's rounded solid crosses itself, so manifold does not unite them
    assert len(calls) == 4


def test_cgal_union_losing_a_body_is_refused(monkeypatch):
    """The probe check turns a silently wrong CGAL union into a refusal."""
    from volumetric_iou import mesh_iou
    real = mesh_iou._cgal_solid

    def lossy(m):                     # return a far-away speck instead of the solid
        return real(box((0.01, 0.01, 0.01), translate=(9, 9, 9)))
    monkeypatch.setattr(mesh_iou, "_cgal_solid", lossy)
    res = volume_iou(*normalize_pair(coiled_bar(), box((2.4, 2.4, 0.6)), pred_frame="gt"))
    assert not res and res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "probe points" in res.detail


def test_balanced_cluster_engine_exception_is_blamed_on_engine(monkeypatch):
    from volumetric_iou import mesh_iou

    def boom(*args, **kwargs):
        raise RuntimeError("cluster engine exploded")

    monkeypatch.setattr(mesh_iou, "_cgal_solid", boom)
    res = volume_iou(*normalize_pair(coiled_bar(), box((2.4, 2.4, 0.6)), pred_frame="gt"))
    assert res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "cluster engine exploded" in res.detail


def test_cgal_intersection_losing_volume_is_refused(monkeypatch):
    from volumetric_iou import mesh_iou

    def boom(a, b):
        raise RuntimeError("engine exploded")

    def lossy(a, b):
        return trimesh.Trimesh(), 0.0
    monkeypatch.setattr(mesh_iou, "_intersect", boom)
    monkeypatch.setattr(mesh_iou, "_cgal_intersection", lossy)
    res = volume_iou(*normalize_pair(box(), box(translate=(0.5, 0, 0)), pred_frame="gt"))
    assert not res and res.status is Status.BOOLEAN_FAILED
    assert "probe points" in res.detail


# a square [0, 4]^2 run counter-clockwise with a loop [3, 6] x [1, 2] that
# crosses its right wall (x = 4 .. 4.5 below and above the loop)
_CUT = [(0, 0), (4, 0), (4, 1), (3, 1), (3, 2), (6, 2), (6, 1), (4.5, 1), (4.5, 4), (0, 4)]
_ADD = [(0, 0), (4.5, 0), (4.5, 1), (6, 1), (6, 2), (3, 2), (3, 1), (4, 1), (4, 4), (0, 4)]


def test_inverted_pocket_is_refused():
    """
    The loop run clockwise: w = 1 - 1 = 0 where it overlaps the square, -1
    on [4.5, 6] x [1, 2] outside it.  Whether that was a feature added inside
    out or a cut leaking through the wall, the surface does not say: no solid,
    a refusal of the side, with the inverted volume (1.5 x 1 before scaling).
    """
    pocket = extruded(_CUT)
    other = box((4, 4, 1), translate=(2, 2, 0.5))
    g, p = normalize_pair(pocket, other, pred_frame="gt")
    res = volume_iou(g, p)
    assert res.status is Status.INVERTED_VOLUME and res.blame == "gt"
    assert res.cause is Cause.INVALID_GEOMETRY
    assert res.diagnostics["inverted_volume"] == pytest.approx(1.5 / 6 ** 3, rel=1e-9)
    p, g = normalize_pair(other, pocket, pred_frame="bbox")
    for mode in ("fail", "exact"):          # pred in "fail": built in CGAL as well
        res = volume_iou(p, g, on_self_intersection=mode)
        assert res.status is Status.INVERTED_VOLUME and res.blame == "pred"


@needs_igl
@pytest.mark.parametrize("loop", [_CUT, _ADD])
def test_solid_and_pocket_match_two_unions(loop):
    """One arrangement gives what the two libigl unions give: the solid of the
    surface and the solid of its flipped copy (its w <= -1 cells)."""
    from volumetric_iou import mesh_iou
    m = trimesh.util.concatenate([extruded(loop), coiled_bar()])
    flipped = trimesh.Trimesh(m.vertices, m.faces[:, ::-1], process=False)
    solid, v, pk_faces, pk_volume = mesh_iou._cgal_solid(m)
    ref, v_ref = mesh_iou._cgal_boolean(m, None, "union")
    pk_ref, pk_v_ref = mesh_iou._cgal_boolean(flipped, None, "union")
    assert len(solid.faces) == len(ref.faces) and v == pytest.approx(v_ref, rel=1e-12)
    assert pk_faces == len(pk_ref.faces) and pk_volume == pytest.approx(pk_v_ref, rel=1e-12)
    assert (pk_faces > 0) is (loop is _CUT)


@needs_igl
def test_intersection_matches_libigl():
    """The module's intersection is libigl's mesh_boolean "intersect": the same
    arrangement and filter, so the same faces and volume -- on a self-crossing
    operand too (its w >= 1 is what enters)."""
    import igl.copyleft.cgal as cgal
    from volumetric_iou import mesh_iou
    a = trimesh.util.concatenate([extruded(_ADD), coiled_bar()])
    b = box((4, 4, 1), translate=(2.5, 2, 0.5))
    out, v = mesh_iou._cgal_intersection(a, b)
    V, F, _ = cgal.mesh_boolean(np.asarray(a.vertices, np.float64), np.asarray(a.faces, np.int64),
                                np.asarray(b.vertices, np.float64), np.asarray(b.faces, np.int64),
                                type_str="intersect")
    assert len(out.faces) == len(F) and v > 0
    assert v == pytest.approx(trimesh.Trimesh(V, F, process=False).volume, rel=1e-12)
    far = b.copy()
    far.apply_translation((100, 0, 0))
    assert mesh_iou._cgal_intersection(a, far)[1] == 0.0


@needs_igl
def test_winding_number_matches_libigl():
    import igl
    from volumetric_iou import _cgal
    m = trimesh.util.concatenate([extruded(_CUT), coiled_bar()])
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    pts = np.random.default_rng(0).uniform(m.bounds[0] - 1, m.bounds[1] + 1, (5000, 3))
    w = _cgal.winding_number(V, F, pts)
    assert np.allclose(w, igl.winding_number(V, F, pts), rtol=0, atol=1e-9)
    assert {-1, 0, 1} <= set(np.round(w).astype(int))
    assert len(_cgal.winding_number(V, F, np.zeros((0, 3)))) == 0
    with pytest.raises(IndexError):
        _cgal.winding_number(V, np.array([[0, 1, len(V)]], np.int64), pts)


def test_winding_exact_matches_winding_number():
    """
    The exact count equals the rounded generalized winding number off the
    surface -- at random points and on a grid aligned with the edges of boxes,
    where an axis-parallel ray would pass through edges and vertices; a point
    on the surface is undecided.
    """
    from volumetric_iou import _cgal
    m = trimesh.util.concatenate([extruded(_CUT), coiled_bar()])
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    pts = np.random.default_rng(0).uniform(m.bounds[0] - 1, m.bounds[1] + 1, (5000, 3))
    w = _cgal.winding_exact(V, F, pts)
    assert np.array_equal(w, np.rint(_cgal.winding_number(V, F, pts)).astype(np.int64))
    assert {-1, 0, 1} <= set(w.tolist())

    two = trimesh.util.concatenate([box(), box(translate=(3, 0, 0))])
    V, F = np.asarray(two.vertices, np.float64), np.asarray(two.faces, np.int64)
    g = np.arange(-1.0, 4.5, 0.25)
    grid = np.stack(np.meshgrid(g, g[:9] - 1, g[:9] - 1), -1).reshape(-1, 3)
    off = np.all(np.abs(np.abs(grid - [[0, 0, 0]]) - 0.5) > 0, axis=1) & \
        np.all(np.abs(np.abs(grid - [[3, 0, 0]]) - 0.5) > 0, axis=1)
    w = _cgal.winding_exact(V, F, grid[off])
    assert np.array_equal(w, np.rint(_cgal.winding_number(V, F, grid[off])).astype(np.int64))
    assert set(w.tolist()) == {0, 1}
    on = np.array([[0.5, 0.0, 0.0], [0.5, 0.5, 0.5], [3.0, 0.5, 0.1]])
    assert np.all(_cgal.winding_exact(V, F, on) == mesh_iou.WINDING_UNDECIDED)
    assert len(_cgal.winding_exact(V, F, np.zeros((0, 3)))) == 0


def test_crossing_without_pocket_is_resolved():
    """The same loop run counter-clockwise: w = 2 on the overlap, never < 0."""
    g, p = normalize_pair(extruded(_ADD), box((4, 4, 1), translate=(2, 2, 0.5)),
                          pred_frame="gt")
    res = volume_iou(g, p)
    assert res, f"{res.status.value}: {res.detail}"
    assert res.diagnostics["gt"]["resolved"]["bodies"] == 1
    assert res.iou == pytest.approx(16 / 18.5, rel=TOL)


def test_arbiter_parallel_matches_serial(monkeypatch):
    """Splitting the points over a process pool must not change a single bit."""
    from volumetric_iou import winding
    monkeypatch.setattr(winding, "PARALLEL_MIN", 0)
    c = coiled_bar()
    pts = np.random.default_rng(3).uniform(-1.5, 1.5, (3_001, 3))
    serial = winding.winding_number(c, pts)
    assert np.array_equal(winding.winding_number(c, pts, workers=3), serial)
    b = box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    assert arbiter_iou(c, b, n=5_000, workers=3) == arbiter_iou(c, b, n=5_000)


def test_on_self_intersection_rejects_unknown_mode():
    for mode in ("warn", ("exact",), ("exact", "warn"), ("exact", "fail", "fail")):
        with pytest.raises(ValueError):
            volume_iou(box(), box(), on_self_intersection=mode)


def test_default_modes_gt_exact_pred_fail():
    """A self-overlapping GT is resolved, the same surface as the pred is refused."""
    c, b = coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))
    res = iou_res(c, b)
    assert res and res.method == "cgal"
    assert res.iou == iou_res(c, b, on_self_intersection=("exact", "fail")).iou
    res = iou_res(b, c)
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"
    assert iou_res(b, c, on_self_intersection="exact")
    res = iou_res(c, b, on_self_intersection=("fail", "exact"))
    assert res.status is Status.SELF_INTERSECTING and res.blame == "gt"


def test_detector_failure_in_fail_mode_is_the_engines(monkeypatch):
    """No proof the body is clean -- no number."""
    from volumetric_iou import mesh_iou

    def fail(mesh, **k):
        raise RuntimeError("detector down")
    monkeypatch.setattr(mesh_iou, "self_intersecting_pairs", fail)
    res = iou_res(box(), box((0.5, 0.5, 0.5)))
    assert res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "pred:" in res.detail and "detector down" in res.detail


def test_no_networkx_needed(monkeypatch):
    """
    Hole closing no longer goes through trimesh's fill_holes, which needed
    networkx and made the verdict depend on the environment.
    """
    import sys
    monkeypatch.setitem(sys.modules, "networkx", None)
    holed = fine_box()
    holed.update_faces(np.arange(len(holed.faces)) != 0)
    holed.remove_unreferenced_vertices()
    assert iou_res(holed, box())

# --------------------------------------------------------------------------- #
# 10. per-mesh triage
# --------------------------------------------------------------------------- #


def test_diagnose_clean_mesh():
    d = diagnose_mesh(unit(box()))
    assert d["loaded"] and d["watertight"]
    assert d["boundary_edges"] == 0
    assert d["nonmanifold_edges"] == 0
    assert d["self_intersections"] == 0
    assert d["components"] == 1
    assert d["volume"] > 0


@pytest.mark.parametrize("parts", [
    [box(), box(translate=(1, 1, 0))],                                  # one edge
    [box(), box(translate=(1, 1, 0)), box(translate=(3, 0, 0))],
    [_prism(0, 1, 0), _prism(1, 2, 3)],                                 # 6-edge contact
])
def test_diagnose_components_count_as_split(parts):
    """`components` counts the parts of trimesh's split(only_watertight=False)."""
    m = unit(trimesh.util.concatenate(parts))
    clean, _ = mesh_iou._clean(m)
    assert diagnose_mesh(m)["components"] == len(clean.split(only_watertight=False))


def test_diagnose_separates_cracks_from_self_intersection():
    g = trimesh.creation.box(extents=(2, 2, 2)).subdivide()
    g.update_faces(g.triangles_center[:, 2] > 0)
    g.remove_unreferenced_vertices()
    d = diagnose_mesh(unit(g))
    assert d["boundary_edges"] > 0            # tessellation defect
    assert d["self_intersections"] == 0       # but the geometry is not lying

    si = unit(trimesh.util.concatenate([box(), box(translate=(0.5, 0, 0))]))
    d = diagnose_mesh(si)
    assert d["boundary_edges"] == 0           # perfectly closed
    assert d["self_intersections"] > 0        # and still invalid


def test_diagnose_reports_nonmanifold_edges():
    nm = unit(trimesh.util.concatenate([box(), box(translate=(1, 1, 0))]))
    d = diagnose_mesh(nm)
    assert d["nonmanifold_edges"] > 0
    assert d["boundary_edges"] == 0


# --------------------------------------------------------------------------- #
# 11. junk that poisons the boolean engine
# --------------------------------------------------------------------------- #


def with_sliver(m):
    """Attach a zero-volume patch to a solid, as OpenCascade tessellation does."""
    lo, hi = m.bounds
    v = np.array([hi, hi + [0, 0, 1e-3], hi + [0, 1e-3, 1e-3]])
    sliver = trimesh.Trimesh(v, np.array([[0, 1, 2]]), process=False)
    return trimesh.util.concatenate([m, sliver])


def test_zero_volume_shells_are_dropped():
    """
    A stray patch contributes no volume but makes manifold return an EMPTY
    intersection for two solids that overlap almost completely -- observed on a
    training prediction where the recorded IoU was 0.99 and the module said 0.
    """
    a = box()
    b = with_sliver(box(extents=(1.001, 1.001, 1.001)))
    res = iou_res(a, b)
    assert res, f"rejected: {res.status.value}: {res.detail}"
    assert res.iou == pytest.approx(1.0, rel=1e-2)
    assert any("zero-volume shells dropped" in w for w in res.warnings)


def test_dropping_slivers_does_not_change_a_clean_answer():
    clean = iou_of(box(), box(translate=(0.5, 0, 0)))
    dirty = iou_of(with_sliver(box()), box(translate=(0.5, 0, 0)))
    assert dirty == pytest.approx(clean, rel=1e-6)


def test_small_crack_is_patched():
    """
    A hexagonal crack survives trimesh's fill_holes.  Six leftover boundary
    edges out of thousands turned a correct 0.92 into a refusal, so loops whose
    fan is a negligible share of the surface get closed.
    """
    # the fan around one vertex: a hexagonal hole, which fill_holes leaves open
    # (a single missing triangle it would close by itself)
    # (a patch of proven error under 1e-3 of the sphere: fine enough)
    m = trimesh.creation.icosphere(subdivisions=5)
    m.update_faces(~(m.faces == 0).any(axis=1))
    m.remove_unreferenced_vertices()
    res = iou_res(m, trimesh.creation.icosphere(subdivisions=5))
    assert res, f"rejected: {res.status.value}: {res.detail}"
    assert res.iou == pytest.approx(1.0, rel=1e-3)
    # the patch is invented geometry and must be declared, with its size:
    # on real data a 3% patch moved the IoU by 0.03, a 0.00% patch by nothing
    assert any("invented geometry" in w for w in res.warnings)


def _patch_across_an_edge():
    """A fine box with an L-shaped hole over its edge x = z = 0.5: one loop,
    not planar and not convex, under 5% of the surface."""
    m = box().subdivide().subdivide().subdivide()
    x, y, z = m.triangles_center.T
    hole = (((x > 0.374) & (z > 0.499) & (np.abs(y) < 0.25))
            | ((x > 0.499) & (z > 0.25) & (y > -0.25) & (y < -0.125)))
    return trimesh.Trimesh(m.vertices, m.faces[~hole], process=False)


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_hole_over_an_edge_is_refused(mode):
    """
    A patch over the edge bends: whatever closes the loop -- its fan runs
    through the faces around the hole, its least-area triangulation cuts the
    corner -- guesses a volume of the order of the hole's size cubed.  Under
    5% of the surface (once a crack, fanned), its proven error is 0.03 of the
    box: open.
    """
    res = iou_res(_patch_across_an_edge(), box(), on_self_intersection=mode)
    assert not res and res.status is Status.OPEN_SURFACE
    pc = res.diagnostics["holes"]
    assert pc["closed"] and not pc["accepted"] and pc["worst_ratio"] > 1e-2
    assert res.diagnostics["loop_faces_added"] == 0


def test_gaping_hole_is_still_refused():
    """Capping a large opening would invent volume, so it must stay a refusal."""
    g = trimesh.creation.box(extents=(2, 2, 2)).subdivide().subdivide()
    g.update_faces(g.triangles_center[:, 2] > 0)    # the whole lid is missing
    g.remove_unreferenced_vertices()
    res = iou_res(g, box())
    assert not res and res.status is Status.OPEN_SURFACE


def test_diagnose_survives_garbage():
    assert diagnose_mesh("/definitely/not/here.obj") == {"loaded": False}
    d = diagnose_mesh(trimesh.Trimesh())
    assert d["loaded"] and d["faces_raw"] == 0


# --------------------------------------------------------------------------- #
# 12. containment without rtree (inside.py)
# --------------------------------------------------------------------------- #


def test_containment_matches_analytic_truth():
    """
    Ray parity against a shape whose inside is known in closed form.

    Points near the surface are excluded: a faceted sphere is not the smooth
    one, so disagreement there is the tessellation, not the tester.
    """
    m = sphere(radius=1.0, subdivisions=4)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 1.5, size=(3000, 3))
    radius = np.linalg.norm(pts, axis=1)
    far = np.abs(radius - 1.0) > 0.02

    got, resolved = contains_resolved(m, pts)
    assert resolved.all()
    assert (got[far] == (radius[far] < 1.0)).all()


def test_containment_handles_a_cavity():
    """A point in the void of a hollow part is outside the material."""
    outer = box(extents=(2, 2, 2))
    inner = box(extents=(1, 1, 1))
    inner.invert()                      # cavity: inward-facing shell
    hollow = trimesh.util.concatenate([outer, inner])

    pts = np.array([[0.0, 0.0, 0.0],    # in the cavity
                    [0.75, 0.0, 0.0],   # in the wall
                    [2.0, 0.0, 0.0]])   # outside altogether
    got, resolved = contains_resolved(hollow, pts)
    assert resolved.all()
    assert list(got) == [False, True, False]


def test_points_on_the_surface_are_unresolved_not_guessed():
    """
    Parity is undefined on the surface itself.  Reporting "unresolved" is what
    lets the callers treat a touching shell pair as ambiguous instead of
    flipping its cavity orientation on a coin toss.
    """
    m = box()
    _, resolved = contains_resolved(m, m.vertices)
    assert not resolved.any()
    # and the boolean view is the conservative reading, never a guess
    assert not contains(m, m.vertices).any()


def test_containment_survives_axis_aligned_probes():
    """
    CAD geometry is axis-aligned, so an axis-aligned ray hits shared edges
    constantly.  A grid of probes over a box is the adversarial case; every
    probe that is not exactly on a face must still be answered.
    """
    m = box(extents=(1, 1, 1))
    g = np.linspace(-1, 1, 11)
    pts = np.stack(np.meshgrid(g, g, g), axis=-1).reshape(-1, 3)
    on_face = (np.abs(np.abs(pts) - 0.5) < 1e-12).any(axis=1)

    got, resolved = contains_resolved(m, pts)
    assert resolved[~on_face].all()
    assert (got[~on_face] == (np.abs(pts[~on_face]) < 0.5).all(axis=1)).all()


def test_containment_needs_no_rtree(monkeypatch):
    """
    The whole point of `inside.py`: the training image has manifold3d, trimesh
    and networkx but no rtree, and `trimesh.contains` reaches for it through
    `mesh.triangles_tree`.  Cavity orientation must survive that.
    """
    m = box()

    def no_rtree(self):
        raise ImportError("No module named 'rtree'")

    monkeypatch.setattr(type(m), "triangles_tree", property(no_rtree))
    # with embreex installed trimesh casts rays through embree and never
    # touches triangles_tree, so the precondition only holds without it
    if not trimesh.ray.has_embree:
        with pytest.raises(ImportError):
            m.contains(np.zeros((1, 3)))

    assert contains(m, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])).tolist() == [True, False]


def test_hollow_iou_without_rtree(monkeypatch):
    """End to end: the cavity case still answers when rtree is unavailable."""
    def no_rtree(self):
        raise ImportError("No module named 'rtree'")

    monkeypatch.setattr(trimesh.Trimesh, "triangles_tree", property(no_rtree))

    outer = box(extents=(2, 2, 2))
    inner = box(extents=(1, 1, 1))
    inner.invert()   # a cavity faces inwards; an outward one would be a body (DEFINITION 4.6)
    hollow = trimesh.util.concatenate([outer, inner])
    res = iou_res(hollow, box(extents=(2, 2, 2)))
    assert res, f"rejected: {res.status.value}: {res.detail}"
    # hollow volume 8 - 1 = 7 against a solid 8
    assert res.iou == pytest.approx(7.0 / 8.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# 13. the detector's verdict does not depend on triangle size or mesh scale
# --------------------------------------------------------------------------- #


def _piercing_pair(scale):
    """Two triangles of edge ~`scale` that cross each other transversally."""
    t1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    t2 = np.array([[0.25, 0.25, -0.5], [0.75, 0.25, 0.5], [0.25, 0.75, 0.5]], dtype=float)
    v = np.vstack([t1, t2]) * scale
    return trimesh.Trimesh(v, np.array([[0, 1, 2], [3, 4, 5]]), process=False)


@pytest.mark.parametrize("scale", [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
def test_self_intersection_detected_at_any_triangle_size(scale):
    """
    Regression from the numpy detector (up to v0.2.0), whose tolerance once
    became a per-triangle threshold: below edge ~1e-3 -- ordinary for a
    tessellated CAD part in a unit cube -- this obvious crossing went
    unreported.  A missed defect is worse than a refusal: the module then
    computes an IoU from a surface that passes through itself.
    """
    hit, n = is_self_intersecting(_piercing_pair(scale))
    assert hit is True and n == 1


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_self_intersection_verdict_is_scale_invariant(scale):
    """
    The same mesh, rescaled, must get the same verdict and the same count.

    `diagnose_mesh` runs the detector on the RAW mesh, and real targets came in
    at scales from 1.0 to 238 -- an absolute tolerance means something different
    to each of them.
    """
    pierced = trimesh.util.concatenate([box(), box(translate=(0.5, 0, 0))])

    base_hit, base_n = is_self_intersecting(pierced)
    scaled = pierced.copy()
    scaled.apply_scale(scale)
    assert is_self_intersecting(scaled) == (base_hit, base_n)
    assert base_hit is True and base_n > 0
