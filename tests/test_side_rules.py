"""
Sides read by the cells of space they cut (docs/DEFINITION.md 4.8, 4.9):

- closed shells written twice -- the solid is every cell they enclose; it is
  refused when enclosed cells that may be a cavity (a hollow body and a body
  inside a body read the same) hold more than 1e-6 of the solid, unless the
  side's cavities are filled;
- a split into open pieces (edges of 3+ faces the package cannot pair) -- by
  the raw surface with its cracks closed: balanced, its w >= 1 in either mode
  (in "fail" refused where it winds 2 or more); unbalanced (a one-sided wall
  inside a closed surface), refused, unless the side's cavities are filled --
  then every enclosed cell;

and the holes (4.3): every hole, a crack as much as a larger one, is closed
only by a patch whose proven volume error is at most 1e-3 of the parts it
closes -- an open box is refused, beside a large part too; a seam of
two open parts (tessellated alike or not), a slit and a junk sheet are closed;

and the shells (4.5, 4.6):

- a shell is dropped as zero-volume only once its winding is agreed and only
  when it is empty -- a shell through itself whose +V and -V cancel is kept;
- orientation is decided per nesting group: another body does not change how
  a group is read;

and the detector's pairs in "fail" (4.7): a body with pairs is built in CGAL,
a touch inside it is a number, a pass through itself (w >= 2) is refused;

and a crack whose points lie on one line (4.3) -- a T-junction (a vertex on
the edge of the neighbouring face) or the hole a sliver face leaves when
cleaning drops it: the fan that closes it has no area, the winding number is
that of the raw surface, and the touch at the junction sends the body to CGAL
-- a number in either mode; the fan that closes a loop starts where its area
is least, so neither the patch nor the crack budget depends on where the
vertex numbering starts the loop, and it does not fold over itself;

and an inverted pocket (w <= -1, refused, 5.3) with the side's cavities
filled (5.5): not refused when it lies in a cavity of the solid -- a cavity of
one body or one closed only by two overlapping bodies -- since both readings
fill to one solid; refused when its void reaches the outside across a face
or only along an edge (the filling would leave that void open).

The expected solid comes from the definition, not from the code: the bodies
each record was written from, by the generalized winding number of each body
(numpy sum of solid angles, `winding.winding_number`) at seeded points, and
their known volumes.
"""

import numpy as np
import pytest
import trimesh

from shapes import coiled_bar, via_stl
from test_cells import cube_with_wall
from volumetric_iou import mesh_iou, seams
from volumetric_iou.mesh_iou import Status, normalize_pair, volume_iou
from volumetric_iou.winding import winding_number


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


def soup(*parts):
    """One mesh of the given meshes and (vertices, faces) pairs, in turn."""
    V, F, n = [], [], 0
    for p in parts:
        v, f = (p.vertices, p.faces) if isinstance(p, trimesh.Trimesh) else p
        V.append(np.asarray(v, np.float64))
        F.append(np.asarray(f, np.int64) + n)
        n += len(v)
    return trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)


def twice(m, times=1):
    """m written both ways, `times` times: winding 0 everywhere."""
    return [(m.vertices, m.faces), (m.vertices, m.faces[:, ::-1])] * times


def prepare(side, mode="exact", fill=False):
    return mesh_iou._prepare(side, "gt" if mode == "exact" else "pred",
                             check_self_intersection=True, on_self_intersection=mode,
                             fill_cavities=fill)


def assert_solid(prep, bodies, volume, n=600):
    """The built solid is the union of `bodies` (each a closed one-sided
    surface): its volume, and inside/outside at seeded points by the winding
    number of each body."""
    assert prep.volume == pytest.approx(volume, rel=1e-9)
    lo = np.min([b.bounds[0] for b in bodies], axis=0) - 0.1
    hi = np.max([b.bounds[1] for b in bodies], axis=0) + 0.1
    pts = lo + np.random.default_rng(0).random((n, 3)) * (hi - lo)
    want = np.zeros(n, bool)
    for b in bodies:
        want |= winding_number(b, pts) > 0.5
    got = winding_number(prep.mesh, pts) > 0.5
    assert np.array_equal(want, got)


CUBE = box([-0.5] * 3, [0.5] * 3)
MODES = ["exact", "fail"]


# --------------------------------------------------------------------------- #
# closed shells written twice (DEFINITION 4.8)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_cube_written_k_times_both_ways_is_the_cube(k, mode):
    # k = 2: every face is a "contact" for the parity reading -- the cube was
    # dropped (empty_mesh); the cell it encloses borders the outer cell
    # through faces of k = 2 > 0: filled under any reading (fact 4b)
    prep, err = prepare(soup(*twice(CUBE, k)), mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    d = prep.info["doubled_shells"]
    assert d["closed"] == 1 and d["cells"] == 1 and d["ambiguous_cells"] == 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_cracked_cube_written_k_times_both_ways_is_the_cube(k, mode):
    # one triangle missing from every copy: the copy's patch goes into the
    # cells as many times as the faces around it -- written once each way at
    # k = 2 it disagreed with them by parity and the cube was refused
    # (degenerate_volume, one ambiguous cell)
    fine = CUBE
    for _ in range(4):
        fine = fine.subdivide()
    cracked = (fine.vertices, fine.faces[1:])
    prep, err = prepare(soup(*[cracked, (fine.vertices, fine.faces[1:, ::-1])] * k), mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    d = prep.info["doubled_shells"]
    assert d["closed"] == 1 and d["ambiguous_cells"] == 0 and d["loop_faces_added"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_body_written_twice_beside_one_written_once(mode):
    # h54 (a): A both ways, B twice both ways -- B was taken for a flat sheet
    a, b = box([0, 0, 0], [1, 1, 1]), box([2, 0, 0], [3, 1, 1])
    prep, err = prepare(soup(*twice(a), *twice(b, 2)), mode)
    assert err is None
    assert_solid(prep, [a, b], 2.0)
    assert prep.info["doubled_shells"]["sheets"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_body_written_twice_on_a_body_written_once(mode):
    # h54 (b): C on D, C twice both ways (k = 3 on the contact, 2 on C) --
    # C's faces were dropped as contacts and only D was left
    d, c = box([0, 0, 0], [1, 1, 1]), box([0, 0, 1], [1, 1, 2])
    prep, err = prepare(soup(*twice(d), *twice(c, 2)), mode)
    assert err is None
    assert_solid(prep, [c, d], 2.0)


@pytest.mark.parametrize("mode", MODES)
def test_nested_bodies_sharing_a_face_both_ways(mode):
    # A inside B, their bottoms on one plane, both written both ways: the
    # cell of A borders the outer cell through the shared bottom (k = 2)
    b, a = box([0, 0, 0], [2, 2, 2]), box([0.5, 0.5, 0], [1.5, 1.5, 1])
    prep, err = prepare(soup(*twice(a), *twice(b)), mode)
    assert err is None
    assert_solid(prep, [b], 8.0)


@pytest.mark.parametrize("shift", [0.0, 1e-12])
def test_copy_shifted_below_the_grid_reads_the_same(shift):
    # equality of vertices by cleaning's grid: a copy shifted by noise in
    # the 12th digit is the same face -- k = 2 either way
    moved = CUBE.copy()
    moved.vertices += shift
    prep, err = prepare(soup(*twice(CUBE), *twice(moved)))
    assert err is None
    assert_solid(prep, [CUBE], 1.0)


def hollow():
    outer, inner = box([-1] * 3, [1] * 3), box([-0.5] * 3, [0.5] * 3)
    return outer, inner


@pytest.mark.parametrize("mode", MODES)
def test_reversed_copy_shifted_below_the_grid_is_unfolded(mode):
    # the copy written back with noise in the 12th digit is the same copy:
    # unfolded by cleaning's grid, the hollow box is refused as when written
    # exactly -- merged exactly, cleaning kept one copy and read it solid
    outer, inner = hollow()
    side = soup(*[(m.vertices + d, f) for m in (outer, inner)
                  for d, f in ((0.0, m.faces), (1e-12, m.faces[:, ::-1]))])
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    assert err.diagnostics["doubled_shells"]["ambiguous_cells"] == 1
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert_solid(prep, [outer], 8.0)


@pytest.mark.parametrize("mode", MODES)
def test_hollow_box_both_ways_is_refused_unless_filled(mode):
    # fact 2: a hollow box and a box with a box inside it are written alike
    outer, inner = hollow()
    side = soup(*twice(outer), *twice(inner))
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    assert "may be a cavity" in err.detail
    assert err.diagnostics["doubled_shells"]["ambiguous_cells"] == 1
    assert "raw_solid" not in err.diagnostics
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert_solid(prep, [outer], 8.0)


def test_hollow_box_both_ways_through_volume_iou():
    outer, _ = hollow()
    side = soup(*twice(outer), *twice(hollow()[1]))
    g, p = normalize_pair(outer, side, pred_frame="gt")
    res = volume_iou(g, p)
    assert res.status is Status.DEGENERATE_VOLUME and res.blame == "pred"
    res = volume_iou(g, p, fill_cavities=True)
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert any("may be cavities; taken as solid" in w for w in res.warnings)


@pytest.mark.parametrize("mode", MODES)
def test_overlapping_boxes_both_ways_are_refused_unless_filled(mode):
    # A and B overlap: their common part is a cell of even parity whose
    # neighbours are A \ B and B \ A at k-distance 1 = k -- the record of A, B
    # is also the record of A xor B (fact 4 holds for neither reading)
    a, b = box([0, 0, 0], [2, 2, 2]), box([1, 1, 1], [3, 3, 3])
    single = box([5, 0, 0], [6, 1, 1])
    side = soup(*twice(a), *twice(b), single)
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    assert err.diagnostics["doubled_shells"]["ambiguous_cells"] == 1
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert_solid(prep, [a, b, single], 8 + 8 - 1 + 1)


@pytest.mark.parametrize("mode", MODES)
def test_surface_of_the_cut_is_built_in_cgal_beside_the_side(mode):
    # the cells' surface with vertices of the exact cut (rounded: cleaning it
    # again would drop slivers and open it -- seen on two-sided CAD parts)
    # goes to CGAL as it is, one more body, united with the side's bodies
    rest = box([0, 0, 0], [1, 1, 1])
    cut = box([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    prep, err = mesh_iou._prepare_shells(rest, "gt", check_self_intersection=True,
                                         on_self_intersection=mode, cut=cut)
    assert err is None and prep.info["resolved"] and prep.info["solids"] == 2
    assert_solid(prep, [rest, cut], 2 - 0.125)
    prep, err = mesh_iou._prepare_shells(trimesh.Trimesh(np.zeros((0, 3)), np.zeros((0, 3), int)),
                                         "gt", check_self_intersection=True,
                                         on_self_intersection=mode, cut=cut)
    assert err is None
    assert_solid(prep, [cut], 1.0)


@pytest.mark.parametrize("check, says", [(True, "(0 crossing pairs)"), (False, "(detector off)")])
def test_cut_in_cgal_warning_tells_detector_off_from_failed(check, says):
    # with the detector off the cut's body goes to CGAL with no pair count
    # (crossing_pairs None, as when the detector fails): the warning says off
    a, b = box([0, 0, 0], [1, 1, 1]), box([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    res = volume_iou(*normalize_pair(soup(*twice(a), *twice(b)), box([0, 0, 0], [1.5, 1.5, 1.5]),
                                     pred_frame="gt"),
                     check_self_intersection=check, fill_cavities=True)
    assert res and res.iou == pytest.approx(15 / 27, rel=1e-9)
    got = [w for w in res.warnings if "resolved by CGAL arrangement" in w]
    assert got == [f"gt: surface resolved by CGAL arrangement {says}"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size, taken", [(0.01, True), (0.03, False)])
def test_cell_that_may_be_a_cavity_is_solid_when_tiny(size, taken, mode):
    # the inner cell may be a cavity: 1e-6 of the solid (8) or less is read
    # as solid -- the volume off by at most that; more is refused
    outer, inner = box([-1] * 3, [1] * 3), box([-size / 2] * 3, [size / 2] * 3)
    side = soup(*twice(outer), *twice(inner))
    prep, err = prepare(side, mode)
    if taken:
        assert err is None
        assert_solid(prep, [outer], 8.0)
        d = prep.info["doubled_shells"]
        assert d["ambiguous_accepted"] and d["ambiguous_cells"] == 1
        assert d["ambiguous_volume"] == pytest.approx(size ** 3, rel=1e-9)
    else:
        assert err is not None and err.status is Status.DEGENERATE_VOLUME
        assert "ambiguous_accepted" not in err.diagnostics["doubled_shells"]


@pytest.mark.parametrize("mode", MODES)
def test_hollow_sphere_both_ways_is_refused_unless_filled(mode):
    outer, inner = trimesh.creation.icosphere(2, 1.0), trimesh.creation.icosphere(2, 0.5)
    side = soup(*twice(outer), *twice(inner))
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert prep.volume == pytest.approx(outer.volume, rel=1e-9)


def test_tiny_cell_that_may_be_a_cavity_through_volume_iou():
    outer, inner = box([-0.5] * 3, [0.5] * 3), box([-0.002] * 3, [0.002] * 3)
    res = volume_iou(outer, soup(*twice(outer), *twice(inner)))
    assert res and res.iou == pytest.approx(1.0, abs=1e-12)
    assert any("within 1e-06 of the solid" in w for w in res.warnings)


@pytest.mark.parametrize("mode", MODES)
def test_inner_box_written_twice_inside_a_box_written_once_is_solid(mode):
    # outer k = 1, inner k = 2: the inner cell is at odd parity (1 + 2) --
    # inside some body under any reading (fact 4a)
    outer, inner = hollow()
    prep, err = prepare(soup(*twice(outer), *twice(inner, 2)), mode)
    assert err is None
    assert_solid(prep, [outer], 8.0)


# --------------------------------------------------------------------------- #
# a split into open pieces (DEFINITION 4.4)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split_mid", [False, True])
def test_one_sided_wall_is_refused_unless_filled(split_mid, mode):
    # winding 1.23 / 0.77 on the two sides of the wall: a partition (a cube)
    # and the boundary of a void (half a cube) read the same.  trimesh's
    # fill_holes closed the 4-edge holes of the plain wall (ok, a cube) and
    # not the 5-edge one (not_manifold)
    side = trimesh.Trimesh(*cube_with_wall(split_mid), process=False)
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.NOT_MANIFOLD
    assert "unbalanced" in err.detail and "raw_solid" not in err.diagnostics
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    assert prep.info["cells_filled"]["cells"] == 2


def test_one_sided_wall_filled_through_volume_iou():
    side = trimesh.Trimesh(*cube_with_wall(False), process=False)
    res = volume_iou(CUBE, side, fill_cavities=(False, True))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert res.method == "cgal:manifold64"
    assert any("every enclosed cell taken as solid" in w for w in res.warnings)
    res = volume_iou(CUBE, side)
    assert res.status is Status.NOT_MANIFOLD and res.blame == "pred"


@pytest.mark.parametrize("mode", MODES)
def test_wall_beside_touching_bodies_with_a_crack_is_filled(mode):
    # cleaning keeps one copy of the contact of b and c, and its patch of
    # c's crack closes no chain (refused, no loop closed); the raw surface's
    # is taken.  The cells were cut with cleaning's patches: the crack open,
    # c's cell broke CGAL (boolean_failed) or joined the outer one
    b = box([2, -.5, -.5], [3, .5, .5])
    for _ in range(4):
        b = b.subdivide()
    c = trimesh.Trimesh(b.vertices * [-1, 1, 1] + [6, 0, 0], b.faces[:, ::-1], process=False)
    far = np.flatnonzero(np.isclose(c.triangles_center[:, 0], 4.0))[0]
    side = soup(cube_with_wall(False), b, (c.vertices, np.delete(c.faces, far, axis=0)))
    prep, err = prepare(side, mode, fill=True)
    assert err is None
    assert_solid(prep, [CUBE, b, c], 3.0)
    assert prep.info["cells_filled"]["patch_faces"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_wall_with_a_crossing_body_is_filled(mode):
    # the pin crosses the walled cube, a component of its own: an overlap of
    # parts -- every bounded cell solid in either mode (in "fail" any pair of
    # the whole surface refused it)
    V, F = cube_with_wall(False)
    pin = box([-0.1, -0.1, 0.3], [0.1, 0.1, 0.8])
    prep, err = prepare(soup((V, F), pin), mode, fill=True)
    assert err is None
    assert_solid(prep, [CUBE, pin], 1.0 + 0.2 * 0.2 * 0.3)


def test_wall_beside_a_part_through_itself_in_fail_is_refused_when_filled():
    # no winding number to find a pass through itself: a crossing pair inside
    # one component (the coil) refuses the filled side in "fail"
    V, F = cube_with_wall(False)
    coil = coiled_bar()
    coil.apply_translation([4.0, 0.0, 0.0])
    _, err = prepare(soup((V, F), coil), "fail", fill=True)
    assert err is not None and err.status is Status.SELF_INTERSECTING
    assert "inside one part" in err.detail


def test_inverted_cube_with_a_wall_is_inverted_when_filled():
    V, F = cube_with_wall(False)
    side = trimesh.Trimesh(V, np.r_[F[:12, ::-1], F[12:]], process=False)
    _, err = prepare(side, fill=True)
    assert err is not None and err.status is Status.INVERTED_VOLUME


def touching_halves():
    """Two half-cubes touching on x = 0 with the same triangulation of the
    contact: one copy each way, every edge balanced, w = 1 on both sides."""
    a = box([-0.5, -0.5, -0.5], [0.0, 0.5, 0.5])
    b = a.copy()
    b.vertices[:, 0] *= -1
    b.invert()
    return a, b


@pytest.mark.parametrize("mode", MODES)
def test_touching_alike_is_built_from_the_raw_winding(mode):
    # trimesh's fill_holes put two faces on each half; now the raw surface
    a, b = touching_halves()
    prep, err = prepare(soup(a, b), mode)
    assert err is None
    assert_solid(prep, [a, b], 1.0)
    raw = prep.info["raw_solid"]
    assert raw["rule"] == "split_open" and raw["refused_as"] == "not_manifold"


def spy_parts(monkeypatch):
    """Counts of raw bodies passing through themselves under the checked reading."""
    from volumetric_iou import raw_solid as R
    got, orig = [], R.resolve

    def spy(raw, check=False, fill=False):
        try:
            res = orig(raw, check=check, fill=fill)
        except R.InvalidReading as err:
            if check and err.kind == "overlap":
                got.append(err.diagnostics["self_overlap_parts"])
            raise
        if check:
            got.append(res[2]["self_overlap_parts"])
        return res
    monkeypatch.setattr(R, "resolve", spy)
    return got


@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_touching_alike_overlapped_by_a_third_body(mode, stl, tmp_path, monkeypatch):
    # w = 2 where the third body overlaps a half: two components winding once
    # each -- an overlap of parts, their union in either mode (in "fail" the
    # sum alone refused it as a pass through itself); from an STL file (no
    # vertex shared; coordinates exact in float32) as well
    a, b = touching_halves()
    c = box([0.25, -0.25, -0.25], [0.75, 0.25, 0.25])
    side = via_stl(soup(a, b, c), tmp_path / "s.stl") if stl else soup(a, b, c)
    parts = spy_parts(monkeypatch)
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [a, b, c], 1.0 + 0.25 * 0.5 * 0.5)
    if mode == "fail":
        assert prep.info["raw_solid"]["winding_max"] == 2 and parts == [0]


@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_touching_alike_cut_across_the_contact_by_a_third_body(mode, stl, tmp_path, monkeypatch):
    # the third body inside both halves, its faces cutting the contact (and
    # the diagonal of its triangulation): the two copies of the contact are
    # cut apart differently, and between them lie cells of zero volume where
    # the halves' component winds 2 -- no volume, not a pass through itself:
    # the union in either mode, from an STL file as well
    a, b = touching_halves()
    c = box([-0.25] * 3, [0.25] * 3)
    side = via_stl(soup(a, b, c), tmp_path / "s.stl") if stl else soup(a, b, c)
    parts = spy_parts(monkeypatch)
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [a, b, c], 1.0)
    assert prep.info["raw_solid"]["rule"] == "split_open"
    if mode == "fail":
        assert prep.info["raw_solid"]["winding_max"] == 2 and parts == [0]


@pytest.mark.parametrize("stl", [False, True])
def test_touching_alike_beside_a_part_through_itself(stl, tmp_path, monkeypatch):
    # the coil is one component winding 2 by itself: refused in "fail" on the
    # raw surface as on shells; "exact" builds w >= 1.  From an STL file (three
    # vertices per face) the components are joined at equal coordinates: the
    # coil is found as when its vertices are shared
    a, b = touching_halves()
    coil = coiled_bar()
    coil.apply_translation([4.0, 0.0, 0.0])
    side = via_stl(soup(a, b, coil), tmp_path / "s.stl") if stl else soup(a, b, coil)
    parts = spy_parts(monkeypatch)
    _, err = prepare(side, "fail")
    assert parts == [1]
    assert err is not None and err.status is Status.SELF_INTERSECTING
    assert err.diagnostics["raw_winding_max"] == 2
    assert err.diagnostics["self_overlap_parts"] == 1
    prep, err = prepare(side, "exact")
    assert err is None and prep.info["raw_solid"]["rule"] == "split_open"


def test_raw_surface_pocket_is_found_before_a_part_through_itself():
    # a coil through itself and a bowtie's pocket on the raw surface: the
    # pocket is refused first, as on the bodies of shells (5.3)
    a, b = touching_halves()
    coil, tie = coiled_bar(), bowtie()
    coil.apply_translation([4.0, 0.0, 0.0])
    tie.apply_translation([0.0, 4.0, 0.0])
    _, err = prepare(soup(a, b, coil, tie), "fail")
    assert err is not None and err.status is Status.INVERTED_VOLUME
    _, err = prepare(soup(a, b, coil), "fail")
    assert err is not None and err.status is Status.SELF_INTERSECTING


def without_a_face(m, pick=lambda m, i: True):
    """m less its first face with every edge on exactly two faces (a hole
    with nothing else at its edges) that `pick` accepts."""
    e = np.sort(m.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    two = (cnt[inv.reshape(-1)] == 2).reshape(-1, 3).all(axis=1)
    i = next(i for i in np.flatnonzero(two) if pick(m, i))
    return trimesh.Trimesh(m.vertices, np.delete(m.faces, i, axis=0), process=False)


@pytest.mark.parametrize("mode", MODES)
def subdivided(m, times):
    v, f = m.vertices, m.faces
    for _ in range(times):
        v, f = trimesh.remesh.subdivide(v, f)
    return trimesh.Trimesh(v, f, process=False)


@pytest.mark.parametrize("mode", MODES)
def test_touching_alike_with_a_crack_is_built_from_the_patched_raw_surface(mode):
    # a hole in a half: its edges were unbalanced on the raw surface, and the
    # side was refused as a wall (not_manifold); cleaning keeps one copy of the
    # contact, so its patch closes no chain there -- the raw surface decides:
    # closed as cleaning closes it, every edge is balanced, w >= 1.  A hole of
    # 1/32 of a face is a guess of 0.014 of the side -- refused
    for times, ok in ((2, False), (4, True)):
        a, b = (subdivided(h, times) for h in touching_halves())
        side = soup(without_a_face(a, lambda m, i: m.face_normals[i, 1] < -0.5), b)
        assert mesh_iou._unbalanced_edges(side) == 3
        prep, err = prepare(side, mode)
        if not ok:
            assert err is not None and err.status is Status.OPEN_SURFACE
            assert not err.diagnostics["raw_holes"]["accepted"]
            continue
        assert err is None
        assert_solid(prep, [a, b], 1.0)
        raw = prep.info["raw_solid"]
        assert raw["rule"] == "split_open" and raw["loop_faces_added"] == 1
        assert prep.info["faces_raw"] == len(side.faces)


def nearly_touching_open_boxes(hole=0.004):
    """Two separate triangular holes whose nearest vertices merge on the grid."""
    originals, open_parts = [], []
    for lo, hi, corner in (([-0.4]*3, [0.0]*3, 0.0),
                           ([4e-10]*3, [0.4000000004]*3, 4e-10)):
        part = box(lo, hi).subdivide()
        v = part.vertices.copy()
        for axis in range(3):
            middle = (lo[axis] + hi[axis]) / 2
            near = corner + (hole if corner == lo[axis] else -hole)
            v[v[:, axis] == middle, axis] = near
        part.vertices = v
        k = int(np.argmin(np.linalg.norm(v - corner, axis=1)))
        face = next(i for i, f in enumerate(part.faces) if k in f)
        originals.append(part)
        open_parts.append(trimesh.Trimesh(v, np.delete(part.faces, face, axis=0), process=False))
    return soup(*open_parts), soup(*originals)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_raw_fallback_patches_holes_before_checking_balance(mode, stl, fill, exact, tmp_path):
    # The grid merges the nearest corners of these two separate holes. Their
    # boundary branches, so cleaning cannot find loops and refuses without a
    # holes record or split_open. Exact merging on the raw surface preserves
    # two simple loops; their patches restore precisely the missing triangles.
    source, reference = nearly_touching_open_boxes()
    if stl:
        source = via_stl(source, tmp_path / "open.stl")
        reference = via_stl(reference, tmp_path / "reference.stl")
    opts = dict(check_self_intersection=True, on_self_intersection=mode, fill_cavities=fill)
    p, err = mesh_iou._prepare_shells(source, "gt", **opts)
    assert p is None and err.status is Status.OPEN_SURFACE
    assert "holes" not in err.diagnostics and not err.diagnostics.get("split_open")
    assert mesh_iou._unbalanced_edges(source) == 6
    p, err = mesh_iou._prepare(source, "gt", exact_booleans=exact, **opts)
    assert err is None
    assert_solid(p, [reference], reference.volume)
    raw = p.info["raw_solid"]
    assert "rule" not in raw                     # the general fallback, not split_open
    assert raw["refused_as"] == "open_surface" and raw["loop_faces_added"] == 2
    assert raw["holes"]["accepted"] and raw["holes"]["ratio"] < 1e-3
    assert p.info["faces_raw"] == len(source.faces)
    for gt, pred in ((source, reference), (reference, source)):
        res = volume_iou(gt, pred, on_self_intersection=(mode, mode),
                         fill_cavities=fill, exact_booleans=exact)
        assert res.status is Status.OK and res.iou == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fill", [False, True])
def test_raw_fallback_does_not_build_a_solid_when_its_patch_is_rejected(mode, fill, monkeypatch):
    source, _ = nearly_touching_open_boxes(hole=0.2)
    label = "gt" if mode == "exact" else "pred"
    _, err = mesh_iou._prepare_shells(source, label, check_self_intersection=True,
                                      on_self_intersection=mode, fill_cavities=fill)
    assert err.status is Status.OPEN_SURFACE and "holes" not in err.diagnostics
    _, patch = mesh_iou._raw_patched(source)
    assert not patch["holes"]["accepted"] and patch["holes"]["closed"]

    def no_solid(*args, **kwargs):
        pytest.fail("a rejected raw patch must not reach solid construction")

    monkeypatch.setattr(mesh_iou, "_raw_solid", no_solid)
    p, refused = prepare(source, mode, fill=fill)
    assert p is None and refused.status is err.status and refused.detail == err.detail
    assert refused.diagnostics == err.diagnostics


@pytest.mark.parametrize("mode", MODES)
def test_patched_raw_fallback_body_preserves_its_checks(mode, tmp_path):
    from volumetric_iou.body import Body, load_body, save_body, to_body

    source, reference = nearly_touching_open_boxes()
    body = to_body(source, frame="gt", gt=reference, on_self_intersection=mode)
    assert isinstance(body, Body)
    save_body(body, tmp_path / "patched.body.ply")
    loaded = load_body(tmp_path / "patched.body.ply")
    raw = loaded.meta["info"]["raw_solid"]
    assert raw["loop_faces_added"] == 2 and raw["holes"]["accepted"]
    gt, pred = normalize_pair(reference, loaded, pred_frame="gt")
    same = volume_iou(gt, pred, on_self_intersection=(mode, mode))
    assert same.status is Status.OK and same.iou == pytest.approx(1.0, rel=1e-12)
    if mode == "exact":
        checked = volume_iou(gt, pred, on_self_intersection=(mode, "fail"))
        assert checked.status is Status.OK and checked.iou == pytest.approx(1.0, rel=1e-12)
        assert raw["assembly"]["fail_validity"] == "valid"


@pytest.mark.parametrize("mode", MODES)
def test_wall_with_a_crack_is_refused_filled_or_not(mode):
    # the wall's edges of 3 faces stay unbalanced: the patch of the crack
    # closes no chain, its error is not bounded -- open, filled too
    wall = trimesh.Trimesh(*cube_with_wall(False), process=False)
    side = without_a_face(subdivided(wall, 4))
    for fill in (False, True):
        _, err = prepare(side, mode, fill=fill)
        assert err is not None and err.status is Status.OPEN_SURFACE
        assert not err.diagnostics["raw_holes"]["closed"]


def assert_region(prep, surfaces, volume, n=600, minus=()):
    """The built solid is w >= 1 of the sum of the windings of `surfaces`
    (each closed and consistently wound as given), less what is inside any
    of `minus`: its volume and inside/outside at seeded points."""
    assert prep.volume == pytest.approx(volume, rel=1e-9)
    lo = np.min([b.bounds[0] for b in surfaces], axis=0) - 0.1
    hi = np.max([b.bounds[1] for b in surfaces], axis=0) + 0.1
    pts = lo + np.random.default_rng(0).random((n, 3)) * (hi - lo)
    w = sum(winding_number(b, pts) for b in surfaces)
    want = w > 0.5
    for b in minus:
        want &= winding_number(b, pts) < 0.5
    got = winding_number(prep.mesh, pts) > 0.5
    assert np.array_equal(want, got)


# --------------------------------------------------------------------------- #
# zero-volume shells: dropped only once wound alike, and only when empty
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("flipped", [4, 6])
def test_cube_with_faces_against_the_rest_is_the_cube(flipped, mode):
    # 4 or 6 of 12 faces the other way: signed volume 0 before the winding is
    # agreed -- the cube was dropped (degenerate_volume); 2 flipped gave 1
    f = CUBE.faces
    side = trimesh.Trimesh(CUBE.vertices, np.r_[f[:flipped, ::-1], f[flipped:]], process=False)
    assert side.volume == pytest.approx(0.0, abs=1e-12) and not side.is_winding_consistent
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    assert prep.info["shells_dropped"] == 0 and prep.info["orientation"] == "parity"


@pytest.mark.parametrize("mode", MODES)
def test_faces_written_twice_one_way_once_the_other(mode):
    # every face up twice, down once (net +1: raw w = 1 inside); cleaning keeps
    # the first copy, the down one for half of the faces -- a shell of signed
    # volume 0 that was dropped
    f, rev = CUBE.faces, CUBE.faces[:, ::-1]
    side = soup((CUBE.vertices, np.r_[rev[:6], f[6:]]), CUBE,
                (CUBE.vertices, np.r_[f[:6], rev[6:]]))
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)


def flat_square(z, flip=False):
    """A closed flat shell: the unit square at height z, both sides, with
    different diagonals; `flip`: one face against the rest."""
    V = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    if flip:
        F[0] = F[0, ::-1]
    return trimesh.Trimesh(V, F, process=False)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("flip", [False, True])
def test_flat_shell_is_dropped_wound_alike_or_not(flip, mode):
    # winds round nothing whichever way its faces go: empty, dropped
    side = soup(box([2, 2, 2], [3, 3, 3]), flat_square(5.0, flip))
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [box([2, 2, 2], [3, 3, 3])], 1.0)
    assert prep.info["shells_dropped"] == 1 and "zero_volume_kept" not in prep.info


def bowtie():
    """A prism over a figure eight: consistently wound, signed
    volume 0, w = +1 round one lobe and -1 round the other (0.016 each)."""
    from shapes import extruded
    m = extruded([(0, 0), (1, 1), (1, 0), (0, 1)], 1.0, centre=(0.2, 0.55))
    m.vertices *= 0.4
    return m


def test_bowtie_winds_both_ways():
    # the premise, by the winding number alone
    b = bowtie()
    assert b.is_winding_consistent and abs(b.volume) < 1e-15
    w = winding_number(b, np.array([[0.36, 0.2, 0.2], [0.04, 0.2, 0.2]]))
    assert sorted(np.round(w).tolist()) == [-1.0, 1.0]


@pytest.mark.parametrize("alone", [True, False])
def test_bowtie_is_not_dropped(alone):
    # dropped as zero-volume: alone degenerate_volume, beside a cube ok with
    # the cube; by the definition the side winds -1 round a lobe -- no solid
    # (inverted_volume) for GT and pred alike (a pred with crossings is built
    # in CGAL as well, its pocket found there)
    side = bowtie() if alone else soup(bowtie(), box([2, 2, 2], [3, 3, 3]))
    for mode in ("exact", "fail"):
        _, err = prepare(side, mode)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert err.diagnostics["inverted_volume"] == pytest.approx(0.016, rel=1e-9)
        assert err.diagnostics["zero_volume_kept"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_bowtie_inside_a_body_is_a_pocket_of_its_own(mode):
    # in a box: the sum winds 2 round one lobe, 0 round the other -- the sum
    # alone gave the box with the w = -1 lobe cut out, a number; the bowtie
    # winds -1 round that lobe by itself, an inverted pocket wherever it lies.
    # With the cavities filled the lobe is a cavity of the box: both readings
    # fill to the box
    big = box([-0.2, -0.2, -0.2], [0.6, 0.6, 0.6])
    b = bowtie()
    _, err = prepare(soup(big, b), mode)
    assert err is not None and err.status is Status.INVERTED_VOLUME
    assert err.diagnostics["inverted_volume"] == pytest.approx(0.016, rel=1e-9)
    prep, err = prepare(soup(big, b), mode, fill=True)
    assert err is None and prep.info["inverted_cavity_volume"] == pytest.approx(0.016, rel=1e-9)
    prep, err = mesh_iou._fill_cavities(prep, "gt")
    assert err is None
    assert prep.volume == pytest.approx(0.8 ** 3, rel=1e-9)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("flip", [False, True])
def test_bowtie_across_a_body_keeps_its_orientation(flip, mode):
    # a box over one lobe only: its signed volume of 0 read as an inside-out
    # body standing on its own flipped the bowtie, and the two answers swapped.
    # The w = -1 lobe is a pocket of the bowtie inside the box or outside it;
    # filling does not take it in either: inside the box it meets the outside
    # along the waist of the eight, on the box's face
    b = inverted(bowtie()) if flip else bowtie()
    lo, hi = box([-0.2, -0.2, -0.2], [0.2, 0.6, 0.6]), box([0.2, -0.2, -0.2], [0.6, 0.6, 0.6])
    for over in (lo, hi):
        for fill in (False, True):
            _, err = prepare(soup(over, b), mode, fill=fill)
            assert err is not None and err.status is Status.INVERTED_VOLUME
            assert err.diagnostics["inverted_volume"] == pytest.approx(0.016, rel=1e-9)


def lobe():
    """The prism over the bowtie's w = -1 lobe, wound outwards: it fills the
    lobe, so the sum with the bowtie winds 0 there."""
    from shapes import extruded
    m = extruded([(0.5, 0.5), (1, 0), (1, 1)], 1.0, centre=(0.8, 0.5))
    m.vertices *= 0.4
    return m


@pytest.mark.parametrize("mode", MODES)
def test_body_in_a_neighbours_pocket_is_refused(mode):
    # the sum winds 1 round one lobe and 0 everywhere else, never -1: by the
    # sum alone, a number (the other lobe).  The bowtie winds -1 round the
    # body by itself -- a defective reading, including with filling (the
    # pocket lies out in the open). Other pairings give a valid body, so
    # strict validity makes this ambiguous. Inside a hollow box the pocket lies in its cavity:
    # with the cavities filled, the box
    side = soup(bowtie(), lobe())
    pts = np.array([[0.36, 0.2, 0.2], [0.04, 0.2, 0.2], [0.2, 0.2, 0.6]])
    assert np.round(winding_number(side, pts)).tolist() == [0.0, 1.0, 0.0]
    for fill in (False, True):
        _, err = prepare(side, mode, fill=fill)
        assert err is not None and err.status is Status.DEGENERATE_VOLUME
        assert err.diagnostics["valid_readings"] > 0 and err.diagnostics["invalid_readings"] > 0
    inside = soup(hollow_box(), *[(m.vertices + 1.0, m.faces) for m in (bowtie(), lobe())])
    _, err = prepare(inside, mode)
    assert err is not None and err.status is Status.DEGENERATE_VOLUME
    prep, err = filled(inside, mode)
    assert err is None
    assert_solid(prep, [box([0, 0, 0], [4, 4, 4])], 64.0)
    assert prep.info["inverted_cavity_volume"] == pytest.approx(0.016, rel=1e-9)


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_body_in_a_neighbours_pocket_through_volume_iou(mode, fill):
    g, p = normalize_pair(soup(bowtie(), lobe()), box([0, 0, 0], [0.4, 0.4, 0.4]),
                          pred_frame="gt")
    for res, side in ((volume_iou(g, p, on_self_intersection=mode, fill_cavities=fill), "gt"),
                      (volume_iou(p, g, on_self_intersection=mode, fill_cavities=fill), "pred")):
        assert res.status is Status.DEGENERATE_VOLUME and res.blame == side


# --------------------------------------------------------------------------- #
# orientation per nesting group
# --------------------------------------------------------------------------- #


def inverted(m):
    m = m.copy()
    m.invert()
    return m


def against(m):
    """m with two faces against the rest: not consistently wound."""
    m = m.copy()
    f = m.faces.copy()
    f[:2] = f[:2, ::-1]
    m.faces = f
    return m


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("other", [0.25, 0.4])
def test_hollow_box_inside_out_beside_a_body(other, mode):
    # the hollow box written inside out as a whole, beside a body A: the sum of
    # the side's volumes was positive with the bigger A, and the outer shell
    # alone was flipped -- the cavity read as a body (0.128 for 0.12)
    ho, hi = box([0, -0.2, -0.2], [0.4, 0.2, 0.2]), box([0.1, -0.1, -0.1], [0.3, 0.1, 0.1])
    a = box([-0.6, -other / 2, -other / 2], [-0.6 + other, other / 2, other / 2])
    side = soup(inverted(ho), hi, a)
    prep, err = prepare(side, mode)
    assert err is None
    assert_region(prep, [ho, a], 0.064 - 0.008 + other ** 3, minus=[hi])
    assert prep.info["cavities"] == 1 and prep.info["flipped_shells"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_body_in_a_body_stays_material_beside_a_shell_wound_against_itself(mode):
    # an insert inside a block (w = 2) and, elsewhere, a body with two faces
    # against the rest: parity was taken for the whole side and the insert
    # read as a cavity (0.0716 for 0.0796)
    block, insert = box([0, -0.2, -0.2], [0.4, 0.2, 0.2]), box([0.1, -0.1, -0.1], [0.3, 0.1, 0.1])
    a = box([-0.6, -0.125, -0.125], [-0.35, 0.125, 0.125])
    prep, err = prepare(soup(block, insert, against(a)), mode)
    assert err is None
    assert_region(prep, [block, insert, a], 0.064 + 0.25 ** 3)
    assert prep.info["cavities"] == 0
    assert prep.info["orientation"] == "parity" and prep.info["parity_groups"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_group_wound_against_itself_is_read_by_parity(mode):
    # the hollow box whose outer shell has faces against the rest: its own
    # group by parity (cavity at depth 1), the body beside it as given
    ho, hi = box([0, -0.2, -0.2], [0.4, 0.2, 0.2]), box([0.1, -0.1, -0.1], [0.3, 0.1, 0.1])
    a = box([-0.6, -0.125, -0.125], [-0.35, 0.125, 0.125])
    prep, err = prepare(soup(against(ho), inverted(hi), a), mode)
    assert err is None
    assert_region(prep, [ho, a], 0.064 - 0.008 + 0.25 ** 3, minus=[hi])
    assert prep.info["cavities"] == 1 and prep.info["parity_groups"] == 1


def test_crowded_edges_are_audited_even_after_successful_splitting(monkeypatch):
    # Ordinary and separate closed shells need no raw reading. A successful
    # split at a crowded edge still needs the ambiguity audit.
    calls = []
    from volumetric_iou import raw_solid as R
    real = R.resolve
    monkeypatch.setattr(R, "resolve", lambda *a, **k: calls.append(1) or real(*a, **k))
    edge = soup(box([0, 0, 0], [1, 1, 1]), box([1, 1, 0], [2, 2, 1]))
    for side in (CUBE, edge, soup(box([0, 0, 0], [1, 1, 1]), box([3, 0, 0], [4, 1, 1]))):
        for mode in MODES:
            prep, err = prepare(side, mode)
            assert err is None
    assert len(calls) == 2
    a, b = touching_halves()
    prepare(soup(a, b), "fail")
    assert len(calls) == 3


@pytest.mark.parametrize("mode", MODES)
def test_edge_touch_with_one_body_inside_out(mode):
    # the faces at the shared edge do not alternate: the module does not
    # split, trimesh does; each cube is closed, the inside-out one is turned
    a, b = box([0, 0, 0], [1, 1, 1]), box([1, 1, 0], [2, 2, 1])
    b.invert()
    prep, err = prepare(soup(a, b), mode)
    assert err is None
    assert prep.volume == pytest.approx(2.0, rel=1e-12)
    assert prep.info["flipped_shells"] == 1


# --------------------------------------------------------------------------- #
# a touch inside one body in "fail": a number; a pass through itself: refused
# --------------------------------------------------------------------------- #


def split_ring(overrun=0.0, n=48):
    """A ring of rectangular section r in [1, 2], z in [0, 1], swept from
    angle 0 to 2 pi + overrun as one closed shell, its section shrinking
    along the sweep (by 0.2 on each side at the end).  overrun 0: the end
    cap lies inside the start cap -- the ring touches itself face on face,
    with no shared vertex; overrun > 0: its end runs on through its start
    (w = 2 there)."""
    rings = []
    for i in range(n + 1):
        t, s = (2 * np.pi + overrun) * i / n, 0.2 * i / n
        if i == n and not overrun:
            t = 0.0                             # the start's plane exactly
        rings.append([((1 + s) * np.cos(t), (1 + s) * np.sin(t), s),
                      ((2 - s) * np.cos(t), (2 - s) * np.sin(t), s),
                      ((2 - s) * np.cos(t), (2 - s) * np.sin(t), 1 - s),
                      ((1 + s) * np.cos(t), (1 + s) * np.sin(t), 1 - s)])
    v = np.asarray(rings, np.float64).reshape(-1, 3)
    f = []
    for i in range(n):
        for k in range(4):
            a, b = 4 * i + k, 4 * i + (k + 1) % 4
            f += [(a, b, 4 * (i + 1) + (k + 1) % 4), (a, 4 * (i + 1) + (k + 1) % 4, 4 * (i + 1) + k)]
    e = 4 * n
    f += [(0, 2, 1), (0, 3, 2), (e, e + 1, e + 2), (e, e + 2, e + 3)]
    m = trimesh.Trimesh(v, np.array(f), process=False)
    if m.volume < 0:
        m.invert()
    return m


def diamond_cavity_on_the_wall():
    """Box [-1, 1]^3 with a cavity of diamond section (diagonals 0.8, height
    0.6) whose vertical edge lies on the wall x = 1: a touch along a line,
    no shared vertex.  The solid: 8 - 0.192."""
    d, h = 0.4, 0.3
    ring = [(1.0, 0.0), (1.0 - d, d), (1.0 - 2 * d, 0.0), (1.0 - d, -d)]
    cav = trimesh.convex.convex_hull([(x, y, z) for x, y in ring for z in (-h, h)])
    cav.invert()
    return soup(box([-1] * 3, [1] * 3), cav), cav


def test_touch_inside_one_body_premise():
    # by the winding number alone: the touching ring winds at most once, the
    # overrunning one twice inside its start; the detector sees pairs in both
    ring, over = split_ring(), split_ring(0.3)
    p = np.array([[1.5, 0.05, 0.5], [1.5, -0.05, 0.5], [-1.5, 0.0, 0.5]])
    assert np.round(winding_number(ring, p)).tolist() == [1.0, 1.0, 1.0]
    assert np.round(winding_number(over, p)).tolist() == [2.0, 1.0, 1.0]
    side, cav = diamond_cavity_on_the_wall()
    q = np.array([[0.6, 0.0, 0.0], [0.99, 0.2, 0.0], [1.01, 0.0, 0.0]])
    assert np.round(winding_number(side, q)).tolist() == [0.0, 1.0, 0.0]
    for m in (ring, over, side):
        assert len(mesh_iou.self_intersecting_pairs(m)) > 0
        assert m.is_winding_consistent and len(m.split(only_watertight=False)) <= 2


@pytest.mark.parametrize("mode", MODES)
def test_ring_touching_itself_is_a_number(mode):
    ring = split_ring()
    prep, err = prepare(ring, mode)
    assert err is None and prep.info["resolved"]["crossing_pairs"] > 0
    assert_solid(prep, [ring], ring.volume)
    if mode == "fail":
        assert prep.info["resolved"]["self_overlap_bodies"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_cavity_touching_the_wall_along_a_line_is_a_number(mode):
    side, cav = diamond_cavity_on_the_wall()
    prep, err = prepare(side, mode)
    assert err is None and prep.info["cavities"] == 1
    assert prep.volume == pytest.approx(8 - 0.192, rel=1e-9)
    pts = np.random.default_rng(0).uniform(-1.1, 1.1, (600, 3))
    want = winding_number(side, pts) > 0.5
    assert np.array_equal(winding_number(prep.mesh, pts) > 0.5, want)


def test_ring_through_itself_is_refused_in_fail_only():
    over = split_ring(0.3)
    _, err = prepare(over, "fail")
    assert err is not None and err.status is Status.SELF_INTERSECTING and err.blame == "pred"
    assert err.diagnostics["self_overlap_bodies"] == 1 and err.diagnostics["self_overlap_volume"] > 0
    prep, err = prepare(over, "exact")                  # GT: its w >= 1
    assert err is None
    pts = np.random.default_rng(0).uniform(-2.1, 2.1, (600, 3)) * [1, 1, 0.3] + [0, 0, 0.5]
    assert np.array_equal(winding_number(prep.mesh, pts) > 0.5, winding_number(over, pts) > 0.5)


def test_two_bodies_overlapping_with_a_touch_inside_one_are_a_number():
    # w = 2 where two bodies overlap is the union, not a pass through itself;
    # the touching ring alone has the pairs that send the side to CGAL
    ring, other = split_ring(), box([1.2, -0.3, 0.2], [2.5, 0.3, 0.8])
    prep, err = prepare(soup(ring, other), "fail")
    assert err is None and prep.info["resolved"]["self_overlap_bodies"] == 0
    assert_solid(prep, [ring, other], prep.volume)
    assert prep.volume > ring.volume


def test_pred_touching_itself_through_volume_iou():
    ring = split_ring()
    g, p = normalize_pair(ring, ring, pred_frame="gt")
    res = volume_iou(g, p)
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert res.diagnostics["pred"]["resolved"]["self_overlap_bodies"] == 0
    g, p = normalize_pair(ring, split_ring(0.3), pred_frame="gt")
    res = volume_iou(g, p)
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"


def test_body_of_a_pred_touching_itself():
    # built in "fail" (a pass through itself looked for): taken in "fail" as
    # its mesh is; built in "exact" (not looked for): refused in "fail"
    from volumetric_iou import to_body
    ring = split_ring()
    ref = volume_iou(*normalize_pair(ring, ring, pred_frame="gt"))
    pb = to_body(ring, frame="gt", gt=ring, label="pred", on_self_intersection="fail")
    res = volume_iou(*normalize_pair(ring, pb, pred_frame="gt"))
    assert res and res.iou == pytest.approx(ref.iou, rel=1e-12)
    pb = to_body(ring, frame="gt", gt=ring, label="pred", on_self_intersection="exact")
    res = volume_iou(*normalize_pair(ring, pb, pred_frame="gt"))
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"


# --------------------------------------------------------------------------- #
# a crack whose points lie on one line
# --------------------------------------------------------------------------- #


def cube_with_t_junction(shift=0.0, sliver=False):
    """CUBE whose top face is triangulated through M, the midpoint of its
    front top edge A-B, while the front face keeps the whole edge A-B: the
    boundary loop A-M-B is a line.  shift moves M off the line within the
    face plane z = 0.5; sliver adds the face A-B-M that closes the loop (a
    sliver cleaning drops when M is that close to the line)."""
    V = np.asarray(CUBE.vertices, np.float64)
    F = np.asarray(CUBE.faces, np.int64)
    top = np.all(np.isclose(V[F][:, :, 2], 0.5), axis=1)

    def at(p):
        return int(np.flatnonzero(np.all(np.isclose(V, p), axis=1))[0])

    a, b, c, d = (at(p) for p in [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)])
    V = np.vstack([V, [0.0, -0.5 + shift, 0.5]])
    m = len(V) - 1
    F = np.vstack([F[~top], [[a, m, d], [m, b, c], [m, c, d]]])
    if sliver:
        F = np.vstack([F, [[b, a, m]] if any(
            tuple(F[i, [j, (j + 1) % 3]]) == (a, b) for i in range(len(F)) for j in range(3))
            else [[a, b, m]]])
    return trimesh.Trimesh(V, F, process=False)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shift, sliver", [(0.0, False), (0.0, True), (-5e-9, False),
                                           (-5e-9, True), (-2.0 ** -53, False)])
def test_crack_on_one_line_is_closed_without_area(shift, sliver, mode):
    side = cube_with_t_junction(shift, sliver)
    if sliver:                          # the raw surface is closed: every edge twice
        assert side.is_watertight and side.is_winding_consistent
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    assert prep.info["loop_faces_added"] == 1
    assert prep.info["patched_area_frac"] < 1e-8
    assert prep.info["degenerate_removed"] == int(sliver)
    if shift == 0.0:
        # M exactly on A-B: a T-junction -- the edge is cut at M, the fan face
        # on one line is dropped, no pair is left: manifold
        assert "resolved" not in prep.info and prep.info["crossing_pairs_internal"] == 0
        assert prep.info["t_junctions_split"] == 1
        assert prep.info["collinear_faces_removed"] == 1
        return
    # M off the line, by a rounding unit too: no T-junction exactly, nothing
    # is cut; the near touch sends the body to CGAL, no pass through itself
    assert prep.info["t_junctions_split"] == prep.info["collinear_faces_removed"] == 0
    assert prep.info["resolved"]["crossing_pairs"] > 0
    if mode == "fail":
        assert prep.info["resolved"]["self_overlap_bodies"] == 0


def test_crack_on_one_line_through_volume_iou():
    res = volume_iou(*normalize_pair(CUBE, cube_with_t_junction(), pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert res.diagnostics["pred"]["t_junctions_split"] == 1
    res = volume_iou(*normalize_pair(cube_with_t_junction(), CUBE, pred_frame="gt"))
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert res.diagnostics["gt"]["t_junctions_split"] == 1


# --------------------------------------------------------------------------- #
# a T-junction split exactly (same surface, same w): manifold, unless a real
# touch or crossing is left
# --------------------------------------------------------------------------- #


def with_t_junction(m, a, b, k=1):
    """m with the face holding the directed edge a -> b (points) split at k
    points of that edge, its neighbour across the edge left whole: a crack
    on one line, k + 2 points, closed by k fan faces on that line.  The edge
    must be along an axis, so the points lie on it exactly."""
    V = np.asarray(m.vertices, np.float64)
    F = np.asarray(m.faces, np.int64)
    ia, ib = (int(np.flatnonzero(np.all(V == p, axis=1))[0]) for p in (a, b))
    f = next(i for i in range(len(F)) for j in range(3)
             if (F[i, j], F[i, (j + 1) % 3]) == (ia, ib))
    c = next(int(x) for x in F[f] if x not in (ia, ib))
    a, b = V[ia], V[ib]
    assert np.count_nonzero(a != b) == 1
    pts = [a + (b - a) * (i + 1) / (k + 1) for i in range(k)]
    idx = list(range(len(V), len(V) + k))
    chain = [ia] + idx + [ib]
    new = [(chain[i], chain[i + 1], c) for i in range(k + 1)]
    return trimesh.Trimesh(np.vstack([V, pts]),
                           np.vstack([np.delete(F, f, axis=0), new]), process=False)


def top_front(m):
    """The directed top front edge (y = min, z = max) as the faces of m hold it."""
    lo, hi = m.bounds
    a, b = (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2])
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    ia, ib = (int(np.flatnonzero(np.all(V == p, axis=1))[0]) for p in (a, b))
    held = any((F[i, j], F[i, (j + 1) % 3]) == (ia, ib) for i in range(len(F)) for j in range(3))
    return (a, b) if held else (b, a)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_t_junction_is_split_and_built_in_manifold(k, mode):
    side = with_t_junction(CUBE, *top_front(CUBE), k=k)
    prep, err = prepare(side, mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    i = prep.info
    assert i["loop_faces_added"] == k and i["collinear_faces_removed"] == k
    # a junction is a vertex on an edge: M on A-B, and on an edge between
    # two fan faces on the line (both dropped) when k > 1
    assert i["t_junctions_split"] >= k
    assert "resolved" not in i and i["crossing_pairs_internal"] == 0


def test_t_junction_split_keeps_the_surface():
    # the split shell, as a sum of oriented triangles, is the cleaned one:
    # balanced, every edge of two faces, the same w off the surface
    from volumetric_iou import tjunction
    side = with_t_junction(CUBE, *top_front(CUBE), k=2)
    m, _ = mesh_iou._clean(side)
    shells, *_ = mesh_iou._orient_shells(m, repair=False)
    assert len(shells) == 1
    before = shells[0].copy()
    pairs = mesh_iou.self_intersecting_pairs(before)
    assert len(pairs) > 0
    rec = {}
    assert tjunction.split_t_junctions(shells, pairs, rec)
    after = shells[0]
    assert rec["collinear_faces_removed"] == 2 and rec["t_junctions_split"] >= 2
    assert tjunction._edge_faults(np.asarray(after.faces)) == (0, 0)
    assert np.array_equal(after.vertices, before.vertices)          # no new vertex
    assert len(mesh_iou.self_intersecting_pairs(after)) == 0
    pts = np.random.default_rng(1).uniform(-0.6, 0.6, (400, 3))
    assert np.allclose(winding_number(after, pts), winding_number(before, pts), atol=1e-9)
    assert after.volume == pytest.approx(1.0, rel=1e-15)


def test_t_junction_predicates_are_exact():
    from volumetric_iou import tjunction
    a, b = np.array([-0.5, -0.5, 0.5]), np.array([0.5, -0.5, 0.5])
    assert tjunction.inside(np.array([0.0, -0.5, 0.5]), a, b)
    # one rounding unit off the line, or at an end: not inside
    assert not tjunction.inside(np.array([0.0, np.nextafter(-0.5, -1), 0.5]), a, b)
    assert not tjunction.inside(a, a, b) and not tjunction.inside(np.array([0.6, -0.5, 0.5]), a, b)
    # a slanted line: 0.1 + 0.2 is not 0.3 in float64, so (0.1, 0.2, 0.3)
    # is off the line through 0 and (1, 2, 3) -- a float test with any
    # tolerance would say "on"
    o, d = np.zeros(3), np.array([1.0, 2.0, 3.0])
    assert tjunction.collinear(o, d, np.array([0.5, 1.0, 1.5]))
    assert not tjunction.collinear(o, d, np.array([0.1, 0.2, 0.1 + 0.2]))


@pytest.mark.parametrize("mode", MODES)
def test_t_junction_beside_a_real_touch_keeps_its_pairs(mode):
    # the cavity touching the wall along a line is a touch no split removes:
    # the T-junction is cut, the pairs of the touch stay, the body goes to CGAL
    side, cav = diamond_cavity_on_the_wall()
    big = box([-1] * 3, [1] * 3)
    side = soup(with_t_junction(big, *top_front(big)), cav)
    prep, err = prepare(side, mode)
    assert err is None and prep.info["cavities"] == 1
    assert prep.volume == pytest.approx(8 - 0.192, rel=1e-9)
    assert prep.info["t_junctions_split"] == 1
    assert prep.info["resolved"]["crossing_pairs"] > 0
    pts = np.random.default_rng(0).uniform(-1.1, 1.1, (600, 3))
    assert np.array_equal(winding_number(prep.mesh, pts) > 0.5,
                          winding_number(soup(big, cav), pts) > 0.5)


def test_t_junction_on_a_ring_through_itself_is_still_refused_in_fail():
    # the split does not change w: a pass through itself stays refused in
    # "fail", and in "exact" the solid is the one without the junction
    over = split_ring(0.3)
    V = np.asarray(over.vertices)
    side = with_t_junction(over, *(tuple(V[i]) for i in _held(over, 0, 1)))
    _, err = prepare(side, "fail")
    assert err is not None and err.status is Status.SELF_INTERSECTING
    assert err.diagnostics["t_junctions_split"] == 1
    prep, err = prepare(side, "exact")
    ref, _ = prepare(over, "exact")
    assert err is None and prep.info["t_junctions_split"] == 1
    assert prep.volume == pytest.approx(ref.volume, rel=1e-12)


def _held(m, i, j):
    """(i, j) or (j, i): the direction in which a face of m holds edge {i, j}."""
    F = np.asarray(m.faces)
    return (i, j) if any((F[f, k], F[f, (k + 1) % 3]) == (i, j)
                         for f in range(len(F)) for k in range(3)) else (j, i)


# --------------------------------------------------------------------------- #
# a hole larger than a crack: patched only with a proven volume bound
# --------------------------------------------------------------------------- #


ETA = seams.SEAM_ETA


def open_box(lo, hi, axis=2, side=1):
    """The box less its face whose normal is +-axis (side): the hole is that
    face, a fifth of the box's surface for a cube."""
    b = box(lo, hi)
    keep = b.face_normals[:, axis] * side < 0.5
    return trimesh.Trimesh(b.vertices, b.faces[keep], process=False)


def assert_hole_refused(err, loops=1):
    assert err is not None and err.status is Status.OPEN_SURFACE
    assert "holes left open" in err.detail
    pc = err.diagnostics["holes"]
    assert not pc["accepted"] and pc["loops"] == loops and pc["worst_ratio"] > ETA


def assert_patched(pc, loops):
    assert pc["accepted"] and pc["loops"] == loops and pc["closed"]
    assert pc["bound"] <= ETA * pc["parts_volume"] and pc["worst_ratio"] <= ETA


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("beside", [False, True])
def test_box_without_its_lid_is_refused_alone_or_beside_a_plate(beside, mode):
    # the hole is 20% of its component; beside the plate 0.35% of the side,
    # which was under the side's 5% and capped (ok, IoU 1) -- the lid is
    # invented: the bound is weighed against the box, not the side
    lid = open_box([0, 0, 0.3], [0.1, 0.1, 0.4])
    side = soup(lid, box([-0.5, -0.5, -0.1], [0.5, 0.5, 0.1])) if beside else lid
    _, err = prepare(side, mode)
    assert_hole_refused(err)


@pytest.mark.parametrize("mode", MODES)
def test_cup_is_refused(mode):
    # w at the centre: 1 less the open face's share of the sphere, 5/6
    cup = open_box([-0.25] * 3, [0.25] * 3)
    w = winding_number(cup, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.2], [0.1, 0.1, 0.2]]))
    assert np.all((w > 0.5) & (w < 0.99)) and w[0] == pytest.approx(5 / 6, abs=1e-9)
    _, err = prepare(cup, mode)
    assert_hole_refused(err)


def seam():
    """Two boxes, A = [0, 1]^3 without its face x = 1 and B = [1 + 1e-7, 2]
    x [0, 1]^2 without its face x = 1 + 1e-7: the holes face each other
    across a gap of 1e-7, each a fifth of its box.  The raw winding number
    is 1 in both boxes off the gap (the two missing faces cancel)."""
    a = open_box([0, 0, 0], [1, 1, 1], axis=0, side=1)
    b = open_box([1 + 1e-7, 0, 0], [2, 1, 1], axis=0, side=-1)
    return soup(a, b), [box([0, 0, 0], [1, 1, 1]), box([1 + 1e-7, 0, 0], [2, 1, 1])]


@pytest.mark.parametrize("mode", MODES)
def test_seam_of_two_open_parts_is_patched(mode):
    side, bodies = seam()
    w = winding_number(side, np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.99, 0.5, 0.5]]))
    assert np.allclose(w, 1.0, atol=1e-5)
    prep, err = prepare(side, mode)
    assert err is None
    # the band closes the gap of 1e-7 too: there w of the surface is 1 as well
    # (two fans left it out: 2 - 1e-7)
    assert_solid(prep, [box([0, 0, 0], [2, 1, 1])], 2.0)
    pc = prep.info["holes"]
    assert_patched(pc, 2)
    assert pc["groups"][0]["patch"] == "band" and pc["bound"] < 1e-9
    assert pc["parts_volume"] == pytest.approx(2.0, rel=1e-6)


def test_seam_against_a_crack_of_a_large_part_is_patched():
    # A = [0, 1]^3 without its face x = 1 (a fifth of A: a large hole); B, a
    # block 20 x 21 x 21 beyond the gap of 1e-7, with the matching unit square
    # missing from its face (0.04% of B: a crack).  The raw winding number is
    # 1 in both; B's crack is A's seam: closed with it, not by its own fan
    x0, lo, hi = 1 + 1e-7, -10.0, 11.0
    a = open_box([0, 0, 0], [1, 1, 1], axis=0, side=1)
    b = open_box([x0, lo, lo], [21, hi, hi], axis=0, side=-1)
    o = [[x0, lo, lo], [x0, hi, lo], [x0, hi, hi], [x0, lo, hi]]
    i = [[x0, 0, 0], [x0, 1, 0], [x0, 1, 1], [x0, 0, 1]]
    ring = []
    for k in range(4):
        n = (k + 1) % 4
        ring += [[o[k], o[n], i[n]], [o[k], i[n], i[k]]]
    ring = np.array(ring, float)
    tri = trimesh.Trimesh(ring.reshape(-1, 3), np.arange(len(ring) * 3).reshape(-1, 3))
    flip = tri.face_normals[:, 0] > 0                  # B's face looks to -x
    faces = tri.faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    side = soup(a, b, trimesh.Trimesh(tri.vertices, faces, process=False))
    assert winding_number(side, np.array([[0.5, 0.5, 0.5]]))[0] == pytest.approx(1.0, abs=1e-5)
    prep, err = prepare(side)
    assert err is None
    assert prep.volume == pytest.approx(1.0 + (21 - x0) * 21 * 21, rel=1e-9)
    pc = prep.info["holes"]
    assert_patched(pc, 2)
    assert pc["groups"][0]["loops"] == 2


def test_two_seams_of_one_part_are_patched():
    # A | B | C in a row, B open at both ends: two groups close one part, and
    # each part is closed only with both patches -- the parts are weighed
    # together, not group by group
    g = 1e-7
    a = open_box([0, 0, 0], [1, 1, 1], axis=0, side=1)
    b0 = box([1 + g, 0, 0], [2, 1, 1])
    b = trimesh.Trimesh(b0.vertices, b0.faces[np.abs(b0.face_normals[:, 0]) < 0.5], process=False)
    c = open_box([2 + 2 * g, 0, 0], [3, 1, 1], axis=0, side=-1)
    prep, err = prepare(soup(a, b, c))
    assert err is None
    assert prep.volume == pytest.approx(3.0, rel=1e-6)
    pc = prep.info["holes"]
    assert_patched(pc, 4)
    assert len(pc["groups"]) == 2


def _open_cup(lo, hi, n, outward):
    """A box lo..hi without its top face, n segments to a side of the rim."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    tx, ty = np.linspace(x0, x1, n + 1)[:-1], np.linspace(y0, y1, n + 1)[:-1]
    sq = np.concatenate([np.c_[tx, np.full(n, y0)], np.c_[np.full(n, x1), ty],
                         np.c_[x1 + x0 - tx, np.full(n, y1)], np.c_[np.full(n, x0), y1 + y0 - ty]])
    k = len(sq)
    V = np.vstack([np.c_[sq, np.full(k, z1)], np.c_[sq, np.full(k, z0)],
                   [[(x0 + x1) / 2, (y0 + y1) / 2, z0]]])
    i = np.arange(k)
    j = (i + 1) % k
    F = np.concatenate([np.c_[i, k + i, k + j], np.c_[i, k + j, j], np.c_[np.full(k, 2 * k), k + j, k + i]])
    return trimesh.Trimesh(V, F if outward else F[:, ::-1], process=False)


def test_open_cups_facing_opposite_ways_do_not_cancel_into_a_sheet():
    # an open cup A and an inside-out wider one B whose bottom cuts A's walls,
    # rims 1e-4 apart: one group, equal volumes of opposite sign.  Weighed by
    # the modulus of their sum the parts were a sheet and the fans (rims too
    # long for a band or a triangulation) went in unchecked -- both cups read
    # as closed bodies; part by part the fans' bound is far above SEAM_ETA
    a, d, n = 0.05, 1e-4, 600
    la = (2 * a + 2 * d) ** 2 * 0.05 / (2 * a) ** 2
    side = soup(_open_cup((-a, -a, a - la), (a, a, a), n, True),
                _open_cup((-a - d, -a - d, a + d - 0.05), (a + d, a + d, a + d), n, False),
                box([0.15] * 3, [0.45] * 3), box([-0.45] * 3, [-0.35] * 3))
    _, info = mesh_iou._clean(side)
    assert info["holes"]["sheets"] == 0 and not info["holes"]["accepted"]
    _, err = prepare(side)
    assert err is not None and err.status is Status.OPEN_SURFACE


def test_box_open_against_a_closed_box_is_refused():
    # A's hole faces a closed box B across the gap: B closes nothing for the
    # winding number -- inside A it is 1 less the hole's share (5/6 at the
    # centre), the body is not defined there
    a = open_box([0, 0, 0], [1, 1, 1], axis=0, side=1)
    side = soup(a, box([1 + 1e-7, -0.1, -0.1], [2, 1.1, 1.1]))
    assert winding_number(side, np.array([[0.5, 0.5, 0.5]]))[0] == pytest.approx(5 / 6, abs=1e-9)
    _, err = prepare(side)
    assert_hole_refused(err)


def halves_of_a_cylinder(lower_sections=64, upper_sections=64, gap=1e-7):
    """A cylinder (R 0.1, h 1) cut across its axis into two halves, neither
    capped at the cut, the second moved off by `gap` along the axis: each hole
    is 9% of its half.  Halves of different sections: the rims of the seam are
    tessellated differently (as two CAD faces exported apart)."""
    cyl = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=lower_sections)
    lower = trimesh.intersections.slice_mesh_plane(cyl, [0, 0, -1], [0, 0, 0])
    up = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=upper_sections)
    upper = trimesh.intersections.slice_mesh_plane(up, [0, 0, 1], [0, 0, 0])
    upper.vertices[:, 2] += gap
    return cyl, soup(lower, upper)


def test_cylinder_cut_by_an_open_seam_is_a_number():
    cyl, halves = halves_of_a_cylinder()
    assert mesh_iou._boundary_edges(halves) > 0
    for side, (g, p) in (("pred", normalize_pair(cyl, halves, pred_frame="gt")),
                         ("gt", normalize_pair(halves, cyl, pred_frame="gt"))):
        res = volume_iou(g, p)
        assert res and res.iou == pytest.approx(1.0, abs=1e-6)
        assert_patched(res.diagnostics[side]["holes"], 2)
        assert any("holes patched" in w for w in res.warnings)


@pytest.mark.parametrize("upper,gap,number", [(48, 1e-7, True), (16, 1e-7, False), (64, 1e-3, False)])
def test_seam_bound_follows_the_rims(upper, gap, number):
    # rims of 64 and 48 sides miss each other by up to 1.7e-3 of the radius:
    # a number, the bound 2e-5 of the cylinder; 64 against 16 and a gap of
    # 1e-3 -- the bound is above 1e-3 of the cylinder: refused
    _, halves = halves_of_a_cylinder(64, upper, gap)
    prep, err = prepare(halves)
    if number:
        assert err is None
        assert_patched(prep.info["holes"], 2)
        # half a 64-gon prism and half a 48-gon one (R 0.1, h 1)
        prism = [n / 2 * np.sin(2 * np.pi / n) * 0.01 for n in (64, upper)]
        assert prep.volume == pytest.approx(sum(prism) / 2, rel=1e-4)
    else:
        assert_hole_refused(err, loops=2)


def slit_sheet_box():
    """A unit cube whose top face (subdivided) is cut along y = 0 for |x| <
    0.45 and the cut opened by 1e-7: one loop run out along the slit and back."""
    b = trimesh.creation.box(extents=[1, 1, 1]).subdivide().subdivide()
    v = b.vertices.copy()
    on_slit = np.isclose(v[:, 2], 0.5) & np.isclose(v[:, 1], 0.0) & (np.abs(v[:, 0]) < 0.45)
    faces = b.faces.copy()
    copy = {}
    top = np.isclose(b.triangles_center[:, 2], 0.5) & (b.triangles_center[:, 1] > 0)
    for f in np.flatnonzero(top):
        for k in range(3):
            i = faces[f, k]
            if on_slit[i]:
                if i not in copy:
                    copy[i] = len(v)
                    v = np.vstack([v, v[i] + [0, 1e-7, 0]])
                faces[f, k] = copy[i]
    return trimesh.Trimesh(v, faces, process=False), int(on_slit.sum())


def test_slit_is_a_number():
    side, n = slit_sheet_box()
    assert n >= 3 and mesh_iou._boundary_edges(side) > 0
    prep, err = prepare(side)
    assert err is None
    assert prep.volume == pytest.approx(1.0, rel=1e-6)


def test_energy_bound_is_above_the_energy():
    # the bound of I(Z) against a Monte Carlo estimate over pairs of points
    rng = np.random.default_rng(11)
    V = rng.random((30, 3))
    Z = [tuple(rng.choice(30, 3, replace=False)) for _ in range(20)]
    t = V[np.array(Z)]
    A = seams.tri_area(V, Z)
    n = 200000

    def sample():
        k = rng.choice(len(A), n, p=A / A.sum())
        r1, r2 = rng.random(n), rng.random(n)
        q = np.sqrt(r1)
        return (t[k, 0] * (1 - q)[:, None] + t[k, 1] * (q * (1 - r2))[:, None]
                + t[k, 2] * (q * r2)[:, None])
    est = A.sum() ** 2 * np.mean(1.0 / np.linalg.norm(sample() - sample(), axis=1))
    assert seams.energy_bound(V, Z) > 1.2 * est


@pytest.mark.parametrize("mode", MODES)
def test_junk_sheet_beside_a_cube_adds_nothing(mode):
    # a flat square of 2 faces: its hole is all of it; patched, it encloses
    # no volume (dropped as a zero-volume shell) -- the body is the cube
    sheet = trimesh.Trimesh([[2, 0, 0], [2.1, 0, 0], [2.1, 0.1, 0], [2, 0.1, 0]],
                            [[0, 1, 2], [0, 2, 3]], process=False)
    prep, err = prepare(soup(CUBE, sheet), mode)
    assert err is None
    assert_solid(prep, [CUBE], 1.0)
    pc = prep.info["holes"]
    assert pc["accepted"] and pc["sheets"] == 1


def cube_without_triangle(subdivisions):
    """The unit cube subdivided, less one triangle near the middle of its face
    +z: a hole of 1 / (2 * 4^subdivisions) of that face."""
    m = trimesh.creation.box(extents=(1, 1, 1))
    for _ in range(subdivisions):
        m = m.subdivide()
    c = m.triangles_center
    on = np.flatnonzero(np.isclose(c[:, 2], 0.5))
    drop = on[np.argmin(np.linalg.norm(c[on, :2], axis=1))]
    return trimesh.Trimesh(m.vertices, np.delete(m.faces, drop, axis=0), process=False)


@pytest.mark.parametrize("mode", MODES)
def test_crack_is_patched_only_with_a_proven_bound(mode):
    # a crack is under the bound as a large hole is: a missing triangle of
    # 1/32 of a face (0.5% of the surface, a crack by area) is a guess of
    # (pi/4) I(Z) = 0.04 of the cube -- refused; of 1/2048 of a face -- patched
    _, err = prepare(cube_without_triangle(2), mode)
    assert_hole_refused(err)
    prep, err = prepare(cube_without_triangle(5), mode)
    assert err is None
    assert_patched(prep.info["holes"], 1)
    assert prep.volume == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_facing_cracks_of_two_parts_are_closed_by_a_band(mode):
    # two cubes one on the other, 1e-7 apart, each with the same small hole
    # in the faces that meet: each loop is a crack, and each fan alone a cap
    # bounded at 0.04 of its cube; together the loops are one seam, closed by
    # the band between them -- of nearly no volume
    lo = cube_without_triangle(2)
    hi = trimesh.Trimesh(lo.vertices * [1, 1, -1] + [0, 0, 1 + 1e-7], lo.faces[:, ::-1], process=False)
    prep, err = prepare(soup(lo, hi), mode)
    assert err is None
    pc = prep.info["holes"]
    assert_patched(pc, 2)
    assert [g["patch"] for g in pc["groups"]] == ["band"]
    assert prep.volume == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_inside_out_part_with_a_crack_is_patched(mode):
    # the parts' volume is taken unsigned: shells are oriented only later (4.6)
    m = cube_without_triangle(5)
    m = trimesh.Trimesh(m.vertices, m.faces[:, ::-1], process=False)
    prep, err = prepare(m, mode)
    assert_patched(prep.info["holes"] if err is None else err.diagnostics["holes"], 1)


def _fan_key(verts, fan):
    """A fan as its triangles by coordinates, each up to rotation (orientation kept)."""
    out = set()
    for t in fan:
        p = [tuple(np.round(verts[i], 12)) for i in t]
        k = min(range(3), key=lambda i: p[i])
        out.add(tuple(p[k:] + p[:k]))
    return out


@pytest.mark.parametrize("shape", ["off_plane", "slit"])
def test_fan_does_not_depend_on_where_the_loop_starts(shape):
    # 4.3: the patch and its area are the same
    # whichever vertex the numbering starts the loop at
    rng = np.random.default_rng(3)
    if shape == "off_plane":
        t = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        verts = np.column_stack([np.cos(t), np.sin(t), 0.6 * np.cos(2 * t) + 0.05 * rng.random(9)])
    else:
        # a slit run out and back along two spokes: a fan from a spoke's tip
        # spans the spokes, from the hub it is slivers
        out = [[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [0.3, 1e-9, 0], [0, 1e-9, 0],
               [0, 0.3, 0.1], [0, 0.6, 0.2], [0, 0.3, 0.1 + 1e-9]]
        verts = np.array(out, float)
    n = len(verts)
    keys, areas = set(), []
    for s in range(n):
        loop = [(s + i) % n for i in range(n)]
        fan = mesh_iou._fan(loop, verts)
        keys.add(frozenset(_fan_key(verts, fan)))
        areas.append(mesh_iou._fan_area(verts, fan))
    assert len(keys) == 1
    assert max(areas) == min(areas)
    from_first = [mesh_iou._fan_area(verts, mesh_iou._fan([(s + i) % n for i in range(n)]))
                  for s in range(n)]
    assert min(areas) == pytest.approx(min(from_first), rel=1e-12)   # the least
    assert max(from_first) > (1 + 1e-3) * min(from_first)     # from the first vertex: it varies
    if shape == "slit":
        assert min(areas) < 1e-6 < 0.01 < max(from_first)


def test_fan_of_a_planar_convex_loop_starts_at_its_least_point():
    # every start gives the same area up to rounding: the start must not
    # follow the rounding of one numbering
    t = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    verts = np.column_stack([np.cos(t) + 3.1, 0.7 * np.sin(t) - 2.3, np.full(13, 0.4)])
    want = int(np.lexsort(verts.T[::-1])[0])
    for s in range(13):
        loop = [(s + i) % 13 for i in range(13)]
        assert mesh_iou._fan(loop, verts)[0][0] == want


def test_patched_area_does_not_depend_on_vertex_numbering():
    # a cube with a hole across one of its edges: the loop bends over the edge,
    # and the fan's area once depended on the loop's first vertex
    box = trimesh.creation.box(extents=[1, 1, 1]).subdivide().subdivide()
    c = box.triangles_center
    keep = np.linalg.norm(c - [0.5, 0.5, 0.0], axis=1) > 0.3
    hole = trimesh.Trimesh(box.vertices, box.faces[keep], process=False)
    rng = np.random.default_rng(7)
    fans, verdicts = set(), set()
    for _ in range(6):
        perm = rng.permutation(len(hole.vertices))
        inv = np.argsort(perm)
        m = trimesh.Trimesh(hole.vertices[perm], inv[hole.faces], process=False)
        verts = np.asarray(m.vertices)
        (loop,) = mesh_iou._boundary_loops(m)
        fans.add(round(mesh_iou._fan_area(verts, mesh_iou._fan(loop, verts)), 14))
        # nor does the verdict on it (refused: the patch bends over the edge)
        added, _, rec = mesh_iou._patch_loops(m)
        verdicts.add((added, rec["accepted"], rec["groups"][0]["patch"], round(rec["worst_ratio"], 12)))
    assert len(fans) == 1 and len(verdicts) == 1


def slab(h, hole=lambda i, j: False):
    """A slab of depth 10 under the heightfield z = h(x, y): its top a grid of
    128 x 128 cells over [0, 1]^2 and a ring of cells out to [-50, 51]^2 (a
    large part, so a patch of the size of [0, 1]^2 is within the bound), its
    walls and its bottom (a fan) flat.  Cells (i, j) with `hole` true are left
    out of the top; i, j = 1..128 are those over [0, 1]^2."""
    t = np.r_[-50.0, np.linspace(0, 1, 129), 51.0]
    n = len(t) - 1
    X, Y = np.meshgrid(t, t, indexing="ij")
    top = np.c_[X.ravel(), Y.ravel(), h(np.clip(X, 0, 1), np.clip(Y, 0, 1)).ravel()]
    i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    keep = ~np.array([hole(a, b) for a, b in zip(i, j)])
    i, j = i[keep], j[keep]
    a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
    F = [np.c_[a, b, c], np.c_[a, c, d]]
    k = np.arange(n)
    ring = np.r_[k * (n + 1), n * (n + 1) + k, (n - k) * (n + 1) + n, n - k]
    bottom = top[ring] * [1, 1, 0] - [0, 0, 10]
    m0, centre = len(top), len(top) + len(ring)
    A, B = m0 + np.arange(len(ring)), m0 + np.roll(np.arange(len(ring)), -1)
    F += [np.c_[ring, A, B], np.c_[ring, B, np.roll(ring, -1)],
          np.c_[np.full(len(ring), centre), B, A]]
    V = np.vstack([top, bottom, [[0.5, 0.5, -10]]])
    return trimesh.Trimesh(V, np.vstack(F), process=False)


def u_hole(i, j):
    """A U of cells: [10, 120)^2 without the tongue [50, 80) x [50, 120) --
    a loop of 578 edges, closed by its fan, which folds over the tongue."""
    return 10 <= i < 120 and 10 <= j < 120 and not (50 <= i < 80 and j >= 50)


def patched(side):
    """The side cleaned and patched (`_clean`), the faces it had before."""
    m, info = mesh_iou._clean(side)
    assert info["holes"]["accepted"] and info["loop_faces_added"] == 578
    return m, len(m.faces) - info["loop_faces_added"]


def test_slab_premise():
    # closed, outward, embedded: it winds 0 or 1 everywhere
    s = slab(lambda x, y: 0.05 * np.abs(x - 0.5))
    assert s.is_watertight and s.is_winding_consistent and s.volume > 0
    assert len(mesh_iou.self_intersecting_pairs(s)) == 0


@pytest.mark.parametrize("mode", MODES)
def test_fold_of_a_patch_round_a_sliver_is_an_inverted_pocket(mode):
    # the top a wave: the fan of the U folds over the tongue, through its
    # faces, and winds -1 round slivers of 6e-6 of volume, 6e-11 of the
    # part -- refused, filled too (no threshold; the pocket lies out in the
    # open).  The slab without its hole is embedded (no pocket of its own)
    def h(x, y):
        return 0.1 * np.sin(6 * x) * y
    assert len(mesh_iou.self_intersecting_pairs(slab(h))) == 0
    m, _ = patched(slab(h, u_hole))
    p = np.array([[0.44953583, 0.46820158, 0.02102229]])
    assert winding_number(m, p)[0] == pytest.approx(-1.0, abs=1e-6)
    for fill in (False, True):
        _, err = prepare(slab(h, u_hole), mode, fill=fill)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert 0 < err.diagnostics["inverted_volume"] < 1e-9 * 101 ** 2 * 10
        assert err.diagnostics["loop_faces_added"] == 578


@pytest.mark.parametrize("mode", MODES)
def test_fold_of_a_patch_round_slivers_of_rounding_has_a_positive_volume(mode):
    # the top a slope: 0.1 x is rounded, the top's vertices are off one plane
    # by a rounding step, and the fan of the U folded over the tongue winds -1
    # round slivers of that size -- refused, as any pocket; their volume is
    # summed exactly (the float64 sum of the rounded pocket faces is -7e-20)
    def h(x, y):
        return 0.1 * x
    for fill in (False, True):
        _, err = prepare(slab(h, u_hole), mode, fill=fill)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert 0 < err.diagnostics["inverted_volume"] < 1e-18
        assert err.diagnostics["loop_faces_added"] == 578


def test_accepted_patch_cutting_its_neighbours_goes_to_the_detector():
    # the top a ridge along x = 0.5: the fan of the U, accepted, runs through
    # the faces of the tongue -- the detector sees its crossings like any
    # other (no pair without the patch): the body is built in CGAL; "exact"
    # takes w >= 1 of the patched surface, "fail" refuses where it winds 2
    def h(x, y):
        return 0.05 * np.abs(x - 0.5)
    side = slab(h, u_hole)
    m, n = patched(side)
    pairs = mesh_iou.self_intersecting_pairs(m)
    assert ((pairs[:, 0] < n) != (pairs[:, 1] < n)).any()        # patch against the side
    assert not ((pairs[:, 0] < n) & (pairs[:, 1] < n)).any()     # nothing else crosses
    prep, err = prepare(side, "exact")
    assert err is None and prep.info["resolved"]["crossing_pairs"] > 0
    pts = np.random.default_rng(0).random((600, 3)) * [1, 1, 0.2] - [0, 0, 0.1]
    got = winding_number(prep.mesh, pts) > 0.5
    assert got.any() and not got.all()
    assert np.array_equal(got, winding_number(m, pts) > 0.5)
    _, err = prepare(side, "fail")
    assert err is not None and err.status is Status.SELF_INTERSECTING
    assert err.diagnostics["self_overlap_bodies"] == 1 and err.diagnostics["self_overlap_volume"] > 0


def test_t_junction_split_failure_leaves_the_path_as_without_it(monkeypatch):
    # the detector call after the first one (all pairs in "fail", the
    # recount in "exact") raises: the shells are put back, no count is
    # left, the error is recorded, and the body goes to CGAL with its
    # original pairs -- the status as without the split
    orig = mesh_iou.self_intersecting_pairs
    for mode in MODES:
        calls = []

        def flaky(mesh, threads=None, first=False):
            calls.append(first)
            if len(calls) > 1:
                raise RuntimeError("detector down")
            return orig(mesh, threads, first)
        monkeypatch.setattr(mesh_iou, "self_intersecting_pairs", flaky)
        prep, err = prepare(cube_with_t_junction(), mode)
        # the third call checks the rounded CGAL solid: it raises too, so
        # the solid stays in CGAL (no manifold copy)
        assert err is None and len(calls) == 3 and prep.solid is None
        assert_solid(prep, [CUBE], 1.0)
        i = prep.info
        assert "detector down" in i["t_junction_error"]
        assert i["t_junctions_split"] == i["collinear_faces_removed"] == 0
        assert i["self_intersection_checked"] is True
        assert i["resolved"]["crossing_pairs"] > 0


def l_touch():
    """Box [0, 2] x [0, 1]^2 and box [1, 2] x [0, 1] x [1, 2] on it, their
    corner (2, 0, 1) one vertex: the upper box's bottom front edge v-b is a
    part of the lower box's top front edge a-b -- v is a T-junction, but
    v-b is already an edge of two faces (the boxes touch along it)."""
    lo, up = box([0, 0, 0], [2, 1, 1]), box([1, 0, 1], [2, 1, 2])
    V = np.vstack([lo.vertices, up.vertices])
    F = np.vstack([lo.faces, up.faces + len(lo.vertices)])
    _, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    return trimesh.Trimesh(V[first], inv.reshape(-1)[F], process=False)


def test_t_junction_split_that_makes_an_edge_of_four_faces_is_undone():
    from volumetric_iou import tjunction
    sh = l_touch()                      # as one shell: the touch along v-b
    faces = sh.faces.copy()
    pairs = mesh_iou.self_intersecting_pairs(sh)
    assert len(pairs) > 0
    rec = {}
    assert not tjunction.split_t_junctions([sh], pairs, rec)
    assert rec == {"t_junction_splits_undone": 1}
    assert np.array_equal(sh.faces, faces)
    rec = {}
    n, bad, unchecked, _ = mesh_iou._crossings([sh], [0], split=rec)
    assert n == len(pairs) and bad == {0} and not unchecked
    assert rec == {"t_junction_splits_undone": 1}


@pytest.mark.parametrize("mode", MODES)
def test_touch_along_a_segment_with_a_t_junction_is_built_in_cgal(mode, monkeypatch):
    # read as one body (as a shell folding onto itself would be): the split
    # is undone, the pairs of the touch send the body to CGAL, volume 3
    monkeypatch.setattr(mesh_iou, "_orient_shells",
                        lambda m, **kw: ([m.copy()], ["solid"], [-1], {}))
    prep, err = prepare(l_touch(), mode)
    assert err is None and prep.volume == pytest.approx(3.0, rel=1e-12)
    i = prep.info
    assert i["t_junctions_split"] == 0
    assert i["t_junction_splits_undone"] == 1
    assert i["resolved"]["crossing_pairs"] > 0


# --------------------------------------------------------------------------- #
# an inverted pocket in a cavity, with fill_cavities (5.3, 5.5)
# --------------------------------------------------------------------------- #

# a square [0, 4]^2 run counter-clockwise with a loop [3, 6] x [1, 2] run
# clockwise across its right wall: w = -1 on [4.5, 6] x [1, 2], outside the
# square (1.5 x 1 x 1, scaled by 0.4: 0.096)
_LEAK = [(0, 0), (4, 0), (4, 1), (3, 1), (3, 2), (6, 2), (6, 1), (4.5, 1), (4.5, 4), (0, 4)]
LEAK_POCKET = 1.5 * 0.4 ** 3


def leak(at=(1.0, 1.0, 1.0)):
    """The figure above, 0.4 of its size, its corner at `at`."""
    from shapes import extruded
    m = extruded(_LEAK)
    m.vertices = m.vertices * 0.4 + np.asarray(at)
    return m


def hollow_box():
    """[0, 4]^3 with the cavity [0.5, 3.5]^3 (an inward shell)."""
    return soup(box([0, 0, 0], [4, 4, 4]), inverted(box([0.5] * 3, [3.5] * 3)))


def filled(side, mode="exact"):
    """The side prepared with fill_cavities and filled: (prep, err)."""
    prep, err = prepare(side, mode, fill=True)
    if err is not None:
        return None, err
    return mesh_iou._fill_cavities(prep, "gt")


@pytest.mark.parametrize("mode", MODES)
def test_pocket_in_a_cavity_is_taken_in_by_the_filling(mode):
    # H a hollow box, F leaks out of the figure but stays in H's cavity: H \ F
    # and H u F differ by the pocket, but both fill to the whole box --
    # refused without filling, the box with it
    side = soup(hollow_box(), leak())
    _, err = prepare(side, mode)
    assert err is not None and err.status is Status.INVERTED_VOLUME
    assert err.diagnostics["inverted_volume"] == pytest.approx(LEAK_POCKET, rel=1e-9)
    prep, err = filled(side, mode)
    assert err is None
    assert_solid(prep, [box([0, 0, 0], [4, 4, 4])], 64.0)
    assert prep.info["inverted_cavity_volume"] == pytest.approx(LEAK_POCKET, rel=1e-9)
    assert prep.info["inverted_cavity_faces"] > 0
    assert "inverted_volume" not in prep.info


def test_pocket_in_a_cavity_through_volume_iou():
    g, p = normalize_pair(soup(hollow_box(), leak()), box([0, 0, 0], [4, 4, 4]),
                          pred_frame="gt")
    res = volume_iou(g, p, fill_cavities=True)
    assert res and res.iou == pytest.approx(1.0, abs=1e-9)
    assert res.diagnostics["gt"]["inverted_cavity_volume"] == pytest.approx(
        LEAK_POCKET / 4 ** 3, rel=1e-9)
    assert any("inverted pocket) lying in a cavity" in w for w in res.warnings)
    res = volume_iou(g, p, fill_cavities=(False, True))
    assert res.status is Status.INVERTED_VOLUME and res.blame == "gt"


@pytest.mark.parametrize("mode", MODES)
def test_pocket_out_in_the_open_is_refused_when_filled(mode):
    # the figure alone: its pocket lies outside, in the outer cell
    for fill in (False, True):
        _, err = prepare(leak(), mode, fill=fill)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert "inverted_cavity_volume" not in err.diagnostics


@pytest.mark.parametrize("mode", MODES)
def test_pocket_touching_the_outside_by_a_face_is_refused_when_filled(mode):
    # a notch [1, 2]^2 x [1, 2] in the top of a block, filled by an inverted
    # box flush with the top: the pocket meets the outer cell across its top
    block = [box([0, 0, 0], [3, 3, 1]), box([0, 0, 1], [1, 3, 2]), box([2, 0, 1], [3, 3, 2]),
             box([1, 0, 1], [2, 1, 2]), box([1, 2, 1], [2, 3, 2])]
    side = soup(*block, inverted(box([1, 1, 1], [2, 2, 2])))
    for fill in (False, True):
        _, err = prepare(side, mode, fill=fill)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert err.diagnostics["inverted_volume"] == pytest.approx(1.0, rel=1e-9)


def edge_contact():
    """A block [0, 3]^3 whose middle layer z in [1, 2] has two unit cells
    left out: P = [1, 2]^2 (an inverted box: the pocket) and C = [2, 3]^2,
    open to the outside; P and C meet along the edge x = y = 2 only."""
    layer = [box([0, 0, 1], [1, 3, 2]), box([1, 0, 1], [3, 1, 2]), box([1, 2, 1], [2, 3, 2]),
             box([2, 1, 1], [3, 2, 2])]
    return soup(box([0, 0, 0], [3, 3, 1]), box([0, 0, 2], [3, 3, 3]), *layer,
                inverted(box([1, 1, 1], [2, 2, 2])))


@pytest.mark.parametrize("mode", MODES)
def test_pocket_touching_the_outside_at_an_edge_is_refused_when_filled(mode):
    # the void around the pocket shares an edge with the outer cell: the
    # solid's boundary is one shell (by vertices), filling leaves the void
    # open, so the two readings differ by the pocket (1 of 26) -- refused
    side = edge_contact()
    from volumetric_iou import _cgal
    got = _cgal.solid_and_pocket(np.ascontiguousarray(side.vertices, np.float64),
                                 np.ascontiguousarray(side.faces, np.int64), cavity=True)
    assert len(got[3]) and got[4] is False
    for fill in (False, True):
        _, err = prepare(side, mode, fill=fill)
        assert err is not None and err.status is Status.INVERTED_VOLUME


def cups():
    """Two cups facing each other, their rims overlapping (w = 2): neither
    encloses anything alone, together they close the cavity [0.5, 3.5]^3."""
    import manifold3d
    def cup(lo, hi, ilo, ihi):
        a = manifold3d.Manifold.cube(np.subtract(hi, lo)).translate(lo)
        b = manifold3d.Manifold.cube(np.subtract(ihi, ilo)).translate(ilo)
        return mesh_iou._from_manifold(a - b)
    return (cup([0, 0, 0], [4, 4, 2.5], [0.5, 0.5, 0.5], [3.5, 3.5, 3]),
            cup([0, 0, 1.5], [4, 4, 4], [0.5, 0.5, 1], [3.5, 3.5, 3.5]))


def test_pocket_in_a_cavity_closed_by_two_bodies():
    # as cadenabench couplings and shafts: the cavity is there only in the
    # union of two overlapping bodies, the pocket inside it
    lower, upper = cups()
    side = soup(lower, upper, leak())
    _, err = prepare(side)
    assert err is not None and err.status is Status.INVERTED_VOLUME
    prep, err = filled(side)
    assert err is None
    assert_solid(prep, [box([0, 0, 0], [4, 4, 4])], 64.0)
    assert prep.info["inverted_cavity_volume"] == pytest.approx(LEAK_POCKET, rel=1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_body_pocket_inside_two_bodies_seeds_the_cavity_walk(mode):
    # the bowtie's w = -1 lobe inside two overlapping boxes: the sum winds
    # 1 there, the cell is a pocket of the bowtie alone and lies in the solid
    # (both readings give the boxes).  Refused without filling; with it a
    # number -- the lobe's cells are not seeds of the walk, they lie in A:
    # - lobe clear of the boxes' faces: the boxes' union;
    # - lobe's bottom flush with both boxes' bottoms (a seed there would walk
    #   across that face into the outer cell): the union as well;
    # - the same inside a hollow box: the hollow box filled
    b = bowtie()
    a, c = box([-0.2] * 3, [0.6] * 3), box([-0.1] * 3, [0.7] * 3)
    af, cf = box([-0.2, -0.2, 0], [0.6, 0.6, 0.6]), box([-0.1, -0.1, 0], [0.7, 0.7, 0.6])
    outer = box([-1] * 3, [1.5] * 3)
    hollow = soup(outer, inverted(box([-0.5] * 3, [1.0] * 3)))
    for side, want in ((soup(a, c, b), ([a, c], 2 * 0.8 ** 3 - 0.7 ** 3)),
                       (soup(af, cf, b), ([af, cf], 2 * 0.8 ** 2 * 0.6 - 0.7 ** 2 * 0.6)),
                       (soup(hollow, af, cf, b), ([outer], 2.5 ** 3))):
        _, err = prepare(side, mode)
        assert err is not None and err.status is Status.INVERTED_VOLUME
        assert err.diagnostics["inverted_volume"] == pytest.approx(0.016, rel=1e-9)
        prep, err = filled(side, mode)
        assert err is None
        assert_solid(prep, *want)
        assert prep.info["inverted_cavity_volume"] == pytest.approx(0.016, rel=1e-9)


def test_solid_and_pocket_cavity_flag():
    # the module call: four items without `cavity`, the flag appended with it;
    # False without a pocket (nothing to decide)
    from volumetric_iou import _cgal
    for m, want in ((CUBE, False), (leak(), False), (soup(hollow_box(), leak()), True)):
        V = np.ascontiguousarray(m.vertices, np.float64)
        F = np.ascontiguousarray(m.faces, np.int64)
        assert len(_cgal.solid_and_pocket(V, F)) == 4
        got = _cgal.solid_and_pocket(V, F, cavity=True)
        assert len(got) == 5 and got[4] is want
        got = _cgal.solid_pocket_overlap(V, F, [len(F)], [0], cavity=True)
        assert len(got) == 8 and got[7] is want
        # the pockets' volume appended last with `volume`, after the flag
        got = _cgal.solid_and_pocket(V, F, cavity=True, volume=True)
        assert len(got) == 6 and got[4] is want and isinstance(got[5], float)
        assert len(_cgal.solid_pocket_overlap(V, F, [len(F)], [0], volume=True)) == 8
        got = _cgal.solid_pocket_overlap(V, F, [len(F)], [0], cavity=True, volume=True)
        assert len(got) == 9 and got[7] is want and isinstance(got[8], float)


def test_pocket_volume_is_summed_exactly():
    # the pockets' volume from the arrangement: of an inverted box alone, of
    # an inverted box half out of a cube (only the half outside winds -1),
    # of a rotated one (vertices born of crossings) -- as the float64 volume
    # of the pocket faces where that is far above rounding; 0.0 without one
    from volumetric_iou import _cgal

    def call(m, fn):
        V = np.ascontiguousarray(m.vertices, np.float64)
        F = np.ascontiguousarray(m.faces, np.int64)
        got = (_cgal.solid_and_pocket(V, F, volume=True) if fn == "one"
               else _cgal.solid_pocket_overlap(V, F, [len(F)], [0], volume=True))
        return got[-1], trimesh.Trimesh(got[2], got[3], process=False)

    turned = inverted(box([1.5, 0.5, 0.5], [2.5, 1.5, 1.5]))
    turned.apply_transform(trimesh.transformations.rotation_matrix(0.3, [1, 2, 3], [2, 1, 1]))
    for fn in ("one", "bodies"):
        assert call(CUBE, fn)[0] == 0.0
        assert call(inverted(box([0, 0, 0], [1, 1, 1])), fn)[0] == 1.0
        assert call(soup(box([0, 0, 0], [2, 2, 2]), inverted(box([1.5, 0.5, 0.5], [2.5, 1.5, 1.5]))),
                    fn)[0] == 0.5
        got, pocket = call(soup(box([0, 0, 0], [2, 2, 2]), turned), fn)
        assert 0 < got < 1 and got == pytest.approx(pocket.volume, rel=1e-12)


def test_parts_labelled_in_the_solids_own_arrangement():
    # components labelled per face, the faces in their order: the solid and
    # pockets are those of `_cgal_solid` (the same triangles; their order
    # varies from call to call either way), with the largest winding of the
    # sum (the third body over a half: 2) and no part through itself
    a, b = touching_halves()
    c = box([0.25, -0.25, -0.25], [0.75, 0.25, 0.25])
    side = soup(a, b, c)
    comp = mesh_iou._face_components(side)
    assert len(set(comp.tolist())) == 2
    want = mesh_iou._cgal_solid(side, cavity=True)
    got, w_max, through = mesh_iou._cgal_solid_parts(side, comp, cavity=True)
    def tris(m):
        t = [sorted(map(tuple, tri)) for tri in m.vertices[m.faces].tolist()]
        return np.array(sorted(t))
    assert np.array_equal(tris(got[0]), tris(want[0]))
    assert got[1] == pytest.approx(want[1], rel=1e-12) and got[2:] == want[2:]
    assert w_max == 2 and through == 0
    # one label: the sum alone, its w >= 2 counted as one part
    _, w_max, through = mesh_iou._cgal_solid_parts(side, np.zeros(len(comp), np.int64))
    assert w_max == 2 and through == 1
