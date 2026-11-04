"""
Cells of space cut by a set of triangles (`volumetric_iou.cells`): shells
written on both sides, twice, with touching and nested parts, a one-sided wall
inside a closed surface.  The winding number of every cell is checked
independently -- the generalized winding number of the input (sum of solid
angles, `winding.winding_number`) at the point the cell reports -- and the
cell volumes against the geometry.
"""

import numpy as np
import pytest
import trimesh

from shapes import extruded
from volumetric_iou import _cgal
from volumetric_iou.cells import cells, merge_vertices
from volumetric_iou.winding import winding_number


def box(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


def joined(*parts):
    """One soup of the given (vertices, faces) or meshes, in turn."""
    V, F, n = [], [], 0
    for p in parts:
        v, f = (p.vertices, p.faces) if isinstance(p, trimesh.Trimesh) else p
        V.append(np.asarray(v, np.float64))
        F.append(np.asarray(f, np.int64) + n)
        n += len(v)
    return np.vstack(V), np.vstack(F)


def two_sided(m, times=1):
    """m written both ways, `times` times."""
    return [(m.vertices, m.faces), (m.vertices, m.faces[:, ::-1])] * times


def generalized_w(V, F, c):
    """The input's winding number at each cell's point, outer cell included."""
    return winding_number(trimesh.Trimesh(V, F, process=False), c.point)


def check(V, F, volumes, winding):
    """Cells with these volumes (bounded cells, sorted) and, cell by cell, the
    winding number of the input at their points (None: not defined)."""
    c = cells(V, F)
    assert c.num_cells == len(volumes) + 1
    assert c.volume[0] == pytest.approx(-sum(volumes), abs=1e-9)
    assert sorted(c.volume[1:]) == pytest.approx(sorted(volumes), abs=1e-9)
    assert np.all(np.isfinite(c.point))
    w = generalized_w(V, F, c)
    if winding is None:
        assert c.winding is None
    else:
        assert c.winding is not None
        assert np.allclose(w, c.winding, atol=1e-6), (w, c.winding)
        # the walk agrees with the definition, and the expected values hold
        assert sorted(c.winding[1:].tolist()) == sorted(winding)
    return c, w


CUBE = box([-0.5] * 3, [0.5] * 3)


def test_one_sided_cube():
    c, _ = check(*joined(CUBE), [1.0], [1])
    assert c.balanced and not c.two_sided


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_cube_written_k_times_both_ways(k):
    # w = 0 everywhere, but the cube's cell is there: k on every triangle,
    # the cell is next to the outer one, so it is filled whatever the bodies
    c, _ = check(*joined(*two_sided(CUBE, k)), [1.0], [0])
    assert c.two_sided
    assert np.all(c.up == k) and np.all(c.down == k)
    assert c.k_distance.tolist() == [0, k]
    assert c.parity.tolist() == [0, k % 2]
    assert c.filled.tolist() == [False, True]


def test_cube_two_up_one_down():
    # 2 copies one way, 1 the other: w = 1, not a two-sided record
    c, _ = check(*joined((CUBE.vertices, CUBE.faces), *two_sided(CUBE)), [1.0], [1])
    assert not c.two_sided and c.filled is None
    assert np.all(c.multiplicity == 2 - 1) or np.all(c.multiplicity == 1 - 2)


@pytest.mark.parametrize("shift", [1e-12, 3e-9])
def test_copy_shifted_below_the_grid_is_the_same_face(shift):
    # vertices merge on cleaning's 1e-8 grid, not exactly: a copy off by noise
    # is the same face, not a thin cell
    v, f = CUBE.vertices, CUBE.faces
    V, F = joined((v, f), (v + shift, f[:, ::-1]), (v, f), (v - shift, f[:, ::-1]))
    c = cells(V, F)
    assert c.num_cells == 2 and c.two_sided
    assert np.all(c.up == 2)
    assert c.winding.tolist() == [0, 0]


def test_merge_vertices_as_cleaning():
    rng = np.random.default_rng(0)
    v = np.repeat(CUBE.vertices, 2, axis=0) + rng.uniform(-2e-9, 2e-9, (16, 3))
    f = np.vstack([2 * CUBE.faces, 2 * CUBE.faces[:, ::-1] + 1])
    vm, fm = merge_vertices(v, f)
    Vc, Fc, *_ = _cgal.clean(np.ascontiguousarray(v), np.ascontiguousarray(f[:12]))
    assert np.array_equal(np.asarray(Vc), vm)            # same points, same order
    assert np.array_equal(np.asarray(Fc), fm[:12])
    assert len(vm) == 8


def test_two_parts_apart_one_written_twice():
    # A two-sided, B two-sided twice: both cells border the outer one
    a = box([-0.45, -0.15, -0.15], [-0.15, 0.15, 0.15])
    b = box([0.15, -0.15, -0.15], [0.45, 0.15, 0.15])
    c, _ = check(*joined(*two_sided(a), *two_sided(b, 2)), [0.027, 0.027], [0, 0])
    assert c.filled[1:].all()
    assert sorted(c.k_distance[1:].tolist()) == [1, 2]


def test_touching_parts_one_written_twice():
    # C on D; C two-sided twice, D once: k = 3 on the contact (cut where the
    # two triangulations differ), 2 on C, 1 on D
    C = box([-0.25, -0.25, 0.0], [0.25, 0.25, 0.5])
    D = box([-0.25, -0.25, -0.5], [0.25, 0.25, 0.0])
    c, _ = check(*joined(*two_sided(C, 2), *two_sided(D)), [0.125, 0.125], [0, 0])
    assert c.two_sided and c.filled[1:].all()
    tri = c.vertices[c.triangles]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    assert area[c.k == 3].sum() == pytest.approx(0.25)        # the contact square
    assert set(c.k.tolist()) == {1, 2, 3}
    assert sorted(c.k_distance[1:].tolist()) == [1, 2]


def test_nested_parts_sharing_a_face():
    # A inside B, their bottoms on one plane (A's bottom overlaps part of
    # B's): the overlap is one triangle stack, cells A and B \ A
    A = box([-0.25, -0.25, -0.5], [0.25, 0.25, 0.0])
    B = box([-0.25, -0.25, -0.5], [0.25, 0.25, 0.5])
    c, _ = check(*joined(*two_sided(A), *two_sided(B)), [0.125, 0.125], [0, 0])
    assert c.filled[1:].all()


def test_hollow_ball_is_ambiguous():
    # outer sphere and inner sphere (a cavity, or a ball in a ball): the inner
    # cell has even parity and its only neighbour is the wall -- not filled
    outer = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    inner = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    hollow = joined(outer, (inner.vertices, inner.faces[:, ::-1]))
    one = check(*hollow, [outer.volume - inner.volume, inner.volume], [1, 0])[0]
    V, F = joined(*two_sided(outer), *two_sided(inner))
    c, _ = check(V, F, [outer.volume - inner.volume, inner.volume], [0, 0])
    wall, core = np.argsort(c.volume[1:])[::-1] + 1
    assert c.k_distance[wall] == 1 and c.k_distance[core] == 2
    assert c.parity[wall] == 1 and c.parity[core] == 0
    assert c.filled[wall] and not c.filled[core]
    assert one.filled is None                                  # one-sided: no k-reading


@pytest.mark.parametrize("same", [True, False])
def test_touching_cubes_balanced(same):
    # two one-sided half-cubes touching on x = 0: with the same triangulation
    # of the contact (coincident faces, one up, one down) or a different one
    # (overlapping coplanar faces, cut exactly); balanced, w = 1 on both sides
    a = box([-0.5, -0.5, -0.5], [0.0, 0.5, 0.5])
    b = a.copy()
    b.vertices[:, 0] *= -1
    b.invert()
    if not same:
        b.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    c, _ = check(*joined(a, b), [0.5, 0.5], [1, 1])
    contact = (c.below != 0) & (c.above != 0)
    assert contact.any()
    assert np.all(c.up[contact] == 1) and np.all(c.down[contact] == 1)


def test_touching_hexagonal_prisms():
    hexagon = [(0.5 * np.cos(t), 0.5 * np.sin(t)) for t in np.arange(6) * np.pi / 3]
    a = extruded(hexagon, height=0.5, centre=(0.01, 0.02))
    b = a.copy()
    b.vertices[:, 2] *= -1
    b.invert()
    vol = a.volume
    check(*joined(a, b), [vol, vol], [1, 1])


def cube_with_wall(split_mid):
    """Cube [-.5, .5]^3 with a one-sided wall in the plane x = z, its edges
    on the cube's edges (3 faces each); optionally a vertex in the middle of
    the wall's edge on y = -.5 (a 5-edge hole after cutting)."""
    P, F = {}, []

    def s(x, y, z):
        return P.setdefault((x - .5, y - .5, z - .5), len(P))

    def quad(a, b, c, d):
        F.extend([[a, b, c], [a, c, d]])
    a, b, cc, d = s(0, 0, 0), s(1, 0, 0), s(1, 0, 1), s(0, 0, 1)
    if split_mid:
        m = s(.5, 0, .5)
        F.extend([[a, b, m], [b, cc, m], [cc, d, m], [d, a, m]])
    else:
        quad(a, b, cc, d)
    quad(s(0, 1, 0), s(0, 1, 1), s(1, 1, 1), s(1, 1, 0))
    quad(s(0, 0, 0), s(0, 0, 1), s(0, 1, 1), s(0, 1, 0))
    quad(s(1, 0, 0), s(1, 1, 0), s(1, 1, 1), s(1, 0, 1))
    quad(s(0, 0, 0), s(0, 1, 0), s(1, 1, 0), s(1, 0, 0))
    quad(s(0, 0, 1), s(1, 0, 1), s(1, 1, 1), s(0, 1, 1))
    if split_mid:
        F.extend([[s(0, 0, 0), m, s(1, 1, 1)], [m, s(1, 0, 1), s(1, 1, 1)],
                  [s(0, 0, 0), s(1, 1, 1), s(0, 1, 0)]])
    else:
        F.extend([[s(0, 0, 0), s(1, 0, 1), s(1, 1, 1)], [s(0, 0, 0), s(1, 1, 1), s(0, 1, 0)]])
    return np.array(sorted(P, key=P.get), float), np.array(F)


@pytest.mark.parametrize("split_mid", [False, True])
def test_one_sided_wall_unbalanced(split_mid):
    # two cells of 0.5 on either side of the wall; the input's winding number
    # is fractional there (not piecewise constant): no w per cell
    V, F = cube_with_wall(split_mid)
    c, w = check(V, F, [0.5, 0.5], None)
    assert np.all(np.abs(w[1:] - np.round(w[1:])) > 0.1)
    wall = (c.below != 0) & (c.above != 0)
    assert wall.any() and np.all(np.abs(c.multiplicity[wall]) == 1)
    # the cube's faces: material inside each cell next to the outer cell
    out = ~wall
    inner = np.where(c.below[out] == 0, c.above[out], c.below[out])
    assert set(inner.tolist()) == {1, 2}


def test_open_sheet_both_sides():
    # an open square written both ways: no bounded cell (its zero-thickness
    # cell between the copies is merged away), one triangle stack per face
    sq = (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
          np.array([[0, 1, 2], [0, 2, 3]]))
    c = cells(*joined(sq, (sq[0], sq[1][:, ::-1])))
    assert c.num_cells == 1
    assert len(c.triangles) == 2 and np.all(c.below == 0) and np.all(c.above == 0)
    assert c.winding.tolist() == [0]


def test_empty_and_collapsed():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1e-10]], float)
    c = cells(v, np.array([[0, 1, 2], [0, 1, 3]]))             # collinear; collapsed
    assert c.num_cells == 1 and len(c.triangles) == 0
