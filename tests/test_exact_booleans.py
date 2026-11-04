"""
Tests for the bounded path (`exact_booleans=True`): every boolean in CGAL, the
result method "cgal" -- and for the default path, a CGAL solid that float64
rounding left with face pairs stays in CGAL instead of going to manifold.
"""

import numpy as np
import pytest
import trimesh

from volumetric_iou import (Body, Status, load_body, mesh_iou, normalize_pair, prepare_side,
                            save_body, to_body, volume_iou)

from shapes import coiled_bar, shallow_crossing


def box(extents=(1, 1, 1), translate=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    return m


def cross_and_plate():
    """GT: two overlapping boxes (union volume 0.040); pred: a plate (0.032);
    their intersection 0.024 -- IoU 0.5 exactly."""
    gt = trimesh.util.concatenate([box((0.6, 0.2, 0.2)), box((0.2, 0.6, 0.2), (0.1, 0, 0))])
    return gt, box((0.4, 0.4, 0.2))


def hollow():
    void = box((0.4, 0.4, 0.4))
    void.invert()
    return trimesh.util.concatenate([box((0.8, 0.8, 0.8)), void])


def test_embedded():
    assert mesh_iou._embedded(box())
    # two boxes touching on part of a face (coplanar faces overlap): a touch is a pair
    touch = trimesh.util.concatenate([box((0.2, 0.2, 0.2)),
                                      box((0.2, 0.2, 0.2), (0.05, 0.2, 0.05))])
    assert not mesh_iou._embedded(touch)


def test_embedded_detector_failure_counts_as_pairs(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("detector down")
    monkeypatch.setattr(mesh_iou, "self_intersecting_pairs", boom)
    assert not mesh_iou._embedded(box())


def test_rounded_cgal_solid_with_pairs_stays_in_cgal():
    """The coiled bar's CGAL solid, rounded to float64, has crossing slivers
    near x = 0 (coordinates 3e-17 against 1.5e-16): manifold does not get it,
    the intersection is CGAL's."""
    gt = coiled_bar()
    solid = to_body(gt, frame="bbox")
    assert not mesh_iou._embedded(trimesh.Trimesh(solid.vertices, solid.faces, process=False))
    res = volume_iou(*normalize_pair(gt, box((2.4, 2.4, 0.6), (0, 0, 0.1)), pred_frame="gt"))
    assert res and res.method == "cgal", res


def test_exact_booleans_union_and_intersection():
    gt, pred = cross_and_plate()
    fast = volume_iou(gt, pred)
    exact = volume_iou(gt, pred, exact_booleans=True)
    assert fast and fast.method == "boolean:manifold64"
    assert exact and exact.method == "cgal", exact
    assert exact.diagnostics["exact_booleans"] is True
    assert exact.diagnostics["gt"]["exact_booleans"] is True
    assert abs(exact.iou - 0.5) <= 1e-12 and abs(exact.diagnostics["volume_gt"] - 0.04) <= 1e-12
    assert abs(exact.iou - fast.iou) <= 1e-12
    # every body is CGAL's, but no crossing: no "resolved by CGAL" warning
    assert exact.warnings == fast.warnings


def test_exact_booleans_zero_is_cgal():
    res = volume_iou(box((0.4, 0.4, 0.4)), box((0.4, 0.4, 0.4), (0.4, 0, 0)), exact_booleans=True)
    assert res and res.iou == 0.0 and res.method == "cgal"
    assert res.diagnostics["zero_proof"] == "cgal"


def test_exact_booleans_fill_cavities():
    res = volume_iou(hollow(), box((0.8, 0.8, 0.8)), fill_cavities=True, exact_booleans=True)
    assert res and res.method == "cgal"
    assert abs(res.iou - 1.0) <= 1e-12
    assert res.diagnostics["gt"]["cavities_filled"] == 1


def test_exact_booleans_prepared_side_must_match():
    gt, pred = cross_and_plate()
    sg = prepare_side(gt, "gt")
    with pytest.raises(ValueError, match="exact_booleans"):
        volume_iou(sg, pred, exact_booleans=True)
    se = prepare_side(gt, "gt", exact_booleans=True)
    assert volume_iou(se, pred, exact_booleans=True).iou == volume_iou(
        gt, pred, exact_booleans=True).iou
    with pytest.raises(ValueError):
        prepare_side(gt, "gt", exact_booleans=1)


def test_exact_booleans_body_must_be_built_so(tmp_path):
    gt, pred = cross_and_plate()
    plain = load_body(save_body(to_body(gt, frame="bbox"), tmp_path / "a.body.ply"))
    res = volume_iou(plain, pred, exact_booleans=True)
    assert res.status is Status.LOAD_FAILED and "exact_booleans" in res.detail
    body = to_body(gt, frame="bbox", exact_booleans=True)
    assert isinstance(body, Body) and body.meta["engine"] == "cgal"
    body = load_body(save_body(body, tmp_path / "b.body.ply"))
    a = volume_iou(*normalize_pair(body, pred, pred_frame="gt"), exact_booleans=True)
    b = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"), exact_booleans=True)
    assert a and a.method == "cgal" and abs(a.iou - b.iou) <= 1e-15
    # a body built so is taken by a default call too (its solid stays in CGAL)
    assert volume_iou(*normalize_pair(body, pred, pred_frame="gt")).method == "cgal"


def test_exact_booleans_with_timeout():
    gt, pred = cross_and_plate()
    res = volume_iou(gt, pred, exact_booleans=True, timeout=60)
    assert res and res.method == "cgal" and abs(res.iou - 0.5) <= 1e-12


def test_exact_booleans_fallback_unites_in_cgal(monkeypatch):
    """A cluster union failing its probe check is redone body by body, and with
    exact_booleans the bodies' solids are united in CGAL, never in manifold."""
    gt, pred = cross_and_plate()
    want = volume_iou(gt, pred, exact_booleans=True)
    real, calls = mesh_iou._lost_probes, []

    def once_wrong(pts, raws, out):
        calls.append(len(out.faces))
        return (len(pts), 1) if len(calls) == 1 else real(pts, raws, out)

    def no_manifold(*a, **k):
        raise AssertionError("manifold used with exact_booleans")
    monkeypatch.setattr(mesh_iou, "_lost_probes", once_wrong)
    monkeypatch.setattr(mesh_iou, "_union", no_manifold)
    monkeypatch.setattr(mesh_iou, "_to_manifold", no_manifold)
    res = volume_iou(gt, pred, exact_booleans=True)
    assert res and res.method == "cgal" and abs(res.iou - want.iou) <= 1e-12


def test_cgal_solid_keeps_bodies_outside_a_shallow_crossing():
    """A body apart from the others, seen past a shallow crossing of two
    bodies, stays in the CGAL union (the arrangement used to take it as nested
    and drop it) -- and the exact path gives a number, not boolean_failed."""
    m, bare = shallow_crossing(), shallow_crossing(with_cubes=False)
    solid, vol = mesh_iou._cgal_solid(m)[:2]
    cubes = [c for c in m.split(only_watertight=False) if c.extents.max() < 0.03]
    centres = np.array([c.bounds.mean(axis=0) for c in cubes])
    assert len(cubes) == 18 and (mesh_iou._winding(solid, centres) > 0.5).all()
    # the cubes stand in pairs, side by side in y and overlapping: union per pair
    ys = sorted({round(c[1], 6) for c in centres})
    gaps = [b - a for a, b in zip(ys[::2], ys[1::2])]
    want = sum(0.02 * 0.02 * (0.02 + g) * 3 for g in gaps)
    assert abs(vol - mesh_iou._cgal_solid(bare)[1] - want) <= 1e-12
    # into the unit cube by 1/16: a power of two, the same arrangement
    small = [x.copy().apply_scale(1 / 16) for x in (bare, m)]
    res = volume_iou(*small, exact_booleans=True)
    assert res and res.method == "cgal", res
    assert abs(res.iou - mesh_iou._cgal_solid(bare)[1] / vol) <= 1e-12


def test_cgal_solid_keeps_body_over_coincident_faces():
    """A cube above a face that two bodies share (coincident faces, a cell of
    no thickness between them) lies in the outer cell, not in that one."""
    m = trimesh.util.concatenate([box((1, 1, 1), (0.5, 0.5, 0.5)),
                                  box((0.6, 0.6, 0.5), (0.5, 0.5, 0.75)),
                                  box((0.1, 0.1, 2), (0.05, 0.05, 1)),
                                  box((0.05, 0.05, 0.05), (0.4, 0.6, 1.2))])
    solid, vol = mesh_iou._cgal_solid(m)[:2]
    assert mesh_iou._winding(solid, np.array([[0.4, 0.6, 1.2]]))[0] > 0.5
    assert abs(vol - (1 + 0.01 + 0.05 ** 3)) <= 1e-12


def test_exact_booleans_key_on_disjoint_boxes():
    res = volume_iou(box((0.2, 0.2, 0.2)), box((0.2, 0.2, 0.2), (0.3, 0, 0)), exact_booleans=True)
    assert res and res.method == "bounds" and res.diagnostics["exact_booleans"] is True
