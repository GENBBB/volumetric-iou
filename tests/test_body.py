"""
Tests for the body format (since 0.4.0): the solid the metric builds, stored and
taken back without repair.
"""

import numpy as np
import pytest
import trimesh

from volumetric_iou import (Body, Status, load_body, normalize_pair, save_body, to_body,
                            volume_iou)
from volumetric_iou import mesh_iou

from shapes import coiled_bar, extruded


def box(extents=(1, 1, 1), translate=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    return m


def _pairs():
    """(gt, pred): clean, multi-body with overlap, self-overlapping (CGAL)."""
    overlapping = trimesh.util.concatenate([box((2, 1, 1)), box((1, 2, 1), translate=(0.3, 0, 0))])
    return {
        "clean": (box((2, 1, 1), translate=(10, 0, 0)), box((1, 1, 1), translate=(10.5, 0.2, 0))),
        "union": (overlapping, box((1.5, 1.5, 1.5), translate=(0.4, 0.2, 0))),
        "cgal": (coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1))),
    }


def _same(a, b):
    """
    Two results equal up to rounding, diagnostics aside.  Not to the bit: the
    body's solid is rebuilt from manifold's own output, whose triangle order
    may differ from the solid it came from, and the intersection volume then
    sums in another order (1 ulp on the server, manifold3d 3.0.0).
    """
    assert a.status is b.status and a.method == b.method
    for x, y in ((a.iou, b.iou), (a.iog, b.iog), (a.iop, b.iop)):
        assert abs(x - y) <= 1e-15
    assert a.warnings == b.warnings


@pytest.mark.parametrize("case", ["clean", "union", "cgal"])
def test_round_trip_is_bitwise(case, tmp_path):
    gt, _ = _pairs()[case]
    body = to_body(gt, frame="bbox")
    assert isinstance(body, Body)
    p1 = save_body(body, tmp_path / "a.body.ply")
    back = load_body(p1)
    assert np.array_equal(back.vertices, body.vertices)
    assert np.array_equal(back.faces, body.faces)
    assert back.meta == body.meta
    p2 = save_body(back, tmp_path / "b.body.ply")
    assert p1.read_bytes() == p2.read_bytes()


@pytest.mark.parametrize("case", ["clean", "union", "cgal"])
def test_gt_body_gives_the_same_result(case, tmp_path):
    gt, pred = _pairs()[case]
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"))
    assert ref, ref
    path = save_body(to_body(gt, frame="bbox", label="gt"), tmp_path / "gt.body.ply")
    for g in (path, load_body(path)):
        _same(volume_iou(*normalize_pair(load_body(path), pred, pred_frame="gt")), ref)
        _same(volume_iou(g, normalize_pair(gt, pred, pred_frame="gt")[1]), ref)


@pytest.mark.parametrize("frame", ["gt", "bbox", "div200"])
def test_pred_body_in_the_pair_frame(frame, tmp_path):
    gt, pred = _pairs()["cgal"]
    pred = pred.copy()
    pred.apply_scale(100.0 if frame == "div200" else 1.0)
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame=frame))
    assert ref, ref
    gb = to_body(gt, frame="bbox", label="gt")
    pb = to_body(pred, frame=frame, gt=gb if frame == "gt" else None, label="pred")
    pb = load_body(save_body(pb, tmp_path / "p.body.ply"))
    _same(volume_iou(*normalize_pair(gb, pb, pred_frame=frame)), ref)
    _same(volume_iou(*normalize_pair(gt, pb, pred_frame=frame)), ref)
    # the pred body from the raw GT mesh is the same body
    if frame == "gt":
        assert to_body(pred, frame="gt", gt=gt).meta == pb.meta


def test_pred_body_of_another_frame_is_refused():
    gt, pred = _pairs()["clean"]
    gb = to_body(gt, frame="bbox")
    other = to_body(box((3, 3, 3)), frame="bbox")
    pb = to_body(pred, frame="gt", gt=other)
    with pytest.raises(ValueError, match="another GT"):
        normalize_pair(gb, pb, pred_frame="gt")
    with pytest.raises(ValueError):
        normalize_pair(gb, pb, pred_frame="bbox")
    res = volume_iou(gb, pb)
    assert res.status is Status.NOT_NORMALIZED


def test_refused_mesh_gives_its_refusal():
    opened = box()
    opened.update_faces(np.arange(len(opened.faces)) > 1)      # one side missing
    res = to_body(opened, frame="bbox", label="gt")
    assert not res and res.status is Status.OPEN_SURFACE and res.blame == "gt"


def test_inverted_pocket_has_no_body():
    """A surface winding -1 somewhere defines no solid: no body to store."""
    loop = [(0, 0), (4, 0), (4, 1), (3, 1), (3, 2), (6, 2), (6, 1), (4.5, 1), (4.5, 4), (0, 4)]
    res = to_body(extruded(loop), frame="bbox", label="gt")
    assert not res and res.status is Status.INVERTED_VOLUME and res.blame == "gt"


def test_body_is_not_cleaned():
    """A body is taken as built: _clean is never called on it."""
    gt, pred = _pairs()["union"]
    gb = to_body(gt, frame="bbox")
    calls = []
    orig = mesh_iou._clean
    mesh_iou._clean = lambda m, fill=True, **kw: calls.append(1) or orig(m, fill, **kw)
    try:
        res = volume_iou(gb, normalize_pair(gt, pred, pred_frame="gt")[1])
    finally:
        mesh_iou._clean = orig
    assert res and len(calls) == 1          # the pred only


def test_edited_file_is_refused(tmp_path):
    path = save_body(to_body(box(), frame="bbox"), tmp_path / "x.body.ply")
    data = bytearray(path.read_bytes())
    data[-20] ^= 1                          # a vertex index bit
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="changed"):
        load_body(path)
    res = volume_iou(path, box((0.5, 0.5, 0.5)))
    assert res.status is Status.LOAD_FAILED and res.blame == "gt"


def test_not_a_body_file(tmp_path):
    path = tmp_path / "y.body.ply"
    box().export(path)
    with pytest.raises(ValueError, match="not a volumetric-iou body"):
        load_body(path)


def test_wrong_recorded_volume_is_refused():
    gb = to_body(box(), frame="bbox")
    gb.meta["volume"] *= 1 + 1e-9
    res = volume_iou(gb, box((0.5, 0.5, 0.5)))
    assert res.status is Status.LOAD_FAILED and "volume" in res.detail


def test_fail_mode_refuses_a_body_with_crossings():
    """A body built in "exact" with crossings inside a body is refused in
    "fail": whether its source passes through itself was not looked for
    there (its source does: refused as passing through itself)."""
    gt, pred = _pairs()["cgal"]
    gb = to_body(gt, frame="bbox")
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"), on_self_intersection="fail")
    res = volume_iou(*normalize_pair(gb, pred, pred_frame="gt"), on_self_intersection="fail")
    assert ref.status is res.status is Status.SELF_INTERSECTING
    assert "winds 2 or more" in ref.detail
    assert res.detail == mesh_iou.crossing_detail("gt") and res.blame == "gt"


def test_cgal_side_body(monkeypatch, tmp_path):
    """A solid manifold would not take stays a mesh; the intersection is CGAL's."""
    monkeypatch.setattr(mesh_iou, "_resolved_to_manifold", lambda *a: None)
    gt, pred = _pairs()["cgal"]
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"))
    assert ref and ref.method == "cgal"
    gb = to_body(gt, frame="bbox")
    assert gb.meta["engine"] == "cgal" and gb.meta["probes"]
    gb = load_body(save_body(gb, tmp_path / "c.body.ply"))
    _same(volume_iou(*normalize_pair(gb, pred, pred_frame="gt")), ref)


def test_default_repairs_the_gt_body_and_refuses_the_pred():
    """
    The default rule (since 0.6.0): a GT is repaired ("exact"), a prediction must
    be a valid solid ("fail").  A GT body built once with crossings is taken as
    built; the same self-overlapping surface as the pred is refused, raw or as
    a body built for the pred in "fail".
    """
    gt, pred = _pairs()["cgal"]
    gb = to_body(gt, frame="bbox", label="gt")
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"))
    assert ref and ref.method == "cgal", ref     # the rounded solid crosses itself
    _same(volume_iou(*normalize_pair(gb, pred, pred_frame="gt")), ref)

    res = volume_iou(*normalize_pair(pred, gt, pred_frame="gt"))
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"
    refused = to_body(gt, frame="gt", gt=pred, label="pred", on_self_intersection="fail")
    assert not refused and refused.status is Status.SELF_INTERSECTING and refused.blame == "pred"
    # a pred body built in "exact" is refused under the default all the same
    pb = to_body(gt, frame="gt", gt=pred, label="pred")
    res = volume_iou(*normalize_pair(pred, pb, pred_frame="gt"))
    assert res.status is Status.SELF_INTERSECTING and res.blame == "pred"


def test_fail_mode_refuses_a_body_the_detector_failed_on():
    gt, pred = _pairs()["clean"]
    pb = to_body(pred, frame="gt", gt=gt, label="pred")
    pb.meta["info"]["self_intersection_checked"] = None
    pb.meta["info"]["detector_error"] = "RuntimeError: detector down"
    res = volume_iou(*normalize_pair(gt, pb, pred_frame="gt"))
    assert res.status is Status.BOOLEAN_FAILED and res.blame == "engine"
    assert "detector down" in res.detail
    assert volume_iou(*normalize_pair(gt, pb, pred_frame="gt"), on_self_intersection="exact")


def test_filled_cavities_of_a_body_match_the_raw_mesh(tmp_path):
    """fill_cavities works on a stored body as on the mesh it came from."""
    void = box((1, 1, 1))
    void.invert()
    gt = trimesh.util.concatenate([box((2, 2, 2)), void])
    pred = box((2, 2, 1.5), translate=(0, 0, 0.2))
    ref = volume_iou(*normalize_pair(gt, pred, pred_frame="gt"), fill_cavities=(True, False))
    assert ref and ref.diagnostics["gt"]["cavities_filled"] == 1
    path = save_body(to_body(gt, frame="bbox", label="gt"), tmp_path / "gt.body.ply")
    res = volume_iou(path, normalize_pair(gt, pred, pred_frame="gt")[1], fill_cavities=(True, False))
    _same(res, ref)
    assert res.iou == pytest.approx(0.75, rel=1e-6)


def test_body_records_cavities_of_its_solid(tmp_path):
    """cavities_solid is stored in the body, and counted for a body built without it."""
    void = box((1, 1, 1))
    void.invert()
    gt = trimesh.util.concatenate([box((2, 2, 2)), void])
    pred = box((2, 2, 1.5), translate=(0, 0, 0.2))
    body = to_body(gt, frame="bbox", label="gt")
    assert body.meta["info"]["cavities_solid"] == 1
    del body.meta["info"]["cavities_solid"]          # as a body of v0.7.0 and older
    path = save_body(body, tmp_path / "gt.body.ply")
    res = volume_iou(path, normalize_pair(gt, pred, pred_frame="gt")[1])
    assert res and res.diagnostics["gt"]["cavities_solid"] == 1
