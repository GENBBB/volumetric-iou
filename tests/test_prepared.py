"""
Tests for a prepared side (`prepare_side`): one side loaded and built once,
then passed to many `volume_iou` calls with the same result as the mesh.
"""

import pickle

import numpy as np
import pytest
import trimesh

from volumetric_iou import (PreparedSide, Status, normalize_pair, prepare_side, save_body,
                            to_body, volume_iou)

from shapes import coiled_bar


def box(extents=(1, 1, 1), translate=(0, 0, 0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    return m


def hollow():
    void = box((1, 1, 1))
    void.invert()
    return trimesh.util.concatenate([box((2, 2, 2)), void])


def opened():
    m = box()
    m.update_faces(np.arange(len(m.faces)) > 1)      # one side missing
    return m


def _cases():
    """name -> (gt, pred, volume_iou options), the pair in the gt frame."""
    overlapping = trimesh.util.concatenate([box((2, 1, 1)), box((1, 2, 1), translate=(0.3, 0, 0))])
    raw = {
        "clean": (box((2, 1, 1)), box((1, 1, 1), translate=(0.5, 0.2, 0)), {}),
        "union": (overlapping, box((1.5, 1.5, 1.5), translate=(0.4, 0.2, 0)), {}),
        "cgal_gt": (coiled_bar(), box((2.4, 2.4, 0.6), translate=(0, 0, 0.1)), {}),
        "pred_refused": (box((2.4, 2.4, 0.6)), coiled_bar(), {}),
        "gt_refused": (coiled_bar(), box((2.4, 2.4, 0.6)), {"on_self_intersection": "fail"}),
        "fill": (hollow(), box((2, 2, 1.5), translate=(0, 0, 0.2)),
                 {"fill_cavities": (True, False)}),
        "disjoint": (box(), box(translate=(3, 0, 0)), {}),
        "no_check": (box((2, 1, 1)), box((1, 1, 1), translate=(0.5, 0.2, 0)),
                     {"check_self_intersection": False}),
    }
    out = {}
    for name, (g, p, opts) in raw.items():
        g, p = normalize_pair(g, p, pred_frame="gt")
        out[name] = (g, p, opts)
    return out


def _side_opts(opts, side):
    i = ("gt", "pred").index(side)
    mode = opts.get("on_self_intersection", ("exact", "fail"))
    fill = opts.get("fill_cavities", False)
    return dict(check_self_intersection=opts.get("check_self_intersection", True),
                on_self_intersection=mode if isinstance(mode, str) else mode[i],
                fill_cavities=fill if isinstance(fill, bool) else fill[i])


def _timeless(d):
    """Diagnostics without wall times ("seconds" of a CGAL build), which vary by run."""
    if isinstance(d, dict):
        return {k: _timeless(v) for k, v in d.items() if k != "seconds"}
    return d


def _identical(a, b):
    assert (a.status, a.iou, a.iog, a.iop, a.method, a.detail, a.blame) == \
           (b.status, b.iou, b.iog, b.iop, b.method, b.detail, b.blame)
    assert a.warnings == b.warnings
    assert repr(_timeless(a.diagnostics)) == repr(_timeless(b.diagnostics))


@pytest.mark.parametrize("case", sorted(_cases()))
def test_prepared_sides_give_the_same_result(case):
    gt, pred, opts = _cases()[case]
    ref = volume_iou(gt, pred, **opts)
    pg = prepare_side(gt, "gt", **_side_opts(opts, "gt"))
    pp = prepare_side(pred, "pred", **_side_opts(opts, "pred"))
    for g, p in ((pg, pred), (gt, pp), (pg, pp)):
        for _ in range(2):           # a call does not change the prepared side
            _identical(volume_iou(g, p, **opts), ref)


def test_result_does_not_share_state_with_the_side():
    gt, pred, _ = _cases()["union"]
    pg = prepare_side(gt, "gt")
    first = volume_iou(pg, pred)
    first.diagnostics["gt"]["solids"] = -1
    first.warnings.append("x")
    _identical(volume_iou(pg, pred), volume_iou(gt, pred))


def test_refusal_is_kept_and_not_shared():
    gt, pred, _ = _cases()["pred_refused"]
    pp = prepare_side(pred, "pred")
    assert pp.error.status is Status.SELF_INTERSECTING and pp.error.blame == "pred"
    first = volume_iou(gt, pp)
    first.warnings.append("x")
    _identical(volume_iou(gt, pp), volume_iou(gt, pred))
    assert prepare_side(gt, "gt").error is None


@pytest.mark.parametrize("bad", ["open", "out_of_cube"])
def test_refused_gt_side(bad):
    pred = box((0.5, 0.5, 0.5))
    gt = opened() if bad == "open" else box((2, 1, 1))
    pg = prepare_side(gt, "gt")
    want = Status.OPEN_SURFACE if bad == "open" else Status.NOT_NORMALIZED
    assert pg.error.status is want
    _identical(volume_iou(pg, pred), volume_iou(gt, pred))


def test_body_path_side(tmp_path):
    gt, pred, _ = _cases()["cgal_gt"]
    path = save_body(to_body(gt, frame="bbox", label="gt"), tmp_path / "gt.body.ply")
    pg = prepare_side(path, "gt")
    assert pg.body is not None
    _identical(volume_iou(pg, pred), volume_iou(path, pred))
    path.unlink()                    # read once: the file is no longer needed
    assert volume_iou(pg, pred)
    bad = prepare_side(tmp_path / "missing.body.ply", "gt")
    assert bad.error.status is Status.LOAD_FAILED


@pytest.mark.parametrize("frame", ["gt", "bbox", "div200"])
def test_normalize_pair_places_the_pred_by_the_body_frame(frame):
    gt, pred = coiled_bar(), box((2.4, 2.4, 0.6), translate=(5, 5, 0.1))
    body = to_body(gt, frame="bbox", label="gt")
    pg = prepare_side(body, "gt")
    g, p = normalize_pair(pg, pred, pred_frame=frame)
    assert g is pg
    _identical(volume_iou(g, p), volume_iou(*normalize_pair(body, pred, pred_frame=frame)))
    with pytest.raises(ValueError, match="from a body"):
        normalize_pair(prepare_side(normalize_pair(gt, pred, pred_frame="gt")[0], "gt"),
                       pred, pred_frame=frame)


def test_options_must_match_the_call():
    gt, pred, _ = _cases()["clean"]
    pg = prepare_side(gt, "gt")
    assert isinstance(pg, PreparedSide) and pg.on_self_intersection == "exact"
    assert prepare_side(pred, "pred").on_self_intersection == "fail"
    with pytest.raises(ValueError, match="prepared side"):
        volume_iou(pg, pred, on_self_intersection="fail")
    with pytest.raises(ValueError, match="prepared side"):
        volume_iou(pg, pred, fill_cavities=True)
    with pytest.raises(ValueError, match="prepared side"):
        volume_iou(pg, pred, check_self_intersection=False)
    with pytest.raises(ValueError, match="prepared side"):
        volume_iou(gt, prepare_side(pred, "gt"))
    # checked before the isolated child is started: ValueError, not "crashed"
    with pytest.raises(ValueError, match="prepared side"):
        volume_iou(pg, pred, on_self_intersection="fail", timeout=60)
    with pytest.raises(ValueError):
        prepare_side(gt, "both")
    with pytest.raises(ValueError):
        prepare_side(gt, "gt", on_self_intersection="repair")


@pytest.mark.parametrize("case", sorted(_cases()))
def test_prepared_side_under_timeout(case):
    """
    A prepared side reaches the isolated child as a pickle: its solid (a
    manifold3d object) does not pickle, so the child builds it again from the
    loaded source -- to the bit the result of the mesh.
    """
    gt, pred, opts = _cases()[case]
    pg = prepare_side(gt, "gt", **_side_opts(opts, "gt"))
    pp = prepare_side(pred, "pred", **_side_opts(opts, "pred"))
    ref = volume_iou(gt, pred, **opts)
    _identical(volume_iou(pg, pp, timeout=60, **opts), ref)
    _identical(volume_iou(pg, pred, timeout=60, **opts), ref)
    assert repr(pg) == repr(pickle.loads(pickle.dumps(pg)))


def test_inputs_reach_the_isolated_child(tmp_path):
    """Every form a side may take gives the same result with a timeout."""
    gt, pred, _ = _cases()["union"]
    gt.export(tmp_path / "gt.ply")
    body = to_body(gt, frame="bbox", label="gt")
    forms = {"mesh": gt, "path": str(tmp_path / "gt.ply"), "pathlib": tmp_path / "gt.ply",
             "arrays": (np.asarray(gt.vertices), np.asarray(gt.faces)), "body": body,
             "body path": str(save_body(body, tmp_path / "gt.body.ply")),
             "prepared": prepare_side(gt, "gt"), "prepared body": prepare_side(body, "gt")}
    for name, g in forms.items():
        ref = volume_iou(g, pred)
        assert ref, (name, ref)
        _identical(volume_iou(g, pred, timeout=60), ref)
