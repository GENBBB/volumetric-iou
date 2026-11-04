"""
Tests of the metric's definition (docs/DEFINITION.md) beyond
test_mesh_iou.py: the solid is the set w >= 1, orientation per 4.6, iog/iop,
the time limit, and agreement with the winding-number arbiter.
"""

import os
import signal
import threading
import time

import numpy as np
import pytest
import trimesh

from volumetric_iou import Status, normalize_pair, volume_iou
from volumetric_iou.winding import arbiter_iou

TOL = 1e-6


def box(extents=(1, 1, 1), translate=(0, 0, 0)):
    return trimesh.creation.box(extents=extents).apply_translation(translate)


def test_iog_and_iop():
    # pred sticks out of gt: gt fully covered (iog 1), half of pred wasted (iop 0.5)
    res = volume_iou(box(), box((2, 1, 1), translate=(0.5, 0, 0)))
    assert res.iou == pytest.approx(0.5, rel=TOL)
    assert res.iog == pytest.approx(1.0, rel=TOL)
    assert res.iop == pytest.approx(0.5, rel=TOL)


def test_body_inside_body_is_material_not_cavity():
    """4.6: two outward shells, one inside the other -> w = 2 inside, still material."""
    nest = trimesh.util.concatenate([box((2, 2, 2)), box()])
    res = volume_iou(*normalize_pair(nest, box((2, 2, 2)), pred_frame="gt"))
    assert res.iou == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["cavities"] == 0
    assert res.diagnostics["gt"]["union"]


def test_inward_shell_outside_any_body_is_flipped():
    """An inside-out body standing on its own is a body, not a cavity of nothing."""
    lone = box(translate=(5, 0, 0))
    lone.invert()
    two = trimesh.util.concatenate([box(), lone])
    res = volume_iou(*normalize_pair(two, trimesh.util.concatenate([box(), box(translate=(5, 0, 0))]), pred_frame="gt"))
    assert res.iou == pytest.approx(1.0, rel=TOL)
    assert res.diagnostics["gt"]["flipped_shells"] == 1
    assert any("inside-out bodies flipped" in w for w in res.warnings)


def test_whole_mesh_inside_out_is_flipped():
    inv = box()
    inv.invert()
    res = volume_iou(inv, box(translate=(0.5, 0, 0)))
    assert res.iou == pytest.approx(1 / 3, rel=TOL)
    assert res.diagnostics["gt"]["flipped_inverted_mesh"]
    # the warning as 4.6 words it; no body flipped on its own
    inv_pred = box(translate=(0.5, 0, 0))
    inv_pred.invert()
    for side, res in (("gt", res), ("pred", volume_iou(box(), inv_pred))):
        assert res.iou == pytest.approx(1 / 3, rel=TOL)
        assert [w for w in res.warnings if "inside" in w] == [f"{side}: mesh was inside out; flipped"]


def test_inconsistent_winding_falls_back_to_parity():
    """4.6 fallback: orientation is meaningless when a shell is not consistently wound."""
    outer = box((2, 2, 2))
    inner = box()
    inner.faces[0] = inner.faces[0][::-1]          # one face against the rest
    m = trimesh.util.concatenate([outer, inner])
    res = volume_iou(*normalize_pair(m, box((2, 2, 2)), pred_frame="gt"))
    assert res.diagnostics["gt"]["orientation"] == "parity"
    assert res.iou == pytest.approx(7 / 8, rel=1e-6)   # parity reads the inner shell as a cavity
    assert any("nesting parity" in w for w in res.warnings)


@pytest.mark.parametrize("parity", [False, True])
def test_ambiguous_containment(parity):
    """
    4.6: an inward shell too thin to place a probe inside (1e-7 thick), in a
    box: whether it is inside is not decided.  With consistent winding the
    orientation is trusted -- a cavity; by nesting parity the pair does not
    count -- an outward body at depth 0, inside the box.
    """
    from volumetric_iou import mesh_iou
    thin = box((1, 1, 1e-7))
    assert mesh_iou._interior_points(thin) is None
    thin.invert()
    host = box((2, 2, 2))
    if parity:
        host.faces[0] = host.faces[0][::-1]
    res = volume_iou(*normalize_pair(trimesh.util.concatenate([host, thin]), box((2, 2, 2)),
                                     pred_frame="gt"))
    assert res
    g = res.diagnostics["gt"]
    assert g["ambiguous_containment"] == 1
    assert any("1 touching shells, containment ambiguous" in w for w in res.warnings)
    assert g["orientation"] == ("parity" if parity else "given")
    assert g["cavities"] == (0 if parity else 1)
    assert res.iou == pytest.approx(1.0 if parity else 1 - 1e-7 / 8, rel=1e-12)


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_cavity_without_a_host_is_refused(mode):
    """
    4.6, nesting parity: a small shell inside three overlapping boxes, none
    of which holds another, is at depth 3 -- a cavity with no shell at depth
    2 to belong to.  A refusal of the side in either mode; with consistent
    winding the same shells are three bodies and an island.
    """
    walls = [box((2, 2, 2)), box((2, 2, 2), translate=(0.5, 0, 0)),
             box((2, 2, 2), translate=(0, 0.5, 0))]
    core = box((0.2, 0.2, 0.2), translate=(0.25, 0.25, 0))
    walls[0].faces[0] = walls[0].faces[0][::-1]          # inconsistent: parity
    side = trimesh.util.concatenate(walls + [core])
    res = volume_iou(*normalize_pair(side, box(), pred_frame="gt"), on_self_intersection=mode)
    assert res.status is Status.SELF_INTERSECTING and res.blame == "gt"
    assert "1 cavity shells have no enclosing body" in res.detail
    walls[0].faces[0] = walls[0].faces[0][::-1]
    side = trimesh.util.concatenate(walls + [core])
    assert volume_iou(*normalize_pair(side, box(), pred_frame="gt"), on_self_intersection=mode)


@pytest.mark.parametrize("case", ["overlap", "cavity", "island"])
def test_agrees_with_arbiter(case):
    if case == "overlap":
        g = trimesh.util.concatenate([box(), box(translate=(0.5, 0.3, 0)),
                                      box((0.4, 0.4, 2), translate=(0.2, 0, 0))])
    elif case == "cavity":
        void = box((1, 1, 1), translate=(0.2, 0, 0))
        void.invert()
        g = trimesh.util.concatenate([box((2, 2, 2)), void])
    else:
        void = box((1.2, 1.2, 1.2))
        void.invert()
        g = trimesh.util.concatenate([box((2, 2, 2)), void, box((0.5, 0.5, 0.5))])
    p = box((1.5, 1.5, 1.5), translate=(0.3, 0.2, 0.1))
    res = volume_iou(*normalize_pair(g, p, pred_frame="gt"))
    assert res, f"{res.status.value}: {res.detail}"
    arb = arbiter_iou(g, p, n=20_000)
    assert abs(res.iou - arb.iou) < 4 * arb.se + 1e-9


# Stand-ins for volume_iou in the isolated child.  The child is forked from
# the package's server, not from the test process, so a patch of a function
# the computation calls does not reach it; the function the child runs is
# replaced instead (volume_iou sends itself by name, looked up at call time).

def _sleeps(gt, pred, **kwargs):
    time.sleep(30)


def _killed(gt, pred, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)


def _raises(gt, pred, **kwargs):
    raise MemoryError("out of memory")


@pytest.mark.parametrize("isolation", ["server", "fork"])
def test_timeout_is_a_status(monkeypatch, isolation):
    from volumetric_iou import mesh_iou

    monkeypatch.setattr(mesh_iou, "volume_iou", _sleeps)
    t = time.perf_counter()
    res = volume_iou(box(), box(translate=(0.5, 0, 0)), timeout=1.0, isolation=isolation)
    assert res.status is Status.TIMEOUT and res.blame == "engine"
    assert time.perf_counter() - t < 10


@pytest.mark.parametrize("isolation", ["server", "fork"])
@pytest.mark.parametrize("how", ["killed", "raised"])
def test_crashed_is_a_status(monkeypatch, how, isolation):
    """
    The isolated child killed by a signal (a segfault or the OOM killer in
    native code) or raising past volume_iou's own handling: CRASHED, blamed
    on the engine, with the cause of death in the detail.
    """
    from volumetric_iou import Cause, mesh_iou

    monkeypatch.setattr(mesh_iou, "volume_iou", _killed if how == "killed" else _raises)
    res = volume_iou(box(), box(translate=(0.5, 0, 0)), timeout=30, isolation=isolation)
    assert not res and res.status is Status.CRASHED
    assert res.blame == "engine" and res.cause is Cause.ENGINE and res.iou is None
    assert ("SIGKILL" if how == "killed" else "MemoryError: out of memory") in res.detail


_HELD = threading.Lock()


def _takes_the_lock():
    return os.getppid(), _HELD.acquire(timeout=5)


def test_child_is_not_forked_from_the_caller():
    """
    A fork copies the memory of the caller's threads, not the threads: a lock
    held by one of them at the fork -- or a thread pool's count of busy
    workers, as oneTBB's in the PyPI manifold3d right after a boolean -- stays
    held in the child forever.  The child comes from a server that has not
    run them, so it starts with the lock free.
    """
    from volumetric_iou import isolate

    release = threading.Event()

    def hold():
        with _HELD:
            release.wait(30)

    t = threading.Thread(target=hold)
    t.start()
    try:
        while not _HELD.locked():
            time.sleep(0.01)
        out = isolate.run_isolated(_takes_the_lock, (), {}, 30)
    finally:
        release.set()
        t.join()
    assert out["status"] == "ok", out
    parent, took = out["value"]
    assert took and parent != os.getpid() and parent == isolate._server.pid


def test_fork_isolation_forks_the_caller():
    """
    isolation="fork": the child is the caller's own, and the call is not
    pickled (a lambda reaches it).  An unknown mode raises before any child.
    """
    from volumetric_iou import isolate

    me = os.getpid()
    out = isolate.run_forked(lambda: os.getppid(), (), {}, 30)
    assert out == {"status": "ok", "value": me}
    with pytest.raises(ValueError, match="isolation"):
        volume_iou(box(), box(), timeout=30, isolation="spawn")


def test_timeout_after_booleans_in_the_caller():
    """
    Calls with a timeout right after in-process ones, which leave manifold3d's
    thread pool (if it has one) busy: a fork of the caller hung about once in
    600-1000 such calls.  A few rounds only check the path; the test above is
    the deterministic one.
    """
    a = trimesh.creation.icosphere(subdivisions=4, radius=0.35)
    b = a.copy().apply_translation((0.1, 0.05, 0))
    for _ in range(8):
        ref = volume_iou(a, b)
        res = volume_iou(a, b, timeout=30)
        assert res.status is Status.OK and res.iou == ref.iou


_CLEAN_WORKER = """
import json, sys, trimesh
from volumetric_iou import volume_iou
print("buffered")                     # not flushed: a child must not write it again
a = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
b = a.copy().apply_translation((0.1, 0.05, 0))
forked = [volume_iou(a, b, timeout=60, isolation="fork") for _ in range(3)]
ref = volume_iou(a, b)
json.dump({"status": [r.status.value for r in forked], "iou": [r.iou for r in forked],
           "ref": ref.iou}, sys.stdout)
"""


def test_fork_isolation_in_a_clean_worker():
    """
    isolation="fork" as meant: a process that computes nothing in-process
    (the in-process reference comes after the forked calls), the same result
    to the bit; the caller's buffered output is written once.
    """
    import json
    import subprocess
    import sys

    run = subprocess.run([sys.executable, "-c", _CLEAN_WORKER], capture_output=True,
                         text=True, timeout=300, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert run.returncode == 0, run.stderr
    head, _, body = run.stdout.partition("\n")
    assert head == "buffered" and "buffered" not in body, run.stdout
    out = json.loads(body)
    assert out["status"] == ["ok"] * 3 and out["iou"] == [out["ref"]] * 3, out


def test_isolated_run_returns_the_same_result():
    a, b = box(), box(translate=(0.5, 0, 0))
    r1, r2 = volume_iou(a, b), volume_iou(a, b, timeout=60)
    assert r2.iou == pytest.approx(r1.iou, rel=1e-12) and r2.iog == pytest.approx(r1.iog)


def test_float64_engine_keeps_a_thin_sliver():
    """
    trimesh hands manifold float32, and a 1e-6 overlap
    came out 1.3% off.  The direct Mesh64 path keeps it.
    """
    eps = 1e-6
    a = box()
    b = box(translate=(1 - eps, 0, 0))
    res = volume_iou(a, b)
    expected = eps / (2 - eps)
    assert res.iou == pytest.approx(expected, rel=1e-3)
