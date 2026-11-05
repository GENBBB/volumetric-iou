"""Thread defaults, caller overrides and libigl's first-use cache."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import trimesh

from volumetric_iou import self_intersect


@pytest.mark.parametrize("setting,expected", [(None, 1), ("4", 4), ("0", 1), ("bad", 1)])
def test_native_thread_default_in_fresh_process(setting, expected):
    env = os.environ.copy()
    env.pop("IGL_NUM_THREADS", None)
    if setting is not None:
        env["IGL_NUM_THREADS"] = setting
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json, os
before = os.environ.get('IGL_NUM_THREADS')
from volumetric_iou import _cgal
print(json.dumps([_cgal.default_num_threads(), before, os.environ.get('IGL_NUM_THREADS')]))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout) == [expected, setting, setting]


def test_native_threads_read_after_import_and_cached_on_first_use():
    env = os.environ.copy()
    env.pop("IGL_NUM_THREADS", None)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    code = """
import json, os
from volumetric_iou import _cgal
os.environ['IGL_NUM_THREADS'] = '2'
first = _cgal.default_num_threads()
os.environ['IGL_NUM_THREADS'] = '4'
print(json.dumps([first, _cgal.default_num_threads()]))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout) == [2, 2]


def test_detector_default_environment_and_per_call_override(monkeypatch):
    mesh = trimesh.creation.box()
    budgets = []

    def capture(vertices, faces, threads, first):
        budgets.append(threads)
        return np.empty((0, 2), dtype=np.int64)

    monkeypatch.setattr(self_intersect._cgal, "self_intersecting_pairs", capture)
    monkeypatch.delenv("IGL_NUM_THREADS", raising=False)
    self_intersect.self_intersecting_pairs(mesh)
    monkeypatch.setenv("IGL_NUM_THREADS", "4")
    self_intersect.self_intersecting_pairs(mesh)
    self_intersect.self_intersecting_pairs(mesh, threads=2)
    monkeypatch.setenv("IGL_NUM_THREADS", "bad")
    self_intersect.self_intersecting_pairs(mesh)
    assert budgets == [1, 4, 2, 1]
