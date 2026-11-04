"""Dataset reports follow the filled metric side and keep run modes separate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import trimesh

from test_cells import cube_with_wall


ROOT = Path(__file__).resolve().parents[1]


def dataset_tool():
    spec = importlib.util.spec_from_file_location("dataset_check", ROOT / "tools/dataset_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_filled_cavity_volume_and_unfilled_body(tmp_path, mode):
    outer = trimesh.creation.box(extents=[2, 2, 2])
    cavity = trimesh.creation.box()
    cavity.invert()
    source = tmp_path / "hollow.stl"
    trimesh.util.concatenate([outer, cavity]).export(source)
    tool = dataset_tool()
    body = tmp_path / "hollow.body.ply"
    raw = tool.check_file(str(source), str(body), mode)
    assert raw["status"] == "ok" and raw["volume"] == pytest.approx(7)
    assert raw["check"]["passed"] and body.exists()
    filled_path = tmp_path / "filled.body.ply"
    filled = tool.check_file(str(source), str(filled_path), mode, fill=True)
    assert filled["status"] == "ok" and filled["volume"] == pytest.approx(8)
    assert filled["fill_cavities"] is True and filled["body"] is None
    assert filled["diag"]["cavities_filled"] == 1
    assert "cavities_filled:1" in filled["repairs"]
    assert filled["diag"]["volume_cavities"] == pytest.approx(1 / 8)
    assert not filled_path.exists()


@pytest.mark.parametrize("mode", ["exact", "fail"])
def test_fill_accepts_a_wall_and_preserves_invalid_input(tmp_path, mode):
    tool = dataset_tool()
    wall = tmp_path / "wall.stl"
    trimesh.Trimesh(*cube_with_wall(False), process=False).export(wall)
    output = str(tmp_path / "unused.body.ply")
    assert tool.check_file(str(wall), output, mode)["status"] == "not_manifold"
    filled = tool.check_file(str(wall), output, mode, fill=True)
    assert filled["status"] == "ok" and filled["volume"] == pytest.approx(1)
    assert filled["diag"]["cells_filled"]["cells"] == 2
    assert "cells_filled:2" in filled["repairs"]
    bad = tmp_path / "missing.stl"
    refused = tool.check_file(str(bad), output, mode, fill=True)
    assert refused["status"] == "load_failed"
    assert refused["blame"] == ("gt" if mode == "exact" else "pred")
    assert refused["fill_cavities"] is True
    open_box = trimesh.creation.box()
    open_box.update_faces(list(range(len(open_box.faces) - 2)))
    source = tmp_path / "open.stl"
    open_box.export(source)
    unclosed = tool.check_file(str(source), output, mode, fill=True)
    assert unclosed["status"] == "open_surface" and "diag" in unclosed


@pytest.mark.parametrize("previous,current", [(None, True), (False, True), (True, False)])
def test_resume_rejects_other_filling_mode(tmp_path, previous, current):
    tool = dataset_tool()
    out = tmp_path / "out"
    out.mkdir()
    row = {"file": "old.stl", "status": "ok", "mode": "exact"}
    if previous is not None:
        row["fill_cavities"] = previous
    (out / "report.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit, match="fill_cavities"):
        tool.run(str(tmp_path), str(out), 1, 0, fill=current)


def test_resume_checks_config_before_rows_exist(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"fill_cavities": True}))
    with pytest.raises(SystemExit, match="fill_cavities"):
        dataset_tool().run(str(tmp_path), str(out), 1, 0)


def test_fill_cli_and_report(tmp_path):
    data, out = tmp_path / "data", tmp_path / "report"
    data.mkdir()
    trimesh.Trimesh(*cube_with_wall(False), process=False).export(data / "wall.stl")
    command = [sys.executable, str(ROOT / "tools/dataset_check.py")]
    subprocess.run(command + ["run", str(data), str(out), "--fill", "--workers", "1"],
                   check=True, capture_output=True, text=True, timeout=120)
    rows = [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert rows[0]["fill_cavities"] is True
    assert json.loads((out / "config.json").read_text())["fill_cavities"] is True
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fill_cavities"] == [True] and summary["bodies_written"] == 0
    assert not (out / "bodies").exists()
    subprocess.run(command + ["report", str(out)], check=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("config", [None, {}, {"fill_cavities": True}])
def test_empty_report_keeps_filling_mode(tmp_path, config):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    dataset_tool().report(str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["files"] == 0
    assert summary["fill_cavities"] == [bool(config and config.get("fill_cavities"))]
