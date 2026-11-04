"""Numeric pocket diagnostics survive reports, including older report formats."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("inverted_volume", "inverted_faces", "inverted_cavity_volume")


def tool(name):
    # Load the tool delivered alongside this test, never one from the cwd/PYTHONPATH.
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("diag", [
    dict(zip(KEYS, (0.123456789, 7, 0.0000123456789))),
    dict(zip(KEYS, (0.0, 0, 0.0))),
    {"inverted_volume": 0.25, "inverted_faces": 4},
    {},
])
def test_check_file_preserves_diagnostics(monkeypatch, tmp_path, accepted, diag):
    from volumetric_iou import body as B

    dataset = tool("dataset_check")
    if accepted:
        result = B.Body(vertices=None, faces=None, meta={
            "info": diag, "source": {"faces_raw": 12}, "engine": "synthetic",
            "volume": 1.0, "frame": {"scale": 1.0},
        })
        monkeypatch.setattr(B, "save_body", lambda b, p: Path(p))
        monkeypatch.setattr(dataset, "_verify", lambda *args: {"passed": True})
    else:
        result = SimpleNamespace(
            status=SimpleNamespace(value="inverted_volume"),
            cause=SimpleNamespace(value="geometry"), blame="mesh",
            detail="rounded diagnostic", diagnostics=diag,
        )
    monkeypatch.setattr(B, "to_body", lambda *args, **kwargs: result)
    row = dataset.check_file("synthetic.stl", str(tmp_path / "synthetic.body.ply"))
    # The JSON representation is what downstream comparisons consume.
    assert json.loads(json.dumps(row))["diag"] == diag


@pytest.mark.parametrize("key", KEYS)
def test_compare_presence_and_exact_changes(key, capsys):
    compare = tool("compare_checks")

    def row(diag):
        return {"status": "inverted_volume", "detail": "volume 0.123", "diag": diag}

    old = {
        "changed": row({key: 0.12341}), "equal-zero": row({key: 0}),
        "only-new": row({}), "only-old": row({key: 0}),
        "missing": {"status": "inverted_volume", "detail": "volume 99"},
        "null": row({key: None}), "old-file": row({key: 100}),
    }
    new = {
        "changed": row({key: 0.12342}), "equal-zero": row({key: 0}),
        "only-new": row({key: 0}), "only-old": row(None),
        "missing": row({}), "null": row({key: None}),
        "new-file": row({key: 200}),
    }
    compare.compare("synthetic", new, old, 20, False)
    output = capsys.readouterr().out
    assert (f"{key}, common 6: comparable 2, only-new 1, only-old 1, "
            "missing 2; changed 1") in output
    assert "changed: 0.12341 -> 0.12342" in output
    assert "status changed: 0" in output

    compare.compare("synthetic", new, old, 0, False)
    output = capsys.readouterr().out
    assert "changed 1" in output
    assert "changed: 0.12341 -> 0.12342" not in output


def test_report_compare_smoke(tmp_path, monkeypatch):
    """No geometry: refusal -> check_file -> JSONL -> report CLI -> compare CLI."""
    from volumetric_iou import body as B

    dataset = tool("dataset_check")
    for label, values in (("old", (0.12341, 7, 0.012341)),
                          ("new", (0.12342, 8, 0.012342))):
        refusal = SimpleNamespace(
            status=SimpleNamespace(value="inverted_volume"),
            cause=SimpleNamespace(value="geometry"), blame="mesh",
            detail="volume 0.123", diagnostics=dict(zip(KEYS, values)),
        )
        monkeypatch.setattr(B, "to_body", lambda *args, **kwargs: refusal)
        row = dataset.check_file("pocket.stl", str(tmp_path / "unused.body.ply"))
        row["file"] = "pocket.stl"
        out = tmp_path / label
        out.mkdir()
        legacy = {"file": "legacy.stl", "status": "inverted_volume", "detail": "volume 9"}
        (out / "report.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(legacy) + "\n")
        reported = subprocess.run(
            [sys.executable, str(ROOT / "tools/dataset_check.py"), "report", str(out)],
            cwd=tmp_path, text=True, capture_output=True, timeout=30,
        )
        assert reported.returncode == 0, reported.stderr
        assert json.loads((out / "summary.json").read_text())["files"] == 2
    compared = subprocess.run(
        [sys.executable, str(ROOT / "tools/compare_checks.py"),
         str(tmp_path / "new"), str(tmp_path / "old")],
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert compared.returncode == 0, compared.stderr
    for key in KEYS:
        assert (f"{key}, common 2: comparable 1, only-new 0, only-old 0, "
                "missing 1; changed 1") in compared.stdout
    for change in ("0.12341 -> 0.12342", "7 -> 8", "0.012341 -> 0.012342"):
        assert change in compared.stdout
    assert "status changed: 0" in compared.stdout
    print(compared.stdout)
