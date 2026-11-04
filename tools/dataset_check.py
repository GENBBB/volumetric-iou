"""
Check a dataset of meshes against the metric, file by file, and write the
metric's solid of every file it accepts as a body file.

"Repair" is only what `volume_iou` does to a side anyway (docs/DEFINITION.md):
merged vertices, degenerate and duplicate faces dropped, holes patched where
the patch's proven volume error is within 1e-3, zero-volume shells dropped, inside-out shells flipped,
overlapping bodies united, self-intersections resolved by CGAL into the
w >= 1 solid (in --mode fail a body passing through itself is refused).  A
side cleaning refuses as open or not manifold may still be built from its raw
surface (DEFINITION 4.9; `diag.raw_solid`) -- not counted in `repairs`.
A file the metric refuses (open surface, edges of 3+ faces, flat, ...) is not
repaired -- it is listed with its status, cause and detail.

    python tools/dataset_check.py run DATASET_DIR OUT_DIR [--workers 24] [--limit N]
                                      [--files LIST] [--mode exact|fail] [--fill]
    python tools/dataset_check.py report OUT_DIR

--mode: the side a file is built as -- "exact" (default) a GT, as
`volume_iou` builds it (label "mesh"); "fail" a prediction
(on_self_intersection="fail", label "pred", placed as
`normalize_pair(pred_frame="gt")` places a prediction equal to its GT -- the
file's own extent, as for a GT; the body's frame is recorded as "bbox"), so
the prediction's path of the metric runs on dataset files.  Every row,
config.json and summary.json carry the mode (a row without it is "exact");
the body check reads the body back in the same mode.  A run resumed into
OUT_DIR with rows of the other mode stops.

--fill: check the side with fill_cavities=True through prepare_side, including
walls accepted only with filling. Write its filled volume and diagnostics to
the report, without body files: to_body stores the solid before filling.
The side labels are gt for exact and pred for fail in this mode.
Rows and run options record fill_cavities; reports of different filling modes
cannot be resumed into the same OUT_DIR. Older rows mean fill_cavities=False.

OUT_DIR/report.jsonl   one line per file: mode, status, cause, blame, detail,
                       repairs, diag (selected diagnostics; of a refused
                       file too -- e.g. raw_winding_max, self_overlap_parts,
                       self_overlap_bodies/_volume of a "fail" refusal), faces, engine,
                       volume (file units; diagnostic volumes remain in the
                       metric's normalised frame), seconds, seconds_outer, and for an
                       accepted file its body path and the check below
OUT_DIR/refusals.csv   files the metric refuses, one per line
OUT_DIR/summary.json, config.json  counts by status; the run's options
OUT_DIR/bodies/<rel>.body.ply  the solid (`volumetric_iou.to_body`, frame
                       "bbox" -- the GT convention of `normalize_pair`), for
                       every accepted file without --fill, repaired or not: the metric takes
                       it with no repair and no detector run

Check of a body file: read back, it must equal the written body bit for bit
and pass the metric's cheap checks with the recorded volume; otherwise it is
deleted and the row says why (`body_rejected`).

Pool scheme as tools/timing.py: forkserver workers, every file in its own
fork with a time limit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

T_LIMIT = 1800          # s per file
TOL = 1e-9              # overlap removed by the union that counts as a repair


def _init_worker():
    import numpy, trimesh, manifold3d  # noqa: F401  warm imports once
    import volumetric_iou  # noqa: F401


def repairs(info: dict) -> list[str]:
    """What the metric changed in a side, beyond merging split vertices."""
    out = []
    if info.get("degenerate_removed"):
        out.append(f"degenerate_faces:{info['degenerate_removed']}")
    if info.get("duplicate_removed"):
        out.append(f"duplicate_faces:{info['duplicate_removed']}")
    if info.get("loop_faces_added"):
        out.append(f"cracks_patched:{info['loop_faces_added']}"
                   f"@{info.get('patched_area_frac', 0.0):.3g}")
    if info.get("shells_dropped"):
        out.append(f"zero_volume_shells:{info['shells_dropped']}")
    if info.get("flipped_inverted_mesh"):
        out.append("inside_out_mesh")
    if info.get("flipped_shells"):
        out.append(f"inside_out_bodies:{info['flipped_shells']}")
    if info.get("orientation") == "parity":
        out.append("orientation_by_parity")
    if info.get("union") and info.get("volume_sum_of_solids") and info.get("volume"):
        # bodies with touching or overlapping extents go through the union;
        # it changed the solid only if their volumes overlapped
        over = (info["volume_sum_of_solids"] - info["volume"]) / info["volume"]
        if over > TOL:
            out.append(f"overlap_removed:{over:.3g}")
    if info.get("resolved"):
        r = info["resolved"]
        out.append(f"self_intersection_resolved:{r.get('crossing_pairs')}")
    if info.get("cavities_filled"):
        out.append(f"cavities_filled:{info['cavities_filled']}")
    if info.get("cells_filled"):
        out.append(f"cells_filled:{info['cells_filled']['cells']}")
    return out


_INFO_KEYS = ("faces_raw", "solids", "cavities", "orientation", "union",
              "self_intersection_checked", "crossing_pairs_internal", "unchecked_bodies",
              "detector_error", "loop_faces_added", "patched_area_frac",
              "boundary_edges_before_fill", "open_edges", "nonmanifold_edges",
              "ambiguous_containment", "resolved", "inverted_volume", "inverted_faces",
              "inverted_cavity_volume", "raw_solid",
              # only on a refusal (in "fail" mostly): why it was refused
              "raw_winding_max", "self_overlap_parts", "self_overlap_bodies",
              "self_overlap_volume", "crossing_pairs", "raw_holes", "volume",
              "cells_filled", "cavities_filled", "volume_cavities", "cavities_solid")


MODES = {"exact": "mesh", "fail": "pred"}      # --mode -> label of the side


def _verify(path: Path, body, mode: str = "exact") -> dict:
    """The body file read back: bit-equal to the written body, and taken by the metric."""
    import numpy as np
    from volumetric_iou import body as B
    try:
        back = B.load_body(path)
    except (OSError, ValueError) as exc:
        return {"passed": False, "status": "load_failed", "detail": str(exc)}
    same = bool(np.array_equal(back.vertices, body.vertices)
                and np.array_equal(back.faces, body.faces) and back.meta == body.meta)
    prep, err = B.prepared(back, "body", mode)
    if err is not None:
        return {"passed": False, "status": err.status.value, "detail": err.detail}
    return {"passed": same and prep.volume == body.volume, "status": "ok", "bitwise": same}


def _check_filled(src: str, mode: str) -> dict:
    """The metric's filled side, in its own bbox frame; no Body is written."""
    from volumetric_iou import body as B, mesh_iou as M
    label = "gt" if mode == "exact" else "pred"
    raw, err = M._load_and_check(src, label)
    if err is None:
        centre, scale = B.frame_for("bbox", raw)
        side = M.prepare_side(M._moved(raw, centre, scale), label,
                              on_self_intersection=mode, fill_cavities=True)
        err = side.error
    if err is not None:
        return dict(status=err.status.value, cause=err.cause.value, blame=err.blame,
                    detail=err.detail,
                    diag=B._plain({k: err.diagnostics[k] for k in _INFO_KEYS if k in err.diagnostics}))
    prep = side._solid()
    info = B._plain(prep.info)
    return dict(status="ok", cause="ok", faces=len(raw.faces),
                repairs=repairs({**info, "volume": prep.volume}),
                volume=prep.volume / scale ** 3,
                engine="manifold" if prep.solid is not None else "cgal",
                diag={k: info[k] for k in _INFO_KEYS if k in info}, body=None)


def check_file(src: str, body_path: str, mode: str = "exact", fill: bool = False) -> dict:
    """Status, repairs and diagnostics of one file; writes its body file."""
    from volumetric_iou import body as B
    t0 = time.perf_counter()
    out: dict = {"mode": mode, "fill_cavities": fill}
    if fill:
        out.update(_check_filled(src, mode))
        out["seconds"] = time.perf_counter() - t0
        return out
    b = B.to_body(src, frame="bbox", label=MODES[mode], on_self_intersection=mode)
    if not isinstance(b, B.Body):
        out.update(status=b.status.value, cause=b.cause.value, blame=b.blame, detail=b.detail,
                   diag={k: b.diagnostics[k] for k in _INFO_KEYS if k in b.diagnostics})
    else:
        info = b.meta["info"]
        out.update(status="ok", cause="ok", faces=b.meta["source"]["faces_raw"],
                   repairs=repairs({**info, "volume": b.volume}),
                   volume=b.volume / b.frame["scale"] ** 3, engine=b.meta["engine"],
                   diag={k: info[k] for k in _INFO_KEYS if k in info})
        path = B.save_body(b, body_path)
        out["body"] = str(path)
        out["check"] = _verify(path, b, mode)
        if not out["check"]["passed"]:
            path.unlink()
            out["body"], out["body_rejected"] = None, (
                f"{out['check']['status']}: {out['check'].get('detail', 'not bit-equal')}")
    out["seconds"] = time.perf_counter() - t0
    return out


def one_file(src: str, rel: str, out_dir: str, mode: str = "exact", fill: bool = False) -> dict:
    from volumetric_iou.body import SUFFIX
    from volumetric_iou.isolate import run_isolated
    body = str(Path(out_dir, "bodies", rel).with_suffix(SUFFIX).resolve())   # rows outlive the cwd
    t = time.perf_counter()
    r = run_isolated(check_file, (src, body, mode, fill), {}, T_LIMIT)
    row = {"file": rel, "mode": mode, "fill_cavities": fill}
    if r["status"] == "ok":
        row.update(r["value"])
    else:           # the fork hung, died or raised: the metric's own statuses
        status = {"timeout": "timeout", "died": "crashed"}.get(r["status"], "crashed")
        row.update(status=status, cause="engine", blame="engine", detail=str(r)[:300])
        Path(body).unlink(missing_ok=True)      # the fork may have saved it before the limit
    row["seconds_outer"] = time.perf_counter() - t
    return row


def _rows(out_dir: Path) -> list[dict]:
    p = out_dir / "report.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def report(out_dir: str) -> None:
    from collections import Counter
    od = Path(out_dir)
    rows = _rows(od)
    refused = [r for r in rows if r.get("status") != "ok"]
    with open(od / "refusals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "cause", "blame", "detail"])
        for r in sorted(refused, key=lambda r: (r["status"], r["file"])):
            w.writerow([r["file"], r["status"], r.get("cause", ""), r.get("blame", ""),
                        r.get("detail", "")])
    ok = [r for r in rows if r.get("status") == "ok"]
    rejected = [r for r in rows if r.get("body_rejected")]
    kinds = Counter(k.split(":")[0] for r in rows for k in r.get("repairs", []))
    secs = sorted(r.get("seconds", r.get("seconds_outer", 0.0)) for r in rows)
    filling = {r.get("fill_cavities", False) for r in rows}
    if not rows:
        config_path = od / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        filling.add(config.get("fill_cavities", False))
    summary = {
        "files": len(rows),
        "mode": sorted({r.get("mode", "exact") for r in rows}),
        "fill_cavities": sorted(filling),
        "status": dict(Counter(r.get("status") for r in rows)),
        "ok_untouched": sum(1 for r in ok if not r.get("repairs")),
        "needed_repair": sum(1 for r in ok if r.get("repairs")),
        "repairs": dict(kinds),
        "bodies_written": sum(1 for r in rows if r.get("body")),
        "bodies_engine": dict(Counter(r.get("engine") for r in ok if r.get("body"))),
        "body_rejected": len(rejected),
        "body_rejected_why": dict(Counter(r["body_rejected"].split(":")[0] for r in rejected)),
        "seconds": {"median": secs[len(secs) // 2] if secs else None,
                    "max": secs[-1] if secs else None, "sum_h": sum(secs) / 3600},
    }
    (od / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def run(dataset: str, out_dir: str, workers: int, limit: int | None,
        files_list: str | None = None, mode: str = "exact", fill: bool = False) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pool_threads import igl_threads
    igl_n = igl_threads(workers)        # before `import igl` and the pool
    root, od = Path(dataset), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    rows = _rows(od)
    other = {r.get("mode", "exact") for r in rows} - {mode}
    if other:               # a resumed run must not mix the two sides in one report
        raise SystemExit(f"{od}/report.jsonl holds rows of mode {sorted(other)}, not {mode!r}")
    other_fill = {r.get("fill_cavities", False) for r in rows} - {fill}
    config_path = od / "config.json"
    if config_path.exists():
        previous_fill = json.loads(config_path.read_text()).get("fill_cavities", False)
        other_fill |= {previous_fill} - {fill}
    if other_fill:
        raise SystemExit(f"{od} holds a run with fill_cavities={sorted(other_fill)}, not {fill}")
    done = {r["file"] for r in rows}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".stl", ".obj", ".ply", ".off"))
    if files_list:          # paths relative to DATASET_DIR, one per line
        keep = {l.strip() for l in open(files_list) if l.strip()}
        files = [p for p in files if str(p.relative_to(root)) in keep]
    files = files[:limit]
    todo = [p for p in files if str(p.relative_to(root)) not in done]
    todo.sort(key=lambda p: p.stat().st_size, reverse=True)      # heavy first: they set the wall
    (od / "config.json").write_text(json.dumps({
        "dataset": str(root), "files": len(files), "workers": workers, "limit": limit,
        "files_list": files_list, "mode": mode, "fill_cavities": fill,
        "IGL_NUM_THREADS": igl_n, "T_LIMIT": T_LIMIT, "TOL": TOL,
        "boolean_so": os.environ.get("VOLUMETRIC_IOU_CGAL_BOOLEAN"),     # mesh_iou.BOOLEAN_ENV
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"{len(files)} files, {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    with open(od / "report.jsonl", "a") as f, ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_init_worker) as ex:
        futs = {ex.submit(one_file, str(p), str(p.relative_to(root)), out_dir, mode, fill): p
                for p in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            p = futs[fu]
            try:
                r = fu.result()
            except Exception as e:      # noqa: BLE001 -- a lost file is a result too
                r = {"file": str(p.relative_to(root)), "mode": mode,
                     "fill_cavities": fill, "status": "crashed",
                     "cause": "engine",
                     "blame": "engine", "detail": f"worker: {e!r}"}
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            if n % 200 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(todo)} files in {time.time() - t0:.0f}s", flush=True)
    report(out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("out_dir")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--limit", type=int)
    r.add_argument("--files", help="only these files: paths relative to DATASET_DIR, one per line")
    r.add_argument("--mode", choices=tuple(MODES), default="exact",
                   help="build each file as a GT (exact) or as a prediction (fail)")
    r.add_argument("--fill", action="store_true",
                   help="check with cavities filled; report volume/diagnostics without body files")
    s = sub.add_parser("report")
    s.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.dataset, a.out_dir, a.workers, a.limit, a.files, a.mode, a.fill)
    else:
        report(a.out_dir)
