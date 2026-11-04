"""
Two tools/dataset_check.py runs on the same dataset(s), file by file: statuses
and every status change, volume of files ok in both, repairs, the detector's
`crossing_pairs_internal` on files checked in both runs, numeric pocket
diagnostics (missing values stay missing), time (ok/ok files
with and without `self_intersection_resolved`; all files by old -> new
status), and with --bodies the body
files on disk against the `ok` rows of the new run.

usage: compare_checks.py <new out dir> <old out dir> [--changes N] [--bodies]

An out dir is one dataset's (it holds report.jsonl) or a parent of several
(work_dirs/bodies_v061): then every dataset present in both is compared.
--bodies checks the paths written in the new report -- absolute, on the
machine that ran it: run it there (the node) or the files count as missing.
Runs of different modes (dataset_check.py --mode; a row without the field is
"exact") stop the comparison before anything is printed.
Pocket diagnostics are compared on common files: comparable means both
values are present, only-new/only-old means one is present, missing means
neither is present (absent keys and null are missing; zero is a value).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from volumetric_iou.body import SUFFIX  # noqa: E402

RESOLVED = "self_intersection_resolved"


def load(out_dir: Path) -> dict:
    return {r["file"]: r for r in map(json.loads, open(out_dir / "report.jsonl"))}


def kinds(r: dict) -> set:
    return {k.split(":")[0] for k in r.get("repairs", [])}


def secs(r: dict) -> float:
    return r.get("seconds", r.get("seconds_outer", 0.0))


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def seconds_by_status(new: dict, old: dict, files) -> None:
    """Time of the same files in both runs, grouped by (old status, new status)."""
    groups: dict = {}
    for f in files:
        groups.setdefault((old[f]["status"], new[f]["status"]), []).append(f)
    print("  seconds by status old -> new: files, sum h new/old, median new/old, max new/old")
    for (o, n), fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        tn, to = [secs(new[f]) for f in fs], [secs(old[f]) for f in fs]
        print(f"    {o} -> {n}: {len(fs)}, {sum(tn) / 3600:.3f}/{sum(to) / 3600:.3f}, "
              f"{_q(tn, .5):.3g}/{_q(to, .5):.3g}, {max(tn):.3g}/{max(to):.3g}")


def bodies(new: dict) -> None:
    """Body files of the new run against its rows: ok without a file, a file without ok."""
    ok = {f: r for f, r in new.items() if r.get("status") == "ok"}
    listed = {f: r["body"] for f, r in ok.items() if r.get("body")}
    missing = [f for f, p in listed.items() if not Path(p).exists()]
    roots = {p[: -len(str(Path(f).with_suffix(SUFFIX)))] for f, p in listed.items()}
    on_disk = set()
    for root in roots:
        on_disk |= {str(p) for p in Path(root).rglob("*" + SUFFIX)}
    extra = on_disk - set(listed.values())
    print(f"  bodies: ok {len(ok)}, with body {len(listed)} (rejected "
          f"{sum(1 for r in ok.values() if r.get('body_rejected'))}), file missing "
          f"{len(missing)}, files under {sorted(roots)} {len(on_disk)}, not in report {len(extra)}")
    for f in missing[:5]:
        print(f"    missing: {f}")
    for p in sorted(extra)[:5]:
        print(f"    not in report: {p}")


def compare(name: str, new: dict, old: dict, n_changes: int, check_bodies: bool) -> None:
    common = new.keys() & old.keys()
    print(f"== {name}: files new {len(new)}, old {len(old)}, common {len(common)}, "
          f"only new {len(new.keys() - old.keys())}, only old {len(old.keys() - new.keys())}")
    for label, rows in (("new", new), ("old", old)):
        print(f"  {label}: " + ", ".join(f"{s} {n}" for s, n in
                                         Counter(r["status"] for r in rows.values()).most_common()))
    moves = Counter((old[f]["status"], new[f]["status"]) for f in common
                    if old[f]["status"] != new[f]["status"])
    print(f"  status changed: {sum(moves.values())}" + "".join(
        f"\n    {o} -> {n}: {c}" for (o, n), c in moves.most_common()))
    changed = sorted(f for f in common if old[f]["status"] != new[f]["status"])
    for f in changed[:n_changes]:
        print(f"    {f}: {old[f]['status']} -> {new[f]['status']}: "
              f"{new[f].get('detail') or old[f].get('detail', '')}"[:200])

    both = [f for f in common if old[f]["status"] == new[f]["status"] == "ok"]
    rel = [abs(new[f]["volume"] - old[f]["volume"]) / abs(old[f]["volume"]) for f in both]
    print(f"  volume, ok/ok {len(both)}: equal {sum(x == 0 for x in rel)}, "
          f"> 1e-12 {sum(x > 1e-12 for x in rel)}, > 1e-6 {sum(x > 1e-6 for x in rel)}, "
          f"max rel {max(rel, default=0.0):.3g}")
    worst = sorted(zip(rel, both), reverse=True)[:min(n_changes, 5)]
    for x, f in worst:
        if x > 1e-12:
            print(f"    {f}: rel {x:.3g}")

    gained, lost = Counter(), Counter()
    for f in both:
        a, b = kinds(old[f]), kinds(new[f])
        gained.update(b - a)
        lost.update(a - b)
    n_diff = sum(1 for f in both if kinds(old[f]) != kinds(new[f]))
    print(f"  repairs, ok/ok: differ in {n_diff}; gained " + (", ".join(
        f"{k} {c}" for k, c in gained.most_common()) or "-") + "; lost " + (", ".join(
        f"{k} {c}" for k, c in lost.most_common()) or "-"))
    for label, rows in (("new", new), ("old", old)):
        c = Counter(k for r in rows.values() if r["status"] == "ok" for k in kinds(r))
        print(f"    {label}: " + ", ".join(f"{k} {n}" for k, n in c.most_common()))

    def cross(r):
        return (r.get("diag") or {}).get("crossing_pairs_internal")
    checked = [f for f in common if cross(old[f]) is not None and cross(new[f]) is not None]
    diff = [f for f in checked if cross(old[f]) != cross(new[f])]
    print(f"  crossing_pairs_internal, both checked {len(checked)} (of common {len(common)}): "
          f"differ in {len(diff)}; checked only new "
          f"{sum(1 for f in common if cross(new[f]) is not None and cross(old[f]) is None)}, only old "
          f"{sum(1 for f in common if cross(old[f]) is not None and cross(new[f]) is None)}")
    for f in diff[:n_changes]:
        print(f"    {f}: {cross(old[f])} -> {cross(new[f])}")

    for key in ("inverted_volume", "inverted_faces", "inverted_cavity_volume"):
        values = [(f, (old[f].get("diag") or {}).get(key),
                   (new[f].get("diag") or {}).get(key)) for f in sorted(common)]
        comparable = [(f, o, n) for f, o, n in values if o is not None and n is not None]
        changed = [(f, o, n) for f, o, n in comparable if o != n]
        print(f"  {key}, common {len(common)}: comparable {len(comparable)}, "
              f"only-new {sum(o is None and n is not None for _, o, n in values)}, "
              f"only-old {sum(o is not None and n is None for _, o, n in values)}, "
              f"missing {sum(o is None and n is None for _, o, n in values)}; "
              f"changed {len(changed)}")
        for f, o, n in changed[:n_changes]:
            print(f"    {f}: {o} -> {n}")

    print("  seconds, ok/ok (by the new run's repairs): group, files, "
          "sum h new/old, median new/old, max new/old")
    for label, pick in ((RESOLVED, lambda f: RESOLVED in kinds(new[f])),
                        ("other", lambda f: RESOLVED not in kinds(new[f]))):
        fs = [f for f in both if pick(f)]
        tn, to = [secs(new[f]) for f in fs], [secs(old[f]) for f in fs]
        print(f"    {label}: {len(fs)}, {sum(tn) / 3600:.3f}/{sum(to) / 3600:.3f}, "
              f"{_q(tn, .5):.3g}/{_q(to, .5):.3g}, {max(tn, default=0):.3g}/{max(to, default=0):.3g}")
    seconds_by_status(new, old, common)
    if check_bodies:
        bodies(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--changes", type=int, default=20, help="changes to list per set and diagnostic")
    ap.add_argument("--bodies", action="store_true", help="check body files of the new run")
    a = ap.parse_args()
    new, old = Path(a.new), Path(a.old)
    if (new / "report.jsonl").exists():
        pairs = [(new.name, new, old)]
    else:
        sets = sorted({p.parent.name for p in new.glob("*/report.jsonl")}
                      & {p.parent.name for p in old.glob("*/report.jsonl")})
        pairs = [(s, new / s, old / s) for s in sets]
    loaded = [(name, load(n), load(o)) for name, n, o in pairs]
    for name, rn, ro in loaded:     # a row without "mode" (before it was written) is "exact"
        mn, mo = (sorted({r.get("mode", "exact") for r in rows.values()}) for rows in (rn, ro))
        if mn != mo:
            raise SystemExit(f"{name}: mode new {mn}, old {mo} -- runs of different modes "
                             "(dataset_check.py --mode) are not compared")
    for name, rn, ro in loaded:
        compare(name, rn, ro, a.changes, a.bodies)


if __name__ == "__main__":
    main()
