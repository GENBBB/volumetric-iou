# Volumetric IoU: historical validation and performance measurements

This report records agreement with an independent point-sampling arbiter,
differences from a legacy component-sum baseline, the frequency of input cases
in real datasets, and computation cost. The metric's definition, treatment of
each input case, and guarantees are in [DEFINITION.md](DEFINITION.md);
“section N” below refers to that document.

**These are historical measurements from different development versions, not
a benchmark of 0.24.0.** The version is given with each measurement. Later
versions changed the input contract as well as implementation and performance;
these historical acceptance rates, statuses, and timings must not be treated as
current results. See [CHANGELOG.md](../CHANGELOG.md) for behavior changes and
DEFINITION for the current contract.

For the specific checks after `0.16.0` (exact zero proof) and `0.17.0` (an open
copy of a shell written twice causes failure; a flat sheet is discarded),
statuses and methods on 1502 pairs and 967 GT inputs that reached CGAL were
compared with the preceding results: no changes, with IoU differences within
1e-14. This observation applies to those checks, not to every later version.

## 1. Data

- **1502 evaluation pairs** (MCB, Fusion360, DeepCAD): 900 random, 630 with
  multi-body GT, 123 previously checked by an arbiter, and 25 disputed cases.
  These selection categories overlap; their counts must not be added.
  The pairs were selected without open GT surfaces. Evaluation frames: GT
  normalized by its own bounding box into `[0, 1]³`, pred transformed by
  `/200 + 0.5`, corresponding to `pred_frame="div200"`.
- **Complete mesh datasets** (`tools/dataset_check.py`, each file normalized
  by its own bounding box): `deepcad_test_mesh` (8046 files),
  `fusion360_test_mesh` (1725), `cadenabench` (3396), `mcb_test_mesh` (5000).
- Server: AMD EPYC 9474F, quota of 48 CPUs; arbiter: 30 thousand points.

## 2. Agreement with the arbiter

The arbiter estimates IoU using points with signed w ≥ 0.5 (section 13),
independently of the Boolean engines. A discrepancy is significant at about
3 standard errors (|z| > 3). Code versions: `v0.1.0` for clean pairs and the
CGAL path through `v0.3.0`.

| | returned a number | \|z\| > 3 | \|z\| > 10 |
|---|---|---|---|
| legacy component-sum baseline | 344 | 63 | 39 |
| metric, clean pairs (manifold) | 147 | 1 (z = 3.8: 0.9994 versus 0.9998 ± 0.0001) | 0 |
| metric, 178 pairs through CGAL | 178 | 1 (z = 3.1; 0.5 expected for an exact answer) | 0 |

On the 178 CGAL pairs, signed z had mean −0.061 ± 0.072 and sd 0.954; on clean
pairs, −0.091 and sd 1.03: sampling noise, without bias. The baseline returned
results on 175 of those 178 pairs: |z| > 3 for 18, and > 10 for 11.

An arbiter using |w| ≥ 0.5 instead of w ≥ 0.5 treats an inverted pocket as a
body: on `mcb_test/00050318`, it gives 0.779 versus the exact 0.846.

## 3. Differences from the legacy component-sum baseline

| legacy component-sum baseline | volumetric-iou |
|---|---|
| sums pairwise component intersections: overlapping bodies on one side are counted twice | a side is the union of its bodies (w ≥ 1) |
| clips (1, 1.05] to 1.0, returns `None` above that | corrects only a small excess (I > min(V) by at most 1e-6 relative, recorded as `volume_inter_raw`); above that, recomputes in CGAL or fails |
| silently drops open parts | patches a hole with proven volume error ≤ 1e-3 of the parts it closes, with a warning; otherwise, `open_surface` (DEFINITION 4.3) |
| `fix_normals` on the intersection inverts a cavity | orients shells before the Boolean operation |
| float32 input to manifold3d | `Mesh64`, float64 |
| a shell passing through itself is weighted by its winding number | exact body in CGAL for GT; failure for pred |
| `None` without a reason, output to stdout | status, cause, blame; no stdout output |
| no time limit | `timeout` produces a status |

On 1282 pairs for which both returned a result, |Δ| > 0.01 in 23 cases, of
which 22 had multi-body GT (double-counted overlap in the baseline). Extreme
cases: `mcb/00044710`: baseline 0.949, metric 0.884, arbiter 0.892 ± 0.005;
`mcb/00031337`: baseline 0.857, metric 0.527. Means over the random subset on
common pairs nearly agree (mcb 0.9235 versus 0.9206, fusion360 0.9784 / 0.9781,
deepcad 0.9816 / 0.9816): the discrepancy is concentrated in multi-body parts.
Union of bodies corrected 26 of 26 baseline overestimates.

## 4. Frequency of input cases

**Self-intersections inside a body** (exact detector, `v0.3.0`): in 125 of
1502 pairs (blame: pred 101, GT 24); 4.4% of the random subset, 8.0% of random
MCB. The previous tolerance-based detector counted more (70 pairs versus 40
in the random subset) because of false alarms. With the default
`("exact", "fail")` (`v0.6.0`), 104 predictions were rejected; 2 pairs had
contact only, without a body passing through itself
(`fusion360/56719_6e59eeff_0000`, `deepcad/00037265`).

**Failures with `"exact"` on both sides** (`v0.3.0`, before pocket rejection):
21 of 1502 pairs (1.4%), all attributed to pred: `open_surface` 20 (deepcad
16, mcb 2, fusion360 2) and `not_manifold` 1 (deepcad): meshes without volume.
No GT or engine failures, and no timeouts.

**Dataset file acceptance rates** (`v0.3.0`): `deepcad_test_mesh` 99.6%,
`fusion360_test_mesh` 97.3%, `cadenabench` 85.2%, `mcb_test_mesh` 73.8% (one
in four files is open). Edges with 3+ faces resolved into shells occurred in
451 of 16 400 accepted files (`v0.4.1`; the count also includes 126 accepted
files from a fifth, small dataset, `fusion_fails`).

**Inverted pockets** (`v0.8.0`): of 967 dataset GT files (mcb, cadenabench,
deepcad, fusion360) that reached CGAL, 239 had a pocket; in 220, it was less
than 1e-4 of body volume (slivers and thin plates along an intersection line),
and in 3, more than 1%.

**Bodies touching along a face, body from raw surface** (`v0.11.0`): of 670
dataset sides with `not_manifold`/`open_surface` failures, 233 had a raw
surface defining a body: 197 became `ok` (IoU against the arbiter 1.0), and
36 became `inverted_volume` (pockets from 0 to 4e-8 of volume; below 1e-12
in 31).

**Shells written twice** (`v0.12.0`): on 34 dataset sides where unwrapping
changed the status, 8 failures became `ok` (IoU against the arbiter 1.0;
0.99984 for one, with se 1.6e-4); one `ok` became `open_surface` (the main
body was written twice, with one copy open at a pin junction). The `0.17.0`
rule for an open copy changed no statuses on the 967 GT inputs (above);
these 34 sides were not remeasured separately.

**Cavities** (`v0.7.0`): 26 of 147 dataset bodies with cavities had a cavity
larger than 20% of volume (up to 84%); predictions on such evaluation pairs
were almost all solid. Filled GT with unfilled pred penalized a prediction
that reproduced a cavity in 1 of 11 pairs with a cavity > 20%. Raw-surface
`cavities` does not detect cavities formed by overlapping bodies or after
CGAL: in mcb, this applied to 70 of 121 GT inputs with a filled cavity.

**Patches:** maximum 0.02% of area on 1284 pairs (measured before `0.1.0`,
then using a 5% area threshold; the patch rule now uses a proven error bound,
section 4.3).

## 5. Performance

### Baseline measurement (`v0.3.0`)

All 1502 pairs, `"exact"` for both sides, 24 processes under a 48-CPU quota,
`IGL_NUM_THREADS=2`, 1800-second limit per pair. Time is the wall time of one
`volume_iou` call in a worker process, excluding STL reading (median reading
time 0.01 s).

**Result:** median 0.04 s per pair, mean 0.76 s, p90 0.35 s, p99 25 s,
maximum 229 s. Sum: 0.32 h of per-call wall time (0.46 h CPU); with 24
processes, the dataset completed in 240 s, with elapsed time determined by
one pair (`mcb/00031558`). The legacy baseline averaged about 0.17 s per
pair: the metric was no more expensive by median and about 4.5 times as
expensive by mean, with almost all the difference due to the 8% of pairs
using CGAL. Summed wall time by version: `v0.1.0`, 1.49 h (one pair timed
out); `v0.2.0`, 0.38 h; `v0.3.0`, 0.32 h.

**By pair computation path:**

| path | pairs | median, s | p90, s | p99, s | max, s | share of summed time |
|---|---|---|---|---|---|---|
| clean pair, one body per side (manifold) | 1005 (67%) | 0.03 | 0.10 | 0.37 | 2.7 | 5% |
| multi-body side, union (manifold) | 352 (23%) | 0.11 | 0.47 | 3.3 | 6.4 | 8% |
| face pairs inside a body → CGAL | 124 (8%) | 0.62 | 27 | 85 | 229 | 87% |
| `open_surface` / `not_manifold` failure | 20 / 1 | 0.02 / 0.08 | — | — | 0.2 | 0% |

**By size** (GT + pred faces), median / p90 / maximum, s:

| faces | manifold | CGAL |
|---|---|---|
| < 10 thousand | 0.03 / 0.09 / 0.23 (n 1137) | 0.29 / 0.59 / 1.7 (n 52) |
| 10–30 thousand | 0.18 / 0.29 / 0.41 (n 148) | 0.52 / 1.3 / 7.3 (n 24) |
| 30–100 thousand | 0.47 / 0.78 / 1.6 (n 55) | 1.7 / 2.9 / 3.8 (n 17) |
| 100–300 thousand | 1.5 / 3.8 / 6.4 (n 15) | 6.9 / 26 / 27 (n 16) |
| > 300 thousand | 5.8 / — / 5.8 (n 2) | 34 / 75 / 229 (n 15) |

At the same size, CGAL was 3–10 times as expensive as manifold.

**By computation stage**, share of total time: body construction in CGAL
69% (128 sides; CGAL core 62%, point checking and assembly 7%); intersection
in manifold 6%; detector 6% (2984 sides, maximum 3.3 s); shell orientation 4%;
cleanup 2%; manifold body assembly 1%. Intersection in CGAL was never needed
(in `v0.3.0`; since `0.16.0`, an unproven empty manifold result is recomputed
in CGAL — none occurred on these pairs).

**Heaviest pairs:** `mcb/00031558`, 229 s (pred 268 thousand faces,
425 thousand intersection pairs, 214 s in the CGAL core); `mcb/00027755`,
100 s (GT with 30 overlapping bodies, union in single-threaded manifold);
another eight took 33–37 s each: MCB predictions with 280–460 thousand faces
and tens of thousands of self-intersections.

**Threads:** CPU time (0.46 h) exceeds wall time (0.32 h) because CGAL is
internally multithreaded. Without limiting `IGL_NUM_THREADS`, densely packed
processes slow CGAL pairs: with 48 processes at one thread each,
`mcb/00031558` took 375 s; with 6 × 8, it took 96 s.

### Subsequent changes

- **Rejecting self-intersecting pred** (`v0.6.0`, default
  `("exact", "fail")`): summed wall time 0.270 → 0.100 h; with GT bodies
  loaded from files, 0.043 h.
- **GT body loaded from a file** (`v0.4.0`): summed pair wall time −21%
  (1227 → 964 s, server overloaded — compare the ratio);
  `mcb/00027755`, 110 → 6 s. Building bodies took 359 s for 1502 GT inputs,
  285 MB; statuses, methods, and warnings were the same, with bitwise-equal
  IoU on 1489 pairs and differences up to 1.8e-15 on the rest.
- **CGAL on GMP** (`v0.5.0`–`0.7`, custom libigl build; since `0.8.0`, module
  built with `VIOU_GMP=ON` or a release wheel): dataset 232 → 160 s,
  `mcb/00031558`, 217 → 148 s; same statuses, IoU differences up to 2.0e-15.
- **Call with prepared GT** (`v0.15.0`, 1491 pairs from the earlier sample,
  pred side): sum 66.2 s, median 16.7 ms, p99 406 ms.
- **Exact mode `exact_booleans=True`** (`0.20.0`, 1501 sample pairs, default
  side modes `("exact", "fail")`, 24 processes with 2 CGAL threads each,
  compared with the same code without the flag): among pairs returning a
  number, calls with raw GT had median 0.714 s versus 0.028 s (×25), sum
  2503 s versus 135 s (×18.5), maximum 64 s versus 5.5 s; calls with GT
  bodies loaded from files had median 0.622 s versus 0.019 s (×33), sum
  2025 s versus 71 s (×28); building GT bodies took 490 s versus 127 s in
  total. iou/iog/iop differed from calls without the flag by up to 1.1e-13.
  One pair changed status (`mcb/00045717`, ok → `boolean_failed`): GT with
  151 bodies, all united in one CGAL union in exact mode, where CGAL loses a
  small body (the point check detects this; no number is returned). Exact
  mode did not finish `mcb/00027755` (GT 328 thousand faces, 30 overlapping
  copies) within 20 minutes — 69 s without the flag.

### Practical implications

- For a benchmark with fixed GT, build GT bodies once (`to_body` or
  `tools/dataset_check.py`), then use `prepare_side` on the body in the
  process; untimed calls can reuse construction. With the default server
  timeout isolation, a prepared side is rebuilt in each child; only loading
  and hashing can be saved. See [API.md](API.md).
- For evaluation inside a training loop, historical measurements show
  medians of tens of milliseconds but a heavy tail. Use `timeout` and decide
  how a timeout affects the reward. Measure the current version on the target
  workload before relying on those timings.
- The most expensive case is self-intersecting GT with hundreds of thousands
  of faces (CGAL); its body can be built once and stored in a file.
  Self-intersecting pred under the default `"fail"` also uses CGAL: it fails
  only where a body passes through itself; contact inside a body returns a
  number. The same applies to raw input (DEFINITION 4.9).
