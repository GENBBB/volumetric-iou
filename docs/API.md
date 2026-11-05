# Python API

See the [definition](DEFINITION.md) for the complete input policy and
[README](../README.md) for a runnable example. The package exports `Body`,
`Cause`, `IouResult`, `PreparedSide`, `Status`, `diagnose_mesh`, `load_body`,
`normalize_pair`, `prepare_side`, `save_body`, `to_body` and `volume_iou`.

## `volume_iou`

```python
volume_iou(
    gt, pred, *, check_self_intersection=True,
    on_self_intersection=("exact", "fail"), fill_cavities=False,
    exact_booleans=False, timeout=None, isolation="server",
)
```

Inputs may be mesh paths, `trimesh.Trimesh`, `trimesh.Scene`, `(vertices,
faces)` arrays, `Body`, `*.body.ply` paths, or `PreparedSide`. Their order
determines IoG/IoP, failure attribution, default policies and the GT cube
precondition. GT must fit `[-0.5, 0.5]³` within `1e-6`; predictions use the
same coordinates. A prediction extending more than `0.1` outside the cube
receives a warning.

| Option | Meaning |
|---|---|
| `on_self_intersection` | One mode or a `(gt, pred)` pair. `"exact"` resolves body self-overlap in CGAL; `"fail"` refuses an individual body with material winding ≥ 2 at any positive volume. Self-contact alone is accepted; inverted pockets are a separate defect. |
| `fill_cavities` | One bool or a pair. Fill enclosed cavities per interpretation, then compare the final materials. Does not make an exterior inverted pocket or self-overlap in `"fail"` valid. |
| `exact_booleans` | Use CGAL for all body, union, filling and intersection Boolean operations. Reusable bodies must have been built with this flag too. Otherwise they return `load_failed`. |
| `check_self_intersection` | Keep `True` for evaluation. `False` disables the detector and the corresponding self-overlap checks; guarantees relying on them no longer apply. Intended for diagnostics. |
| `timeout` | Seconds for isolated computation. `None` runs in the caller; native hangs cannot then be interrupted by the metric. |
| `isolation` | `"server"` by default. `"fork"` is for callers that have not initialized native geometry thread pools. Applies to timed calls. |

Ambiguous raw assemblies require agreement of every admissible interpretation
on material and validity. `"exact"` does not bypass that requirement.
See [raw input](DEFINITION.md#section-4-9).

## `normalize_pair(gt, pred, *, pred_frame)`

Normalize GT by its bounding box: center at the origin and longest extent
equal to one. `pred_frame` is mandatory:

- `"gt"`: apply the GT transform to the prediction; preserve relative errors.
- `"bbox"`: normalize the prediction independently; remove scale and position errors.
- `"div200"`: divide prediction coordinates by 200 about the origin.

A `Body` GT retains its stored frame; a GT prepared from a `Body` does too.
The prediction is transformed using that frame. A GT prepared from a raw
mesh has no original-frame transform, so passing it to `normalize_pair`
raises `ValueError`. A prepared prediction is not accepted by this helper.
Normalize a raw pair before preparation, or prepare a saved GT body.

## Body files

```python
to_body(mesh, *, frame, gt=None, label="mesh",
        on_self_intersection="exact", exact_booleans=False)
save_body(body, path)
load_body(path)
```

`to_body` builds a side once and returns a `Body`, or an `IouResult` failure.
The `frame` options correspond to the normalization conventions above;
`frame="gt"` needs a reference raw GT mesh or a GT `Body` in its `"bbox"`
frame. The `gt` argument is required only for this frame. Body files record vertices, faces,
metadata and the coordinate transform. `save_body` writes `*.body.ply`.

```python
from volumetric_iou import Body, load_body, save_body, to_body, volume_iou

body = to_body(gt_raw, frame="bbox", label="gt")
if not isinstance(body, Body):
    raise RuntimeError(f"{body.status.value}: {body.detail}")
save_body(body, "part.body.ply")
```

A saved body reuses the rules of the version that created it. Loading it
does **not** revalidate the original STL under newer rules. To apply current
rules, rebuild from the original mesh; resaving an old `Body` is insufficient.
New raw bodies retain the source interpretation classification for `"fail"`;
older bodies without that metadata retain their older limited checks.

`to_body` builds without cavity filling. If an input is unambiguous only
after filling, use its source directly in `volume_iou(..., fill_cavities=True)`
or `prepare_side(..., fill_cavities=True)` instead.

## `prepare_side`

```python
prepare_side(obj, side, *, check_self_intersection=True,
             on_self_intersection=None, fill_cavities=False,
             exact_booleans=False)
```

`side` is `"gt"` or `"pred"`. The default `None` selects `"exact"` for GT
and `"fail"` for predictions. Inspect `prepared.error` before reuse; it is
an `IouResult` on failure and `None` on success. The object is caller-owned:
the package has no global body cache. A call using different side policies
raises `ValueError`.

```python
from volumetric_iou import normalize_pair, prepare_side, volume_iou

gt = prepare_side("part.body.ply", "gt")
if gt.error is not None:
    raise RuntimeError(gt.error.detail)
for pred_raw in predictions:
    pair = normalize_pair(gt, pred_raw, pred_frame="gt")
    result = volume_iou(*pair, timeout=60)
```

Without a timeout, reuse avoids construction as well as file loading.
With the default `isolation="server"` timeout isolation, a prepared side is rebuilt in the child from its
loaded source on every call: native manifold objects are not serialized.
Loading and hashing a body file can still be saved.

## Results and failures

`IouResult` has `status`, `cause`, `iou`, `iog`, `iop`, `method`, `blame`,
`detail`, `warnings` and `diagnostics`. `bool(result)` is true only for
`Status.OK`. Scores are `None` on failure. Successful scores lie in `[0, 1]`.

| Status | Cause | Meaning |
|---|---|---|
| `ok` | `ok` | Boolean result passed the checks |
| `load_failed`, `empty_mesh`, `invalid_coords` | `input` | Unreadable input, no faces, or nonfinite coordinates |
| `open_surface` | `tessellation` | Surface cannot be closed within the accepted patch bound, including unaccepted open doubled copies |
| `not_manifold` | `tessellation` | Unresolved nonmanifold structure, unbalanced cluster edges, or an internal wall without filling |
| `degenerate_volume` | `invalid_geometry` | Flat/zero-thickness input or ambiguous material/validity; includes unresolved doubled surfaces |
| `self_intersecting` | `invalid_geometry` | Individual body passes through itself in `"fail"`; also a cavity without an owner |
| `inverted_volume` | `invalid_geometry` | Unresolved inverted pocket, including a pocket hidden inside a neighboring body |
| `not_normalized` | `precondition` | GT violates the unit-cube contract, or a prediction body has another GT frame |
| `no_engine` | `environment` | manifold3d unavailable |
| `boolean_failed` | `engine` | Boolean construction/check failed, or raw-interpretation resources were exhausted |
| `timeout`, `crashed` | `engine` | Isolated computation exceeded the deadline, died, or raised an exception |

`blame` identifies `gt`, `pred` or `engine`; it is unset for
`not_normalized`. The [failure reference](DEFINITION.md#section-10)
specifies precedence and special cases.

Methods are `boolean:manifold64`, `cgal:manifold64` (CGAL body and manifold
intersection), `cgal`, and `bounds` for disjoint bounding boxes.

Diagnostics include `volume_gt`, `volume_pred`, `volume_inter`, and per-side
records. `volume_inter_raw` preserves an intersection above the smaller
operand that was accepted only as rounding within the `1e-6` check.
`zero_proof` identifies `bounds`, `disjoint_surfaces` or `cgal`.
Warnings record accepted input changes and conditions worth inspecting.
See [the result reference](DEFINITION.md#section-9) for the full catalogue.

## Timeouts and native threads

Timed calls use a child forked from a private server that imports dependencies
but performs no geometry. Forking the active caller can copy a locked native
thread pool and hang, so `isolation="server"` is the default. The server
starts on the first timed call and inherits environment settings then;
later changes to `os.environ` do not propagate to it.

The timeout starts when the request is sent to the child. Initial server
startup (up to 300 seconds) and caller-side input serialization are outside
that deadline. Child-side preparation is inside it. After expiry, termination
can add approximately one to two seconds for the grace period and signals.
Inputs/results use pickle; unserializable inputs raise in the caller.
Server startup failures also raise; they are not `crashed` results.

`isolation="fork"` bypasses the server and input serialization. Use it only
in a worker that has not computed geometry or prepared native bodies.
There is no automatic cleanliness check; a copied busy thread pool may
cause a false timeout. Timeout isolation is supported on Linux only.

CGAL/libigl and the self-intersection detector default to **one thread**
when `IGL_NUM_THREADS` is unset. Set a positive integer to override both:

```bash
IGL_NUM_THREADS=4 python your_script.py
```

Set it before the first CGAL call and before the first timed call: libigl
caches the budget on first use, and the timeout server inherits its
environment when it starts. The detector reads the variable on each call;
`volumetric_iou.self_intersect.self_intersecting_pairs(mesh, threads=4)`
also accepts a per-call override. Missing or invalid/nonpositive budgets
fall back to one. The package does not modify `os.environ`.
These defaults apply to the package's CGAL/libigl code and detector;
NumPy/BLAS/OpenMP and manifold3d have their own threading settings.
Coordinate those settings with the number of worker processes.

## Diagnostics and independent checks

`diagnose_mesh(mesh)` inspects a mesh without computing pair IoU.
`volumetric_iou.winding.arbiter_iou(gt, pred, n=..., seed=..., workers=...)`
is a slower independent sampling check. It requires a surface whose winding
convention represents the intended material; it cannot establish which
interpretation of an ambiguous raw assembly is correct. See
[the arbiter reference](DEFINITION.md#section-13).
