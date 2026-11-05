# volumetric-iou

Volumetric intersection over union for triangle meshes. Overlapping parts
are counted once, enclosed cavities are
preserved, and invalid or ambiguous geometry returns a structured failure
instead of a misleading score.

For ground-truth material $G$ and predicted material $P$:

```math
\mathrm{IoU}=\frac{|G\cap P|}{|G\cup P|},\qquad
\mathrm{IoG}=\frac{|G\cap P|}{|G|},\qquad
\mathrm{IoP}=\frac{|G\cap P|}{|P|}.
```

Each body's material is defined by its signed winding number $w\geq1$;
the side is the union of its bodies, with cavities assigned to their body
or assembly. A threshold on the global winding sum of an arbitrary STL
does not, in general, implement this definition.

- Boolean volume computation with `manifold3d.Mesh64` and an exact CGAL fallback.
- Validation of open surfaces, inverted pockets, self-overlap and ambiguous assemblies.
- Separate policies for ground truth and predictions; optional cavity filling.
- Reusable body files, prepared inputs, diagnostics and process timeouts on Linux.
- `exact_booleans=True` for CGAL throughout the Boolean pipeline and the
  computational error bound described in the definition.

## Installation

Python **3.12 or newer**. Linux is the supported platform; timeout isolation
uses Linux process primitives. Other platforms have not been validated.

From a checkout of this repository:

```bash
python -m pip install .
python -c "import volumetric_iou._cgal; print('ok')"
```

This builds the required C++ extension. You need a C++17 compiler and
CMake 3.18 or newer. The isolated Python build installs `scikit-build-core`
and `nanobind`; CMake downloads pinned CGAL and libigl sources, and Boost
and Eigen if they are not found locally. See [source builds](docs/BUILDING.md)
for offline source directories and GMP/MPFR builds.

Runtime dependencies are NumPy, SciPy, trimesh ≥ 4.12, manifold3d and
NetworkX. Python `libigl` is optional and used only by reference tests;
the production CGAL implementation is part of this package. `rtree` is
not required by the metric.

Prebuilt GMP/MPFR wheels are available in the
[GitHub Releases](https://github.com/GENBBB/volumetric-iou/releases).
For version 0.24.1 on Linux x86_64:

```bash
python -m pip install https://github.com/GENBBB/volumetric-iou/releases/download/v0.24.1/volumetric_iou-0.24.1-cp312-abi3-manylinux_2_34_x86_64.whl
```

The wheel target is `cp312-abi3-manylinux_2_34_x86_64`: CPython ≥ 3.12,
Linux x86_64 and glibc ≥ 2.34. GMP/MPFR runtime libraries and their notices
are included. Checksums, tagged source and the build recipe accompany the
wheel in the release. These instructions install directly from GitHub Releases.

## Quick start

This example needs no external mesh files:

```python
import trimesh
from volumetric_iou import normalize_pair, volume_iou

gt = trimesh.creation.box()
pred = trimesh.creation.box()
pred.apply_translation([0.5, 0, 0])
gt, pred = normalize_pair(gt, pred, pred_frame="gt")

result = volume_iou(gt, pred, timeout=60)
if result:
    print(result.iou, result.iog, result.iop)  # approximately 1/3, 1/2, 1/2
else:
    print(result.status.value, result.cause.value, result.blame, result.detail)
```

For real inputs, load both meshes with `trimesh.load(path, force="mesh")`
and use the same normalization step. **Choose the prediction frame explicitly:**

| `pred_frame` | Transformation | Evaluation meaning |
|---|---|---|
| `"gt"` | Same centering and scaling as GT | Preserves prediction scale and placement errors |
| `"bbox"` | Each mesh normalized by its own bounding box | Removes those errors; use only when intended |
| `"div200"` | Prediction divided by 200 about the origin | For predictions already expressed in that convention |

`volume_iou` does not move inputs. GT must lie in `[-0.5, 0.5]³`; predictions
must use the same frame and may extend outside the cube.

## Validation policy

CGAL/libigl and the self-intersection detector use one thread by default.
Set `IGL_NUM_THREADS` to a positive integer before the first metric call
to select a different budget; see [native threads](docs/API.md#timeouts-and-native-threads).

The default `on_self_intersection=("exact", "fail")` resolves GT self-overlap
into its exact material and refuses a prediction whose individual body
passes through itself. Overlap between different bodies is a union;
mere self-contact is not automatically a failure. Inverted pockets are
refused in both modes, except pockets removed by the explicitly requested
filling of an enclosed cavity.

If a raw assembly has multiple admissible interpretations, **all must be
valid and define the same material**. Different materials or mixed valid
and invalid interpretations return `degenerate_volume`. Exceeding the
search budget returns an engine failure rather than selecting the first
interpretation. This is a defined input policy, not a claim that every
possible mesh can be processed within finite resource limits.

`fill_cavities=True` fills enclosed cavities; a pair `(gt_flag, pred_flag)`
sets it separately per side. It changes the evaluation target, so report
the setting along with results. It does not repair an exterior pocket
or make self-overlap valid in `"fail"` mode.

Successful scores are in `[0, 1]`. Failures have no score and include
`status`, `cause`, `blame`, `detail` and diagnostics. For dataset reporting,
exclude invalid GT from the metric population; report prediction failures
separately from the mean over successful pairs, always with sample counts.

## Documentation

- [API and result statuses](docs/API.md)
- [Metric definition, complete algorithm, guarantees and case catalogue](docs/DEFINITION.md)
- [Builds and development](docs/BUILDING.md)
- [Validation evidence and historical measurements](docs/REPORT.md)
- [Changes](CHANGELOG.md) and [contributing](CONTRIBUTING.md)

## Limitations

The metric measures tessellated meshes, not the underlying CAD B-rep.
Coarse tessellation can systematically change IoU. Small self-overlap or
inverted pockets caused by tessellation are still geometry defects under
the validation policy. Fixing the tessellation belongs to the caller.

The default manifold path has checks and exact fallbacks but no proven
general numerical error bound. Use `exact_booleans=True` when the CGAL
bound is required. Input cleanup and accepted crack patches have their
own bounds and are distinct from arithmetic error.

Exact arrangements and ambiguity searches can be expensive. Use `timeout`
for untrusted or batch inputs and decide explicitly how failures enter your
evaluation or training objective. See the definition for resource limits,
large-coordinate limitations and the scope of the guarantees.

## License

The project is distributed under **GPL-3.0-or-later**, except for files with
their own retained notices. The compiled extension uses GPL-covered CGAL
components. See [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).
