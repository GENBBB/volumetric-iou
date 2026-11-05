# Changelog

This public source release starts at **0.24.0**. Earlier version numbers in
the validation report refer to development versions; they are not claims
that earlier public packages or releases are available.

## 0.24.1

- Default CGAL/libigl and self-intersection detection to one thread when
  `IGL_NUM_THREADS` is unset. Positive caller overrides remain supported;
  the detector also retains its explicit per-call `threads` argument.
- Preserve libigl's first-use environment lookup and cache without changing
  the application's environment or unrelated native thread pools.

## 0.24.0

- Validated volumetric IoU, IoG and IoP for triangle meshes, with
  structured input, geometry and engine failures.
- Float64 manifold Booleans, exact CGAL body construction and fallbacks,
  exact zero proofs, and optional CGAL throughout (`exact_booleans=True`).
- Distinct GT/prediction self-overlap policies, optional per-side cavity
  filling, bounded crack patches and handling of doubled surfaces.
- Strict raw-assembly interpretation: all admissible face connections and
  cavity assignments must agree on material and validity. Ambiguity is an
  explicit failure; resource exhaustion does not select a partial answer.
- Entirely inverted raw assemblies are reoriented using exact winding
  before interpretation validation. Sufficient certificates can establish
  agreement across all interpretations or an unavoidable inverted pocket;
  the full search remains the fallback with explicit budgets.
- Body files retain source classification for later `"fail"` use. Loading
  an older body does not revalidate its original STL under current rules.
- Reusable prepared inputs, process-isolated timeouts on Linux, synthetic
  regression tests and portable dataset validation tools.
- English installation/API documentation, the complete metric definition,
  historical validation evidence and third-party license notices.
