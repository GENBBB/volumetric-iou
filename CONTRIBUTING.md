# Contributing

Build and run the tests as described in [BUILDING.md](docs/BUILDING.md).
The synthetic suite needs no external dataset. The definition in
[DEFINITION.md](docs/DEFINITION.md) is the input and output contract.

For a geometry bug, include a minimal synthetic mesh or a mesh you have
permission to share. Report the package version, Python and platform,
normalization convention, `on_self_intersection`, `fill_cavities`,
`exact_booleans`, timeout, and the result's status, detail and diagnostics.
For performance issues, include face counts, process/thread settings and
whether the timing includes reading files or building a reusable GT body.
Do not attach proprietary CAD models without permission.

Keep fixes small. Add a regression test that demonstrates the geometric
failure and its expected result, rather than duplicating implementation
details. Changes to input interpretation must update the definition and
the affected cases. Rebuild the extension after changes under `cpp/`.

Preserve third-party copyright and license notices, especially in
`cpp/libigl/closest_facet.cpp`. Contributions are provided under the
project's GPL-3.0-or-later license unless a file's existing upstream license
requires otherwise. Do not add dependencies with incompatible terms.

When reporting evaluation results, publish the number of accepted pairs and
the failure rates with the mean score. Distinguish a geometry refusal from
an engine failure, and report cavity filling and coordinate conventions.
