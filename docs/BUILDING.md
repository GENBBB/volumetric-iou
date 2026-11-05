# Building and developing

## Standard source build

Use Linux, CPython ≥ 3.12, a C++17 compiler and CMake ≥ 3.18:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
```

The required `volumetric_iou._cgal` extension is built by scikit-build-core
and nanobind. `cp312-abi3` is the stable Python ABI target, not a promise
of operating-system or CPU portability.

Build inputs:

| Dependency | Source selected by CMake |
|---|---|
| CGAL | 6.0.1 source archive |
| libigl headers | `f68f01c23ac82351e2543ce8c529b6d4c89fdf16` |
| Eigen | system Eigen ≥ 3.4, or 3.4.0 archive |
| Boost | system Boost ≥ 1.72, or 1.86.0 CMake source archive |

Build isolation installs the Python build dependencies. CMake downloads
missing C++ headers; an offline environment must supply the source trees
and the Python build requirements in advance.

The build copies the libigl headers into the build directory and replaces
`closest_facet.cpp` there with the included correction. The upstream file
must have MD5 `f44cdb57c6874ef6ab00adf9794335e6`; a mismatch is a build error.
The source tree of libigl is not modified.

## Local dependency sources

```bash
python -m pip install . \
  -C cmake.define.VIOU_CGAL_DIR=/path/to/CGAL-6.0.1 \
  -C cmake.define.VIOU_IGL_DIR=/path/to/libigl \
  -C cmake.define.VIOU_EIGEN_DIR=/path/to/eigen \
  -C cmake.define.VIOU_BOOST_DIR=/path/to/boost-1.86.0-cmake
```

The Eigen directory contains `Eigen/`; the CGAL tree contains `include/CGAL`;
the libigl tree contains `include/igl`. The explicit Boost directory uses
the source archive layout `libs/*/include`. You may omit options for
dependencies discoverable in the system.

## GMP/MPFR variant

The default exact number implementation uses Boost multiprecision and needs
no GMP runtime. For GMP/MPFR, install their development headers and libraries
in your chosen environment, then build with:

```bash
python -m pip wheel . --no-deps --wheel-dir dist \
  -C cmake.define.VIOU_GMP=ON
```

CMake finds GMP/MPFR and sets the extension's installation RPATH to the
directory containing GMP. A wheel for redistribution needs a separate
platform repair and dependency audit; a wheel built on your machine is
not automatically manylinux-compatible. Check its minimum glibc version,
architecture, bundled dependencies, license texts and source availability.

## Running the source tests

Tests import the working source tree. Build the extension into that tree:

```bash
tools/build_cgal.sh python
python -m pytest -q tests
```

`build_cgal.sh` builds a wheel and extracts its extension into
`src/volumetric_iou/`; it does not install into or change the interpreter's
environment. Dependencies must already be installed. Extra CMake options
can be supplied through `CMAKE_ARGS`, for example `-DVIOU_GMP=ON` and the
local source paths above. Rebuild after editing `cpp/`.

The test extra includes pytest, Shapely and the optional `libigl==2.6.3`
reference bindings. Tests requiring unavailable optional reference bindings
are skipped. The synthetic tests create their own geometry; no private
dataset or server access is required.

The portable tools included here are:

- `dataset_check.py run DATASET OUT --workers N [--mode exact|fail] [--fill]`:
  per-file validation, reports and body files for accepted unfilled inputs.
- `dataset_check.py report OUT`: regenerate the report summary.
- `compare_checks.py NEW OLD`: compare validation runs and diagnostics.
- `build_cgal.sh PYTHON [CGAL_DIR] [JOBS]`: build the source extension.
- `doc_figures.py --out docs/img`: regenerate the definition figures
  (requires the optional `matplotlib` package).

Set `IGL_NUM_THREADS` before the first CGAL call or before starting worker
pools. `dataset_check.py` supplies a default per-worker thread budget when
the caller has not set one. The library itself does not change your thread
environment. Set BLAS/OpenMP environment limits before importing their
libraries when running many worker processes.

## Release validation

The [v0.24.0 release](https://github.com/GENBBB/volumetric-iou/releases/tag/v0.24.0)
includes a GMP/MPFR wheel for CPython ≥ 3.12, Linux x86_64 and glibc ≥ 2.34,
along with `SHA256SUMS`, the tagged source archive, `build-public-wheel.sh`
and `WHEEL-BUILD.md`. The build notes record dependency versions, source
links, build options and validation results. The wheel includes the
system GMP/MPFR copyright notices in addition to the source-tree licenses.

Build source and wheel distributions from the intended public commit.
Test the sdist in a fresh directory, install the wheel in a clean virtual
environment, and run the quick-start example outside the checkout. Inspect
both archives for required C++ sources, the libigl correction, documentation
and license notices. Test GMP and Boost builds separately when distributing
both variants. Publish the corresponding source and build instructions
with each binary release.
