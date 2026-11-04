#!/usr/bin/env bash
# Build the package wheel (with volumetric_iou._cgal) and drop the module into
# src/volumetric_iou/, where tests and tools/ import the package from.  The
# python env only runs pip (isolated build env under TMPDIR); it is not changed.
#
#   tools/build_cgal.sh PYTHON [CGAL_DIR] [JOBS]
# CGAL_DIR: an unpacked CGAL-6.0.1 (else CMake downloads it from GitHub).
# Extra CMake settings: CMAKE_ARGS="-DVIOU_GMP=ON -DVIOU_BOOST_DIR=..." (scikit-build-core reads it).
set -euo pipefail
PY=$1; CGAL=${2:-}; JOBS=${3:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export TMPDIR=${TMPDIR:-/tmp/viou_build}; mkdir -p "$TMPDIR"
OUT=$TMPDIR/wheel; rm -rf "$OUT"
args=()
[ -n "$CGAL" ] && args+=(-C "cmake.define.VIOU_CGAL_DIR=$(realpath "$CGAL")")
cd "$ROOT"
CMAKE_BUILD_PARALLEL_LEVEL=$JOBS PIP_NO_CACHE_DIR=1 "$PY" -m pip wheel . --no-deps -q -w "$OUT" "${args[@]}"
whl=$(ls "$OUT"/volumetric_iou-*.whl)
"$PY" - "$whl" "$ROOT/src/volumetric_iou" <<'EOF'
import sys, zipfile
whl, dest = sys.argv[1:]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.startswith("volumetric_iou/_cgal"):
            open(f"{dest}/{n.split('/')[-1]}", "wb").write(z.read(n))
            print("module:", f"{dest}/{n.split('/')[-1]}")
EOF
echo "wheel: $whl"
