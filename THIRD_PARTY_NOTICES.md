# License and third-party notices

volumetric-iou is distributed under **GNU GPL version 3 or, at your option,
any later version** (`GPL-3.0-or-later`). The full text is in [LICENSE](LICENSE).
Files carrying their own upstream license notices retain those notices.

The following components are used to build the native extension. Their
licenses are not replaced by the project's license.

| Component | Selected source | License |
|---|---|---|
| CGAL | 6.0.1 | GPL-3.0-or-later and LGPL-3.0-or-later, per header |
| libigl headers | `f68f01c23ac82351e2543ce8c529b6d4c89fdf16` | MPL-2.0; CGAL integration also requires compliance with CGAL's licenses |
| Eigen | 3.4.0 fallback, or the system version | MPL-2.0, with exceptions identified in upstream files |
| Boost | 1.86.0 fallback, or system ≥ 1.72 | Boost Software License 1.0 |
| nanobind | build requirement ≥ 2.4 | BSD-3-Clause |
| Tessil robin-map | headers supplied with nanobind | MIT |
| GMP / MPFR | optional, discovered at build time | LGPL-3.0-or-later for current library versions; verify the versions distributed in each wheel |

`cpp/libigl/closest_facet.cpp` is derived from libigl's file at the commit
above. Its original notice is retained:

> Copyright (C) 2015 Qingnan Zhou

The local modification resolves a closest point inside a face using
`process_face_case_lifted`, correcting cell ownership for shallow crossings.
It remains under MPL-2.0; its modified source is included in this repository
and source distributions. CMake checks the upstream file's MD5 before
applying the replacement in the build directory.

CGAL's `box_intersection_d.h`, used by the detector, explicitly declares
`GPL-3.0-or-later OR LicenseRef-Commercial` in CGAL 6.0.1. This distribution
uses the open-source licensing path. Changing the exact number backend from
GMP to Boost does not remove that GPL-covered CGAL dependency.

Full license texts supplied here:

- [GPL version 3](LICENSE)
- [LGPL version 3](LICENSES/LGPL-3.0.txt)
- [MPL version 2.0](LICENSES/MPL-2.0.txt)
- [Boost Software License 1.0](LICENSES/BSL-1.0.txt)
- [nanobind BSD notice](LICENSES/nanobind-BSD-3-Clause.txt)
- [robin-map MIT notice](LICENSES/robin-map-MIT.txt)

Corresponding upstream source and licensing information:

- [CGAL 6.0.1 sources](https://github.com/CGAL/cgal/tree/v6.0.1)
  and [license policy](https://www.cgal.org/license.html)
- [libigl pinned source](https://github.com/libigl/libigl/tree/f68f01c23ac82351e2543ce8c529b6d4c89fdf16)
  and [license policy](https://libigl.github.io/license/)
- [Eigen 3.4.0](https://gitlab.com/libeigen/eigen/-/tree/3.4.0)
- [Boost 1.86.0](https://github.com/boostorg/boost/tree/boost-1.86.0)
- [nanobind source](https://github.com/wjakob/nanobind)
- [robin-map source](https://github.com/Tessil/robin-map)
- [GMP source distributions](https://gmplib.org/)
- [MPFR source distributions](https://www.mpfr.org/)

NumPy, SciPy, trimesh, manifold3d and NetworkX are separately installed
Python dependencies. Their own distributions carry their license notices.
They are not vendored in this source tree. Optional test/figure dependencies
include pytest, Shapely, Python libigl bindings and Matplotlib.

When distributing a binary wheel, record the exact native dependencies and
build options, include their required notices, and provide the corresponding
source and build instructions alongside the binary. A local wheel containing
additional bundled libraries needs an additional audit of those libraries.
