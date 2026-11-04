// volumetric_iou._cgal -- the CGAL path of the metric, compiled into the wheel.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <igl/winding_number.h>

#include "cells.h"
#include "clean.h"
#include "mesh_boolean.h"
#include "raw_arrangement.h"
#include "self_intersections.h"
#include "winding_exact.h"

namespace nb = nanobind;
using namespace nb::literals;

using Vertices = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Faces = nb::ndarray<const int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

using MapV = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>;
using MapF = Eigen::Map<const Eigen::Matrix<int64_t, Eigen::Dynamic, 3, Eigen::RowMajor>>;

static void check_faces(Vertices V, Faces F) {
  const int64_t nv = int64_t(V.shape(0)), nf = int64_t(F.shape(0));
  const int64_t* f = F.data();
  for (int64_t i = 0; i < 3 * nf; ++i)
    if (f[i] < 0 || f[i] >= nv)
      throw std::out_of_range("face index " + std::to_string(f[i]) + " out of range [0, " +
                              std::to_string(nv) + ")");
}

// an n x 3 numpy array owning a copy of a row-major Eigen matrix
template <typename T, typename M>
static nb::ndarray<nb::numpy, T, nb::shape<-1, 3>> to_numpy(const M& m) {
  auto* out = new T[3 * size_t(m.rows())];
  std::copy(m.data(), m.data() + 3 * m.rows(), out);
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T, nb::shape<-1, 3>>(out, {size_t(m.rows()), 3}, owner);
}

static nb::tuple solid_and_pocket(Vertices V, Faces F, bool cavity, bool volume) {
  check_faces(V, F);
  viou::SolidAndPocket r;
  {
    nb::gil_scoped_release release;
    r = viou::solid_and_pocket(MapV(V.data(), V.shape(0), 3), MapF(F.data(), F.shape(0), 3),
                               cavity, volume);
  }
  if (!r.valid) throw std::runtime_error("Mesh boolean failed");  // as igl.copyleft.cgal
  nb::tuple out = nb::make_tuple(to_numpy<double>(r.V), to_numpy<int64_t>(r.F),
                                 to_numpy<double>(r.Vp), to_numpy<int64_t>(r.Fp));
  if (cavity) out = nb::tuple(out + nb::make_tuple(r.in_cavity));
  if (volume) out = nb::tuple(out + nb::make_tuple(r.pocket_volume));
  return out;
}

static nb::tuple solid_pocket_overlap(Vertices V, Faces F, std::vector<int64_t> sizes,
                                      std::vector<int64_t> check, bool cavity,
                                      std::vector<int64_t> pockets, std::vector<int64_t> labels,
                                      bool volume) {
  check_faces(V, F);
  int64_t total = 0;
  for (const int64_t s : sizes) {
    if (s < 0) throw std::invalid_argument("sizes must be >= 0");
    total += s;
  }
  if (sizes.empty() || total != int64_t(F.shape(0)))
    throw std::invalid_argument("sizes must sum to the number of faces");
  for (const int64_t k : check)
    if (k < 0 || k >= int64_t(sizes.size()))
      throw std::out_of_range("check: body " + std::to_string(k) + " out of range");
  for (const int64_t k : pockets)
    if (k < 0 || k >= int64_t(sizes.size()))
      throw std::out_of_range("pockets: body " + std::to_string(k) + " out of range");
  if (!labels.empty() && int64_t(labels.size()) != int64_t(F.shape(0)))
    throw std::invalid_argument("labels must give one body per face");
  for (const int64_t k : labels)
    if (k < 0 || k >= int64_t(sizes.size()))
      throw std::out_of_range("labels: body " + std::to_string(k) + " out of range");
  const std::vector<Eigen::Index> sz(sizes.begin(), sizes.end());
  viou::SolidPocketOverlap r;
  {
    nb::gil_scoped_release release;
    r = viou::solid_pocket_overlap(MapV(V.data(), V.shape(0), 3), MapF(F.data(), F.shape(0), 3),
                                   sz, check, cavity, pockets, labels, volume);
  }
  if (!r.valid) throw std::runtime_error("Mesh boolean failed");  // as igl.copyleft.cgal
  nb::tuple out = nb::make_tuple(to_numpy<double>(r.V), to_numpy<int64_t>(r.F),
                                 to_numpy<double>(r.Vp), to_numpy<int64_t>(r.Fp),
                                 nb::cast(r.overlap_faces), nb::cast(r.overlap_volume),
                                 r.winding_max);
  if (cavity) out = nb::tuple(out + nb::make_tuple(r.in_cavity));
  if (volume) out = nb::tuple(out + nb::make_tuple(r.pocket_volume));
  return out;
}

static nb::tuple intersection(Vertices VA, Faces FA, Vertices VB, Faces FB) {
  check_faces(VA, FA);
  check_faces(VB, FB);
  viou::Boolean r;
  {
    nb::gil_scoped_release release;
    r = viou::intersection(MapV(VA.data(), VA.shape(0), 3), MapF(FA.data(), FA.shape(0), 3),
                           MapV(VB.data(), VB.shape(0), 3), MapF(FB.data(), FB.shape(0), 3));
  }
  if (!r.valid) throw std::runtime_error("Mesh boolean failed");  // as igl.copyleft.cgal
  return nb::make_tuple(to_numpy<double>(r.V), to_numpy<int64_t>(r.F));
}

static nb::ndarray<nb::numpy, double, nb::shape<-1>> winding_number(Vertices V, Faces F,
                                                                     Vertices O) {
  check_faces(V, F);
  const size_t n = O.shape(0);
  auto* out = new double[n];
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    nb::gil_scoped_release release;
    // the copies igl.winding_number makes: WindingNumberAABB wants MatrixX* and MatrixXi
    Eigen::MatrixXd Vc = MapV(V.data(), V.shape(0), 3);
    Eigen::MatrixXi Fc = MapF(F.data(), F.shape(0), 3).cast<int>();
    Eigen::VectorXd W;
    if (n > 0 && Fc.rows() > 0) {
      igl::winding_number(Vc, Fc, MapV(O.data(), O.shape(0), 3), W);
      std::copy(W.data(), W.data() + n, out);
    } else {
      std::fill(out, out + n, 0.0);
    }
  }
  return nb::ndarray<nb::numpy, double, nb::shape<-1>>(out, {n}, owner);
}

static nb::ndarray<nb::numpy, int64_t, nb::shape<-1>> winding_exact(Vertices V, Faces F,
                                                                     Vertices O) {
  check_faces(V, F);
  const size_t n = O.shape(0);
  std::vector<int64_t> w;
  {
    nb::gil_scoped_release release;
    w = viou::winding_exact(V.data(), int64_t(V.shape(0)), F.data(), int64_t(F.shape(0)),
                            O.data(), int64_t(n));
  }
  auto* out = new int64_t[n];
  std::copy(w.begin(), w.end(), out);
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
  return nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(out, {n}, owner);
}

static nb::ndarray<nb::numpy, int64_t, nb::shape<-1, 2>> self_intersecting_pairs(
    Vertices V, Faces F, int threads, bool first) {
  if (threads < 1) throw std::invalid_argument("threads must be >= 1");
  check_faces(V, F);
  const int64_t nf = int64_t(F.shape(0));
  const int64_t* f = F.data();
  std::vector<viou::Pair> pairs;
  {
    nb::gil_scoped_release release;
    pairs = viou::self_intersecting_pairs(V.data(), f, nf, threads, first);
  }
  auto* out = new int64_t[2 * pairs.size()];
  for (size_t i = 0; i < pairs.size(); ++i) {
    out[2 * i] = pairs[i].first;
    out[2 * i + 1] = pairs[i].second;
  }
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
  return nb::ndarray<nb::numpy, int64_t, nb::shape<-1, 2>>(out, {pairs.size(), 2}, owner);
}

static nb::dict detector_profile(Vertices V, Faces F) {
  check_faces(V, F);
  viou::DetectorProfile p;
  {
    nb::gil_scoped_release release;
    p = viou::detector_profile(V.data(), F.data(), int64_t(F.shape(0)));
  }
  nb::dict d;
  d["candidates"] = p.candidates;
  d["pairs"] = p.pairs;
  d["first_hit"] = p.first_hit;
  d["seconds_boxes"] = p.seconds_boxes;
  d["seconds_tests"] = p.seconds_tests;
  return d;
}

// an n x 3 numpy array owning a copy of a flat row-major vector
template <typename T>
static nb::ndarray<nb::numpy, T, nb::shape<-1, 3>> rows_numpy(const std::vector<T>& v) {
  auto* out = new T[v.size()];
  std::copy(v.begin(), v.end(), out);
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T, nb::shape<-1, 3>>(out, {v.size() / 3, 3}, owner);
}

static nb::tuple clean(Vertices V, Faces F) {
  check_faces(V, F);
  viou::Cleaned r;
  {
    nb::gil_scoped_release release;
    r = viou::clean(V.data(), int64_t(V.shape(0)), F.data(), int64_t(F.shape(0)));
  }
  return nb::make_tuple(rows_numpy(r.V), rows_numpy(r.F), r.degenerate_removed,
                        r.duplicate_removed, r.boundary_edges, r.manifold_edges,
                        r.nonmanifold_edges);
}

static nb::object shells(Vertices V, Faces F) {
  check_faces(V, F);
  bool ok = false;
  std::vector<viou::Shell> r;
  {
    nb::gil_scoped_release release;
    r = viou::shells(V.data(), int64_t(V.shape(0)), F.data(), int64_t(F.shape(0)), ok);
  }
  if (!ok) return nb::none();
  nb::list out;
  for (const auto& s : r)
    out.append(nb::make_tuple(rows_numpy(s.V), rows_numpy(s.F), s.volume,
                              s.winding_consistent));
  return out;
}

static nb::tuple cells(Vertices V, Faces F) {
  check_faces(V, F);
  viou::Cells r;
  {
    nb::gil_scoped_release release;
    r = viou::cells(V.data(), int64_t(V.shape(0)), F.data(), int64_t(F.shape(0)));
  }
  const size_t nf = r.birth.size();
  auto* birth = new int64_t[nf];
  std::copy(r.birth.begin(), r.birth.end(), birth);
  nb::capsule own_b(birth, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
  auto* side = new int64_t[2 * nf];
  std::copy(r.side.begin(), r.side.end(), side);
  nb::capsule own_s(side, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
  return nb::make_tuple(rows_numpy(r.V), rows_numpy(r.F),
                        nb::ndarray<nb::numpy, int64_t, nb::shape<-1>>(birth, {nf}, own_b),
                        nb::ndarray<nb::numpy, int64_t, nb::shape<-1, 2>>(side, {nf, 2}, own_s),
                        r.num_cells);
}

NB_MODULE(_cgal, m) {
  m.doc() = "CGAL path of volumetric_iou (exact predicates).";
  m.def("convex_interface", [](Vertices V, Faces F) {
    check_faces(V, F);
    nb::gil_scoped_release release;
    return viou::convex_interface(MapV(V.data(), V.shape(0), 3), MapF(F.data(), F.shape(0), 3));
  });
  nb::class_<viou::RawArrangement>(m, "RawArrangement")
      .def("__init__", [](viou::RawArrangement* self, Vertices V, Faces F) {
        check_faces(V, F);
        nb::gil_scoped_release release;
        new (self) viou::RawArrangement(MapV(V.data(), V.shape(0), 3), MapF(F.data(), F.shape(0), 3));
      })
      .def_prop_ro("live", [](const viou::RawArrangement& a) { return a.live; })
      .def("winding", [](const viou::RawArrangement& a, const std::vector<int64_t>& labels) {
        viou::RawArrangement::IntRows r;
        { nb::gil_scoped_release release; r = a.winding(labels); }
        auto* data = new int[size_t(r.size())];
        std::copy(r.data(), r.data() + r.size(), data);
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<int*>(p); });
        return nb::ndarray<nb::numpy, int, nb::ndim<2>>(data,
            {size_t(r.rows()), size_t(r.cols())}, owner);
      })
      .def("region", [](const viou::RawArrangement& a, const std::vector<int64_t>& mask) {
        viou::Boolean r;
        { nb::gil_scoped_release release; r = a.region(mask); }
        return nb::make_tuple(to_numpy<double>(r.V), to_numpy<int64_t>(r.F));
      })
      .def("volume", &viou::RawArrangement::volume, nb::call_guard<nb::gil_scoped_release>())
      .def("filled", &viou::RawArrangement::filled, nb::call_guard<nb::gil_scoped_release>())
      .def("components", &viou::RawArrangement::components, nb::call_guard<nb::gil_scoped_release>())
      .def("in_cavity", &viou::RawArrangement::in_cavity, nb::call_guard<nb::gil_scoped_release>());
  m.def("self_intersecting_pairs", &self_intersecting_pairs, "V"_a, "F"_a, "threads"_a = 1,
        "first"_a = false,
        "Face pairs (i < j, sorted, unique) that intersect beyond their shared vertices or\n"
        "edge; V float64 (n, 3), F int64 (m, 3), both C-contiguous.  first: stop at the\n"
        "first such pair and return it alone.  The GIL is released.");
  m.def("detector_profile", &detector_profile, "V"_a, "F"_a,
        "{candidates, pairs, first_hit, seconds_boxes, seconds_tests}: the detector on one\n"
        "thread, timed by phase -- box traversal, exact tests -- with the traversal index\n"
        "of the first intersecting candidate (-1: none).  The GIL is released.");
  m.def("solid_and_pocket", &solid_and_pocket, "V"_a, "F"_a, "cavity"_a = false,
        "volume"_a = false,
        "(V, F, Vp, Fp): the boundaries of {w >= 1} (the solid) and {w <= -1} (inverted\n"
        "pockets) of one surface, from one exact CGAL arrangement -- libigl's\n"
        "mesh_boolean(V, F, 'union') and the same union of the flipped surface.  cavity:\n"
        "(V, F, Vp, Fp, in_cavity) -- with a pocket, in_cavity tells whether the pockets lie\n"
        "in cavities of the solid: the cells reached from the pocket cells through cells of\n"
        "w <= 0 (or of zero volume) sharing faces of non-zero area hold no unbounded cell and\n"
        "share no vertex with a cell of w <= 0 and non-zero volume outside them (False without\n"
        "a pocket).  Exact, on the same arrangement.  volume: one more item, the pockets'\n"
        "volume summed exactly on the arrangement and rounded once -- never below 0, unlike\n"
        "the float64 volume of (Vp, Fp) for slivers of a rounding step (0.0 without a\n"
        "pocket).  Raises RuntimeError if the winding number is not piecewise constant.  The\n"
        "GIL is released.");
  m.def("solid_pocket_overlap", &solid_pocket_overlap, "V"_a, "F"_a, "sizes"_a, "check"_a,
        "cavity"_a = false, "pockets"_a = std::vector<int64_t>{},
        "labels"_a = std::vector<int64_t>{}, "volume"_a = false,
        "(V, F, Vp, Fp, overlap_faces, overlap_volume, winding_max): solid_and_pocket of\n"
        "bodies stacked in F (sizes[k] faces of body k in turn, the winding summed over them),\n"
        "and from the same arrangement, per body k in check, the faces and divergence volume\n"
        "of the boundary of {w_k >= 2} -- where body k alone winds twice or more; cells of\n"
        "zero volume left out.  winding_max: with check, the largest winding of the sum over\n"
        "the cells of non-zero volume (0 without check).  labels: if given, the body of each\n"
        "face (the faces in any order; sizes still count them).  The pockets (Vp,\n"
        "Fp) are {w <= -1} of the sum and, per body k in pockets, {w_k <= -1}.  cavity:\n"
        "in_cavity appended, as in solid_and_pocket, seeded by the same pockets.  volume: the\n"
        "pockets' exact volume appended last, as in solid_and_pocket.  Raises\n"
        "RuntimeError as solid_and_pocket.  The GIL is released.");
  m.def("intersection", &intersection, "VA"_a, "FA"_a, "VB"_a, "FB"_a,
        "(V, F): the boundary of {w_a >= 1 and w_b >= 1} from one exact CGAL arrangement of\n"
        "both surfaces -- libigl's mesh_boolean(VA, FA, VB, FB, 'intersect').  Raises\n"
        "RuntimeError as solid_and_pocket.  The GIL is released.");
  m.def("winding_number", &winding_number, "V"_a, "F"_a, "O"_a,
        "Generalized winding number of the surface (V, F) at each point of O (k, 3),\n"
        "float64 -- libigl's igl::winding_number with its hierarchy, as igl.winding_number;\n"
        "threads: IGL_NUM_THREADS.  The GIL is released.");
  m.def("winding_exact", &winding_exact, "V"_a, "F"_a, "O"_a,
        "Winding number of the closed surface (V, F) at each point of O (k, 3), int64, exact:\n"
        "signed crossings of a ray, every decision an exact predicate.  A point on the surface,\n"
        "or one every ray tried meets at an edge or a vertex, gets the minimum of int64.  The\n"
        "GIL is released.");
  m.def("clean", &clean, "V"_a, "F"_a,
        "(V, F, degenerate_removed, duplicate_removed, boundary_edges, manifold_edges,\n"
        "nonmanifold_edges): trimesh's merge_vertices, nondegenerate_faces, unique_faces and\n"
        "remove_unreferenced_vertices, with the same result (except that vertices beyond\n"
        "~9.2e10 are not merged into one); then the edges used by one face,\n"
        "by two and by three or more.  The GIL is released.");
  m.def("shells", &shells, "V"_a, "F"_a,
        "[(V, F, volume, winding_consistent)] per shell, as trimesh's split(only_watertight=\n"
        "False) of a surface whose every edge has exactly two faces (faces inside a shell in\n"
        "index order); at an edge of 4, 6, ... faces they are paired around it across wedges\n"
        "of material.  None if some edge has an odd number of faces, its faces cannot be so\n"
        "paired, or a shell would be pinched at an edge.  The GIL is released.");
  m.def("cells", &cells, "V"_a, "F"_a,
        "(V, F, birth, side, num_cells): the cells of space cut by the triangles (V, F),\n"
        "closed or not -- libigl's remesh_self_intersections (exact, equal points stitched)\n"
        "and extract_cells.  F: the resolved faces, birth: the input face each was born on,\n"
        "side (nf, 2): the cell on the positive (normal) / negative side; coincident faces\n"
        "are kept, a zero-thickness cell between neighbours.  Cell 0 is the unbounded one.\n"
        "Input faces with exactly collinear points are dropped.  V: exact coordinates rounded\n"
        "to float64.  The GIL is released.");
}
