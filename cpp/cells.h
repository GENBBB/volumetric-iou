// Cells of space cut by a set of triangles (not necessarily closed): the steps
// of mesh_boolean.h's arrange() up to libigl's extract_cells, without the
// winding numbers -- those need a balanced surface, the cells do not.
//
//   cells(V, F) -> the resolved arrangement (every intersection and coplanar
//   overlap remeshed in exact arithmetic, equal points stitched), each output
//   face with the input face it was born on and the cell on each side of it.
//   Cells around an edge follow the exact angular order of its faces
//   (order_facets_around_edge); coincident faces (same three vertices) are
//   kept, one zero-thickness cell between neighbours of a stack.  Cell 0 is
//   the unbounded one.  Input faces whose three points are collinear (exactly)
//   are dropped first: they bound nothing.
#pragma once

#include <igl/copyleft/cgal/assign_scalar.h>
#include <igl/copyleft/cgal/extract_cells.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/extract_manifold_patches.h>
#include <igl/unique_edge_map.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace viou {

struct Cells {
  std::vector<double> V;         // nv x 3, exact coordinates rounded to float64
  std::vector<int64_t> F;        // nf x 3, the resolved faces
  std::vector<int64_t> birth;    // nf: index of the input face each was born on
  std::vector<int64_t> side;     // nf x 2: cell on the positive (normal) / negative side
  int64_t num_cells = 0;
};

// V: nv x 3 float64, F: m x 3 int64 (indices checked by the caller)
inline Cells cells(const double* Vin, int64_t nv, const int64_t* Fin, int64_t nf) {
  using P3 = CGAL::Epick::Point_3;
  using ExactV =
      Eigen::Matrix<CGAL::Epeck::FT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixV = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixF = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Cells out;
  std::vector<int64_t> kept;
  auto pt = [&](int64_t v) { return P3(Vin[3 * v], Vin[3 * v + 1], Vin[3 * v + 2]); };
  for (int64_t f = 0; f < nf; ++f)
    if (!CGAL::collinear(pt(Fin[3 * f]), pt(Fin[3 * f + 1]), pt(Fin[3 * f + 2])))
      kept.push_back(f);
  if (kept.empty()) {
    out.num_cells = 1;
    return out;
  }
  MatrixV VV = Eigen::Map<const MatrixV>(Vin, nv, 3);
  MatrixF FF(kept.size(), 3);
  for (size_t i = 0; i < kept.size(); ++i)
    for (int c = 0; c < 3; ++c) FF(Eigen::Index(i), c) = Fin[3 * kept[i] + c];
  ExactV V;
  MatrixF F;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> CJ;
  {
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.stitch_all = true;
    Eigen::MatrixXi IF;
    igl::copyleft::cgal::remesh_self_intersections(VV, FF, params, V, F, IF, CJ);
  }
  Eigen::MatrixXi E, uE;
  Eigen::VectorXi EMAP, uEC, uEE;
  igl::unique_edge_map(F, E, uE, EMAP, uEC, uEE);
  Eigen::VectorXi P;
  igl::extract_manifold_patches(F, EMAP, uEC, uEE, P);
  Eigen::MatrixXi per_patch_cells;
  out.num_cells = int64_t(
      igl::copyleft::cgal::extract_cells(V, F, P, uE, EMAP, uEC, uEE, per_patch_cells));
  out.V.resize(3 * size_t(V.rows()));
  for (Eigen::Index i = 0; i < V.rows(); ++i)
    for (Eigen::Index c = 0; c < 3; ++c)
      igl::copyleft::cgal::assign_scalar(V(i, c), false, out.V[size_t(3 * i + c)]);
  out.F.assign(F.data(), F.data() + F.size());
  out.birth.resize(size_t(F.rows()));
  out.side.resize(2 * size_t(F.rows()));
  for (Eigen::Index i = 0; i < F.rows(); ++i) {
    out.birth[size_t(i)] = kept[size_t(CJ(i))];
    out.side[size_t(2 * i)] = per_patch_cells(P(i), 0);
    out.side[size_t(2 * i + 1)] = per_patch_cells(P(i), 1);
  }
  return out;
}

}  // namespace viou
