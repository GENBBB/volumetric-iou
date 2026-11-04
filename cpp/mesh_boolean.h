// Booleans of the metric from CGAL arrangements.
//
// libigl's copyleft::cgal::mesh_boolean (the one igl.copyleft.cgal.mesh_boolean
// runs: the same kernel, Epeck, and the same steps) up to the winding number
// on both sides of every face of the resolved arrangement, w.r.t. each input,
// then extractions from that W:
//   solid_and_pocket -- one surface, two extractions from one arrangement
//                       instead of two arrangements:
//     solid  -- the boundary of {w >= 1}: mesh_boolean(V, F, "union") with an
//               empty second operand;
//     pocket -- the boundary of {w <= -1}: what the same union of the surface
//               with its faces flipped returns (flipping negates w).
//     cavity -- on request, when there is a pocket: whether every pocket
//               cell lies in a cavity of the solid (pocket_in_cavity below);
//   solid_pocket_overlap -- solid_and_pocket of stacked bodies (w summed;
//                       pockets also {w_body <= -1} of bodies asked for)
//                       and, per body, the boundary of {w_body >= 2}
//                       (cells of zero volume left out) with the largest
//                       w of the sum; bodies by blocks or per-face labels;
//   intersection     -- the boundary of {w_a >= 1 and w_b >= 1}:
//                       mesh_boolean(VA, FA, VB, FB, "intersect").
// Each output is cleaned as mesh_boolean cleans it: duplicated faces resolved,
// unreferenced vertices removed, exact coordinates rounded to float64.
#pragma once

#include <igl/copyleft/cgal/assign_scalar.h>
#include <igl/copyleft/cgal/extract_cells.h>
#include <igl/copyleft/cgal/propagate_winding_numbers.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/extract_manifold_patches.h>
#include <igl/parallel_for.h>
#include <igl/remove_unreferenced.h>
#include <igl/resolve_duplicated_faces.h>
#include <igl/unique_edge_map.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace viou {

using MatrixV = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixF = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct SolidAndPocket {
  MatrixV V, Vp;   // solid {w >= 1}, pocket {w <= -1}
  MatrixF F, Fp;
  bool valid = true;   // propagate_winding_numbers: winding piecewise constant
  // asked for (cavity) and a pocket found: the pockets lie in cavities of
  // the solid, apart from the outside and other voids (pocket_in_cavity)
  bool in_cavity = false;
  // asked for (volume): the pockets' volume, exact on the arrangement and
  // rounded once (pocket_volume); 0 without a pocket
  double pocket_volume = 0.0;
};

struct Boolean {
  MatrixV V;
  MatrixF F;
  bool valid = true;   // as SolidAndPocket::valid
};

namespace detail {

using ExactV =
    Eigen::Matrix<CGAL::Epeck::FT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// The faces between a cell inside and one outside, oriented outward, cleaned
// as at the end of mesh_boolean; in(W.row(f), s) tells whether the cell above
// (s = 0) or below (s = 1) face f is inside.
template <typename Inside>
void extract(const ExactV& V, const MatrixF& F, const Eigen::MatrixXi& W, Inside in,
             MatrixV& VC, MatrixF& FC) {
  // KeepInside on the per-side result (0/1) of the winding operation
  std::vector<int64_t> selected;
  for (Eigen::Index i = 0; i < F.rows(); ++i) {
    const bool out = in(W.row(i), 0), inn = in(W.row(i), 1);
    if (!out && inn)
      selected.push_back(i + 1);
    else if (out && !inn)
      selected.push_back(-(i + 1));
  }
  MatrixF kept(selected.size(), 3);
  for (size_t i = 0; i < selected.size(); ++i) {
    const auto idx = std::abs(selected[i]) - 1;
    if (selected[i] > 0)
      kept.row(i) = F.row(idx);
    else
      kept.row(i) = F.row(idx).reverse();
  }
  MatrixF G;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> JJ;
  igl::resolve_duplicated_faces(kept, G, JJ);
  Eigen::VectorXi I, J;
  igl::remove_unreferenced(V.rows(), G, I, J);
  FC = G;
  for (Eigen::Index k = 0; k < FC.size(); ++k) FC.data()[k] = I(FC.data()[k]);
  VC.resize(J.size(), 3);
  igl::parallel_for(
      J.size(),
      [&](Eigen::Index i) {
        for (Eigen::Index c = 0; c < 3; ++c)
          igl::copyleft::cgal::assign_scalar(V(J(i), c), false, VC(i, c));
      },
      1000);
}

// The cells of an arrangement (extract_cells): the patch of each face and the
// cell on the positive (normal, column 0) / negative (1) side of each patch.
// Cell 0 is the unbounded one.
struct CellsOf {
  Eigen::VectorXi P;
  Eigen::MatrixXi per_patch_cells;
  size_t num_cells = 0;
};

// The resolved arrangement of VV, FF (FF's rows: sizes(k) faces of input k in
// turn) and W(f, 2k / 2k+1): the winding number w.r.t. input k above / below
// face f.  Returns propagate_winding_numbers' verdict.  `cells`: if given,
// receives the cells (moved out, no copy).  `face_label`: if given, the input
// of each row of FF instead of the blocks of `sizes` (sizes.size() inputs):
// the rows stay in their order, and the arrangement is that of the unlabelled
// call.
template <typename DerivedV, typename DerivedF>
bool arrange(const Eigen::MatrixBase<DerivedV>& VV, const Eigen::MatrixBase<DerivedF>& FF,
             const std::vector<Eigen::Index>& sizes, ExactV& V, MatrixF& F,
             Eigen::MatrixXi& W, CellsOf* cells = nullptr,
             const std::vector<int64_t>* face_label = nullptr) {
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
  const size_t num_patches = igl::extract_manifold_patches(F, EMAP, uEC, uEE, P);
  Eigen::MatrixXi per_patch_cells;
  const size_t num_cells =
      igl::copyleft::cgal::extract_cells(V, F, P, uE, EMAP, uEC, uEE, per_patch_cells);
  // labels(f) = k: face f was born on input k
  const Eigen::Index n = Eigen::Index(sizes.size());
  Eigen::VectorXi labels(F.rows());
  for (Eigen::Index i = 0; i < F.rows(); ++i) {
    if (face_label != nullptr) {
      labels(i) = int((*face_label)[size_t(CJ(i))]);
      continue;
    }
    Eigen::Index k = 0, end = sizes[0];
    while (CJ(i) >= end && k + 1 < n) end += sizes[++k];
    labels(i) = int(k);
  }
  bool valid = true;
  if (F.rows() > 0)
    valid = igl::copyleft::cgal::propagate_winding_numbers(
        V, F, uE, uEC, uEE, num_patches, P, num_cells, per_patch_cells, labels, W);
  else
    W.resize(0, 2 * n);
  if (W.cols() < 2 * n) {   // an input with no faces left: zero, as mesh_boolean pads
    const Eigen::Index old = W.cols();
    W.conservativeResize(F.rows(), 2 * n);
    W.rightCols(2 * n - old).setConstant(0);
  }
  if (cells != nullptr) {
    cells->P = std::move(P);
    cells->per_patch_cells = std::move(per_patch_cells);
    cells->num_cells = num_cells;
  }
  return valid;
}

// The volume of the cells where in(W.row(f), s) holds: the divergence sum
// over the faces between such a cell and another, each oriented out of it,
// on the exact coordinates of the arrangement -- summed exactly, then rounded
// once, so it is not negative and the faces' order does not matter.  A cell
// of zero volume (between coincident faces) adds 0.  The rounded output of
// extract, summed in float64, may come out below 0 for slivers of the size
// of a rounding step.
template <typename Inside>
double exact_volume(const ExactV& V, const MatrixF& F, const Eigen::MatrixXi& W, Inside in) {
  using ET = CGAL::Epeck::FT::ET;
  ET sum = 0;
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    const bool out = in(W.row(f), 0), inn = in(W.row(f), 1);
    if (out == inn) continue;
    ET p[3][3];
    for (int k = 0; k < 3; ++k)
      for (int d = 0; d < 3; ++d) p[k][d] = CGAL::exact(V(F(f, k), d));
    const ET t = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) -
                 p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) +
                 p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
    // inside below the face (s = 1): its normal leaves the region
    if (inn)
      sum += t;
    else
      sum -= t;
  }
  return CGAL::to_double(sum / 6);
}

// Per cell of an arrangement: its faces of non-zero area between two
// different cells, and whether it has zero volume -- exactly: the sum of
// a.(b x c) over those faces, the normal leaving the cell on its negative
// side (faces of no area add 0: left out).  Summed in intervals first,
// exactly (the exact type, not a lazy chain: a chain of 10^5 lazy sums
// overflows the stack) only if 0 is inside; computed on first use.  A cell
// of zero volume lies between coincident faces of a stack, triangulated
// differently on its two sides.
struct CellVolumes {
  using K = CGAL::Epeck;
  const ExactV& V;
  const MatrixF& F;
  const CellsOf& c;
  std::vector<std::vector<Eigen::Index>> faces_of;
  std::vector<signed char> zero;   // -1: not computed

  CellVolumes(const ExactV& V_, const MatrixF& F_, const CellsOf& c_)
      : V(V_), F(F_), c(c_), faces_of(c_.num_cells), zero(c_.num_cells, -1) {
    auto pt = [&](int64_t v) { return K::Point_3(V(v, 0), V(v, 1), V(v, 2)); };
    for (Eigen::Index f = 0; f < F.rows(); ++f) {
      const size_t a = cell(f, 0), b = cell(f, 1);
      if (a == b || CGAL::collinear(pt(F(f, 0)), pt(F(f, 1)), pt(F(f, 2)))) continue;
      faces_of[a].push_back(f);
      faces_of[b].push_back(f);
    }
  }
  // the cell on side s of face f (0: above, along the normal)
  size_t cell(Eigen::Index f, int s) const { return size_t(c.per_patch_cells(c.P(f), s)); }

  bool zero_volume(size_t x) {
    if (zero[x] < 0) {
      using IT = CGAL::Interval_nt<>;
      using ET = K::FT::ET;
      auto det = [](const auto& a, const auto& b, const auto& d) {
        return a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) +
               a[2] * (b[0] * d[1] - b[1] * d[0]);
      };
      IT si = 0;
      for (const Eigen::Index f : faces_of[x]) {
        IT p[3][3];
        for (int k = 0; k < 3; ++k)
          for (int d = 0; d < 3; ++d) {
            const auto& a = V(F(f, k), d).approx();
            p[k][d] = IT(a.inf(), a.sup());
          }
        const IT t = det(p[0], p[1], p[2]);
        si = cell(f, 1) == x ? si + t : si - t;
      }
      if (si.inf() > 0 || si.sup() < 0) {
        zero[x] = 0;
      } else {
        ET se = 0;
        for (const Eigen::Index f : faces_of[x]) {
          ET p[3][3];
          for (int k = 0; k < 3; ++k)
            for (int d = 0; d < 3; ++d) p[k][d] = CGAL::exact(V(F(f, k), d));
          const ET t = det(p[0], p[1], p[2]);
          if (cell(f, 1) == x)
            se += t;
          else
            se -= t;
        }
        zero[x] = CGAL::is_zero(se) ? 1 : 0;
      }
    }
    return zero[x] == 1;
  }
};

// Whether the pockets {w <= -1} lie in cavities of the solid A = {w >= 1},
// as the package fills them -- a cavity is a shell of A's boundary apart
// from the rest (no vertex in common):
//   Q = the cells reached from the pocket cells through cells of w <= 0 or
//       of zero volume, consecutive cells sharing a face of non-zero area;
//   true iff Q holds no unbounded cell (0) and no vertex of a face of a cell
//       of Q belongs to a face of a cell of w <= 0 and non-zero volume
//       outside Q.
// Then the boundary of Q is a shell of its own: filling A fills Q, pockets
// included, and A and A u {pockets} fill to one solid.  Contact of Q with
// the outside (or another void) at an edge or a vertex alone: false -- the
// boundaries share that vertex, filling leaves Q open, and the two readings
// differ by the pockets.
//   - a cell of zero volume (between coincident faces of a stack) has no
//     interior and separates nothing, whatever its w: crossed freely, and
//     touching it is not touching a void;
//   - a face whose three points are collinear (exactly) has no area: not a
//     passage (its vertices still count for contact).
// W is per face (total(W.row(f), 0 / 1): w above / below face f), valid
// (piecewise constant).  Exact throughout: the cells, the stitched vertices,
// the area test (CGAL::collinear) and the zero volume (the exact sum of the
// cell's boundary tetrahedra) -- no point in a cell is used.
// `pocket(W.row(f), s)`: whether the cell on side s of face f is a pocket
// cell (w <= -1 by default; with stacked bodies also a cell where one body
// alone winds -1 or less, its total may be 0 or more there).
// A pocket cell of total w >= 1 (one body winds -1 or less inside two or
// more others) lies in A itself: A u {it} = A, nothing to prove for it -- it
// is not a seed (a seed flush with the outside would end in false though
// both readings give A).  Pocket cells all of total w >= 1: true.
template <typename Total, typename Pocket>
bool pocket_in_cavity(const ExactV& V, const MatrixF& F, const Eigen::MatrixXi& W,
                      const CellsOf& c, Total total, Pocket pocket) {
  const size_t n = c.num_cells;
  if (n == 0) return false;
  std::vector<int> w(n, 0);
  std::vector<char> seed(n, 0);
  for (Eigen::Index f = 0; f < F.rows(); ++f)
    for (int s = 0; s < 2; ++s) {
      const size_t x = size_t(c.per_patch_cells(c.P(f), s));
      w[x] = total(W.row(f), s);
      seed[x] = pocket(W.row(f), s) ? 1 : 0;
    }
  CellVolumes vol(V, F, c);
  const auto& faces_of = vol.faces_of;
  auto other = [&](Eigen::Index f, size_t x) {
    const size_t a = vol.cell(f, 0);
    return a == x ? vol.cell(f, 1) : a;
  };
  auto zero_volume = [&](size_t x) { return vol.zero_volume(x); };
  // from every pocket cell, across cells of w <= 0 or of zero volume
  std::vector<char> seen(n, 0);
  std::vector<size_t> stack;
  bool any = false;
  for (size_t x = 0; x < n; ++x)
    if (seed[x]) {
      any = true;
      if (w[x] >= 1) continue;       // in A: A u {it} = A
      seen[x] = 1;
      stack.push_back(x);
    }
  if (!any) return false;            // no pocket cell: nothing proven
  if (stack.empty()) return true;    // every pocket cell in A
  while (!stack.empty()) {
    const size_t x = stack.back();
    stack.pop_back();
    if (x == 0) return false;
    for (const Eigen::Index f : faces_of[x]) {
      const size_t y = other(f, x);
      if (seen[y]) continue;
      if (w[y] <= 0 || zero_volume(y)) {
        seen[y] = 1;
        stack.push_back(y);
      }
    }
  }
  // contact at an edge or a vertex: every face, of any area, of a cell of Q
  auto side = [&](Eigen::Index f, int s) { return size_t(c.per_patch_cells(c.P(f), s)); };
  std::vector<char> mark(size_t(V.rows()), 0);
  for (Eigen::Index f = 0; f < F.rows(); ++f)
    if (seen[side(f, 0)] || seen[side(f, 1)])
      for (int k = 0; k < 3; ++k) mark[size_t(F(f, k))] = 1;
  for (Eigen::Index f = 0; f < F.rows(); ++f) {
    if (!mark[size_t(F(f, 0))] && !mark[size_t(F(f, 1))] && !mark[size_t(F(f, 2))]) continue;
    for (int s = 0; s < 2; ++s) {
      const size_t y = side(f, s);
      if (!seen[y] && w[y] <= 0 && (y == 0 || !zero_volume(y))) return false;
    }
  }
  return true;
}

}  // namespace detail

// V: n x 3 float64, F: m x 3 int64 (indices checked by the caller).
// cavity: with a pocket, also decide in_cavity (the same arrangement); costs
// nothing without a pocket.  volume: with a pocket, also its exact volume
// (exact_volume).
template <typename DerivedV, typename DerivedF>
SolidAndPocket solid_and_pocket(const Eigen::MatrixBase<DerivedV>& VV,
                                const Eigen::MatrixBase<DerivedF>& FF, bool cavity = false,
                                bool volume = false) {
  SolidAndPocket out;
  detail::ExactV V;
  MatrixF F;
  Eigen::MatrixXi W;
  detail::CellsOf cells;
  out.valid = detail::arrange(VV, FF, {FF.rows()}, V, F, W, cavity ? &cells : nullptr);
  auto total = [](const auto& w, int s) { return int(w(s)); };
  detail::extract(V, F, W, [](const auto& w, int s) { return w(s) > 0; }, out.V, out.F);
  detail::extract(V, F, W, [](const auto& w, int s) { return w(s) < 0; }, out.Vp, out.Fp);
  if (cavity && out.valid && out.Fp.rows() > 0)
    out.in_cavity = detail::pocket_in_cavity(V, F, W, cells, total,
                                             [](const auto& w, int s) { return w(s) < 0; });
  if (volume && out.Fp.rows() > 0)
    out.pocket_volume = detail::exact_volume(V, F, W, [](const auto& w, int s) { return w(s) < 0; });
  return out;
}

// solid_and_pocket of bodies stacked in FF (sizes(k) faces of body k in
// turn; the winding summed over the bodies, so the solid is that of
// solid_and_pocket on the same input), and from the same arrangement, per
// body k in `check`, the boundary of {w_k >= 2}: where body k alone winds
// twice or more -- it passes through itself; cells of zero volume left out
// (CellVolumes).  An overlap of two bodies is not counted (each winds once
// there).  The pockets are {w <= -1} of the sum and,
// per body k in `pockets`, {w_k <= -1}: an inverted pocket of one body is a
// pocket wherever it lies -- inside another body the sum there is 0 or more.
// With `check`, also the largest winding of the sum over the cells of non-zero
// volume (the unbounded one included): one arrangement tells a pass through
// itself (w_k >= 2) from an overlap of bodies (w >= 2, each w_k <= 1).
// `labels`: if not empty, the body of each face of FF instead of the blocks of
// `sizes` (whose size is then the number of bodies).
struct SolidPocketOverlap : SolidAndPocket {
  std::vector<int64_t> overlap_faces;   // per body in `check`
  std::vector<double> overlap_volume;   // divergence volume of that boundary
  int winding_max = 0;                  // with `check`
};

template <typename DerivedV, typename DerivedF>
SolidPocketOverlap solid_pocket_overlap(const Eigen::MatrixBase<DerivedV>& VV,
                                        const Eigen::MatrixBase<DerivedF>& FF,
                                        const std::vector<Eigen::Index>& sizes,
                                        const std::vector<int64_t>& check, bool cavity = false,
                                        const std::vector<int64_t>& pockets = {},
                                        const std::vector<int64_t>& labels = {},
                                        bool volume = false) {
  SolidPocketOverlap out;
  detail::ExactV V;
  MatrixF F;
  Eigen::MatrixXi W;
  detail::CellsOf cells;
  out.valid = detail::arrange(VV, FF, sizes, V, F, W,
                              cavity || !check.empty() ? &cells : nullptr,
                              labels.empty() ? nullptr : &labels);
  const Eigen::Index n = Eigen::Index(sizes.size());
  auto total = [n](const auto& w, int s) {
    int t = 0;
    for (Eigen::Index k = 0; k < n; ++k) t += w(2 * k + s);
    return t;
  };
  auto pocket = [&](const auto& w, int s) {
    if (total(w, s) < 0) return true;
    for (const int64_t k : pockets)
      if (w(2 * k + s) < 0) return true;
    return false;
  };
  detail::extract(V, F, W, [&](const auto& w, int s) { return total(w, s) > 0; }, out.V, out.F);
  detail::extract(V, F, W, pocket, out.Vp, out.Fp);
  if (cavity && out.valid && out.Fp.rows() > 0)
    out.in_cavity = detail::pocket_in_cavity(V, F, W, cells, total, pocket);
  if (volume && out.Fp.rows() > 0) out.pocket_volume = detail::exact_volume(V, F, W, pocket);
  // a cell of zero volume (between coincident faces of one body, triangulated
  // differently on their two sides) holds no volume, whatever its w_k: not
  // in {w_k >= 2} -- as pocket_in_cavity crosses it freely
  std::unique_ptr<detail::CellVolumes> vol;
  if (!check.empty() && out.valid) vol = std::make_unique<detail::CellVolumes>(V, F, cells);
  if (vol) {
    // the cells by total w, largest first: the first of non-zero volume
    std::vector<int> w(cells.num_cells, 0);
    for (Eigen::Index f = 0; f < F.rows(); ++f)
      for (int s = 0; s < 2; ++s) w[vol->cell(f, s)] = total(W.row(f), s);
    std::vector<size_t> order(cells.num_cells);
    for (size_t x = 0; x < order.size(); ++x) order[x] = x;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return w[a] > w[b]; });
    for (const size_t x : order)
      if (x == 0 || !vol->zero_volume(x)) {
        out.winding_max = w[x];
        break;
      }
  }
  for (const int64_t k : check) {
    MatrixV Vk;
    MatrixF Fk;
    // per face and side: inside {w_k >= 2} and of non-zero volume
    Eigen::MatrixXi in(F.rows(), 2);
    for (Eigen::Index f = 0; f < F.rows(); ++f)
      for (int s = 0; s < 2; ++s)
        in(f, s) = W(f, 2 * k + s) >= 2 && !(vol && vol->zero_volume(vol->cell(f, s)));
    detail::extract(V, F, in, [](const auto& w, int s) { return w(s) != 0; }, Vk, Fk);
    // summed in sorted order: the arrangement's face order varies by run
    std::vector<double> terms(Fk.rows());
    for (Eigen::Index i = 0; i < Fk.rows(); ++i) {
      const Eigen::Vector3d a = Vk.row(Fk(i, 0)), b = Vk.row(Fk(i, 1)), c = Vk.row(Fk(i, 2));
      terms[i] = a.dot(b.cross(c));
    }
    std::sort(terms.begin(), terms.end());
    double vol = 0;
    for (const double t : terms) vol += t;
    out.overlap_faces.push_back(Fk.rows());
    out.overlap_volume.push_back(vol / 6);
  }
  return out;
}

// The two operands stacked as mesh_boolean stacks them: VB after VA, FB
// re-indexed after FA.
template <typename DerivedV, typename DerivedF>
Boolean intersection(const Eigen::MatrixBase<DerivedV>& VA, const Eigen::MatrixBase<DerivedF>& FA,
                     const Eigen::MatrixBase<DerivedV>& VB, const Eigen::MatrixBase<DerivedF>& FB) {
  MatrixV VV(VA.rows() + VB.rows(), 3);
  MatrixF FF(FA.rows() + FB.rows(), 3);
  VV.topRows(VA.rows()) = VA;
  VV.bottomRows(VB.rows()) = VB;
  FF.topRows(FA.rows()) = FA;
  FF.bottomRows(FB.rows()) = FB.array() + int64_t(VA.rows());
  Boolean out;
  detail::ExactV V;
  MatrixF F;
  Eigen::MatrixXi W;
  out.valid = detail::arrange(VV, FF, {FA.rows(), FB.rows()}, V, F, W);
  // BinaryWindingNumberOperations<INTERSECT>: every input's w > 0
  detail::extract(V, F, W, [](const auto& w, int s) { return w(s) > 0 && w(2 + s) > 0; },
                  out.V, out.F);
  return out;
}

}  // namespace viou
