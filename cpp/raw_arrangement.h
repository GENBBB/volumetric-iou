// Exact arrangement retained while testing alternative oriented shell readings.
// All material comparisons use non-zero-volume cells before double conversion.
#pragma once

#include "mesh_boolean.h"
#include <array>
#include <map>

namespace viou {

// Certificate: after cancelling exactly coincident opposite triangles, the
// remaining chain is outward on a convex hull; all cancelled triangles lie
// in one plane. Any closed subchain with a negative-volume region would need
// an inward hull face, or a volume bounded only by that plane. Neither is
// possible. All pairings therefore have the same nonnegative material.
// A second certificate has only hull triangles, each once or twice outward
// plus once inward (net one). Closed subchains are constant on that hull,
// with fields in [-1,2] summing to one: [1], [1,1,-1], or [2,-1], apart from
// zeros. Assigning the negative field to either containing host leaves only
// 0/1 bodies. This case must never be mixed with an internal interface.
template <typename DerivedV, typename DerivedF>
bool convex_interface(const Eigen::MatrixBase<DerivedV>& vertices,
                      const Eigen::MatrixBase<DerivedF>& faces) {
  using K = CGAL::Epeck;
  using P = K::Point_3;
  std::map<std::array<int64_t, 3>, std::pair<int, std::array<int64_t, 3>>> groups;
  std::map<std::array<int64_t, 3>, size_t> copies;
  std::map<P, int64_t> unique;
  std::vector<P> points;
  auto index = [&](int64_t v) {
    P p(vertices(v, 0), vertices(v, 1), vertices(v, 2));
    auto inserted = unique.emplace(p, int64_t(points.size()));
    if (inserted.second) points.push_back(p);
    return inserted.first->second;
  };
  for (Eigen::Index i = 0; i < faces.rows(); ++i) {
    std::array<int64_t, 3> f{index(faces(i, 0)), index(faces(i, 1)), index(faces(i, 2))};
    if (CGAL::collinear(points[f[0]], points[f[1]], points[f[2]])) continue;
    auto key = f;
    std::sort(key.begin(), key.end());
    const int inversions = (f[0] > f[1]) + (f[0] > f[2]) + (f[1] > f[2]);
    auto& g = groups[key];
    ++copies[key];
    g.first += inversions % 2 ? -1 : 1;
    g.second = key;
  }
  std::vector<std::array<int64_t, 3>> outer;
  std::vector<P> cancelled;
  bool repeated_outer = false;
  for (auto& item : groups) {
    auto f = item.second.second;
    const int sign = item.second.first;
    if (sign == 0) {
      for (auto k : f) cancelled.push_back(points[k]);
    } else {
      if (std::abs(sign) != 1 || (copies[item.first] != 1 && copies[item.first] != 3)) return false;
      repeated_outer = repeated_outer || copies[item.first] == 3;
      if (sign < 0) std::swap(f[0], f[1]);
      outer.push_back(f);
    }
  }
  if (outer.empty()) return false;
  if (cancelled.empty()) {
    if (!repeated_outer) return false;
  } else {
    if (repeated_outer || cancelled.size() < 3) return false;
    size_t b = 1, c = 2;
    while (b < cancelled.size() && cancelled[b] == cancelled[0]) ++b;
    if (b == cancelled.size()) return false;
    while (c < cancelled.size() && CGAL::collinear(cancelled[0], cancelled[b], cancelled[c])) ++c;
    if (c == cancelled.size()) return false;
    const auto plane = K::Plane_3(cancelled[0], cancelled[b], cancelled[c]);
    for (const auto& p : cancelled) if (!plane.has_on(p)) return false;
  }
  std::vector<K::Plane_3> support;
  for (const auto& f : outer) {
    bool known = false;
    for (const auto& p : support)
      if (p.has_on(points[f[0]]) && p.has_on(points[f[1]]) && p.has_on(points[f[2]])) {
        // Coplanar faces must have the same outward orientation.
        known = p.orthogonal_vector() * CGAL::cross_product(points[f[1]]-points[f[0]],
                                                            points[f[2]]-points[f[0]]) > 0;
        if (!known) return false;
        break;
      }
    if (known) continue;
    if (support.size() >= 64) return false;  // only a certificate accelerator
    K::Plane_3 p(points[f[0]], points[f[1]], points[f[2]]);
    for (const auto& q : points) if (p.has_on_positive_side(q)) return false;
    support.push_back(p);
  }
  return support.size() >= 4;
}

class RawArrangement {
 public:
  using IntRows = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  detail::ExactV V;
  MatrixF F;
  Eigen::Matrix<int64_t, Eigen::Dynamic, 1> birth;
  Eigen::MatrixXi uE;
  Eigen::VectorXi uEC, uEE;
  detail::CellsOf cells;
  Eigen::Index input_faces = 0;
  std::vector<int64_t> live;
  IntRows side;

  template <typename DerivedV, typename DerivedF>
  RawArrangement(const Eigen::MatrixBase<DerivedV>& v, const Eigen::MatrixBase<DerivedF>& f)
      : input_faces(f.rows()) {
    igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
    params.stitch_all = true;
    Eigen::MatrixXi IF;
    igl::copyleft::cgal::remesh_self_intersections(v, f, params, V, F, IF, birth);
    Eigen::MatrixXi E;
    Eigen::VectorXi EMAP;
    igl::unique_edge_map(F, E, uE, EMAP, uEC, uEE);
    const size_t patches = igl::extract_manifold_patches(F, EMAP, uEC, uEE, cells.P);
    cells.num_cells = igl::copyleft::cgal::extract_cells(
        V, F, cells.P, uE, EMAP, uEC, uEE, cells.per_patch_cells);
    (void)patches;
    side.resize(F.rows(), 2);
    for (Eigen::Index i = 0; i < F.rows(); ++i)
      side.row(i) = cells.per_patch_cells.row(cells.P(i));
    detail::CellVolumes volumes(V, F, cells);
    live.resize(cells.num_cells);
    for (size_t i = 1; i < cells.num_cells; ++i) live[i] = !volumes.zero_volume(i);
  }

  IntRows winding(const std::vector<int64_t>& labels) const {
    if (Eigen::Index(labels.size()) != input_faces)
      throw std::invalid_argument("raw winding: one label required per input face");
    const int64_t n = labels.empty() ? 1 : *std::max_element(labels.begin(), labels.end()) + 1;
    if (n <= 0 || std::any_of(labels.begin(), labels.end(), [](int64_t k) { return k < 0; }))
      throw std::invalid_argument("raw winding: labels must be nonnegative");
    // A resource refusal, never a substitute for examining a possible reading.
    if (uint64_t(F.rows()) * uint64_t(n) * 2 * sizeof(int) > uint64_t(512) * 1024 * 1024)
      throw std::runtime_error("raw winding: labelled arrangement exceeds 512 MiB");
    Eigen::VectorXi cut_labels(F.rows());
    for (Eigen::Index i = 0; i < F.rows(); ++i) cut_labels(i) = int(labels[size_t(birth(i))]);
    Eigen::MatrixXi W;
    const bool valid = F.rows() == 0 || igl::copyleft::cgal::propagate_winding_numbers(
        V, F, uE, uEC, uEE, size_t(cells.per_patch_cells.rows()), cells.P,
        cells.num_cells, cells.per_patch_cells, cut_labels, W);
    if (!valid) throw std::runtime_error("raw winding is not piecewise constant");
    IntRows result = IntRows::Zero(cells.num_cells, n);
    for (Eigen::Index i = 0; i < F.rows(); ++i)
      for (int s = 0; s < 2; ++s)
        for (int64_t k = 0; k < n; ++k)
          if (2 * k + s < W.cols()) result(side(i, s), k) = W(i, 2 * k + s);
    return result;
  }

  void check_mask(const std::vector<int64_t>& mask) const {
    if (mask.size() != cells.num_cells || (!mask.empty() && mask[0]))
      throw std::invalid_argument("raw region: bounded-cell mask required");
  }

  Boolean region(const std::vector<int64_t>& mask) const {
    check_mask(mask);
    Boolean r;
    detail::extract(V, F, side, [&](const auto& row, int s) { return mask[size_t(row(s))] != 0; },
                    r.V, r.F);
    return r;
  }

  double volume(const std::vector<int64_t>& mask) const {
    check_mask(mask);
    return detail::exact_volume(V, F, side,
        [&](const auto& row, int s) { return mask[size_t(row(s))] != 0; });
  }

  bool in_cavity(const std::vector<int64_t>& material, const std::vector<int64_t>& pocket) const {
    check_mask(material);
    check_mask(pocket);
    return detail::pocket_in_cavity(V, F, side, cells,
        [&](const auto& row, int s) { return material[size_t(row(s))] ? 1 : 0; },
        [&](const auto& row, int s) { return pocket[size_t(row(s))] != 0; });
  }

  std::vector<int64_t> filled(const std::vector<int64_t>& material) const {
    check_mask(material);
    // Void components connect through a face, edge or vertex, as the public
    // cavity filling does. Zero-volume cells do not form a material barrier.
    std::vector<std::vector<size_t>> at_vertex(size_t(V.rows()));
    for (Eigen::Index f = 0; f < F.rows(); ++f)
      for (int j = 0; j < 3; ++j)
        for (int s = 0; s < 2; ++s) {
          const size_t c = size_t(side(f, s));
          if (!material[c] && (c == 0 || live[c])) at_vertex[size_t(F(f, j))].push_back(c);
        }
    std::vector<std::vector<size_t>> neighbours(cells.num_cells);
    for (auto& group : at_vertex) {
      if (group.empty()) continue;
      std::sort(group.begin(), group.end());
      group.erase(std::unique(group.begin(), group.end()), group.end());
      for (size_t k = 1; k < group.size(); ++k) {
        neighbours[group[0]].push_back(group[k]);
        neighbours[group[k]].push_back(group[0]);
      }
    }
    std::vector<char> outside(cells.num_cells, 0);
    std::vector<size_t> stack(1, 0);
    outside[0] = 1;
    while (!stack.empty()) {
      const size_t c = stack.back(); stack.pop_back();
      for (const size_t other : neighbours[c])
        if (!outside[other]) { outside[other] = 1; stack.push_back(other); }
    }
    std::vector<int64_t> out(cells.num_cells, 0);
    for (size_t i = 1; i < out.size(); ++i) out[i] = material[i] || !outside[i];
    return out;
  }

  std::vector<std::vector<int64_t>> components(const std::vector<int64_t>& mask) const {
    check_mask(mask);
    detail::CellVolumes volumes(V, F, cells);
    std::vector<char> seen(cells.num_cells, 0);
    std::vector<std::vector<int64_t>> result;
    for (size_t first = 1; first < cells.num_cells; ++first) {
      if (!mask[first] || seen[first]) continue;
      std::vector<size_t> stack(1, first);
      std::vector<int64_t> group(cells.num_cells, 0);
      seen[first] = 1;
      bool nonempty = false;
      while (!stack.empty()) {
        const size_t c = stack.back(); stack.pop_back();
        group[c] = 1;
        nonempty = nonempty || bool(live[c]);
        for (const auto f : volumes.faces_of[c]) {
          const size_t a = volumes.cell(f, 0), b = volumes.cell(f, 1), other = a == c ? b : a;
          if (mask[other] && !seen[other]) { seen[other] = 1; stack.push_back(other); }
        }
      }
      if (nonempty) result.push_back(std::move(group));
    }
    return result;
  }
};
}  // namespace viou
