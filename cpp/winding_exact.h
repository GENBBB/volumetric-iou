// Exact winding number of a closed triangle surface at points off it: the
// signed number of crossings of a ray from the point, every decision an exact
// predicate (CGAL::orientation on Epick), no constructed point.
#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace viou {

// a point the winding number was not decided for: it lies on the surface, or
// every ray tried passes through an edge or a vertex of it
constexpr int64_t WINDING_UNDECIDED = std::numeric_limits<int64_t>::min();

namespace detail {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using P3 = K::Point_3;

// ray directions: generic, so that a ray along an axis-aligned CAD edge is not
// the first try
constexpr double RAY_DIRS[][3] = {
    {0.5773502691896257, 0.3141592653589793, 0.7536594627415634},
    {-0.2718281828459045, 0.8660254037844386, 0.4142135623730950},
    {0.1234567890123456, -0.6180339887498949, 0.7763932022500210},
    {-0.7071067811865476, -0.2236067977499790, -0.6708203932499369},
    {0.9238795325112867, 0.1950903220161283, -0.3290067137428394},
    {-0.4472135954999579, 0.5345224838248488, -0.7171371656006362},
    {0.3826834323650898, -0.9128709291752769, 0.1428571428571428},
    {-0.8164965809277260, -0.4082482904638630, 0.4082482904638631},
};

// winding number of the surface at p along the ray p -> p + d; false if the
// ray passes through an edge or a vertex of a face in front of p, or p lies
// on a face.  u, v span the plane normal to d; pu, pv are the (u, v) of the
// vertices, pad bounds their rounding: a face whose padded (u, v) box misses
// p's cannot meet the line, all others go to the exact tests.
inline bool winding_along(const std::vector<P3>& pts, const int64_t* f, int64_t nf,
                          const P3& p, const P3& q, const std::vector<double>& pu,
                          const std::vector<double>& pv, double p_u, double p_v, double pad,
                          int64_t& w) {
  w = 0;
  for (int64_t t = 0; t < nf; ++t) {
    const int64_t ia = f[3 * t], ib = f[3 * t + 1], ic = f[3 * t + 2];
    if (std::min({pu[ia], pu[ib], pu[ic]}) - pad > p_u ||
        std::max({pu[ia], pu[ib], pu[ic]}) + pad < p_u ||
        std::min({pv[ia], pv[ib], pv[ic]}) - pad > p_v ||
        std::max({pv[ia], pv[ib], pv[ic]}) + pad < p_v)
      continue;
    const P3 &a = pts[ia], &b = pts[ib], &c = pts[ic];
    // the line p q meets the closed face iff these signs do not disagree;
    // their common sign is that of n . (q - p), n the face normal
    const int o1 = CGAL::orientation(p, q, a, b), o2 = CGAL::orientation(p, q, b, c),
              o3 = CGAL::orientation(p, q, c, a);
    const bool pos = o1 > 0 || o2 > 0 || o3 > 0, neg = o1 < 0 || o2 < 0 || o3 < 0;
    if (pos && neg) continue;           // the line misses the face
    if (!pos && !neg) return false;     // the line lies in the face's plane
    const int sigma = pos ? 1 : -1;
    // sign of n . (p - a): the crossing is in front of p iff it is -sigma
    const int s = CGAL::orientation(a, b, c, p);
    if (s == sigma) continue;           // the crossing is behind p
    if (s == 0) return false;           // p is on the face
    if (o1 == 0 || o2 == 0 || o3 == 0) return false;   // through an edge or a vertex
    w += sigma;                         // leaving along n counts +1
  }
  return true;
}

}  // namespace detail

// Winding numbers of the closed surface (V, F) at points O, exact; a point on
// the surface or one no ray decides gets WINDING_UNDECIDED.
inline std::vector<int64_t> winding_exact(const double* V, int64_t nv, const int64_t* F,
                                          int64_t nf, const double* O, int64_t no) {
  using namespace detail;
  std::vector<P3> pts;
  pts.reserve(nv);
  double scale = 0.0;
  for (int64_t i = 0; i < nv; ++i) {
    pts.emplace_back(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    for (int k = 0; k < 3; ++k) scale = std::max(scale, std::abs(V[3 * i + k]));
  }
  for (int64_t i = 0; i < no; ++i)
    for (int k = 0; k < 3; ++k) scale = std::max(scale, std::abs(O[3 * i + k]));
  // a projection rounds by a few ulp of the largest coordinate, and the line
  // p -> q (q = p + d rounded) drifts from direction d by ulp(q) per unit of
  // length, over at most ~2 * scale: both far below the pad, which only costs
  // exact tests of a few more faces
  const double s1 = std::max(scale, 1.0), pad = 1e-9 * s1 * s1;

  std::vector<int64_t> out(no, WINDING_UNDECIDED);
  std::vector<bool> done(no, false);
  std::vector<double> pu(nv), pv(nv);
  for (const auto& d : RAY_DIRS) {
    // u, v: a basis of the plane normal to d; a point moving along d keeps its (u, v)
    double u[3] = {-d[1], d[0], 0.0};
    // not std::hypot: its versioned symbol needs glibc 2.35, the wheel targets 2.34
    const double nu = std::sqrt(u[0] * u[0] + u[1] * u[1]);
    for (double& x : u) x /= nu;
    const double v[3] = {d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2],
                         d[0] * u[1] - d[1] * u[0]};
    for (int64_t i = 0; i < nv; ++i) {
      const double* x = V + 3 * i;
      pu[i] = u[0] * x[0] + u[1] * x[1] + u[2] * x[2];
      pv[i] = v[0] * x[0] + v[1] * x[1] + v[2] * x[2];
    }
    bool all = true;
    for (int64_t i = 0; i < no; ++i) {
      if (done[i]) continue;
      const double* o = O + 3 * i;
      const P3 p(o[0], o[1], o[2]), q(o[0] + d[0], o[1] + d[1], o[2] + d[2]);
      int64_t w = 0;
      if (p != q && winding_along(pts, F, nf, p, q, pu, pv,
                                  u[0] * o[0] + u[1] * o[1] + u[2] * o[2],
                                  v[0] * o[0] + v[1] * o[1] + v[2] * o[2], pad, w)) {
        out[i] = w;
        done[i] = true;
      } else {
        all = false;
      }
    }
    if (all) break;
  }
  return out;
}

}  // namespace viou
