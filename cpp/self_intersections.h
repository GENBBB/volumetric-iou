// Self-intersection detector: the detect-only path of libigl's
// copyleft::cgal::SelfIntersectMesh (the one igl.copyleft.cgal.
// remesh_self_intersections(detect_only=True) runs on float64 input), with the
// same kernel and the same per-pair tests, but with the number of threads given
// per call instead of libigl's process-wide IGL_NUM_THREADS, and the coplanarity
// of two faces sharing an edge decided by an exact predicate (libigl rounds a
// plane, which made the count depend on the order of the faces).
//
// Candidates: CGAL box_self_intersection_d over the bounding boxes of the
// non-degenerate faces.  A pair is reported when the faces meet beyond what
// they share by construction: identical faces (3 shared vertices, by index or
// by position) are skipped; 2 shared -- only a coplanar overlap counts; 1
// shared -- the opposite edge of either face meets the other face; none --
// the triangles intersect.  All predicates are exact (Epick: filtered doubles,
// exact fallback).
#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/box_intersection_d.h>
#include <CGAL/intersections.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace viou {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Segment_3 = Kernel::Segment_3;
using Triangle_3 = Kernel::Triangle_3;
using Pair = std::pair<int64_t, int64_t>;

namespace detail {

// Coplanar tests in 2D.  Epick filters orientation statically, but not
// coplanar_orientation (its static filter is commented out in CGAL), and the
// coplanar branches of CGAL's do_intersect run on it, through interval
// arithmetic.  Coplanar points are projected along a coordinate axis on which
// the plane does not project to a line -- an affine bijection of the plane, so
// "which side of a line" is kept -- and decided by the static Orientation_2.
// Exact, as CGAL's predicates: the same answers.
using Point_2 = Kernel::Point_2;

struct Plane2 {
  int drop;  // the coordinate left out
  Point_2 operator()(const Point_3& p) const {
    return drop == 0 ? Point_2(p.y(), p.z()) : drop == 1 ? Point_2(p.x(), p.z())
                                                         : Point_2(p.x(), p.y());
  }
};

// an axis along which the non-collinear p, q, r stay non-collinear: the largest
// normal component first (a guess in doubles), then the others, checked exactly
inline Plane2 plane2(const Point_3& p, const Point_3& q, const Point_3& r) {
  const double ux = q.x() - p.x(), uy = q.y() - p.y(), uz = q.z() - p.z();
  const double vx = r.x() - p.x(), vy = r.y() - p.y(), vz = r.z() - p.z();
  const double n[3] = {std::abs(uy * vz - uz * vy), std::abs(uz * vx - ux * vz),
                       std::abs(ux * vy - uy * vx)};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int i, int j) { return n[i] > n[j]; });
  for (int k = 0; k < 3; ++k) {
    const Plane2 P{order[k]};
    if (CGAL::orientation(P(p), P(q), P(r)) != CGAL::COLLINEAR) return P;
  }
  return Plane2{-1};  // p, q, r collinear: not for the 2D path
}

// two closed coplanar convex polygons are disjoint iff a line through an edge of
// one has the other strictly on its outer side (Minkowski difference)
inline bool coplanar_triangles_meet(const Plane2& P, const Triangle_3& A, const Triangle_3& B) {
  auto separates = [&](const Triangle_3& X, const Triangle_3& Y) {
    const int ox = int(CGAL::orientation(P(X.vertex(0)), P(X.vertex(1)), P(X.vertex(2))));
    for (int i = 0; i < 3; ++i) {
      const Point_2 a = P(X.vertex(i)), b = P(X.vertex((i + 1) % 3));
      bool out = true;
      for (int j = 0; j < 3 && out; ++j)
        out = ox * int(CGAL::orientation(a, b, P(Y.vertex(j)))) < 0;
      if (out) return true;
    }
    return false;
  };
  return !separates(A, B) && !separates(B, A);
}

// the closed segment pq and triangle B, coplanar: the same, with the line pq
// as the segment's own edge
inline bool coplanar_segment_meets(const Plane2& P, const Point_3& p, const Point_3& q,
                                   const Triangle_3& B) {
  const Point_2 p2 = P(p), q2 = P(q);
  const int ob = int(CGAL::orientation(P(B.vertex(0)), P(B.vertex(1)), P(B.vertex(2))));
  for (int i = 0; i < 3; ++i) {
    const Point_2 a = P(B.vertex(i)), b = P(B.vertex((i + 1) % 3));
    if (ob * int(CGAL::orientation(a, b, p2)) < 0 && ob * int(CGAL::orientation(a, b, q2)) < 0)
      return false;
  }
  const int s0 = int(CGAL::orientation(p2, q2, P(B.vertex(0))));
  return !(s0 != 0 && int(CGAL::orientation(p2, q2, P(B.vertex(1)))) == s0 &&
           int(CGAL::orientation(p2, q2, P(B.vertex(2)))) == s0);
}

inline bool double_shared_vertex(const Triangle_3& A, const Triangle_3& B,
                                 const std::pair<int, int>* shared) {
  auto opposite = [](int a0, int a1) {
    for (int c = 0; c < 3; ++c)
      if (c != a0 && c != a1) return c;
    return -1;
  };
  const int a2 = opposite(shared[0].first, shared[1].first);
  // not coplanar: the faces meet only along the shared edge.  An exact
  // predicate: libigl's B.supporting_plane().has_on() tests a plane rounded
  // to double, so its answer depended on which face of the pair was B
  if (!CGAL::coplanar(B.vertex(0), B.vertex(1), B.vertex(2), A.vertex(a2))) return false;
  const int b2 = opposite(shared[0].second, shared[1].second);
  // coplanar, on opposite sides of the shared edge (coplanar_orientation, in 2D:
  // one projection for both faces keeps the plane's orientations coherent)
  const Plane2 P = plane2(B.vertex(0), B.vertex(1), B.vertex(2));
  if (int(CGAL::orientation(P(A.vertex(shared[0].first)), P(A.vertex(shared[1].first)),
                            P(A.vertex(a2)))) *
          int(CGAL::orientation(P(B.vertex(shared[0].second)), P(B.vertex(shared[1].second)),
                                P(B.vertex(b2)))) <
      0)
    return false;
  auto edges_meet = [](const Triangle_3& A, int va, const Triangle_3& B, int vb) {
    Segment_3 sa(A.vertex((va + 1) % 3), A.vertex((va + 2) % 3));
    Segment_3 sb(B.vertex((vb + 1) % 3), B.vertex((vb + 2) % 3));
    return CGAL::do_intersect(sa, sb);
  };
  return CGAL::do_intersect(A.vertex(a2), B) || CGAL::do_intersect(B.vertex(b2), A) ||
         edges_meet(A, shared[0].first, B, shared[1].second) ||
         edges_meet(A, shared[1].first, B, shared[0].second);
}

inline bool single_shared_vertex(const Triangle_3& A, const Triangle_3& B, int va) {
  // the edge of A opposite the shared vertex against B
  const Point_3& p = A.vertex((va + 1) % 3);
  const Point_3& q = A.vertex((va + 2) % 3);
  const CGAL::Orientation op = CGAL::orientation(B.vertex(0), B.vertex(1), B.vertex(2), p);
  const CGAL::Orientation oq = CGAL::orientation(B.vertex(0), B.vertex(1), B.vertex(2), q);
  if (op == oq && op != CGAL::COPLANAR) return false;  // strictly on one side of B's plane
  if (op == CGAL::COPLANAR && oq == CGAL::COPLANAR)
    return coplanar_segment_meets(plane2(B.vertex(0), B.vertex(1), B.vertex(2)), p, q, B);
  return CGAL::do_intersect(Segment_3(p, q), B);
}

inline bool pair_intersects(const int64_t* F, const std::vector<Triangle_3>& T, int64_t fa,
                            int64_t fb) {
  const Triangle_3& A = T[fa];
  const Triangle_3& B = T[fb];
  // shared vertex pairs (index in A, index in B), in the order found; a fixed
  // array, not a vector: no allocation per candidate
  std::pair<int, int> shared[9];
  int total = 0;
  for (int ea = 0; ea < 3; ++ea)
    for (int eb = 0; eb < 3; ++eb)
      if (F[3 * fa + ea] == F[3 * fb + eb] || A.vertex(ea) == B.vertex(eb))
        shared[total++] = {ea, eb};
  if (total == 3) return false;  // duplicate face (by index or by position)
  if (total == 2) return double_shared_vertex(A, B, shared);
  if (total == 1)
    return single_shared_vertex(A, B, shared[0].first) ||
           single_shared_vertex(B, A, shared[0].second);
  // the signs CGAL's do_intersect starts with, its coplanar branch in 2D
  int o[3];
  for (int i = 0; i < 3; ++i)
    o[i] = int(CGAL::orientation(B.vertex(0), B.vertex(1), B.vertex(2), A.vertex(i)));
  if (o[0] != 0 && o[1] == o[0] && o[2] == o[0]) return false;  // A strictly on one side
  if (o[0] == 0 && o[1] == 0 && o[2] == 0)
    return coplanar_triangles_meet(plane2(B.vertex(0), B.vertex(1), B.vertex(2)), A, B);
  int s0 = int(CGAL::orientation(A.vertex(0), A.vertex(1), A.vertex(2), B.vertex(0)));
  if (s0 != 0 &&
      int(CGAL::orientation(A.vertex(0), A.vertex(1), A.vertex(2), B.vertex(1))) == s0 &&
      int(CGAL::orientation(A.vertex(0), A.vertex(1), A.vertex(2), B.vertex(2))) == s0)
    return false;  // B strictly on one side
  return CGAL::do_intersect(A, B);
}

// thrown from the box callback to stop the traversal at the first hit
struct FirstHit {};

inline std::vector<Triangle_3> triangles(const double* V, const int64_t* F, int64_t nf) {
  std::vector<Triangle_3> T;
  T.reserve(nf);
  for (int64_t f = 0; f < nf; ++f) {
    const double* a = V + 3 * F[3 * f];
    const double* b = V + 3 * F[3 * f + 1];
    const double* c = V + 3 * F[3 * f + 2];
    T.emplace_back(Point_3(a[0], a[1], a[2]), Point_3(b[0], b[1], b[2]),
                   Point_3(c[0], c[1], c[2]));
  }
  return T;
}

using Box = CGAL::Box_intersection_d::Box_with_info_d<double, 3, int64_t>;

inline std::vector<Box> face_boxes(const std::vector<Triangle_3>& T) {
  std::vector<Box> boxes;
  boxes.reserve(T.size());
  for (size_t f = 0; f < T.size(); ++f)
    if (!T[f].is_degenerate()) boxes.emplace_back(T[f].bbox(), int64_t(f));
  return boxes;
}

// libigl's default cutoff: a switch of the algorithm, no limit
constexpr std::ptrdiff_t box_cutoff = 1000;

}  // namespace detail

// V: n x 3 row-major float64, F: m x 3 row-major int64 (indices checked by the
// caller).  Returns the intersecting pairs (i < j), sorted, without repeats.
// first: stop at the first intersecting pair and return it alone -- enough to
// tell that the surface is not embedded.  With one thread each candidate is
// tested as the box traversal finds it, so an early hit also saves the rest of
// the traversal; with more, the candidates are listed first and the workers
// stop at the first hit (which pair is returned then depends on the timing).
inline std::vector<Pair> self_intersecting_pairs(const double* V, const int64_t* F, int64_t nf,
                                                 int threads, bool first = false) {
  const std::vector<Triangle_3> T = detail::triangles(V, F, nf);
  std::vector<detail::Box> boxes = detail::face_boxes(T);

  if (first && threads <= 1) {
    std::vector<Pair> out;
    try {
      CGAL::box_self_intersection_d(
          boxes.begin(), boxes.end(),
          [&](const detail::Box& a, const detail::Box& b) {
            if (detail::pair_intersects(F, T, a.info(), b.info())) {
              out.emplace_back(std::min(a.info(), b.info()), std::max(a.info(), b.info()));
              throw detail::FirstHit{};
            }
          },
          detail::box_cutoff);
    } catch (const detail::FirstHit&) {
    }
    return out;
  }

  std::vector<Pair> candidates;
  CGAL::box_self_intersection_d(
      boxes.begin(), boxes.end(),
      [&](const detail::Box& a, const detail::Box& b) {
        candidates.emplace_back(a.info(), b.info());
      },
      detail::box_cutoff);

  const size_t n = candidates.size();
  const size_t nthreads =
      std::max<size_t>(1, std::min<size_t>(size_t(std::max(threads, 1)), n / 256 + 1));
  constexpr size_t chunk = 1024;
  std::atomic<size_t> next{0};
  std::vector<std::vector<Pair>> found(nthreads);
  std::exception_ptr error;
  std::mutex error_lock;
  auto work = [&](size_t t) {
    try {
      for (size_t lo; (lo = next.fetch_add(chunk)) < n;) {
        const size_t hi = std::min(n, lo + chunk);
        for (size_t i = lo; i < hi; ++i) {
          auto [fa, fb] = candidates[i];
          if (detail::pair_intersects(F, T, fa, fb)) {
            found[t].emplace_back(std::min(fa, fb), std::max(fa, fb));
            if (first) {
              next.store(n);
              return;
            }
          }
        }
      }
    } catch (...) {
      std::lock_guard<std::mutex> g(error_lock);
      if (!error) error = std::current_exception();
      next.store(n);
    }
  };
  std::vector<std::thread> pool;
  for (size_t t = 1; t < nthreads; ++t) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
  if (error) std::rethrow_exception(error);

  std::vector<Pair> out;
  for (auto& v : found) out.insert(out.end(), v.begin(), v.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  if (first && out.size() > 1) out.resize(1);
  return out;
}

// Where the detector spends its time on one mesh, one thread: the box
// traversal listing the candidates, the exact test of every candidate, and the
// position of the first intersecting candidate in traversal order (-1: none) --
// how much of both phases a stop at the first hit would skip.
struct DetectorProfile {
  int64_t candidates = 0, pairs = 0, first_hit = -1;
  double seconds_boxes = 0, seconds_tests = 0;
};

inline DetectorProfile detector_profile(const double* V, const int64_t* F, int64_t nf) {
  using clock = std::chrono::steady_clock;
  DetectorProfile p;
  const auto t0 = clock::now();
  const std::vector<Triangle_3> T = detail::triangles(V, F, nf);
  std::vector<detail::Box> boxes = detail::face_boxes(T);
  std::vector<Pair> candidates;
  CGAL::box_self_intersection_d(
      boxes.begin(), boxes.end(),
      [&](const detail::Box& a, const detail::Box& b) {
        candidates.emplace_back(a.info(), b.info());
      },
      detail::box_cutoff);
  const auto t1 = clock::now();
  for (size_t i = 0; i < candidates.size(); ++i)
    if (detail::pair_intersects(F, T, candidates[i].first, candidates[i].second)) {
      ++p.pairs;
      if (p.first_hit < 0) p.first_hit = int64_t(i);
    }
  const auto t2 = clock::now();
  p.candidates = int64_t(candidates.size());
  p.seconds_boxes = std::chrono::duration<double>(t1 - t0).count();
  p.seconds_tests = std::chrono::duration<double>(t2 - t1).count();
  return p;
}

}  // namespace viou
