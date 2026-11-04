// The cleaning of one side and its split into shells, as trimesh 4.12 does
// them for mesh_iou._clean and mesh_iou._orient_shells -- the same faces kept,
// the same vertices and their order, the same shells in the same order --
// except that vertices beyond ~9.2e10 are not merged into one (round_key).
//
// clean():  merge_vertices() (positions rounded to 1e-8, first occurrence
//           kept), nondegenerate_faces() (both heights of the triangle's box
//           over its edges v0v1, v0v2 > 1e-8), unique_faces() (the vertex set,
//           first face kept), remove_unreferenced_vertices(), then the edges
//           used by one face, by two and by three or more.
// shells(): split(only_watertight=False) of a surface whose every edge has
//           exactly two faces -- connected components over shared edges, in
//           the order of their first face, each with its vertices in index
//           order; then per shell is_winding_consistent and the signed volume
//           of mass_properties().  trimesh orders the faces inside a shell by
//           numpy's unstable argsort; here they keep their index order, so a
//           shell's volume may differ from trimesh's in the last bits.
//           An edge of 4, 6, ... faces (bodies touching along it) is not cut
//           as trimesh does: its faces are ordered around it by exact
//           predicates and each is joined to its neighbour across a wedge of
//           material, so every shell gets two faces at the edge (see
//           pair_around_edge).  The faces are the same, only their grouping
//           into shells is chosen, so the winding number of the surface -- the
//           sum over shells -- does not change.
//
// Floating point follows numpy's operation order; the module is built with
// -ffp-contract=off so that no product is fused into an addition.
#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace viou {

struct Cleaned {
  std::vector<double> V;     // n x 3
  std::vector<int64_t> F;    // m x 3
  int64_t degenerate_removed = 0, duplicate_removed = 0;
  int64_t boundary_edges = 0;     // used by one face
  int64_t manifold_edges = 0;     // used by exactly two
  int64_t nonmanifold_edges = 0;  // used by three or more
};

namespace detail {

struct Key3 {
  int64_t a, b, c;
  bool operator==(const Key3& o) const { return a == o.a && b == o.b && c == o.c; }
};
struct Key3Hash {
  size_t operator()(const Key3& k) const noexcept {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : {uint64_t(k.a), uint64_t(k.b), uint64_t(k.c)}) {
      h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return size_t(h);
  }
};

inline uint64_t mix64(uint64_t x) {     // splitmix64 finaliser
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
struct Key3Mix {
  size_t operator()(const Key3& k) const noexcept {
    return size_t(mix64(uint64_t(k.c) ^ mix64(uint64_t(k.b) ^ mix64(uint64_t(k.a)))));
  }
};
struct U64Mix {
  size_t operator()(uint64_t k) const noexcept { return size_t(mix64(k)); }
};

// Insert-only hash map, open addressing with linear probing, load <= 1/2: one
// array of slots, zeroed by calloc (fresh pages, no write pass).  Only lookup
// and insertion -- what is found or inserted does not depend on the table's
// layout, so it gives the results std::unordered_map gave (whose node per
// entry cost ~3/4 of the cleaning and of the split on 10^5-face meshes).
// K and V trivially copyable; iteration order is not used for any result.
template <class K, class V, class H>
class FlatMap {
 public:
  explicit FlatMap(size_t expect) {
    size_t cap = 16;
    while (cap < 2 * expect) cap <<= 1;
    alloc(cap);
  }
  // the value of k and whether it was inserted (as v); the pointer is valid
  // until the next emplace
  std::pair<V*, bool> emplace(const K& k, const V& v) {
    if (2 * (size_ + 1) > cap_) grow();
    size_t i = H{}(k) & (cap_ - 1);
    while (slots_[i].full) {
      if (slots_[i].key == k) return {&slots_[i].val, false};
      i = (i + 1) & (cap_ - 1);
    }
    Slot& s = slots_[i];
    s.key = k;
    s.val = v;
    s.full = 1;
    ++size_;
    return {&s.val, true};
  }
  template <class Fn> void for_each(Fn fn) const {
    for (size_t i = 0; i < cap_; ++i)
      if (slots_[i].full) fn(slots_[i].key, slots_[i].val);
  }

 private:
  struct Slot { K key; V val; uint8_t full; };
  struct Free { void operator()(Slot* p) const noexcept { std::free(p); } };
  void alloc(size_t cap) {
    Slot* p = static_cast<Slot*>(std::calloc(cap, sizeof(Slot)));
    if (!p) throw std::bad_alloc();
    slots_.reset(p);
    cap_ = cap;
    size_ = 0;
  }
  void grow() {
    std::unique_ptr<Slot[], Free> old(slots_.release());
    const size_t n = cap_;
    alloc(2 * n);
    for (size_t i = 0; i < n; ++i)
      if (old[i].full) emplace(old[i].key, old[i].val);
  }
  std::unique_ptr<Slot[], Free> slots_;
  size_t cap_ = 0, size_ = 0;
};

// numpy: (v * 10**8).round() -- round half to even.  The key is the bits of
// that double (-0.0 made +0.0), not its int64 conversion: numpy's astype(int64)
// turns |v| >~ 9.2e10 into INT64_MIN on x86 and merged every such vertex into
// one; within the int64 range equal keys are the same either way
inline int64_t round_key(double v) {
  const double r = std::nearbyint(v * 1e8) + 0.0;
  int64_t k;
  std::memcpy(&k, &r, sizeof k);
  return k;
}

// numpy's pairwise summation of a contiguous float64 array (the inner loop of
// np.add.reduce): blocks of 8 accumulators below 128 elements, halving above
inline double pairwise_sum(const double* a, size_t n) {
  if (n < 8) {
    double res = 0.0;
    for (size_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// undirected edge (min, max) of face f, k-th edge (f0f1, f1f2, f2f0)
inline uint64_t edge_key(int64_t a, int64_t b) {
  const uint64_t lo = uint64_t(a < b ? a : b), hi = uint64_t(a < b ? b : a);
  return (hi << 32) | lo;
}

// One use of an edge (a, b), a < b: the face, whether it runs the edge a -> b,
// and its third vertex.
struct EdgeUse {
  int64_t face;
  bool forward;
  int64_t opposite;
};

// Pairs of the faces around one edge (a, b) of 4, 6, ... faces, so that each
// pair bounds a wedge of material; false if there is no such pairing (as many
// faces run the edge a -> b as b -> a, alternating around it).
//
// Looking along d = b - a, the angle of a face is that of its third vertex
// about the edge, increasing counter-clockwise (right hand about d).  A face
// running a -> b has its outward normal d x u at its angle + 90 degrees, so its
// material lies towards smaller angles; a face running b -> a has it towards
// larger ones.  The material wedges are therefore (b->a face, the next a->b
// face counter-clockwise).  Faces at the same angle (coplanar, on one side of
// the edge) are ordered a->b first: a b->a face and an a->b face lying on each
// other then bound no wedge, and each is joined to the neighbour beyond -- two
// bodies touching on a face stay two bodies, as trimesh's cut-and-fill gives.
inline bool pair_around_edge(const double* V, int64_t a, int64_t b, std::vector<EdgeUse>& uses,
                             std::vector<std::pair<int64_t, int64_t>>& pairs) {
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  auto P = [&](int64_t i) { return K::Point_3(V[3 * i], V[3 * i + 1], V[3 * i + 2]); };
  const K::Point_3 pa = P(a), pb = P(b), r = P(uses[0].opposite);
  // 0: angle in [0, pi) from the reference face, 1: in [pi, 2 pi)
  auto half = [&](const K::Point_3& p) {
    const auto s = CGAL::orientation(pa, pb, r, p);
    if (s != CGAL::COPLANAR) return s == CGAL::POSITIVE ? 0 : 1;
    return CGAL::coplanar_orientation(pa, pb, r, p) == CGAL::POSITIVE ? 0 : 1;
  };
  struct Item { EdgeUse u; K::Point_3 p; int h; };
  std::vector<Item> items;
  items.reserve(uses.size());
  int64_t fwd = 0;
  for (const auto& u : uses) {
    const K::Point_3 p = P(u.opposite);
    items.push_back({u, p, half(p)});
    fwd += u.forward;
  }
  if (2 * fwd != int64_t(uses.size())) return false;
  std::sort(items.begin(), items.end(), [&](const Item& x, const Item& y) {
    if (x.h != y.h) return x.h < y.h;
    const auto s = CGAL::orientation(pa, pb, x.p, y.p);
    if (s != CGAL::COPLANAR) return s == CGAL::POSITIVE;
    if (x.u.forward != y.u.forward) return x.u.forward;
    return x.u.face < y.u.face;
  });
  const size_t n = items.size();
  size_t start = 0;
  while (items[start].u.forward) ++start;     // a b->a face; one exists
  for (size_t k = 0; k < n; k += 2) {
    const Item& lo = items[(start + k) % n];
    const Item& hi = items[(start + k + 1) % n];
    if (lo.u.forward || !hi.u.forward) return false;
    pairs.emplace_back(lo.u.face, hi.u.face);
  }
  return true;
}

}  // namespace detail

// V: nv x 3 float64, F: nf x 3 int64 (indices checked by the caller)
inline Cleaned clean(const double* V, int64_t nv, const int64_t* F, int64_t nf) {
  using namespace detail;
  Cleaned out;
  // merge_vertices: referenced vertices only, keyed by rounded position,
  // in index order; the first occurrence keeps its exact coordinates
  std::vector<char> referenced(size_t(nv), nf == 0);
  for (int64_t i = 0; i < 3 * nf; ++i) referenced[size_t(F[i])] = 1;
  std::vector<int64_t> inverse(size_t(nv), 0);
  std::vector<int64_t> keep_v;
  {
    FlatMap<Key3, int64_t, Key3Mix> seen(size_t(std::min(nv, 3 * nf)));
    for (int64_t v = 0; v < nv; ++v) {
      if (!referenced[size_t(v)]) continue;
      const Key3 k{round_key(V[3 * v]), round_key(V[3 * v + 1]), round_key(V[3 * v + 2])};
      auto it = seen.emplace(k, int64_t(keep_v.size()));
      if (it.second) keep_v.push_back(v);
      inverse[size_t(v)] = *it.first;
    }
  }
  std::vector<double> Vm(3 * keep_v.size());
  for (size_t i = 0; i < keep_v.size(); ++i)
    for (int c = 0; c < 3; ++c) Vm[3 * i + c] = V[3 * keep_v[i] + c];

  // nondegenerate_faces(height=1e-8) on the merged mesh
  std::vector<int64_t> Fk;
  Fk.reserve(size_t(3 * nf));
  for (int64_t f = 0; f < nf; ++f) {
    const int64_t i0 = inverse[size_t(F[3 * f])], i1 = inverse[size_t(F[3 * f + 1])],
                  i2 = inverse[size_t(F[3 * f + 2])];
    const double* p0 = &Vm[3 * i0];
    const double* p1 = &Vm[3 * i1];
    const double* p2 = &Vm[3 * i2];
    // triangles.cross: np.cross(v1 - v0, v2 - v1)
    const double ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
    const double wx = p2[0] - p1[0], wy = p2[1] - p1[1], wz = p2[2] - p1[2];
    const double cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
    const double area = std::sqrt((cx * cx + cy * cy) + cz * cz) / 2.0;
    // triangles.extents: heights over the edges v0v1 and v0v2
    const double bx = p2[0] - p0[0], by = p2[1] - p0[1], bz = p2[2] - p0[2];
    const double la = std::sqrt((ux * ux + uy * uy) + uz * uz);
    const double lb = std::sqrt((bx * bx + by * by) + bz * bz);
    const double h0 = la > 1e-8 ? (area * 2) / la : 0.0;
    const double h1 = lb > 1e-8 ? (area * 2) / lb : 0.0;
    if (h0 > 1e-8 && h1 > 1e-8) {
      Fk.push_back(i0);
      Fk.push_back(i1);
      Fk.push_back(i2);
    }
  }
  out.degenerate_removed = nf - int64_t(Fk.size() / 3);

  // unique_faces: the sorted vertex triple, first face kept
  std::vector<int64_t> Fu;
  Fu.reserve(Fk.size());
  {
    FlatMap<Key3, char, Key3Mix> seen(Fk.size() / 3);
    for (size_t f = 0; f < Fk.size() / 3; ++f) {
      int64_t a = Fk[3 * f], b = Fk[3 * f + 1], c = Fk[3 * f + 2];
      if (a > b) std::swap(a, b);
      if (b > c) std::swap(b, c);
      if (a > b) std::swap(a, b);
      if (seen.emplace(Key3{a, b, c}, 1).second)
        Fu.insert(Fu.end(), Fk.begin() + 3 * f, Fk.begin() + 3 * f + 3);
    }
  }
  out.duplicate_removed = int64_t(Fk.size() / 3) - int64_t(Fu.size() / 3);

  // remove_unreferenced_vertices: prefix numbering, index order kept
  const int64_t nm = int64_t(keep_v.size());
  std::vector<int64_t> renum(size_t(nm), -1);
  for (int64_t i : Fu) renum[size_t(i)] = 0;
  int64_t k = 0;
  for (int64_t i = 0; i < nm; ++i)
    if (renum[size_t(i)] == 0) renum[size_t(i)] = k++;
  out.V.resize(size_t(3 * k));
  for (int64_t i = 0; i < nm; ++i)
    if (renum[size_t(i)] >= 0)
      for (int c = 0; c < 3; ++c) out.V[size_t(3 * renum[size_t(i)] + c)] = Vm[size_t(3 * i + c)];
  out.F.resize(Fu.size());
  for (size_t i = 0; i < Fu.size(); ++i) out.F[i] = renum[size_t(Fu[i])];

  // edge multiplicities (edges_sorted grouped)
  FlatMap<uint64_t, int32_t, U64Mix> count(out.F.size() / 2);
  for (size_t f = 0; f < out.F.size() / 3; ++f)
    for (int e = 0; e < 3; ++e)
      ++*count.emplace(edge_key(out.F[3 * f + e], out.F[3 * f + (e + 1) % 3]), 0).first;
  count.for_each([&](uint64_t, int32_t n) {
    if (n == 1) ++out.boundary_edges;
    else if (n == 2) ++out.manifold_edges;
    else ++out.nonmanifold_edges;
  });
  return out;
}

struct Shell {
  std::vector<double> V;
  std::vector<int64_t> F;
  double volume = 0.0;
  bool winding_consistent = true;
};

// shells of a closed surface: every edge has two faces, or an even number of
// faces that pair_around_edge pairs; `ok` false (and no shells) otherwise, or
// when one shell would get two pairs at one edge -- the caller then splits it
// as trimesh does
inline std::vector<Shell> shells(const double* V, int64_t nv, const int64_t* F, int64_t nf,
                                 bool& ok) {
  using namespace detail;
  std::vector<Shell> out;
  ok = false;
  // the first two faces of every edge, and the direction each runs it in
  struct Use { int64_t face0 = -1, face1 = -1; int64_t first0 = -1, first1 = -1; int n = 0; };
  std::vector<Use> edges;     // per edge, in the order of its first use
  edges.reserve(size_t(3 * nf / 2 + 1));
  FlatMap<uint64_t, int64_t, U64Mix> edge_of(size_t(3 * nf / 2 + 1));
  // every face of an edge of 3+ faces
  std::unordered_map<uint64_t, std::vector<EdgeUse>> crowded;
  for (int64_t f = 0; f < nf; ++f)
    for (int e = 0; e < 3; ++e) {
      const int64_t a = F[3 * f + e], b = F[3 * f + (e + 1) % 3];
      const uint64_t key = edge_key(a, b);
      const auto at = edge_of.emplace(key, int64_t(edges.size()));
      if (at.second) edges.emplace_back();
      Use& u = edges[size_t(*at.first)];
      if (u.n == 0) { u.face0 = f; u.first0 = a; }
      else if (u.n == 1) { u.face1 = f; u.first1 = a; }
      ++u.n;
      if (u.n >= 3) {
        auto& list = crowded[key];
        auto use = [&](int64_t g, int64_t first) {
          int64_t opp = -1;
          for (int k = 0; k < 3; ++k) {
            const int64_t v = F[3 * g + k];
            if (v != a && v != b) opp = v;
          }
          return EdgeUse{g, first == std::min(a, b), opp};
        };
        if (u.n == 3) {
          list.push_back(use(u.face0, u.first0));
          list.push_back(use(u.face1, u.first1));
        }
        list.push_back(use(f, a));
      }
    }
  std::vector<int64_t> root(static_cast<size_t>(nf));
  for (int64_t f = 0; f < nf; ++f) root[size_t(f)] = f;
  auto find = [&](int64_t i) {
    while (root[size_t(i)] != i) {
      root[size_t(i)] = root[size_t(root[size_t(i)])];
      i = root[size_t(i)];
    }
    return i;
  };
  auto join = [&](int64_t fa, int64_t fb) {
    const int64_t ra = find(fa), rb = find(fb);
    if (ra != rb) root[size_t(std::max(ra, rb))] = std::min(ra, rb);
  };
  for (const Use& u : edges) {
    if (u.n == 2) {
      if (u.face0 != u.face1) join(u.face0, u.face1);
    } else if (u.n < 2 || u.n % 2) {
      return out;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> pairs;
  std::vector<size_t> pairs_end;     // per crowded edge, the end of its pairs
  for (auto& kv : crowded) {
    const int64_t lo = int64_t(kv.first & 0xffffffffull), hi = int64_t(kv.first >> 32);
    if (!pair_around_edge(V, lo, hi, kv.second, pairs)) return out;
    pairs_end.push_back(pairs.size());
  }
  for (const auto& p : pairs) join(p.first, p.second);
  // a shell with two pairs at one edge is pinched there: not a manifold
  for (size_t k = 0, begin = 0; k < pairs_end.size(); begin = pairs_end[k++]) {
    std::vector<int64_t> seen;
    for (size_t i = begin; i < pairs_end[k]; ++i) seen.push_back(find(pairs[i].first));
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) return out;
  }
  ok = true;
  // components in the order of their first face, faces in index order
  std::vector<int64_t> label(size_t(nf), -1), comp_of_root(size_t(nf), -1);
  int64_t nc = 0;
  for (int64_t f = 0; f < nf; ++f) {
    const int64_t r = find(f);
    if (comp_of_root[size_t(r)] < 0) comp_of_root[size_t(r)] = nc++;
    label[size_t(f)] = comp_of_root[size_t(r)];
  }
  std::vector<std::vector<int64_t>> faces_of(static_cast<size_t>(nc));
  for (int64_t f = 0; f < nf; ++f) faces_of[size_t(label[size_t(f)])].push_back(f);
  // winding: the two uses of an edge run it in opposite directions
  std::vector<char> consistent(size_t(nc), 1);
  for (const Use& u : edges) {
    // a crowded edge's pairs run it both ways by construction
    if (u.n == 2 && u.first0 == u.first1) consistent[size_t(label[size_t(u.face0)])] = 0;
  }
  // vertices of each shell in index order (np.unique of its face indices):
  // one pass over the vertices with the shells of each, not a sort per shell
  std::vector<std::vector<int64_t>> verts_of(static_cast<size_t>(nc));
  {
    std::vector<int64_t> start(size_t(nv) + 1, 0);
    for (int64_t i = 0; i < 3 * nf; ++i) ++start[size_t(F[i]) + 1];
    for (int64_t v = 0; v < nv; ++v) start[size_t(v) + 1] += start[size_t(v)];
    std::vector<int64_t> shell_at(size_t(3 * nf)), fill(start.begin(), start.end() - 1);
    for (int64_t i = 0; i < 3 * nf; ++i)
      shell_at[size_t(fill[size_t(F[i])]++)] = label[size_t(i / 3)];
    std::vector<int64_t> last(size_t(nc), -1);     // the last vertex given to a shell
    for (int64_t v = 0; v < nv; ++v)
      for (int64_t i = start[size_t(v)]; i < start[size_t(v) + 1]; ++i) {
        const int64_t c = shell_at[size_t(i)];
        if (last[size_t(c)] != v) {
          last[size_t(c)] = v;
          verts_of[size_t(c)].push_back(v);
        }
      }
  }
  out.resize(size_t(nc));
  std::vector<int64_t> local(size_t(nv), -1);
  std::vector<double> terms;
  for (int64_t c = 0; c < nc; ++c) {
    Shell& s = out[size_t(c)];
    const auto& fs = faces_of[size_t(c)];
    const auto& verts = verts_of[size_t(c)];
    s.V.resize(3 * verts.size());
    for (size_t i = 0; i < verts.size(); ++i) {
      local[size_t(verts[i])] = int64_t(i);
      for (int k = 0; k < 3; ++k) s.V[3 * i + k] = V[3 * verts[i] + k];
    }
    s.F.resize(3 * fs.size());
    terms.resize(fs.size());
    for (size_t i = 0; i < fs.size(); ++i) {
      for (int e = 0; e < 3; ++e) s.F[3 * i + e] = local[size_t(F[3 * fs[i] + e])];
      const double* p0 = &V[3 * F[3 * fs[i]]];
      const double* p1 = &V[3 * F[3 * fs[i] + 1]];
      const double* p2 = &V[3 * F[3 * fs[i] + 2]];
      // mass_properties integral[0]: cross_x(v1 - v0, v2 - v1) * ((x0 + x1) + x2)
      const double cx = (p1[1] - p0[1]) * (p2[2] - p1[2]) - (p1[2] - p0[2]) * (p2[1] - p1[1]);
      terms[i] = cx * ((p0[0] + p1[0]) + p2[0]);
    }
    s.volume = (0.0 + pairwise_sum(terms.data(), terms.size())) / 6.0;
    s.winding_consistent = consistent[size_t(c)] != 0;
  }
  return out;
}

}  // namespace viou
