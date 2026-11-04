"""
Exact rational triangle-triangle intersection, as an independent arbiter.

Float coordinates are converted to Fractions, so every predicate below is
decided without rounding: the answer is the truth about these two triangles,
not the truth up to a tolerance.  A reference for the measuring tools (the
crossing detector is checked against it); the metric does not call it.
"""
from fractions import Fraction as F


def _v(p):
    return tuple(F(float(x)) for x in p)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _dists(tri, other):
    n = _cross(_sub(other[1], other[0]), _sub(other[2], other[0]))
    return n, [_dot(n, _sub(p, other[0])) for p in tri]


def _interval(tri, dist, axis):
    """Parameter range where `tri` meets the two planes' intersection line."""
    proj = [p[axis] for p in tri]
    hits = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = dist[a], dist[b]
        if da == 0:
            hits.append(proj[a])
        if db == 0:
            hits.append(proj[b])
        if (da > 0) != (db > 0) and da != 0 and db != 0:
            t = da / (da - db)
            hits.append(proj[a] + (proj[b] - proj[a]) * t)
    return (min(hits), max(hits)) if hits else None


def intersect(t1, t2):
    """
    'disjoint' | 'coplanar' | 'touch' | 'cross'

    'touch' means the shared set exists but lies on the boundary of both
    triangles -- legal contact between two faces, not a self-intersection.
    'cross' means the shared segment passes through the interior of at least
    one of them, which is what a self-intersection is.
    """
    t1, t2 = [_v(p) for p in t1], [_v(p) for p in t2]
    n2, d1 = _dists(t1, t2)
    if all(x > 0 for x in d1) or all(x < 0 for x in d1):
        return "disjoint"
    n1, d2 = _dists(t2, t1)
    if all(x > 0 for x in d2) or all(x < 0 for x in d2):
        return "disjoint"
    if all(x == 0 for x in d1):
        return "coplanar"

    direction = _cross(n1, n2)
    axis = max(range(3), key=lambda k: abs(direction[k]))
    i1, i2 = _interval(t1, d1, axis), _interval(t2, d2, axis)
    if i1 is None or i2 is None:
        return "disjoint"
    lo, hi = max(i1[0], i2[0]), min(i1[1], i2[1])
    if lo > hi:
        return "disjoint"

    # a shared segment of zero length is a single point: contact, not crossing
    if lo == hi:
        return "touch"
    # does the shared segment run through the interior of either triangle?
    mid_t = (lo + hi) / 2
    for tri, dist, other_n in ((t1, d1, n2), (t2, d2, n1)):
        pt = _point_at(tri, dist, axis, mid_t)
        if pt is not None and _strictly_inside(tri, pt):
            return "cross"
    return "touch"


def _point_at(tri, dist, axis, target):
    """3D point on `tri`'s crossing chord at parameter `target` along `axis`."""
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = dist[a], dist[b]
        if da == 0:
            pts.append(tri[a])
        if db == 0:
            pts.append(tri[b])
        if (da > 0) != (db > 0) and da != 0 and db != 0:
            t = da / (da - db)
            pts.append(tuple(tri[a][k] + (tri[b][k] - tri[a][k]) * t for k in range(3)))
    if len(pts) < 2:
        return None
    lo = min(pts, key=lambda p: p[axis])
    hi = max(pts, key=lambda p: p[axis])
    if hi[axis] == lo[axis]:
        return lo
    f = (target - lo[axis]) / (hi[axis] - lo[axis])
    return tuple(lo[k] + (hi[k] - lo[k]) * f for k in range(3))


def _strictly_inside(tri, pt):
    """Barycentric interior test for a point known to lie in the triangle's plane."""
    n = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
    tot = _dot(n, n)
    if tot == 0:
        return False
    w = []
    for a, b in ((1, 2), (2, 0), (0, 1)):
        c = _cross(_sub(tri[a], pt), _sub(tri[b], pt))
        w.append(_dot(n, c))
    return all(x > 0 for x in w) or all(x < 0 for x in w)
