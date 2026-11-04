"""Synthetic meshes for tests that trimesh.creation does not provide."""

import numpy as np
import trimesh


def coiled_bar(turns: float = 1.25, radius: float = 1.0, width: float = 0.4,
               height: float = 0.4, pitch: float = 0.2, segments: int = 60) -> trimesh.Trimesh:
    """
    One closed, consistently wound shell that passes through itself.

    A bar of square cross-section coiled into a helix; with more than one turn
    and a pitch smaller than the bar's height the last part of the coil runs
    through the first, so the surface winds twice around that region (w = 2)
    while every cheap check -- watertight, winding consistent, one component --
    passes.
    """
    n = segments + 1
    t = np.linspace(0.0, 2 * np.pi * turns, n)
    rings = []
    for (dr, dz) in ((-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2)):
        r = radius + dr
        z = pitch * t / (2 * np.pi) + dz
        rings.append(np.stack([r * np.cos(t), r * np.sin(t), z], axis=1))
    v = np.stack(rings, axis=1).reshape(-1, 3)           # vertex (i, k) -> 4 i + k
    faces = []
    for i in range(n - 1):
        for k in range(4):
            a, b = 4 * i + k, 4 * i + (k + 1) % 4
            c, d = 4 * (i + 1) + (k + 1) % 4, 4 * (i + 1) + k
            faces += [(a, b, c), (a, c, d)]
    faces += [(0, 2, 1), (0, 3, 2)]
    e = 4 * (n - 1)
    faces += [(e, e + 1, e + 2), (e, e + 2, e + 3)]
    m = trimesh.Trimesh(v, np.array(faces), process=False)
    if m.volume < 0:
        m.invert()
    return m


def klein_bottle(n: int = 48, m: int = 24, radius: float = 3.0) -> trimesh.Trimesh:
    """
    The figure-8 Klein bottle: closed, passes through itself, and cannot be
    oriented -- some edges keep both faces running one way, so its winding
    number is not piecewise constant and CGAL's arrangement refuses it.  The
    grid is offset by half a step so the self-intersection runs through faces,
    not along vertices that cleaning would merge.
    """
    u = np.arange(n) * 2 * np.pi / n
    v = (np.arange(m) + 0.5) * 2 * np.pi / m
    U, V = np.meshgrid(u, v, indexing="ij")
    r = radius + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)
    pts = np.stack([r * np.cos(U), r * np.sin(U),
                    np.sin(U / 2) * np.sin(V) + np.cos(U / 2) * np.sin(2 * V)], -1)

    def idx(i, j):          # u = 2 pi is u = 0 with v -> -v
        return i * m + j % m if i < n else (-j - 1) % m
    faces = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [(a, b, c), (a, c, d)]
    return trimesh.Trimesh(pts.reshape(-1, 3), np.array(faces), process=False)


def shallow_crossing(with_cubes: bool = True) -> trimesh.Trimesh:
    """
    A slab (with a pillar standing in it, to reach up) crossed by a small plate
    at 3 degrees, and 18 cubes of side 0.02 apart from both, above the slab
    beyond the plate.  The slab's top meets the plate's lower face along edges
    of four faces; the cubes stand over thin faces of the top that have such an
    edge, seen at a steeper angle than the plate's.  libigl's arrangement
    (closest_facet, a closest point inside a face) took 3 of them as nested in
    the plate -- the union dropped them.  The cube positions are found for this
    arrangement: another triangulation of the top misses them.
    """
    slab = trimesh.creation.box(extents=[10, 10, 1])
    slab.apply_translation([0, 0, -0.5])
    pillar = trimesh.creation.box(extents=[0.5, 0.5, 5])
    pillar.apply_translation([-4.5, -4.5, 2])
    plate = trimesh.creation.box(extents=[0.4, 0.3, 0.05])
    plate.apply_translation([0.1, 0, 0.025])
    plate = plate.subdivide().subdivide()
    plate.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(3), [0, 1, 0]))
    parts = [slab, plate, pillar]
    if with_cubes:
        for x, ys, zs in ((2.50005, (-2.5563, -2.57505), (0.36136, 0.67989, 1.4534)),
                          (3.50007, (-3.53382, -3.54507), (0.5019, 0.94784, 2.03077)),
                          (4.250085, (-4.26696, -4.272585), (0.60731, 1.14881, 2.46379))):
            for y in ys:
                for z in zs:
                    c = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
                    c.apply_translation([x, y, z])
                    parts.append(c)
    return trimesh.util.concatenate(parts)


def extruded(polygon, height: float = 1.0, centre=(0.3, 0.7)) -> trimesh.Trimesh:
    """
    A prism over a closed polyline that may cross itself: one closed shell
    whose winding number between the caps is the polyline's winding number
    in the plane.  A loop run clockwise gives w = -1 where it leaves the
    counter-clockwise outline (an inverted pocket).  The caps are fans from
    `centre`, which must not be collinear with two vertices.
    """
    p = np.asarray(polygon, np.float64)
    n = len(p)
    c = np.asarray(centre, np.float64)
    v = np.concatenate([np.c_[p, np.zeros(n)], np.c_[p, np.full(n, height)],
                        [[*c, 0.0], [*c, height]]])
    cb, ct = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]        # side, outward for CCW
        faces += [(ct, n + i, n + j), (cb, j, i)]          # top and bottom fans
    return trimesh.Trimesh(v, np.array(faces), process=False)


def via_stl(m: trimesh.Trimesh, path) -> trimesh.Trimesh:
    """`m` written to an STL file at `path` and read back as the package reads
    a file (`process=False`): three vertices of its own per face, coordinates
    rounded to float32."""
    from volumetric_iou import mesh_iou
    m.export(str(path))
    out = mesh_iou._as_mesh(str(path))
    assert len(out.vertices) == 3 * len(out.faces)             # nothing merged
    return out
