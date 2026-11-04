"""
Figures of docs/DEFINITION.md: 2D schemes and 3D renders, written to docs/img/.

    python tools/doc_figures.py [--out docs/img]

Gaps and offsets are drawn enlarged; the patches of 4.3 are built by the
package's own `seams` functions, so the pictures show what the code builds.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Polygon                              # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volumetric_iou import seams                                    # noqa: E402

MAT = "#b9cde5"          # material
MAT2 = "#7f9fc6"         # material covered twice
EDGE = "#3b5675"
LOOP = "#d62728"         # hole loops
PATCH = "#ff9f1c"        # patches
NEG = "#f2a3a3"          # w = -1
WALL = "#7b4fa0"         # one-sided wall
GREY = "#888888"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------- helpers

def ax2(ax, title=None, lim=None):
    ax.set_aspect("equal")
    ax.axis("off")
    if lim:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    if title:
        ax.set_title(title)


def path2(ax, pts, color=EDGE, closed=True, lw=1.6, arrows=True, ls="-", head=15, at=0.55):
    """A polyline with an arrow head on every segment (the direction of the faces)."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P, P[:1]])
    ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round")
    if arrows:
        for a, b in zip(P[:-1], P[1:]):
            m = a + at * (b - a)
            ax.annotate("", xy=m + 0.01 * (b - a), xytext=m - 0.01 * (b - a),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1, mutation_scale=head))


def fill2(ax, pts, color=MAT, alpha=1.0, z=0):
    ax.add_patch(Polygon(np.asarray(pts, float), closed=True, fc=color, ec="none", alpha=alpha, zorder=z))


def square(x0, y0, x1, y1, ccw=True):
    p = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return p if ccw else p[::-1]


def ax3(ax, elev=22, azim=-58, title=None, box=None):
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.computed_zorder = False          # drawn in the order added: loops over faces
    if box is not None:
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title)


def box_faces(lo, hi, skip=()):
    """The six sides of an axis box as quads (outward), minus those in `skip`."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    f = {"bottom": [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
         "top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
         "y0": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
         "y1": [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
         "x0": [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
         "x1": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]}
    return [v for k, v in f.items() if k not in skip]


def poly3(ax, polys, color=MAT, alpha=0.85, edge=EDGE, lw=0.6):
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor=edge, linewidths=lw,
                                         alpha=alpha))


def loop3(ax, pts, color=LOOP, lw=2.2, arrows=True, closed=True):
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P, P[:1]])
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=lw)
    if arrows:
        for a, b in zip(P[:-1], P[1:]):
            m = a + 0.5 * (b - a)
            d = 0.2 * (b - a) / max(np.linalg.norm(b - a), 1e-12)
            ax.quiver(*(m - d / 2), *d, color=color, arrow_length_ratio=0.7, lw=2.0)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name)


# ----------------------------------------------------------------- 4.2

def fig_merge(out):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    T = [np.array([(0, 0), (2, 0), (1, 1.6)]), np.array([(2, 0), (3, 1.6), (1, 1.6)])]
    ax = axs[0]
    for t in T:
        c = t.mean(axis=0)
        s = c + 0.86 * (t - c)
        fill2(ax, s, MAT)
        path2(ax, s, arrows=False)
        ax.plot(s[:, 0], s[:, 1], "o", color=EDGE, ms=3.5)
    ax.annotate("a₁", (1.80, 0.12), fontsize=9)
    ax.annotate("a₂", (2.02, 0.32), fontsize=9)
    ax.annotate("b₁", (0.92, 1.52), fontsize=9, ha="right")
    ax.annotate("b₂", (1.25, 1.62), fontsize=9)
    ax2(ax, "In file: three separate vertices per face\n(gap exaggerated)", (-0.2, 3.2, -0.3, 1.9))

    ax = axs[1]
    for k in range(-1, 3):
        ax.axvline(k + 0.5, color=GREY, ls="--", lw=0.7)
        ax.axhline(k + 0.5, color=GREY, ls="--", lw=0.7)
    for i in range(-1, 3):
        for j in range(-1, 3):
            ax.plot(i, j, "+", color=GREY, ms=6)
    ax.plot([0.18, -0.27], [-0.22, 0.12], "o", color=EDGE, ms=5)
    ax.annotate("a₁", (0.22, -0.30))
    ax.annotate("a₂", (-0.47, 0.18))
    ax.annotate("", xy=(0.02, 0.0), xytext=(0.16, -0.19),
                arrowprops=dict(arrowstyle="->", color=EDGE))
    ax.plot([1.47, 1.53], [1.0, 1.0], "o", color=LOOP, ms=5)
    ax.annotate("c₁   c₂", (1.18, 1.12), color=LOOP)
    ax.text(-1.4, -1.55, "+ marks grid nodes, spacing 1e-8; dashes mark cell borders.\n"
            "a₁, a₂ in one cell: same key, one vertex\n(coordinates of a₁, the first copy).\n"
            "c₁, c₂ are 1e-12 apart, across a cell border:\ndifferent keys leave a gap, patched in 4.3.",
            fontsize=8, va="top")
    ax2(ax, "Merging: key = nearest grid node", (-1.4, 2.6, -1.5, 2.6))

    ax = axs[2]
    for t in T:
        fill2(ax, t, MAT)
        path2(ax, t, arrows=False)
    ax.plot([0, 2, 1, 3], [0, 0, 1.6, 1.6], "o", color=EDGE, ms=4)
    ax.annotate("a", (2.05, -0.2))
    ax.annotate("b", (0.85, 1.7))
    ax2(ax, "After merging: shared vertices a, b;\nedge ab belongs to two faces", (-0.2, 3.2, -0.3, 1.9))
    save(fig, out, "merge_2d.png")


def tri_cube():
    v = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, [t for a, b, c, d in q for t in ((a, b, c), (a, c, d))]


def fig_merge3d(out):
    v, F = tri_cube()
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    polys = []
    for f in F:
        t = v[list(f)]
        c = t.mean(axis=0)
        polys.append(c + 0.82 * (t - c))
    poly3(ax, polys, alpha=0.9)
    ax3(ax, title="In file: 12 triangles, 36 vertices", box=((0, 0, 0), (1, 1, 1)))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    poly3(ax, [v[list(f)] for f in F], alpha=0.9)
    ax.scatter(*v.T, color=EDGE, s=10)
    ax3(ax, title="After merging: 8 vertices, two faces\nper edge; closed surface",
        box=((0, 0, 0), (1, 1, 1)))
    save(fig, out, "merge_3d.png")


# ----------------------------------------------------------------- 4.3

def fig_holes(out):
    fig = plt.figure(figsize=(9, 8))
    # (a) seam of two parts: two halves, rims facing each other
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    poly3(ax, box_faces((0, 0, 0), (1, 1, 0.42), skip=("top",)))
    poly3(ax, box_faces((0, 0, 0.58), (1, 1, 1), skip=("bottom",)))
    lo = [(0, 0, 0.42), (0, 1, 0.42), (1, 1, 0.42), (1, 0, 0.42)]           # clockwise from above
    hi = [(0, 0, 0.58), (1, 0, 0.58), (1, 1, 0.58), (0, 1, 0.58)]           # counterclockwise
    loop3(ax, lo)
    loop3(ax, hi)
    ax3(ax, title="(a) Seam between two parts: opposite\nloop directions form one group",
        box=((0, 0, 0), (1, 1, 1)))
    # (b) slit: one loop, its two long sides face each other
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    x0, x1, y0, y1 = 0.15, 0.85, 0.4, 0.6
    top = [[(0, 0, 1), (1, 0, 1), (1, y0, 1), (0, y0, 1)], [(0, y1, 1), (1, y1, 1), (1, 1, 1), (0, 1, 1)],
           [(0, y0, 1), (x0, y0, 1), (x0, y1, 1), (0, y1, 1)], [(x1, y0, 1), (1, y0, 1), (1, y1, 1), (x1, y1, 1)]]
    poly3(ax, box_faces((0, 0, 0), (1, 1, 1), skip=("top",)) + top)
    loop3(ax, [(x0, y0, 1), (x0, y1, 1), (x1, y1, 1), (x1, y0, 1)])
    ax3(ax, title="(b) Slit in one part: one loop\nwith oppositely directed sides",
        box=((0, 0, 0), (1, 1, 1)))
    # (c) open box
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    poly3(ax, box_faces((0, 0, 0), (1, 1, 1), skip=("top",)))
    loop3(ax, [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)])
    ax3(ax, title="(c) Open box: patch is an entire wall;\nlarge error bound, rejected",
        box=((0, 0, 0), (1, 1, 1)))
    # (d) a sheet
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    s = [(0, 0, 0.45), (1, 0, 0.55), (1, 1, 0.45), (0, 1, 0.55)]
    poly3(ax, [[s[0], s[1], s[2]], [s[0], s[2], s[3]]], color="#d9d9d9")
    loop3(ax, s)
    ax3(ax, title="(d) Sheet: no volume even with a patch;\nclosed, excluded from the bound",
        box=((0, 0, 0), (1, 1, 1)))
    fig.text(0.5, 0.02, "Red: hole boundary loops; arrows: face traversal direction; gaps exaggerated",
             ha="center", fontsize=8.5)
    save(fig, out, "holes_3d.png")


def _fan(V, seq):
    """The fan of the vertex with the least total area (as the side's fan)."""
    best = None
    for k in range(len(seq)):
        r = seq[k:] + seq[:k]
        T = [(r[0], r[i], r[i + 1]) for i in range(1, len(r) - 1)]
        a = seams.tri_area(V, T).sum()
        if best is None or a < best[0] - 1e-12:
            best = (a, T)
    assert best is not None
    return best[1]


def tris2(ax, V, T, color=PATCH, alpha=0.45):
    for t in T:
        p = V[list(t)][:, :2]
        fill2(ax, p, color, alpha=alpha, z=1)
        path2(ax, p, color="#b36b00", arrows=False, lw=0.6)


def fig_candidates(out):
    fig = plt.figure(figsize=(12, 6.7))
    fig.subplots_adjust(hspace=0.45)
    # row 1: one loop -- fan and least-area triangulation
    P = [(0, 0), (4, 0), (4, 1), (1.2, 1), (1.2, 2), (4, 2), (4, 3), (0, 3)]
    V = np.array([(x, y, 0.0) for x, y in P])
    seq = list(range(len(P)))
    for i, (name, T) in enumerate([("Fan: triangles from one vertex;\n"
                                    "extends outside a nonconvex loop (darker)", _fan(V, seq)),
                                   ("Minimum-area triangulation:\nloop with up to 400 points",
                                    seams.poly_close(V, seq) or [])]):
        ax = fig.add_subplot(2, 3, 1 + i)
        tris2(ax, V, T)
        path2(ax, P, color=LOOP)
        ax2(ax, name, (-0.3, 4.3, -0.3, 3.3))
    ax = fig.add_subplot(2, 3, 3)
    ax.text(0, 1.0, "Candidates are not exclusive:\nbuild all applicable types; choose\n"
            "the smallest proven volume\nerror bound (π/4)·I(Z).\n\n"
            "One loop: fan and triangulation.\nTwo loops: fans of both, band,\nbridged triangulation.\n"
            "Three or more: fans of all,\nbridged triangulation.",
            va="top", fontsize=8.8, transform=ax.transAxes)
    ax.axis("off")
    # row 2: a seam -- fans of both loops against the band
    lo = [(0, 0, 0.42), (0, 1, 0.42), (1, 1, 0.42), (1, 0, 0.42)]
    hi = [(0, 0, 0.58), (1, 0, 0.58), (1, 1, 0.58), (0, 1, 0.58)]
    Vs = np.array(lo + hi, float)
    a, b = [0, 1, 2, 3], [4, 5, 6, 7]
    bounds = []
    for i, (name, Z) in enumerate([("Fans of both loops: two squares;\n"
                                    "winding cancels, error bounds do not", _fan(Vs, a) + _fan(Vs, b)),
                                   ("Band between loops: narrow strip\nin the gap, small error bound",
                                    seams.stitch_close(Vs, a, b) or [])]):
        bounds.append(seams.volume_bound(Vs, Z))
        ax = fig.add_subplot(2, 3, 4 + i, projection="3d")
        poly3(ax, box_faces((0, 0, 0), (1, 1, 0.42), skip=("top",)), alpha=0.25)
        poly3(ax, box_faces((0, 0, 0.58), (1, 1, 1), skip=("bottom",)), alpha=0.25)
        poly3(ax, [Vs[list(t)] for t in Z], color=PATCH, alpha=0.8, edge="#b36b00")
        loop3(ax, lo, arrows=False)
        loop3(ax, hi, arrows=False)
        ax3(ax, title=name, box=((0, 0, 0), (1, 1, 1)))
    ax = fig.add_subplot(2, 3, 6)
    ax.text(0, 1.0, "Shown gap: 0.16, millions of times\nwider than a real seam.\n"
            "Fans: two lids, area 1 each,\nbound %.2g.\nBand: strip 4 × 0.16, bound %.2g.\n"
            "For a seam of width 1e-7, the band\nbound is ≈ 2e-11: accepted.\n"
            "Fan bound is unchanged: rejected."
            % tuple(bounds), va="top", fontsize=8.8, transform=ax.transAxes)
    ax.axis("off")
    save(fig, out, "patch_candidates.png")


def fig_bridge(out):
    """One patch with an inner boundary: contours, doubled bridge, triangles."""
    outer = [(0, 0), (4, 0), (4, 3), (0, 3)]
    inner = [(1, 1), (1, 2), (3, 2), (3, 1)]
    V = np.array([(x, y, 0.0) for x, y in outer + inner])
    loops = [list(range(4)), list(range(4, 8))]
    poly = seams.bridged(V, loops)
    T = seams.poly_close(V, poly)
    i = next(k for k, p in enumerate(poly) if p >= 4)
    a, b = V[[poly[i - 1], poly[i]], :2]
    fig = plt.figure(figsize=(12, 4.4))
    grid = fig.add_gridspec(2, 3, height_ratios=(3.1, 1), hspace=0.06, wspace=0.22)
    titles = ["1. Two loops in one group", "2. Bridge: out and back",
              "3. Triangulated patch"]
    notes = ["Outer boundary and inner cutout.\nPatch boundaries shown in red.",
             "Follow the bridge to the inner loop,\nwalk around it, and return.",
             "Region between the loops is closed.\nInner cutout remains open."]
    for k in range(3):
        ax = fig.add_subplot(grid[0, k])
        if k == 2:
            tris2(ax, V, T)
        path2(ax, outer, color=LOOP, head=12)
        path2(ax, inner, color=LOOP, head=12)
        if k == 1:
            normal = np.array([-(b-a)[1], (b-a)[0]])
            normal *= 0.045 / np.linalg.norm(normal)
            for start, end, off in ((a, b, normal), (b, a, -normal)):
                ax.annotate("", xy=end + off, xytext=start + off,
                            arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.8))
        if k == 2:
            ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=EDGE, lw=1.8)
        ax2(ax, titles[k], (-0.25, 4.25, -0.25, 3.3))
        caption = fig.add_subplot(grid[1, k])
        caption.axis("off")
        caption.text(0.5, 0.85, notes[k], ha="center", va="top", fontsize=9,
                     transform=caption.transAxes)
    save(fig, out, "patch_bridge.png")


def fig_parts(out):
    fig = plt.figure(figsize=(9, 3.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    poly3(ax, box_faces((0, 0, 0), (4, 3, 0.5)), color="#d9d9d9")
    poly3(ax, box_faces((4.8, 1, 0), (5.8, 2, 1), skip=("top",)), color=MAT)
    loop3(ax, [(4.8, 1, 1), (4.8, 2, 1), (5.8, 2, 1), (5.8, 1, 1)])
    ax3(ax, elev=28, azim=-62, title="One side: closed plate and open cube",
        box=((0, 0, 0), (5.8, 3, 1)))
    lid = np.array([(4.8, 1, 1), (4.8, 2, 1), (5.8, 2, 1), (5.8, 1, 1)], float)
    bound = seams.volume_bound(lid, _fan(lid, [0, 1, 2, 3]))
    ax = fig.add_subplot(1, 2, 2)
    ax.text(0, 0.9, "Parts being closed: components with patch loops,\nincluding their patches. "
            "Here: the cube\nwith its lid (volume 1), not the whole side (volume 7).\n\n"
            "Lid-patch bound %.3g > 1e-3 · 1: rejected.\nA side-volume threshold "
            "grows with the plate;\na large enough plate would make the cube pass.\n\n" % bound +
            "Components with loops in one group\ncount as one part being closed\n"
            "(two seam halves form one part).", va="top", fontsize=9, transform=ax.transAxes)
    ax.axis("off")
    save(fig, out, "patch_parts.png")


def fig_chain(out):
    fig = plt.figure(figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.18, bottom=0.18)
    ax = fig.add_subplot(1, 2, 1)
    fill2(ax, square(0, 0, 4, 3), MAT)
    path2(ax, [(0, 3), (0, 1.75)], closed=False)
    path2(ax, [(0, 1.25), (0, 0), (4, 0), (4, 3), (2, 3), (0, 3)], closed=False)
    path2(ax, [(0, 1.75), (0, 1.25)], color=PATCH, closed=False, ls="--")
    path2(ax, [(2, 0), (2, 3)], color=WALL, closed=False, lw=2.2)
    for p in [(2, 0), (2, 3)]:
        ax.plot(*p, "o", color=WALL, ms=6)
    ax.annotate("crack\npatch", xy=(0, 1.5), xytext=(-1.6, 1.5),
                va="center", color="#b36b00", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#b36b00"))
    ax.text(2.12, 1.35, "partition", color=WALL, fontsize=9)
    ax.annotate("joint: three faces", xy=(2, 3), xytext=(2.5, 3.65),
                color=WALL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=WALL))
    ax2(ax, "(a) Box cross-section", (-1.85, 4.4, -0.3, 4.0))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    poly3(ax, box_faces((0, 0, 0), (2, 1, 1)), alpha=0.18)
    poly3(ax, [[(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]], color=WALL, alpha=0.7, edge=WALL)
    ax.add_collection3d(Line3DCollection([[(1, 0, 0), (1, 1, 0)], [(1, 1, 0), (1, 1, 1)],
                                          [(1, 1, 1), (1, 0, 1)], [(1, 0, 1), (1, 0, 0)]],
                                         colors=WALL, linewidths=2.5))
    ax3(ax, title="(b) Partition and its wall joints",
        box=((0, 0, 0), (2, 1, 1)))
    fig.text(0.5, 0.04, "The patch closes the crack. Three faces still meet at the partition joint: "
             "direction balance is not restored.", ha="center", fontsize=9)
    save(fig, out, "patch_chain.png")


# ----------------------------------------------------------------- 4.4

def fig_split(out):
    fig = plt.figure(figsize=(13, 3.8))
    ax = fig.add_subplot(1, 4, 1)
    fill2(ax, square(0, 0, 1, 1))
    fill2(ax, square(1, 1, 2, 2), MAT)
    path2(ax, square(0, 0, 1, 1))
    path2(ax, square(1, 1, 2, 2), color="#2a7a3a")
    ax.plot(1, 1, "o", color="k", ms=5)
    ax.text(0.4, 0.45, "A")
    ax.text(1.4, 1.45, "B")
    ax2(ax, "(a) Four segments, alternating:\nwedges separate A and B into two shells",
        (-0.2, 2.2, -0.2, 2.2))
    ax = fig.add_subplot(1, 4, 2)
    fill2(ax, square(0, 0, 1, 1))
    fill2(ax, square(1, 1, 2, 2), MAT)
    path2(ax, square(0, 0, 1, 1))
    path2(ax, square(1, 1, 2, 2, ccw=False), color="#2a7a3a")
    ax.plot(1, 1, "o", color="k", ms=5)
    ax2(ax, "(b) B is inverted:\nnonalternating at the point;\ncutting leaves two closed pieces;\n"
        "4.6 flips the inverted piece", (-0.2, 2.2, -0.2, 2.2))
    ax = fig.add_subplot(1, 4, 3)
    fill2(ax, square(0, 0, 2, 1.4))
    path2(ax, square(0, 0, 2, 1.4))
    path2(ax, [(1, 0), (1, 1.4)], color=WALL, closed=False, lw=2.2)
    ax.plot([1, 1], [0, 1.4], "o", color="k", ms=5)
    ax2(ax, "(c) Partition:\nthree segments per point (odd);\ncutting leaves open pieces;\n"
        "the side proceeds to 4.9", (-0.2, 2.2, -0.4, 1.8))
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    poly3(ax, box_faces((0, 0, 0), (1, 1, 1)))
    poly3(ax, box_faces((1, 1, 0), (2, 2, 1)), color="#a8d5b0", edge="#2a7a3a")
    ax.plot([1, 1], [1, 1], [0, 1], color="k", lw=2.5)
    ax3(ax, elev=30, azim=-70, title="(a) In 3D: cubes touch at an edge\nwith four incident faces",
        box=((0, 0, 0), (2, 2, 1)))
    fig.text(0.5, 0.0, "In 2D: a face is a segment, an edge is its endpoint; arrows show traversal, "
             "material lies to the left", ha="center", fontsize=8.5)
    save(fig, out, "split_2d.png")


# ----------------------------------------------------------------- 4.5

def fig_eight(out):
    fig = plt.figure(figsize=(9, 3.6))
    ax = fig.add_subplot(1, 2, 1)
    fill2(ax, [(1, 1), (2, 2), (0, 2)], MAT)
    fill2(ax, [(1, 1), (2, 0), (0, 0)], NEG)
    path2(ax, [(0, 0), (2, 2), (0, 2), (2, 0)])
    ax.text(0.93, 1.6, "+1")
    ax.text(0.93, 0.3, "−1")
    ax2(ax, "Figure eight: one self-crossing contour;\nsigned volume 1 − 1 = 0, yet not empty",
        (-0.3, 2.3, -0.3, 2.3))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    top = [(0, 0), (2, 2), (0, 2), (2, 0)]
    sides = []
    for (x0, y0), (x1, y1) in zip(top, top[1:] + top[:1]):
        sides.append([(x0, y0, 0), (x1, y1, 0), (x1, y1, 1), (x0, y0, 1)])
    poly3(ax, sides, alpha=0.55)
    for z in (0, 1):
        poly3(ax, [[(1, 1, z), (2, 2, z), (0, 2, z)]], color=MAT, alpha=0.8)
        poly3(ax, [[(1, 1, z), (2, 0, z), (0, 0, z)]], color=NEG, alpha=0.8)
    ax3(ax, title="In 3D: prism over the figure eight", box=((0, 0, 0), (2, 2, 1)))
    save(fig, out, "figure_eight.png")


# ----------------------------------------------------------------- 4.6

def fig_orient(out):
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(2, 3, height_ratios=(3.4, 1.1), hspace=0.12, wspace=0.23)
    axs = [fig.add_subplot(grid[0, k]) for k in range(3)]
    ax = axs[0]
    fill2(ax, square(0, 0, 6, 6))
    fill2(ax, square(1, 1, 5, 5), "white")
    fill2(ax, square(2.3, 2.3, 3.7, 3.7))
    path2(ax, square(0, 0, 6, 6))
    path2(ax, square(1, 1, 5, 5, ccw=False), color=LOOP)
    path2(ax, square(2.3, 2.3, 3.7, 3.7))
    ax.text(0.3, 5.5, "A", color=EDGE, weight="bold")
    ax.text(1.3, 4.3, "B", color=LOOP, weight="bold")
    ax.text(2.8, 2.8, "C", color=EDGE, weight="bold")
    ax2(ax, "(a) Housing, cavity, and inner part", (-0.3, 6.3, -0.3, 6.8))
    ax = axs[1]
    fill2(ax, square(0, 0, 6, 6))
    fill2(ax, square(1, 1, 5, 5), MAT2)
    path2(ax, square(0, 0, 6, 6))
    path2(ax, square(1, 1, 5, 5))
    ax.text(0.3, 5.5, "A", color=EDGE, weight="bold")
    ax.text(2.8, 2.8, "B", color=EDGE, weight="bold")
    ax2(ax, "(b) Body inside another body", (-0.3, 6.3, -0.3, 6.8))
    ax = axs[2]
    fill2(ax, square(0, 0, 6, 6))
    fill2(ax, square(1, 1, 5, 5), "white")
    path2(ax, [(0, 0), (6, 0), (6, 6)], closed=False)
    path2(ax, [(0, 6), (6, 6)], closed=False, color=LOOP)        # one side written backwards
    path2(ax, [(0, 6), (0, 0)], closed=False)
    path2(ax, square(1, 1, 5, 5, ccw=False))
    ax.text(0.3, 5.5, "A", color=EDGE, weight="bold")
    ax.text(2.8, 2.8, "B", color=EDGE, weight="bold")
    ax.text(3, 6.25, "reversed face", color=LOOP, ha="center", fontsize=8.5)
    ax2(ax, "(c) Inconsistent winding", (-0.3, 6.3, -0.3, 6.8))
    notes = ["A: outward, body. B: inward, cavity.\nC: outward, separate body in the cavity.",
             "A and B face outward: both are bodies.\nInside B: w = 2, material counted once.",
             "Original directions are disregarded.\nDepth A = 0: body; B = 1: cavity."]
    for k, note in enumerate(notes):
        ax = fig.add_subplot(grid[1, k])
        ax.axis("off")
        ax.text(0.5, 0.9, note, ha="center", va="top", fontsize=9, transform=ax.transAxes)
    fig.text(0.5, 0.01, "Arrows show cross-section traversal. Counterclockwise: outward shell; "
             "clockwise: inward.", ha="center", fontsize=9)
    save(fig, out, "orientation_2d.png")


# ----------------------------------------------------------------- 4.8

def double2(ax, pts, color=EDGE, closed=True, gap=0.07):
    """Every segment twice, drawn apart by `gap`, the copies running opposite ways."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P, P[:1]])
    for a, b in zip(P[:-1], P[1:]):
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]]) * gap
        path2(ax, [a + n, b + n], color=color, closed=False, lw=1.1)
        path2(ax, [b - n, a - n], color=color, closed=False, lw=1.1)


def fig_doubled(out):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    ax = axs[0]
    fill2(ax, square(0, 0, 2, 2))
    path2(ax, square(0, 0, 2, 2))
    fill2(ax, square(3, 0, 5, 2), "#eeeeee")
    double2(ax, square(3, 0, 5, 2))
    ax.text(0.75, 0.9, "w = 1")
    ax.text(3.75, 0.9, "w = 0")
    ax2(ax, "One-sided versus two-sided:\neach face in both directions", (-0.2, 5.2, -0.3, 2.3))
    for ax, inner, name in [(axs[1], "white", "hollow square"), (axs[2], MAT, "square with an inner square")]:
        fill2(ax, square(0, 0, 4, 4))
        fill2(ax, square(1, 1, 3, 3), inner)
        double2(ax, square(0, 0, 4, 4))
        double2(ax, square(1, 1, 3, 3))
        ax2(ax, name + "\nsame record, different volume", (-0.2, 4.2, -0.2, 4.4))
    ax = axs[3]
    double2(ax, [(0, 2), (0, 0), (2, 0), (2, 2)], closed=False)
    path2(ax, [(2, 2), (0, 2)], color=PATCH, closed=False, ls="--", arrows=False)
    double2(ax, [(3.6, 1), (5.6, 1)], closed=False)
    ax.text(-0.2, -0.3, "cup: trial patch\n(dashed) encloses\nan area;\nrejected", fontsize=8, va="top")
    ax.text(3.6, -0.3, "sheet: nothing\nto enclose; both\ncopies discarded", fontsize=8, va="top")
    ax2(ax, "Open copy (rule 5):\ncup or sheet", (-0.3, 5.9, -1.2, 2.6))
    fig.text(0.5, 0.0, "Two-sided record: each segment appears twice, in opposite directions "
             "(copies drawn apart)", ha="center", fontsize=8.5)
    save(fig, out, "doubled_2d.png")


# ----------------------------------------------------------------- 5.3

def fig_nonorientable(out):
    fig = plt.figure(figsize=(10, 4.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    nu, nv = 40, 2
    u = np.linspace(0, 2 * np.pi, nu + 1)
    v = np.linspace(-0.35, 0.35, nv + 1)

    def p(a, b):
        return np.array([(1 + b * np.cos(a / 2)) * np.cos(a), (1 + b * np.cos(a / 2)) * np.sin(a),
                         b * np.sin(a / 2)])
    quads = [[p(u[i], v[j]), p(u[i + 1], v[j]), p(u[i + 1], v[j + 1]), p(u[i], v[j + 1])]
             for i in range(nu) for j in range(nv)]
    poly3(ax, quads, alpha=0.75, lw=0.3)
    for i in range(0, nu, 4):
        a = u[i] + 0.5 * (u[1] - u[0])
        c = p(a, 0.0)
        n = np.cross(p(a + 1e-3, 0) - p(a - 1e-3, 0), p(a, 1e-3) - p(a, -1e-3))
        n = 0.35 * n / np.linalg.norm(n)
        ax.quiver(*c, *n, color=LOOP if i == 0 else EDGE, lw=1.2, arrow_length_ratio=0.3)
    ax.plot(*np.array([p(0, b) for b in v]).T, color=LOOP, lw=3)
    ax3(ax, elev=35, azim=-60, title="Möbius strip: after one circuit,\nthe normal points the other way",
        box=((-1.3, -1.3, -0.6), (1.3, 1.3, 0.6)))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 30))
    r = 2.0
    x = (r + np.cos(a / 2) * np.sin(b) - np.sin(a / 2) * np.sin(2 * b)) * np.cos(a)
    y = (r + np.cos(a / 2) * np.sin(b) - np.sin(a / 2) * np.sin(2 * b)) * np.sin(a)
    z = np.sin(a / 2) * np.sin(b) + np.cos(a / 2) * np.sin(2 * b)
    ax.plot_surface(x, y, z, color=MAT, edgecolor=EDGE, linewidth=0.15, alpha=0.8)
    ax3(ax, elev=30, azim=-50, title="Klein bottle (closed, nonorientable):\nin 3D it always passes "
        "through itself", box=((-3, -3, -1.5), (3, 3, 1.5)))
    fig.text(0.5, 0.02, "Any face winding of a nonorientable surface has an edge "
             "(red) traversed by both incident faces in the same direction: unbalanced",
             ha="center", fontsize=8.5)
    save(fig, out, "nonorientable_3d.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures of docs/DEFINITION.md, written to docs/img/.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "img"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig_merge, fig_merge3d, fig_holes, fig_candidates, fig_bridge, fig_parts, fig_chain, fig_split,
              fig_eight, fig_orient, fig_doubled, fig_nonorientable):
        f(args.out)


if __name__ == "__main__":
    main()
