"""
The solid of a side, stored so the metric can take it without repairing it
again (since 0.4.0).

`volume_iou` rebuilds the w >= 1 solid of each side on every call: merging,
patching, orientation, the crossing detector, the union of bodies and CGAL.
For a GT scored many times, or a prediction kept for later, that work is done
once by `to_body` and kept in a `*.body.ply` file:

    binary little-endian PLY, float64 vertices, int32 triangles -- the solid
    exactly as the metric built it, in the metric's frame (the unit cube of
    the pair), and one header line `comment volumetric-iou-body {json}`.

The vertices are NOT moved back to the source's coordinates: moving them
rounds, and a rounded solid, read through the metric's cleaning, often no
longer is the same solid (merged slivers open it).  The
source's frame -- centre and scale, which `normalize_pair` needs to place a
prediction -- is kept in the metadata instead.  The file opens in any PLY
viewer (in the normalised frame); read through anything other than
`load_body`, the metadata is lost and the metric repairs it as a raw mesh.

A body is taken by the metric with cheap checks only (manifold status or
closedness, volume as recorded; the hash is checked when `load_body` reads
the file) and never cleaned.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh

from . import mesh_iou as M

FORMAT = 1
SUFFIX = ".body.ply"
_MAGIC = "comment volumetric-iou-body "
_VOLUME_TOL = 1e-12          # rebuilt volume against the recorded one, relative
_FACE = np.dtype([("n", "u1"), ("i", "<i4", (3,))])


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("volumetric-iou")
    except Exception:            # noqa: BLE001 -- run from a source tree
        return "unknown"


def _plain(x):
    """JSON-safe copy: numpy scalars and arrays to Python, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _sha256_arrays(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256(np.ascontiguousarray(vertices, "<f8").tobytes())
    h.update(np.ascontiguousarray(faces, "<i4").tobytes())
    return h.hexdigest()


@dataclass(eq=False)
class Body:
    """The metric's solid of one side in its frame, with its metadata."""

    vertices: np.ndarray            # (n, 3) float64
    faces: np.ndarray               # (m, 3) int32
    meta: dict
    _mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False)

    @property
    def frame(self) -> dict:
        """{"kind": "bbox" | "div200" | "gt", "centre": [3], "scale": float}."""
        return self.meta["frame"]

    @property
    def volume(self) -> float:
        return self.meta["volume"]

    @property
    def mesh(self) -> trimesh.Trimesh:
        if self._mesh is None:
            self._mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
        return self._mesh

    def __repr__(self) -> str:
        return (f"<Body {len(self.faces)} faces, volume {self.volume:.6g}, "
                f"frame {self.frame['kind']}, {self.meta['engine']}>")


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #


def _frame_of_gt(gt) -> tuple[np.ndarray, float]:
    """Centre and scale of the GT's own-extent frame, from a Body or a raw mesh."""
    if isinstance(gt, Body):
        if gt.frame["kind"] != "bbox":
            raise ValueError("a GT body must be in its own extent (frame 'bbox'), "
                             f"got {gt.frame['kind']!r}")
        return np.asarray(gt.frame["centre"], np.float64), float(gt.frame["scale"])
    m = M._as_mesh(gt)
    if m is None:
        raise ValueError("gt: not loadable as a mesh")
    return M._unit_box(m)


def frame_for(frame: str, mesh: trimesh.Trimesh, gt=None) -> tuple[np.ndarray, float]:
    """
    Centre and scale that `normalize_pair` would apply to `mesh` in `frame`:
    "bbox" -- its own extent (the GT's frame); "div200" -- divided by 200
    about the origin; "gt" -- the frame of `gt` (a raw mesh or a GT body).
    """
    if frame not in M.PRED_FRAMES:
        raise ValueError(f"frame must be one of {M.PRED_FRAMES}, got {frame!r}")
    if (frame == "gt") != (gt is not None):
        raise ValueError("gt is required for frame 'gt' and only for it")
    if frame == "bbox":
        return M._unit_box(mesh)
    if frame == "div200":
        return np.zeros(3), 1.0 / 200
    return _frame_of_gt(gt)


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def to_body(mesh, *, frame: str, gt=None, label: str = "mesh",
            on_self_intersection: str = "exact",
            exact_booleans: bool = False) -> Union[Body, M.IouResult]:
    """
    The solid the metric builds from `mesh` (a Trimesh, a path, a (V, F)
    pair) in the given frame, as a Body -- or the metric's refusal of it (an
    IouResult, falsy; blame is `label`, or "engine" when the engine failed).

    A GT is taken with frame="bbox".  A prediction is taken in the frame of
    its pair: frame="gt" with the GT (raw mesh or GT body), or "bbox" /
    "div200" -- the same choice as `normalize_pair(pred_frame=...)`.

    The solid is the one `volume_iou` would build for that side in the given
    `on_self_intersection` mode -- "exact" for a GT, "fail" for a prediction
    (a self-intersecting one is refused, as `volume_iou` would refuse it).
    Probe points are stored for the CGAL intersection check. For a raw
    assembly, probes check the constructed solid's representation; they
    are not an independent check of its interpretation of the source.
    This call does not fill cavities: a source that becomes unambiguous
    only after filling cannot first be converted to a Body by this call.
    `exact_booleans`: built as `volume_iou(..., exact_booleans=True)` builds
    it (in CGAL only); only such a body is taken by a call with that option.
    """
    if on_self_intersection not in M.SELF_INTERSECTION_MODES:
        raise ValueError(f"on_self_intersection must be one of {M.SELF_INTERSECTION_MODES}, "
                         f"got {on_self_intersection!r}")
    raw, err = M._load_and_check(mesh, label)
    if err is not None:
        return err
    centre, scale = frame_for(frame, raw, gt)
    prep, err = M._prepare(M._moved(raw, centre, scale), label,
                           check_self_intersection=True,
                           on_self_intersection=on_self_intersection,
                           exact_booleans=exact_booleans)
    if err is not None:
        return err

    probes = prep.probe_points()
    if len(probes):
        want = M._winding(prep.shells, probes) >= 0.5
        got = M._winding(prep.mesh, probes) >= 0.5
        if (want != got).any():
            return M.IouResult(
                M.Status.BOOLEAN_FAILED,
                detail=f"{label}: the built solid disagrees with the raw surface's "
                       f"winding at {int((want != got).sum())} of {len(probes)} probe points",
                diagnostics=prep.info, blame="engine")

    # copies: manifold's output is a read-only view, which manifold3d rejects
    vertices = np.array(prep.mesh.vertices, np.float64, order="C")
    faces = np.array(prep.mesh.faces, np.int32, order="C")
    source = {"faces_raw": len(raw.faces)}
    if isinstance(mesh, (str, os.PathLike)):
        source["name"] = Path(mesh).name
        source["sha256"] = hashlib.sha256(Path(mesh).read_bytes()).hexdigest()
    meta = {
        "format": FORMAT,
        "package": _package_version(),
        "frame": {"kind": frame, "centre": [float(c) for c in centre], "scale": float(scale)},
        "engine": "manifold" if prep.solid is not None else "cgal",
        "volume": float(prep.volume),
        "body_sha256": _sha256_arrays(vertices, faces),
        "source": source,
        "probes": _plain(np.asarray(probes, np.float64)),
        "info": _plain(prep.info),
    }
    return Body(vertices, faces, meta)


# --------------------------------------------------------------------------- #
# file
# --------------------------------------------------------------------------- #


def _header(meta: dict, n_vertices: int, n_faces: int) -> bytes:
    return ("ply\nformat binary_little_endian 1.0\n"
            f"{_MAGIC}{json.dumps(meta, separators=(',', ':'))}\n"
            f"element vertex {n_vertices}\n"
            "property double x\nproperty double y\nproperty double z\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\nend_header\n").encode("ascii")


def save_body(body: Body, path) -> Path:
    """Write `body` to `path` (atomically); returns the path."""
    path = Path(path)
    rec = np.empty(len(body.faces), _FACE)
    rec["n"], rec["i"] = 3, body.faces
    data = (_header(body.meta, len(body.vertices), len(body.faces))
            + np.ascontiguousarray(body.vertices, "<f8").tobytes() + rec.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def load_body(path) -> Body:
    """Read a body file; ValueError if it is not one, or was changed after writing."""
    data = Path(path).read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"{path}: not a PLY file")
    head = data[:end + len(b"end_header\n")].decode("ascii", "replace")
    lines = head.split("\n")
    if len(lines) < 3 or not lines[2].startswith(_MAGIC):
        raise ValueError(f"{path}: not a volumetric-iou body")
    meta = json.loads(lines[2][len(_MAGIC):])
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: body format {meta.get('format')!r}, this reader knows {FORMAT}")
    try:
        nv, nf = int(lines[3].split()[2]), int(lines[7].split()[2])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: unexpected PLY header") from None
    if head.encode("ascii") != _header(meta, nv, nf):
        raise ValueError(f"{path}: unexpected PLY header")
    body = data[len(head):]
    if len(body) != nv * 24 + nf * _FACE.itemsize:
        raise ValueError(f"{path}: {len(body)} data bytes, header says "
                         f"{nv * 24 + nf * _FACE.itemsize}")
    vertices = np.frombuffer(body, "<f8", nv * 3).reshape(nv, 3).astype(np.float64)
    rec = np.frombuffer(body, _FACE, nf, offset=nv * 24)
    if (rec["n"] != 3).any():
        raise ValueError(f"{path}: non-triangular faces")
    faces = rec["i"].astype(np.int32)
    if _sha256_arrays(vertices, faces) != meta.get("body_sha256"):
        raise ValueError(f"{path}: vertices or faces changed after the body was written")
    return Body(vertices, faces, meta)


def is_body_path(obj) -> bool:
    return isinstance(obj, (str, os.PathLike)) and str(obj).endswith(SUFFIX)


# --------------------------------------------------------------------------- #
# the metric's side
# --------------------------------------------------------------------------- #


def prepared(body: Body, label: str, on_self_intersection: str, exact_booleans: bool = False
             ) -> tuple[Optional["M._Prepared"], Optional[M.IouResult]]:
    """
    A body as the metric's prepared side, after cheap checks only: finite
    coordinates, then manifold status (engine "manifold") or no open edge
    (engine "cgal"), and the rebuilt volume against the recorded one.
    `exact_booleans`: only a body built so (`to_body(exact_booleans=True)`)
    is taken -- another one may hold a union made in manifold.
    """
    meta = body.meta
    info = {**meta["info"], "body": {"format": meta["format"], "package": meta["package"],
                                     "engine": meta["engine"], "source": meta.get("source")}}
    if exact_booleans and not info.get("exact_booleans"):
        return None, M.IouResult(
            M.Status.LOAD_FAILED,
            detail=f"{label}: body was built without exact_booleans; this call needs one built "
                   "with it (to_body(..., exact_booleans=True))",
            diagnostics=info, blame=label)
    raw = info.get("raw_solid") or {}
    raw_fail = (raw.get("assembly") or {}).get("fail_validity")
    if on_self_intersection != "exact" and raw_fail in ("ambiguous", "self_intersecting"):
        status = M.Status.DEGENERATE_VOLUME if raw_fail == "ambiguous" else M.Status.SELF_INTERSECTING
        detail = ("raw-source readings include both valid and defective bodies in fail mode"
                  if raw_fail == "ambiguous" else "every raw-source reading has a self-overlapping body")
        return None, M.IouResult(status, detail=f"{label}: {detail}", diagnostics=info, blame=label)
    resolved = info.get("resolved") or {}
    if (on_self_intersection != "exact" and raw_fail != "valid"
            and info.get("crossing_pairs_internal") and resolved.get("self_overlap_bodies", 1)):
        # the raw surface this body came from is refused in "fail" mode: a body
        # passing through itself (w >= 2) -- or, in a file that does not say
        # (written before that was looked for), any crossing inside a body
        detail = (M.through_detail(label, resolved["self_overlap_volume"])
                  if "self_overlap_bodies" in resolved else M.crossing_detail(label))
        return None, M.IouResult(M.Status.SELF_INTERSECTING, detail=detail,
                                 diagnostics=info, blame=label)
    if on_self_intersection != "exact" and raw and raw_fail != "valid" and not (
            raw.get("winding_max", 2) < 2 or raw.get("self_overlap_parts") == 0):
        # built from the raw surface after cleaning refused it: "fail" takes
        # it where the raw surface winds below 2, or 2 or more only where
        # different parts overlap -- found in "fail" only: a body built in
        # "exact" (no winding_max) stays refused here
        return None, M.IouResult(M.Status(raw["refused_as"]), detail=raw["refused_detail"],
                                 diagnostics=info, blame=label)
    if on_self_intersection != "exact" and info.get("self_intersection_checked") is None:
        return None, M.IouResult(
            M.Status.BOOLEAN_FAILED,
            detail=f"{label}: self-intersection check failed on the body's source "
                   f"({info.get('detector_error', '')})",
            diagnostics=info, blame="engine")

    def bad(why: str):
        return None, M.IouResult(M.Status.LOAD_FAILED, detail=f"{label}: body {why}",
                                 diagnostics=info, blame=label)

    if not np.isfinite(body.vertices).all():
        return bad("has NaN/inf vertices")
    mesh = body.mesh
    if meta["engine"] == "manifold":
        if not M._HAS_ENGINE:
            return None, M.IouResult(M.Status.NO_ENGINE,
                                     detail="no boolean engine available (pip install manifold3d)",
                                     diagnostics=info, blame="engine")
        solid = M._to_manifold([mesh])
        if solid.status() != M.manifold3d.Error.NoError:
            return bad(f"is not a valid manifold ({solid.status()})")
        volume = float(solid.volume())
    else:
        solid = None
        if M._boundary_edges(mesh):
            return bad("has open edges")
        volume = float(mesh.volume)
    if abs(volume - body.volume) > _VOLUME_TOL * abs(body.volume):
        return bad(f"volume {volume!r} differs from the recorded {body.volume!r}")
    if "cavities_solid" not in info:            # a body built before v0.7.1
        info["cavities_solid"] = M._solid_cavities(mesh)
    probes = np.asarray(meta["probes"], np.float64).reshape(-1, 3)
    return M._Prepared(mesh, solid, body.volume, info, shells=mesh, probes=probes), None
