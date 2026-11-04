"""Exact raw readings: assembly cavities, hidden pockets and ambiguity."""
import numpy as np
import pytest
import trimesh

from volumetric_iou import mesh_iou as M
from volumetric_iou.raw_solid import partitions, resolve, InvalidReading
from volumetric_iou.winding import winding_number
from test_side_rules import bowtie, touching_halves


def box(lo, hi, inward=False):
    m = trimesh.creation.box(bounds=[lo, hi])
    if inward:
        m.invert()
    return m


def hollow_halves():
    import manifold3d as mf
    result = []
    for lo, hi, c0, c1 in (([-.5]*3, [0, .5, .5], [-.25]*3, [.05, .25, .25]),
                            ([0, -.5, -.5], [.5]*3, [-.05, -.25, -.25], [.25]*3)):
        a, b = box(lo, hi), box(c0, c1)
        def solid(m):
            return mf.Manifold(mf.Mesh64(np.asarray(m.vertices), np.asarray(m.faces, np.uint64)))
        p = (solid(a) - solid(b)).to_mesh64()
        result.append(trimesh.Trimesh(p.vert_properties[:, :3], p.tri_verts, process=False))
    return result


def side(parts, trigger=False, stl=False, tmp_path=None):
    if trigger:
        extra = list(touching_halves())
        for m in extra:
            m.apply_translation([10, 0, 0])
        parts = parts + extra
    result = trimesh.util.concatenate(parts)
    result.apply_scale(.04)    # common frame, including the distant trigger
    if stl:
        path = tmp_path / "raw.stl"
        result.export(path)
        result = trimesh.load(path, process=False)
    return result


def prepared(m, mode, fill):
    p, err = M._prepare(m, "gt" if mode == "exact" else "pred",
                        check_self_intersection=True, on_self_intersection=mode,
                        fill_cavities=fill, exact_booleans=True)
    if p is not None and fill:
        p, err = M._fill_cavities(p, "gt", exact=True)
    return p, err


@pytest.mark.parametrize("shape", ["notched_halves", "common_cavity"])
@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
def test_hollow_cube_from_two_halves(shape, trigger, mode, fill, stl, inverted, tmp_path):
    parts = hollow_halves() if shape == "notched_halves" else [*touching_halves(),
                                                               box([-.25]*3, [.25]*3, True)]
    m = side(parts, trigger, stl, tmp_path)
    if inverted:
        m.invert()
    p, err = prepared(m, mode, fill)
    assert err is None, err
    assert p.volume / .04**3 == pytest.approx((1 if fill else .875) + trigger, rel=2e-7)
    # Independent witnesses in the cavity and the surrounding material.
    points = .04 * np.array([[0, 0, 0], [.4, 0, 0], [2, 0, 0]])
    assert (winding_number(p.mesh, points) > .5).tolist() == [fill, True, False]
    public = M.volume_iou(m, m, on_self_intersection=mode,
                          fill_cavities=fill, exact_booleans=True)
    assert public.status is M.Status.OK and public.iou == pytest.approx(1.0)


@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
def test_ambiguous_minimal_assembly_owners(trigger, mode, fill):
    parts = [box([0, 0, 0], [2, 1, 1]), box([1, 0, 0], [4, 1, 1]),
             box([1.5, 0, 0], [4, 1, 1]), box([.5, .25, .25], [3, .75, .75], True)]
    p, err = prepared(side(parts, trigger), mode, fill)
    if fill and mode == "exact":
        assert err is None, err
        assert p.volume / .04**3 == pytest.approx(4 + trigger)
    else:
        assert p is None and err.status is M.Status.DEGENERATE_VOLUME
        assert "different material" in err.detail or "both valid and defective" in err.detail


@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("stl", [False, True])
def test_inside_out_overlapping_assembly_keeps_its_cavity(trigger, mode, fill, stl, tmp_path):
    parts = [box([0, 0, 0], [2, 1, 1]), box([1.5, 0, 0], [4, 1, 1]),
             box([1, .25, .25], [3, .75, .75], True)]
    mesh = side(parts, trigger, stl, tmp_path)
    mesh.invert()
    p, err = prepared(mesh, mode, fill)
    assert err is None, err
    assert p.volume / .04**3 == pytest.approx((4 if fill else 3.5) + trigger, rel=2e-7)
    points = .04 * np.array([[2, .5, .5], [.5, .5, .5], [5, .5, .5]])
    assert (winding_number(p.mesh, points) > .5).tolist() == [fill, True, False]


@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("open_pocket", [False, True])
def test_hidden_bowtie_cannot_be_masked_by_a_distant_trigger(trigger, mode, fill, open_pocket):
    lo = [.2, -.2, -.2] if open_pocket else [-.2]*3
    p, err = prepared(side([box(lo, [.6]*3), bowtie()], trigger), mode, fill)
    if fill and not open_pocket:
        assert err is None, err
        assert p.volume / .04**3 == pytest.approx(.512 + trigger)
    else:
        assert p is None and err.status is M.Status.INVERTED_VOLUME


@pytest.mark.parametrize("fill", [False, True])
def test_assembly_cavity_is_subtracted_from_the_union(fill):
    parts = [box([0, 0, 0], [2, 1, 1]), box([1.5, 0, 0], [4, 1, 1]),
             box([1, .25, .25], [3, .75, .75], True)]
    m, volume, info = resolve(trimesh.util.concatenate(parts), fill=fill)
    assert volume == pytest.approx(4 if fill else 3.5)
    assert info["assembly_owners"] == 1


def test_search_limit_is_explicit_and_does_not_choose_a_reading(monkeypatch):
    import volumetric_iou.raw_solid as R
    m = trimesh.util.concatenate(touching_halves())
    with pytest.raises(RuntimeError, match="resource limit"):
        list(partitions(m.vertices, m.faces, max_states=1))
    monkeypatch.setattr(R, "MAX_SEARCH", 1)
    p, err = prepared(m, "exact", False)
    assert p is None and err.status is M.Status.BOOLEAN_FAILED and err.blame == "engine"


def test_completed_winding_fields_avoid_the_full_partition_limit():
    # Two half-cubes share a cap triangulated into a fan. Many topological
    # partitions differ only in completed zero-volume surface pieces.
    def prism(x0, x1, n=9):
        polygon = [[-.5, -.5], [.5, -.5], [.5, .5]] + [
            [-.5 + (n-i)/n, .5] for i in range(1, n+1)]
        count = len(polygon)
        vertices = [[x, y, z] for x in (x0, x1) for y, z in polygon]
        faces = []
        for i in range(1, count-1):
            faces.extend([[0, i+1, i], [count, count+i, count+i+1]])
        for i in range(count):
            j = (i+1) % count
            faces.extend([[i, j, count+j], [i, count+j, count+i]])
        return trimesh.Trimesh(vertices, faces, process=False)

    m = trimesh.util.concatenate([prism(-.5, 0), prism(0, .5)])
    with pytest.raises(RuntimeError, match="resource limit"):
        list(partitions(m.vertices, m.faces))
    from volumetric_iou import _cgal
    from volumetric_iou.raw_solid import Search
    from volumetric_iou.raw_topology import readings
    arena = _cgal.RawArrangement(m.vertices, m.faces)
    _, families, _, topology = readings(m.vertices, m.faces, arena, Search())
    assert len(families) < 4096 and topology["frontier_peak_states"] <= 4096
    result, volume, info = resolve(m)
    assert volume == pytest.approx(1)
    assert result.is_watertight
    assert info["raw_reading_certificate"]["kind"] == "nonnegative_unit_components"


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("shape", ["contact", "overlap", "double_tetra", "triple_tetra",
                                    "open_pocket", "enclosed_pocket", "bowtie_plug", "cavity"])
def test_reading_certificates_match_exhaustive_verdict_and_material(monkeypatch, shape, check, fill):
    import volumetric_iou.raw_solid as R
    from volumetric_iou import _cgal
    from test_side_rules import lobe
    tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0],
                                                [0, 1, 0], [.6, .1, 1]]))
    inward = tetra.copy()
    inward.invert()
    shapes = {
        "contact": list(touching_halves()),
        "overlap": [box([0]*3, [1]*3), box([.5, 0, 0], [1.5, 1, 1])],
        "double_tetra": [tetra, tetra.copy()],
        "triple_tetra": [tetra, tetra.copy(), inward],
        "open_pocket": [bowtie()],
        "enclosed_pocket": [box([-.2]*3, [.6]*3), bowtie()],
        "bowtie_plug": [bowtie(), lobe()],
        "cavity": [box([-.5]*3, [.5]*3), box([-.2]*3, [.2]*3, True)],
    }
    mesh = trimesh.util.concatenate(shapes[shape])

    def outcome():
        try:
            return R.resolve(mesh, check=check, fill=fill)
        except R.InvalidReading as exc:
            return exc.kind

    actual = outcome()
    monkeypatch.setattr(R, "_certified_masks", lambda *args: None)
    expected = outcome()
    if isinstance(expected, str):
        assert actual == expected
    else:
        assert not isinstance(actual, str)
        a, va, ia = actual
        b, vb, ib = expected
        assert va == pytest.approx(vb, rel=1e-13, abs=1e-15)
        assert ia["fail_validity"] == ib["fail_validity"]
        if fill:
            assert ia["volume_cavities"] == pytest.approx(ib["volume_cavities"], abs=1e-15)
        # Exact equality of spatial material, independent of output triangulation
        # and artificial zero-volume cells between coincident copies.
        b = b.copy()
        b.invert()
        difference = trimesh.util.concatenate([a, b])
        arena = _cgal.RawArrangement(difference.vertices, difference.faces)
        total = arena.winding([0]*len(difference.faces))[:, 0]
        assert np.all(total[np.asarray(arena.live, bool)] == 0)


@pytest.mark.parametrize("seed", [0, 7, 23])
@pytest.mark.parametrize("shape", ["contact_cavity", "double_tetra", "bowtie_plug"])
def test_frontier_preserves_full_fields_multiplicity_and_regular_flags(seed, shape):
    from volumetric_iou import _cgal
    from volumetric_iou.raw_solid import Search
    from volumetric_iou.raw_topology import readings
    if shape == "contact_cavity":
        m = trimesh.util.concatenate([*touching_halves(), box([-.25]*3, [.25]*3, True)])
    elif shape == "double_tetra":
        tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0], [.6, .1, 1]]))
        m = trimesh.util.concatenate([tetra, tetra.copy()])
    else:
        from test_side_rules import lobe
        m = trimesh.util.concatenate([bowtie(), lobe()])
    # Face order changes both crowded-edge insertion order and block IDs.
    # The independent full-partition oracle below must still yield exactly
    # the same complete fields, multiplicities and regular-mixed flags.
    permutation = np.random.default_rng(seed).permutation(len(m.faces))
    m = trimesh.Trimesh(m.vertices, m.faces[permutation], process=False)
    arena = _cgal.RawArrangement(m.vertices, m.faces)
    fields, families, regular, _ = readings(m.vertices, m.faces, arena, Search())
    actual = {tuple(sorted((fields[k].tobytes(), regular[k]) for k in family)) for family in families}
    expected = set()
    live = np.asarray(arena.live, bool)
    _, indices = np.unique(m.vertices, axis=0, return_inverse=True)
    triangles = indices[m.faces]
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, edge_id = np.unique(edges, axis=0, return_inverse=True)
    for labels in partitions(m.vertices, m.faces):
        W = arena.winding(labels.tolist())
        family = []
        for k in range(W.shape[1]):
            if not np.any(W[live, k]):
                continue
            counts = np.bincount(edge_id.reshape(-1, 3)[labels == k].ravel())
            mixed = np.any(W[live, k] > 0) and np.any(W[live, k] < 0)
            flag = bool(mixed and np.all(counts[counts != 0] == 2))
            family.append((W[:, k].astype(np.int32).tobytes(), flag))
        expected.add(tuple(sorted(family)))
    assert actual == expected


def test_native_exact_masks_ignore_zero_volume_contact_cells():
    from volumetric_iou import _cgal
    m = trimesh.util.concatenate([*touching_halves(), box([-.25]*3, [.25]*3, True)])
    a = _cgal.RawArrangement(m.vertices, m.faces)
    w = a.winding([0]*len(m.faces))[:, 0]
    mask = (w >= 1).astype(int).tolist()
    assert a.volume(mask) == pytest.approx(.875)
    assert a.volume(a.filled(mask)) == pytest.approx(1.0)


def test_open_block_formal_potentials_are_additive_on_zero_volume_cells():
    from volumetric_iou import _cgal
    tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0],
                                                [0, 1, 0], [.6, .1, 1]]))
    m = trimesh.util.concatenate([tetra, tetra.copy()])
    # Every edge has four faces: each individual (open) triangle is a fixed
    # block. Coincident sheets produce zero-volume cells with different fields.
    arena = _cgal.RawArrangement(m.vertices, m.faces)
    live = np.asarray(arena.live, bool)
    assert np.any(~live[1:])
    basis = np.column_stack([arena.winding((np.arange(len(m.faces)) != k).astype(int).tolist())[:, 0]
                             for k in range(len(m.faces))]).astype(np.int64)
    assert np.any(basis[~live])
    # Check every subset, including open partial groups. Native propagation
    # chooses the same infinity cell and BFS tree regardless of labels.
    for bits in range(1 << len(m.faces)):
        member = np.array([bool(bits & (1 << k)) for k in range(len(m.faces))])
        expected = arena.winding((~member).astype(int).tolist())[:, 0]
        assert np.array_equal(basis[:, member].sum(axis=1, dtype=np.int64), expected)


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("shape", ["negative_block", "double_cover"])
def test_impossible_success_certificate_stops_before_expensive_tail(monkeypatch, shape, fill):
    from volumetric_iou import _cgal, raw_solid as R
    from volumetric_iou.raw_topology import fixed_topology
    from test_side_rules import lobe
    if shape == "negative_block":
        mesh = trimesh.util.concatenate([bowtie(), lobe()])
    else:
        tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0],
                                                    [0, 1, 0], [.6, .1, 1]]))
        mesh = trimesh.util.concatenate([tetra, tetra.copy()])
    arena = _cgal.RawArrangement(mesh.vertices, mesh.faces)
    live = np.asarray(arena.live, bool)
    raw_sum = np.asarray(arena.winding([0]*len(mesh.faces)))[:, 0]
    _, _, fixed = fixed_topology(mesh.vertices, mesh.faces)
    n_fixed = int(fixed.max())+1

    class MeteredArena:
        calls = 0

        def winding(self, labels):
            self.calls += 1
            # A backend that cannot afford the complete extra basis pass.
            # An impossible sufficient certificate must still leave enough
            # work for the unchanged exhaustive resolver.
            if self.calls >= n_fixed:
                raise RuntimeError("unnecessary certificate tail exhausted backend budget")
            return arena.winding(labels)

    metered = MeteredArena()
    assert R._certified_masks(mesh.vertices, mesh.faces, metered, live, raw_sum, fill, R.Search()) is None
    assert 0 < metered.calls < n_fixed

    def outcome():
        try:
            _, volume, info = R.resolve(mesh, check=True, fill=fill)
            return volume, info["fail_validity"]
        except R.InvalidReading as exc:
            return exc.kind

    actual = outcome()
    monkeypatch.setattr(R, "_certified_masks", lambda *args: None)
    assert actual == outcome() == "ambiguous"


def test_filled_volume_is_not_a_difference_of_rounded_totals():
    tiny = 1e-7
    m = trimesh.util.concatenate([box([-.5]*3, [.5]*3),
                                  box([-tiny/2]*3, [tiny/2]*3, True)])
    _, volume, info = resolve(m, fill=True)
    assert volume == 1.0
    assert info["volume_cavities"] == pytest.approx(tiny**3, rel=1e-12, abs=0)


def test_crossing_assembly_owners_are_independent_of_cavity_order():
    from itertools import permutations
    from volumetric_iou.raw_solid import material
    W = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
    fields = [np.array([0, -1, -1, 0, 0, 0, 0, 0]),
              np.array([0, 0, 0, -1, -1, 0, 0, 0]),
              np.array([0, 0, 0, 0, 0, -1, -1, -1])]
    owners = [frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])]
    results = []
    for order in permutations(range(3)):
        m, pocket, overlap, count = material(W, [0, 1, 2], [fields[k] for k in order],
                                             [owners[k] for k in order])
        assert count == 1 and not pocket.any() and not m[-1]
        results.append(m)
    assert all(np.array_equal(results[0], m) for m in results)


@pytest.mark.parametrize("fill", [False, True])
def test_pocket_connected_to_outside_by_an_assembly_cavity_is_not_filled(fill):
    a, b = box([0]*3, [2]*3), box([1, 0, 0], [3, 2, 2])
    c = trimesh.convex.convex_hull(np.array([[1.5, 0, 1], [.5, 1, .5], [2.5, 1, .5],
                                            [2.5, 1, 1.5], [.5, 1, 1.5]]))
    c.invert()
    d = box([.5, .2, .98], [2.5, .3, 1.02], True)
    with pytest.raises(InvalidReading) as refused:
        resolve(trimesh.util.concatenate([a, b, c, d]), fill=fill)
    assert refused.value.kind == "pocket"


@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("fill", [False, True])
def test_multiple_single_hosts_use_the_same_rule_on_both_routes(trigger, fill):
    parts = [box([0]*3, [2]*3), box([.2, 0, 0], [2.2, 2, 2]),
             box([.5]*3, [1.3]*3, True), box([.9, .5, .5], [1.7, 1.3, 1.3], True)]
    p, err = prepared(side(parts, trigger), "exact", fill)
    if fill:
        assert err is None, err
        assert p.volume / .04**3 == pytest.approx(8.8 + trigger)
    else:
        # Assigning both overlapping cavities to one host is defective;
        # distributing them between hosts is valid. Equal material among
        # the valid choices does not remove validity ambiguity.
        assert p is None and err.status is M.Status.DEGENERATE_VOLUME
        assert "both valid and defective" in err.detail


@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
def test_raw_audit_preserves_small_cavity_dropped_only_by_cleaned_path(mode, fill):
    tiny = 1e-3
    mesh = trimesh.util.concatenate([box([0]*3, [1]*3),
                                     box([.2]*3, [.4]*3, True),
                                     box([.7]*3, [.7 + tiny]*3, True)])
    mesh.apply_translation([-.5]*3)
    ordinary, error = M._prepare_shells(
        mesh, "gt", check_self_intersection=True, on_self_intersection=mode,
        fill_cavities=fill, exact=True)
    assert error is None, error
    assert ordinary.info["shells_dropped"] == 1
    assert ordinary.volume == pytest.approx(1 - .2**3, abs=1e-14, rel=0)
    audited, error = prepared(mesh, mode, fill)
    assert error is None, error
    assert audited.volume == pytest.approx(1 if fill else 1 - .2**3 - tiny**3,
                                           abs=1e-14, rel=0)
    assert "shells_dropped" not in audited.info
    assert audited.info["raw_solid"]["cleaned_shells_dropped"] == 1
    public = M.volume_iou(mesh, mesh, on_self_intersection=mode,
                          fill_cavities=fill, exact_booleans=True)
    assert public.status is M.Status.OK
    assert not any("zero-volume shells dropped" in warning for warning in public.warnings)


def test_convex_interface_certificate_rejects_an_inner_cavity():
    from volumetric_iou import _cgal
    plain = trimesh.util.concatenate(touching_halves())
    assert _cgal.convex_interface(plain.vertices, plain.faces)
    hollow = trimesh.util.concatenate([plain, box([-.25]*3, [.25]*3, True)])
    assert not _cgal.convex_interface(hollow.vertices, hollow.faces)


def test_triple_hull_certificate_keeps_strictly_valid_readings():
    from volumetric_iou import _cgal
    a = box([0]*3, [1]*3)
    triple = trimesh.util.concatenate([a, a.copy(), box([0]*3, [1]*3, True)])
    assert _cgal.convex_interface(triple.vertices, triple.faces)
    for check in (False, True):
        _, volume, info = resolve(triple, check=check)
        assert volume == pytest.approx(1) and info["certified_interfaces"] == 1
    double = trimesh.util.concatenate([a, a.copy()])
    assert not _cgal.convex_interface(double.vertices, double.faces)
    # A negative portion can be bounded by inward hull faces plus an internal
    # interface: that combination has no triple-hull certificate.
    contact = trimesh.util.concatenate(touching_halves())
    mixed = trimesh.util.concatenate([contact, box([-.5]*3, [.5]*3),
                                      box([-.5]*3, [.5]*3, True)])
    assert not _cgal.convex_interface(mixed.vertices, mixed.faces)


def test_local_convex_certificate_does_not_hide_an_external_assembly_owner():
    m = trimesh.util.concatenate([*touching_halves(), box([-.25]*3, [.25]*3, True),
                                  box([-.1, -.3, -.3], [.7, .3, .3])])
    with pytest.raises(InvalidReading) as refused:
        resolve(m)
    assert refused.value.kind == "ambiguous"


def test_native_arrangement_reports_bad_indices_and_masks():
    from volumetric_iou import _cgal
    m = box([0]*3, [1]*3)
    bad = m.faces.copy()
    bad[0, 0] = len(m.vertices)
    with pytest.raises(IndexError):
        _cgal.RawArrangement(m.vertices, bad)
    a = _cgal.RawArrangement(m.vertices, m.faces)
    with pytest.raises(ValueError):
        a.winding([0])
    with pytest.raises(ValueError):
        a.region([1, 1])


def test_collapsed_raw_faces_add_no_material_or_readings():
    collapsed = trimesh.Trimesh([[2, 0, 0], [2, 0, 0], [3, 0, 0]], [[0, 1, 2]], process=False)
    source = trimesh.util.concatenate([*touching_halves(), collapsed])
    _, volume, _ = resolve(source, check=True)
    assert volume == pytest.approx(1)


def test_regular_cavity_body_keeps_its_checked_fail_contract(tmp_path):
    from volumetric_iou.body import Body, to_body, save_body, load_body
    m = trimesh.util.concatenate([box([-.4]*3, [.4]*3), box([-.2]*3, [.2]*3, True)])
    b = to_body(m, frame="bbox", on_self_intersection="exact")
    assert isinstance(b, Body)
    save_body(b, tmp_path / "cavity.body.ply")
    loaded = load_body(tmp_path / "cavity.body.ply")
    result = M.volume_iou(loaded, loaded, on_self_intersection="fail")
    assert result.status is M.Status.OK and result.iou == pytest.approx(1.0)


@pytest.mark.parametrize("exact", [False, True])
def test_body_roundtrip_keeps_raw_validity_ambiguity_in_fail(tmp_path, exact):
    from volumetric_iou.body import Body, to_body, save_body, load_body
    tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.6, .1, 1]]))
    source = trimesh.util.concatenate([tetra, tetra.copy()])
    b = to_body(source, frame="bbox", on_self_intersection="exact", exact_booleans=exact)
    assert isinstance(b, Body)
    path = tmp_path / "ambiguous.body.ply"
    save_body(b, path)
    for candidate in (b, load_body(path)):
        accepted = M.volume_iou(candidate, candidate, on_self_intersection="exact", exact_booleans=exact)
        assert accepted.status is M.Status.OK and accepted.iou == pytest.approx(1)
        refused = M.volume_iou(candidate, candidate, on_self_intersection="fail", exact_booleans=exact)
        assert refused.status is M.Status.DEGENERATE_VOLUME
        assert "both valid and defective" in refused.detail


@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("fill", [False, True])
def test_bowtie_and_plug_have_validity_ambiguity_even_with_filling(trigger, fill):
    from test_side_rules import lobe
    p, err = prepared(side([bowtie(), lobe()], trigger), "exact", fill)
    assert p is None and err.status is M.Status.DEGENERATE_VOLUME
    assert err.diagnostics["valid_readings"] > 0 and err.diagnostics["invalid_readings"] > 0


@pytest.mark.parametrize("mode", ["exact", "fail"])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("trigger", [False, True])
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_coincident_positive_bodies_are_ambiguous_only_in_fail(
        mode, fill, trigger, inverted, stl, exact, tmp_path):
    tetra = trimesh.convex.convex_hull(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.6, .1, 1]]))
    mesh = side([tetra, tetra.copy()], trigger, stl, tmp_path)
    if inverted:
        mesh.invert()
    p, err = M._prepare(mesh, "gt" if mode == "exact" else "pred",
                         check_self_intersection=True, on_self_intersection=mode,
                         fill_cavities=fill, exact_booleans=exact)
    if mode == "exact":
        assert err is None, err
        assert p.volume / .04**3 == pytest.approx(1/6 + trigger)
    else:
        assert p is None and err.status is M.Status.DEGENERATE_VOLUME
        assert "both valid and defective" in err.detail


@pytest.mark.parametrize("open_point", [False, True])
@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("trigger", [False, True])
def test_cavity_and_insert_with_shared_base_use_strict_validity(open_point, fill, trigger):
    base = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    insert = trimesh.convex.convex_hull(np.array(base + [[.6, .1, 1]]))
    cavity = trimesh.convex.convex_hull(np.array(base + [[.2, .2, 1]]))
    cavity.invert()
    host = box([-.2]*3, [1.2, 1.2, 1 if open_point else 1.2])
    mesh = side([host, insert, cavity], trigger)
    p, err = prepared(mesh, "exact", fill)
    if fill and not open_point:
        assert err is None, err
        assert p.volume / .04**3 == pytest.approx(1.4**3 + trigger)
    else:
        assert p is None and err.status is M.Status.DEGENERATE_VOLUME
        assert "both valid and defective" in err.detail
        public = M.volume_iou(mesh, mesh, fill_cavities=fill, on_self_intersection="exact")
        assert public.status is M.Status.DEGENERATE_VOLUME
