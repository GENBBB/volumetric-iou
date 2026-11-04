"""Oriented raw surfaces, assembly cavities and exact ambiguity checks.

Interpretations preserve every input triangle and every ordinary two-face
edge. Only the pairing of opposite half-edges at crowded edges may change.
Each resulting negative sheet belongs to a minimal containing positive sheet
or assembly. Crossing assembly owners are merged; already nested owners keep
their hierarchy. Every interpretation must be valid and give the same
material, after filling when requested. Mixed validity is ambiguity.
A search limit is an engine failure, never a
choice of the first interpretation. See docs/DEFINITION.md 4.9.
"""
from collections import Counter
from itertools import product

import numpy as np

MAX_STATES = 4096
MAX_SEARCH = 100000
MAX_INPUT_FACES = 500000


class InvalidReading(Exception):
    """No valid reading or different material under valid readings."""
    def __init__(self, kind, detail, diagnostics=None):
        super().__init__(detail)
        self.kind = kind
        self.diagnostics = diagnostics or {}


def partitions(vertices, faces, max_states=MAX_STATES, certified=()):
    """All face partitions induced by orientation-preserving edge pairings."""
    if len(faces) > MAX_INPUT_FACES:
        raise RuntimeError("raw topology input exceeded its face resource limit")
    points = np.ascontiguousarray(np.asarray(vertices, np.float64) + 0.0)
    _, inv = np.unique(points.view(np.dtype((np.void, 24))).ravel(), return_inverse=True)
    merged = inv.reshape(-1)[faces]
    edges = {}
    for face, tri in enumerate(merged):
        for a, b in zip(tri, np.roll(tri, -1)):
            if a == b:
                raise ValueError("raw surface has a collapsed face")
            edges.setdefault((min(int(a), int(b)), max(int(a), int(b))), ([], []))[int(a > b)].append(face)
    if any(len(a) != len(b) for a, b in edges.values()):
        raise ValueError("raw surface has unbalanced edges")
    root = list(range(len(faces)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b in edges.values():
        if len(a) == 1:
            x, y = find(a[0]), find(b[0])
            root[max(x, y)] = min(x, y)
    for group in certified:
        for face in group[1:]:
            x, y = find(int(group[0])), find(int(face))
            root[max(x, y)] = min(x, y)
    roots = [find(i) for i in range(len(faces))]
    rank = {r: k for k, r in enumerate(sorted(set(roots)))}
    fixed = np.asarray([rank[r] for r in roots], np.int64)

    def join(state, a, b):
        a, b = state[a], state[b]
        lo, hi = min(a, b), max(a, b)
        return tuple(lo if k == hi else k for k in state)

    def limit(n, width):
        if n > max_states or n * max(width, 1) * 8 > 512 * 1024**2:
            raise RuntimeError("raw shell enumeration exceeded its resource limit")

    states = {tuple(range(len(rank)))}
    for a, b in edges.values():
        if len(a) == 1:
            continue
        forward, backward = fixed[a], fixed[b]
        next_states = set()
        for state in states:
            counts = tuple(sorted(Counter(state[k] for k in backward).items()))
            work = {(state, counts)}
            for component in forward:
                next_work = set()
                for current, counts in work:
                    for other, _ in counts:
                        joined = join(current, int(component), other)
                        remaining = Counter()
                        for k, count in counts:
                            count -= k == other
                            if count:
                                remaining[joined[k]] += count
                        next_work.add((joined, tuple(sorted(remaining.items()))))
                        limit(len(next_work), len(rank))
                work = next_work
            next_states.update(s for s, remaining in work)
            limit(len(next_states), len(rank))
        states = next_states
    limit(len(states), len(faces))
    for state in sorted(states):
        _, labels = np.unique(np.asarray(state, np.int64)[fixed], return_inverse=True)
        yield labels.reshape(-1).astype(np.int64)


class Search:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps > MAX_SEARCH:
            raise RuntimeError("raw assembly enumeration exceeded its resource limit")


def covers(positive, cavity, live, search):
    """Minimal covering owners, by exact cell membership, not sample points."""
    hosts = [k for k, p in enumerate(positive) if not np.any(cavity & ~p & live)]
    if hosts:
        # Prefer a genuinely smaller enclosing shell to an outer containing one.
        return [frozenset([k]) for k in hosts if not any(
            h != k and np.any(positive[k] & ~positive[h] & live)
            and not np.any(positive[h] & ~positive[k] & live) for h in hosts)]
    candidates = [k for k, p in enumerate(positive) if np.any(cavity & p & live)]
    found, seen = set(), set()

    def walk(chosen, covered):
        search.step()
        if chosen in seen or any(done <= chosen for done in found):
            return
        seen.add(chosen)
        missing = np.flatnonzero(cavity & ~covered & live)
        if not len(missing):
            found.difference_update([done for done in found if chosen < done])
            found.add(chosen)
            return
        options = min(([k for k in candidates if positive[k][c]] for c in missing), key=len)
        for k in options:
            walk(chosen | {k}, covered | positive[k])

    walk(frozenset(), np.zeros(len(live), bool))
    return sorted(found, key=lambda s: (len(s), tuple(sorted(s))))


def material(W, positive, negative, ownership):
    """Evaluate the laminar assembly hierarchy induced by chosen minimal covers."""
    owners = list(ownership)
    while True:
        parent = list(range(len(owners)))
        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        changed = False
        for i, a in enumerate(owners):
            for j, b in enumerate(owners[:i]):
                if a & b and not (a <= b or b <= a):
                    x, y = find(i), find(j)
                    parent[x] = y
                    changed = True
        if not changed:
            break
        merged = {}
        for i, owner in enumerate(owners):
            merged[find(i)] = merged.get(find(i), frozenset()) | owner
        owners = [merged[find(i)] for i in range(len(owners))]
    groups = set(owners) | {frozenset([k]) for k in range(len(positive))}
    root = frozenset(range(len(positive)))
    groups.add(root)
    values, pocket, through = {}, np.zeros(len(W), bool), np.zeros(len(W), bool)
    for group in sorted(groups, key=len):
        if len(group) == 1:
            field = W[:, positive[next(iter(group))]].copy()
        else:
            children = [g for g in groups if g < group and not any(g < h < group for h in groups)]
            field = np.logical_or.reduce([values[g] for g in children]).astype(np.int64)
        for field_negative, owner in zip(negative, owners):
            if owner == group:
                field += field_negative
        pocket |= field < 0
        through |= field >= 2
        values[group] = field >= 1
    return values[root], pocket, through, sum(len(g) > 1 for g in set(owners))


def _certified_masks(V, F, arena, live, raw_sum, fill, search):
    """Sufficient certificates for ALL readings; otherwise return None.

    A negative total forces a pocket in every laminar ownership hierarchy:
    if all node fields were nonnegative, OR(children) <= sum(children)
    inductively implies 0 <= root_field <= raw_sum, a contradiction.

    Native winding propagation chooses its infinity cell and BFS tree before
    inspecting labels. Thus potentials of open ordinary-edge blocks add
    exactly, on every cell, to those of any completed group. Nonnegative
    block potentials and a total <=1 in each edge-connected component imply
    every possible group is 0/1 on live cells. This certifies material AND
    fail validity; overlap between distinct components remains permitted.
    """
    from .raw_topology import fixed_topology

    _, edges, fixed = fixed_topology(V, F)
    negative = (raw_sum < 0) & live
    if negative.any() and not fill:
        return None, negative, {"kind": "negative_total", "all_readings": True}
    n_fixed = int(fixed.max()) + 1
    root = list(range(n_fixed))

    def find(k):
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    for forward, backward in edges.values():
        first = find(int(fixed[forward[0]]))
        for face in forward + backward:
            root[find(int(fixed[face]))] = first
    components = {}
    for k in range(n_fixed):
        components.setdefault(find(k), []).append(k)
    # O(cells) workspace; no labelled faces x all-blocks allocation and no
    # memo of active-group fields. Native calls still use just two labels.
    if len(live) * 32 > 512 * 1024**2:
        raise RuntimeError("raw certificate workspace exceeded its memory limit")
    envelope = np.zeros(len(live), bool)
    total = np.zeros(len(live), np.int64)
    nonnegative, unit = True, True
    for blocks in components.values():
        component = np.zeros(len(live), np.int64)
        for k in blocks:
            search.step()
            potential = np.asarray(arena.winding((fixed != k).astype(int).tolist()))[:, 0]
            nonnegative &= not np.any(potential[live] < 0)
            # With no negative total, only the nonnegative/unit success
            # certificate applies. A negative block disproves its premise;
            # reading the remaining blocks cannot restore that premise.
            if not negative.any() and not nonnegative:
                return None
            envelope |= potential > 0
            # Python's search cap bounds the number of int32 columns, so their
            # int64 sum cannot overflow. Never sum in native int32 dtype.
            component += potential.astype(np.int64)
            # Either later blocks stay nonnegative (and cannot lower this
            # sum), or a later negative block also disproves the certificate.
            if not negative.any() and np.any(component[live] > 1):
                return None
        unit &= not np.any(component[live] > 1)
        total += component
    if not np.array_equal(total, raw_sum):
        raise RuntimeError("raw certificate block potentials are not additive")
    certificate = {"fixed_blocks": n_fixed, "edge_components": len(components),
                   "winding_calls": n_fixed, "all_readings": True}
    if negative.any():
        # Any possible material is a subset of envelope. Filling is monotone
        # on the fixed live-cell vertex adjacency graph used by arena.filled.
        # A mandatory pocket still outside filled(envelope) cannot be filled
        # in ANY reading. Do not infer this from in_cavity(False): its vertex
        # contact rejection, unlike filled(), is not a monotone barrier test.
        envelope[0] = False
        maximum_filled = np.asarray(arena.filled(envelope.astype(int).tolist()), bool)
        witness = negative & ~maximum_filled
        if witness.any():
            return None, witness, {"kind": "unfillable_negative_total", **certificate}
    elif nonnegative and unit and np.any(raw_sum[live] > 0):
        return raw_sum >= 1, None, {"kind": "nonnegative_unit_components", **certificate}
    return None


def resolve(mesh, check=False, fill=False):
    """Return exact solid mesh, volume and diagnostics, or an explicit refusal."""
    import trimesh
    from . import _cgal
    from .raw_topology import readings
    from .mesh_iou import _face_components

    V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    if len(F) > MAX_INPUT_FACES:
        raise RuntimeError("raw topology input exceeded its face resource limit")
    triangles = V[F]
    keep = (np.any(triangles[:, 0] != triangles[:, 1], axis=1)
            & np.any(triangles[:, 1] != triangles[:, 2], axis=1)
            & np.any(triangles[:, 0] != triangles[:, 2], axis=1))
    if not np.all(keep):
        F = F[keep]
        mesh = trimesh.Trimesh(V, F, process=False)
    if not len(F):
        raise InvalidReading("empty", "raw surface has no noncollapsed triangles")
    arena = _cgal.RawArrangement(np.ascontiguousarray(V), np.ascontiguousarray(F))
    live = np.asarray(arena.live, bool)
    raw_sum = np.asarray(arena.winding([0] * len(F)))[:, 0]
    if np.any(raw_sum[live] < 0) and not np.any(raw_sum[live] > 0):
        # A globally inside-out surface needs the same orientation repair
        # before raw readings as before the ordinary shell audit. Rebuild
        # from the reversed input, retaining every ambiguity and fail check.
        corrected = mesh.copy()
        corrected.invert()
        solid, volume, info = resolve(corrected, check=check, fill=fill)
        info["flipped_inverted_mesh"] = True
        return solid, volume, info
    search = Search()
    certificate = _certified_masks(V, F, arena, live, raw_sum, fill, search)
    if certificate is not None:
        inside, pocket, proof = certificate
        if pocket is not None:
            raise InvalidReading("pocket", "every raw reading has an inverted pocket",
                                 {"raw_reading_certificate": proof,
                                  "inverted_volume": arena.volume(pocket.astype(int).tolist()),
                                  "inverted_faces": len(arena.region(pocket.astype(int).tolist())[1]),
                                  "inverted_witness": "all_readings_negative_total"})
        before = inside.copy()
        if fill:
            inside = np.asarray(arena.filled(inside.astype(int).tolist()), bool)
        v, f = arena.region(inside.astype(int).tolist())
        info = {"raw_reading_certificate": proof, "fail_validity": "valid", "invalid_readings": 0,
                "assembly_owners": 0, "search_steps": search.steps,
                "winding_max": int(raw_sum[live].max()), "pocket_filled": False,
                "certified_interfaces": 0}
        if check:
            info["self_overlap_parts"] = 0
        if fill:
            from .mesh_iou import _solid_cavities
            v0, f0 = arena.region(before.astype(int).tolist())
            count = _solid_cavities(trimesh.Trimesh(v0, f0, process=False))
            added = arena.volume((inside & ~before).astype(int).tolist())
            info.update(cavities_filled_range=[count, count], volume_cavities_range=[added, added],
                        cavities_filled=count, volume_cavities=added)
        return trimesh.Trimesh(v, f, process=False), arena.volume(inside.astype(int).tolist()), info
    # This accelerator is valid only in isolation. Otherwise collapsing a
    # positive component can remove owners of another component's cavity.
    components = _face_components(mesh)
    groups = [np.flatnonzero(components == k) for k in range(int(components.max()) + 1)]
    bounds = [(V[F[g]].min(axis=(0, 1)), V[F[g]].max(axis=(0, 1))) for g in groups]
    certified = []
    for k, group in enumerate(groups):
        lo, hi = bounds[k]
        if not all(j == k or np.any(hi < a) or np.any(b < lo)
                   for j, (a, b) in enumerate(bounds)):
            continue
        if not _cgal.convex_interface(np.ascontiguousarray(V), np.ascontiguousarray(F[group])):
            continue
        field = np.asarray(arena.winding((components != k).astype(int).tolist()))[:, 0]
        if not np.any((field[live] < 0) | (field[live] > 1)):
            certified.append(group)
    fields, alternatives, _, topology_info = readings(
        V, F, arena, search, max_states=MAX_STATES, certified=certified)
    winding_max = int(raw_sum[live].max()) if live.any() else 0
    signatures, representative = set(), None
    valid, invalid, assembly_count = 0, 0, 0
    fail_overlaps, fail_clear = 0, 0

    def reject_mixed_validity():
        if valid and invalid:
            raise InvalidReading("ambiguous", "raw-surface readings include both valid and defective bodies",
                                 {"raw_assembly_candidates": True,
                                  "valid_readings": valid, "invalid_readings": invalid})

    def defective():
        nonlocal invalid
        invalid += 1
        reject_mixed_validity()
    before_fill, accepted_pockets = None, np.zeros(len(live), bool)
    fill_volumes, fill_counts, fill_cache = [], [], {}
    pocket_witness = None
    has_pocket, has_owners, has_overlap = False, False, False

    def record_pocket(mask):
        nonlocal has_pocket, pocket_witness
        has_pocket = True
        if pocket_witness is None:
            pocket_witness = mask & live

    for family in sorted(alternatives):
        search.step()
        W = np.column_stack([fields[k] for k in family]) if family else np.empty((len(live), 0), np.int32)
        positive = [k for k in range(W.shape[1]) if np.any(W[live, k] > 0)]
        negative = [np.where(np.asarray(region, bool), W[:, k], 0)
                    for k in range(W.shape[1]) if not np.any(W[live, k] > 0)
                    for region in arena.components((W[:, k] < 0).astype(int).tolist())]
        if not positive:
            defective()
            if negative:
                record_pocket(np.any(W < 0, axis=1))
            continue
        support = [W[:, k] >= 1 for k in positive]
        options = [covers(support, field < 0, live, search) for field in negative]
        has_owners |= any(any(len(owner) > 1 for owner in choices) for choices in options)
        if any(not choices for choices in options):
            defective()
            covered = np.logical_or.reduce(support)
            record_pocket(np.any(W < 0, axis=1) & ~covered)
            continue
        for owners in product(*options):
            search.step()
            inside, pocket, overlap, n_assemblies = material(W, positive, negative, owners)
            pocket &= live
            if pocket.any():
                if not fill or not arena.in_cavity((inside & ~pocket).astype(int).tolist(),
                                                   pocket.astype(int).tolist()):
                    record_pocket(pocket)
                    defective()
                    continue
            if check and np.any(overlap & live):
                has_overlap = True
                defective()
                continue
            if fill:
                before_fill = inside.copy()
                inside = np.asarray(arena.filled(inside.astype(int).tolist()), bool)
            if not np.any(inside & live):
                defective()
                continue
            signature = inside[live].tobytes()
            signatures.add(signature)
            if len(signatures) > 1:
                raise InvalidReading("ambiguous", "valid raw-surface readings define different material",
                                     {"raw_assembly_candidates": True})
            representative = inside
            valid += 1
            reject_mixed_validity()
            if np.any(overlap & live):
                fail_overlaps += 1
            else:
                fail_clear += 1
            if fill:
                from .mesh_iou import _solid_cavities
                key = before_fill[live].tobytes()
                if key not in fill_cache:
                    v0, f0 = arena.region(before_fill.astype(int).tolist())
                    fill_cache[key] = (_solid_cavities(trimesh.Trimesh(v0, f0, process=False)),
                                       arena.volume((inside & ~before_fill).astype(int).tolist()))
                count, added_volume = fill_cache[key]
                fill_counts.append(count)
                fill_volumes.append(added_volume)
            accepted_pockets |= pocket
            assembly_count = max(assembly_count, n_assemblies)
    if not valid:
        diagnostics = {"raw_assembly_candidates": has_owners}
        if has_pocket:
            diagnostics["inverted_volume"] = arena.volume(pocket_witness.astype(int).tolist())
            diagnostics["inverted_faces"] = len(arena.region(pocket_witness.astype(int).tolist())[1])
            diagnostics["inverted_witness"] = "one_defective_reading"
            raise InvalidReading("pocket", "no valid raw reading: an inverted pocket remains", diagnostics)
        if has_overlap:
            raise InvalidReading("overlap", "a raw body winds twice or more",
                                 {"raw_winding_max": winding_max, "self_overlap_parts": 1})
        raise InvalidReading("empty", "raw surface defines no nonempty solid", diagnostics)
    v, f = arena.region(representative.astype(int).tolist())
    volume = arena.volume(representative.astype(int).tolist())
    info = {"winding_families": len(alternatives), "valid_readings": valid, "invalid_readings": 0,
            "fail_validity": ("ambiguous" if fail_overlaps and fail_clear else
                              "self_intersecting" if fail_overlaps else "valid"),
            **topology_info,
            "certified_interfaces": len(certified),
            "assembly_owners": assembly_count, "search_steps": search.steps,
            "winding_max": winding_max,
            "pocket_filled": bool(accepted_pockets.any() and fill)}
    if check:
        info["self_overlap_parts"] = 0
    if accepted_pockets.any():
        info["inverted_cavity_volume"] = arena.volume(accepted_pockets.astype(int).tolist())
        info["inverted_cavity_faces"] = len(arena.region(accepted_pockets.astype(int).tolist())[1])
    if fill:
        info["cavities_filled_range"] = [min(fill_counts), max(fill_counts)]
        info["volume_cavities_range"] = [min(fill_volumes), max(fill_volumes)]
        info["cavities_filled"] = max(fill_counts)
        info["volume_cavities"] = max(fill_volumes)
    return trimesh.Trimesh(v, f, process=False), volume, info
