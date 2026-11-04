"""Exact shell enumeration with completed winding fields.

Only closed groups leaving the frontier are evaluated by native winding.
Their FULL cell fields, multiplicities and regular-mixed flags survive.
The native RawArrangement.winding allocation guard remains independent:
this helper requests at most two labels (the group and its complement).
"""
from collections import Counter
import sys

import numpy as np


def fixed_topology(vertices, faces, certified=()):
    """Exact merged edges and blocks forced by ordinary two-face edges."""
    V, F = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    points = np.ascontiguousarray(V + 0.0)
    _, inv = np.unique(points.view(np.dtype((np.void, 24))).ravel(), return_inverse=True)
    merged = inv.reshape(-1)[F]
    edges = {}
    for face, tri in enumerate(merged):
        for a, b in zip(tri, np.roll(tri, -1)):
            if a == b:
                raise ValueError("raw surface has a collapsed face")
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges.setdefault(key, ([], []))[int(a > b)].append(face)
    if any(len(a) != len(b) for a, b in edges.values()):
        raise ValueError("raw surface has unbalanced edges")
    root = list(range(len(F)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for a, b in edges.values():
        if len(a) == 1:
            x, y = find(a[0]), find(b[0])
            root[max(x, y)] = min(x, y)
    # The caller may supply only isolated components with a proven unique
    # nonnegative material, not merely components with the same total winding.
    for group in certified:
        for face in group[1:]:
            x, y = find(int(group[0])), find(int(face))
            root[max(x, y)] = min(x, y)
    roots = [find(i) for i in range(len(F))]
    rank = {r: k for k, r in enumerate(sorted(set(roots)))}
    fixed = np.asarray([rank[r] for r in roots], np.int64)
    return merged, edges, fixed


def readings(vertices, faces, arena, search, max_states=4096,
             max_memory=512 * 1024**2, certified=()):
    """Return fields, families, regular_mixed_flags, diagnostics.

    ``families`` is a set of tuples of field indices, preserving multiplicity.
    ``fields[k]`` covers every arena cell, including zero-volume cells.
    ``regular_mixed_flags[k]`` marks a two-face-edge sheet with both positive
    and negative live winding; this flag records topology, not validity.
    A supplied search object must provide step(), sharing the caller's budget.
    """
    V, F = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    live = np.asarray(arena.live, bool)
    merged, edges, fixed = fixed_topology(V, F, certified)
    n_fixed = int(fixed.max()) + 1 if len(fixed) else 0
    # Bound the initial bitset representation before allocating large integers.
    if n_fixed * (40 + (n_fixed + 7) // 8) > max_memory:
        raise RuntimeError("raw frontier initial bitsets exceed its memory limit")
    crowded = [(tuple(int(fixed[f]) for f in a), tuple(int(fixed[f]) for f in b))
               for a, b in edges.values() if len(a) > 1]
    edge_blocks = [set(a + b) for a, b in crowded]
    remaining = set(range(len(crowded)))
    seen, order = set(), []
    counts = Counter(k for group in edge_blocks for k in group)
    while remaining:
        def score(i):
            after = seen | edge_blocks[i]
            frontier = {k for k in after if counts[k] - (k in edge_blocks[i]) > 0}
            return len(frontier), len(edge_blocks[i] - seen), -sum(counts[k] == 1 for k in edge_blocks[i]), i

        # Ordering is a heuristic only; every edge is still processed. Bound
        # its quadratic cost without spending the interpretation search budget.
        candidates = sorted(remaining)[:64]
        i = min(candidates, key=score)
        order.append(i)
        remaining.remove(i)
        seen |= edge_blocks[i]
        for k in edge_blocks[i]:
            counts[k] -= 1
    crowded = [crowded[i] for i in order]
    future, all_incident = [], 0
    for a, b in reversed(crowded):
        future.append(all_incident)
        for k in a + b:
            all_incident |= 1 << k
    future.reverse()
    edge_rows = np.sort(merged[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, edge_id = np.unique(edge_rows, axis=0, return_inverse=True)
    edge_id = edge_id.reshape(-1, 3)
    fields, flags, field_ids, cache = [], [], {}, {}
    field_bytes = 0
    next_memory_check, sampled_storage = 0, 0

    def budget(*dynamic, force=False):
        # Actual Python container/key allocations and owned numpy storage.
        # Sampling avoids a quadratic walk over unchanged caches. Count and
        # field-byte guards run at every insertion; this is an allocation
        # checkpoint, not a hard process RSS ceiling. Native storage is separate.
        nonlocal next_memory_check, sampled_storage
        if not force and search.steps < next_memory_check:
            return sampled_storage
        next_memory_check = search.steps + 256
        objects = [fields, flags, field_ids, cache, *dynamic]
        visited, total = set(), 0
        while objects:
            item = objects.pop()
            key = id(item)
            if key in visited:
                continue
            visited.add(key)
            total += sys.getsizeof(item)
            if isinstance(item, dict):
                objects.extend(item.keys())
                objects.extend(item.values())
            elif isinstance(item, (tuple, list, set, frozenset)):
                objects.extend(item)
        if total > max_memory:
            raise RuntimeError("raw frontier storage exceeded its memory limit")
        sampled_storage = max(sampled_storage, total)
        return total

    def field(bits):
        nonlocal field_bytes
        if bits in cache:
            return cache[bits]
        search.step()
        member = np.asarray([bool(bits & (1 << k)) for k in range(n_fixed)])[fixed]
        # Both labels are closed; original total minus this group is closed too.
        labels = (~member).astype(np.int64)
        W = np.asarray(arena.winding(labels.tolist()))[:, 0]
        if not np.any(W[live]):
            cache[bits] = None
            return None
        mixed = bool(np.any(W[live] > 0) and np.any(W[live] < 0))
        regular = False
        if mixed:
            edge_counts = np.bincount(edge_id[member].ravel())
            regular = bool(np.all(edge_counts[edge_counts != 0] == 2))
        values = np.ascontiguousarray(W, dtype=np.int32)
        # FULL W is needed: zero-volume cells can join negative components.
        key = (values.tobytes(), regular)
        if key not in field_ids:
            field_bytes += values.nbytes + len(key[0])
            if field_bytes > max_memory:
                raise RuntimeError("raw frontier fields exceed its memory limit")
            field_ids[key] = len(fields)
            fields.append(values.copy())
            flags.append(regular)
        cache[bits] = field_ids[key]
        return cache[bits]

    def complete(active, finished, incidence):
        keep, done = [], list(finished)
        for group in active:
            if group & incidence:
                keep.append(group)
            else:
                token = field(group)
                if token is not None:
                    done.append(token)
        return tuple(sorted(keep)), tuple(sorted(done))

    def limit(size):
        if size > max_states:
            raise RuntimeError("raw frontier state count exceeded its resource limit")

    states = {complete(tuple(1 << k for k in range(n_fixed)), (), all_incident)}
    peak_states, peak_storage = 1, budget(states, force=True)
    for (a, b), incidence in zip(crowded, future):
        next_states = set()
        for active, finished in states:
            position = {k: group for group in active for k in range(n_fixed) if group & (1 << k)}
            counts = tuple(sorted(Counter(position[k] for k in b).items()))
            work = {(active, counts)}
            for component in a:
                next_work, bit = set(), 1 << component
                for groups, counts in work:
                    own = next(g for g in groups if g & bit)
                    for other, _ in counts:
                        search.step()
                        joined = own | other
                        updated = tuple(sorted([g for g in groups if g != own and g != other] + [joined]))
                        remaining_counts = Counter()
                        for g, count in counts:
                            count -= g == other
                            if count:
                                remaining_counts[joined if g == own or g == other else g] += count
                        next_work.add((updated, tuple(sorted(remaining_counts.items()))))
                        limit(len(next_work))
                peak_storage = max(peak_storage, budget(states, next_states, work, next_work))
                work = next_work
            for active, _ in work:
                next_states.add(complete(active, finished, incidence))
                limit(len(next_states))
        peak_storage = max(peak_storage, budget(states, next_states, force=True))
        states = next_states
        peak_states = max(peak_states, len(states))
    if any(active for active, finished in states):
        raise RuntimeError("raw frontier left an unfinished group")
    families = {finished for active, finished in states}
    info = {"frontier_readings": len(families), "frontier_peak_states": peak_states,
            "frontier_fixed_blocks": n_fixed, "frontier_crowded_edges": len(crowded),
            "frontier_winding_calls": len(cache), "frontier_fields": len(fields),
            "frontier_field_bytes": field_bytes, "frontier_storage_sample_peak_bytes": peak_storage}
    return fields, families, flags, info
