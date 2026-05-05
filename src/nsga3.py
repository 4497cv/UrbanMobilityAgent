import random
import heapq
import numpy as np
import os
import osmnx as ox
import workspace
from urban_mobility import calculate_toblers_time, reconstruct_graph_from_graphml, set_elevation_weight, path_to_prev, reconstruct_route, plot_route
from moad import weighted_astar


# ── Objectives ───────────────────────────────────────────────────────────────

def evaluate(G, path):
    """
    Returns (travel_time_s, elevation_gain_m, veg_cost) for a path.
    - travel_time: sum of Tobler time per edge
    - elevation_gain: total positive elevation accumulated (uphill only)
    - veg_cost: average (1 - indice_veg) per edge — 0 = full vegetation, 1 = none
    """
    if len(path) < 2:
        return (float('inf'), float('inf'), float('inf'))

    total_time = 0.0
    total_gain = 0.0
    total_veg_cost = 0.0
    edge_count = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            return (float('inf'), float('inf'), float('inf'))
        data = edge_data.get(0, {})

        length = float(data.get('length', 1.0))
        elev_u = float(G.nodes[u].get('elevation', 0.0))
        elev_v = float(G.nodes[v].get('elevation', 0.0))
        iv = float(data.get('indice_veg', 0.0))

        total_time += calculate_toblers_time(elev_u, elev_v, length)
        gain = elev_v - elev_u
        if gain > 0:
            total_gain += gain
        total_veg_cost += (1.0 - iv)
        edge_count += 1

    veg_cost = total_veg_cost / edge_count if edge_count > 0 else 1.0
    return (total_time, total_gain, veg_cost)


# ── Path generation ───────────────────────────────────────────────────────────

def _init_population(G, start, end, pop_size):
    """Seed population with paths across the (w_time, w_elev, w_veg) trade-off simplex."""
    population = []
    weights = np.random.dirichlet([1, 1, 1], pop_size)

    for w in weights:
        path = weighted_astar(G, start, end, w_time=w[0], w_elev=w[1], w_veg=w[2])
        if path:
            population.append(path)

    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 4:
        w = np.random.dirichlet([1, 1, 1])
        path = weighted_astar(G, start, end, w_time=w[0], w_elev=w[1], w_veg=w[2])
        if path:
            population.append(path)
        attempts += 1

    return population


# ── Non-dominated sort ────────────────────────────────────────────────────────

def _dominates(a, b):
    """True if solution a dominates b (minimization of all objectives)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_nondominated_sort(objectives):
    n = len(objectives)
    dominated_count = [0] * n
    dominates_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dominates_set[i].append(j)
                dominated_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominates_set[j].append(i)
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


# ── NSGA-III specific: reference points, normalization, niching ───────────────

def _das_dennis_points(num_obj, divisions):
    """
    Das-Dennis structured reference points on the unit simplex.
    For M objectives and p divisions, returns C(M+p-1, p) points summing to 1.
    """
    def gen(left, num_remaining):
        if num_remaining == 1:
            yield [left]
            return
        for i in range(left + 1):
            for rest in gen(left - i, num_remaining - 1):
                yield [i] + rest

    points = []
    for combo in gen(divisions, num_obj):
        points.append([c / divisions for c in combo])
    return np.array(points)


def _normalize(objectives, ideal_point):
    """
    Normalize objectives via ideal-point translation and per-axis intercepts
    derived from extreme points (Achievement Scalarizing Function).
    """
    obj = np.array(objectives, dtype=float)
    M = obj.shape[1]

    translated = obj - ideal_point

    extreme_points = np.zeros((M, M))
    for i in range(M):
        weights = np.full(M, 1e-6)
        weights[i] = 1.0
        asf = np.max(translated / weights, axis=1)
        idx = np.argmin(asf)
        extreme_points[i] = translated[idx]

    try:
        b = np.ones(M)
        intercepts = np.linalg.solve(extreme_points, b)
        intercepts = 1.0 / intercepts
        if np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = np.max(translated, axis=0)
        intercepts[intercepts <= 1e-6] = 1.0

    return translated / intercepts


def _associate(normalized_obj, ref_points):
    """
    Associate each individual with the nearest reference line (perpendicular distance).
    Returns: assignments (list of ref-point indices), distances (list of floats).
    """
    assignments = []
    distances = []

    norms_sq = np.sum(ref_points ** 2, axis=1)

    for s in normalized_obj:
        proj_scalars = (ref_points @ s) / norms_sq
        projected = proj_scalars[:, None] * ref_points
        dists = np.linalg.norm(s - projected, axis=1)
        idx = int(np.argmin(dists))
        assignments.append(idx)
        distances.append(float(dists[idx]))

    return assignments, distances


def _niching(K, fronts, last_front_idx, assignments, distances, num_ref_points):
    """
    NSGA-III niching: select K individuals from fronts[last_front_idx] using
    reference-point niche counts built from fronts[:last_front_idx].
    """
    niche_count = np.zeros(num_ref_points, dtype=int)
    for f in fronts[:last_front_idx]:
        for i in f:
            niche_count[assignments[i]] += 1

    last_front = list(fronts[last_front_idx])
    selected = []

    ref_to_last = {r: [] for r in range(num_ref_points)}
    for i in last_front:
        ref_to_last[assignments[i]].append(i)

    while len(selected) < K:
        candidates_refs = [r for r in range(num_ref_points) if ref_to_last[r]]
        if not candidates_refs:
            break

        min_niche = min(niche_count[r] for r in candidates_refs)
        min_refs = [r for r in candidates_refs if niche_count[r] == min_niche]
        chosen_ref = random.choice(min_refs)

        candidates = ref_to_last[chosen_ref]
        if niche_count[chosen_ref] == 0:
            chosen = min(candidates, key=lambda i: distances[i])
        else:
            chosen = random.choice(candidates)

        selected.append(chosen)
        ref_to_last[chosen_ref].remove(chosen)
        niche_count[chosen_ref] += 1

    return selected


# ── Variation operators ───────────────────────────────────────────────────────

def _crossover(p1, p2):
    """Path crossover at a shared intermediate node."""
    shared = list(set(p1[1:-1]) & set(p2[1:-1]))
    if not shared:
        return list(p1), list(p2)

    cp = random.choice(shared)
    i1 = p1.index(cp)
    i2 = p2.index(cp)
    return p1[:i1] + p2[i2:], p2[:i2] + p1[i1:]


def _mutate(G, path, mutation_rate=0.15):
    """Replace a random interior segment with an alternative A* sub-path."""
    if random.random() > mutation_rate or len(path) < 4:
        return path

    idx1, idx2 = sorted(random.sample(range(1, len(path) - 1), 2))
    w = np.random.dirichlet([1, 1, 1])
    segment = weighted_astar(G, path[idx1], path[idx2], w_time=w[0], w_elev=w[1], w_veg=w[2])

    if segment:
        return path[:idx1] + segment + path[idx2 + 1:]
    return path


def _random_parent(population):
    """NSGA-III uses random parent selection; reference points handle diversity."""
    return random.choice(population)


# ── Entry point ───────────────────────────────────────────────────────────────

def nsga3(G, start, end, pop_size=60, generations=80, mutation_rate=0.15, divisions=12):
    """
    Multi-objective route optimization with NSGA-III.
    Objectives (all minimized):
      1. Tobler travel time (seconds)
      2. Elevation gain (meters, uphill only)
      3. Vegetation cost: avg (1 - indice_veg) per edge — 0 = full greenery

    Differs from NSGA-II in diversity preservation: instead of crowding distance,
    it builds Das-Dennis reference points on the unit simplex and associates
    individuals to the nearest reference line, then performs niche-based
    selection on the splitting front.

    Returns:
      List of (path, (time_s, elev_gain_m, veg_cost)) for the final Pareto front,
      sorted by ascending travel time.
    """
    print(f"[NSGA-III] Generating reference points (divisions={divisions})...")
    ref_points = _das_dennis_points(num_obj=3, divisions=divisions)
    print(f"[NSGA-III] {len(ref_points)} reference points generated")

    print(f"[NSGA-III] Seeding population ({pop_size} individuals)...")
    population = _init_population(G, start, end, pop_size)

    if not population:
        print("[NSGA-III] Could not generate initial population.")
        return []

    print(f"[NSGA-III] Population ready: {len(population)} paths")

    for gen in range(generations):
        # Generate offspring (random parent selection, NSGA-III style)
        offspring = []
        while len(offspring) < pop_size:
            p1 = _random_parent(population)
            p2 = _random_parent(population)
            c1, c2 = _crossover(p1, p2)
            offspring.append(_mutate(G, c1, mutation_rate))
            offspring.append(_mutate(G, c2, mutation_rate))

        # Combine parents + offspring
        combined = population + offspring[:pop_size]
        obj_combined = [evaluate(G, p) for p in combined]

        # Non-dominated sort
        fronts = _fast_nondominated_sort(obj_combined)

        # Fill next population front by front
        next_pop_idx = []
        last_front_idx = 0
        for i, front in enumerate(fronts):
            if len(next_pop_idx) + len(front) <= pop_size:
                next_pop_idx.extend(front)
                last_front_idx = i + 1
            else:
                last_front_idx = i
                break

        # Niching for the splitting front
        if len(next_pop_idx) < pop_size and last_front_idx < len(fronts):
            relevant_fronts = fronts[:last_front_idx + 1]
            relevant_idx = [i for f in relevant_fronts for i in f]
            relevant_obj = np.array([obj_combined[i] for i in relevant_idx], dtype=float)

            valid_mask = np.all(np.isfinite(relevant_obj), axis=1)
            finite_obj = relevant_obj[valid_mask] if np.any(valid_mask) else relevant_obj

            if len(finite_obj) == 0:
                population = [combined[i] for i in next_pop_idx[:pop_size]]
                continue

            ideal_point = np.min(finite_obj, axis=0)
            normalized = _normalize(relevant_obj, ideal_point)

            assignments_local, distances_local = _associate(normalized, ref_points)
            assignments = {relevant_idx[k]: assignments_local[k] for k in range(len(relevant_idx))}
            distances = {relevant_idx[k]: distances_local[k] for k in range(len(relevant_idx))}

            K = pop_size - len(next_pop_idx)
            chosen = _niching(
                K,
                relevant_fronts,
                last_front_idx,
                assignments,
                distances,
                len(ref_points),
            )
            next_pop_idx.extend(chosen)

        population = [combined[i] for i in next_pop_idx[:pop_size]]

        if gen % 10 == 0 or gen == generations - 1:
            front0_size = len(fronts[0]) if fronts else 0
            print(f"[NSGA-III] Gen {gen + 1:>3}/{generations}  |  Pareto front: {front0_size} solutions")

    # Build and return final Pareto front
    final_obj = [evaluate(G, p) for p in population]
    final_fronts = _fast_nondominated_sort(final_obj)
    pareto = sorted(
        [(population[i], final_obj[i]) for i in final_fronts[0]],
        key=lambda x: x[1][0],
    )

    print(f"[NSGA-III] Done. {len(pareto)} non-dominated solutions.")
    return pareto


def run_nsga3(G, user):
    start_node = ox.distance.nearest_nodes(G, user.start_coordinates.x, user.start_coordinates.y)
    end_node   = ox.distance.nearest_nodes(G, user.end_coordinates.x,   user.end_coordinates.y)

    pareto = nsga3(G, start_node, end_node, pop_size=60, generations=20)

    print("\n── Pareto Front ──────────────────────────────────")
    print(f"  {'#':>3}  {'Time (s)':>12}  {'Elev gain (m)':>14}  {'Veg cost':>10}  {'Nodes':>6}")
    for idx, (raw_path, (t, e, v)) in enumerate(pareto):
        prev = path_to_prev(raw_path)
        path = reconstruct_route(prev, start_node, end_node)
        if path:
            print(f"  {idx:>3}  {t:>12.1f}  {e:>14.2f}  {v:>10.4f}  {len(path):>6}")
            plot_route("NSGA3", G, path, solution_index=idx)
        else:
            print(f"  {idx:>3}  (ruta inválida)")