import random
import heapq
import numpy as np
import os
import osmnx as ox
import workspace
import profiles
from urban_mobility import calculate_toblers_time, weighted_astar, path_to_prev, reconstruct_route, plot_route

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


# ── NSGA-II core ──────────────────────────────────────────────────────────────

def _dominates(a, b):
    """True if solution a dominates b (minimization of all objectives)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_nondominated_sort(objectives):
    """
    NSGA-II fast non-dominated sort.
    Returns a list of fronts; each front is a list of indices into objectives.
    """
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


def _crowding_distance(front, objectives):
    """
    Crowding distance for each index in front.
    Boundary solutions (min/max per objective) get infinite distance.
    """
    n = len(front)
    cd = {i: 0.0 for i in front}

    if n <= 2:
        for i in front:
            cd[i] = float('inf')
        return cd

    num_obj = len(objectives[0])

    for m in range(num_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        obj_min = objectives[sorted_front[0]][m]
        obj_max = objectives[sorted_front[-1]][m]

        cd[sorted_front[0]] = float('inf')
        cd[sorted_front[-1]] = float('inf')

        if obj_max == obj_min:
            continue

        for k in range(1, n - 1):
            prev_val = objectives[sorted_front[k - 1]][m]
            next_val = objectives[sorted_front[k + 1]][m]
            cd[sorted_front[k]] += (next_val - prev_val) / (obj_max - obj_min)

    return cd


def _tournament(population, rank, cd, k=2):
    """
    Binary tournament selection.
    Lower rank wins; ties resolved by larger crowding distance.
    """
    candidates = random.sample(range(len(population)), k)

    def better(a, b):
        if rank[a] != rank[b]:
            return a if rank[a] < rank[b] else b
        return a if cd.get(a, 0.0) >= cd.get(b, 0.0) else b

    winner = candidates[0]
    for c in candidates[1:]:
        winner = better(winner, c)
    return population[winner]


def _crossover(p1, p2):
    """
    Path crossover at a shared intermediate node.
    When no shared intermediate nodes exist, returns copies of the parents.
    """
    shared = list(set(p1[1:-1]) & set(p2[1:-1]))
    if not shared:
        return list(p1), list(p2)

    cp = random.choice(shared)
    i1 = p1.index(cp)
    i2 = p2.index(cp)
    return p1[:i1] + p2[i2:], p2[:i2] + p1[i1:]


def _mutate(G, path, mutation_rate=0.15):
    """
    Mutation: replace a random interior segment with an alternative A* sub-path.
    The replacement uses a random (w_time, w_elev) pair to introduce diversity.
    """
    if random.random() > mutation_rate or len(path) < 4:
        return path

    idx1, idx2 = sorted(random.sample(range(1, len(path) - 1), 2))
    w = np.random.dirichlet([1, 1, 1])
    segment = weighted_astar(G, path[idx1], path[idx2], w_time=w[0], w_elev=w[1], w_veg=w[2])

    if segment:
        return path[:idx1] + segment + path[idx2 + 1:]
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def nsga2(G, start, end, pop_size=60, generations=80, mutation_rate=0.15):
    """
    Multi-objective route optimization with NSGA-II.
    Objectives (all minimized):
      1. Tobler travel time (seconds)
      2. Elevation gain (meters, uphill only)
      3. Vegetation cost: avg (1 - indice_veg) per edge — 0 = full greenery

    Returns:
      List of (path, (time_s, elev_gain_m, veg_cost)) for the final Pareto front,
      sorted by ascending travel time.
    """
    print(f"[NSGA-II] Seeding population ({pop_size} individuals)...")
    population = _init_population(G, start, end, pop_size)

    if not population:
        print("[NSGA-II] Could not generate initial population.")
        return []

    print(f"[NSGA-II] Population ready: {len(population)} paths")

    for gen in range(generations):
        objectives = [evaluate(G, p) for p in population]

        fronts = _fast_nondominated_sort(objectives)
        rank = {}
        cd = {}
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r
            cd.update(_crowding_distance(front, objectives))

        # Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = _tournament(population, rank, cd)
            p2 = _tournament(population, rank, cd)
            c1, c2 = _crossover(p1, p2)
            offspring.append(_mutate(G, c1, mutation_rate))
            offspring.append(_mutate(G, c2, mutation_rate))

        # Combine parents + offspring, then select next generation
        combined = population + offspring[:pop_size]
        obj_combined = [evaluate(G, p) for p in combined]
        fronts_c = _fast_nondominated_sort(obj_combined)
        cd_c = {}
        for front in fronts_c:
            cd_c.update(_crowding_distance(front, obj_combined))

        next_pop = []
        for front in fronts_c:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(combined[i] for i in front)
            else:
                remaining = pop_size - len(next_pop)
                by_cd = sorted(front, key=lambda i: cd_c.get(i, 0.0), reverse=True)
                next_pop.extend(combined[i] for i in by_cd[:remaining])
                break

        population = next_pop

        if gen % 10 == 0 or gen == generations - 1:
            pareto_size = len(fronts_c[0]) if fronts_c else 0
            print(f"[NSGA-II] Gen {gen + 1:>3}/{generations}  |  Pareto front: {pareto_size} solutions")

    # Build and return final Pareto front
    final_obj = [evaluate(G, p) for p in population]
    final_fronts = _fast_nondominated_sort(final_obj)
    pareto = sorted(
        [(population[i], final_obj[i]) for i in final_fronts[0]],
        key=lambda x: x[1][0]  # sort by travel time
    )

    print(f"[NSGA-II] Done. {len(pareto)} non-dominated solutions.")
    return pareto


def run_nsga2(G, user):
    start_lon, start_lat = -103.376624, 20.630163
    end_lat,   end_lon   =  20.697814, -103.384384

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node   = ox.distance.nearest_nodes(G, end_lon,   end_lat)

    pareto = nsga2(G, start_node, end_node, pop_size=60, generations=20)

    print("\n── Pareto Front ──────────────────────────────────")
    print(f"  {'#':>3}  {'Time (s)':>12}  {'Elev gain (m)':>14}  {'Veg cost':>10}  {'Nodes':>6}")
    for idx, (raw_path, (t, e, v)) in enumerate(pareto):
        prev = path_to_prev(raw_path)
        path = reconstruct_route(prev, start_node, end_node)
        if path:
            print(f"  {idx:>3}  {t:>12.1f}  {e:>14.2f}  {v:>10.4f}  {len(path):>6}")
            plot_route("NSGA2", G, path, solution_index=idx)
        else:
            print(f"  {idx:>3}  (ruta inválida)")