import time
import numpy as np
import matplotlib.pyplot as plt
import os
import osmnx as ox

import workspace
import nsga2
import nsga3
from urban_mobility import reconstruct_graph_from_graphml, set_elevation_weight


# ── Metrics ───────────────────────────────────────────────────────────────────

def _hypervolume_2d(pareto_objs, ref_point):
    """
    2D hypervolume dominated by the Pareto front relative to ref_point.
    Both objectives are minimized; ref_point must be worse than all solutions.
    Higher is better.
    """
    pts = [(t, e) for t, e in pareto_objs
           if np.isfinite(t) and np.isfinite(e)
           and t < ref_point[0] and e < ref_point[1]]
    if not pts:
        return 0.0

    pts.sort(key=lambda x: x[0])

    hv = 0.0
    prev_e = ref_point[1]
    for t, e in pts:
        height = prev_e - e
        if height > 0:
            hv += (ref_point[0] - t) * height
        prev_e = min(prev_e, e)

    return hv


def _spacing(pareto_objs):
    """
    Spacing metric: std dev of distances between consecutive solutions
    (normalized to [0,1] per axis). Lower = more uniform distribution.
    """
    valid = [(t, e) for t, e in pareto_objs if np.isfinite(t) and np.isfinite(e)]
    if len(valid) < 2:
        return 0.0

    pts = np.array(sorted(valid, key=lambda x: x[0]))
    ranges = pts.max(axis=0) - pts.min(axis=0)
    ranges[ranges == 0] = 1.0
    pts_norm = (pts - pts.min(axis=0)) / ranges

    dists = [np.linalg.norm(pts_norm[i + 1] - pts_norm[i])
             for i in range(len(pts_norm) - 1)]
    return float(np.std(dists))


# ── Main comparison ───────────────────────────────────────────────────────────

def compare(pop_size=60, generations=20):
    print("Loading graph...")
    if not os.path.exists(workspace.get_graphml_gdl_path()):
        print("Graph not found. Run main.py first.")
        return

    G = reconstruct_graph_from_graphml(workspace.get_graphml_gdl_path())
    set_elevation_weight(G)

    start_lon, start_lat = -103.376624, 20.630163
    end_lat,   end_lon   =  20.697814, -103.384384
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node   = ox.distance.nearest_nodes(G, end_lon,   end_lat)

    # ── NSGA-II ──────────────────────────────────────────────────────────────
    print("\nRunning NSGA-II...")
    t0 = time.perf_counter()
    pareto2 = nsga2.nsga2(G, start_node, end_node,
                          pop_size=pop_size, generations=generations)
    time2 = time.perf_counter() - t0
    objs2 = [obj for _, obj in pareto2]

    # ── NSGA-III ─────────────────────────────────────────────────────────────
    print("\nRunning NSGA-III...")
    t0 = time.perf_counter()
    pareto3 = nsga3.nsga3(G, start_node, end_node,
                          pop_size=pop_size, generations=generations)
    time3 = time.perf_counter() - t0
    objs3 = [obj for _, obj in pareto3]

    # ── Reference point for hypervolume ──────────────────────────────────────
    all_valid = [(t, e) for t, e in objs2 + objs3
                 if np.isfinite(t) and np.isfinite(e)]
    if not all_valid:
        print("No valid solutions found.")
        return

    ref = (
        max(t for t, e in all_valid) * 1.1,
        max(e for t, e in all_valid) * 1.1,
    )

    hv2  = _hypervolume_2d(objs2, ref)
    hv3  = _hypervolume_2d(objs3, ref)
    sp2  = _spacing(objs2)
    sp3  = _spacing(objs3)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 57)
    print(f"  {'Metric':<26} {'NSGA-II':>12} {'NSGA-III':>12}")
    print("=" * 57)
    print(f"  {'Pareto solutions':<26} {len(objs2):>12} {len(objs3):>12}")
    print(f"  {'Hypervolume':<26} {hv2:>12.2f} {hv3:>12.2f}")
    print(f"  {'Spread (spacing std)':<26} {sp2:>12.4f} {sp3:>12.4f}")
    print(f"  {'Compute time (s)':<26} {time2:>12.1f} {time3:>12.1f}")
    print("=" * 57)
    print("  Hypervolume : higher is better  (larger dominated area)")
    print("  Spread      : lower  is better  (more uniform front)")

    _plot(objs2, objs3)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(objs2, objs3):
    fig, ax = plt.subplots(figsize=(8, 6))

    if objs2:
        pts = sorted([(t, e) for t, e in objs2 if np.isfinite(t) and np.isfinite(e)],
                     key=lambda x: x[0])
        if pts:
            t2, e2 = zip(*pts)
            ax.plot(t2, e2, 'o-', color='#e67e00',
                    label='NSGA-II', linewidth=1.5, markersize=6)

    if objs3:
        pts = sorted([(t, e) for t, e in objs3 if np.isfinite(t) and np.isfinite(e)],
                     key=lambda x: x[0])
        if pts:
            t3, e3 = zip(*pts)
            ax.plot(t3, e3, 's--', color='#0077cc',
                    label='NSGA-III', linewidth=1.5, markersize=6)

    ax.set_xlabel("Travel time (s)")
    ax.set_ylabel("Elevation gain (m)")
    ax.set_title("Pareto Front: NSGA-II vs NSGA-III")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(workspace.get_workspace_path(), "src", "pareto_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    compare(pop_size=60, generations=20)
