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
    2D hypervolume for a pair of objectives relative to ref_point.
    Both objectives are minimized; ref_point must be worse than all solutions.
    Higher is better.
    """
    pts = [(a, b) for a, b in pareto_objs
           if np.isfinite(a) and np.isfinite(b)
           and a < ref_point[0] and b < ref_point[1]]
    if not pts:
        return 0.0

    pts.sort(key=lambda x: x[0])

    hv = 0.0
    prev_b = ref_point[1]
    for a, b in pts:
        height = prev_b - b
        if height > 0:
            hv += (ref_point[0] - a) * height
        prev_b = min(prev_b, b)

    return hv


def _hypervolume_3d(pareto_objs, ref_point):
    """
    Approximated 3D hypervolume as the sum of 2D hypervolumes for each pair
    of objectives (time/elev, time/veg, elev/veg), normalized and averaged.
    Higher is better.
    """
    pairs = [
        ([(t, e) for t, e, v in pareto_objs], (ref_point[0], ref_point[1])),
        ([(t, v) for t, e, v in pareto_objs], (ref_point[0], ref_point[2])),
        ([(e, v) for t, e, v in pareto_objs], (ref_point[1], ref_point[2])),
    ]
    total = 0.0
    for pts, ref in pairs:
        total += _hypervolume_2d(pts, ref)
    return total / 3.0


def _spacing(pareto_objs):
    """
    Spacing metric using all 3 objectives (normalized to [0,1] per axis).
    Lower = more uniform distribution.
    """
    valid = [obj for obj in pareto_objs if all(np.isfinite(x) for x in obj)]
    if len(valid) < 2:
        return 0.0

    pts = np.array(valid)
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
    all_valid = [obj for obj in objs2 + objs3 if all(np.isfinite(x) for x in obj)]
    if not all_valid:
        print("No valid solutions found.")
        return

    ref = (
        max(t for t, e, v in all_valid) * 1.1,
        max(e for t, e, v in all_valid) * 1.1,
        max(v for t, e, v in all_valid) * 1.1,
    )

    hv2  = _hypervolume_3d(objs2, ref)
    hv3  = _hypervolume_3d(objs3, ref)
    sp2  = _spacing(objs2)
    sp3  = _spacing(objs3)

    avg2 = tuple(np.mean([obj[i] for obj in objs2]) for i in range(3)) if objs2 else (0, 0, 0)
    avg3 = tuple(np.mean([obj[i] for obj in objs3]) for i in range(3)) if objs3 else (0, 0, 0)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 57)
    print(f"  {'Metric':<26} {'NSGA-II':>12} {'NSGA-III':>12}")
    print("=" * 57)
    print(f"  {'Pareto solutions':<26} {len(objs2):>12} {len(objs3):>12}")
    print(f"  {'Hypervolume (3D avg)':<26} {hv2:>12.2f} {hv3:>12.2f}")
    print(f"  {'Spread (spacing std)':<26} {sp2:>12.4f} {sp3:>12.4f}")
    print(f"  {'Avg time (s)':<26} {avg2[0]:>12.1f} {avg3[0]:>12.1f}")
    print(f"  {'Avg elev gain (m)':<26} {avg2[1]:>12.2f} {avg3[1]:>12.2f}")
    print(f"  {'Avg veg cost':<26} {avg2[2]:>12.4f} {avg3[2]:>12.4f}")
    print(f"  {'Compute time (s)':<26} {time2:>12.1f} {time3:>12.1f}")
    print("=" * 57)
    print("  Hypervolume : higher is better  (larger dominated area)")
    print("  Spread      : lower  is better  (more uniform front)")
    print("  Veg cost    : lower  is better  (0 = full vegetation)")

    _plot(objs2, objs3)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(objs2, objs3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    pairs = [
        (0, 1, "Travel time (s)",    "Elevation gain (m)", "Time vs Elevation"),
        (0, 2, "Travel time (s)",    "Veg cost (0=green)", "Time vs Vegetation"),
        (1, 2, "Elevation gain (m)", "Veg cost (0=green)", "Elevation vs Vegetation"),
    ]

    for ax, (ix, iy, xlabel, ylabel, title) in zip(axes, pairs):
        if objs2:
            pts = sorted(
                [(obj[ix], obj[iy]) for obj in objs2 if all(np.isfinite(x) for x in obj)],
                key=lambda x: x[0]
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, 'o-', color='#e67e00',
                        label='NSGA-II', linewidth=1.5, markersize=6)

        if objs3:
            pts = sorted(
                [(obj[ix], obj[iy]) for obj in objs3 if all(np.isfinite(x) for x in obj)],
                key=lambda x: x[0]
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, 's--', color='#0077cc',
                        label='NSGA-III', linewidth=1.5, markersize=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pareto Front: NSGA-II vs NSGA-III (3 objectives)", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(workspace.get_workspace_path(), "src", "pareto_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    compare(pop_size=60, generations=20)
