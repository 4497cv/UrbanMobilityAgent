import heapq
import numpy as np
import osmnx as ox
from urban_mobility import calculate_toblers_time, plot_route

def weighted_astar(G, start, end, w_time, w_elev, w_veg=0.0):
    dist = {node: float('inf') for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = set()

    end_x = G.nodes[end]['x']
    end_y = G.nodes[end]['y']

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            break

        for neighbor in G.neighbors(current):
            if neighbor in visited:
                continue
            edge_data = G.get_edge_data(current, neighbor)
            if edge_data is None:
                continue
            data = edge_data.get(0, {})

            length  = float(data.get('length', 1.0))
            elev_u  = float(G.nodes[current].get('elevation', 0.0))
            elev_v  = float(G.nodes[neighbor].get('elevation', 0.0))
            iv      = float(data.get('indice_veg', 0.0))

            t         = calculate_toblers_time(elev_u, elev_v, length)
            gain      = max(0.0, elev_v - elev_u)
            edge_cost = w_time * t + w_elev * gain + w_veg * (1.0 - iv)

            g = dist[current] + edge_cost
            if g < dist[neighbor]:
                dist[neighbor] = g
                prev[neighbor] = current
                dx = G.nodes[neighbor]['x'] - end_x
                dy = G.nodes[neighbor]['y'] - end_y
                h  = np.sqrt(dx ** 2 + dy ** 2) * w_time * 0.1
                heapq.heappush(pq, (g + h, neighbor))

    path = []
    cur  = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if path and path[0] == start:
        return path
    return None


def run(G, user):
    start_node = ox.distance.nearest_nodes(G, user.start_coordinates.x, user.start_coordinates.y)
    end_node   = ox.distance.nearest_nodes(G, user.end_coordinates.x,   user.end_coordinates.y)

    print(f"[Weighted A*] w_time={user.w_time}  w_elev={user.w_elev}  w_veg={user.w_veg}")
    path = weighted_astar(G, start_node, end_node, w_time=user.w_time, w_elev=user.w_elev, w_veg=user.w_veg)

    if path:
        print(f"[Weighted A*] Route found with {len(path)} nodes")
        params = f"_t{user.w_time}_e{user.w_elev}_v{user.w_veg}"
        plot_route("weighted_astar", G, path, parameters=params)
    else:
        print("[Weighted A*] Route not found")
