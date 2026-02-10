import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import heapq

# ----------------------------------
# DIJKSTRA (USANDO travel_time)
# ----------------------------------
def dijkstra(G, start, weight="length"):
    dist = {node: float("inf") for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[start] = 0
    pq = [(0, start)]  # cola de prioridad

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current_dist > dist[current]:
            continue

        for neighbor in G.neighbors(current):
            edge_data = min(G[current][neighbor].values(), key=lambda d: d.get(weight, 1))
            w = edge_data.get(weight, 1)

            new_dist = dist[current] + w
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return dist, prev


def reconstruir_ruta(prev, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    if path and path[0] == start:
        return path
    return []


def plot_route(G, route):
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.to_crs(epsg=3857)
    edges = edges.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    edges.plot(ax=ax, linewidth=1, color="lightgray")
    nodes.plot(ax=ax, color="black", markersize=5)

    if len(route) > 1:
        route_edges = ox.routing.route_to_gdf(G, route)
        route_edges = route_edges.to_crs(epsg=3857)
        route_edges.plot(ax=ax, linewidth=3, color="red")

        nodes.loc[[route[0]]].plot(ax=ax, color="green", markersize=50, label="Inicio")
        nodes.loc[[route[-1]]].plot(ax=ax, color="blue", markersize=50, label="Destino")
    else:
        nodes.loc[[route[0]]].plot(ax=ax, color="orange", markersize=80, label="Inicio=Destino")

    ax.legend()
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
    ax.set_axis_off()
    plt.show()


def main():
    print("Cargando grafo OSM...")

    G = ox.graph_from_place("Guadalajara, Mexico", network_type="drive")

    # -----------------------------
    # Añadir travel_time a las aristas
    # -----------------------------
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" in data:
            speed_kmh = 50  # valor por defecto
            if "maxspeed" in data:
                try:
                    speed_kmh = float(str(data["maxspeed"]).split()[0])
                except:
                    pass
            speed_ms = speed_kmh * 1000 / 3600
            data["travel_time"] = data["length"] / speed_ms

    # coordenadas de inicio
    start_lat, start_lon = 20.679248, -103.377080
    # coordenadas de destino
    end_lat, end_lon = 20.697814, -103.384384

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    print("Número de nodos:", len(G.nodes))
    print("Número de aristas:", len(G.edges))

    print("Nodo inicio:", start_node)
    print("Nodo destino:", end_node)

    if start_node == end_node:
        print("Inicio y destino son el mismo nodo, cambia coordenadas.")
        return

    if not nx.has_path(G, start_node, end_node):
        print("No hay conexión entre inicio y destino")
        return

    print("Ejecutando Dijkstra (por tiempo)...")
    dist, prev = dijkstra(G, start_node, weight="travel_time")
    path = reconstruir_ruta(prev, start_node, end_node)

    if path:
        print("Tiempo total (segundos):", dist[end_node])
        print("Ruta encontrada con", len(path), "nodos")
        plot_route(G, path)
    else:
        print("No se encontró ruta")


if __name__ == "__main__":
    main()
