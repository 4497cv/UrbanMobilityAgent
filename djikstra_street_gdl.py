import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx


def dijkstra(G, start):
    dist = {node: float("inf") for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[start] = 0
    visited = set()

    while len(visited) < len(G.nodes):
        current = None
        current_dist = float("inf")

        for node in G.nodes:
            if node not in visited and dist[node] < current_dist:
                current = node
                current_dist = dist[node]

        if current is None:
            break

        visited.add(current)

        for neighbor in G.neighbors(current):
            edge_data = min(G[current][neighbor].values(), key=lambda d: d.get("length", 1))
            weight = edge_data.get("length", 1)

            if dist[current] + weight < dist[neighbor]:
                dist[neighbor] = dist[current] + weight
                prev[neighbor] = current

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
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    ax.set_axis_off()
    plt.show()


def main():
    print("Cargando grafo OSM...")

    # Usa solo uno de los dos métodos:
    # G = ox.graph_from_xml("provi.osm", simplify=True, retain_all=True)
    G = ox.graph_from_place("Guadalajara, Mexico", network_type="walk", simplify=True, retain_all=True)

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

    print("Ejecutando Dijkstra...")
    dist, prev = dijkstra(G, start_node)
    path = reconstruir_ruta(prev, start_node, end_node)

    if path:
        print("Distancia total:", dist[end_node])
        print("Ruta encontrada con", len(path), "nodos")
        plot_route(G, path)
    else:
        print("No se encontró ruta")


if __name__ == "__main__":
    main()
