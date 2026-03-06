import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
from math import sqrt, inf
from math import inf, sqrt


def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def distance_euclidean(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def dijistra_heuristic(G, start, end, weight="length", heuristic=distance_euclidean):
    dist = {node: float("inf") for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[start] = 0
    visited = set()

    while(len(visited) < len(G.nodes)):
        current = None
        current_f = float("inf")

       
        for node in G.nodes:
            if(node not in visited):
                # h(n) - estimation cost
                h_n = heuristic(
                    (G.nodes[node]['x'], G.nodes[node]['y']),
                    (G.nodes[end]['x'], G.nodes[end]['y'])
                )

                # f(n) = g(n) + h(n)
                f_n = dist[node] + h_n

                if(f_n < current_f):
                    current = node
                    current_f = f_n

        if(current is None):
            break

        if(current == end):
            break

        visited.add(current)

        for neighbor in G.neighbors(current):
            edge_data = min(
                G[current][neighbor].values(),
                key=lambda d: d.get(weight, 1)
            )

            edge_weight = edge_data.get(weight, 1)

            # g(n)
            g_n = dist[current] + edge_weight

            # 🔹 Comparación SOLO con g(n)
            if(g_n < dist[neighbor]):
                dist[neighbor] = g_n
                prev[neighbor] = current

    return dist, prev


def dijkstra(G, start, weight = "length"):
    dist = {node: float("inf") for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[start] = 0
    visited = set()

    while(len(visited) < len(G.nodes)):
        current = None
        current_dist = float("inf")

        for node in G.nodes:
            if((node not in visited) and (dist[node] < current_dist)):
                current = node
                current_dist = dist[node]

        if(current is None):
            break

        visited.add(current)

        for neighbor in G.neighbors(current):
            edge_data = min(G[current][neighbor].values(), key=lambda d: d.get("length", 1))
            weight = edge_data.get(weight, 1)

            if((dist[current] + weight) < (dist[neighbor])):
                dist[neighbor] = dist[current] + weight
                prev[neighbor] = current

    return dist, prev


def reconstruir_ruta(prev, start, end):
    path = []
    current = end
    while(current is not None):
        path.append(current)
        current = prev[current]
    path.reverse()

    if path and path[0] == start:
        return path
    return []


def plot_route(G, route, online_mode = 1):
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.to_crs(epsg=3857)
    edges = edges.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    edges.plot(ax=ax, linewidth=1, color="lightgray")
    nodes.plot(ax=ax, color="black", markersize=5)

    if(len(route) > 1):
        route_edges = ox.routing.route_to_gdf(G, route)
        route_edges = route_edges.to_crs(epsg=3857)
        route_edges.plot(ax=ax, linewidth=3, color="red")

        nodes.loc[[route[0]]].plot(ax=ax, color="green", markersize=50, label="Inicio")
        nodes.loc[[route[-1]]].plot(ax=ax, color="blue", markersize=50, label="Destino")
    else:
        nodes.loc[[route[0]]].plot(ax=ax, color="orange", markersize=80, label="Inicio=Destino")

    ax.legend()
    
    if(online_mode):
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
        except:
            print(">> ERROR: could not invoke contextily.")

    ax.set_axis_off()
    plt.show()


def add_weight_to_graph(G):
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" in data:
            speed_kmh = 60  # valor por defecto
            if "maxspeed" in data:
                try:
                    speed_kmh = float(str(data["maxspeed"]).split()[0])
                except:
                    pass
            speed_ms = speed_kmh * 1000 / 3600
            data["travel_time"] = data["length"] / speed_ms
    return G


def main():
    print("Cargando grafo de Guadalajara...")
    G = ox.graph_from_place("Guadalajara, Mexico", network_type="walk", simplify=True, retain_all=True)

    print("Agregando elevaciones de nodos...")
    G = ox.elevation.add_node_elevations_google(G, api_key="AIzaSyAM3AJEapQcpVRglfgmg7hw8o9VSuS0p8I")
    print("Calculando pendientes de aristas...")
    G = ox.elevation.add_edge_grades(G)

    print("Ajustando tiempo de viaje según pendiente...")
    for u, v, k, data in G.edges(keys=True, data=True):
        base_travel_time = data.get("length", 1)/15  # default 15 m/s
        slope_factor = 1 + 5 * data.get("grade_abs", 0)  # penaliza rutas empinadas
        data["weighted_travel_time"] = base_travel_time * slope_factor

    # add corresponding weight to graph
    #add_weight_to_graph(G)

    start_coordinates = [20.68312, -103.36822]
    end_coordinates = [20.67061, -103.36169]

    # coordenadas de inicio
    start_lat, start_lon = start_coordinates[0], start_coordinates[1]
    # coordenadas de destino
    end_lat, end_lon = end_coordinates[0], end_coordinates[1]

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

    algorithm = "manhattan"

    if(algorithm == "djikstra"):
        print("Ejecutando Dijkstra Clasico...")
        # search for the optimal route from start to end
        dist, prev = dijkstra(G, start_node, "travel_time")
    elif(algorithm == "euclidean"):
        print("Ejecutando Dijkstra con Heuristica Euclidiana (A*)...")
        dist, prev = dijistra_heuristic(G, start_node, end_node, weight="travel_time", heuristic=distance_euclidean)
    elif(algorithm == "manhattan"):
        print("Ejecutando Dijkstra con Heuristica Manhattan (A*)...")
        dist, prev = dijistra_heuristic(G, start_node, end_node, weight="travel_time", heuristic=distance_manhattan)
    else:
        print("Ejecutando Dijkstra Clasico...")
        dist, prev = dijkstra(G, start_node, "travel_time")

    print("Reconstruyendo ruta....")
    path = reconstruir_ruta(prev, start_node, end_node)

    if path:
        print("Distancia total:", dist[end_node])
        print("Ruta encontrada con", len(path), "nodos")
        plot_route(G, path)
    else:
        print("No se encontró ruta")


if __name__ == "__main__":
    main()
