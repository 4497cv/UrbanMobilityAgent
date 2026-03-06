import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import heapq
import os
import sys
import workspace
import math 

def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def distance_euclidean(p1, p2):
    """
    Calculates the Euclidean distance (straight-line distance) between two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def a_star(G, start, end, heuristic, weight):
    """
    A* algorithm to find the shortest path from start to goal using a NetworkX graph.
    
    Parameters:
    - G: The graph (assumed to be a NetworkX graph).
    - start: The starting node.
    - end: The goal node.
    - heuristic: A function that returns the heuristic value (h(n)) for a given node.
    - weight: The edge weight (e.g., travel time).
    
    Returns:
    - dist: A dictionary of the shortest distance to each node from start.
    - prev: A dictionary of the previous node on the shortest path.
    - path: A list of nodes representing the shortest path from start to goal.
    """
    # Initialize distances and previous node tracker
    distance = {node: float('inf') for node in G.nodes}  # g(n)
    prev = {node: None for node in G.nodes}  # To store the previous node in the shortest path
    distance[start] = 0  # g(start) = 0
    
    # Priority queue to store nodes and their f values (f = g + h)
    pq = []
    
    # Heuristic for the start node
    start_coords = (G.nodes[start]['x'], G.nodes[start]['y'])
    end_coords = (G.nodes[end]['x'], G.nodes[end]['y'])
    h = heuristic(start_coords, end_coords)
    heapq.heappush(pq, (h, start))  # Push with (f, node), initially f = h
    
    # A dictionary to store the f value of each node
    f_score = {node: float('inf') for node in G.nodes}
    f_score[start] = h
    
    # Set to track visited nodes
    visited = set()

    while pq:
        # Get the node with the lowest f value
        _, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        
        if current == end:
            break  # Path found to the goal
        
        # Check each neighbor
        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)
            weight = edge_data.get('weight', 1)  # Default to 1 if no weight is specified
            
            g = distance[current] + weight  # Actual cost from start to neighbor
            neighbor_coords = (G.nodes[neighbor]['x'], G.nodes[neighbor]['y'])
            h = heuristic(neighbor_coords, end_coords)  # Heuristic from neighbor to goal
            f = g + h  # Total estimated cost
            
            if g < distance[neighbor]:  # Found a shorter path to the neighbor
                distance[neighbor] = g
                prev[neighbor] = current
                if f < f_score[neighbor]:  # Push to the queue with the new f value
                    f_score[neighbor] = f
                    heapq.heappush(pq, (f, neighbor))  # Push the node with its f value
    
    return distance, prev

def reconstruct_route(prev, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    if path and path[0] == start:
        return path
    return []


def plot_route(algorithm_used, G, route):
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
    
    # save resulting route in shape file
    route_gdf = ox.routing.route_to_gdf(G, route)

    if("djikstra" == algorithm_used):
        route_gdf.to_file(os.path.join(workspace.get_route_djikstra_gdl_path(), "ruta_dijkstra.shp"))
    elif("a_star" == algorithm_used):
        route_gdf.to_file(os.path.join(workspace.get_route_a_star_gdl_path(), "ruta_a_star.shp"))

    ax.legend()
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
    ax.set_axis_off()
    plt.show()

def reconstruct_graph_from_shp():


    nodes = gpd.read_file(workspace.get_qgis_gdl_nodes_path())
    edges = gpd.read_file(workspace.get_qgis_gdl_edges_path())

    G = ox.graph_from_gdfs(nodes, edges)
    return G

def reconstruct_graph_from_graphml(graphml_path):
    G = ox.load_graphml(graphml_path)
    return G

def save_shp_files_from_graph(G):
    # convertir a GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    os.makedirs(workspace.get_qgis_gdl_shp_path(), exist_ok=True)
    
    # guardar shapefiles
    nodes.to_file(workspace.get_qgis_gdl_nodes_path())
    edges.to_file(workspace.get_qgis_gdl_edges_path())
    # guardarlo en formato networkx
    ox.save_graphml(G, workspace.get_graphml_gdl_path())

def set_max_speed_weight(G, speed_kmh = 100):
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
            
    return G

def main():
    # verify if graph exists
    if(True == os.path.exists(workspace.get_graphml_gdl_path())):
        print("Reconstructing path from shape files: %s" % workspace.get_qgis_gdl_shp_path())
        G = reconstruct_graph_from_graphml(workspace.get_graphml_gdl_path())
    else:
        print("Loading graph from Guadalajara using OSMX")
        G = ox.graph_from_place("Guadalajara, Mexico", network_type="drive")
        # mapping node elevations to graph
        print("Adding Elevations to Graph of Guadalajara using OSMX")
        ox.add_node_elevations_google(G, api_key='AIzaSyAM3AJEapQcpVRglfgmg7hw8o9VSuS0p8I')  
        print("Saving shape files")
        save_shp_files_from_graph(G)
    
    set_max_speed_weight(G, 100)
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

    algorithm_used = "A_Star_Manhattan"

    if(algorithm_used == "Djikstra"):
        print("Executing Dijkstra ...")
        dist, prev = dijkstra(G, start_node, weight="travel_time")
        path = reconstruct_route(prev, start_node, end_node)
    elif(algorithm_used == "A_Star_Manhattan"):
        print("Executing A Star...")
        dist, prev = a_star(G, start_node, end_node, distance_manhattan, weight="travel_time")
        path = reconstruct_route(prev, start_node, end_node)

    if path:
        print("Tiempo total (segundos):", dist[end_node])
        print("Ruta encontrada con", len(path), "nodos")
        plot_route("djikstra", G, path)
    else:
        print("No se encontró ruta")


if __name__ == "__main__":
    main()
