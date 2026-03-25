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
import time
import numpy as np

def toblers_hiking_function(slope) -> float:
    """
        Tobler's hiking function is an exponential function determining the hiking speed, taking into account the slope angle.

        Parameters:
        - slope (float): it corresponds to the absolute value of the difference in elevation of two different points.

        Returns:
        - walking velocity [km/h] (float): it is the walking velocity of a hiker, considering the slope into account.
    """
    walking_velocity = 6 * np.exp(-3.5 * abs(slope + 0.05))

    return walking_velocity

def calculate_slope(elevation_x1:float, elevation_x2:float, distance:float) -> float:
    """
        Function to calculate the slope between two different elevation points

        Parameters:
        - elevation_x1 (float): elevation value of the first node.
        - elevation_x2 (float): elevation value of the second node.
        - distance [km](float): total distance between the two different points

        Returns:
        - slope: angle of inclination to the horizontal
    """
    return (elevation_x2 - elevation_x1) / distance    

def distance_manhattan(p1, p2):
    """
        Function to calculate the manhattan distance between two different points. 
        The Manhattan distance is a way to measure distance between two points in a grid by only moving horizontally and vertically.

        Parameters:
        - p1 (float): elevation value of the first node.
        - p2 (float): elevation value of the second node.

        Returns:
        - distance (float): manhattan distance between two points (grid movement)
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def calculate_toblers_time(elevation_x1, elevation_x2, distance):
    """
        Tobler's hiking function is an exponential function determining the hiking speed, taking into account the slope angle.
    
        Parameters:
        - elevation_x1 [km](float): elevation value of the first node.
        - elevation_x2 [km](float): elevation value of the second node.
        - distance [m](float): total distance between the two different points

        Returns:
        - tobblers_time [m/s]: angle of inclination to the horizontal
    """
    tobblers_time = 0
    # calculate the slope between the two elevation points
    slope = calculate_slope(elevation_x1, elevation_x2, distance)
    # calculate the total walking speed
    speed_kmh = toblers_hiking_function(slope)
    # convert the speed to ms
    speed_ms = speed_kmh * 1000 / 3600

    if speed_ms == 0:
        tobblers_time = float("inf")
    else:
        tobblers_time = distance / speed_ms

    return tobblers_time

def distance_euclidean(p1, p2):
    """
        Function to calculate the euclidean distance between two different points. 
        The Euclidean distance is the straight-line distance between two points in space.
        
        Parameters:
        - p1 (float): elevation value of the first node.
        - p2 (float): elevation value of the second node.

        Returns:
        - distance (float): manhattan distance between two points (grid movement)
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

def a_star(G, start, end, heuristic, weight_d):
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
            edge_dt = G.get_edge_data(current, neighbor)
            edge_data = edge_dt.get(0, {})

            if(True == workspace.get_elevation_flag()):
                weight = edge_data.get('ele_diff', 1)

                # Check if 'ele_diff' is missing
                if 'ele_diff' not in edge_data:
                    print(f"Missing 'ele_diff' in edge between {current} and {neighbor}: {edge_dt}")            #weight = edge.get('ele_diff', 1)  # Default to 1 if no weight is specified
                    print(weight)
                    sys.exit()
            else:
                weight = 0

            g = float(distance[current]) + float(weight)  # Actual cost from start to neighbor
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
    """
    This function reconstructs the route with the solution, with a graph given

    Parameters:
    - prev:
    - start:
    - end:
    
    Returns:
    - path: list of nodes that correspond that connect the start node to the end node
    """
    path = []
    # start in the end node
    current = end
    while(current is not None):
        # we insert the current node from finish to start
        path.append(current)
        # we get the reference to the previous node
        current = prev[current]
    path.reverse()

    if(len(path) > 0) and (path[0] == start):
        return path
    
    return []


def plot_route(algorithm_used, G, route, local_plot = False):
    """
    This function loads the graph from a given networkx graph, 
    saves the nodes and edges information from the graph; when
    local_plot is activated the graph can be shown locally.

    Parameters:
    - algorithm_used (str): string that indicates the algorithm that is executed.
    - G (networkx graph): network X graph.
    - route (list of networkx nodes): list of nodes containing the path from the start to end.
    - local_plot (bool): flag to indicate if the graph will be plotted locally.s
    
    Returns:
    - None
    """
    # load nodes and edges from the network x graph G
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
    
    if("Djikstra" == algorithm_used):
        if(workspace.get_elevation_flag() == True):
            route_gdf.to_file(os.path.join(workspace.get_route_djikstra_gdl_path(), "ruta_dijkstra_Elevation.shp"))
        else:
            route_gdf.to_file(os.path.join(workspace.get_route_djikstra_gdl_path(), "ruta_dijkstra.shp"))
    elif("A_Star_Manhattan" == algorithm_used):
        if(workspace.get_elevation_flag() == True):
            route_gdf.to_file(workspace.get_a_star_manhattan_ele_shp())
        else:
            route_gdf.to_file(workspace.get_a_star_manhattan_shp())
    elif("A_Star_Euclidean" == algorithm_used):
        if(workspace.get_elevation_flag() == True):
            route_gdf.to_file(workspace.get_a_star_euclidean_ele_shp())
        else:
            route_gdf.to_file(workspace.get_a_star_euclidean_shp())
    else:
        print("Algorithm not found %s" % algorithm_used)
        sys.exit()

    if(local_plot == True):
        ax.legend()
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
        ax.set_axis_off()
        plt.show()

def reconstruct_graph_from_shp():
    """
    This function loads the nodes and edges from the .shp files
    and reconstructs a graph based in these files. This newly
    created graph does not include the nodes information since,
    the shp file only contains the information about the trace.
    If this method is used it would be necesssary to remap, all
    the attributes again to the new networkx graph.

    Parameters:
    - algorithm_used (str): string that indicates the algorithm that is executed.
    - G (networkx graph): network X graph.
    - route (list of networkx nodes): list of nodes containing the path from the start to end.
    - local_plot (bool): flag to indicate if the graph will be plotted locally.s
    
    Returns:
    - G (networkx graph): networkx graph 
    """
    nodes = gpd.read_file(workspace.get_qgis_gdl_nodes_path())
    edges = gpd.read_file(workspace.get_qgis_gdl_edges_path())

    G = ox.graph_from_gdfs(nodes, edges)
    return G

def reconstruct_graph_from_graphml(graphml_path):
    """
    This function reconstructs a networkx graph based in a graphml file.
    The graphml format retains all the nodes attributes that were present 
    when the graph had been stored.

    Parameters:
    - graphml_path (str): path pointing to the location of the graphml file.
    Returns:
    - G (networkx graph): network X graph.
    """
    G = ox.load_graphml(graphml_path)

    return G

def save_shp_files_from_graph(G):
    """
    This function generates the shp files of a given networkx graph and
    stores the graph information in a graphml format.

    Parameters:
    - G (networkx graph): network X graph.
    Returns:
    - None
    """
    # convertir a GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    os.makedirs(workspace.get_qgis_gdl_shp_path(), exist_ok=True)
    
    # guardar shapefiles
    nodes.to_file(workspace.get_qgis_gdl_nodes_path())
    edges.to_file(workspace.get_qgis_gdl_edges_path())
    # guardarlo en formato networkx
    ox.save_graphml(G, workspace.get_graphml_gdl_path())

def set_max_speed_weight(G, speed_kmh = 60):
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" in data:
            speed_kmh = 50  # valor por defecto
            if "maxspeed" in data:
                try:
                    speed_kmh = float(str(data["maxspeed"]).split()[0])
                except:
                    pass
            speed_ms = speed_kmh * 1000 / 3600
            data["trv_time"] = data["length"] / speed_ms
            
    return G

def set_elevation_weight(G, network_type):

    if(network_type == "walk"):
        workspace.set_elevation_flag(True)
        print("> Elevation is activated")

        for u, v, k, data in G.edges(keys=True, data=True):
            # Get elevations of the start and end nodes
            elevation_u = G.nodes[u].get('elevation', 0)  # Default to 0 if no elevation
            elevation_v = G.nodes[v].get('elevation', 0)  # Default to 0 if no elevation

            # Use the difference in elevation as the edge weight
            elevation_difference = abs(elevation_u - elevation_v)
            
            data['ele_diff'] = calculate_toblers_time(elevation_u, elevation_v, data["length"])

        ox.save_graphml(G, workspace.get_graphml_gdl_path())