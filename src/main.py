from urban_mobility import *
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import os
import workspace
import profiles

def __main__(algorithm_used, elevation):

    print("\nExecuting Algorithm for %s" %algorithm_used)

    user = profiles.UserProfile("network_type")

    # verify if graph existss
    if(True == os.path.exists(workspace.get_graphml_gdl_path())):
        print("Reconstructing path from shape files: %s" % workspace.get_qgis_gdl_shp_path())
        G = reconstruct_graph_from_graphml(workspace.get_graphml_gdl_path())
    else:
        print("Loading graph from Guadalajara using OSMX")
        G = ox.graph_from_place("Guadalajara, Mexico", network_type=user.get_network_type())
        # mapping node elevations to graph
        print("Adding Elevations to Graph of Guadalajara using OSMX")
        ox.add_node_elevations_google(G, api_key=workspace.gg_key())  
        print("Saving shape files")
        save_shp_files_from_graph(G)
    
    if(elevation == True):
        set_elevation_weight(G)
        workspace.set_elevation_flag(True)
    else:
        workspace.set_elevation_flag(False)

    
    #set_max_speed_weight(G, 80)
    
    start_lon, start_lat = -103.376624,20.630163
    # destination coordinates
    end_lat, end_lon = 20.697814, -103.384384

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    print("Number of nodes:", len(G.nodes))
    print("Number de edges:", len(G.edges))

    print("Start Node:", start_node)
    print("Destination Node:", end_node)

    if start_node == end_node:
        print("Start and destination node have the same coordinates or are to close to each other.")
        return

    if not nx.has_path(G, start_node, end_node):
        print("There is no conection between the start and end node")
        return

    if(algorithm_used == "Djikstra"):
        print("Executing Dijkstra ...")
        dist, prev = dijkstra(G, start_node, weight="elevation")
        path = reconstruct_route(prev, start_node, end_node)
    elif(algorithm_used == "A_Star_Manhattan"):
        print("Executing A Star Manhattan...")
        dist, prev = a_star(G, start_node, end_node, distance_manhattan, weight_d="elevation")
        path = reconstruct_route(prev, start_node, end_node)
    elif(algorithm_used == "A_Star_Euclidean"):
        print("Executing A Star Euclidean...")
        dist, prev = a_star(G, start_node, end_node, distance_euclidean, weight_d="elevation")
        path = reconstruct_route(prev, start_node, end_node)

    if path:
        print("Total time (sec):", dist[end_node])
        print("Route found with", len(path), "nodes")
        plot_route(algorithm_used, G, path)
    else:
        print("Route not found")


if __name__ == "__main__":
    #__main__("Djikstra", True)
    #__main__("A_Star_Manhattan", True)
    __main__("A_Star_Euclidean", True)

    #__main__("Djikstra", False)
    #__main__("A_Star_Manhattan", False)
    __main__("A_Star_Euclidean", False)
  