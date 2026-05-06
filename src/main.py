from urban_mobility import reconstruct_graph_from_graphml, set_elevation_weight, plot_route
import osmnx as ox
import networkx as nx
import os
import workspace
import profiles
import moad
import nsga2
import nsga3
import vegetation
import elevation

def __main__(algorithm_used):

    print("\nExecuting Algorithm for %s" % algorithm_used)

    user = profiles.UserProfile("drive", "ZMG",
                                w_time=0, w_elev=1, w_veg=0.5)

    user.set_start_coordinates()
    user.set_end_coordinates()

    if os.path.exists(workspace.get_graphml_gdl_path()):
        print("Reconstructing path from graphml file: %s" % workspace.get_qgis_gdl_shp_path())
        G = reconstruct_graph_from_graphml(workspace.get_graphml_gdl_path())

        edges_data = [d for _, _, d in G.edges(data=True)]
        if not all('ele_diff' in d for d in edges_data):
            print("Missing 'ele_diff' in edges — running elevation...")
            elevation.run(G, user)
        else:
            print("'ele_diff' OK")

        if not all('indice_veg' in d for d in edges_data):
            print("Missing 'indice_veg' in edges — running vegetation...")
            vegetation.run(G, user)
        else:
            print("'indice_veg' OK")
    else:
        print("Downloading graph for ZMG (%d municipios) using OSMX" % len(user.get_place()))
        G = ox.graph_from_place(user.get_place(), network_type=user.network_type)

        if user.elevation_active:
            print("Adding 'ele_diff' in edges — running elevation...")
            elevation.run(G, user)

        if user.vegetation_active:
            print("Adding 'indice_veg' in edges — running vegetation...")
            vegetation.run(G, user)

    start_node = ox.distance.nearest_nodes(G, user.start_coordinates.x, user.start_coordinates.y)
    end_node   = ox.distance.nearest_nodes(G, user.end_coordinates.x,   user.end_coordinates.y)

    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))
    print("Start Node:", start_node)
    print("Destination Node:", end_node)

    if start_node == end_node:
        print("Start and destination node have the same coordinates or are too close to each other.")
        return G, user

    if not nx.has_path(G, start_node, end_node):
        print("There is no connection between the start and end node")
        return G, user

    if algorithm_used == "weighted_astar":
        moad.run(G, user)
    elif algorithm_used == "nsga2":
        nsga2.run_nsga2(G, user)
    elif algorithm_used == "nsga3":
        nsga3.run_nsga3(G, user)

    return G, user


if __name__ == "__main__":
    #__main__("weighted_astar")
    __main__("nsga2")
    #__main__("nsga3")

    #workspace.open_qgis_project()
