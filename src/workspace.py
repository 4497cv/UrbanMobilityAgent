import os
import sys

workspace_path = ""
route_path = ""
elevation_flag = False

def init_workspace_path():
    global workspace_path
    os.chdir("..")
    workspace_path = os.getcwd()

def set_workspace_path(path):
    global workspace_path
    workspace_path = path

def get_workspace_path():
    global workspace_path
    return workspace_path

def get_qgis_gdl_shp_path():
    return os.path.join(get_workspace_path(), "QGIS", "Graph", "Guadalajara")

def get_qgis_gdl_edges_path(edges_shp="edges.shp"):
    return os.path.join(get_qgis_gdl_shp_path(), edges_shp)

def get_qgis_gdl_nodes_path(nodes_shp="nodes.shp"):
    return os.path.join(get_qgis_gdl_shp_path(), nodes_shp)

def get_graphml_gdl_path(graphml="grafo_guadalajara.graphml"):
    return os.path.join(get_qgis_gdl_shp_path(), graphml)

def get_route_gdl_path():
    return os.path.join(get_workspace_path(), "QGIS", "Graph", "Guadalajara", "Route")

def get_route_djikstra_gdl_path():
    path = os.path.join(get_route_gdl_path(), "Djikstra")
    os.makedirs(path, exist_ok=True)
    return path

def get_route_a_star_gdl_path():
    path = os.path.join(get_route_gdl_path(), "A_Star")
    os.makedirs(path, exist_ok=True)
    return path

def get_a_star_manhattan_shp():
    path = os.path.join(get_route_a_star_gdl_path(), "ruta_a_star_manhattan.shp")
    return path

def get_a_star_manhattan_ele_shp():
    path = os.path.join(get_route_a_star_gdl_path(), "ruta_a_star_manhattan_Elevation.shp")
    return path

def get_a_star_euclidean_shp():
    path = os.path.join(get_route_a_star_gdl_path(), "ruta_a_star_euclidean.shp")
    return path

def get_a_star_euclidean_ele_shp():
    path = os.path.join(get_route_a_star_gdl_path(), "ruta_a_star_euclidean_Elevation.shp")
    return path

def gg_key():
    # todo: use encryption to hide key and store it cfg file
    return 'AIzaSyAM3AJEapQcpVRglfgmg7hw8o9VSuS0p8I'

def set_elevation_flag(value):
    global elevation_flag
    elevation_flag = value

def get_elevation_flag():
    global elevation_flag
    return elevation_flag

init_workspace_path()