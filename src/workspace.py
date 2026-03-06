import os
import sys

workspace_path = ""

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



init_workspace_path()