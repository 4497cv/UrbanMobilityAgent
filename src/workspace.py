import os
import sys
import subprocess

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

def get_route_a_star_gdl_path():
    path = os.path.join(get_route_gdl_path(), "A_Star")
    os.makedirs(path, exist_ok=True)
    return path

def get_weighted_astar_shp(parameters):
    return os.path.join(get_route_a_star_gdl_path(), "ruta_weighted_astar" + parameters + ".shp")

def gg_key():
    # todo: use encryption to hide key and store it cfg file
    return 'AIzaSyAM3AJEapQcpVRglfgmg7hw8o9VSuS0p8I'

def set_elevation_flag(value):
    global elevation_flag
    elevation_flag = value

def get_elevation_flag():
    global elevation_flag
    return elevation_flag

def get_route_nsga2_gdl_path():
    path = os.path.join(get_route_gdl_path(), "NSGA2")
    os.makedirs(path, exist_ok=True)
    return path

def get_nsga2_shp(index=0):
    return os.path.join(get_route_nsga2_gdl_path(), f"ruta_nsga2_{index}.shp")

def get_route_nsga3_gdl_path():
    path = os.path.join(get_route_gdl_path(), "NSGA3")
    os.makedirs(path, exist_ok=True)
    return path

def get_nsga3_shp(index=0):
    return os.path.join(get_route_nsga3_gdl_path(), f"ruta_nsga3_{index}.shp")

def get_qgis_project_path():
    return os.path.join(get_workspace_path(), "QGIS", "Guadalajara_QGIS.qgz")

def get_qgis_setup_script_path():
    return os.path.join(get_workspace_path(), "QGIS", "setup_project.py")

def _find_qgis_exe():
    import glob as _glob
    patterns = [
        r"C:\Program Files\QGIS*\bin\qgis-bin.exe",
        r"C:\Program Files\QGIS*\bin\qgis-ltr-bin.exe",
        r"C:\OSGeo4W\bin\qgis-bin.exe",
        r"C:\OSGeo4W64\bin\qgis-bin.exe",
    ]
    for pattern in patterns:
        matches = _glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    return None

def open_qgis_project():
    setup_script = get_qgis_setup_script_path()
    qgis_exe = _find_qgis_exe()

    if qgis_exe and os.path.exists(setup_script):
        print(f"Opening QGIS: {qgis_exe}")
        subprocess.Popen([qgis_exe, "--code", setup_script])
        return

    # Fallback: open project file directly
    project_path = get_qgis_project_path()
    if os.path.exists(project_path):
        print(f"Opening QGIS project: {project_path}")
        os.startfile(project_path)
    else:
        print("QGIS executable not found and project file is missing.")
        print(f"Run QGIS/setup_project.py manually from the QGIS Python Console.")

def get_insecurity_path():
    path = os.path.join(get_workspace_path(), "src", "insecurity")
    return path


init_workspace_path()