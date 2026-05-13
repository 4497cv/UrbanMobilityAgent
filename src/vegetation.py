import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import warnings
import urban_mobility
import workspace
from joblib import Parallel, delayed
import profiles
import sys
from tqdm import tqdm

tags_vegetacion = {
    "leisure": ["park", "garden", "nature_reserve", "forest"],
    "landuse": ["forest", "grass"],
    "natural": ["grassland", "tree_row"],
}

def _to_polygon(gdf, buffer_m=3):
    """Buffer LineString/MultiLineString geometries to thin polygons (metric CRS assumed)."""
    mask = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    if mask.any():
        gdf = gdf.copy()
        gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].buffer(buffer_m)
    return gdf

def descargar_areas_verdes(lugar):
    # lista de areas verdes
    areas_verdes_lista = []

    # iterar en los tags de vegetaciones de osmnx
    for tag_key, tag_values in tags_vegetacion.items():
        try:
            # obtener geometria de los diferentes tipos de vegetacion usando osmnx
            gdf = ox.features_from_place(lugar, tags={tag_key: tag_values})
            # verificar que la geometria descargada no se encuentre vacia
            if not gdf.empty:
                gdf = gdf[gdf.geometry.type.isin(
                    ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
                )]
                # valida que la geometria sea correcta
                if not gdf.empty:
                    # Convert LineStrings to thin polygons for shapefile compatibility
                    gdf_proj = gdf[["geometry"]].to_crs(epsg=6372)
                    gdf_proj = _to_polygon(gdf_proj)
                    gdf_wgs = gdf_proj.to_crs(epsg=4326)
                    areas_verdes_lista.append(gdf_wgs)
                    print(f"  {tag_key}: {len(gdf)} elementos encontrados")
        except Exception as e:
            print(f"  {tag_key}: No se encontraron datos ({e})")

    # También buscar árboles individuales (puntos)
    try:
        arboles = ox.features_from_place(lugar, tags={"natural": "tree"})
        if not arboles.empty:
            # Crear un buffer de 5m alrededor de cada árbol
            arboles_proj = arboles.to_crs(epsg=6372)  # Proyección para México
            arboles_proj["geometry"] = arboles_proj.geometry.buffer(5)
            arboles_buffer = arboles_proj.to_crs(epsg=4326)
            areas_verdes_lista.append(arboles_buffer[["geometry"]])
            print(f"  Árboles individuales: {len(arboles)} encontrados")
    except Exception:
        print("  Árboles individuales: No se encontraron")

    # Combinar todas las áreas verdes en un solo GeoDataFrame
    if areas_verdes_lista:
        areas_verdes = gpd.GeoDataFrame(
            pd.concat(areas_verdes_lista, ignore_index=True),
            crs="EPSG:4326"
        )
        print(f"\nTotal de elementos de vegetación: {len(areas_verdes)}")
    else:
        raise ValueError("No se encontraron áreas verdes. Verifica la zona de estudio.")
    return areas_verdes

def calcular_indice_veg(arista_geom, vegetacion_union, radio=profiles.VEG_RADIO_INFLUENCIA_M):
    """
    Calcula un índice de vegetación [0, 1] para una arista del grafo.

    Método: Proporción de la longitud de la arista que tiene vegetación
    dentro del radio de influencia.

    Parámetros:
    -----------
    arista_geom : shapely geometry
        Geometría de la arista (LineString)
    vegetacion_union : shapely geometry
        Unión de todas las áreas verdes
    radio : float
        Radio de influencia en metros

    Retorna:
    --------
    float : Índice entre 0 (sin vegetación) y 1 (totalmente rodeada de vegetación)
    """
    # verifica que la geometría de la arista sea válida y que sea mayor que cero
    if((arista_geom is None) or\
      (arista_geom.is_empty) or \
      (arista_geom.length == 0)):
        return 0.0

    # Muestrear puntos a lo largo de la arista cada 10 metros
    num_muestras = max(int(arista_geom.length / 10), 2)
    # separar el número de muestras en fracciones
    fracciones = np.linspace(0, 1, num_muestras)

    puntos_cerca_verde = 0

    for frac_arista in fracciones:
        # verificar si hubo una intersección entre la geometría de la arista y la geometria de la vegetacion
        punto = arista_geom.interpolate(frac_arista, normalized=True)
        # obtenemos la distancia entre el área verde y el punto fraccionado de la arista
        distancia_a_verde = vegetacion_union.distance(punto)

        # si la distancia es mayor al radio el punto se ignora
        if(distancia_a_verde <= radio):
            # Peso inversamente proporcional a la distancia
            puntos_cerca_verde += 1 - (distancia_a_verde / radio)

    # calculamos la proporcion de la arista que tiene vegetacion
    indice = puntos_cerca_verde / num_muestras

    return round(min(indice, 1.0), 4)

def calcular_indice_veg_gpu(arista_geom, vegetacion_union, radio=profiles.VEG_RADIO_INFLUENCIA_M):
    if arista_geom is None or arista_geom.is_empty or arista_geom.length == 0:
        return 0.0

    num_muestras = max(int(arista_geom.length / 10), 2)
    fracciones = np.linspace(0, 1, num_muestras)

    distancias = np.array([
        vegetacion_union.distance(arista_geom.interpolate(f, normalized=True))
        for f in fracciones
    ])

    distancias_gpu = cp.asarray(distancias)
    mascara = distancias_gpu <= radio
    pesos = cp.where(mascara, 1.0 - (distancias_gpu / radio), 0.0)
    indice = float(pesos.sum() / num_muestras)

    return round(min(indice, 1.0), 4)
	
def procesar_inidices_veg(G, user, aristas_proj, areas_verdes_proj, aristas_gdf):
    """
    Procesar los indices de vegetación para cada una de las aristas. Se obtiene la proporción de la longitud de la arista que tiene vegetación
    dentro del radio de influencia.

    Parámetros:
    -----------
    G (networkx graph): Grafo en formato de networkx
    user (UserProfile): contiene la información del perfil de usuario
    aristas_proj: 
    areas_verdes_proj:
    aristas_gdf:
    processing_mode: modo de procesamiento del GPU
    --------
    float : Índice entre 0 (sin vegetación) y 1 (totalmente rodeada de vegetación)
    """
    vegetacion_union = unary_union(areas_verdes_proj.geometry)
    total = len(aristas_proj)

    if(user.get_processing_mode() == profiles.MODE_CPU):
        print(f"  Calculando índice de vegetación para {total} aristas (Paralelo en CPU)...")
        indices_vegetacion = Parallel(n_jobs = user.get_cpu_threads())(
            delayed(calcular_indice_veg)(arista.geometry, vegetacion_union)
            # muestra el progreso del prosamiento de las aristas en una barra
            for _, arista in tqdm(aristas_proj.iterrows(), total=total, desc="Vegetación CPU")
        )
    elif(user.get_processing_mode() == profiles.MODE_NORMAL):
        print(f"  Calculando índice de vegetación para {total} aristas (Sequential)...")
        indices_vegetacion = []
        # muestra el progreso del prosamiento de las aristas en una barra
        for _, arista in tqdm(aristas_proj.iterrows(), total=total, desc="Vegetación Secuencial"):
            indices_vegetacion.append(calcular_indice_veg(arista.geometry, vegetacion_union))
    elif(user.get_processing_mode() == profiles.MODE_GPU):
        print(f"  Calculando índice de vegetación para {total} aristas (Paralelo en GPU)...")
		print(f"  Calculando índice de vegetación para {total} aristas (Sequential)...")
        indices_vegetacion = []
        # muestra el progreso del prosamiento de las aristas en una barra
        for _, arista in tqdm(aristas_proj.iterrows(), total=total, desc="Vegetación Secuencial"):
            indices_vegetacion.append(calcular_indice_veg_gpu(arista.geometry, vegetacion_union))
    else:
        sys.exit("Pefil de procesamiento no encontrado %s" % user.get_processing_mode())

    # pasa el calculo de los indices de vegetación para todas las aristas
    aristas_gdf["indice_veg"] = indices_vegetacion

    edge_attrs = {}
    for (u, v, key), row in aristas_gdf.iterrows():
        # se hace el mapeo de indice_veg a cada uno de las aristas
        edge_attrs[(u, v, key)] = {"indice_veg": row["indice_veg"]}

    # se la nueva lista de ejes, con los atributos agregados
    nx.set_edge_attributes(G, edge_attrs)
    # se guarda nuevamente el el grafo en formato de networkx
    urban_mobility.save_shp_files_from_graph(G)

def run(G, user):
    # obtener las aristas de nuestro grafo
    _, aristas_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    print("Descargando areas verdes de %s" % user.get_place())
    areas_verdes = descargar_areas_verdes(user.get_place())

    # Proyectar geometrías a sistema métrico (EPSG:6372 para México)
    aristas_proj = aristas_gdf.to_crs(epsg=6372)
    areas_verdes_proj = areas_verdes.to_crs(epsg=6372)

    # comenzar procesamiento de los indices de vegetación
    procesar_inidices_veg(G, 
                          user,
                          aristas_proj,
                          areas_verdes_proj,
                          aristas_gdf)

    # guardar geometrías de areas verdes en formato shp
    areas_verdes.to_file(workspace.get_areas_verdes_shp_path())
        
        