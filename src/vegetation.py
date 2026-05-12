import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
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

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

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

def calcular_indice_veg(arista_geom, veg_tree, veg_geoms, radio=profiles.VEG_RADIO_INFLUENCIA_M):
    """
    Calcula un índice de vegetación [0, 1] para una arista del grafo.

    Usa STRtree para encontrar la sub-geometría más cercana a cada punto de muestra
    en O(log n) en vez de O(n) sobre todo el MultiPolygon.
    """
    if arista_geom is None or arista_geom.is_empty or arista_geom.length == 0:
        return 0.0

    num_muestras  = max(int(arista_geom.length / 10), 2)
    puntos        = shapely.line_interpolate_point(arista_geom, np.linspace(0, 1, num_muestras), normalized=True)
    nearest_idx   = veg_tree.nearest(puntos)
    distancias    = shapely.distance(puntos, veg_geoms[nearest_idx])

    pesos  = np.where(distancias <= radio, 1.0 - distancias / radio, 0.0)
    return round(float(min(pesos.sum() / num_muestras, 1.0)), 4)


def _calcular_indices_veg_gpu(aristas_proj, vegetacion_union, radio):
    """
    Misma lógica que calcular_indice_veg pero vectorizada para todas las aristas:
    STRtree.nearest + shapely.distance en batch, pesos en GPU.
    """
    veg_geoms = (np.array(list(vegetacion_union.geoms), dtype=object)
                 if hasattr(vegetacion_union, "geoms")
                 else np.array([vegetacion_union], dtype=object))
    veg_tree = shapely.STRtree(veg_geoms)

    # Generar todos los puntos de muestra para todas las aristas
    all_pts    = []
    boundaries = []
    idx        = 0

    for geom in aristas_proj.geometry:
        if geom is None or geom.is_empty or geom.length == 0:
            boundaries.append((idx, idx))
            continue
        n   = max(int(geom.length / 10), 2)
        pts = shapely.line_interpolate_point(geom, np.linspace(0, 1, n), normalized=True)
        all_pts.append(pts)
        boundaries.append((idx, idx + n))
        idx += n

    if not all_pts:
        return [0.0] * len(boundaries)

    all_pts_flat = np.concatenate(all_pts)
    print(f"    {len(all_pts_flat)} puntos de muestra para {len(aristas_proj)} aristas")

    # Distancias con STRtree — igual que calcular_indice_veg, maneja interior (dist=0)
    nearest_idx = veg_tree.nearest(all_pts_flat)
    distancias  = shapely.distance(all_pts_flat, veg_geoms[nearest_idx]).astype(np.float32)

    # Pesos en GPU
    radio_f  = np.float32(radio)
    dist_gpu = cp.asarray(distancias)
    pesos    = cp.asnumpy(cp.where(dist_gpu <= radio_f, 1.0 - dist_gpu / radio_f, cp.float32(0.0)))

    # Agregar por arista
    indices = []
    for start_i, end_i in boundaries:
        if start_i == end_i:
            indices.append(0.0)
            continue
        w = pesos[start_i:end_i]
        indices.append(round(float(min(w.sum() / (end_i - start_i), 1.0)), 4))

    return indices


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


    # STRtree sobre sub-geometrías: O(log n) por consulta en vez de O(n)
    veg_geoms = (np.array(list(vegetacion_union.geoms), dtype=object)
                 if hasattr(vegetacion_union, "geoms")
                 else np.array([vegetacion_union], dtype=object))
    veg_tree  = shapely.STRtree(veg_geoms)

    if(user.get_processing_mode() == profiles.MODE_CPU):
        print(f"  Calculando índice de vegetación para {total} aristas (Paralelo en CPU)...")
        # threading: comparte veg_tree sin pickle (STRtree es thread-safe en Shapely 2.x)
        indices_vegetacion = Parallel(n_jobs=user.get_cpu_threads(), backend="threading")(
            delayed(calcular_indice_veg)(arista.geometry, veg_tree, veg_geoms)
            for _, arista in tqdm(aristas_proj.iterrows(), total=total, desc="Vegetación CPU")
        )
    elif(user.get_processing_mode() == profiles.MODE_NORMAL):
        print(f"  Calculando índice de vegetación para {total} aristas (Sequential)...")
        indices_vegetacion = []
        for _, arista in tqdm(aristas_proj.iterrows(), total=total, desc="Vegetación Secuencial"):
            indices_vegetacion.append(calcular_indice_veg(arista.geometry, veg_tree, veg_geoms))
    elif(user.get_processing_mode() == profiles.MODE_GPU):
        if not _CUPY_AVAILABLE:
            sys.exit("CuPy no está instalado. Instálalo con: pip install cupy-cuda12x")
        print(f"  Calculando índice de vegetación para {total} aristas (GPU con CuPy)...")
        indices_vegetacion = _calcular_indices_veg_gpu(
            aristas_proj, vegetacion_union, profiles.VEG_RADIO_INFLUENCIA_M
        )
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

    user.set_processing_mode(profiles.MODE_CPU)

    # comenzar procesamiento de los indices de vegetación
    procesar_inidices_veg(G, 
                          user,
                          aristas_proj,
                          areas_verdes_proj,
                          aristas_gdf)

    # guardar geometrías de areas verdes en formato shp
    areas_verdes.to_file(workspace.get_areas_verdes_shp_path())
        