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
from joblib import Parallel, delayed

tags_vegetacion = {
    "leisure": ["park", "garden", "nature_reserve", "forest"],
    "landuse": ["forest", "grass"],
    "natural": ["grassland", "tree_row"],
}

def descargar_areas_verdes(lugar):
    areas_verdes_lista = []
    for tag_key, tag_values in tags_vegetacion.items():
        try:
            gdf = ox.features_from_place(lugar, tags={tag_key: tag_values})
            if not gdf.empty:
                # Filtrar solo geometrías tipo Polygon/MultiPolygon y LineString
                gdf = gdf[gdf.geometry.type.isin(
                    ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
                )]
                if not gdf.empty:
                    areas_verdes_lista.append(gdf[["geometry"]])
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
    return areas_verdes, areas_verdes_lista

RADIO_INFLUENCIA_M = 50  # metros

def calcular_indice_veg(arista_geom, vegetacion_union, radio=RADIO_INFLUENCIA_M):
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
    if arista_geom is None or arista_geom.is_empty:
        return 0.0

    longitud_total = arista_geom.length
    if longitud_total == 0:
        return 0.0

    # Muestrear puntos a lo largo de la arista cada 10 metros
    num_muestras = max(int(longitud_total / 10), 2)
    fracciones = np.linspace(0, 1, num_muestras)

    puntos_cerca_verde = 0
    for frac in fracciones:
        punto = arista_geom.interpolate(frac, normalized=True)
        distancia_a_verde = vegetacion_union.distance(punto)
        if distancia_a_verde <= radio:
            # Peso inversamente proporcional a la distancia
            puntos_cerca_verde += 1 - (distancia_a_verde / radio)

    indice = puntos_cerca_verde / num_muestras
    return round(min(indice, 1.0), 4)

MODO_NORMAL = "normal"
MODO_CPU    = "cpu"

def procesar_inidices_veg(G, aristas_proj, areas_verdes_proj, aristas_gdf, modo=MODO_NORMAL):
    vegetacion_union = unary_union(areas_verdes_proj.geometry)
    total = len(aristas_proj)

    if modo == MODO_CPU:
        print(f"  Calculando índice de vegetación para {total} aristas (paralelo CPU)...")
        indices_vegetacion = Parallel(n_jobs=-1)(
            delayed(calcular_indice_veg)(row.geometry, vegetacion_union)
            for _, row in aristas_proj.iterrows()
        )
    else:
        print(f"  Calculando índice de vegetación para {total} aristas (secuencial)...")
        indices_vegetacion = []
        for i, (_, row) in enumerate(aristas_proj.iterrows()):
            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  Procesando arista {i + 1}/{total}...")
            indices_vegetacion.append(calcular_indice_veg(row.geometry, vegetacion_union))

    aristas_gdf["indice_veg"] = indices_vegetacion

    edge_attrs = {
        (u, v, key): {"indice_veg": row["indice_veg"]}
        for (u, v, key), row in aristas_gdf.iterrows()
    }
    nx.set_edge_attributes(G, edge_attrs)

    urban_mobility.save_shp_files_from_graph(G)

def run(G, user):

    lugar = user.place
    nodos_gdf, aristas_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    print("Descargando areas verdes de %s" % lugar)
    areas_verdes, areas_verdes_lista = descargar_areas_verdes(lugar)

    # Proyectar a sistema métrico para cálculos de distancia (EPSG:6372 para México)
    aristas_proj = aristas_gdf.to_crs(epsg=6372)
    areas_verdes_proj = areas_verdes.to_crs(epsg=6372)

    procesar_inidices_veg(G,
                          aristas_proj, 
                          areas_verdes_proj,
                          aristas_gdf,
                          modo=MODO_CPU)
        