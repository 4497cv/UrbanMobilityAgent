"""
Indice de inseguridad (SI) por coordenadas y hora, basado en doc/Prediccion_SI_Coordenadas.ipynb.

- La ciudad se divide en celdas de 500 m y el dia en 24 horas.
- SI calculado por celda-hora: N[ALPHA * violentos + (1 - ALPHA) * robos], 0-100.
- Un HistGradientBoosting (perdida Poisson) aprende el SI a partir de (lat, lon, hora)
  y suaviza el ruido de celdas con pocos delitos.
- SI final = PESO_CALCULADO * calculado + (1 - PESO_CALCULADO) * predicho.

run(G, user) agrega a cada arista 'indice_inseg' en [0, 1] (SI final / 100) para la hora del usuario.
"""
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import workspace

ALPHA = 0.7            # peso de delitos violentos vs robos
PERCENTIL = 0.99       # tope de la normalizacion N
TAM_CELDA = 0.5        # km
PESO_CALCULADO = 0.3   # mezcla calculado/predicho (mejor R2 contra 2024-2026 en el notebook)

# proyeccion local a km (a 20.6 grados de latitud): suficiente para una ciudad
LAT0, LON0 = 20.6, -103.4
KM_POR_GRADO_LAT, KM_POR_GRADO_LON = 110.7, 104.2

DELITOS_VIOLENTOS = ["Violencia familiar", "Lesiones dolosas", "Abuso sexual infantil",
                     "Homicidio doloso", "Violacion", "Feminicidio"]

_modelo = None  # cache en memoria: {"hgb": modelo, "si_calculado": Series indexada por (ix, iy, hora)}


def _celda(lat, lon):
    ix = np.floor((np.asarray(lon) - LON0) * KM_POR_GRADO_LON / TAM_CELDA).astype(int)
    iy = np.floor((np.asarray(lat) - LAT0) * KM_POR_GRADO_LAT / TAM_CELDA).astype(int)
    return ix, iy


def _features(lat, lon, hora):
    hora = np.asarray(hora) % 24
    return pd.DataFrame({"lat": lat, "lon": lon, "hora": hora,
                         "hora_sin": np.sin(2 * np.pi * hora / 24),
                         "hora_cos": np.cos(2 * np.pi * hora / 24)})


def entrenar(csv_path=None):
    """Construye la malla celda-hora, calcula el SI y entrena el modelo con todos los anios."""
    csv_path = csv_path or workspace.get_crime_population_csv_path()
    print("Entrenando modelo de inseguridad con %s" % csv_path)

    d = pd.read_csv(csv_path, usecols=["delito", "x", "y", "hora"])
    d = d.dropna(subset=["x", "y", "hora"])
    d["hora"] = d["hora"].astype(int)
    d["n_violence"] = d["delito"].isin(DELITOS_VIOLENTOS).astype(int)
    d["n_property"] = d["delito"].str.startswith("Robo").astype(int)
    d["ix"], d["iy"] = _celda(d["y"], d["x"])

    # malla: celdas con al menos un delito x 24 horas; las horas sin delitos entran en cero
    celdas = d[["ix", "iy"]].drop_duplicates()
    malla = celdas.merge(pd.DataFrame({"hora": range(24)}), how="cross")
    malla["lon"] = LON0 + (malla["ix"] + 0.5) * TAM_CELDA / KM_POR_GRADO_LON
    malla["lat"] = LAT0 + (malla["iy"] + 0.5) * TAM_CELDA / KM_POR_GRADO_LAT

    c = d.groupby(["ix", "iy", "hora"])[["n_violence", "n_property"]].sum()
    c = c.reindex(pd.MultiIndex.from_frame(malla[["ix", "iy", "hora"]]), fill_value=0)
    x = ALPHA * c["n_violence"].to_numpy() + (1 - ALPHA) * c["n_property"].to_numpy()
    tope = np.quantile(x, PERCENTIL)
    malla["SI"] = 100 * np.clip(x, None, tope) / tope

    # perdida Poisson: el SI viene de conteos de delitos (muchos ceros, sin negativos)
    hgb = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05,
                                        min_samples_leaf=40, random_state=0)
    hgb.fit(_features(malla["lat"], malla["lon"], malla["hora"]), malla["SI"])

    print("Modelo de inseguridad: %d celdas x 24 horas" % len(celdas))
    return {"hgb": hgb, "si_calculado": malla.set_index(["ix", "iy", "hora"])["SI"]}


def cargar_modelo(reentrenar=False):
    """Carga el modelo del cache en disco; si no existe (o reentrenar=True) lo entrena y lo guarda."""
    global _modelo
    if _modelo is not None and not reentrenar:
        return _modelo

    path = workspace.get_insecurity_model_path()
    if os.path.exists(path) and not reentrenar:
        _modelo = joblib.load(path)
    else:
        _modelo = entrenar()
        joblib.dump(_modelo, path)
        print("Modelo de inseguridad guardado en %s" % path)
    return _modelo


def si_en(lat, lon, hora):
    """
    SI final (0-100) para uno o varios puntos a la hora dada (0-23).
    lat, lon pueden ser escalares o arreglos; hora escalar o arreglo del mismo tamanio.
    Un punto fuera de la huella urbana usa 0 como SI calculado y solo la prediccion del modelo.
    """
    modelo = cargar_modelo()
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    hora = np.broadcast_to(np.asarray(hora, dtype=int) % 24, lat.shape)

    predicho = np.clip(modelo["hgb"].predict(_features(lat, lon, hora)), 0, 100)
    ix, iy = _celda(lat, lon)
    claves = pd.MultiIndex.from_arrays([ix, iy, hora])
    calculado = modelo["si_calculado"].reindex(claves, fill_value=0.0).to_numpy()

    si = PESO_CALCULADO * calculado + (1 - PESO_CALCULADO) * predicho
    return si if si.size > 1 else float(si[0])


def run(G, user, hora=None):
    """Asigna 'indice_inseg' (0 = seguro, 1 = maximo riesgo) a cada arista, evaluado en su punto medio."""
    if hora is None:
        hora = getattr(user, "hour", None)
    if hora is None:
        hora = datetime.now().hour
    print("Calculando indice de inseguridad para las %02d:00" % hora)

    aristas = list(G.edges(keys=True))
    lat = np.array([(G.nodes[u]["y"] + G.nodes[v]["y"]) / 2 for u, v, _ in aristas])
    lon = np.array([(G.nodes[u]["x"] + G.nodes[v]["x"]) / 2 for u, v, _ in aristas])

    si = np.atleast_1d(si_en(lat, lon, hora)) / 100.0
    for (u, v, k), valor in zip(aristas, si):
        G.edges[u, v, k]["indice_inseg"] = float(valor)

    G.graph["hora_inseg"] = int(hora)
    return G
