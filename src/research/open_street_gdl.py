import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

points = [
    (20.69582239196953, -103.39611397651713),
    (20.687732284959427, -103.39334814155488),
    (20.6874458179614,  -103.37880310021951),
    (20.69620669072897, -103.37321474223175),
    (20.699644061239827, -103.37505201061083)
]

lats = [p[0] for p in points]
lons = [p[1] for p in points]

north = max(lats)
south = min(lats)
east  = max(lons)
west  = min(lons)

bbox = (north, south, east, west)
print("bbox:", bbox)

# Grafo de Providencia
G = ox.graph_from_bbox(
    bbox = bbox,
    network_type="walk"
)

nodes, edges = ox.graph_to_gdfs(G)

# Reproyectar a Web Mercator (necesario para mapas base)
nodes = nodes.to_crs(epsg=3857)
edges = edges.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))

# Calles (aristas)
edges.plot(ax=ax, linewidth=1, color="red", alpha=0.8)

# Intersecciones (nodos)
nodes.plot(ax=ax, color="blue", markersize=5)

# Mapa base real
ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik
)

ax.set_axis_off()
plt.show()