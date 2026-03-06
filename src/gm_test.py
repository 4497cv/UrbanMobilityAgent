import globalmapper
print("Loaded from:", globalmapper.__file__)
print("Available attributes:", dir(globalmapper))


gm.GM_Initialize()

# Load DEM file (must cover your coordinates)
gm.GM_LoadLayer(r"C:\data\guadalajara_dem.tif")

# Coordinates (Longitude, Latitude) - example in Guadalajara
lon = -103.3496
lat = 20.6597

# Query elevation
elev = gm.GM_GetElevation(lat, lon)

print("Elevation (meters):", elev)

gm.GM_Terminate()