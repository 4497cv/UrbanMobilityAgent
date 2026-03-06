import requests

lat = 20.6597
lon = -103.3496

url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
response = requests.get(url).json()

print(response["results"][0]["elevation"])