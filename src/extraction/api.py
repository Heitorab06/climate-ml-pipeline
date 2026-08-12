import requests
import pandas

url = "https://archive-api.open-meteo.com/v1/archive"

lat = -12.9822499
long = -38.4812772
data = ["temperature_2m"]

params = {
    "latitude": lat,
    "longitude": long,
    "hourly": data
}

r = requests.get(url=url, params=params)

json = r.json()

for key in json:
    print(key)
