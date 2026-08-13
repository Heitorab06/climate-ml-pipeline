import requests
import pandas as pd

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

lat = -12.9822499
long = -38.4812772
features = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain",
            "pressure_msl", "cloud_cover", "wind_speed_10m", 
            "wind_direction_10m", "wind_gusts_10m"
            ]


params = {
    "latitude": lat,
    "longitude": long,
    "start_date": "2023-01-01",
	"end_date": "2025-12-31",
    "hourly": features
}

r = requests.get(url=url, params=params)
json = r.json()

df = pd.DataFrame(json["hourly"])

print(df.shape)