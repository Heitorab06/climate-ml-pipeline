URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

LAT = -12.9822499
LONG = -38.4812772
FEATURES = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain",
            "pressure_msl", "cloud_cover", "wind_speed_10m", 
            "wind_direction_10m", "wind_gusts_10m"
            ]


PARAMS = {
    "latitude": LAT,
    "longitude": LONG,
    "start_date": "2023-01-01",
	"end_date": "2025-12-31",
    "hourly": FEATURES
}

METRICS_FILE = "reports/metrics.csv"