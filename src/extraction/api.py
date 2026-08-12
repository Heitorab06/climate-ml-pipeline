import requests
import pandas

url = "https://archive-api.open-meteo.com/v1/archive"

r = requests.get(url=url)
lat = -12.9822499
long = -38.4812772