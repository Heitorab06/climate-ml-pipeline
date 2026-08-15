import pandas as pd
from src.extraction.api import get_weather_data

df = pd.read_parquet("data/weather_raw.parquet")

print(df.head())