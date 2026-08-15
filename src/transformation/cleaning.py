import pandas as pd
from src.extraction.api import get_weather_data
from datetime import datetime

df = pd.read_parquet("data/weather_raw.parquet")

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    format_date = "%Y-%m-%dT%H:%M"
    
    df["time"] = pd.to_datetime(df["time"], format=format_date)
         
    return df


df_clean = convert_data_types(df=df)
df_clean.info()