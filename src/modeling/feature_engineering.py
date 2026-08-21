import pandas as pd
import numpy as np
from src.loading.database import select

df = select("SELECT * FROM WEATHER")
print(df.columns)

def create_lag_features(df:pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        if col == "precipitation":
            df[f"cumulative_{col}_3h"] = df[col].rolling(3).sum()
        else: 
            df[f"{col}_variation_3h"] = df[col] - df[col].shift(3)
    
    return df


def create_time_cyclical_features(df:pd.DataFrame) -> pd.DataFrame:
    
    df["hour_sin"] = np.sin(np.pi * df["time"].dt.hour/24)
    df["hour_cos"] = np.cos(np.pi * df["time"].dt.hour/24)
    
    df["day_sin"] = np.sin(np.pi * df["time"].dt.dayofyear/365)
    df["day_cos"] = np.cos(np.pi * df["time"].dt.dayofyear/365)    
    
    return df

def create_target(df:pd.DataFrame) -> pd.DataFrame:
    
    return df

def run_feature_engineering(df:pd.DataFrame)-> pd.DataFrame:
    lag_features = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "precipitation"]
    
    return (df.copy()
            .pipe(create_lag_features, cols=lag_features, lags=[1,2,3,6])
            .pipe(create_time_cyclical_features)
            .pipe(create_target))
    
df = run_feature_engineering(df.copy())
print(df.head(10))