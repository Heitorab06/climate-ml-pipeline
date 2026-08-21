import pandas as pd
from src.loading.database import select

df = select("SELECT * FROM WEATHER")
print(df.columns)

def create_lag_features(df:pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    
    return df

def create_time_cyclical_features(df:pd.DataFrame) -> pd.DataFrame:
    
    return df

def create_target(df:pd.DataFrame) -> pd.DataFrame:
    
    return df

def run_feature_engineering(df:pd.DataFrame)-> pd.DataFrame:
    lag_features = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "precipitation"]
    return (df.copy()
            .pipe(create_lag_features(cols=lag_features, lags=[1,2,3,6]))
            .pipe(create_time_cyclical_features)
            .pipe(create_target))
    
df = run_feature_engineering(df.copy())
print(df.columns)