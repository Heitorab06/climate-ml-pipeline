import pandas as pd

df = pd.read_parquet("data/weather_raw.parquet")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return (df.copy()
            .pipe(convert_data_types)
            .pipe(set_index_time))
    
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:    
    format_date = "%Y-%m-%dT%H:%M"
    
    df["time"] = pd.to_datetime(df["time"], format=format_date)
         
    return df

def set_index_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("time")
    return df

df = clean_data(df=df)

