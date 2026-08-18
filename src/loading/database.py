import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

def connect() -> Engine:
    db_host = os.getenv("HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    port = os.getenv("PORT")
    db_name = os.getenv("DB_NAME")

    conn_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{port}/{db_name}"
    engine = create_engine(conn_string)
    return engine


def upload_to_postgres() -> None:
    engine = connect()
    df = pd.read_parquet("data/weather_clean.parquet")
    
    df.to_sql(
        name = "weather",
        con = engine,
        if_exists="replace",
        index=True,
        index_label="time"
    )


def select(query:str) -> pd.DataFrame:
    engine = connect()
    df = pd.read_sql(query, con=engine)
    return df

