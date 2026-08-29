import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

def connect() -> Engine:
    """Create and return a SQLAlchemy database engine using environment variables.

    Returns:
        Engine: SQLAlchemy Engine connected to the PostgreSQL database.
    """
    db_host = os.getenv("HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    port = os.getenv("PORT")
    db_name = os.getenv("DB_NAME")

    conn_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{port}/{db_name}"
    engine = create_engine(conn_string)
    return engine


def save_to_database(df: pd.DataFrame) -> None:
    """Save a DataFrame into the PostgreSQL database table.

    Args:
        df (pd.DataFrame): DataFrame containing weather records to persist.
    """
    engine = connect()
    
    df.to_sql(
        name = "weather",
        con = engine,
        if_exists="replace",
        index=True,
        index_label="time"
    )


def select(query:str) -> pd.DataFrame:
    """Execute a SQL query and return the resulting data as a DataFrame.

    Args:
        query (str): SQL query string to execute.

    Returns:
        pd.DataFrame: Query results loaded into a pandas DataFrame.
    """
    engine = connect()
    df = pd.read_sql(query, con=engine)
    return df

def check_database_has_data(table_name: str = "weather") -> bool:
    """Check whether a database table exists and contains records.

    Args:
        table_name (str): Name of the database table to verify. Defaults to "weather".

    Returns:
        bool: True if the table contains at least one record, False otherwise.
    """
    query = f"SELECT 1 FROM {table_name} LIMIT 1;"
    logger = logging.getLogger(__name__)
 
    try:
        result = select(query)  
        
        if result is None:
            return False

        return not result.empty
        
    except Exception as e:
        logger.error("Error when accessing database", exc_info=e)
        return False

