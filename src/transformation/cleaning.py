import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Execute data cleaning pipeline by converting data types and setting the index.

    Args:
        df (pd.DataFrame): Raw weather DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with formatted datetime and time index.
    """
    return (df.copy()
            .pipe(convert_data_types)
            .pipe(set_index_time))
    
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:    
    """Convert the 'time' column to datetime format.

    Args:
        df (pd.DataFrame): DataFrame containing the 'time' column in string format.

    Returns:
        pd.DataFrame: DataFrame with the 'time' column converted to datetime.
    """
    format_date = "%Y-%m-%dT%H:%M"
    
    df["time"] = pd.to_datetime(df["time"], format=format_date)
         
    return df

def set_index_time(df: pd.DataFrame) -> pd.DataFrame:
    """Set the 'time' column as the DataFrame index.

    Args:
        df (pd.DataFrame): DataFrame containing the 'time' column.

    Returns:
        pd.DataFrame: DataFrame indexed by the 'time' column.
    """
    df = df.set_index("time")
    return df


