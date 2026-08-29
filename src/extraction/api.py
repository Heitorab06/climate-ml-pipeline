from typing import Any

import pandas as pd
import requests

from src import config


def get_weather_data(
    url: str | None=None, 
    params: dict[str, Any] | None=None
    ) -> pd.DataFrame:
    """Fetch hourly weather forecast data from the Open-Meteo API.

    Args:
        url (str | None): Endpoint URL for the weather API. Defaults to None
            (uses config.URL).
        params (dict[str, Any] | None): Query parameters for the API request.
            Defaults to None (uses config.PARAMS).

    Returns:
        pd.DataFrame: DataFrame containing hourly weather records extracted
            from the API response.
    """
    url = url or config.URL
    params = params or config.PARAMS
    
    r = requests.get(url=url, params=params)
    r.raise_for_status() 
    json_data = r.json()
    df = pd.DataFrame(json_data["hourly"])
    
    return df