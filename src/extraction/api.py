import requests
import pandas as pd
import src.config as config
from typing import Any

def get_weather_data(
    url: str | None=None, 
    params: dict[str, Any] | None=None
    ) -> pd.DataFrame:
    
    url = url or config.URL
    params = params or config.PARAMS
    
    r = requests.get(url=url, params=params)
    r.raise_for_status() #trata erros na requisição
    json_data = r.json()
    df = pd.DataFrame(json_data["hourly"])
    
    return df