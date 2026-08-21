from src.loading.database import select
from src.modeling.feature_engineering import run_feature_engineering

#from scikit-learn import (model)

df = select("SELECT * FROM WEATHER")

df = run_feature_engineering(df=df)

print(df.head(10))