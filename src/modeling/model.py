from src.loading.database import select

df = select("SELECT * FROM WEATHER")
print(df.head(10))