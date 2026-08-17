import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
port = os.getenv("PORT")
db_name = os.getenv("DB_NAME")

print("Diretório atual de execução:", os.getcwd())
print("HOST lido:", repr(os.getenv("HOST")))
print("USER lido:", repr(os.getenv("DB_USER")))
print("PASSWORD lida:", repr(os.getenv("DB_PASSWORD")))

conn = psycopg2.connect(
    host=db_host,
    dbname=db_name,
    port=port,
    user=db_user,
    password=db_password)


conn.close()