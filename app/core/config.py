import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
DATABASE_HOST = os.getenv("PGHOST", "")
DATABASE_PORT = os.getenv("PGPORT", "")
DATABASE_NAME = os.getenv("PGDATABASE", "")
DATABASE_USER = os.getenv("PGUSER", "")
DATABASE_PASSWORD = os.getenv("PGPASSWORD", "")
SECRET_KEY = os.getenv("SECRET_KEY")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

INPUT_FOLDER = "data_file"
OUTPUT_FOLDER = "output/"