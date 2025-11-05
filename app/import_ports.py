import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

df = pd.read_csv('importer_data/貨物船_位置情報（国土数値情報）.csv', encoding='shift_jis').fillna('')

conn = psycopg2.connect(host='localhost', port=5432, database='pgrouting', user='postgres', password='pgrouting')
cursor = conn.cursor()
execute_values(cursor, "INSERT INTO ports (name, geom) VALUES %s", 
               [tuple([row[6], f"POINT({row[0]} {row[1]})"]) for row in df.values])
conn.commit()
conn.close()
