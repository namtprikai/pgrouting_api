import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from typing import Optional
import os

class DatabaseManager:
    def __init__(self, host: str, port: int, database: str, username: str, password: str):
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': username,
            'password': password
        }
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Connect Database"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.cursor = self.connection.cursor()
            print("Connect database success")
            return True
        except psycopg2.Error as e:
            print(f"Error connect database: {e}")
            return False
    
    def disconnect(self):
        """Close connect database"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Closed connect database")
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, batch_size: int = 1000) -> bool:
        """Insert DataFrame into database"""
        try:
            df_clean = df.copy()
            
            df_clean = df_clean.fillna('')
            
            insert_sql = f"""
            INSERT INTO {table_name} ("name", "geom") 
            VALUES %s
            """
            
            data_tuples = [tuple([row[1], f"POINT({row[3]} {row[2]})"]) for row in df_clean.values]
            
            total_rows = len(data_tuples)
            inserted_rows = 0
            
            for i in range(0, total_rows, batch_size):
                batch_data = data_tuples[i:i + batch_size]
                
                execute_values(
                    self.cursor,
                    insert_sql,
                    batch_data,
                    template=None,
                    page_size=batch_size
                )
                
                inserted_rows += len(batch_data)
                print(f"Inserted {inserted_rows}/{total_rows} rows")
            
            self.connection.commit()
            print(f"Insert success {total_rows} rows into table {table_name}")
            return True
            
        except psycopg2.Error as e:
            print(f"Error insert data: {e}")
            self.connection.rollback()
            return False

class ExcelProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
    
    def read_excel(self, sheet_name: Optional[str] = None, header: int = 0) -> pd.DataFrame:
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not exist: {self.file_path}")

            if sheet_name:
                self.data = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header)
            else:
                self.data = pd.read_excel(self.file_path, header=header)
            
            return self.data
        
        except Exception as e:
            print(f"Error read Excel: {e}")
            return pd.DataFrame()
    
    def clean_data(self) -> pd.DataFrame:
        if self.data is None or self.data.empty:
            print("File is empty")
            return pd.DataFrame()
        
        try:
            cleaned_data = self.data.copy()
            cleaned_data.columns = cleaned_data.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
            cleaned_data = cleaned_data.dropna(how='all')
            
            # Reset index
            cleaned_data = cleaned_data.reset_index(drop=True)            
            print(f"Data have cleaned: {len(cleaned_data)} rows")
            return cleaned_data
            
        except Exception as e:
            print(f"Error clean data: {e}")
            return pd.DataFrame()

def main():    
    # Config database
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'pgrouting',
        'username': 'postgres',
        'password': 'pgrouting'
    }
    
    excel_file_path = 'importer_data/貨物駅_位置情報.xlsx'
    table_name = 'freight_stations'
    sheet_name = None #Read first sheet
    
    try:
        excel_processor = ExcelProcessor(excel_file_path)
        
        raw_data = excel_processor.read_excel(sheet_name=sheet_name)
        if raw_data.empty:
            print("Data is empty")
            return
        
        cleaned_data = excel_processor.clean_data()
        if cleaned_data.empty:
            print("Data is empty")
            return
        
        # 2. Connect database
        db_manager = DatabaseManager(**db_config)
        if not db_manager.connect():
            print("Can't connect database")
            return
        
        success = db_manager.insert_dataframe(cleaned_data, table_name, batch_size=1000)
        
        if success:
            print("Import success!")
        else:
            print("Import fail!")
        
        db_manager.disconnect()
        
    except Exception as e:
        print(f"Error handle import data: {e}")

if __name__ == "__main__":
    main()

