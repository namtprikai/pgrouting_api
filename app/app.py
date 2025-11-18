import json
import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat, conint
from pydantic_settings import BaseSettings
from shapely import LineString
import geopandas as gpd
from route_optimizer import RouteOptimizer

from constant import *
import globals

# =========================
# 環境変数
# =========================
class Settings(BaseSettings):
    # PGHOST: str = os.getenv("PGHOST")
    # PGPORT: int = os.getenv("PGPORT")
    # PGDATABASE: str = os.getenv("PGDATABASE")
    # PGUSER: str = os.getenv("PGUSER")
    # PGPASSWORD: str = os.getenv("PGPASSWORD")

    PGHOST: str = "localhost"
    PGPORT: int = 5434
    PGDATABASE: str = "pgrouting"
    PGUSER: str = "postgres"
    PGPASSWORD: str = "pgrouting"

    PORT: int = 8080

    # デフォルト料金・船速（必要に応じて上書き）
    TOLL_PER_KM: float = 30.0
    SHIP_SPEED_KPH: float = 30.0

    # CO2 係数（g-CO2/トンkm）— 仮の既定値。運用で調整してください
    EF_TRUCK_G_PER_TKM: float = 120.0
    EF_TRAIN_G_PER_TKM: float = 22.0
    EF_SHIP_G_PER_TKM: float = 12.0
    PAYLOAD_TON: float = 10.0

    class Config:
        env_file = ".env"


settings = Settings()
monitoring = False

# =========================
# DB プール
# =========================
POOL: psycopg2.pool.SimpleConnectionPool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=settings.PGHOST,
    port=settings.PGPORT,
    database=settings.PGDATABASE,
    user=settings.PGUSER,
    password=settings.PGPASSWORD,
)

def sql_one(query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            if not row:
                return None
            # psycopg2 の場合は列名を得る
            cols = [desc.name for desc in cur.description]
            return {c: v for c, v in zip(cols, row)}
    finally:
        POOL.putconn(conn)

def sql_all(query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return None
            # psycopg2 の場合は列名を得る
            cols = [desc.name for desc in cur.description]
            result = []
            for row in rows:
                result.append({c: v for c, v in zip(cols, row)})
            return result
    finally:
        POOL.putconn(conn)

# =========================
# 入出力スキーマ
# =========================
class MultimodalBody(BaseModel):
    origin_name: str
    origin_lat: confloat(ge=-90, le=90)
    origin_lon: confloat(ge=-180, le=180)
    destination_name: str
    destination_lat: confloat(ge=-90, le=90)
    destination_lon: confloat(ge=-180, le=180)
    mode: str = STREET_TYPE["TRUCK_ONLY"]
    weight: int = 1
    max_transfers: int = 1
    show_all: bool = True
    departure_hour: int

# =========================
# FastAPI
# =========================
app = FastAPI(title="Multimodal Truck/Train/Ship Router (FastAPI)")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/multimodal/route")
def multimodal_route(payload: MultimodalBody):

    if (payload.mode.upper() in STREET_TYPE):
        # 基本パラメータ
        origin_name = payload.origin_name if payload.origin_name is not None else None
        origin_lat = float(payload.origin_lat) if payload.origin_lat is not None else None
        origin_lon = float(payload.origin_lon) if payload.origin_lon is not None else None
        destination_name = payload.destination_name if payload.destination_name is not None else None
        destination_lat = float(payload.destination_lat) if payload.destination_lat is not None else None
        destination_lon = float(payload.destination_lon) if payload.destination_lon is not None else None
        mode = payload.mode.lower() if payload.mode is not None else STREET_TYPE["TRUCK_ONLY"].lower()
        weight = payload.weight if payload.weight is not None else 1
        max_transfers = payload.max_transfers if payload.max_transfers is not None else 1
        show_all = payload.show_all if payload.show_all is not None else True

        if (payload.departure_hour < 0 or payload.departure_hour > 23 or not isinstance(payload.departure_hour, int)):
            return {
                'mode': payload.mode,
                'msg': 'Error format: departure_hour'
            }

        departure_hour = payload.departure_hour if payload.departure_hour is not None else 0

        data_folder_path = FOLDER_DATA
        criteria = 'fastest'
        db_config = {
            'host': settings.PGHOST,
            'port': settings.PGPORT,
            'database': settings.PGDATABASE,
            'user': settings.PGUSER,
            'password': settings.PGPASSWORD
        }
        optimizer = RouteOptimizer(data_folder_path, db_config)

        # Find route with automatic transfer detection
        results = optimizer.find_route(
            origin_lat, origin_lon, 
            destination_lat, destination_lon,
            origin_name,
            destination_name,
            departure_hour,
            weight,mode,
            enable_transfer=True,  # Automatically enabled
            max_transfers=max_transfers,
            show_all=show_all
        )

        if 'isError' in results and results.get('isError'):
            summary = {
                'mode': payload.mode,
                'msg': results.get('message')
            }
        else:
            # Save to file
            file_name = mode + '.geojson'
            
            # Determine what to save based on criteria
            if show_all:
                # Save all routes
                save_results = results
            else:
                # Save only optimal route for the specified criteria
                optimal_routes = results.get('optimal_routes', {})
                if criteria in optimal_routes:
                    # Find the full route with geometry from the original routes list
                    optimal_route_summary = optimal_routes[criteria]
                    all_routes = results.get('routes', [])
                    
                    # Find the corresponding full route by matching name and mode
                    optimal_route_full = None
                    for route in all_routes:
                        if (route.get('name') == optimal_route_summary.get('name') and 
                            route.get('mode') == optimal_route_summary.get('mode')):
                            optimal_route_full = route
                            break
                    
                    
                    # Use full route if found, otherwise use summary
                    selected_route = optimal_route_full if optimal_route_full else optimal_route_summary
                    
                    save_results = {
                        'origin': results.get('origin', {}),
                        'destination': results.get('destination', {}),
                        'weight_tons': results.get('weight_tons', 10.0),
                        'routes': [selected_route],
                        'optimal_routes': {criteria: optimal_route_summary},
                        'criteria_used': criteria,
                        'show_all': show_all,
                        'mode': mode,
                        'enable_transfer': True,  # Automatically enabled
                        'max_transfers': max_transfers
                    }
                else:
                    # Fallback to all routes if optimal route not found
                    save_results = results
            
            optimizer.save_results(save_results, 'output/' + file_name)
            print(f"\nResults saved to: output/{file_name}")
            
            geojson = optimizer._convert_to_geojson(save_results)
            if not show_all:
                print(f"Saved optimal route by criteria: {criteria}")

            data = results['optimal_routes'][criteria]
            departure_time = globals.GLOBAL_STATE["departure_time"]
            arrival_time = globals.GLOBAL_STATE["arrival_time"]

            summary = {
                'mode': mode.upper(),

                'origin_name': origin_name,
                'destination_name': destination_name,
                
                'departure_time': departure_time,
                'arrival_time': arrival_time,

                'origin_lat': origin_lat,
                'origin_lon': origin_lon,
                'destination_lat': destination_lat,
                'destination_lon': destination_lon,

                'total_time_minutes': data['total_time_minutes'] if 'total_time_minutes' in data and data['total_time_minutes'] else None,
                'total_distance_meters': data['total_distance_meters'] if 'total_distance_meters' in data and data['total_distance_meters'] else None,
                'total_co2_emissions_grams': data['co2_emissions_grams'] if 'co2_emissions_grams' in data and data['co2_emissions_grams'] else None,
                
                'message': '', #"時刻表データがないため、簡易試算となります"

                'geojson': geojson
            }
    else:
        summary = {
            'mode': payload.mode,
            'msg': 'Street cann\'t found'
        }
    return summary


# 直接起動用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=settings.PORT, reload=True)
