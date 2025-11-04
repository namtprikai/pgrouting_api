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
    dest_name: str
    dest_lat: confloat(ge=-90, le=90)
    dest_lon: confloat(ge=-180, le=180)
    mode: str = STREET_TYPE["TRUCK_ONLY"]
    weight: int = 1
    max_transfers: int = 1
    show_all: bool = True

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
        origin_lat = float(payload.origin_lat) if payload.origin_lat is not None else None
        origin_lon = float(payload.origin_lon) if payload.origin_lon is not None else None
        dest_lat = float(payload.dest_lat) if payload.dest_lat is not None else None
        dest_lon = float(payload.dest_lon) if payload.dest_lon is not None else None
        mode = payload.mode.lower() if payload.mode is not None else STREET_TYPE["TRUCK_ONLY"].lower()
        weight = payload.weight if payload.weight is not None else 1
        max_transfers = payload.max_transfers if payload.max_transfers is not None else 1
        show_all = payload.show_all if payload.show_all is not None else True

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
            dest_lat, dest_lon, 
            weight, mode,
            enable_transfer=True,  # Automatically enabled
            max_transfers=max_transfers,
            show_all=show_all
        )

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
            
        optimizer.save_results(save_results, file_name)
        print(f"\nResults saved to: {file_name}")
        if not show_all:
            print(f"Saved optimal route by criteria: {criteria}")
        
        data = results['optimal_routes'][criteria]

        summary = {
            'time': data['total_time_minutes'] / 60 if 'total_time_minutes' in data and data['total_time_minutes'] else None,
            'distance': data['total_distance_km'] if 'total_distance_km' in data and data['total_distance_km'] else None,
            'co2': data['co2_emissions_grams'] if 'co2_emissions_grams' in data and data['co2_emissions_grams'] else None,
            'mode': mode,
            'origin_port': data['origin_port'] if 'origin_port' in data and data['origin_port'] else None,
            'dest_port': data['dest_port'] if 'dest_port' in data and data['dest_port'] else None,
            'origin_station': data['origin_station'] if 'origin_station' in data and data['origin_station'] else None,
            'dest_station': data['dest_station'] if 'dest_station' in data and data['dest_station'] else None,
            'transfer_port': data['transfer_port'] if 'transfer_port' in data and data['transfer_port'] else None,
            'transfer_station': data['transfer_station'] if 'transfer_station' in data and data['transfer_station'] else None,
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
