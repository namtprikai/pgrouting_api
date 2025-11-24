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
from helper import process_ship_data, process_train_data
from constant import *
import globals
from helper import create_response



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
    PGPORT: int = 5432
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
    mode: list
    origin_name: str
    origin_lat: confloat(ge=-90, le=90)
    origin_lon: confloat(ge=-180, le=180)
    destination_name: str
    destination_lat: confloat(ge=-90, le=90)
    destination_lon: confloat(ge=-180, le=180)
    max_transfers: int = 1
    show_all: bool = True
    find_station_radius_km: int
    find_port_radius_km: int
    departure_hour: int
    weight_tons: float



# =========================
# FastAPI
# =========================
app = FastAPI(title="Multimodal Truck/Train/Ship Router (FastAPI)")


@app.get("/api/routes-map")
def get_route_map():
    data_folder_path = FOLDER_DATA
    db_config = {
        "host": settings.PGHOST,
        "port": settings.PGPORT,
        "database": settings.PGDATABASE,
        "user": settings.PGUSER,
        "password": settings.PGPASSWORD,
    }

    optimizer = RouteOptimizer(data_folder_path, db_config)

    # Load ship data
    optimizer._load_ferry_schedule()
    optimizer._load_port_data()

    # Load train data
    optimizer._load_station_data()
    optimizer._load_train_schedule()

    # Process Ship data
    ship_schedule = optimizer.ferry_time
    ship_ports = optimizer.minato_gdf

    ship_data = process_ship_data(schedules_data=ship_schedule, ports_data=ship_ports)

    # Process Train data
    train_stations = optimizer.station_gdf
    train_schedules = optimizer.train_time

    train_data = process_train_data(
        stations_data=train_stations, schedules_data=train_schedules
    )

    # Combine data
    combined_data = {"ship_routes": ship_data, "train_routes": train_data}

    return combined_data


@app.post("/api/search-route")
def multimodal_route(payload: MultimodalBody):
    origin_name = payload.origin_name if payload.origin_name is not None else None
    origin_lat = float(payload.origin_lat) if payload.origin_lat is not None else None
    origin_lon = float(payload.origin_lon) if payload.origin_lon is not None else None
    destination_name = payload.destination_name if payload.destination_name is not None else None
    destination_lat = float(payload.destination_lat) if payload.destination_lat is not None else None
    destination_lon = float(payload.destination_lon) if payload.destination_lon is not None else None
    mode = [m.lower() for m in payload.mode]
    weight_tons = payload.weight_tons if payload.weight_tons is not None else 1
    max_transfers = payload.max_transfers if payload.max_transfers is not None else 1
    show_all = payload.show_all if payload.show_all is not None else True

    if (payload.departure_hour < 0 or payload.departure_hour > 23 or not isinstance(payload.departure_hour, int)):
        results = {
            'mode': mode,
            'message': 'Error format: departure_hour'
        }
        return create_response(origin_name, origin_lat, origin_lon, destination_name, destination_lat, destination_lon, results)

    departure_hour = payload.departure_hour if payload.departure_hour is not None else 0

    data_folder_path = FOLDER_DATA
    criteria = "fastest"
    db_config = {
        "host": settings.PGHOST,
        "port": settings.PGPORT,
        "database": settings.PGDATABASE,
        "user": settings.PGUSER,
        "password": settings.PGPASSWORD,
    }
    optimizer = RouteOptimizer(data_folder_path, db_config)

    all_results = []
    
    time_list = []
    distance_list = []
    co2_list = []
    
    # Xử lý từng mode riêng biệt
    for current_mode in mode:
        try:
            if current_mode.upper() not in STREET_TYPE:
                all_results.append({
                    'mode': current_mode,
                    'message': f'Mode {current_mode} not supported'
                })
                continue

            # Tìm route cho từng mode
            mode_results = optimizer.find_route(
                origin_lat,
                origin_lon,
                destination_lat,
                destination_lon,
                origin_name,
                destination_name,
                departure_hour,
                weight_tons,
                [current_mode],  # Chỉ truyền một mode
                enable_transfer=True,
                max_transfers=max_transfers,
                show_all=show_all,
            )

            if 'isError' in mode_results and mode_results.get('isError'):
                all_results.append({
                    'mode': current_mode,
                    'message': mode_results.get('message')
                })
            else:
                # Lấy thông tin từ kết quả trả về thay vì global state
                if 'routes' in mode_results and mode_results['routes']:
                    route = mode_results['routes'][0]  # Lấy route đầu tiên
                    
                    # Tạo geojson cho route này
                    single_route_result = {
                        "origin": mode_results.get("origin", {}),
                        "destination": mode_results.get("destination", {}),
                        "weight_tons": mode_results.get("weight_tons", 10.0),
                        "routes": [route],
                        "mode": [current_mode],
                        "enable_transfer": True,
                        "max_transfers": max_transfers,
                    }
                    
                    geojson = optimizer._convert_to_geojson(single_route_result)
                    
                    # Lấy thông tin từ route thay vì global state
                    route_info = {
                        'mode': current_mode,
                        'departure_time': globals.GLOBAL_STATE["departure_time"],
                        'arrival_time': globals.GLOBAL_STATE["arrival_time"],
                        'total_time_minutes': globals.GLOBAL_STATE["total_time_minutes"],
                        'total_move_time_minutes': globals.GLOBAL_STATE["total_move_time_minutes"],
                        'total_distance_km': globals.GLOBAL_STATE["total_distance_km"],
                        'total_co2_emissions_grams': globals.GLOBAL_STATE["total_co2_emissions_grams"],
                        'message': route.get('warning_message', ''),
                        'geojson': geojson
                    }
                    all_results.append(route_info)
                    
                    time_list.append(route_info["total_time_minutes"])
                    distance_list.append(route_info["total_distance_km"])
                    co2_list.append(route_info["total_co2_emissions_grams"])
                    
                    if all_results:
                        min_time = min(time_list)
                        min_distance = min(distance_list)
                        min_co2 = min(co2_list)

                        for r in all_results:
                            r["minimum_time_flag"] = bool(r["total_time_minutes"] == min_time)
                            r["minimum_distance_flag"] = bool(r["total_distance_km"] == min_distance)
                            r["minimum_co2_flag"] = bool(r["total_co2_emissions_grams"] == min_co2)

                else:
                    all_results.append({
                        'mode': current_mode,
                        'message': 'No route found'
                    })

        except Exception as e:
            all_results.append({
                'mode': current_mode,
                'message': f'Error processing {current_mode}: {str(e)}'
            })

    summary = create_response(origin_name, origin_lat, origin_lon, destination_name, destination_lat, destination_lon, all_results)
    return summary



# 直接起動用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=settings.PORT, reload=True)
