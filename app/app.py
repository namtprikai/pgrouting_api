from typing import Any, Dict, Optional
import asyncio
import asyncpg
from fastapi import FastAPI
from pydantic import BaseModel, confloat
from pydantic_settings import BaseSettings
from route_optimizer import RouteOptimizer
from helper import process_ship_data, process_train_data
from constant import *
from helper import create_response, save_summary_to_geojson
import threading

# =========================
# 環境変数
# =========================
class Settings(BaseSettings):
    PGHOST: str
    PGPORT: int
    PGDATABASE: str
    PGUSER: str
    PGPASSWORD: str

    TOLL_PER_KM: float
    SHIP_SPEED_KPH: float

    EF_TRUCK_G_PER_TKM: float
    EF_TRAIN_G_PER_TKM: float
    EF_SHIP_G_PER_TKM: float
    PAYLOAD_TON: float
    
    PORT: int
    
    class Config:
        env_file = ".env"


settings = Settings()
monitoring = False

# =========================
# Async DB Pool
# =========================
class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def create_pool(self):
        """Create async database connection pool"""
        self.pool = await asyncpg.create_pool(
            host=settings.PGHOST,
            port=settings.PGPORT,
            database=settings.PGDATABASE,
            user=settings.PGUSER,
            password=settings.PGPASSWORD,
            min_size=2,
            max_size=20,
            command_timeout=60
        )
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

# Global database instance
db = Database()


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


# Startup event to create DB pool
@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool on startup"""
    await db.create_pool()
    print("Database connection pool created")


# Shutdown event to close DB pool
@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection pool on shutdown"""
    await db.close_pool()
    print("Database connection pool closed")


# API endpoint for getting available routes
@app.get("/api/routes-map")
async def get_route_map():
    """Get available ship and train routes (async version)"""
    data_folder_path = FOLDER_DATA
    db_config = {
        "host": settings.PGHOST,
        "port": settings.PGPORT,
        "database": settings.PGDATABASE,
        "user": settings.PGUSER,
        "password": settings.PGPASSWORD,
    }

    # Create RouteOptimizer instance
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

    return {"ship_routes": ship_data, "train_routes": train_data}


# API endpoint for multimodal route search
@app.post("/api/search-route")
async def multimodal_route(payload: MultimodalBody):
    """Find multimodal routes (fully async version)"""
    # Input validation and processing
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

    # Validate departure hour
    if (payload.departure_hour < 0 or payload.departure_hour > 23 or not isinstance(payload.departure_hour, int)):
        results = {
            'mode': mode,
            'message': 'Error format: departure_hour'
        }
        return create_response(origin_name, origin_lat, origin_lon, destination_name, destination_lat, destination_lon, results)

    departure_hour = payload.departure_hour if payload.departure_hour is not None else 0

    data_folder_path = FOLDER_DATA
    db_config = {
        "host": settings.PGHOST,
        "port": settings.PGPORT,
        "database": settings.PGDATABASE,
        "user": settings.PGUSER,
        "password": settings.PGPASSWORD,
    }

    all_results = []
    time_list = []
    distance_list = []
    co2_list = []
    
    async def process_single_mode(current_mode: str) -> Dict[str, Any]:
        """Process single transportation mode asynchronously"""
        try:
            if current_mode.upper() not in STREET_TYPE:
                return {
                    'mode': current_mode,
                    'message': f'Mode {current_mode} not supported'
                }

            # Create RouteOptimizer instance for this mode
            optimizer = RouteOptimizer(data_folder_path, db_config)
            
            # Initialize database connection
            await optimizer._init_database()
            
            # Find route using async method
            mode_results = await optimizer.find_route(
                origin_lat,
                origin_lon,
                destination_lat,
                destination_lon,
                origin_name,
                destination_name,
                departure_hour,
                weight_tons,
                [current_mode],
                enable_transfer=True,
                max_transfers=max_transfers,
                show_all=show_all,
            )
            
            # Clean up database connection
            if optimizer.db_pool:
                await optimizer.db_pool.close()

            if 'isError' in mode_results and mode_results.get('isError'):
                return {
                    'mode': current_mode,
                    'message': mode_results.get('message')
                }
            else:
                if 'routes' in mode_results and mode_results['routes']:
                    all_routes = mode_results['routes']
                    global_info = mode_results.get("global_state_info", {})

                    single_route_result = {
                        "origin": mode_results.get("origin", {}),
                        "destination": mode_results.get("destination", {}),
                        "weight_tons": mode_results.get("weight_tons", 10.0),
                        "routes": all_routes,
                        "mode": [current_mode],
                        "enable_transfer": True,
                        "max_transfers": max_transfers,
                    }

                    geojson = optimizer._convert_to_geojson(single_route_result)

                    route_info = {
                        'mode': current_mode,
                        'departure_time': global_info.get("departure_time", ""),
                        'arrival_time': global_info.get("arrival_time", ""),
                        'total_time_minutes': global_info.get("total_time_minutes", 0),
                        'total_move_time_minutes': global_info.get("total_move_time_minutes", 0),
                        'total_distance_km': global_info.get("total_distance_km", 0),
                        'total_co2_emissions_grams': global_info.get("total_co2_emissions_grams", 0),
                        'geojson': geojson
                    }

                    return route_info

                else:
                    return {
                        'mode': current_mode,
                        'message': 'No route found'
                    }

        except Exception as e:
            print(f"Error processing {current_mode}: {str(e)}")
            return {
                'mode': current_mode,
                'message': f'Error processing {current_mode}: {str(e)}'
            }
    
    # Process all modes concurrently using asyncio.gather
    tasks = [process_single_mode(current_mode) for current_mode in mode]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            all_results.append({
                'mode': 'unknown',
                'message': f'Unexpected error: {str(result)}'
            })
        else:
            all_results.append(result)
            
            # Add to calculation lists if valid result
            if 'total_time_minutes' in result:
                time_list.append(result["total_time_minutes"])
                distance_list.append(result["total_distance_km"])
                co2_list.append(result["total_co2_emissions_grams"])

    # Calculate flags after all results are collected
    if all_results and time_list:
        min_time = min(time_list)
        min_distance = min(distance_list)
        min_co2 = min(co2_list)

        for r in all_results:
            if 'total_time_minutes' in r:
                r["minimum_time_flag"] = bool(r["total_time_minutes"] == min_time)
                r["minimum_distance_flag"] = bool(r["total_distance_km"] == min_distance)
                r["minimum_co2_flag"] = bool(r["total_co2_emissions_grams"] == min_co2)

    summary = create_response(
        origin_name, origin_lat, origin_lon, 
        destination_name, destination_lat, destination_lon, 
        all_results
    )
    
    # Check the returned data using geojson file
    save_summary_to_geojson(summary, "output/result.geojson")
    
    return summary


# API endpoint for health check connection to DB
@app.get("api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        if db.pool:
            async with db.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy", "database": "connected"}
        else:
            return {"status": "unhealthy", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# API endpoint for testing route finding with debug info
@app.post("/api/test-route")
async def test_route():
    """Test route finding with debug information"""
    try:
        data_folder_path = FOLDER_DATA
        db_config = {
            "host": settings.PGHOST,
            "port": settings.PGPORT,
            "database": settings.PGDATABASE,
            "user": settings.PGUSER,
            "password": settings.PGPASSWORD,
        }
        
        # Create RouteOptimizer instance
        optimizer = RouteOptimizer(data_folder_path, db_config)
        
        # Initialize database connection
        await optimizer._init_database()
        
        # Test coordinates (Tokyo to Osaka)
        origin_lat, origin_lon = 35.6895, 139.6917  # Tokyo
        dest_lat, dest_lon = 34.6937, 135.5023  # Osaka
        
        result = await optimizer.find_route(
            origin_lat, origin_lon,
            dest_lat, dest_lon,
            "Tokyo", "Osaka",
            8,  # departure_hour
            10.0,  # weight_tons
            ["truck_only"],  # mode
            enable_transfer=True,
            max_transfers=1,
            show_all=True
        )
        
        # Clean up
        if optimizer.db_pool:
            await optimizer.db_pool.close()
            
        return result
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}


# API endpoint for getting thread info
@app.get("/thread-info")
async def thread_info():
    return {
        "total_threads": threading.active_count(),
        "thread_names": [t.name for t in threading.enumerate()]
    }


# 直接起動用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=settings.PORT, 
        reload=True,
        workers=1,
        loop="asyncio"
    )