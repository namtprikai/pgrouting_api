from fastapi import APIRouter
import os

router = APIRouter()


@router.get("/api/routes-map")
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