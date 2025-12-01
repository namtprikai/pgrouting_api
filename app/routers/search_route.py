from fastapi import APIRouter
import asyncio

from app.schemas.search_route import SeachRouteBody
from app.services.response_services import create_response, save_to_geojson
from app.core.constant import FOLDER_DATA
from app.core.config import DATABASE_HOST, DATABASE_PORT, DATABASE_USER, DATABASE_PASSWORD

router = APIRouter()
@router.post("/api/search-route")
async def multimodal_route(payload: SeachRouteBody):
    origin_name = payload.origin_name if payload.origin_name is not None else None
    origin_lat = float(payload.origin_lat) if payload.origin_lat is not None else None
    origin_lon = float(payload.origin_lon) if payload.origin_lon is not None else None
    destination_name = payload.destination_name if payload.destination_name is not None else None
    destination_lat = float(payload.destination_lat) if payload.destination_lat is not None else None
    destination_lon = float(payload.destination_lon) if payload.destination_lon is not None else None
    find_station_radius_km = payload.find_station_radius_km if payload.find_station_radius_km is not None else 100
    find_port_radius_km = payload.find_port_radius_km if payload.find_port_radius_km is not None else 100
    mode = payload.mode if payload.mode is not None else ["truck_only"]
    weight_tons = payload.weight_tons if payload.weight_tons is not None else 1
    max_transfers = (
        payload.max_transfers if payload.max_transfers is not None else 1
    )
    show_all = payload.show_all if payload.show_all is not None else True

    # Validate departure hour
    if (payload.departure_hour < 0 or payload.departure_hour > 23 or not isinstance(payload.departure_hour, int)):
        results = {
            'mode': mode,
            'message': 'Error format: departure_hour'
        }
        return create_response(origin_name, origin_lat, origin_lon, destination_name, destination_lat, destination_lon,
                               results)

    departure_hour = payload.departure_hour if payload.departure_hour is not None else 0

    data_folder_path = FOLDER_DATA
    db_config = {
        "host": DATABASE_HOST,
        "port": DATABASE_PORT,
        "database": DATABASE_PASSWORD,
        "user": DATABASE_USER,
        "password": DATABASE_PASSWORD,
    }

    all_results = []
    time_list = []
    distance_list = []
    co2_list = []

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
    save_to_geojson(summary, "output/result.geojson")

    return summary

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
            find_station_radius_km,
            find_port_radius_km,
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