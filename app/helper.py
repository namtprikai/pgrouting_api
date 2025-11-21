from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import linemerge
import math
import json
from datetime import datetime, timedelta
from shapely import wkt
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Any, Dict, List, Optional

RouteDict = Dict[str, Any]

def _is_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _clean_points(coords):
    out = []
    for p in coords:
        if (
            isinstance(p, (list, tuple))
            and len(p) == 2
            and _is_num(p[0])
            and _is_num(p[1])
        ):
            out.append((float(p[0]), float(p[1])))
    return out


def _flatten_if_singleton_segments(coords):
    for _ in range(2):
        if (
            isinstance(coords, (list, tuple))
            and len(coords) > 0
            and isinstance(coords[0], (list, tuple))
            and len(coords[0]) > 0
            and isinstance(coords[0][0], (list, tuple))
        ):
            coords = coords[0]
        else:
            break
    return coords


def extract_linestring(geom):
    # 1) Shapely
    if isinstance(geom, LineString):
        return geom

    if isinstance(geom, MultiLineString):
        try:
            merged = linemerge(geom)
            
            if isinstance(merged, LineString):
                return merged
            if isinstance(merged, MultiLineString) and len(merged.geoms) > 0:
                
                return LineString(list(merged.geoms[0].coords))
        except Exception:
            
            if len(geom.geoms) > 0:
                return LineString(list(geom.geoms[0].coords))
            return LineString()

    # 2) GeoJSON dict
    coords = None
    if isinstance(geom, dict):
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "LineString":
            pass  
        elif gtype == "MultiLineString":
            coords = _flatten_if_singleton_segments(coords)
        else:
            return LineString()

    if coords is None and isinstance(geom, (list, tuple)):
        coords = geom

    coords = _flatten_if_singleton_segments(coords)

    pts = _clean_points(coords if coords is not None else [])

    if len(pts) >= 2:
        return LineString(pts)
    return LineString()


def _nz(x):
    """None/NaN/inf -> 0.0"""
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _get_name(x):
    if isinstance(x, dict):
        return x.get("name") or x.get("C02_005") or next(iter(x.values()), None)
    if isinstance(x, (str, int, float)):
        return str(x)
    return str(x)


def _label_stations(prefix, stations):
    stations = [s for s in (stations or []) if s is not None]
    if not stations:
        return {}
    if len(stations) == 1:
        return {f"{prefix}_station": _get_name(stations[0])}
    return {f"{prefix}_station_{i}": _get_name(s) for i, s in enumerate(stations, 1)}


def build_data_infos(
    origin_port,
    dest_port,
    origin_stations,
    dest_stations,  
    emissions, 
    ship_time=0,
    train_time_minutes=0,
    truck_time_minutes=0,
    truck_distances_km=None,
):
    co2_emissions = math.fsum(_nz(v) for v in (emissions or []))

    truck_distance_km = math.fsum(_nz(d) for d in (truck_distances_km or []))

    data = {
        "origin_port": (
            origin_port["C02_005"]
            if isinstance(origin_port, dict)
            else _get_name(origin_port)
        ),
        "dest_port": (
            dest_port["C02_005"]
            if isinstance(dest_port, dict)
            else _get_name(dest_port)
        ),
        "co2_emissions": co2_emissions,
        "ship_time": ship_time,
        "train_time_minutes": train_time_minutes,
        "truck_time_minutes": truck_time_minutes,
        "truck_distance_km": truck_distance_km,
    }
    data.update(
        _label_stations(
            "origin",
            (
                origin_stations
                if isinstance(origin_stations, (list, tuple))
                else [origin_stations]
            ),
        )
    )
    data.update(
        _label_stations(
            "dest",
            (
                dest_stations
                if isinstance(dest_stations, (list, tuple))
                else [dest_stations]
            ),
        )
    )
    return data


def build_result_segment(
    idx: int,
    merged_count: int,
    geometry,
    total_distance_meters: float,
    data_infos: dict,
    mode: str
):
    total_distance_km = (
        float(total_distance_meters) / 1000.0
        if isinstance(total_distance_meters, (int, float))
        else None
    )

    fixed_fields = {
        "total_co2_emissions_grams": data_infos.get("co2_emissions"),
        "origin_port": data_infos.get("origin_port"),
        "dest_port": data_infos.get("dest_port"),
        "ship_time_hours": data_infos.get("ship_time"),
        "train_time_minutes": data_infos.get("train_time_minutes"),
        "truck_time_minutes": data_infos.get("truck_time_minutes"),
        "truck_distance_km": data_infos.get("truck_distance_km"),
    }

    station_fields = {
        k: v
        for k, v in data_infos.items()
        if k.startswith("origin_station") or k.startswith("dest_station")
    }

    return {
        "mode": mode,
        "segment_index": idx,
        "total_segments": merged_count,
        "is_continuous": (merged_count == 1),
        "total_time_minutes": "",  
        "total_distance_meters": total_distance_meters,
        "total_distance_km": total_distance_km,
        "transfer_station": "",
        "geometry": geometry,
        **fixed_fields,
        **station_fields,  
    }


def linestring_to_geojson_feature(geom, props=None, precision=6):
    """geom: shapely.geometry.LineString (EPSG:4326)
    props: dict properties (optional)"""
    assert geom.geom_type == "LineString"
    m = mapping(geom) 

    coords = [[round(x, precision), round(y, precision)] for x, y in m["coordinates"]]

    feature = {
        "type": "Feature",
        "properties": props or {},
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }
    print("---------------------------------------------------------", "\n")
    print(json.dumps(feature, ensure_ascii=False))
    print("---------------------------------------------------------", "\n")
    return feature

def add_hours(time_str: str, hours: float = 3.0) -> str:
    # Parse "HH:MM"
    base_time = datetime.strptime(time_str, "%H:%M")
    new_time = base_time + timedelta(hours=hours)
    return new_time.strftime("%H:%M")

def create_features(route):
    try:
        geometry = route["geometry"]

        # Handle different geometry types
        if isinstance(geometry, str):
            # If it's a WKT string, parse it
            try:
                geometry = wkt.loads(geometry)
            except:
                # If WKT parsing fails, try to parse as JSON
                try:
                    geometry = json.loads(geometry)
                except:
                    print(
                        f"Warning: Could not parse geometry string: {geometry[:100]}..."
                    )
                    return None
        elif isinstance(geometry, LineString):
            # If it's already a LineString object, use it directly
            pass
        elif isinstance(geometry, dict):
            # Already GeoJSON, use as is
            pass
        elif isinstance(geometry, list):
            # List of coordinates → convert to LineString
            try:
                geometry = LineString(geometry)
            except:
                print(f"Warning: Could not convert list geometry: {geometry[:100]}...")
                return None
        else:
            # Try to convert other geometry types
            try:
                geometry = wkt.loads(str(geometry))
            except:
                print(
                    f"Warning: Could not convert geometry: {type(geometry)}"
                )
                return None

        # Create feature
        if isinstance(geometry, dict):
            # Already GeoJSON
            new_feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": convert_numpy_types(
                    {
                        "vehicle": route.get("vehicle", ""),
                        "origin_name": route.get("origin_name", ""),
                        "destination_name": route.get(
                            "destination_name", ""
                        ),
                        "departure_time": route.get(
                            "departure_time", "00:00"
                        ),
                        "arrival_time": route.get("arrival_time", "00:00"),
                        "total_wait_time_before_departure_minutes": route.get(
                            "total_wait_time_before_departure_minutes", 0
                        ),
                        "total_time_minutes": route.get(
                            "total_time_minutes", 0
                        ),
                        "total_distance_km": route.get(
                            "total_distance_km", 0
                        ),
                        "total_co2_emissions_grams": (route.get(
                            "total_co2_emissions_grams", 0
                        )),
                    }
                ),
            }
        else:
            # Shapely geometry
            new_feature = {
                "type": "Feature",
                "geometry": {
                    "type": geometry.geom_type,
                    "coordinates": list(geometry.coords),
                },
                "properties": convert_numpy_types(
                    {
                        "vehicle": route.get("vehicle", ""),
                        "origin_name": route.get("origin_name", ""),
                        "destination_name": route.get(
                            "destination_name", ""
                        ),
                        "departure_time": route.get(
                            "departure_time", "00:00"
                        ),
                        "arrival_time": route.get("arrival_time", "00:00"),
                        "total_wait_time_before_departure_minutes": route.get(
                            "total_wait_time_before_departure_minutes", 0
                        ),
                        "total_time_minutes": route.get(
                            "total_time_minutes", 0
                        ),
                        "total_distance_km": route.get(
                            "total_distance_km", 0
                        ),
                        "total_co2_emissions_grams": (route.get(
                            "total_co2_emissions_grams", 0
                        )),
                    }
                ),
            }
        
        return new_feature

    except Exception as e:
        print(
            f"Warning: Could not convert geometry for route {route.get('name', '')}: {e}"
        )

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def create_response(origin_name, origin_lat, origin_lon, destination_name, destination_lat, destination_lon, result):
    final_results = []

    if isinstance(result, dict):
        final_results.append(result)
    else:
        final_results = result

    summary = {
        'origin_name': origin_name,
        'origin_lat': origin_lat,
        'origin_lon': origin_lon,
        
        'destination_name': destination_name,
        'destination_lat': destination_lat,
        'destination_lon': destination_lon,
        
        'results': final_results
    }
    return summary

def calc_total_wait_time_before_departure_minutes(departure_time, arrival_time):
    return departure_time - arrival_time

def parse_time(time_str):
    return datetime.strptime(time_str, "%H:%M")

def find_ship_by_departure_time(routes: pd.DataFrame, arrival_time_buffer_wait_time):
    try:
        prev_arrival = parse_time(arrival_time_buffer_wait_time)
        ships = []
        
        for _, row in routes.iterrows():
            dep_str = row["Departure_Time"]
            dep_time = parse_time(dep_str)

            # Nếu Departure_Time < arrival_time trước → coi như xuất phát ngày hôm sau
            # if dep_time < prev_arrival:
            #     dep_time += timedelta(days=1)

            # Giữ những chuyến phù hợp
            if dep_time >= prev_arrival:
                ships.append(row)

        # Trả về DataFrame
        return pd.DataFrame(ships)
    except Exception as e:
        print(f"Error find_ship_by_departure_time: {e}")
        return []
    
def calc_wait_minutes(arrival_time: str, departure_time: str) -> float:
    """
    Calculate the waiting time (in minutes) from when the truck arrives
    until the train departs.
    - truck_arrival_time, train_departure_time: strings in format "H:MM" or "HH:MM"
    - If the train departure time is earlier than or equal to the truck arrival time
      on the same day → assume the train departs on the NEXT day.
    """
    # Parse "H:MM" or "HH:MM"
    arrival_time_hours, arrival_time_minutes = map(int, arrival_time.split(":"))
    departure_time_hours, departure_time_minutes = map(int, departure_time.split(":"))

    # Assign both times to the same dummy date
    base_date = datetime(1900, 1, 1)
    arrival_time_dt = base_date.replace(hour=arrival_time_hours, minute=arrival_time_minutes)
    departure_time_dt = base_date.replace(hour=departure_time_hours, minute=departure_time_minutes)

    # If train departs earlier or at the same time → move to the next day
    if departure_time_dt <= arrival_time_dt:
        departure_time_dt += timedelta(days=1)

    delta = departure_time_dt - arrival_time_dt
    return delta.total_seconds() / 60.0  # minutes


def to_nullable_str(v: Any) -> Optional[str]:
    """Return None if value is NaN/empty, else str(v)."""
    if pd.isna(v):
        return None
    s = str(v)
    if s.strip() == "":
        return None
    return s


def to_nullable_float(v: Any) -> Optional[float]:
    """Return None if value is NaN, else float(v)."""
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_hhmm_or_none(v: Any) -> Optional[str]:
    """Convert datetime-like to 'HH:MM', or None if NaN/invalid."""
    if pd.isna(v):
        return None
    try:
        return pd.to_datetime(v).strftime("%H:%M")
    except Exception:
        return None


def to_hhmm(t):
    t = str(t)
    if " " in t:  # "1900-01-02 11:04:52"
        return t.split(" ")[1][:5]
    return t[:5]


def process_train_data(
    schedules_data: pd.DataFrame, stations_data: gpd.GeoDataFrame
) -> List[RouteDict]:
    """
    Convert train schedule DataFrame to list[dict] in the unified route format.
    """
    routes: List[RouteDict] = []

    unique_stations = stations_data.drop_duplicates(subset="Station_Name", keep="first")

    station_lookup: Dict[str, Dict[str, float]] = {}
    for _, srow in unique_stations.iterrows():
        name = to_nullable_str(srow.get("Station_Name"))
        if name is None:
            continue
        station_lookup[name] = {
            "lon": to_nullable_float(srow.get("lon")),
            "lat": to_nullable_float(srow.get("lat")),
        }

    for _, row in schedules_data.iterrows():
        dep_name = to_nullable_str(row.get("Departure_Station_Name"))
        arr_name = to_nullable_str(row.get("Arrival_Station_Name"))

        dep_station = station_lookup.get(dep_name) if dep_name is not None else None
        arr_station = station_lookup.get(arr_name) if arr_name is not None else None

        if dep_station is not None:
            dep_lat = to_nullable_float(dep_station.get("lat"))
            dep_lon = to_nullable_float(dep_station.get("lon"))
        else:
            dep_lat = None
            dep_lon = None

        if arr_station is not None:
            arr_lat = to_nullable_float(arr_station.get("lat"))
            arr_lon = to_nullable_float(arr_station.get("lon"))
        else:
            arr_lat = None
            arr_lon = None

        duration_value = row.get("train_Duration")
        route_minutes = None

        if not pd.isna(duration_value):
            if isinstance(duration_value, pd.Timedelta):
                route_minutes = duration_value.total_seconds() / 60.0
            else:
                route_minutes = to_nullable_float(duration_value)

        routes.append(
            {
                "depature_name": dep_name,
                "depature_lat": dep_lat,
                "depature_lon": dep_lon,
                "arrival_name": arr_name,
                "arrival_lat": arr_lat,
                "arrival_lon": arr_lon,
                "departure_time": to_hhmm_or_none(
                    row.get("Departure_Time")
                ),  # "HH:MM" or None
                "arrival_time": to_hhmm_or_none(
                    row.get("Arrival_Time")
                ),  # "HH:MM" or None
                "route_time_minutes": route_minutes,  # float or None
            }
        )

    return routes


def process_ship_data(
    ports_data: gpd.GeoDataFrame,
    schedules_data: pd.DataFrame,
) -> list[dict]:
    """
    Convert ship schedule data into list of dicts in the format:
    {
        "depature_name": string,
        "depature_lat": float,
        "depature_lon": float,
        "arrival_name": string,
        "arrival_lat": float,
        "arrival_lon": float,
        "departure_time": string,  # HH:MM
        "arrival_time": string,    # HH:MM
        "route_time_minutes": float
    }
    """
    routes = []

    station_lookup = {}
    for _, row in ports_data.iterrows():
        port_name = row["C02_005"]  # port code/tên
        station_lookup[port_name] = {
            "name": port_name,
            "lat": float(row["Y"]),
            "lon": float(row["X"]),
        }

    for _, row in schedules_data.iterrows():
        dep_code = row[
            "Departure_Location_(National_Land_Numerical_Information_Format)"
        ]
        arr_code = row["Arrival_Location_(National_Land_Numerical_Information_Format)"]

        dep_station = station_lookup.get(dep_code)
        arr_station = station_lookup.get(arr_code)

        if not dep_station or not arr_station:
            continue

        dep_time = str(row["Departure_Time"])
        arr_time = str(row["Arrival_Time"])
        route_time = (
            float(row["Route_Time"]) if not pd.isna(row["Route_Time"]) else None
        )

        routes.append(
            {
                "depature_name": dep_station["name"],
                "depature_lat": dep_station["lat"],
                "depature_lon": dep_station["lon"],
                "arrival_name": arr_station["name"],
                "arrival_lat": arr_station["lat"],
                "arrival_lon": arr_station["lon"],
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "route_time_minutes": round(route_time * 60, 2),
            }
        )

    return routes