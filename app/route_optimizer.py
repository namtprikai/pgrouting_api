"""
Route Optimizer for Multimodal Transportation
Optimizes multimodal transportation routes (truck, train, ship)
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from datetime import timedelta, datetime
import pyproj
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from shapely import wkt
import math
from constant import *
import globals
from helper import (
    build_data_infos,
    build_result_segment,
    extract_linestring,
    add_hours,
    create_features,
    find_ship_by_departure_time,
    find_train_by_departure_time,
    calc_wait_minutes,
    reset_global_states
)
from decoratorr import timeit

import time

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class RouteOptimizer:
    """
    Class for optimizing multimodal transportation routes
    """

    def __init__(self, data_folder_path: str, db_config: Optional[Dict] = None):
        """
        Initialize RouteOptimizer

        Args:
            data_folder_path: Path to data folder
            db_config: Database configuration (host, port, database, user, password)
        """
        self.data_folder_path = data_folder_path

        # Database configuration
        if db_config is None:
            db_config = {
                "host": "localhost",
                "port": 5432,
                "database": "pgrouting",
                "user": "postgres",
                "password": "pgrouting",
            }

        self.db_config = db_config
        self.db_pool = None

        # Initialize data attributes
        self.odlist_gdf = None
        self.minato_gdf = None
        self.station_gdf = None
        self.ferry_time = None
        self.train_time = None
        self.track_route = None
        self.port_list = None

        # CO2 emission factors (g CO2 per ton-kilometer)
        self.co2_factors = {"truck": 207, "train": 19, "ship": 42}

        # Initialize database connection
        self._init_database()

        # Load all data
        self._load_all_data()
        
        self._node_cache = {}
        self._route_cache = {}

    def _load_all_data(self):
        """Load all required data"""
        print("Loading data...")

        # Load OD data
        self._load_od_data()

        # Load port data
        self._load_port_data()

        # Load station data
        self._load_station_data()

        # Load ferry schedule
        self._load_ferry_schedule()

        # Load train schedule
        self._load_train_schedule()

        print("Data loading completed!")

    def _load_od_data(self):
        """Load origin-destination data"""
        path = f"{self.data_folder_path}/L013101物流拠点出発地到着地リスト.csv"
        odlist = pd.read_csv(path, encoding="utf-8-sig")
        odlist = odlist.loc[:, ~odlist.columns.str.startswith("Unnamed")]
        self.odlist_gdf = gpd.GeoDataFrame(
            odlist,
            geometry=gpd.points_from_xy(odlist["Origin_lon"], odlist["Origin_lat"]),
        )
        self.odlist_gdf.set_crs(epsg=4326, inplace=True)

    def _load_port_data(self):
        """Load port data"""
        minato = pd.read_csv(
            f"{self.data_folder_path}/貨物船_位置情報（国土数値情報）.csv",
            encoding="SHIFT_JIS",
        )
        self.minato_gdf = gpd.GeoDataFrame(
            minato, geometry=gpd.points_from_xy(minato["X"], minato["Y"])
        )
        self.minato_gdf.set_crs(epsg=4326, inplace=True)
        self.minato_gdf["C02_005"] = self.minato_gdf["C02_005"] + "港"

    def _load_station_data(self):
        """Load freight station data"""
        station = pd.read_csv(f"{self.data_folder_path}/貨物駅_位置情報.csv")
        self.station_gdf = gpd.GeoDataFrame(
            station, geometry=gpd.points_from_xy(station["lon"], station["lat"])
        )
        self.station_gdf.set_crs(epsg=4326, inplace=True)

    def _load_ferry_schedule(self):
        """Load ferry schedule"""
        self.ferry_time = pd.read_csv(
            f"{self.data_folder_path}/貨物船_時刻表（2024年版海上定期便ガイド）.csv"
        )
        dep_col = "Departure_Location_(National_Land_Numerical_Information_Format)"
        arr_col = "Arrival_Location_(National_Land_Numerical_Information_Format)"
        self.ferry_time[dep_col] = self.ferry_time[dep_col] + "港"
        self.ferry_time[arr_col] = self.ferry_time[arr_col] + "港"

        # Remove records with NaN Route_Time
        self.ferry_time = self.ferry_time.dropna(subset=["Route_Time"])

        # Create port list
        dep_list = self.ferry_time[dep_col].dropna().astype(str).unique()
        arr_list = self.ferry_time[arr_col].dropna().astype(str).unique()
        self.port_list = list(set(list(dep_list) + list(arr_list)))

        # Filter ports
        self.minato_gdf = self.minato_gdf[
            self.minato_gdf["C02_005"].isin(self.port_list)
        ]

    def _load_train_schedule(self):
        """Load train schedule"""
        self.train_time = pd.read_csv(f"{self.data_folder_path}/貨物駅_時刻表.csv")

        # Replace arrival date
        self.train_time["Arrival_Date_Before"] = self.train_time["Arrival_Date"]
        self.train_time["Arrival_Date"] = self.train_time["Arrival_Date"].replace(
            {
                "当日": 0,
                "翌日": 1,
                "翌々日": 2,
                "７日目": 6,
                "４日目": 3,
                "５日目": 4,
                "６日目": 5,
            }
        )

        # Convert time format
        self.train_time["Track_Entry_Time"] = pd.to_datetime(
            self.train_time["Track_Entry_Time"], format="%H:%M:%S"
        ).dt.time
        self.train_time["Track_Exit_Time"] = pd.to_datetime(
            self.train_time["Track_Exit_Time"], format="%H:%M:%S"
        ).dt.time

        # Convert to datetime
        self.train_time["Track_Entry_Time"] = pd.to_datetime(
            self.train_time["Track_Entry_Time"].astype(str), format="%H:%M:%S"
        )
        self.train_time["Track_Exit_Time"] = pd.to_datetime(
            self.train_time["Track_Exit_Time"].astype(str), format="%H:%M:%S"
        )

        # Calculate duration
        self.train_time["Updated_Entry_Time"] = self.train_time[
            "Track_Entry_Time"
        ] + self.train_time["Arrival_Date"].apply(lambda x: timedelta(days=x))
        self.train_time["train_Duration"] = (
            self.train_time["Updated_Entry_Time"] - self.train_time["Track_Exit_Time"]
        )
        self.train_time["train_od"] = (
            self.train_time["Departure_Station_Name"].str.replace(" ", "")
            + "_"
            + self.train_time["Arrival_Station_Name"].str.replace(" ", "")
        )

        self.train_time["Arrival_Time"] = self.train_time["Arrival_Time"].astype(str).str.strip()
        self.train_time["Arrival_Time"].replace("", pd.NaT, inplace=True)

        arrival_time_str = self.train_time["Arrival_Time"].astype(str).str.extract(
            r'(\d{2}:\d{2}:\d{2})'
        )[0]

        self.train_time["Parse_Arrival_Time"] = pd.to_datetime(
            arrival_time_str,
            format="%H:%M:%S",
            errors="coerce"
        )

        self.train_time["Departure_Time"] = self.train_time["Departure_Time"].astype(str).str.strip()
        self.train_time["Departure_Time"].replace("", pd.NaT, inplace=True)

        departure_time_str = self.train_time["Departure_Time"].astype(str).str.extract(
            r'(\d{2}:\d{2}:\d{2})'
        )[0]

        self.train_time["Parse_Departure_Time"] = pd.to_datetime(
            departure_time_str,
            format="%H:%M:%S",
            errors="coerce"   
        )

        # Add days to arrival time based on Arrival_Date
        self.train_time["Updated_Arrival_Time"] = self.train_time["Parse_Arrival_Time"] + \
            self.train_time["Arrival_Date"].apply(lambda x: timedelta(days=x))


        # Compute real train duration
        self.train_time["train_Duration2"] = (
            self.train_time["Updated_Arrival_Time"] - self.train_time["Parse_Departure_Time"]
        )

        # Compute duration in minutes
        self.train_time["train_Duration_Minutes"] = (
            self.train_time["train_Duration"].dt.total_seconds() / 60
        )

        # od key
        self.train_time["train_od2"] = (
            self.train_time["Departure_Station_Name"].str.replace(" ", "")
            + "_"
            + self.train_time["Arrival_Station_Name"].str.replace(" ", "")
        )

        # Remove duplicates
        # self.train_time = self.train_time.drop_duplicates(subset=["train_od"])
        # self.train_time = self.train_time.drop_duplicates(subset=["train_od2"])

    async def find_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        origin_name,
        destination_name,
        input_departure_hour: str,
        weight_tons: float = 10.0,
        mode: list = ["truck_only"],
        enable_transfer: bool = False,
        max_transfers: int = 10,
        show_all: bool = False,
    ) -> Dict:
        """
        Find optimal route between two points with transfer capability

        Args:
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            dest_lat: Destination latitude
            dest_lon: Destination longitude
            weight_tons: Cargo weight (tons)
            mode: Route type ('all', 'truck_only', 'truck_ship', 'truck_train', 'truck_ship_train', 'truck_train_ship', 'truck_train_ship_train')
            enable_transfer: Enable transfer mode
            max_transfers: Maximum number of transfers (default: 10)
            show_all: Show all routes (False: only optimal)

        Returns:
            Dict containing optimal route information
        """
        print(
            f"Finding route from ({origin_lat}, {origin_lon}) to ({dest_lat}, {dest_lon}) with mode: {mode}"
        )

        # Validate mode parameter
        valid_modes = [
            "all",
            "truck_only",
            "truck_ship",
            "truck_train",
            "truck_ship_train",
            "truck_train_ship",
            "truck_train_ship_train",
        ]
        for m in mode:
            if m not in valid_modes:
                raise ValueError(f"Invalid mode '{m}'. Must be one of: {valid_modes}")

        # Create origin and destination points
        origin_point = Point(origin_lon, origin_lat)
        dest_point = Point(dest_lon, dest_lat)

        # Find nearest ports and stations
        nearest_ports = self._find_nearest_ports(origin_point, dest_point)
        nearest_stations = self._find_nearest_stations(origin_point, dest_point)

        # Calculate routes for different transportation modes
        routes = await self._calculate_routes_by_mode(
            origin_point,
            dest_point,
            nearest_ports,
            nearest_stations,
            origin_name,
            destination_name,
            weight_tons,
            mode,
            input_departure_hour,
            enable_transfer,
            max_transfers,
            show_all,
        )
        
        if 'isError' in routes and routes.get('isError'):
            return {
                'isError': routes.get('isError'),
                'message': routes.get('message')
            }
        else:
            # Find optimal routes
            optimal_routes = self._find_optimal_routes(routes)
            
            result = {
                "origin": {"lat": origin_lat, "lon": origin_lon},
                "destination": {"lat": dest_lat, "lon": dest_lon},
                "weight_tons": 0,
                "routes": routes,
                "optimal_routes": optimal_routes,
                "global_state_info": {
                    'departure_time': globals.GLOBAL_STATE["departure_time"],
                    'arrival_time': globals.GLOBAL_STATE["arrival_time"],
                    'total_time_minutes': globals.GLOBAL_STATE["total_time_minutes"],
                    'total_move_time_minutes': globals.GLOBAL_STATE["total_move_time_minutes"],
                    'total_distance_km': globals.GLOBAL_STATE["total_distance_km"],
                    'total_co2_emissions_grams': globals.GLOBAL_STATE["total_co2_emissions_grams"],
                }
            }
            return result

    def _find_nearest_ports(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest ports to origin and destination points"""
        # Create temporary GeoDataFrame for origin and destination
        temp_gdf = gpd.GeoDataFrame(
            {"name": ["origin", "destination"]},
            geometry=[origin_point, dest_point],
            crs="EPSG:4326",
        )

        # Convert to projected CRS for accurate distance calculations
        temp_gdf = temp_gdf.to_crs("EPSG:3857")  # Web Mercator
        minato_gdf_projected = self.minato_gdf.to_crs("EPSG:3857")

        # Find nearest ports
        origin_ports = gpd.sjoin_nearest(
            temp_gdf.iloc[[0]],
            minato_gdf_projected,
            how="inner",
            distance_col="distance",
        )
        dest_ports = gpd.sjoin_nearest(
            temp_gdf.iloc[[1]],
            minato_gdf_projected,
            how="inner",
            distance_col="distance",
        )

        return {
            "origin_port": origin_ports.iloc[0] if not origin_ports.empty else None,
            "dest_port": dest_ports.iloc[0] if not dest_ports.empty else None,
        }

    def _find_nearest_stations(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest stations to origin and destination points"""
        # Create temporary GeoDataFrame for origin and destination
        temp_gdf = gpd.GeoDataFrame(
            {"name": ["origin", "destination"]},
            geometry=[origin_point, dest_point],
            crs="EPSG:4326",
        )

        # Convert to projected CRS for accurate distance calculations
        temp_gdf = temp_gdf.to_crs("EPSG:3857")  # Web Mercator
        station_gdf_projected = self.station_gdf.to_crs("EPSG:3857")

        # Find nearest stations
        origin_stations = gpd.sjoin_nearest(
            temp_gdf.iloc[[0]],
            station_gdf_projected,
            how="inner",
            distance_col="distance",
        )
        dest_stations = gpd.sjoin_nearest(
            temp_gdf.iloc[[1]],
            station_gdf_projected,
            how="inner",
            distance_col="distance",
        )

        return {
            "origin_station": (
                origin_stations.iloc[0] if not origin_stations.empty else None
            ),
            "dest_station": dest_stations.iloc[0] if not dest_stations.empty else None,
        }

    async def _nearest_station(self, lon: float, lat: float):
        """Find nearest station to given coordinates using database query"""
        return await self._db_query_one("SELECT * FROM nearest_station($1, $2)", (lon, lat))
    
    def _calculate_travel_time(self, input_departure_hour, travel_hours: float, wait: bool = False):
        # --- Parse input departure time ---
        if isinstance(input_departure_hour, int) or isinstance(input_departure_hour, float):
            departure_dt = datetime.strptime(f"{int(input_departure_hour):02d}:00", "%H:%M")
        elif isinstance(input_departure_hour, str):
            departure_dt = datetime.strptime(input_departure_hour, "%H:%M")
        else:
            raise ValueError("input_departure_hour must be an int or HH:MM string")

        # --- Add travel time ---
        arrival_dt = departure_dt + timedelta(hours=travel_hours)

        # --- Add waiting time if needed ---
        if wait:
            arrival_dt += timedelta(hours=1.5)

        # --- Return formatted times ---
        return {
            "departure_time": departure_dt.strftime("%H:%M"),
            "arrival_time": arrival_dt.strftime("%H:%M")
        }

    def _calculate_travel_time2(self, input_departure_hour: str, travel_hours: float):
        """
        input_departure_hour: "HH" hoặc "HH:MM"
        travel_hours: số giờ (float)
        """
        time_str = str(input_departure_hour).strip()
        parts = time_str.split(":")

        try:
            departure_hours = int(parts[0])
            departure_minutes = int(parts[1]) if len(parts) > 1 else 0
        except ValueError as e:
            raise ValueError(
                f"Invalid departure time format: {input_departure_hour}"
            ) from e

        travel_minutes = int(round(travel_hours * 60))

        start_total_minutes = departure_hours * 60 + departure_minutes

        end_total_minutes = (start_total_minutes + travel_minutes) % (24 * 60)

        arrival_hours = end_total_minutes // 60
        arrival_minutes = end_total_minutes % 60

        return {
            "departure_time": f"{departure_hours:02d}:{departure_minutes:02d}",
            "arrival_time": f"{arrival_hours:02d}:{arrival_minutes:02d}",
        }

    async def _calculate_routes_by_mode(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_ports: Dict,
        nearest_stations: Dict,
        origin_name: str,
        destination_name: str,
        weight_tons: float,
        mode: list,
        input_departure_hour: str,
        enable_transfer: bool = False,
        max_transfers: int = 10,
        show_all: bool = False,
    ) -> List[Dict]:
        """Calculate routes by selected mode"""
        routes = []
        
        for mode in mode:

            # Route 1: Truck only
            if mode == "truck_only":
                reset_global_states()
                truck_route = await self._calculate_truck_route(
                    origin_point,
                    dest_point,
                    weight_tons,
                    input_departure_hour,
                    origin_name,
                    destination_name,
                )
                if truck_route:
                    routes.append(truck_route)
                else:
                    return {'isError': True, 'data': [], 'message': 'Truck route not found'}

            # Route 2: Truck + Ship
            if mode == "truck_ship":
                if (
                    nearest_ports["origin_port"] is not None
                    and nearest_ports["dest_port"] is not None
                ):
                    reset_global_states()
                    # Step 1: Find truck routes to nearest ports
                    truck_routes = await self._get_truck_routes_to_ports(
                        origin_point, dest_point, nearest_ports, input_departure_hour, origin_name, destination_name, weight_tons
                    )

                    if truck_routes:
                        # Step 2: Find ship routes between ports (direct or via transfer)
                        ship_routes = self._find_ship_routes_between_ports(
                            nearest_ports, weight_tons, max_transfers, show_all
                        )

                        if ship_routes:
                            # Step 3: Combine truck routes + ship routes
                            combined_routes = self._combine_truck_ship_routes(
                                truck_routes, ship_routes, nearest_ports, weight_tons
                            )
                        
                            routes.extend(combined_routes)
                        else:
                            return {'isError': True, 'data': [], 'message': MESSAGES['ship_route_not_found']}
                    else:
                        return {'isError': True, 'data': [], 'message': 'Truck route not found'}

            # Route 3: Truck + Train
            if mode == "truck_train":
                if (nearest_stations['origin_station'] is not None and 
                    nearest_stations['dest_station'] is not None):
                    reset_global_states()
                    # Step 1: Find truck routes to nearest stations
                    truck_routes = await self._get_truck_routes_to_stations(
                        origin_point, dest_point, nearest_stations, input_departure_hour, origin_name, destination_name, weight_tons
                    )

                    if truck_routes:
                        # Step 2: Find train routes between stations (direct or via transfer)
                        train_routes = self._find_train_routes_between_stations(
                            nearest_stations, weight_tons, max_transfers, show_all
                        )

                        if train_routes:
                            # Step 3: Combine truck routes + train routes
                            combined_routes = self._combine_truck_train_routes(
                                truck_routes, train_routes, nearest_stations, weight_tons
                            )
                            routes.extend(combined_routes)

        # Route 4: Truck + Ship + Train
        if mode == "truck_ship_train":
            reset_global_states()
            # 1. Find origin port
            origin_port = nearest_ports["origin_port"]

            # 2. Find dest port
            dest_port = nearest_ports["dest_port"]

            # 3. Find origin train station
            stO = self._nearest_station(float(dest_port["X"]), float(dest_port["Y"]))

            # 4. Find destination train station
            stD = self._nearest_station(dest_point.x, dest_point.y)

            # Step 1: Find route from origin point to ship port
            truck_O_ptO = await self._get_truck_routes_to_ports(
                origin_point, dest_point, nearest_ports
            )

            if truck_O_ptO:
                isLineString = isinstance(
                    truck_O_ptO["origin_to_port"]["geometry"], LineString
                )
                if truck_O_ptO["origin_to_port"]["geometry"] and isLineString:
                    truck_O_ptO_geometry = truck_O_ptO["origin_to_port"]["geometry"]
                else:
                    truck_O_ptO_geometry = truck_O_ptO["origin_to_port"]["geometry"][
                        "coordinates"
                    ]
            else:
                truck_O_ptO_geometry = LineString()

            truck_O_ptO_distance = truck_O_ptO["origin_to_port"]["distance"]
            truck_O_ptO_co2_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_O_ptO_distance / 1000
            )
            truck_O_ptO_time = truck_O_ptO["origin_to_port"]["time"]

            # Step 2: Find ship route
            ptO_ptD_routes = self._find_ship_routes_between_ports(
                nearest_ports, weight_tons, max_transfers, show_all
            )

            if ptO_ptD_routes and ptO_ptD_routes[0]["transfer_ports"]:
                ptO_ptD_geometry = self._create_ship_coords(
                    origin_port, dest_port, ptO_ptD_routes[0]["transfer_ports"]
                )
            else:
                ptO_ptD_geometry = self._create_ship_coords(origin_port, dest_port, [])

            ptO_ptD_distance = ptO_ptD_routes[0]["route_info"]["distance"]
            ptO_ptD_time = ptO_ptD_routes[0]["route_info"]["time"]

            ptO_ptD_co2_emissions = self._calculate_co2_emissions(
                "ship", weight_tons, ptO_ptD_distance / 1000
            )

            # Step 3: Find route from port -> train station
            route_truck_mm = await self._route_truck_mm(
                float(dest_port["X"]), float(dest_port["Y"]), stO["slon"], stO["slat"]
            )
            if route_truck_mm:
                ptD_stO_routes = LineString(route_truck_mm["geometry"]["coordinates"])
                ptD_stO_distance = route_truck_mm["distance_km"]
            else:
                dest_port_point = (dest_port["X"], dest_port["Y"])
                origin_train_point = (stO["slon"], stO["slat"])
                ptD_stO_routes = LineString([dest_port_point, origin_train_point])
                ptD_stO_distance = self._calculate_distance(
                    float(dest_port["Y"]),
                    float(dest_port["X"]),
                    stO["slat"],
                    stO["slon"],
                )

            ptD_stO_co2_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, ptD_stO_distance / 1000
            )

            # Step 4: Find train route
            train_coords = LineString(
                [(stO["slon"], stO["slat"]), (stD["slon"], stD["slat"])]
            )
            stO_stD_distance = self._calculate_distance(
                stO["slat"], stO["slon"], stD["slat"], stD["slon"]
            )
            stO_stD_co2_emissions = self._calculate_co2_emissions(
                "train", weight_tons, stO_stD_distance / 1000
            )

            # Step 5: Find route from destination station route to destination point
            stD_dest_routes = await self._route_truck_mm(
                stD["slon"], stD["slat"], dest_point.x, dest_point.y
            )
            if stD_dest_routes:
                stD_dest_geometry = LineString(
                    stD_dest_routes["geometry"]["coordinates"]
                )
                stD_dest_distance = stD_dest_routes["distance_km"]
            else:
                stD_point = (stD["slon"], stD["slat"])
                dest_point = (dest_point.x, dest_point.y)
                ptD_stO_routes = LineString([stD_point, dest_point])
                stD_dest_distance = self._calculate_distance(
                    stD["lat"], stD["lon"], dest_point.y, dest_point.x
                )

            stD_dest_co2_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, stD_dest_distance / 1000
            )

            total_distances = self._calc_total_distance(
                [
                    truck_O_ptO_distance,
                    ptO_ptD_distance,
                    ptD_stO_distance,
                    stO_stD_distance,
                    stD_dest_distance,
                ]
            )

            data_infos = {
                "origin_port": origin_port["C02_005"],
                "dest_port": dest_port["C02_005"],
                "origin_station": stO["name"],
                "dest_station": stD["name"],
                "co2_emissions": truck_O_ptO_co2_emissions
                + ptO_ptD_co2_emissions
                + ptD_stO_co2_emissions
                + stO_stD_co2_emissions
                + stD_dest_co2_emissions,
                "ship_time": 0,
                "train_time_minutes": 0,
                "truck_time_minutes": 0,
                "truck_distance_km": truck_O_ptO_distance
                + ptD_stO_distance
                + stD_dest_distance,
            }

            listLineStrings = [
                truck_O_ptO_geometry,
                ptO_ptD_geometry,
                ptD_stO_routes,
                train_coords,
                stD_dest_geometry,
            ]
            combined_routes = self._combine_linestrings(
                mode, listLineStrings, total_distances, data_infos
            )
            routes.extend(combined_routes)

        # Route 5: Truck + Train + Ship + Train
        if mode == "truck_train_ship_train":
            reset_global_states()
            # FIND IMPORTANT POINTS
            # 3. Find origin port
            origin_port = nearest_ports["origin_port"]
            origin_port_point = Point(origin_port["X"], origin_port["Y"])
            # 1. Find FIRST origin train station
            nearest_stations_1 = self._find_nearest_stations(
                origin_point, origin_port_point
            )

            stO_1 = nearest_stations_1["origin_station"]

            # 2. Find FIRST destination train station
            stD_1 = nearest_stations_1["dest_station"]

            # 4. Find dest port
            dest_port = nearest_ports["dest_port"]

            # 5. Find SECOND origin train station (nearest to dest port)
            stO_2 = self._nearest_station(float(dest_port["X"]), float(dest_port["Y"]))

            # 6. Find SECOND destination train station
            stD_2 = self._nearest_station(dest_point.x, dest_point.y)

            # CALCULATING ROUTES
            # 1. Get the truck path from the Origin to stO_1, geometry, distance & CO2 emissions
            truck_route_1 = await self._get_truck_routes_to_stations(
                origin_point, dest_point, nearest_stations
            )
            if truck_route_1:
                print("Debug: Found truck_route_1")

                origin_to_station = truck_route_1.get("origin_to_station", None)
                geom_in = origin_to_station.get("geometry", None)

                truck_route_1_geometry = extract_linestring(geom_in)
                truck_route_1_distance = truck_route_1["origin_to_station"]["distance"]
            else:
                print("Debug: Not found truck_route_1, initializing fallback route")
                stO_1_point = Point(stO_1["lon"], stO_1["lat"])
                truck_route_1_geometry = LineString(origin_point, stO_1_point)
                truck_route_1_distance = self._calculate_distance(
                    origin_point.y, origin_point.x, stO_1["lat"], stO_1["lon"]
                )

            truck_route_1_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_1_distance / 1000
            )

            # 2. Get the train path from stO_1 to stD_1, geometry, distance & CO2 emissions
            train_route_1 = self._find_train_routes_between_stations(
                nearest_stations_1, weight_tons, max_transfers, show_all
            )
            train_route_1_geometry = LineString(
                [
                    (float(stO_1["lon"]), float(stO_1["lat"])),
                    (float(stD_1["lon"]), float(stD_1["lat"])),
                ]
            )
            train_route_1_distance = self._calculate_distance(
                stO_1["lat"], stO_1["lon"], stD_1["lat"], stD_1["lon"]
            )

            if isinstance(train_route_1, (list, LineString)) and train_route_1:
                first = train_route_1[0]

                if isinstance(first, dict):
                    ri = first.get("route_info") or {}
                    train_route_1_distance = ri.get("distance")
                    geom = ri.get("geometry")

                    if geom:
                        print(
                            "Debug: Initializing new value for train_route_1_geometry"
                        )
                        train_route_1_geometry = geom

            train_route_1_emissions = self._calculate_co2_emissions(
                "train", weight_tons, train_route_1_distance / 1000
            )

            # 3. Get the truck path from stD_1 to origin_port, geometry, distance & CO2 emissions
            stD_1_point = Point(float(stD_1["lon"]), float(stD_1["lat"]))
            truck_route_2 = await self._get_truck_routes_to_ports(
                stD_1_point, dest_point, nearest_ports
            )

            if truck_route_2:
                print("Debug: Found truck_route_2")
                truck_route_2_distance = truck_route_2["origin_to_port"]["distance"]
                isLineString = isinstance(
                    truck_route_2["origin_to_port"]["geometry"], LineString
                )

                if truck_route_2["origin_to_port"]["geometry"] and isLineString:
                    truck_route_2_geometry = truck_route_2["origin_to_port"]["geometry"]
                else:
                    truck_route_2_geometry = truck_route_2["origin_to_port"][
                        "geometry"
                    ]["coordinates"]

            else:
                print("Debug: NOT found truck_route_2")
                truck_route_2_geometry = LineString(stD_1_point, origin_port_point)
                truck_route_2_distance = self._calculate_distance(
                    stD_1["lat"], stD_1["lon"], dest_port["Y"], dest_port["X"]
                )

            truck_route_2_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_2_distance / 1000
            )

            # 4. Get the ship path form origin_port to dest_port
            ship_route = self._find_ship_routes_between_ports(
                nearest_ports, weight_tons, max_transfers, show_all
            )

            if ship_route and ship_route[0]["transfer_ports"]:
                ship_route_geometry = self._create_ship_coords(
                    origin_port, dest_port, ship_route[0]["transfer_ports"]
                )
            else:
                ship_route_geometry = self._create_ship_coords(
                    origin_port, dest_port, []
                )

            # Find ship_route_distance
            ship_route_distance = None
            item = (
                ship_route[0]
                if isinstance(ship_route, list) and ship_route
                else ship_route
            )
            if isinstance(item, dict):
                ri = item.get("route_info")
                if isinstance(ri, dict):
                    d = ri.get("distance")
                    if d is not None:
                        try:
                            ship_route_distance = float(d)
                        except:
                            pass

            if ship_route_distance is None:
                ship_route_distance = self._calculate_distance(
                    float(origin_port["Y"]),
                    float(origin_port["X"]),
                    float(dest_port["Y"]),
                    float(dest_port["X"]),
                )

            ship_route_emissions = self._calculate_co2_emissions(
                "ship", weight_tons, ship_route_distance / 1000
            )

            # 5. Get the truck path from dest_port to stO_2
            truck_route_3 = await self._route_truck_mm(
                float(dest_port["X"]),
                float(dest_port["Y"]),
                float(stO_2["slon"]),
                float(stO_2["slat"]),
            )

            if truck_route_3:
                print("Debug: Found truck_route_3")
                truck_route_3_geometry = LineString(
                    truck_route_3["geometry"]["coordinates"]
                )

                truck_route_3_distance = truck_route_3["distance_km"]
            else:
                print(
                    "Debug: Initializing new value for truck_route_3_geometry & truck_route_3_distance"
                )
                dest_port_point = (float(dest_port["X"]), float(dest_port["Y"]))
                origin_train_point = (float(stO_2["slon"]), float(stO_2["slat"]))
                truck_route_3_geometry = LineString(
                    [dest_port_point, origin_train_point]
                )
                truck_route_3_distance = self._calculate_distance(
                    float(dest_port["Y"]),
                    float(dest_port["X"]),
                    float(stO_2["slat"]),
                    float(stO_2["slon"]),
                )

            truck_route_3_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_3_distance / 1000
            )

            # 6. Get the train path from stO_2 to stD_2
            train_route_2 = self._find_train_routes_between_stations(
                nearest_stations, weight_tons, max_transfers, show_all
            )

            train_route_2_geometry = LineString(
                [
                    (float(stO_2["slon"]), float(stO_2["slat"])),
                    (float(stD_2["slon"]), float(stD_2["slat"])),
                ]
            )

            train_route_2_distance = self._calculate_distance(
                float(stO_2["slat"]),
                float(stO_2["slon"]),
                float(stD_2["slat"]),
                float(stD_2["slon"]),
            )

            if train_route_2:
                if isinstance(train_route_2, list) and train_route_2:
                    first = train_route_2[0]

                    if isinstance(first, dict):
                        ri = first.get("route_info") or {}
                        train_route_2_distance = ri.get("distance")
                        geom = ri.get("geometry", None)

                        if geom:
                            train_route_2_geometry = geom

                    train_route_2_distance = first["route_info"]["distance"]

            train_route_2_emissions = self._calculate_co2_emissions(
                "train", weight_tons, train_route_2_distance / 1000
            )

            # 7. Get the truck path from stD_2 to Destination
            truck_route_4 = await self._route_truck_mm(
                float(stD_2["slon"]), float(stD_2["slat"]), dest_point.x, dest_point.y
            )

            if truck_route_4:
                truck_route_4_geometry = LineString(
                    truck_route_4["geometry"]["coordinates"]
                )
                truck_route_4_distance = truck_route_4["distance_km"]
            else:
                stD_2_point = Point(float(stD_2["slon"]), float(stD_2["slat"]))
                truck_route_4_geometry = LineString([stD_2_point, dest_point])
                truck_route_4_distance = self._calculate_distance(
                    float(stD_2["slat"]),
                    float(stD_2["slon"]),
                    dest_point.y,
                    dest_point.x,
                )

            truck_route_4_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_4_distance / 1000
            )

            # 8. Sum up
            total_distances = self._calc_total_distance(
                [
                    truck_route_1_distance,
                    truck_route_2_distance,
                    truck_route_3_distance,
                    truck_route_4_distance,
                    train_route_1_distance,
                    train_route_2_distance,
                    ship_route_distance,
                ]
            )

            emissions = [
                truck_route_1_emissions,
                truck_route_2_emissions,
                truck_route_3_emissions,
                truck_route_4_emissions,
                train_route_1_emissions,
                train_route_2_emissions,
                ship_route_emissions,
            ]

            truck_distances_km = [
                truck_route_1_distance,
                truck_route_2_distance,
                truck_route_3_distance,
                truck_route_4_distance,
            ]

            data_infos = build_data_infos(
                origin_port=origin_port["C02_005"],
                dest_port=dest_port["C02_005"],
                origin_stations=stO_1["Station_Name"] + ", " + stO_2["name"],
                dest_stations=stD_1["Station_Name"] + ", " + stD_2["name"],
                emissions=emissions,
                ship_time=0,
                train_time_minutes=0,
                truck_time_minutes=0,
                truck_distances_km=truck_distances_km,
            )

            listLineStrings = [
                truck_route_1_geometry,
                truck_route_2_geometry,
                truck_route_3_geometry,
                truck_route_4_geometry,
                ship_route_geometry,
                train_route_1_geometry,
                train_route_2_geometry,
            ]

            combined_routes = self._combine_linestrings(
                mode, listLineStrings, total_distances, data_infos
            )

            routes.extend(combined_routes)

        # Route 6: Truck + Train + Ship
        if mode == "truck_train_ship":
            reset_global_states()
            # FIND IMPORTANT POINTS
            # 3. Find origin port
            origin_port = nearest_ports["origin_port"]
            origin_port_point = Point(origin_port["X"], origin_port["Y"])
            # 1. Find FIRST origin train station
            nearest_stations_1 = self._find_nearest_stations(
                origin_point, origin_port_point
            )

            stO_1 = nearest_stations_1["origin_station"]

            # 2. Find FIRST destination train station
            stD_1 = nearest_stations_1["dest_station"]

            # 4. Find dest port
            dest_port = nearest_ports["dest_port"]

            # CALCULATING ROUTES
            # 1. Get the truck path from the Origin to stO_1, geometry, distance & CO2 emissions
            truck_route_1 = await self._get_truck_routes_to_stations(
                origin_point, dest_point, nearest_stations
            )
            if truck_route_1:
                print("Debug: Found truck_route_1")

                origin_to_station = truck_route_1.get("origin_to_station", None)
                geom_in = origin_to_station.get("geometry", None)

                truck_route_1_geometry = extract_linestring(geom_in)
                truck_route_1_distance = truck_route_1["origin_to_station"]["distance"]
            else:
                print("Debug: Not found truck_route_1, initializing fallback route")
                stO_1_point = Point(stO_1["lon"], stO_1["lat"])
                truck_route_1_geometry = LineString(origin_point, stO_1_point)
                truck_route_1_distance = self._calculate_distance(
                    origin_point.y, origin_point.x, stO_1["lat"], stO_1["lon"]
                )

            truck_route_1_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_1_distance / 1000
            )

            # 2. Get the train path from stO_1 to stD_1, geometry, distance & CO2 emissions
            train_route_1 = self._find_train_routes_between_stations(
                nearest_stations_1, weight_tons, max_transfers, show_all
            )
            train_route_1_geometry = LineString(
                [
                    (float(stO_1["lon"]), float(stO_1["lat"])),
                    (float(stD_1["lon"]), float(stD_1["lat"])),
                ]
            )
            train_route_1_distance = self._calculate_distance(
                stO_1["lat"], stO_1["lon"], stD_1["lat"], stD_1["lon"]
            )

            if isinstance(train_route_1, (list, LineString)) and train_route_1:
                first = train_route_1[0]

                if isinstance(first, dict):
                    ri = first.get("route_info") or {}
                    train_route_1_distance = ri.get("distance")
                    geom = ri.get("geometry")

                    if geom:
                        print(
                            "Debug: Initializing new value for train_route_1_geometry"
                        )
                        train_route_1_geometry = geom

            train_route_1_emissions = self._calculate_co2_emissions(
                "train", weight_tons, train_route_1_distance / 1000
            )

            # 3. Get the truck path from stD_1 to origin_port, geometry, distance & CO2 emissions
            stD_1_point = Point(float(stD_1["lon"]), float(stD_1["lat"]))
            truck_route_2 = await self._get_truck_routes_to_ports(
                stD_1_point, dest_point, nearest_ports
            )

            if truck_route_2:
                print("Debug: Found truck_route_2")
                truck_route_2_distance = truck_route_2["origin_to_port"]["distance"]
                isLineString = isinstance(
                    truck_route_2["origin_to_port"]["geometry"], LineString
                )

                if truck_route_2["origin_to_port"]["geometry"] and isLineString:
                    truck_route_2_geometry = truck_route_2["origin_to_port"]["geometry"]
                else:
                    truck_route_2_geometry = truck_route_2["origin_to_port"][
                        "geometry"
                    ]["coordinates"]

            else:
                print("Debug: NOT found truck_route_2")
                truck_route_2_geometry = LineString(stD_1_point, origin_port_point)
                truck_route_2_distance = self._calculate_distance(
                    stD_1["lat"], stD_1["lon"], dest_port["Y"], dest_port["X"]
                )

            truck_route_2_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_2_distance / 1000
            )

            # 4. Get the ship path form origin_port to dest_port
            ship_route = self._find_ship_routes_between_ports(
                nearest_ports, weight_tons, max_transfers, show_all
            )

            if ship_route and ship_route[0]["transfer_ports"]:
                ship_route_geometry = self._create_ship_coords(
                    origin_port, dest_port, ship_route[0]["transfer_ports"]
                )
            else:
                ship_route_geometry = self._create_ship_coords(
                    origin_port, dest_port, []
                )

            # Find ship_route_distance
            ship_route_distance = None
            item = (
                ship_route[0]
                if isinstance(ship_route, list) and ship_route
                else ship_route
            )
            if isinstance(item, dict):
                ri = item.get("route_info")
                if isinstance(ri, dict):
                    d = ri.get("distance")
                    if d is not None:
                        try:
                            ship_route_distance = float(d)
                        except:
                            pass

            if ship_route_distance is None:
                ship_route_distance = self._calculate_distance(
                    float(origin_port["Y"]),
                    float(origin_port["X"]),
                    float(dest_port["Y"]),
                    float(dest_port["X"]),
                )

            ship_route_emissions = self._calculate_co2_emissions(
                "ship", weight_tons, ship_route_distance / 1000
            )

            # 5. Get the truck path from dest_port to Destination
            truck_route_3 = await self._route_truck_mm(
                float(dest_port["X"]), float(dest_port["Y"]), dest_point.x, dest_point.y
            )

            if truck_route_3:
                truck_route_3_geometry = LineString(
                    truck_route_3["geometry"]["coordinates"]
                )
                truck_route_3_distance = truck_route_3["distance_km"]
            else:
                stD_2_point = Point(float(stD_2["slon"]), float(stD_2["slat"]))
                truck_route_3_geometry = LineString([stD_2_point, dest_point])
                truck_route_3_distance = self._calculate_distance(
                    float(stD_2["slat"]),
                    float(stD_2["slon"]),
                    dest_point.y,
                    dest_point.x,
                )

            truck_route_3_emissions = self._calculate_co2_emissions(
                "truck", weight_tons, truck_route_3_distance / 1000
            )

            # 8. Sum up
            total_distances = self._calc_total_distance(
                [
                    truck_route_1_distance,
                    truck_route_2_distance,
                    truck_route_3_distance,
                    train_route_1_distance,
                    ship_route_distance,
                ]
            )

            emissions = [
                truck_route_1_emissions,
                truck_route_2_emissions,
                truck_route_3_emissions,
                train_route_1_emissions,
                ship_route_emissions,
            ]

            truck_distances_km = [
                truck_route_1_distance,
                truck_route_2_distance,
                truck_route_3_distance,
            ]

            data_infos = build_data_infos(
                origin_port=origin_port["C02_005"],
                dest_port=dest_port["C02_005"],
                origin_stations=stO_1["Station_Name"],
                dest_stations=stD_1["Station_Name"],
                emissions=emissions,
                ship_time=0,
                train_time_minutes=0,
                truck_time_minutes=0,
                truck_distances_km=truck_distances_km,
            )

            listLineStrings = [
                truck_route_1_geometry,
                truck_route_2_geometry,
                truck_route_3_geometry,
                ship_route_geometry,
                train_route_1_geometry,
            ]

            combined_routes = self._combine_linestrings(
                mode, listLineStrings, total_distances, data_infos
            )

            routes.extend(combined_routes)

        # print("=" * 100, "\n", routes[0]["geometry"], "\n", "=" * 100)
        return routes

    def _calc_total_distance(self, list_distances):
        return sum(list_distances)

    def _combine_linestrings(
        self,
        mode,
        linestrings,
        total_distances,
        data_infos,
        target_crs="EPSG:4326",
        max_gap_meters=100,
    ):
        if not linestrings:
            return None

        transformer = pyproj.Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )

        segments = []
        indx = 0
        for part in linestrings:
            if part is None:
                continue

            coords = None

            if isinstance(part, str):
                try:
                    geom = wkt.loads(part)
                    if geom.geom_type == "LineString":
                        coords = list(geom.coords)
                    elif geom.geom_type == "MultiLineString":
                        for line in geom.geoms:
                            segment_coords = list(line.coords)
                            normalized = self._normalize_coords(
                                segment_coords, transformer, target_crs
                            )
                            if normalized:
                                segments.append(normalized)
                        continue
                except Exception as e:
                    print(f"Can't parse WKT string: {part[:50]}... - {e}")
                    continue

            elif hasattr(part, "coords"):
                coords = list(part.coords)
                print("processed part number by coords: ", indx)

            elif isinstance(part, (list, tuple)):
                if len(part) > 0:
                    if isinstance(part[0], (list, tuple)):
                        coords = part
                    elif len(part) == 2 and isinstance(part[0], (int, float)):
                        coords = [part]
                    else:
                        coords = part
                print("processed part number by lt: ", indx)

            elif isinstance(part, dict):
                if "type" in part and "coordinates" in part:
                    from shapely.geometry import shape

                    geom = shape(part)
                    coords = list(geom.coords)
                elif "coordinates" in part:
                    coords = part["coordinates"]
                print("processed part number by dict: ", indx)

            if coords and len(coords) >= 1:
                normalized = self._normalize_coords(coords, transformer, target_crs)
                if normalized:
                    segments.append(normalized)

            indx = indx + 1

        if not segments:

            return None

        def distance_meters(p1, p2):
            if target_crs == "EPSG:4326":
                lat1, lon1 = p1[1], p1[0]
                lat2, lon2 = p2[1], p2[0]

                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)

                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(lat1))
                    * math.cos(math.radians(lat2))
                    * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.asin(math.sqrt(a))

                return 6371000 * c
            else:
                return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        merged_segments = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            gap = distance_meters(current_segment[-1], next_segment[0])

            if gap <= max_gap_meters:
                if current_segment[-1] == next_segment[0]:
                    current_segment.extend(next_segment[1:])
                else:
                    current_segment.extend(next_segment)
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment

        merged_segments.append(current_segment)

        results = []
        for idx, segment in enumerate(merged_segments):
            if len(segment) < 2:
                continue

            geometry = LineString(segment)

            if geometry.length == 0:
                continue

            total_distance_m = 0.0
            for a, b in zip(segment, segment[1:]):
                if a == b:
                    continue
                total_distance_m += distance_meters(a, b)

            results.append(
                build_result_segment(
                    idx=idx,
                    merged_count=len(merged_segments),
                    geometry=geometry,
                    total_distance_meters=int(round(total_distance_m)),
                    data_infos=data_infos,
                    mode=mode,
                )
            )
            print("appended part: ", idx)

        return results if results else None

    def _normalize_coords(self, coords, transformer, target_crs):
        normalized = []

        for coord in coords:
            try:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    x = float(coord[0])
                    y = float(coord[1])
                else:
                    continue

                if abs(x) > 200 or abs(y) > 200:
                    if target_crs == "EPSG:4326":
                        x, y = transformer.transform(x, y)
                else:
                    if target_crs == "EPSG:3857":
                        transformer_to_3857 = pyproj.Transformer.from_crs(
                            "EPSG:4326", "EPSG:3857", always_xy=True
                        )
                        x, y = transformer_to_3857.transform(x, y)

                normalized.append((x, y))

            except (ValueError, TypeError, IndexError) as e:
                print(f"coord error: {coord} - {e}")
                continue

        return normalized if len(normalized) >= 2 else None

    async def _calculate_truck_route(
        self,
        origin_point: Point,
        dest_point: Point,
        weight_tons: float,
        input_departure_hour: int,
        origin_name: str,
        destination_name: str,
    ) -> Optional[Dict]:
        """Calculate pure truck route"""
        try:
            # Get truck route from NITAS data
            truck_info = await self._get_truck_route_info(origin_point, dest_point)

            if truck_info:
                distance_km = truck_info["distance"] / 1000
                co2_emissions = self._calculate_co2_emissions(
                    "truck", weight_tons, distance_km
                )

                travel_time_hours = round(truck_info["time"], 2) / 60
                travel_times = self._calculate_travel_time(
                    input_departure_hour, travel_time_hours
                )

                departure_time = travel_times["departure_time"]
                arrival_time = travel_times["arrival_time"]

                globals.GLOBAL_STATE["departure_time"] = departure_time
                globals.GLOBAL_STATE["arrival_time"] = arrival_time
                globals.GLOBAL_STATE["total_time_minutes"] = round(truck_info["time"], 2)
                globals.GLOBAL_STATE["total_move_time_minutes"] = round(truck_info["time"], 2)
                globals.GLOBAL_STATE["total_distance_km"] = round(distance_km, 2)
                globals.GLOBAL_STATE["total_co2_emissions_grams"] = round(co2_emissions, 2)

                return {
                    "mode": "truck_only",
                    "vehicle": VEHICLES['truck'],
                    "departure_time": departure_time,
                    "arrival_time": arrival_time,
                    "origin_name": origin_name,
                    "destination_name": destination_name,
                    "total_time_minutes": (
                        0
                        if np.isnan(truck_info["time"])
                        else round(truck_info["time"], 2)
                    ),
                    "total_distance_meters": (
                        0
                        if np.isnan(truck_info["distance"])
                        else round(truck_info["distance"], 2)
                    ),
                    "total_distance_km": 0 if np.isnan(distance_km) else round(distance_km, 2),
                    "total_co2_emissions_grams": (
                        0 if np.isnan(co2_emissions) else round(co2_emissions, 2)
                    ),
                    "truck_time_minutes": (
                        0
                        if np.isnan(truck_info["time"])
                        else round(truck_info["time"], 2)
                    ),
                    "truck_distance_km": (
                        0 if np.isnan(distance_km) else round(distance_km, 2)
                    ),
                    "geometry": truck_info["geometry"],
                }
        except Exception as e:
            print(f"Error calculating truck route: {e}")

        return None

    async def _calculate_ship_route(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_ports: Dict,
        weight_tons: float,
    ) -> Optional[Dict]:
        """Calculate truck + ship route"""
        try:
            origin_port = nearest_ports["origin_port"]
            dest_port = nearest_ports["dest_port"]

            # Get truck routes to/from ports using DATABASE
            origin_to_port = await self._get_truck_route_info(
                origin_point, Point(origin_port["X"], origin_port["Y"])
            )
            port_to_dest = await self._get_truck_route_info(
                Point(dest_port["X"], dest_port["Y"]), dest_point
            )

            # Get ship route using OLD LOGIC (CSV data)
            ship_info = self._get_ship_route_info(
                origin_port["C02_005"], dest_port["C02_005"]
            )

            if origin_to_port and port_to_dest:
                if ship_info:
                    # Direct ship route found
                    total_time = (
                        origin_to_port["time"]
                        + port_to_dest["time"]
                        + ship_info["time"] * 60  # Convert hours to minutes
                    )

                    total_distance = (
                        origin_to_port["distance"]
                        + port_to_dest["distance"]
                        + ship_info["distance"]
                    )

                    co2_emissions = (
                        self._calculate_co2_emissions(
                            "truck", weight_tons, origin_to_port["distance"] / 1000
                        )
                        + self._calculate_co2_emissions(
                            "truck", weight_tons, port_to_dest["distance"] / 1000
                        )
                        + self._calculate_co2_emissions(
                            "ship", weight_tons, ship_info["distance"] / 1000
                        )
                    )
                else:
                    # No direct ship route, save truck routes for transfer logic
                    self._last_truck_routes_ship = {
                        "origin_to_port": origin_to_port,
                        "port_to_dest": port_to_dest,
                    }
                    self._last_ports = {
                        "origin_port": origin_port,
                        "dest_port": dest_port,
                    }
                    return {
                        "truck_routes": {
                            "origin_to_port": origin_to_port,
                            "port_to_dest": port_to_dest,
                        },
                        "ports": {"origin_port": origin_port, "dest_port": dest_port},
                    }

                # Use actual geometries from database for truck routes
                truck_geom_1 = origin_to_port.get("geometry")
                truck_geom_2 = port_to_dest.get("geometry")

                # Combine geometries: truck1 + ship + truck2
                # Use actual truck geometries from database
                if truck_geom_1 and truck_geom_2:
                    # Use actual truck geometries from database
                    truck1_coords = self._get_geometry_coords(truck_geom_1)
                    truck2_coords = self._get_geometry_coords(truck_geom_2)

                    # Combine: truck1 + ship + truck2
                    # Create simple ship segment between ports
                    ship_coords = [
                        (origin_port["X"], origin_port["Y"]),
                        (dest_port["X"], dest_port["Y"]),
                    ]

                    # Combine all coordinates
                    combined_coords = (
                        truck1_coords + ship_coords + truck2_coords[1:]
                    )  # Skip first point of truck2 to avoid duplication
                    # Fallback to straight line if no truck geometries
                    combined_coords = [
                        (origin_point.x, origin_point.y),
                        (origin_port["X"], origin_port["Y"]),
                        (dest_port["X"], dest_port["Y"]),
                        (dest_point.x, dest_point.y),
                    ]

                # Ensure all coordinates are tuples, not Point objects
                try:
                    # Convert all coordinates to tuples
                    clean_coords = []
                    for coord in combined_coords:
                        if hasattr(coord, "x") and hasattr(coord, "y"):
                            # It's a Point object
                            clean_coords.append((float(coord.x), float(coord.y)))
                        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                            # It's already a tuple/list
                            clean_coords.append((float(coord[0]), float(coord[1])))

                    geometry = LineString(clean_coords)
                except Exception as e:
                    pass
                    # Fallback to simple straight line
                    geometry = LineString(
                        [
                            (origin_point.x, origin_point.y),
                            (origin_port["X"], origin_port["Y"]),
                            (dest_port["X"], dest_port["Y"]),
                            (dest_point.x, dest_point.y),
                        ]
                    )

                return {
                    "mode": "truck_ship",
                    "total_time_minutes": total_time,
                    "total_distance_meters": total_distance,
                    "total_distance_km": total_distance / 1000,
                    "total_co2_emissions_grams": co2_emissions,
                    "origin_port": origin_port["C02_005"],
                    "dest_port": dest_port["C02_005"],
                    "origin_station": "",  # No station for ship route
                    "dest_station": "",  # No station for ship route
                    "transfer_port": "",  # No transfer port
                    "transfer_station": "",  # No transfer station
                    "ship_time_hours": ship_info["time"],
                    "train_time_minutes": 0,  # No train time
                    "geometry": geometry,
                }
        except Exception as e:
            print(f"Error calculating ship route: {e}")

        return None

    async def _calculate_train_route(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_stations: Dict,
        weight_tons: float,
    ) -> Optional[Dict]:
        """Calculate truck + train route"""
        try:
            origin_station = nearest_stations["origin_station"]
            dest_station = nearest_stations["dest_station"]

            # Get truck routes to/from stations using DATABASE FIRST
            origin_to_station = await self._get_truck_route_info(
                origin_point, Point(origin_station["lon"], origin_station["lat"])
            )
            station_to_dest = await self._get_truck_route_info(
                Point(dest_station["lon"], dest_station["lat"]), dest_point
            )

            # Now find train route between stations
            train_info = self._get_train_route_info(
                origin_station["Station_Code"], dest_station["Station_Code"]
            )

            if origin_to_station and station_to_dest:
                if train_info:
                    # Direct train route found
                    total_time = (
                        origin_to_station["time"]
                        + station_to_dest["time"]
                        + train_info["time"]
                    )

                    total_distance = (
                        origin_to_station["distance"]
                        + station_to_dest["distance"]
                        + train_info["distance"] * 1000  # Convert km to meters
                    )

                    co2_emissions = (
                        self._calculate_co2_emissions(
                            "truck", weight_tons, origin_to_station["distance"] / 1000
                        )
                        + self._calculate_co2_emissions(
                            "truck", weight_tons, station_to_dest["distance"] / 1000
                        )
                        + self._calculate_co2_emissions(
                            "train", weight_tons, train_info["distance"]
                        )
                    )
                else:
                    # No direct train route, save truck routes and stations for transfer logic
                    self._last_truck_routes = {
                        "origin_to_station": origin_to_station,
                        "station_to_dest": station_to_dest,
                    }
                    self._last_stations = {
                        "origin_station": origin_station,
                        "dest_station": dest_station,
                    }
                    return {
                        "truck_routes": {
                            "origin_to_station": origin_to_station,
                            "station_to_dest": station_to_dest,
                        },
                        "stations": {
                            "origin_station": origin_station,
                            "dest_station": dest_station,
                        },
                    }

                # Use actual geometries from database for truck routes
                truck_geom_1 = origin_to_station.get("geometry")
                truck_geom_2 = station_to_dest.get("geometry")

                # Create train geometry (straight line between stations)
                train_geom = LineString(
                    [
                        (origin_station["lon"], origin_station["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                    ]
                )

                # Combine geometries: truck1 + train + truck2
                # Use actual truck geometries from database
                if truck_geom_1 and truck_geom_2:
                    # Use actual truck geometries from database
                    truck1_coords = self._get_geometry_coords(truck_geom_1)
                    truck2_coords = self._get_geometry_coords(truck_geom_2)

                    # Combine: truck1 + train + truck2
                    # Create simple train segment between stations
                    train_coords = [
                        (origin_station["lon"], origin_station["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                    ]

                    # Combine all coordinates
                    combined_coords = (
                        truck1_coords + train_coords + truck2_coords[1:]
                    )  # Skip first point of truck2 to avoid duplication
                    # Fallback to straight line if no truck geometries
                    combined_coords = [
                        (origin_point.x, origin_point.y),
                        (origin_station["lon"], origin_station["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                        (dest_point.x, dest_point.y),
                    ]

                # Ensure all coordinates are tuples, not Point objects
                try:
                    # Convert all coordinates to tuples
                    clean_coords = []
                    for coord in combined_coords:
                        if hasattr(coord, "x") and hasattr(coord, "y"):
                            # It's a Point object
                            clean_coords.append((float(coord.x), float(coord.y)))
                        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                            # It's already a tuple/list
                            clean_coords.append((float(coord[0]), float(coord[1])))

                    geometry = LineString(clean_coords)
                except Exception as e:
                    pass
                    # Fallback to simple straight line
                    geometry = LineString(
                        [
                            (origin_point.x, origin_point.y),
                            (origin_station["lon"], origin_station["lat"]),
                            (dest_station["lon"], dest_station["lat"]),
                            (dest_point.x, dest_point.y),
                        ]
                    )

                return {
                    "mode": "truck_train",
                    "total_time_minutes": total_time,
                    "total_distance_meters": total_distance,
                    "total_distance_km": total_distance / 1000,
                    "total_co2_emissions_grams": co2_emissions,
                    "origin_port": "",  # No port for train route
                    "dest_port": "",  # No port for train route
                    "origin_station": origin_station["Station_Name"],
                    "dest_station": dest_station["Station_Name"],
                    "transfer_port": "",  # No transfer port
                    "transfer_station": "",  # No transfer station
                    "ship_time_hours": 0,  # No ship time
                    "train_time_minutes": train_info["time"],
                    "geometry": geometry,
                }
        except Exception as e:
            print(f"Error calculating train route: {e}")

        return None

    async def _calculate_train_route_modified(
        self,
        input_departure_hour: str,
        origin_point: Point,
        dest_point: Point,
        origin_name: str,
        destination_name: str,
        nearest_stations: Dict,
        weight_tons: float,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        try:
            origin_station = nearest_stations["origin_station"]
            dest_station = nearest_stations["dest_station"]

            origin_to_station = await self._get_truck_route_info(
                origin_point,
                Point(origin_station["lon"], origin_station["lat"]),
            )

            station_to_dest = await self._get_truck_route_info(
                Point(dest_station["lon"], dest_station["lat"]),
                dest_point,
            )

            train_info = self._get_train_route_info_test(
                origin_station["Station_Name"],
                dest_station["Station_Name"],
            )

            if not origin_to_station or not station_to_dest or not train_info:
                if origin_to_station and station_to_dest:
                    self._last_truck_routes = {
                        "origin_to_station": origin_to_station,
                        "station_to_dest": station_to_dest,
                    }
                    self._last_stations = {
                        "origin_station": origin_station,
                        "dest_station": dest_station,
                    }
                return None, None, None

            co2_truck_1 = round(
                self._calculate_co2_emissions(
                    "truck",
                    weight_tons,
                    origin_to_station["distance"] / 1000,
                ),
                2,
            )
            co2_truck_2 = round(
                self._calculate_co2_emissions(
                    "truck",
                    weight_tons,
                    station_to_dest["distance"] / 1000,
                ),
                2,
            )
            co2_train = round(
                self._calculate_co2_emissions(
                    "train",
                    weight_tons,
                    train_info["distance"],
                ),
                2,
            )

            truck_geom_1 = origin_to_station.get("geometry")
            if not truck_geom_1:
                truck_geom_1 = LineString(
                    [
                        (origin_point.x, origin_point.y),
                        (origin_station["lon"], origin_station["lat"]),
                    ]
                )

            train_geom = LineString(
                [
                    (origin_station["lon"], origin_station["lat"]),
                    (dest_station["lon"], dest_station["lat"]),
                ]
            )

            truck_geom_2 = station_to_dest.get("geometry")
            if not truck_geom_2:
                truck_geom_2 = LineString(
                    [
                        (dest_station["lon"], dest_station["lat"]),
                        (dest_point.x, dest_point.y),
                    ]
                )

            # Calculate truck_1 arrival time
            truck_1_travel_time_hours = round(origin_to_station["time"], 2) / 60
            truck_1_travel_times = self._calculate_travel_time(
                input_departure_hour,
                truck_1_travel_time_hours,
            )
            truck_1_departure_time = truck_1_travel_times["departure_time"]
            truck_1_arrival_time = truck_1_travel_times["arrival_time"]

            # Calculate train arrival time
            def to_hhmm(t):
                t = str(t)
                if " " in t:  # "1900-01-02 11:04:52"
                    return t.split(" ")[1][:5]
                return t[:5]

            train_departure_time_raw = train_info["departure_time"]
            train_arrival_time_raw = train_info["arrival_time"]

            train_departure_time = to_hhmm(train_departure_time_raw)
            train_arrival_time = to_hhmm(train_arrival_time_raw)

            truck_route_1 = {
                "vehicle": VEHICLES['truck'],
                "origin_name": origin_name,
                "destination_name": origin_station["Station_Name"],
                "departure_time": truck_1_departure_time,
                "arrival_time": truck_1_arrival_time,
                "total_wait_time_before_departure_minutes": 0,
                "total_time_minutes": round(origin_to_station["time"], 2),
                "total_distance_meters": origin_to_station["distance"],
                "total_distance_km": round(origin_to_station["distance"] / 1000, 2),
                "total_co2_emissions_grams": co2_truck_1,
                "geometry": truck_geom_1,
            }
            globals.GLOBAL_STATE['departure_time'] = truck_1_departure_time
            globals.GLOBAL_STATE['total_move_time_minutes'] += truck_route_1['total_time_minutes']
            globals.GLOBAL_STATE['total_time_minutes'] += (truck_route_1['total_time_minutes'] + truck_route_1['total_wait_time_before_departure_minutes'])
            globals.GLOBAL_STATE['total_distance_km'] += truck_route_1['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += truck_route_1['total_co2_emissions_grams']

            train_route = {
                "vehicle": VEHICLES["train"],
                "departure_time": train_departure_time,
                "arrival_time": train_arrival_time,
                "origin_name": origin_station["Station_Name"],
                "destination_name": dest_station["Station_Name"],
                "total_wait_time_before_departure_minutes": calc_wait_minutes(truck_route_1['arrival_time'], train_departure_time),
                "total_time_minutes": round(train_info["time"], 2),
                "total_distance_meters": (train_info["distance"] * 1.5) * 1000,
                "total_distance_km": round((train_info["distance"] * 1.5), 2),
                "total_co2_emissions_grams": co2_train,
                "geometry": train_geom,
            }
            globals.GLOBAL_STATE['total_time_minutes'] += (train_route['total_time_minutes'] + train_route['total_wait_time_before_departure_minutes'])
            globals.GLOBAL_STATE['total_move_time_minutes'] += train_route['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += train_route['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += train_route['total_co2_emissions_grams']

            truck_2_departure_time = add_hours(train_arrival_time, 1.5)
            truck2_travel_time = self._calculate_travel_time(truck_2_departure_time, round(station_to_dest["time"], 2) / 60, False)
            truck_route_2 = {
                "vehicle": VEHICLES['truck'],
                "departure_time": truck_2_departure_time,
                "arrival_time": truck2_travel_time['arrival_time'],
                "origin_name": dest_station["Station_Name"],
                "destination_name": destination_name,
                "total_wait_time_before_departure_minutes": calc_wait_minutes(train_route['arrival_time'], truck2_travel_time['departure_time']),
                "total_time_minutes": round(station_to_dest["time"], 2),
                "total_distance_meters": station_to_dest["distance"],
                "total_distance_km": round(station_to_dest["distance"] / 1000, 2),
                "total_co2_emissions_grams": co2_truck_2,
                "geometry": truck_geom_2,
            }
            globals.GLOBAL_STATE['arrival_time'] = truck2_travel_time['arrival_time']
            globals.GLOBAL_STATE['total_time_minutes'] += (truck_route_2['total_time_minutes'] + truck_route_2['total_wait_time_before_departure_minutes'])
            globals.GLOBAL_STATE['total_move_time_minutes'] += truck_route_2['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += truck_route_2['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += truck_route_2['total_co2_emissions_grams']

            return truck_route_1, train_route, truck_route_2

        except Exception as e:
            print(f"Error calculating train route: {e}")
            return None, None, None

    async def _get_truck_routes_to_stations(
        self, origin_point: Point, dest_point: Point, nearest_stations: Dict, input_departure_hour: str = '', origin_name: str = '', destination_name: str = '', weight_tons: float = 0
    ) -> Optional[Dict]:
        """Step 1: Find truck routes to nearest stations"""
        try:
            origin_station = nearest_stations["origin_station"]
            dest_station = nearest_stations["dest_station"]

            origin_to_station = await self._get_truck_route_info(
                origin_point, Point(origin_station["lon"], origin_station["lat"])
            )
            station_to_dest = await self._get_truck_route_info(
                Point(dest_station["lon"], dest_station["lat"]), dest_point
            )

            if origin_to_station and station_to_dest:
                truck_routes = {
                    "origin_to_station": origin_to_station,
                    "station_to_dest": station_to_dest,
                    "origin_station": origin_station,
                    "dest_station": dest_station,
                }

                origin_distance_km = 0 if np.isnan(origin_to_station["distance"]) else round(origin_to_station["distance"]/1000, 2)
                origin_co2_emissions = self._calculate_co2_emissions('truck', weight_tons, origin_distance_km)
                origin_travel_time = self._calculate_travel_time(input_departure_hour, round(origin_to_station["time"]/60, 2), False)
                
                globals.GLOBAL_STATE["departure_time"] = origin_travel_time['departure_time']
                globals.GLOBAL_STATE["arrival_time_buffer_wait_time"] = add_hours(origin_travel_time['arrival_time'], 1.5)
                globals.GLOBAL_STATE["arrival_time_previous_route"] = origin_travel_time['arrival_time']
                
                origin_to_station_routes = {
                    "departure_time": origin_travel_time['departure_time'],
                    "arrival_time": origin_travel_time['arrival_time'],
                    "total_wait_time_before_departure_minutes": 0,
                    "origin_name": origin_name,
                    "destination_name": origin_station["Station_Name"],
                    "total_time_minutes": (
                        0 if np.isnan(origin_to_station["time"]) else round(origin_to_station["time"], 2)
                    ),
                    "total_distance_meters": (
                        0
                        if np.isnan(origin_to_station["distance"])
                        else round(origin_to_station["distance"], 2)
                    ),
                    "total_distance_km": origin_distance_km,
                    "total_co2_emissions_grams": (
                        0 if np.isnan(origin_co2_emissions) else round(origin_co2_emissions, 2)
                    ),
                    "geometry": origin_to_station["geometry"],
                }
                truck_routes['origin_to_station_routes'] = origin_to_station_routes

                dest_distance_km = 0 if np.isnan(station_to_dest["distance"]) else round(station_to_dest["distance"]/1000, 2)
                dest_co2_emissions = self._calculate_co2_emissions('truck', weight_tons, dest_distance_km)
                dest_travel_time = self._calculate_travel_time(input_departure_hour, station_to_dest["time"]/60, False)
                dest_travel_time_wait = self._calculate_travel_time(input_departure_hour, round(station_to_dest["time"]/60, 2), True)
                
                station_to_dest_routes = {
                    "departure_time": dest_travel_time['departure_time'],
                    "arrival_time": dest_travel_time['arrival_time'],
                    "total_wait_time_before_departure": dest_travel_time_wait['arrival_time'],
                    "origin_name": dest_station['Station_Name'],
                    "destination_name": destination_name,
                    "total_time_minutes": (
                        0 if np.isnan(station_to_dest["time"]) else round(station_to_dest["time"], 2)
                    ),
                    "total_distance_meters": (
                        0
                        if np.isnan(station_to_dest["distance"])
                        else round(station_to_dest["distance"], 2)
                    ),
                    "total_distance_km": origin_distance_km,
                    "total_co2_emissions_grams": (
                        0 if np.isnan(dest_co2_emissions) else round(dest_co2_emissions, 2)
                    ),
                    "geometry": station_to_dest["geometry"],
                }
                truck_routes['station_to_dest_routes'] = station_to_dest_routes
                return truck_routes
            else:
                return None

        except Exception as e:
            print(f"Error getting truck routes to stations: {e}")
            return None

    def _find_train_routes_between_stations(
        self,
        nearest_stations: Dict,
        weight_tons: float,
        max_transfers: int,
        show_all: bool,
    ) -> List[Dict]:
        """Step 2: Find train routes between stations (direct or via transfer)"""
        try:
            origin_station = nearest_stations["origin_station"]
            dest_station = nearest_stations["dest_station"]

            # Try to find direct train route first
            direct_train = self._get_train_route_info(
                origin_station["Station_Code"], dest_station["Station_Code"]
            )

            if direct_train:
                co2_emissions = self._calculate_co2_emissions('train', weight_tons, direct_train["distance"] / 1000)
                total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], direct_train['departure_time'])
                globals.GLOBAL_STATE["arrival_time_previous_route"] = direct_train['arrival_time']
                return [
                    {
                        "type": "direct",
                        "route_info": direct_train,
                        "transfer_stations": [],
                        "vehicle": VEHICLES['train'],
                        "departure_time": direct_train['departure_time'],
                        "arrival_time": direct_train['arrival_time'],
                        "origin_name": origin_station["Station_Name"],
                        "destination_name": dest_station["Station_Name"],
                        "total_wait_time_before_departure_minutes": total_wait_time_before_departure_minutes,
                        "total_time_minutes": (
                            0 if np.isnan(direct_train['time']) else round(direct_train['time'], 2)
                        ),
                        "total_distance_meters": (
                            0
                            if np.isnan(direct_train["distance"])
                            else round(direct_train["distance"], 2)
                        ),
                        "total_distance_km": 0 if np.isnan(direct_train["distance"]) else round(direct_train["distance"] / 1000, 2),
                        "total_co2_emissions_grams": co2_emissions
                    }
                ]
            else:
                # Find train routes via transfer
                transfer_routes = self._find_train_transfer_routes(
                    origin_station, dest_station, max_transfers, weight_tons
                )
                return transfer_routes

        except Exception as e:
            print(f"Error finding train routes between stations: {e}")
            return []

    def _combine_truck_train_routes(
        self,
        truck_routes: Dict,
        train_routes: List[Dict],
        nearest_stations: Dict,
        weight_tons: float,
    ) -> List[Dict]:
        """Step 3: Combine truck routes + train routes"""
        try:
            combined_routes = []
            origin_station = nearest_stations["origin_station"]
            dest_station = nearest_stations["dest_station"]

            for train_route in train_routes:
                if train_route["type"] == "direct":
                    # Direct train route
                    route = self._create_combined_route(
                        truck_routes,
                        train_route["route_info"],
                        train_route,
                        origin_station,
                        dest_station,
                        weight_tons,
                        [],
                    )
                    if route:
                        combined_routes.append(route)
                else:
                    # Transfer train route
                    route = self._create_combined_route(
                        truck_routes,
                        train_route["route_info"],
                        train_route,
                        origin_station,
                        dest_station,
                        weight_tons,
                        train_route["transfer_stations"],
                    )
                    if route:
                        combined_routes.append(route)

            return combined_routes

        except Exception as e:
            print(f"Error combining truck train routes: {e}")
            return []

    def _find_train_transfer_routes(
        self, origin_station: Dict, dest_station: Dict, max_transfers: int, weight_tons: float
    ) -> List[Dict]:
        """Find train routes via transfer"""
        try:
            # Logic to find transfer routes (using old logic but only for train routes)
            # Find all stations reachable from origin station
            from_origin = self.train_time[
                self.train_time["Departure_Station_Code"]
                == origin_station["Station_Code"]
            ]["Arrival_Station_Code"].unique()

            # Find all stations that can reach destination station
            to_dest = self.train_time[
                self.train_time["Arrival_Station_Code"] == dest_station["Station_Code"]
            ]["Departure_Station_Code"].unique()

            # Find transfer stations
            transfer_stations = set(from_origin) & set(to_dest)

            routes = []
            for transfer_station in list(transfer_stations)[:max_transfers]:
                # Find train routes via transfer
                leg1 = self._get_train_route_info(
                    origin_station["Station_Code"], transfer_station
                )
                leg2 = self._get_train_route_info(
                    transfer_station, dest_station["Station_Code"]
                )

                if leg1 and leg2:
                    # Calculate total information
                    total_time = leg1["time"] + leg2["time"]
                    total_distance = leg1["distance"] + leg2["distance"]

                    leg1_co2_emissions = self._calculate_co2_emissions('train', weight_tons, (leg1["distance"] / 1000) * 1.5)
                    leg2_co2_emissions = self._calculate_co2_emissions('train', weight_tons, (leg2["distance"] / 1000) * 1.5)

                    leg1_total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], leg1['departure_time'])
                    globals.GLOBAL_STATE["arrival_time_previous_route"] = leg1['arrival_time']

                    leg2_total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], leg2['departure_time'])
                    globals.GLOBAL_STATE["arrival_time_previous_route"] = leg2['arrival_time']


                    routes.append(
                        {
                            "type": "transfer",
                            "route_info": {
                                "time": total_time,
                                "distance": total_distance,
                            },
                            "transfer_stations": [str(transfer_stations)],
                            "leg1": {
                                "vehicle": VEHICLES['train'],
                                "origin_name": origin_station["Station_Name"],
                                "destination_name": transfer_station,
                                "departure_time": leg1['departure_time'],
                                "arrival_time": leg1['arrival_time'],
                                "total_wait_time_before_departure_minutes": leg1_total_wait_time_before_departure_minutes,
                                "total_time_minutes": (
                                    0 if np.isnan(leg1['time']) else round(leg1['time'], 2)
                                ),
                                "total_distance_km": 0 if np.isnan(leg1["distance"]) else round((leg1["distance"] / 1000)*1.5, 2),
                                "total_co2_emissions_grams": leg1_co2_emissions
                            },
                            "leg2": {
                                "vehicle": VEHICLES['train'],
                                "origin_name": transfer_station,
                                "destination_name": dest_station["Station_Name"],
                                "departure_time": leg2['departure_time'],
                                "arrival_time": leg2['arrival_time'],
                                "total_wait_time_before_departure_minutes": leg2_total_wait_time_before_departure_minutes,
                                "total_time_minutes": (
                                    0 if np.isnan(leg2['time']) else round(leg2['time'], 2)
                                ),
                                "total_distance_km": 0 if np.isnan(leg2["distance"]) else round((leg2["distance"] / 1000)*1.5, 2),
                                "total_co2_emissions_grams": leg2_co2_emissions
                            }
                        }
                    )

            return routes

        except Exception as e:
            print(f"Error finding train transfer routes: {e}")
            return []

    def _create_combined_route(
        self,
        truck_routes: Dict,
        train_info: Dict,
        train_route: Dict,
        origin_station: Dict,
        dest_station: Dict,
        weight_tons: float,
        transfer_stations: List,
    ) -> Optional[Dict]:
        """Create combined truck + train route"""
        try:
            origin_to_station = truck_routes["origin_to_station"]
            station_to_dest = truck_routes["station_to_dest"]

            # Calculate total time and distance
            total_time = (
                origin_to_station["time"] + station_to_dest["time"] + train_info["time"]
            )

            total_distance = (
                origin_to_station["distance"] + station_to_dest["distance"] + train_info["distance"] * 1000  # Convert km to meters
            )

            # Calculate CO2 emissions
            co2_emissions = (
                self._calculate_co2_emissions(
                    "truck", weight_tons, origin_to_station["distance"] / 1000
                )
                + self._calculate_co2_emissions(
                    "truck", weight_tons, station_to_dest["distance"] / 1000
                )
                + self._calculate_co2_emissions(
                    "train", weight_tons, train_info["distance"]
                )
            )

            # Create combined geometry
            geometry = self._create_combined_geometry(
                origin_to_station,
                station_to_dest,
                origin_station,
                dest_station,
                transfer_stations,
            )

            # Create train geometry
            train_geometry = self._create_train_coords(origin_station, dest_station, transfer_stations)

            # Create route name
            if transfer_stations:
                train_segments = self.split_coords_into_segments(train_geometry)
                route_name = (
                    f"Truck + Train (transfer via {', '.join(transfer_stations)})"
                )
                mode = "truck_train_transfer"
            else:
                route_name = "Truck + Train"
                mode = "truck_train"
            
            features = []
            # ---- Feature 1: Truck (Origin → Origin Station)
            f1 = {
                "vehicle": VEHICLES['truck'],
                "departure_time": truck_routes['origin_to_station_routes']['departure_time'],
                "arrival_time": truck_routes['origin_to_station_routes']['arrival_time'],
                "total_wait_time_before_departure_minutes": truck_routes['origin_to_station_routes']['total_wait_time_before_departure_minutes'],
                "origin_name": truck_routes['origin_to_station_routes']['origin_name'],
                "destination_name": truck_routes['origin_to_station_routes']['destination_name'],
                "total_time_minutes": (
                    0 if np.isnan(truck_routes['origin_to_station_routes']['total_time_minutes']) else truck_routes['origin_to_station_routes']['total_time_minutes']
                ),
                "total_distance_meters": (
                    0
                    if np.isnan(truck_routes['origin_to_station_routes']['total_distance_meters'])
                    else truck_routes['origin_to_station_routes']['total_distance_meters']
                ),
                "total_distance_km": 0 if np.isnan(truck_routes['origin_to_station_routes']['total_distance_km']) else truck_routes['origin_to_station_routes']['total_distance_km'],
                "total_co2_emissions_grams": (
                    0 if np.isnan(truck_routes['origin_to_station_routes']['total_co2_emissions_grams']) else truck_routes['origin_to_station_routes']['total_co2_emissions_grams']
                ),
                "geometry": origin_to_station["geometry"],
            }
            features.append(f1)
            globals.GLOBAL_STATE['total_time_minutes'] += f1['total_time_minutes'] + f1['total_wait_time_before_departure_minutes']
            globals.GLOBAL_STATE['total_move_time_minutes'] += f1['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += f1['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += f1['total_co2_emissions_grams']

            if transfer_stations:
                f2 = {
                    "vehicle": VEHICLES['train'],
                    "departure_time": train_route['leg1']['departure_time'],
                    "arrival_time": train_route['leg1']['arrival_time'],
                    "total_wait_time_before_departure_minutes": train_route['leg1']['total_wait_time_before_departure_minutes'],
                    "origin_name": train_route['leg1']['origin_name'],
                    "destination_name": train_route['leg1']['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(train_route['leg1']['total_time_minutes']) else train_route['leg1']['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(train_route['leg1']['total_distance_km']) else train_route['leg1']['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(train_route['leg1']['total_co2_emissions_grams']) else train_route['leg1']['total_co2_emissions_grams']
                    ),
                    "geometry": train_segments[0] if transfer_stations else train_geometry,
                }
                features.append(f2)
                globals.GLOBAL_STATE['total_time_minutes'] += (f2['total_time_minutes'] + f2['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f2['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f2['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f2['total_co2_emissions_grams']

                f3 = {
                    "vehicle": VEHICLES['train'],
                    "departure_time": train_route['leg2']['departure_time'],
                    "arrival_time": train_route['leg2']['arrival_time'],
                    "total_wait_time_before_departure_minutes": train_route['leg2']['total_wait_time_before_departure_minutes'],
                    "origin_name": train_route['leg2']['origin_name'],
                    "destination_name": train_route['leg2']['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(train_route['leg2']['total_time_minutes']) else train_route['leg2']['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(train_route['leg2']['total_distance_km']) else train_route['leg2']['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(train_route['leg2']['total_co2_emissions_grams']) else train_route['leg2']['total_co2_emissions_grams']
                    ),
                    "geometry": train_segments[1],
                }
                features.append(f3)
                globals.GLOBAL_STATE['total_time_minutes'] += (f3['total_time_minutes'] + f3['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f3['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f3['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f3['total_co2_emissions_grams']
            else:
                f2 = {
                    "vehicle": VEHICLES['train'],
                    "departure_time": train_route['departure_time'],
                    "arrival_time": train_route['arrival_time'],
                    "total_wait_time_before_departure_minutes": train_route['total_wait_time_before_departure_minutes'],
                    "origin_name": train_route['origin_name'],
                    "destination_name": train_route['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(train_route['total_time_minutes']) else train_route['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(train_route['total_distance_km']) else train_route['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(train_route['total_co2_emissions_grams']) else train_route['total_co2_emissions_grams']
                    ),
                    "geometry": train_segments[0] if transfer_stations else train_geometry,
                }
                features.append(f2)
                globals.GLOBAL_STATE['total_time_minutes'] += (f2['total_time_minutes'] + f2['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f2['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f2['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f2['total_co2_emissions_grams']

            ship_arrival_time = f3['arrival_time'] if transfer_stations else f2['arrival_time']
            truck_departure_time = add_hours(ship_arrival_time, 1.5)
            truck_travel_time = self._calculate_travel_time(truck_departure_time, truck_routes['station_to_dest_routes']['total_time_minutes'] / 60, False)
            globals.GLOBAL_STATE["arrival_time"] = truck_travel_time['arrival_time']
            f4 = {
                "vehicle": VEHICLES['truck'],
                "departure_time": truck_departure_time,
                "arrival_time": truck_travel_time['arrival_time'],
                "total_wait_time_before_departure_minutes": calc_wait_minutes(globals.GLOBAL_STATE['arrival_time_previous_route'], truck_departure_time),
                "origin_name": truck_routes['station_to_dest_routes']['origin_name'],
                "destination_name": truck_routes['station_to_dest_routes']['destination_name'],
                "total_time_minutes": (
                    0 if np.isnan(truck_routes['station_to_dest_routes']['total_time_minutes']) else truck_routes['station_to_dest_routes']['total_time_minutes']
                ),
                "total_distance_meters": (
                    0
                    if np.isnan(truck_routes['station_to_dest_routes']['total_distance_meters'])
                    else truck_routes['station_to_dest_routes']['total_distance_meters']
                ),
                "total_distance_km": 0 if np.isnan(truck_routes['station_to_dest_routes']['total_distance_km']) else truck_routes['station_to_dest_routes']['total_distance_km'],
                "total_co2_emissions_grams": (
                    0 if np.isnan(truck_routes['station_to_dest_routes']['total_co2_emissions_grams']) else truck_routes['station_to_dest_routes']['total_co2_emissions_grams']
                ),
                "geometry": station_to_dest.get("geometry"),
            }
            features.append(f4)
            globals.GLOBAL_STATE['total_time_minutes'] += (f4['total_time_minutes'] + f4['total_wait_time_before_departure_minutes'])
            globals.GLOBAL_STATE['total_move_time_minutes'] += f4['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += f4['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += f4['total_co2_emissions_grams']

            return {
                "mode": mode,
                "total_time_minutes": total_time,
                "total_distance_meters": total_distance,
                "total_distance_km": total_distance / 1000,
                "total_co2_emissions_grams": co2_emissions,
                "origin_port": "",
                "dest_port": "",
                "origin_station": origin_station["Station_Name"],
                "dest_station": dest_station["Station_Name"],
                "transfer_port": "",
                "transfer_station": ", ".join(transfer_stations) if transfer_stations else "",
                "ship_time_hours": 0,
                "train_time_minutes": train_info["time"],
                "truck_time_minutes": origin_to_station["time"] + station_to_dest["time"],
                "truck_distance_km": (origin_to_station["distance"] + station_to_dest["distance"])/ 1000,
                "geometry": geometry,
                "features": features
            }

        except Exception as e:
            print(f"Error creating combined route: {e}")
            return None

    def _create_combined_geometry(
        self,
        origin_to_station: Dict,
        station_to_dest: Dict,
        origin_station: Dict,
        dest_station: Dict,
        transfer_stations: List,
    ) -> LineString:
        """Create combined truck + train geometry"""
        try:
            # Get truck geometries
            truck_geom_1 = origin_to_station.get("geometry")
            truck_geom_2 = station_to_dest.get("geometry")

            if truck_geom_1 and truck_geom_2:
                # Use truck geometries from database
                truck1_coords = self._get_geometry_coords(truck_geom_1)
                truck2_coords = self._get_geometry_coords(truck_geom_2)

                # Create train segment
                if transfer_stations:
                    # Train route via transfer - need to find coordinates of transfer stations
                    train_coords = [(origin_station["lon"], origin_station["lat"])]

                    # Add coordinates of transfer stations
                    for transfer_station in transfer_stations:
                        # Find coordinates of transfer station
                        transfer_station_info = self.station_gdf[
                            self.station_gdf["Station_Code"] == transfer_station
                        ]
                        if not transfer_station_info.empty:
                            transfer_coords = (
                                transfer_station_info["lon"].iloc[0],
                                transfer_station_info["lat"].iloc[0],
                            )
                            train_coords.append(transfer_coords)

                    # Add final destination
                    train_coords.append((dest_station["lon"], dest_station["lat"]))
                else:
                    # Direct train route
                    train_coords = [
                        (origin_station["lon"], origin_station["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                    ]

                # Combine: truck1 + train + truck2
                combined_coords = truck1_coords + train_coords + truck2_coords[1:]

                # Ensure coordinates are tuples
                clean_coords = []
                for coord in combined_coords:
                    if hasattr(coord, "x") and hasattr(coord, "y"):
                        clean_coords.append((float(coord.x), float(coord.y)))
                    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                        clean_coords.append((float(coord[0]), float(coord[1])))

                return LineString(clean_coords)
            else:
                # Fallback: straight line
                return LineString(
                    [
                        (
                            origin_to_station.get("start_point", Point(0, 0)).x,
                            origin_to_station.get("start_point", Point(0, 0)).y,
                        ),
                        (origin_station["lon"], origin_station["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                        (
                            station_to_dest.get("end_point", Point(0, 0)).x,
                            station_to_dest.get("end_point", Point(0, 0)).y,
                        ),
                    ]
                )

        except Exception as e:
            print(f"Error creating combined geometry: {e}")
            # Fallback to simple straight line
            return LineString([(0, 0), (0, 0), (0, 0), (0, 0)])

    async def _get_truck_routes_to_ports(
        self, origin_point: Point, dest_point: Point, nearest_ports: Dict, input_departure_hour: str = '', origin_name: str = '', destination_name: str = '', weight_tons: float = 0
    ) -> Optional[Dict]:
        """Step 1: Find truck routes to nearest ports"""
        try:
            origin_port = nearest_ports["origin_port"]
            dest_port = nearest_ports["dest_port"]

            origin_to_port = await self._get_truck_route_info(
                origin_point, Point(origin_port["X"], origin_port["Y"])
            )
            port_to_dest = await self._get_truck_route_info(
                Point(dest_port["X"], dest_port["Y"]), dest_point
            )

            if origin_to_port and port_to_dest:
                truck_routes = {
                    "origin_to_port": origin_to_port,
                    "port_to_dest": port_to_dest,
                    "origin_port": origin_port,
                    "dest_port": dest_port,
                }
                
                origin_distance_km = 0 if np.isnan(origin_to_port["distance"]) else round(origin_to_port["distance"]/1000, 2)
                origin_co2_emissions = self._calculate_co2_emissions('truck', weight_tons, origin_distance_km)
                origin_travel_time = self._calculate_travel_time(input_departure_hour, round(origin_to_port["time"]/60, 2), False)
                
                globals.GLOBAL_STATE["departure_time"] = origin_travel_time['departure_time']
                globals.GLOBAL_STATE["arrival_time_buffer_wait_time"] = add_hours(origin_travel_time['arrival_time'], 1.5)
                globals.GLOBAL_STATE["arrival_time_previous_route"] = origin_travel_time['arrival_time']
                
                origin_to_port_routes = {
                    "departure_time": origin_travel_time['departure_time'],
                    "arrival_time": origin_travel_time['arrival_time'],
                    "total_wait_time_before_departure_minutes": 0,
                    "origin_name": origin_name,
                    "destination_name": origin_port["C02_005"],
                    "total_time_minutes": (
                        0 if np.isnan(origin_to_port["time"]) else round(origin_to_port["time"], 2)
                    ),
                    "total_distance_meters": (
                        0
                        if np.isnan(origin_to_port["distance"])
                        else round(origin_to_port["distance"], 2)
                    ),
                    "total_distance_km": origin_distance_km,
                    "total_co2_emissions_grams": (
                        0 if np.isnan(origin_co2_emissions) else round(origin_co2_emissions, 2)
                    ),
                    "geometry": origin_to_port["geometry"],
                }
                truck_routes['origin_to_port_routes'] = origin_to_port_routes

                dest_distance_km = 0 if np.isnan(port_to_dest["distance"]) else round(port_to_dest["distance"]/1000, 2)
                dest_co2_emissions = self._calculate_co2_emissions('truck', weight_tons, dest_distance_km)
                dest_travel_time = self._calculate_travel_time(input_departure_hour, port_to_dest["time"]/60, False)
                dest_travel_time_wait = self._calculate_travel_time(input_departure_hour, round(port_to_dest["time"]/60, 2), True)
                
                port_to_dest_routes = {
                    "departure_time": dest_travel_time['departure_time'],
                    "arrival_time": dest_travel_time['arrival_time'],
                    "total_wait_time_before_departure": dest_travel_time_wait['arrival_time'],
                    "origin_name": dest_port['C02_005'],
                    "destination_name": destination_name,
                    "total_time_minutes": (
                        0 if np.isnan(port_to_dest["time"]) else round(port_to_dest["time"], 2)
                    ),
                    "total_distance_meters": (
                        0
                        if np.isnan(port_to_dest["distance"])
                        else round(port_to_dest["distance"], 2)
                    ),
                    "total_distance_km": origin_distance_km,
                    "total_co2_emissions_grams": (
                        0 if np.isnan(dest_co2_emissions) else round(dest_co2_emissions, 2)
                    ),
                    "geometry": port_to_dest["geometry"],
                }
                truck_routes['port_to_dest_routes'] = port_to_dest_routes

                return truck_routes
            else:
                return None

        except Exception as e:
            print(f"Error getting truck routes to ports: {e}")
            return None

    def _find_ship_routes_between_ports(
        self,
        nearest_ports: Dict,
        weight_tons: float,
        max_transfers: int,
        show_all: bool,
    ) -> List[Dict]:
        """Step 2: Find ship routes between ports (direct or via transfer)"""
        try:
            origin_port = nearest_ports["origin_port"]
            dest_port = nearest_ports["dest_port"]

            # Try to find direct ship route first
            direct_ship = self._get_ship_route_info(
                origin_port["C02_005"], dest_port["C02_005"]
            )

            if direct_ship:
                co2_emissions = self._calculate_co2_emissions('ship', weight_tons, direct_ship["distance"] / 1000)
                total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], direct_ship['departure_time'])
                globals.GLOBAL_STATE["arrival_time_previous_route"] = direct_ship['arrival_time']
                return [
                    {
                        "type": "direct",
                        "route_info": direct_ship,
                        "transfer_ports": [],
                        "vehicle": VEHICLES['ship'],
                        "departure_time": direct_ship['departure_time'],
                        "arrival_time": direct_ship['arrival_time'],
                        "origin_name": origin_port["C02_005"],
                        "destination_name": dest_port["C02_005"],
                        "total_wait_time_before_departure_minutes": total_wait_time_before_departure_minutes,
                        "total_time_minutes": (
                            0 if np.isnan(direct_ship['time']) else round(direct_ship['time'] * 60, 2)
                        ),
                        "total_distance_meters": (
                            0
                            if np.isnan(direct_ship["distance"])
                            else round(direct_ship["distance"], 2)
                        ),
                        "total_distance_km": 0 if np.isnan(direct_ship["distance"]) else round(direct_ship["distance"] / 1000, 2),
                        "total_co2_emissions_grams": co2_emissions
                    }
                ]
            else:
                # Find ship routes via transfer
                transfer_routes = self._find_ship_transfer_routes(
                    origin_port, dest_port, max_transfers, weight_tons
                )
                return transfer_routes

        except Exception as e:
            print(f"Error finding ship routes between ports: {e}")
            return []

    def _combine_truck_ship_routes(
        self,
        truck_routes: Dict,
        ship_routes: List[Dict],
        nearest_ports: Dict,
        weight_tons: float,
    ) -> List[Dict]:
        """Step 3: Combine truck routes + ship routes"""
        try:
            combined_routes = []
            origin_port = nearest_ports["origin_port"]
            dest_port = nearest_ports["dest_port"]

            for ship_route in ship_routes:
                if ship_route["type"] == "direct":
                    # Direct ship route
                    route = self._create_combined_ship_route(
                        truck_routes,
                        ship_route["route_info"],
                        ship_route,
                        origin_port,
                        dest_port,
                        weight_tons,
                        [],
                    )
                    if route:
                        combined_routes.append(route)
                else:
                    # Transfer ship route
                    route = self._create_combined_ship_route(
                        truck_routes,
                        ship_route["route_info"],
                        ship_route,
                        origin_port,
                        dest_port,
                        weight_tons,
                        ship_route["transfer_ports"],
                    )
                    if route:
                        combined_routes.append(route)
            return combined_routes

        except Exception as e:
            print(f"Error combining truck ship routes: {e}")
            return []

    def _find_ship_transfer_routes(
        self, origin_port: Dict, dest_port: Dict, max_transfers: int, weight_tons: float
    ) -> List[Dict]:
        """Find ship routes via transfer"""
        try:
            # Logic to find transfer routes (using old logic but only for ship routes)
            # Find all ports reachable from origin port
            from_origin = self.ferry_time[
                self.ferry_time[
                    "Departure_Location_(National_Land_Numerical_Information_Format)"
                ]
                == origin_port["C02_005"]
            ]["Arrival_Location_(National_Land_Numerical_Information_Format)"].unique()

            # Find all ports that can reach destination port
            to_dest = self.ferry_time[
                self.ferry_time[
                    "Arrival_Location_(National_Land_Numerical_Information_Format)"
                ]
                == dest_port["C02_005"]
            ][
                "Departure_Location_(National_Land_Numerical_Information_Format)"
            ].unique()

            # Find transfer ports
            transfer_ports = set(from_origin) & set(to_dest)
            routes = []
            for transfer_port in list(transfer_ports)[:max_transfers]:
                # Find ship routes via transfer
                leg1 = self._get_ship_route_info(origin_port["C02_005"], transfer_port)
                if leg1:
                    leg1_total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], leg1['departure_time'])
                    globals.GLOBAL_STATE["arrival_time_previous_route"] = leg1['arrival_time']

                leg2 = self._get_ship_route_info(transfer_port, dest_port["C02_005"])
                if leg2:
                    leg2_total_wait_time_before_departure_minutes = calc_wait_minutes(globals.GLOBAL_STATE["arrival_time_previous_route"], leg2['departure_time'])
                    globals.GLOBAL_STATE["arrival_time_previous_route"] = leg2['arrival_time']

                if leg1 and leg2:
                    # Calculate total information
                    total_time = leg1["time"] + leg2["time"]
                    total_distance = leg1["distance"] + leg2["distance"]

                    leg1_co2_emissions = self._calculate_co2_emissions('ship', weight_tons, (leg1["distance"] / 1000) * 1.5)
                    leg2_co2_emissions = self._calculate_co2_emissions('ship', weight_tons, (leg2["distance"] / 1000) * 1.5)

                    routes.append(
                        {
                            "type": "transfer",
                            "route_info": {
                                "time": total_time,
                                "distance": total_distance,
                            },
                            "transfer_ports": [str(transfer_port)],
                            "leg1": {
                                "vehicle": VEHICLES['ship'],
                                "origin_name": origin_port["C02_005"],
                                "destination_name": transfer_port,
                                "departure_time": leg1['departure_time'],
                                "arrival_time": leg1['arrival_time'],
                                "total_wait_time_before_departure_minutes": leg1_total_wait_time_before_departure_minutes,
                                "total_time_minutes": (
                                    0 if np.isnan(leg1['time']) else round(leg1['time'] * 60, 2)
                                ),
                                "total_distance_km": 0 if np.isnan(leg1["distance"]) else round((leg1["distance"] / 1000)*1.5, 2),
                                "total_co2_emissions_grams": leg1_co2_emissions
                            },
                            "leg2": {
                                "vehicle": VEHICLES['ship'],
                                "origin_name": transfer_port,
                                "destination_name": dest_port["C02_005"],
                                "departure_time": leg2['departure_time'],
                                "arrival_time": leg2['arrival_time'],
                                "total_wait_time_before_departure_minutes": leg2_total_wait_time_before_departure_minutes,
                                "total_time_minutes": (
                                    0 if np.isnan(leg2['time']) else round(leg2['time'] * 60, 2)
                                ),
                                "total_distance_km": 0 if np.isnan(leg2["distance"]) else round((leg2["distance"] / 1000)*1.5, 2),
                                "total_co2_emissions_grams": leg2_co2_emissions
                            }
                        }
                    )

            return routes

        except Exception as e:
            print(f"Error finding ship transfer routes: {e}")
            return []

    def _create_combined_ship_route(
        self,
        truck_routes: Dict,
        ship_info: Dict,
        ship_route: Dict,
        origin_port: Dict,
        dest_port: Dict,
        weight_tons: float,
        transfer_ports: List,
    ) -> Optional[Dict]:
        """Create combined truck + ship route"""
        try:
            origin_to_port = truck_routes["origin_to_port"]
            port_to_dest = truck_routes["port_to_dest"]

            # Calculate total time and distance
            total_time = (
                origin_to_port["time"]
                + port_to_dest["time"]
                + ship_info["time"] * 60  # Convert hours to minutes
            )

            total_distance = (
                origin_to_port["distance"]
                + port_to_dest["distance"]
                + ship_info["distance"]
            )

            # Calculate CO2 emissions
            co2_emissions = (
                self._calculate_co2_emissions(
                    "truck", weight_tons, origin_to_port["distance"] / 1000
                )
                + self._calculate_co2_emissions(
                    "truck", weight_tons, port_to_dest["distance"] / 1000
                )
                + self._calculate_co2_emissions(
                    "ship", weight_tons, ship_info["distance"] / 1000
                )
            )

            # Create combined geometry
            geometry = self._create_combined_ship_geometry(
                origin_to_port, port_to_dest, origin_port, dest_port, transfer_ports
            )

            # Create ship geometry
            ship_geometry = self._create_ship_coords(origin_port, dest_port, transfer_ports)

            if transfer_ports:
                ship_segments = self.split_coords_into_segments(ship_geometry)
            
            # Create route name
            if transfer_ports:
                route_name = f"Truck + Ship (transfer via {', '.join(transfer_ports)})"
                mode = "truck_ship_transfer"
            else:
                route_name = "Truck + Ship"
                mode = "truck_ship"

            features = []
            # ---- Feature 1: Truck (Origin → Origin Port)\
            f1 = {
                "vehicle": VEHICLES['truck'],
                "departure_time": truck_routes['origin_to_port_routes']['departure_time'],
                "arrival_time": truck_routes['origin_to_port_routes']['arrival_time'],
                "total_wait_time_before_departure_minutes": truck_routes['origin_to_port_routes']['total_wait_time_before_departure_minutes'],
                "origin_name": truck_routes['origin_to_port_routes']['origin_name'],
                "destination_name": truck_routes['origin_to_port_routes']['destination_name'],
                "total_time_minutes": (
                    0 if np.isnan(truck_routes['origin_to_port_routes']['total_time_minutes']) else truck_routes['origin_to_port_routes']['total_time_minutes']
                ),
                "total_distance_meters": (
                    0
                    if np.isnan(truck_routes['origin_to_port_routes']['total_distance_meters'])
                    else truck_routes['origin_to_port_routes']['total_distance_meters']
                ),
                "total_distance_km": 0 if np.isnan(truck_routes['origin_to_port_routes']['total_distance_km']) else truck_routes['origin_to_port_routes']['total_distance_km'],
                "total_co2_emissions_grams": (
                    0 if np.isnan(truck_routes['origin_to_port_routes']['total_co2_emissions_grams']) else truck_routes['origin_to_port_routes']['total_co2_emissions_grams']
                ),
                "geometry": origin_to_port["geometry"],
            }
            features.append(f1)
            globals.GLOBAL_STATE['total_time_minutes'] += f1['total_time_minutes'] + f1['total_wait_time_before_departure_minutes']
            globals.GLOBAL_STATE['total_move_time_minutes'] += f1['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += f1['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += f1['total_co2_emissions_grams']
            

            if transfer_ports:
                f2 = {
                    "vehicle": VEHICLES['ship'],
                    "departure_time": ship_route['leg1']['departure_time'],
                    "arrival_time": ship_route['leg1']['arrival_time'],
                    "total_wait_time_before_departure_minutes": ship_route['leg1']['total_wait_time_before_departure_minutes'],
                    "origin_name": ship_route['leg1']['origin_name'],
                    "destination_name": ship_route['leg1']['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(ship_route['leg1']['total_time_minutes']) else ship_route['leg1']['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(ship_route['leg1']['total_distance_km']) else ship_route['leg1']['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(ship_route['leg1']['total_co2_emissions_grams']) else ship_route['leg1']['total_co2_emissions_grams']
                    ),
                    "geometry": ship_segments[0] if transfer_ports else ship_geometry,
                }
                features.append(f2)
                globals.GLOBAL_STATE['total_time_minutes'] += (f2['total_time_minutes'] + f2['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f2['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f2['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f2['total_co2_emissions_grams']

                f3 = {
                    "vehicle": VEHICLES['ship'],
                    "departure_time": ship_route['leg2']['departure_time'],
                    "arrival_time": ship_route['leg2']['arrival_time'],
                    "total_wait_time_before_departure_minutes": ship_route['leg2']['total_wait_time_before_departure_minutes'],
                    "origin_name": ship_route['leg2']['origin_name'],
                    "destination_name": ship_route['leg2']['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(ship_route['leg2']['total_time_minutes']) else ship_route['leg2']['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(ship_route['leg2']['total_distance_km']) else ship_route['leg2']['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(ship_route['leg2']['total_co2_emissions_grams']) else ship_route['leg2']['total_co2_emissions_grams']
                    ),
                    "geometry": ship_segments[1],
                }
                features.append(f3)
                globals.GLOBAL_STATE['total_time_minutes'] += (f3['total_time_minutes'] + f3['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f3['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f3['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f3['total_co2_emissions_grams']
            else:
                f2 = {
                    "vehicle": VEHICLES['ship'],
                    "departure_time": ship_route['departure_time'],
                    "arrival_time": ship_route['arrival_time'],
                    "total_wait_time_before_departure_minutes": ship_route['total_wait_time_before_departure_minutes'],
                    "origin_name": ship_route['origin_name'],
                    "destination_name": ship_route['destination_name'],
                    "total_time_minutes": (
                        0 if np.isnan(ship_route['total_time_minutes']) else ship_route['total_time_minutes']
                    ),
                    "total_distance_km": 0 if np.isnan(ship_route['total_distance_km']) else ship_route['total_distance_km'],
                    "total_co2_emissions_grams": (
                        0 if np.isnan(ship_route['total_co2_emissions_grams']) else ship_route['total_co2_emissions_grams']
                    ),
                    "geometry": ship_segments[0] if transfer_ports else ship_geometry,
                }
                features.append(f2)
                globals.GLOBAL_STATE['total_time_minutes'] += (f2['total_time_minutes'] + f2['total_wait_time_before_departure_minutes'])
                globals.GLOBAL_STATE['total_move_time_minutes'] += f2['total_time_minutes']
                globals.GLOBAL_STATE['total_distance_km'] += f2['total_distance_km']
                globals.GLOBAL_STATE['total_co2_emissions_grams'] += f2['total_co2_emissions_grams']

            ship_arrival_time = f3['arrival_time'] if transfer_ports else f2['arrival_time']
            truck_departure_time = add_hours(ship_arrival_time, 1.5)
            truck_travel_time = self._calculate_travel_time(truck_departure_time, truck_routes['port_to_dest_routes']['total_time_minutes'] / 60, False)
            globals.GLOBAL_STATE["arrival_time"] = truck_travel_time['arrival_time']
            f4 = {
                "vehicle": VEHICLES['truck'],
                "departure_time": truck_departure_time,
                "arrival_time": truck_travel_time['arrival_time'],
                "total_wait_time_before_departure_minutes": calc_wait_minutes(globals.GLOBAL_STATE['arrival_time_previous_route'], truck_departure_time),
                "origin_name": truck_routes['port_to_dest_routes']['origin_name'],
                "destination_name": truck_routes['port_to_dest_routes']['destination_name'],
                "total_time_minutes": (
                    0 if np.isnan(truck_routes['port_to_dest_routes']['total_time_minutes']) else truck_routes['port_to_dest_routes']['total_time_minutes']
                ),
                "total_distance_meters": (
                    0
                    if np.isnan(truck_routes['port_to_dest_routes']['total_distance_meters'])
                    else truck_routes['port_to_dest_routes']['total_distance_meters']
                ),
                "total_distance_km": 0 if np.isnan(truck_routes['port_to_dest_routes']['total_distance_km']) else truck_routes['port_to_dest_routes']['total_distance_km'],
                "total_co2_emissions_grams": (
                    0 if np.isnan(truck_routes['port_to_dest_routes']['total_co2_emissions_grams']) else truck_routes['port_to_dest_routes']['total_co2_emissions_grams']
                ),
                "geometry": port_to_dest.get("geometry"),
            }
            features.append(f4)
            globals.GLOBAL_STATE['total_time_minutes'] += (f4['total_time_minutes'] + f4['total_wait_time_before_departure_minutes'])
            globals.GLOBAL_STATE['total_move_time_minutes'] += f4['total_time_minutes']
            globals.GLOBAL_STATE['total_distance_km'] += f4['total_distance_km']
            globals.GLOBAL_STATE['total_co2_emissions_grams'] += f4['total_co2_emissions_grams']
            
            return {
                "mode": mode,
                "total_time_minutes": total_time,
                "total_distance_meters": total_distance,
                "total_distance_km": total_distance / 1000,
                "total_co2_emissions_grams": co2_emissions,
                "origin_port": origin_port["C02_005"],
                "dest_port": dest_port["C02_005"],
                "origin_station": "",
                "dest_station": "",
                "transfer_port": ", ".join(transfer_ports) if transfer_ports else "",
                "transfer_station": "",
                "ship_time_hours": ship_info["time"],
                "train_time_minutes": 0,
                "truck_time_minutes": origin_to_port["time"] + port_to_dest["time"],
                "truck_distance_km": (
                    origin_to_port["distance"] + port_to_dest["distance"]
                )
                / 1000,
                "geometry": geometry,
                "features": features
            }

        except Exception as e:
            print(f"Error creating combined ship route: {e}")
            return None

    def split_coords_into_segments(self, coords):
        segments = []
        for i in range(len(coords) - 1):
            segments.append([coords[i], coords[i + 1]])
        return segments
    
    def _create_combined_ship_geometry(
        self,
        origin_to_port: Dict,
        port_to_dest: Dict,
        origin_port: Dict,
        dest_port: Dict,
        transfer_ports: List,
    ) -> LineString:
        """Create combined truck + ship geometry"""
        try:
            # Get truck geometries
            truck_geom_1 = origin_to_port.get("geometry")
            truck_geom_2 = port_to_dest.get("geometry")

            if truck_geom_1 and truck_geom_2:
                # Use truck geometries from database
                truck1_coords = self._get_geometry_coords(truck_geom_1)
                truck2_coords = self._get_geometry_coords(truck_geom_2)

                # Create ship segment
                if transfer_ports:
                    # Ship route via transfer - need to find coordinates of transfer port
                    ship_coords = [(origin_port["X"], origin_port["Y"])]

                    # Add coordinates of transfer ports
                    for transfer_port in transfer_ports:
                        # Find coordinates of transfer port
                        transfer_port_info = self.minato_gdf[
                            self.minato_gdf["C02_005"] == transfer_port
                        ]
                        if not transfer_port_info.empty:
                            transfer_coords = (
                                transfer_port_info["X"].iloc[0],
                                transfer_port_info["Y"].iloc[0],
                            )
                            ship_coords.append(transfer_coords)

                    # Add final destination
                    ship_coords.append((dest_port["X"], dest_port["Y"]))
                else:
                    # Direct ship route
                    ship_coords = [
                        (origin_port["X"], origin_port["Y"]),
                        (dest_port["X"], dest_port["Y"]),
                    ]

                # Combine: truck1 + ship + truck2
                combined_coords = truck1_coords + ship_coords + truck2_coords[1:]

                # Ensure coordinates are tuples
                clean_coords = []
                for coord in combined_coords:
                    if hasattr(coord, "x") and hasattr(coord, "y"):
                        clean_coords.append((float(coord.x), float(coord.y)))
                    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                        clean_coords.append((float(coord[0]), float(coord[1])))

                return LineString(clean_coords)
            else:
                # Fallback: straight line
                return LineString(
                    [
                        (
                            origin_to_port.get("start_point", Point(0, 0)).x,
                            origin_to_port.get("start_point", Point(0, 0)).y,
                        ),
                        (origin_port["X"], origin_port["Y"]),
                        (dest_port["X"], dest_port["Y"]),
                        (
                            port_to_dest.get("end_point", Point(0, 0)).x,
                            port_to_dest.get("end_point", Point(0, 0)).y,
                        ),
                    ]
                )

        except Exception as e:
            print(f"Error creating combined ship geometry: {e}")
            # Fallback to simple straight line
            return LineString([(0, 0), (0, 0), (0, 0), (0, 0)])

    def _create_ship_coords(self, origin_port, dest_port, transfer_ports):
        # Create ship segment
        if transfer_ports:
            # Ship route via transfer - need to find coordinates of transfer port
            ship_coords = [(origin_port["X"], origin_port["Y"])]

            # Add coordinates of transfer ports
            for transfer_port in transfer_ports:
                # Find coordinates of transfer port
                transfer_port_info = self.minato_gdf[
                    self.minato_gdf["C02_005"] == transfer_port
                ]
                if not transfer_port_info.empty:
                    transfer_coords = (
                        transfer_port_info["X"].iloc[0],
                        transfer_port_info["Y"].iloc[0],
                    )
                    ship_coords.append(transfer_coords)

            # Add final destination
            ship_coords.append((dest_port["X"], dest_port["Y"]))
            return ship_coords
        else:
            # Direct ship route
            return [
                (origin_port["X"], origin_port["Y"]),
                (dest_port["X"], dest_port["Y"]),
            ]
        
    def _create_train_coords(self, origin_station, dest_station, transfer_stations):
        # Create train segment
        if transfer_stations:
            # Train route via transfer - need to find coordinates of transfer stations
            train_coords = [(origin_station["lon"], origin_station["lat"])]

            # Add coordinates of transfer stations
            for transfer_station in transfer_stations:
                # Find coordinates of transfer station
                transfer_station_info = self.station_gdf[
                    self.station_gdf["Station_Code"] == transfer_station
                ]
                if not transfer_station_info.empty:
                    transfer_coords = (
                        transfer_station_info["lon"].iloc[0],
                        transfer_station_info["lat"].iloc[0],
                    )
                    train_coords.append(transfer_coords)

            # Add final destination
            train_coords.append((dest_station["lon"], dest_station["lat"]))
        else:
            # Direct train route
            train_coords = [
                (origin_station["lon"], origin_station["lat"]),
                (dest_station["lon"], dest_station["lat"]),
            ]

        return train_coords

    def _get_geometry_coords(self, geometry) -> List[Tuple[float, float]]:
        """Get coordinates from geometry (handles both Shapely and GeoJSON)"""
        if isinstance(geometry, dict):
            # GeoJSON geometry
            if geometry.get("type") == "LineString":
                return geometry.get("coordinates", [])
            elif geometry.get("type") == "MultiLineString":
                # Flatten all LineString coordinates
                all_coords = []
                for line in geometry.get("coordinates", []):
                    all_coords.extend(line)
                return all_coords
        elif isinstance(geometry, str):
            # WKT string - try to parse
            try:
                from shapely import wkt

                geom = wkt.loads(geometry)
                if hasattr(geom, "coords"):
                    return list(geom.coords)
                elif hasattr(geom, "geoms"):
                    all_coords = []
                    for g in geom.geoms:
                        all_coords.extend(list(g.coords))
                    return all_coords
            except:
                pass
        else:
            # Shapely geometry
            if hasattr(geometry, "coords"):
                return list(geometry.coords)
            elif hasattr(geometry, "geoms"):
                # MultiLineString
                all_coords = []
                for geom in geometry.geoms:
                    all_coords.extend(list(geom.coords))
                return all_coords
        return []

    async def _get_truck_route_info(
        self, start_point: Point, end_point: Point
    ) -> Optional[Dict]:
        """Get truck route info from database or fallback"""
        # Try database first
        if self.db_pool:
            print(
                f"Debug: Trying database for truck route from ({start_point.x}, {start_point.y}) to ({end_point.x}, {end_point.y})"
            )
            db_result = await self._get_truck_route_info_db(start_point, end_point)
            if db_result:
                coords = self._get_geometry_coords(db_result.get("geometry", {}))
                print(f"Debug: Database returned route with {len(coords)} points")
                return db_result
            else:
                print("Debug: Database returned None, using fallback")
        else:
            print("Debug: No database pool, using fallback")

        # Fallback to straight-line calculation if database fails
        print("Debug: Using fallback straight line calculation")
        distance = self._calculate_distance(
            start_point.y, start_point.x, end_point.y, end_point.x
        )

        # Estimate truck speed as 60 km/h
        time_minutes = (distance / 1000) / 60 * 60

        return {
            "distance": distance,
            "time": time_minutes,
            "geometry": LineString([start_point, end_point]),
        }

    def _get_ship_route_info(self, origin_port: str, dest_port: str) -> Optional[Dict]:
        """Get ship route information"""
        route = self.ferry_time[
            (
                self.ferry_time[
                    "Departure_Location_(National_Land_Numerical_Information_Format)"
                ]
                == origin_port
            )
            & (
                self.ferry_time[
                    "Arrival_Location_(National_Land_Numerical_Information_Format)"
                ]
                == dest_port
            )
        ]

        if not route.empty:
            ship_data = find_ship_by_departure_time(route, globals.GLOBAL_STATE['arrival_time_buffer_wait_time'])

            route_time = ship_data["Route_Time"].iloc[0] if ship_data["Route_Time"].iloc[0] else None
            departure_time = ship_data["Departure_Time"].iloc[0] if ship_data["Departure_Time"].iloc[0] else add_hours(globals.GLOBAL_STATE["arrival_time_previous_route"])
            arrival_time = ship_data["Arrival_Time"].iloc[0] if ship_data["Arrival_Time"].iloc[0] else None
            speed_upper = ship_data["Speed_Upper_(km/h)"].iloc[0] if ship_data["Speed_Upper_(km/h)"].iloc[0] else SHIP_SPEED_DEFAULT

            # Calculate distance between ports
            origin_port_data = self.minato_gdf[
                self.minato_gdf["C02_005"] == origin_port
            ]
            dest_port_data = self.minato_gdf[self.minato_gdf["C02_005"] == dest_port]

            if not origin_port_data.empty and not dest_port_data.empty:
                distance = self._calculate_distance(
                    origin_port_data["Y"].iloc[0],
                    origin_port_data["X"].iloc[0],
                    dest_port_data["Y"].iloc[0],
                    dest_port_data["X"].iloc[0],
                )

                if not route_time or not arrival_time or not ship_data["Departure_Time"].iloc[0] or not ship_data["Speed_Upper_(km/h)"].iloc[0]:
                    route_time = (distance / 1000) * 1.5 / speed_upper
                    arrival_time = add_hours(departure_time, route_time)
                    globals.GLOBAL_STATE["warning_message"] = MESSAGES['no_time_data']

                return {"time": route_time, "distance": distance * 1.5, 'departure_time': departure_time, 'arrival_time': arrival_time}

        return None

    def _get_train_route_info_test(
        self, origin_station_name: str, dest_station_name: str
    ) -> Optional[Dict]:
        """Get train route information"""
        route = self.train_time[
            (self.train_time["Departure_Station_Name"] == origin_station_name)
            & (self.train_time["Arrival_Station_Name"] == dest_station_name)
        ]

        if not route.empty:
            duration = route["train_Duration2"].iloc[0]
            distance = (
                route["Distance_(km)"].iloc[0]
                if "Distance_(km)" in route.columns
                else 0
            )
            departure_time = route["Departure_Time"].iloc[0]
            arrival_time = route["Arrival_Time"].iloc[0]

            # Convert duration to minutes
            if hasattr(duration, "total_seconds"):
                time_minutes = duration.total_seconds() / 60
            else:
                time_minutes = 0

            return {
                "time": time_minutes,
                "distance": distance,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
            }

        return None

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate geodesic distance between two points"""
        geod = pyproj.Geod(ellps="WGS84")
        _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
        return distance

    def _calculate_co2_emissions(
        self, mode: str, weight_tons: float, distance_km: float
    ) -> float:
        """Calculate CO2 emissions"""
        if mode in self.co2_factors:
            return weight_tons * distance_km * self.co2_factors[mode]
        return 0

    def _find_optimal_routes(self, routes: List[Dict]) -> Dict:
        """Find optimal routes by different criteria"""
        if not routes:
            return {}

        # Find minimum time route
        min_time_route = min(routes, key=lambda x: x["total_time_minutes"])

        # Find minimum distance route
        min_distance_route = min(routes, key=lambda x: x["total_distance_meters"])

        # Find minimum CO2 route
        min_co2_route = min(routes, key=lambda x: x["total_co2_emissions_grams"])

        # Create copies without geometry for optimal routes (to avoid JSON serialization issues)
        def create_route_summary(route):
            summary = {k: v for k, v in route.items() if k != "geometry"}
            return summary

        return {
            "fastest": create_route_summary(min_time_route),
            "shortest": create_route_summary(min_distance_route),
            "greenest": create_route_summary(min_co2_route),
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to GeoJSON file"""
        # Convert results to GeoJSON format
        geojson_data = self._convert_to_geojson(results)

        with open(OUTPUT_FOLDER + output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    def _convert_to_geojson(self, results: Dict) -> Dict:
        """Convert results to GeoJSON format"""
        from shapely import wkt
        from shapely.geometry import LineString
        import numpy as np

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

        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": [],
            # "properties": convert_numpy_types(
            #     {
            #         "origin": results.get("origin", {}),
            #         "destination": results.get("destination", {}),
            #         "weight_tons": results.get("weight_tons", 10.0),
            #         "optimal_routes": results.get("optimal_routes", {}),
            #         "search_options": {
            #             "criteria_used": results.get("criteria_used", "fastest"),
            #             "show_all": results.get("show_all", False),
            #             "mode": results.get("mode", "all"),
            #             "enable_transfer": results.get("enable_transfer", False),
            #             "max_transfers": results.get("max_transfers", 2),
            #         },
            #     }
            # ),
        }

        # Convert each route to a GeoJSON feature
        for route in results.get("routes", []):
            if "features" in route and route["features"]:
                for feature in route["features"]:
                    new_feature = create_features(feature)
                    if not new_feature:
                        continue
                    
                    geojson["features"].append(new_feature)
            else:
                if "geometry" in route and route["geometry"]:
                    new_feature = create_features(route)
                    if not new_feature:
                        continue
                    
                    geojson["features"].append(new_feature)

        return geojson

    async def _find_ship_routes_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_ports: Dict,
        weight_tons: float,
        max_transfers: int,
    ) -> List[Dict]:
        """
        Find ship routes with transfer

        Args:
            origin_point: Origin point
            dest_point: Destination point
            nearest_ports: Nearest ports information
            weight_tons: Cargo weight
            max_transfers: Maximum number of transfers

        Returns:
            List of routes with transfer
        """
        routes = []

        if self.ferry_time is None:
            return routes

        origin_port = nearest_ports["origin_port"]
        dest_port = nearest_ports["dest_port"]

        if origin_port is None or dest_port is None:
            return routes

        # Find all ports reachable from origin port
        from_origin = self.ferry_time[
            self.ferry_time[
                "Departure_Location_(National_Land_Numerical_Information_Format)"
            ]
            == origin_port["C02_005"]
        ]["Arrival_Location_(National_Land_Numerical_Information_Format)"].unique()

        # Find all ports that can reach destination port
        to_dest = self.ferry_time[
            self.ferry_time[
                "Arrival_Location_(National_Land_Numerical_Information_Format)"
            ]
            == dest_port["C02_005"]
        ]["Departure_Location_(National_Land_Numerical_Information_Format)"].unique()

        # Find transfer ports
        transfer_ports = set(from_origin) & set(to_dest)

        for transfer_port in list(transfer_ports)[:max_transfers]:
            try:
                # Try to get truck routes and ports from main calculation first
                truck_routes = None
                ports = None
                if hasattr(self, "_last_truck_routes_ship") and hasattr(
                    self, "_last_ports"
                ):
                    truck_routes = self._last_truck_routes_ship
                    ports = self._last_ports

                if truck_routes and ports:
                    # Use existing truck routes and ports
                    origin_to_port = truck_routes["origin_to_port"]
                    port_to_dest = truck_routes["port_to_dest"]
                    # Use saved ports instead of parameters
                    origin_port = ports["origin_port"]
                    dest_port = ports["dest_port"]
                else:
                    # Calculate truck routes if not available
                    origin_to_port = await self._get_truck_route_info(
                        origin_point, Point(origin_port["X"], origin_port["Y"])
                    )
                    port_to_dest = await self._get_truck_route_info(
                        Point(dest_port["X"], dest_port["Y"]), dest_point
                    )

                if origin_to_port and port_to_dest:
                    route = self._calculate_ship_route_with_transfer(
                        origin_point,
                        dest_point,
                        origin_port,
                        dest_port,
                        transfer_port,
                        weight_tons,
                        origin_to_port,
                        port_to_dest,
                    )
                else:
                    route = None
                if route:
                    routes.append(route)
            except Exception as e:
                print(f"Error calculating transfer route via {transfer_port}: {e}")
                continue

        return routes

    def _calculate_ship_route_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        origin_port: Dict,
        dest_port: Dict,
        transfer_port: str,
        weight_tons: float,
        origin_to_port: Dict,
        port_to_dest: Dict,
    ) -> Optional[Dict]:
        """
        Calculate ship route with transfer
        """
        try:
            # Find transfer port information
            transfer_port_info = self.minato_gdf[
                self.minato_gdf["C02_005"] == transfer_port
            ]

            if transfer_port_info.empty:
                return None

            transfer_port_data = transfer_port_info.iloc[0]

            # Calculate ship route using OLD LOGIC (CSV data) - truck routes already passed
            leg1 = self._get_ship_route_info(origin_port["C02_005"], transfer_port)
            leg2 = self._get_ship_route_info(transfer_port, dest_port["C02_005"])

            if origin_to_port and port_to_dest and leg1 and leg2:
                total_time = (
                    origin_to_port["time"]
                    + port_to_dest["time"]
                    + (leg1["time"] + leg2["time"]) * 60
                )

                total_distance = (
                    origin_to_port["distance"]
                    + port_to_dest["distance"]
                    + leg1["distance"]
                    + leg2["distance"]
                )

                co2_emissions = (
                    self._calculate_co2_emissions(
                        "truck", weight_tons, origin_to_port["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "truck", weight_tons, port_to_dest["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "ship", weight_tons, leg1["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "ship", weight_tons, leg2["distance"] / 1000
                    )
                )

                geometry = LineString(
                    [
                        (origin_point.x, origin_point.y),
                        (origin_port["X"], origin_port["Y"]),
                        (transfer_port_data["X"], transfer_port_data["Y"]),
                        (dest_port["X"], dest_port["Y"]),
                        (dest_point.x, dest_point.y),
                    ]
                )

                return {
                    "mode": "truck_ship_transfer",
                    "total_time_minutes": total_time,
                    "total_distance_meters": total_distance,
                    "total_distance_km": total_distance / 1000,
                    "total_co2_emissions_grams": co2_emissions,
                    "origin_port": origin_port["C02_005"],
                    "dest_port": dest_port["C02_005"],
                    "transfer_port": transfer_port,
                    "ship_time_hours": leg1["time"] + leg2["time"],
                    "geometry": geometry,
                }
        except Exception as e:
            print(f"Error calculating ship route with transfer: {e}")

        return None

    async def _find_train_routes_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_stations: Dict,
        weight_tons: float,
        max_transfers: int,
    ) -> List[Dict]:
        """
        Find train routes with transfer
        """
        routes = []

        if self.train_time is None:
            return routes

        origin_station = nearest_stations["origin_station"]
        dest_station = nearest_stations["dest_station"]

        if origin_station is None or dest_station is None:
            return routes

        # Find all stations reachable from origin station
        from_origin = self.train_time[
            self.train_time["Departure_Station_Code"] == origin_station["Station_Code"]
        ]["Arrival_Station_Code"].unique()

        # Find all stations that can reach destination station
        to_dest = self.train_time[
            self.train_time["Arrival_Station_Code"] == dest_station["Station_Code"]
        ]["Departure_Station_Code"].unique()

        # Find transfer stations
        transfer_stations = set(from_origin) & set(to_dest)

        for transfer_station in list(transfer_stations)[:max_transfers]:
            try:
                # Try to get truck routes and stations from main calculation first
                truck_routes = None
                stations = None
                if hasattr(self, "_last_truck_routes") and hasattr(
                    self, "_last_stations"
                ):
                    truck_routes = self._last_truck_routes
                    stations = self._last_stations

                if truck_routes and stations:
                    # Use existing truck routes and stations
                    origin_to_station = truck_routes["origin_to_station"]
                    station_to_dest = truck_routes["station_to_dest"]
                    # Use saved stations instead of parameters
                    origin_station = stations["origin_station"]
                    dest_station = stations["dest_station"]
                else:
                    # Calculate truck routes if not available
                    origin_to_station = await self._get_truck_route_info(
                        origin_point,
                        Point(origin_station["lon"], origin_station["lat"]),
                    )
                    station_to_dest = await self._get_truck_route_info(
                        Point(dest_station["lon"], dest_station["lat"]), dest_point
                    )

                if origin_to_station and station_to_dest:
                    route = self._calculate_train_route_with_transfer(
                        origin_point,
                        dest_point,
                        origin_station,
                        dest_station,
                        transfer_station,
                        weight_tons,
                        origin_to_station,
                        station_to_dest,
                    )
                else:
                    route = None
                if route:
                    routes.append(route)
            except Exception as e:
                print(
                    f"Error calculating train transfer route via {transfer_station}: {e}"
                )
                continue

        return routes

    def _calculate_train_route_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        origin_station: Dict,
        dest_station: Dict,
        transfer_station: int,
        weight_tons: float,
        origin_to_station: Dict,
        station_to_dest: Dict,
    ) -> Optional[Dict]:
        """
        Calculate train route with transfer
        """
        try:
            # Find transfer station information
            transfer_station_info = self.station_gdf[
                self.station_gdf["Station_Code"] == transfer_station
            ]

            if transfer_station_info.empty:
                return None

            transfer_station_data = transfer_station_info.iloc[0]

            # Calculate train route (truck routes already passed)
            leg1 = self._get_train_route_info(
                origin_station["Station_Code"], transfer_station
            )
            leg2 = self._get_train_route_info(
                transfer_station, dest_station["Station_Code"]
            )
            if origin_to_station and station_to_dest and leg1 and leg2:
                total_time = (
                    origin_to_station["time"]
                    + station_to_dest["time"]
                    + leg1["time"]
                    + leg2["time"]
                )

                total_distance = (
                    origin_to_station["distance"]
                    + station_to_dest["distance"]
                    + leg1["distance"]
                    + leg2["distance"]
                )

                co2_emissions = (
                    self._calculate_co2_emissions(
                        "truck", weight_tons, origin_to_station["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "truck", weight_tons, station_to_dest["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "train", weight_tons, leg1["distance"] / 1000
                    )
                    + self._calculate_co2_emissions(
                        "train", weight_tons, leg2["distance"] / 1000
                    )
                )

                geometry = LineString(
                    [
                        (origin_point.x, origin_point.y),
                        (origin_station["lon"], origin_station["lat"]),
                        (transfer_station_data["lon"], transfer_station_data["lat"]),
                        (dest_station["lon"], dest_station["lat"]),
                        (dest_point.x, dest_point.y),
                    ]
                )

                return {
                    "mode": "truck_train_transfer",
                    "total_time_minutes": total_time,
                    "total_distance_meters": total_distance,
                    "total_distance_km": total_distance / 1000,
                    "total_co2_emissions_grams": co2_emissions,
                    "origin_station": origin_station["Station_Name"],
                    "dest_station": dest_station["Station_Name"],
                    "transfer_station": transfer_station,
                    "train_time_minutes": leg1["time"] + leg2["time"],
                    "geometry": geometry,
                }
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error calculating train route with transfer: {e}")

        return None

    def _get_train_route_info(
        self, origin_station_code: int, dest_station_code: int
    ) -> Optional[Dict]:
        """
        Get train route information between 2 stations
        """
        if self.train_time is None:
            return None

        route = self.train_time[
            (self.train_time["Departure_Station_Code"] == origin_station_code)
            & (self.train_time["Arrival_Station_Code"] == dest_station_code)
        ]

        print(route, '///////')

        if not route.empty:
            train_data = find_train_by_departure_time(route, globals.GLOBAL_STATE['arrival_time_buffer_wait_time'])

            departure_time = train_data["Departure_Time"].iloc[0] if train_data["Departure_Time"].iloc[0] else globals.GLOBAL_STATE["arrival_time_previous_route"]
            arrival_time = train_data["Arrival_Time"].iloc[0] if train_data["Arrival_Time"].iloc[0] else None

            # Calculate travel time from train_Duration
            duration = route["train_Duration2"].iloc[0]
            if hasattr(duration, "total_seconds"):
                time_minutes = duration.total_seconds() / 60
            else:
                time_minutes = 0

            # Estimate distance (can be improved by using real data)
            distance_km = time_minutes * 50  # Assume average speed 50 km/h

            departure_time_dt = datetime.strptime(departure_time.strip(), "%H:%M:%S")
            departure_time = departure_time_dt.strftime("%H:%M")

            arrival_time_dt = datetime.strptime(arrival_time.strip(), "%H:%M:%S")
            arrival_time = arrival_time_dt.strftime("%H:%M")


            return {"time": time_minutes, "distance": distance_km * 1000, 'departure_time': departure_time, 'arrival_time': arrival_time}

        return None

    async def _find_single_ship_route_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_ports: Dict,
        weight_tons: float,
        max_transfers: int,
    ) -> Optional[Dict]:
        """
        Find only 1 optimal ship route with transfer
        Uses similar logic as _find_ship_routes_with_transfer but returns only the best route
        """
        try:
            # Use old function to find all routes with transfer
            all_transfer_routes = await self._find_ship_routes_with_transfer(
                origin_point, dest_point, nearest_ports, weight_tons, max_transfers
            )

            if not all_transfer_routes:
                return None

            # Find best route (shortest time)
            best_route = None
            best_time = float("inf")

            for route in all_transfer_routes:
                if route["total_time_minutes"] < best_time:
                    best_time = route["total_time_minutes"]
                    best_route = route

            return best_route

        except Exception as e:
            print(f"Error finding single ship route with transfer: {e}")
            return None

    async def _find_single_train_route_with_transfer(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_stations: Dict,
        weight_tons: float,
        max_transfers: int,
    ) -> Optional[Dict]:
        """
        Find only 1 optimal train route with transfer
        Uses similar logic as _find_train_routes_with_transfer but returns only the best route
        """
        try:
            # Use old function to find all routes with transfer
            all_transfer_routes = await self._find_train_routes_with_transfer(
                origin_point, dest_point, nearest_stations, weight_tons, max_transfers
            )

            if not all_transfer_routes:
                return None

            # Find best route (shortest time)
            best_route = None
            best_time = float("inf")

            for route in all_transfer_routes:
                if route["total_time_minutes"] < best_time:
                    best_time = route["total_time_minutes"]
                    best_route = route

            return best_route

        except Exception as e:
            print(f"Error finding single train route with transfer: {e}")
            return None

    @timeit("_init_database")
    async def _init_database(self):
        """Initialize async database connection pool"""
        if not ASYNCPG_AVAILABLE:
            print(
                "Warning: asyncpg not available, truck routes will use fallback method"
            )
            self.db_pool = None
            return

        if not self.db_config:
            print(
                "Warning: No database config provided, truck routes will use fallback method"
            )
            self.db_pool = None
            return

        try:
            self.db_pool = await asyncpg.create_pool(
                min_size=1,
                max_size=10,
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                command_timeout=60,
            )
            print("Async database connection pool initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize async database pool: {e}")
            print("Truck routes will use fallback method")
            self.db_pool = None

    @timeit("_get_node_component")    
    async def _get_node_component(
        self, start_node: int, end_node: int
    ) -> Optional[List[Dict]]:
        """
        Get node components to check connectivity using precomputed table.
        """
        sql = "SELECT component FROM jpn_components WHERE node = ANY($1::bigint[])"
        params = [start_node, end_node]
        rows = await self._db_query_all(sql, [params])
        
        if not rows:
            return None
        return rows

    @timeit("_db_query_one")
    async def _db_query_one(self, query: str, params: tuple) -> Optional[Dict]:
        """Execute a single query and return one result asynchronously."""
        if not self.db_pool:
            await self._init_database()
            if not self.db_pool:
                return None

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, *(params or ()))
                return dict(row) if row else None

        except Exception as e:
            print(f"[ERROR] _db_query_one: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            return None

    @timeit("_db_query_all")
    async def _db_query_all(self, query: str, params: tuple) -> Optional[List[Dict]]:
        """Execute an async query and return all results"""
        if not self.db_pool:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                if not rows:
                    return None

                result = [dict(r) for r in rows]
                return result

        except Exception as e:
            print(f"[ERROR] _db_query_all: {e}")
            return None
   
    @timeit("_nearest_node_id")
    async def _nearest_node_id(self, lon: float, lat: float) -> Optional[Dict]:
        cache_key = (round(lon, 6), round(lat, 6))
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]

        result = await self._db_query_one("SELECT nearest_node_id($1, $2) AS nid", (lon, lat))

        if result is None:
            return None

        if isinstance(result, int):
            result = {"nid": result}

        self._node_cache[cache_key] = result
        return result

    @timeit("_route_truck_mm")    
    async def _route_truck_mm(
        self,
        o_lon: float,
        o_lat: float,
        d_lon: float,
        d_lat: float,
        toll_per_km: float = 30.0,
    ) -> Optional[Dict]:
        """Compute truck route asynchronously"""

        # 1 Cache key
        cache_key = (
            round(o_lon, 6),
            round(o_lat, 6),
            round(d_lon, 6),
            round(d_lat, 6),
            round(toll_per_km, 2),
        )
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        # 2 Find nearest nodes
        src_node = await self._db_query_one(
            "SELECT nearest_node_id($1::float, $2::float) as nid", 
            (o_lon, o_lat)
        )
        dst_node = await self._db_query_one(
            "SELECT nearest_node_id($1::float, $2::float) as nid", 
            (d_lon, d_lat)
        )

        if not src_node or not dst_node or src_node["nid"] is None or dst_node["nid"] is None:
            return None
        src_id = src_node["nid"]
        dst_id = dst_node["nid"]
        
        start = time.time()
        
        route_query = """
            WITH route AS (
                SELECT * FROM pgr_bdAstar(
                    $$
                    SELECT gid AS id, source, target, cost_s AS cost, reverse_cost,
                        ST_X(ST_StartPoint(geom)) AS x1, ST_Y(ST_StartPoint(geom)) AS y1,
                        ST_X(ST_EndPoint(geom)) AS x2, ST_Y(ST_EndPoint(geom)) AS y2
                    FROM jpn_ways
                    WHERE NOT blocked
                    $$,
                    $1::bigint, $2::bigint, directed := true
                )
            ),
            edge_details AS (
                SELECT 
                    w.gid AS id, 
                    w.source, 
                    w.target, 
                    w.cost_s,
                    CASE 
                        WHEN w.oneway = 'YES' THEN 1e15
                        ELSE w.cost_s
                    END AS reverse_cost,
                    w.length_m,
                    w.highway,
                    w.geom,
                    w.maxspeed_forward,
                    r.seq,
                    r.node,
                    r.edge,
                    r.cost AS route_cost
                FROM jpn_ways w
                JOIN route r ON w.gid = r.edge
                WHERE NOT w.blocked
                ORDER BY r.seq
            )
            SELECT * FROM edge_details;
        """

        edges_detail = await self._db_query_all(
            route_query,
            [src_id, dst_id]
        )

        end = time.time()
        print("tg3 = ", end-start)

        if not edges_detail:
            return None

        # 6 Merge geometry
        geoms = [e["geom"] for e in edges_detail]
        merged_geom = await self._db_query_one(
            "SELECT ST_AsGeoJSON(ST_LineMerge(ST_Union(geom))) AS gj FROM unnest($1::geometry[]) AS geom",
            (geoms,)
        )
        if not merged_geom or not merged_geom["gj"]:
            return None
        distance_km = sum(e["length_m"] for e in edges_detail) / 1000.0

        # Travel time and motorway km
        travel_time_h = 0.0
        motorway_km = 0.0
        for edge in edges_detail:
            length_m = edge["length_m"]
            highway = edge["highway"]
            maxspeed = edge["maxspeed_forward"]
            speed = float(maxspeed) if maxspeed else {
                'motorway': 120.0, 'motorway_link': 120.0,
                'trunk': 100.0, 'trunk_link': 100.0,
                'primary': 80.0, 'primary_link': 80.0,
                'secondary': 60.0, 'secondary_link': 60.0,
                'tertiary': 40.0, 'tertiary_link': 40.0
            }.get(highway, 30.0)
            travel_time_h += length_m / (speed * 1000)
            if highway in ("motorway", "motorway_link"):
                motorway_km += length_m / 1000.0

        toll_estimate_yen = round(motorway_km * toll_per_km)

        # 7 Entry/Exit IC
        motorway_edges_in_route = [e for e in edges_detail if e["highway"] in ("motorway", "motorway_link")]
        entry_ic = exit_ic = None
        if motorway_edges_in_route:
            first_motorway = motorway_edges_in_route[0]
            last_motorway = motorway_edges_in_route[-1]

            entry_point = await self._db_query_one("SELECT ST_StartPoint($1) as pt", (first_motorway["geom"],))
            exit_point = await self._db_query_one("SELECT ST_EndPoint($1) as pt", (last_motorway["geom"],))

            if entry_point:
                entry_ic = await self._db_query_one(
                    "SELECT name, lon, lat FROM motorway_ic ORDER BY ST_SetSRID(ST_MakePoint(lon,lat),4326) <-> $1 LIMIT 1",
                    (entry_point["pt"],)
                )
            if exit_point:
                exit_ic = await self._db_query_one(
                    "SELECT name, lon, lat FROM motorway_ic ORDER BY ST_SetSRID(ST_MakePoint(lon,lat),4326) <-> $1 LIMIT 1",
                    (exit_point["pt"],)
                )

        # 8 Build result
        route = {
            "geometry": json.loads(merged_geom["gj"]),
            "distance_km": distance_km,
            "travel_time_h": travel_time_h,
            "motorway_km": motorway_km,
            "toll_estimate_yen": toll_estimate_yen,
            "entry_ic": {"name": entry_ic["name"], "lon": entry_ic["lon"], "lat": entry_ic["lat"]} if entry_ic else None,
            "exit_ic": {"name": exit_ic["name"], "lon": exit_ic["lon"], "lat": exit_ic["lat"]} if exit_ic else None,
        }

        self._route_cache[cache_key] = route
        return route

    # @timeit("Route Truck MM (async)")
    # async def _route_truck_mm(
    #     self,
    #     o_lon: float,
    #     o_lat: float,
    #     d_lon: float,
    #     d_lat: float,
    #     toll_per_km: float = 30.0,
    # ) -> Optional[Dict]:
    #     """Get truck route using SQL function route_truck_mm() asynchronously"""

    #     result = await self._db_query_one(
    #         """
    #         SELECT geom_geojson, distance_km, travel_time_h, motorway_km, toll_estimate_yen,
    #             entry_ic_name, entry_ic_lon, entry_ic_lat,
    #             exit_ic_name,  exit_ic_lon,  exit_ic_lat
    #         FROM route_truck_mm($1, $2, $3, $4, $5)
    #         """,
    #         (o_lon, o_lat, d_lon, d_lat, toll_per_km),
    #     )

    #     if not result or not result.get("geom_geojson"):
    #         return None

    #     return {
    #         "geometry": json.loads(result["geom_geojson"]),
    #         "distance_km": float(result["distance_km"]),
    #         "travel_time_h": float(result["travel_time_h"]),
    #         "motorway_km": float(result["motorway_km"]),
    #         "toll_estimate_yen": (
    #             float(result["toll_estimate_yen"])
    #             if result["toll_estimate_yen"] is not None
    #             else None
    #         ),
    #         "entry_ic": (
    #             {
    #                 "name": result["entry_ic_name"],
    #                 "lon": result["entry_ic_lon"],
    #                 "lat": result["entry_ic_lat"],
    #             }
    #             if result.get("entry_ic_name") is not None
    #             else None
    #         ),
    #         "exit_ic": (
    #             {
    #                 "name": result["exit_ic_name"],
    #                 "lon": result["exit_ic_lon"],
    #                 "lat": result["exit_ic_lat"],
    #             }
    #             if result.get("exit_ic_name") is not None
    #             else None
    #         ),
    #     }

    @timeit("_get_truck_route_info_db")
    async def _get_truck_route_info_db(
        self, start_point: Point, end_point: Point, toll_per_km: float = 30.0
    ) -> Optional[Dict]:
        """Get truck route info from database - using logic from app.py"""
        try:
            print(
                f"Debug: Finding nearest nodes for ({start_point.x}, {start_point.y}) and ({end_point.x}, {end_point.y})"
            )
            start_node = await self._nearest_node_id(start_point.x, start_point.y)
            end_node = await self._nearest_node_id(end_point.x, end_point.y)

            if not start_node or not end_node:
                print("Debug: Could not find nearest nodes")
                return None

            start_nid = start_node["nid"]
            end_nid = end_node["nid"]

            print(f"Debug: Found nodes - start: {start_nid}, end: {end_nid}")

            components = await self._get_node_component(start_nid, end_nid)
            
            if not components or len(components) < 2:
                print("Debug: Nodes not connected or components not found")
                return None

            start_component = components[0]["component"]
            end_component = components[1]["component"]

            if start_component != end_component:
                print(
                    f"Debug: Nodes in different components - start: {start_component}, end: {end_component}"
                )
                return None

            print("Debug: Nodes are connected, getting route from database")

            # Query route, loi o day
            route_result = await self._route_truck_mm(
                start_point.x, start_point.y, end_point.x, end_point.y, toll_per_km
            )
            
            if not route_result:
                print("Debug: route_truck_mm returned None")
                return None

            print(
                f"Debug: Got route from database - distance: {route_result['distance_km']} km, time: {route_result['travel_time_h']} h"
            )

            geometry = route_result["geometry"]

            return {
                "time": route_result["travel_time_h"] * 60,  # minutes
                "distance": route_result["distance_km"] * 1000,  # meters
                "geometry": geometry,
            }

        except Exception as e:
            print(f"Error getting truck route from database: {e}")
            import traceback
            traceback.print_exc()
            return None
