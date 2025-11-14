"""
Route Optimizer for Multimodal Transportation
Optimizes multimodal transportation routes (truck, train, ship)
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from datetime import timedelta
import pyproj
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from shapely import wkt
import math

from helper import (
    build_data_infos,
    build_result_segment,
    extract_linestring,
)

try:
    import psycopg2
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from decoratorr import timeit


class RouteOptimizer:
    """
    Class for optimizing multimodal transportation routes
    """

    # Constants
    CO2_FACTORS = {"truck": 43, "train": 20, "ship": 10}
    VALID_MODES = [
        "all", "truck_only", "truck_ship", "truck_train", 
        "truck_ship_train", "truck_train_ship", "truck_train_ship_train"
    ]

    @timeit("Init RouteOptimizer")
    def __init__(self, data_folder_path: str, db_config: Optional[Dict] = None):
        """
        Initialize RouteOptimizer

        Args:
            data_folder_path: Path to data folder
            db_config: Database configuration (host, port, database, user, password)
        """
        self.data_folder_path = data_folder_path

        # Database configuration
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5432,
            "database": "pgrouting",
            "user": "postgres",
            "password": "pgrouting",
        }
        
        self.db_pool = None

        # Initialize data attributes
        self.odlist_gdf = None
        self.minato_gdf = None
        self.station_gdf = None
        self.ferry_time = None
        self.train_time = None
        self.track_route = None
        self.port_list = None

        # Caches
        self._node_cache = {}
        self._route_cache = {}
        self._node_component_cache = {}

        # Initialize database connection and load data
        self._init_database()
        self._load_all_data()

    def _load_all_data(self):
        """Load all required data"""
        print("Loading data...")
        
        loaders = [
            ("OD Data", self._load_od_data),
            ("Port Data", self._load_port_data),
            ("Station Data", self._load_station_data),
            ("Ferry Schedule", self._load_ferry_schedule),
            ("Train Schedule", self._load_train_schedule),
        ]
        
        for name, loader in loaders:
            try:
                loader()
                print(f"✓ {name} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        
        print("Data loading completed!")

    @timeit("Load OD Data")
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

    @timeit("Load Port Data")
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

    @timeit("Load Station Data")
    def _load_station_data(self):
        """Load freight station data"""
        station = pd.read_csv(f"{self.data_folder_path}/貨物駅_位置情報.csv")
        self.station_gdf = gpd.GeoDataFrame(
            station, geometry=gpd.points_from_xy(station["lon"], station["lat"])
        )
        self.station_gdf.set_crs(epsg=4326, inplace=True)

    @timeit("Load Ferry Schedule")
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

    @timeit("Load Train Schedule")
    def _load_train_schedule(self):
        """Load train schedule"""
        self.train_time = pd.read_csv(f"{self.data_folder_path}/貨物駅_時刻表.csv")

        # Replace arrival date
        date_mapping = {
            "当日": 0, "翌日": 1, "翌々日": 2, "７日目": 6,
            "４日目": 3, "５日目": 4, "６日目": 5,
        }
        self.train_time["Arrival_Date"] = self.train_time["Arrival_Date"].replace(date_mapping)

        # Convert time format
        time_cols = ["Track_Entry_Time", "Track_Exit_Time"]
        for col in time_cols:
            self.train_time[col] = pd.to_datetime(
                self.train_time[col], format="%H:%M:%S"
            ).dt.time
            self.train_time[col] = pd.to_datetime(
                self.train_time[col].astype(str), format="%H:%M:%S"
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

        # Remove duplicates
        self.train_time = self.train_time.drop_duplicates(subset=["train_od"])

    @timeit("Find Route")
    def find_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        weight_tons: float = 10.0,
        mode: str = "all",
        enable_transfer: bool = False,
        max_transfers: int = 10,
        show_all: bool = False,
    ) -> Dict:
        """
        Find optimal route between two points with transfer capability
        """
        self._validate_mode(mode)
        print(
            f"Finding route from ({origin_lat}, {origin_lon}) to ({dest_lat}, {dest_lon}) with mode: {mode}"
        )

        origin_point = Point(origin_lon, origin_lat)
        dest_point = Point(dest_lon, dest_lat)

        # Find nearest ports and stations
        nearest_ports = self._find_nearest_ports(origin_point, dest_point)
        nearest_stations = self._find_nearest_stations(origin_point, dest_point)

        # Calculate routes for different transportation modes
        routes = self._calculate_routes_by_mode(
            origin_point, dest_point, nearest_ports, nearest_stations,
            weight_tons, mode, enable_transfer, max_transfers, show_all
        )

        # Find optimal routes
        optimal_routes = self._find_optimal_routes(routes)

        return {
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "destination": {"lat": dest_lat, "lon": dest_lon},
            "weight_tons": weight_tons,
            "routes": routes,
            "optimal_routes": optimal_routes,
        }

    def _validate_mode(self, mode: str):
        """Validate mode parameter"""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

    @timeit("Find Nearest Ports")
    def _find_nearest_ports(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest ports to origin and destination points"""
        return self._find_nearest_features(
            origin_point, dest_point, self.minato_gdf, ["origin_port", "dest_port"]
        )

    @timeit("Find Nearest Stations")
    def _find_nearest_stations(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest stations to origin and destination points"""
        return self._find_nearest_features(
            origin_point, dest_point, self.station_gdf, ["origin_station", "dest_station"]
        )

    def _find_nearest_features(self, origin_point: Point, dest_point: Point, 
                             gdf: gpd.GeoDataFrame, keys: List[str]) -> Dict:
        """Generic method to find nearest features"""
        temp_gdf = gpd.GeoDataFrame(
            {"name": ["origin", "destination"]},
            geometry=[origin_point, dest_point],
            crs="EPSG:4326",
        )

        # Convert to projected CRS for accurate distance calculations
        temp_gdf = temp_gdf.to_crs("EPSG:3857")
        gdf_projected = gdf.to_crs("EPSG:3857")

        results = {}
        for i, key in enumerate(keys):
            nearest = gpd.sjoin_nearest(
                temp_gdf.iloc[[i]], gdf_projected, how="inner", distance_col="distance"
            )
            results[key] = nearest.iloc[0] if not nearest.empty else None

        return results

    @timeit("Calculate Routes by Mode")
    def _calculate_routes_by_mode(
        self,
        origin_point: Point,
        dest_point: Point,
        nearest_ports: Dict,
        nearest_stations: Dict,
        weight_tons: float,
        mode: str,
        enable_transfer: bool = False,
        max_transfers: int = 10,
        show_all: bool = False,
    ) -> List[Dict]:
        """Calculate routes by selected mode"""
        mode_handlers = {
            "truck_only": self._calculate_truck_only_route,
            "truck_ship": self._calculate_truck_ship_route,
            "truck_train": self._calculate_truck_train_route,
            "truck_ship_train": self._calculate_truck_ship_train_route,
            "truck_train_ship": self._calculate_truck_train_ship_route,
            "truck_train_ship_train": self._calculate_truck_train_ship_train_route,
        }

        routes = []
        
        if mode == "all":
            # Calculate all available modes
            for handler in mode_handlers.values():
                try:
                    route = handler(origin_point, dest_point, nearest_ports, 
                                  nearest_stations, weight_tons, max_transfers, show_all)
                    if route:
                        routes.extend(route)
                except Exception as e:
                    print(f"Error calculating route: {e}")
        elif mode in mode_handlers:
            # Calculate specific mode
            route = mode_handlers[mode](origin_point, dest_point, nearest_ports,
                                      nearest_stations, weight_tons, max_transfers, show_all)
            if route:
                routes.extend(route)

        return routes

    def _calculate_truck_only_route(self, origin_point: Point, dest_point: Point,
                                  nearest_ports: Dict, nearest_stations: Dict,
                                  weight_tons: float, max_transfers: int, 
                                  show_all: bool) -> List[Dict]:
        """Calculate truck-only route"""
        truck_route = self._calculate_truck_route(origin_point, dest_point, weight_tons)
        return [truck_route] if truck_route else []

    def _calculate_truck_ship_route(self, origin_point: Point, dest_point: Point,
                                  nearest_ports: Dict, nearest_stations: Dict,
                                  weight_tons: float, max_transfers: int,
                                  show_all: bool) -> List[Dict]:
        """Calculate truck + ship route"""
        origin_port = nearest_ports["origin_port"]
        dest_port = nearest_ports["dest_port"]

        if not origin_port or not dest_port:
            return []

        # Get truck routes
        truck_routes = self._get_truck_routes_to_ports(origin_point, dest_point, nearest_ports)
        if not truck_routes:
            return []

        # Calculate segments
        segments = self._calculate_truck_ship_segments(
            truck_routes, origin_port, dest_port, weight_tons, max_transfers, show_all
        )
        
        return self._build_combined_route(segments, "truck_ship", weight_tons)

    def _calculate_truck_ship_segments(self, truck_routes: Dict, origin_port: Dict,
                                     dest_port: Dict, weight_tons: float,
                                     max_transfers: int, show_all: bool) -> List[Dict]:
        """Calculate segments for truck + ship route"""
        # Truck to origin port
        truck_to_port = self._extract_truck_route_info(truck_routes["origin_to_port"])
        
        # Ship route
        ship_route = self._find_ship_routes_between_ports(
            {"origin_port": origin_port, "dest_port": dest_port},
            weight_tons, max_transfers, show_all
        )
        ship_info = self._extract_ship_route_info(ship_route, origin_port, dest_port)
        
        # Truck from dest port
        truck_from_port = self._get_truck_route_from_point(
            float(dest_port["X"]), float(dest_port["Y"]),
            truck_routes["port_to_dest"]["end_point"] if "port_to_dest" in truck_routes else None
        )

        return [truck_to_port, ship_info, truck_from_port]

    def _extract_truck_route_info(self, truck_route: Dict) -> Dict:
        """Extract truck route information"""
        geometry = extract_linestring(truck_route.get("geometry"))
        distance = truck_route.get("distance", 0)
        
        return {
            "geometry": geometry,
            "distance": distance,
            "emissions": self._calculate_co2_emissions("truck", 1, distance / 1000),
            "type": "truck"
        }

    def _extract_ship_route_info(self, ship_route: List, origin_port: Dict, 
                               dest_port: Dict) -> Dict:
        """Extract ship route information"""
        if ship_route and isinstance(ship_route[0], dict) and "geometry" in ship_route[0]:
            geometry = ship_route[0]["geometry"]
            distance = ship_route[0].get("route_info", {}).get("distance", 0)
        else:
            # Fallback to straight line
            geometry = LineString([
                (float(origin_port["X"]), float(origin_port["Y"])),
                (float(dest_port["X"]), float(dest_port["Y"]))
            ])
            distance = self._calculate_distance(
                float(origin_port["Y"]), float(origin_port["X"]),
                float(dest_port["Y"]), float(dest_port["X"])
            )

        return {
            "geometry": geometry,
            "distance": distance,
            "emissions": self._calculate_co2_emissions("ship", 1, distance / 1000),
            "type": "ship"
        }

    def _get_truck_route_from_point(self, from_lon: float, from_lat: float,
                                  dest_point: Point) -> Dict:
        """Get truck route from a point to destination"""
        route = self._route_truck_mm(from_lon, from_lat, dest_point.x, dest_point.y)
        
        if route:
            geometry = LineString(route["geometry"]["coordinates"])
            distance = route["distance_km"] * 1000  # Convert to meters
        else:
            # Fallback
            geometry = LineString([(from_lon, from_lat), (dest_point.x, dest_point.y)])
            distance = self._calculate_distance(from_lat, from_lon, dest_point.y, dest_point.x)

        return {
            "geometry": geometry,
            "distance": distance,
            "emissions": self._calculate_co2_emissions("truck", 1, distance / 1000),
            "type": "truck"
        }

    def _build_combined_route(self, segments: List[Dict], mode: str, 
                            weight_tons: float) -> List[Dict]:
        """Build combined route from segments"""
        if not segments:
            return []

        total_distance = sum(segment["distance"] for segment in segments)
        emissions = [segment["emissions"] for segment in segments]
        geometries = [segment["geometry"] for segment in segments]

        data_infos = build_data_infos(
            origin_port="", dest_port="", origin_stations="", dest_stations="",
            emissions=emissions, ship_time=0, train_time_minutes=0,
            truck_time_minutes=0, truck_distances_km=[s["distance"]/1000 for s in segments if s["type"] == "truck"]
        )

        combined_routes = self._combine_linestrings(
            mode, geometries, total_distance, data_infos
        )

        return combined_routes if combined_routes else []

    # Similar refactoring would be applied to other mode handlers...
    # _calculate_truck_train_route, _calculate_truck_ship_train_route, etc.

    @timeit("Calculate CO2 Emissions")
    def _calculate_co2_emissions(self, mode: str, weight_tons: float, distance_km: float) -> float:
        """Calculate CO2 emissions"""
        if mode in self.CO2_FACTORS:
            return weight_tons * distance_km * self.CO2_FACTORS[mode]
        return 0

    @timeit("Calculate Distance")
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate geodesic distance between two points"""
        geod = pyproj.Geod(ellps="WGS84")
        _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
        return distance

    # Keep other existing methods but refactor them similarly...
    # _combine_linestrings, _find_optimal_routes, database methods, etc.

    @timeit("Initialize Database")
    def _init_database(self):
        """Initialize database connection pool"""
        if not PSYCOPG2_AVAILABLE:
            print("Warning: psycopg2 not available, truck routes will use fallback method")
            return

        if not self.db_config:
            print("Warning: No database config provided, truck routes will use fallback method")
            return

        try:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1, maxconn=10, **self.db_config
            )
            print("Database connection initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize database connection: {e}")
            print("Truck routes will use fallback method")
            self.db_pool = None

    # Database query methods remain largely the same but can be optimized further...