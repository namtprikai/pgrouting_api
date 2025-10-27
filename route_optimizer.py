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
import json
try:
    import psycopg2
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class RouteOptimizer:
    """
    Class for optimizing multimodal transportation routes
    """
    
    def __init__(self, data_folder_path: str, 
                 db_config: Optional[Dict] = None):
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
                'host': 'localhost',
                'port': 5435,
                'database': 'pgrouting_japan_logistics',
                'user': 'postgres',
                'password': 'pgrouting'
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
        self.co2_factors = {
            'truck': 43,
            'train': 20,
            'ship': 10
        }
        
        # Initialize database connection
        self._init_database()
        
        # Load all data
        self._load_all_data()
    
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
        
        # Load truck route data
        # self._load_truck_route_data()
        
        print("Data loading completed!")
    
    def _load_od_data(self):
        """Load origin-destination data"""
        path = f"{self.data_folder_path}/L013101物流拠点出発地到着地リスト.csv"
        odlist = pd.read_csv(path, encoding='utf-8-sig')
        odlist = odlist.loc[:, ~odlist.columns.str.startswith('Unnamed')]
        self.odlist_gdf = gpd.GeoDataFrame(
            odlist, 
            geometry=gpd.points_from_xy(odlist['Origin_lon'], odlist['Origin_lat'])
        )
        self.odlist_gdf.set_crs(epsg=4326, inplace=True)
    
    def _load_port_data(self):
        """Load port data"""
        minato = pd.read_csv(
            f"{self.data_folder_path}/貨物船_位置情報（国土数値情報）.csv",
            encoding="SHIFT_JIS"
        )
        self.minato_gdf = gpd.GeoDataFrame(
            minato, 
            geometry=gpd.points_from_xy(minato['X'], minato['Y'])
        )
        self.minato_gdf.set_crs(epsg=4326, inplace=True)
        self.minato_gdf["C02_005"] = self.minato_gdf["C02_005"] + "港"
    
    def _load_station_data(self):
        """Load freight station data"""
        station = pd.read_excel(f"{self.data_folder_path}/貨物駅_位置情報.xlsx")
        self.station_gdf = gpd.GeoDataFrame(
            station, 
            geometry=gpd.points_from_xy(station['lon'], station['lat'])
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
        self.ferry_time = self.ferry_time.dropna(subset=['Route_Time'])
        
        # Create port list
        dep_list = self.ferry_time[dep_col].dropna().astype(str).unique()
        arr_list = self.ferry_time[arr_col].dropna().astype(str).unique()
        self.port_list = list(set(list(dep_list) + list(arr_list)))
        
        # Filter ports
        self.minato_gdf = self.minato_gdf[self.minato_gdf['C02_005'].isin(self.port_list)]
    
    def _load_train_schedule(self):
        """Load train schedule"""
        self.train_time = pd.read_excel(f"{self.data_folder_path}/貨物駅_時刻表.xlsx")
        
        # Replace arrival date
        self.train_time['Arrival_Date'] = self.train_time['Arrival_Date'].replace({
            '当日': 0, '翌日': 1, '翌々日': 2, '７日目': 6, 
            '４日目': 3, '５日目': 4, '６日目': 5
        })
        
        # Convert time format
        self.train_time["Track_Entry_Time"] = pd.to_datetime(
            self.train_time["Track_Entry_Time"], format='%H:%M:%S'
        ).dt.time
        self.train_time["Track_Exit_Time"] = pd.to_datetime(
            self.train_time["Track_Exit_Time"], format='%H:%M:%S'
        ).dt.time
        
        # Convert to datetime
        self.train_time["Track_Entry_Time"] = pd.to_datetime(
            self.train_time["Track_Entry_Time"].astype(str), format='%H:%M:%S'
        )
        self.train_time["Track_Exit_Time"] = pd.to_datetime(
            self.train_time["Track_Exit_Time"].astype(str), format='%H:%M:%S'
        )
        
        # Calculate duration
        self.train_time["Updated_Entry_Time"] = self.train_time["Track_Entry_Time"] + \
            self.train_time["Arrival_Date"].apply(lambda x: timedelta(days=x))
        self.train_time["train_Duration"] = (
            self.train_time["Updated_Entry_Time"] - self.train_time["Track_Exit_Time"]
        )
        self.train_time["train_od"] = (
            self.train_time["Departure_Station_Name"].str.replace(" ", "") + "_" + 
            self.train_time["Arrival_Station_Name"].str.replace(" ", "")
        )
        
        # Remove duplicates
        self.train_time = self.train_time.drop_duplicates(subset=["train_od"])
    
    def _load_truck_route_data(self):
        """Load truck route data"""
        self.track_route = gpd.read_file(
            f"{self.data_folder_path}/_NITAS自動車経路探索結果データ.gpkg"
        )
    
    def find_route(self, origin_lat: float, origin_lon: float, 
                   dest_lat: float, dest_lon: float, 
                   weight_tons: float = 10.0, 
                   mode: str = 'all',
                   enable_transfer: bool = False,
                   max_transfers: int = 10,
                   show_all: bool = False) -> Dict:
        """
        Find optimal route between two points with transfer capability
        
        Args:
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            dest_lat: Destination latitude
            dest_lon: Destination longitude
            weight_tons: Cargo weight (tons)
            mode: Route type ('all', 'truck_only', 'truck_ship', 'truck_train')
            enable_transfer: Enable transfer mode
            max_transfers: Maximum number of transfers (default: 10)
            show_all: Show all routes (False: only optimal)
            
        Returns:
            Dict containing optimal route information
        """
        print(f"Finding route from ({origin_lat}, {origin_lon}) to ({dest_lat}, {dest_lon}) with mode: {mode}")
        
        # Validate mode parameter
        valid_modes = ['all', 'truck_only', 'truck_ship', 'truck_train']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        
        # Create origin and destination points
        origin_point = Point(origin_lon, origin_lat)
        dest_point = Point(dest_lon, dest_lat)
        
        # Find nearest ports and stations
        nearest_ports = self._find_nearest_ports(origin_point, dest_point)
        nearest_stations = self._find_nearest_stations(origin_point, dest_point)
        
        # Calculate routes for different transportation modes
        routes = self._calculate_routes_by_mode(
            origin_point, dest_point, nearest_ports, nearest_stations, weight_tons, mode,
            enable_transfer, max_transfers, show_all
        )
        
        # Find optimal routes
        optimal_routes = self._find_optimal_routes(routes)
        
        return {
            'origin': {'lat': origin_lat, 'lon': origin_lon},
            'destination': {'lat': dest_lat, 'lon': dest_lon},
            'weight_tons': weight_tons,
            'routes': routes,
            'optimal_routes': optimal_routes
        }
    
    def _find_nearest_ports(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest ports to origin and destination points"""
        # Create temporary GeoDataFrame for origin and destination
        temp_gdf = gpd.GeoDataFrame(
            {'name': ['origin', 'destination']},
            geometry=[origin_point, dest_point],
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS for accurate distance calculations
        temp_gdf = temp_gdf.to_crs('EPSG:3857')  # Web Mercator
        minato_gdf_projected = self.minato_gdf.to_crs('EPSG:3857')
        
        # Find nearest ports
        origin_ports = gpd.sjoin_nearest(
            temp_gdf.iloc[[0]], minato_gdf_projected, 
            how='inner', distance_col='distance'
        )
        dest_ports = gpd.sjoin_nearest(
            temp_gdf.iloc[[1]], minato_gdf_projected, 
            how='inner', distance_col='distance'
        )
        
        return {
            'origin_port': origin_ports.iloc[0] if not origin_ports.empty else None,
            'dest_port': dest_ports.iloc[0] if not dest_ports.empty else None
        }
    
    def _find_nearest_stations(self, origin_point: Point, dest_point: Point) -> Dict:
        """Find nearest stations to origin and destination points"""
        # Create temporary GeoDataFrame for origin and destination
        temp_gdf = gpd.GeoDataFrame(
            {'name': ['origin', 'destination']},
            geometry=[origin_point, dest_point],
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS for accurate distance calculations
        temp_gdf = temp_gdf.to_crs('EPSG:3857')  # Web Mercator
        station_gdf_projected = self.station_gdf.to_crs('EPSG:3857')
        
        # Find nearest stations
        origin_stations = gpd.sjoin_nearest(
            temp_gdf.iloc[[0]], station_gdf_projected, 
            how='inner', distance_col='distance'
        )
        dest_stations = gpd.sjoin_nearest(
            temp_gdf.iloc[[1]], station_gdf_projected, 
            how='inner', distance_col='distance'
        )
        
        return {
            'origin_station': origin_stations.iloc[0] if not origin_stations.empty else None,
            'dest_station': dest_stations.iloc[0] if not dest_stations.empty else None
        }
    
    def _calculate_routes_by_mode(self, origin_point: Point, dest_point: Point,
                                nearest_ports: Dict, nearest_stations: Dict, 
                                weight_tons: float, mode: str,
                                enable_transfer: bool = False,
                                max_transfers: int = 10,
                                show_all: bool = False) -> List[Dict]:
        """Calculate routes by selected mode"""
        routes = []
        
        # Route 1: Truck only
        if mode in ['all', 'truck_only']:
            truck_route = self._calculate_truck_route(origin_point, dest_point, weight_tons)
            if truck_route:
                routes.append(truck_route)
        
        # Route 2: Truck + Ship
        if mode in ['all', 'truck_ship']:
            if (nearest_ports['origin_port'] is not None and 
                nearest_ports['dest_port'] is not None):
                
                # Step 1: Find truck routes to nearest ports
                truck_routes = self._get_truck_routes_to_ports(
                    origin_point, dest_point, nearest_ports
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
        
        # Route 3: Truck + Train
        if mode in ['all', 'truck_train']:
            if (nearest_stations['origin_station'] is not None and 
                nearest_stations['dest_station'] is not None):
                
                # Step 1: Find truck routes to nearest stations
                truck_routes = self._get_truck_routes_to_stations(
                    origin_point, dest_point, nearest_stations
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
        
        return routes
    
    def _calculate_truck_route(self, origin_point: Point, dest_point: Point, 
                              weight_tons: float) -> Optional[Dict]:
        """Calculate pure truck route"""
        try:
            # Get truck route from NITAS data
            truck_info = self._get_truck_route_info(origin_point, dest_point)
            
            if truck_info:
                distance_km = truck_info['distance'] / 1000
                co2_emissions = self._calculate_co2_emissions('truck', weight_tons, distance_km)
                
                return {
                    'mode': 'truck_only',
                    'name': 'Truck only',
                    'total_time_minutes': truck_info['time'],
                    'total_distance_meters': truck_info['distance'],
                    'total_distance_km': distance_km,
                    'co2_emissions_grams': co2_emissions,
                    'truck_time_minutes': truck_info['time'],
                    'truck_distance_km': distance_km,
                    'geometry': truck_info['geometry']
                }
        except Exception as e:
            print(f"Error calculating truck route: {e}")
        
        return None
    
    def _calculate_ship_route(self, origin_point: Point, dest_point: Point,
                            nearest_ports: Dict, weight_tons: float) -> Optional[Dict]:
        """Calculate truck + ship route"""
        try:
            origin_port = nearest_ports['origin_port']
            dest_port = nearest_ports['dest_port']
            
            # Get truck routes to/from ports using DATABASE
            origin_to_port = self._get_truck_route_info(origin_point, Point(origin_port['X'], origin_port['Y']))
            port_to_dest = self._get_truck_route_info(Point(dest_port['X'], dest_port['Y']), dest_point)
            
            # Get ship route using OLD LOGIC (CSV data)
            ship_info = self._get_ship_route_info(origin_port['C02_005'], dest_port['C02_005'])
            
            if origin_to_port and port_to_dest:
                if ship_info:
                    # Direct ship route found
                    total_time = (
                        origin_to_port['time'] + 
                        port_to_dest['time'] + 
                        ship_info['time'] * 60  # Convert hours to minutes
                    )
                    
                    total_distance = (
                        origin_to_port['distance'] + 
                        port_to_dest['distance'] + 
                        ship_info['distance']
                    )
                    
                    co2_emissions = (
                        self._calculate_co2_emissions('truck', weight_tons, origin_to_port['distance']/1000) +
                        self._calculate_co2_emissions('truck', weight_tons, port_to_dest['distance']/1000) +
                        self._calculate_co2_emissions('ship', weight_tons, ship_info['distance']/1000)
                    )
                else:
                    # No direct ship route, save truck routes for transfer logic
                    self._last_truck_routes_ship = {
                        'origin_to_port': origin_to_port,
                        'port_to_dest': port_to_dest
                    }
                    self._last_ports = {
                        'origin_port': origin_port,
                        'dest_port': dest_port
                    }
                    return {
                        'truck_routes': {
                            'origin_to_port': origin_to_port,
                            'port_to_dest': port_to_dest
                        },
                        'ports': {
                            'origin_port': origin_port,
                            'dest_port': dest_port
                        }
                    }
                
                # Use actual geometries from database for truck routes
                truck_geom_1 = origin_to_port.get('geometry')
                truck_geom_2 = port_to_dest.get('geometry')
                
                # Combine geometries: truck1 + ship + truck2
                # Use actual truck geometries from database
                if truck_geom_1 and truck_geom_2:
                    # Use actual truck geometries from database
                    truck1_coords = self._get_geometry_coords(truck_geom_1)
                    truck2_coords = self._get_geometry_coords(truck_geom_2)
                    
                    
                    # Combine: truck1 + ship + truck2
                    # Create simple ship segment between ports
                    ship_coords = [
                        (origin_port['X'], origin_port['Y']),
                        (dest_port['X'], dest_port['Y'])
                    ]
                    
                    # Combine all coordinates
                    combined_coords = truck1_coords + ship_coords + truck2_coords[1:]  # Skip first point of truck2 to avoid duplication
                    # Fallback to straight line if no truck geometries
                    combined_coords = [
                        (origin_point.x, origin_point.y),
                        (origin_port['X'], origin_port['Y']),
                        (dest_port['X'], dest_port['Y']),
                        (dest_point.x, dest_point.y)
                    ]
                
                # Ensure all coordinates are tuples, not Point objects
                try:
                    # Convert all coordinates to tuples
                    clean_coords = []
                    for coord in combined_coords:
                        if hasattr(coord, 'x') and hasattr(coord, 'y'):
                            # It's a Point object
                            clean_coords.append((float(coord.x), float(coord.y)))
                        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                            # It's already a tuple/list
                            clean_coords.append((float(coord[0]), float(coord[1])))
                    
                    geometry = LineString(clean_coords)
                except Exception as e:
                    pass
                    # Fallback to simple straight line
                    geometry = LineString([
                        (origin_point.x, origin_point.y),
                        (origin_port['X'], origin_port['Y']),
                        (dest_port['X'], dest_port['Y']),
                        (dest_point.x, dest_point.y)
                    ])
                
                
                return {
                    'mode': 'truck_ship',
                    'total_time_minutes': total_time,
                    'total_distance_meters': total_distance,
                    'total_distance_km': total_distance / 1000,
                    'co2_emissions_grams': co2_emissions,
                    'origin_port': origin_port['C02_005'],
                    'dest_port': dest_port['C02_005'],
                    'origin_station': '',  # No station for ship route
                    'dest_station': '',    # No station for ship route
                    'transfer_port': '',   # No transfer port
                    'transfer_station': '',  # No transfer station
                    'ship_time_hours': ship_info['time'],
                    'train_time_minutes': 0,  # No train time
                    'geometry': geometry
                }
        except Exception as e:
            print(f"Error calculating ship route: {e}")
        
        return None
    
    def _calculate_train_route(self, origin_point: Point, dest_point: Point,
                             nearest_stations: Dict, weight_tons: float) -> Optional[Dict]:
        """Calculate truck + train route"""
        try:
            origin_station = nearest_stations['origin_station']
            dest_station = nearest_stations['dest_station']

            # Get truck routes to/from stations using DATABASE FIRST
            origin_to_station = self._get_truck_route_info(
                origin_point, Point(origin_station['lon'], origin_station['lat'])
            )
            station_to_dest = self._get_truck_route_info(
                Point(dest_station['lon'], dest_station['lat']), dest_point
            )
            
            # Now find train route between stations
            train_info = self._get_train_route_info(
                origin_station['Station_Code'], dest_station['Station_Code']
            )
            
            if origin_to_station and station_to_dest:
                if train_info:
                    # Direct train route found
                    total_time = (
                        origin_to_station['time'] + 
                        station_to_dest['time'] + 
                        train_info['time']
                    )
                    
                    total_distance = (
                        origin_to_station['distance'] + 
                        station_to_dest['distance'] + 
                        train_info['distance'] * 1000  # Convert km to meters
                    )
                    
                    co2_emissions = (
                        self._calculate_co2_emissions('truck', weight_tons, origin_to_station['distance']/1000) +
                        self._calculate_co2_emissions('truck', weight_tons, station_to_dest['distance']/1000) +
                        self._calculate_co2_emissions('train', weight_tons, train_info['distance'])
                    )
                else:
                    # No direct train route, save truck routes and stations for transfer logic
                    self._last_truck_routes = {
                        'origin_to_station': origin_to_station,
                        'station_to_dest': station_to_dest
                    }
                    self._last_stations = {
                        'origin_station': origin_station,
                        'dest_station': dest_station
                    }
                    return {
                        'truck_routes': {
                            'origin_to_station': origin_to_station,
                            'station_to_dest': station_to_dest
                        },
                        'stations': {
                            'origin_station': origin_station,
                            'dest_station': dest_station
                        }
                    }
                
                # Use actual geometries from database for truck routes
                truck_geom_1 = origin_to_station.get('geometry')
                truck_geom_2 = station_to_dest.get('geometry')
                
                # Create train geometry (straight line between stations)
                train_geom = LineString([
                    (origin_station['lon'], origin_station['lat']),
                    (dest_station['lon'], dest_station['lat'])
                ])
                
                # Combine geometries: truck1 + train + truck2
                # Use actual truck geometries from database
                if truck_geom_1 and truck_geom_2:
                    # Use actual truck geometries from database
                    truck1_coords = self._get_geometry_coords(truck_geom_1)
                    truck2_coords = self._get_geometry_coords(truck_geom_2)
                    
                    
                    # Combine: truck1 + train + truck2
                    # Create simple train segment between stations
                    train_coords = [
                        (origin_station['lon'], origin_station['lat']),
                        (dest_station['lon'], dest_station['lat'])
                    ]
                    
                    # Combine all coordinates
                    combined_coords = truck1_coords + train_coords + truck2_coords[1:]  # Skip first point of truck2 to avoid duplication
                    # Fallback to straight line if no truck geometries
                    combined_coords = [
                        (origin_point.x, origin_point.y),
                        (origin_station['lon'], origin_station['lat']),
                        (dest_station['lon'], dest_station['lat']),
                        (dest_point.x, dest_point.y)
                    ]
                
                # Ensure all coordinates are tuples, not Point objects
                try:
                    # Convert all coordinates to tuples
                    clean_coords = []
                    for coord in combined_coords:
                        if hasattr(coord, 'x') and hasattr(coord, 'y'):
                            # It's a Point object
                            clean_coords.append((float(coord.x), float(coord.y)))
                        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                            # It's already a tuple/list
                            clean_coords.append((float(coord[0]), float(coord[1])))
                    
                    geometry = LineString(clean_coords)
                except Exception as e:
                    pass
                    # Fallback to simple straight line
                    geometry = LineString([
                        (origin_point.x, origin_point.y),
                        (origin_station['lon'], origin_station['lat']),
                        (dest_station['lon'], dest_station['lat']),
                        (dest_point.x, dest_point.y)
                    ])
                
                
                return {
                    'mode': 'truck_train',
                    'total_time_minutes': total_time,
                    'total_distance_meters': total_distance,
                    'total_distance_km': total_distance / 1000,
                    'co2_emissions_grams': co2_emissions,
                    'origin_port': '',  # No port for train route
                    'dest_port': '',    # No port for train route
                    'origin_station': origin_station['Station_Name'],
                    'dest_station': dest_station['Station_Name'],
                    'transfer_port': '',   # No transfer port
                    'transfer_station': '',  # No transfer station
                    'ship_time_hours': 0,  # No ship time
                    'train_time_minutes': train_info['time'],
                    'geometry': geometry
                }
        except Exception as e:
            print(f"Error calculating train route: {e}")
        
        return None
    
    def _get_truck_routes_to_stations(self, origin_point: Point, dest_point: Point, 
                                     nearest_stations: Dict) -> Optional[Dict]:
        """Step 1: Find truck routes to nearest stations"""
        try:
            origin_station = nearest_stations['origin_station']
            dest_station = nearest_stations['dest_station']
            
            origin_to_station = self._get_truck_route_info(
                origin_point, Point(origin_station['lon'], origin_station['lat'])
            )
            station_to_dest = self._get_truck_route_info(
                Point(dest_station['lon'], dest_station['lat']), dest_point
            )
            
            if origin_to_station and station_to_dest:
                return {
                    'origin_to_station': origin_to_station,
                    'station_to_dest': station_to_dest,
                    'origin_station': origin_station,
                    'dest_station': dest_station
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error getting truck routes to stations: {e}")
            return None
    
    def _find_train_routes_between_stations(self, nearest_stations: Dict, weight_tons: float,
                                          max_transfers: int, show_all: bool) -> List[Dict]:
        """Step 2: Find train routes between stations (direct or via transfer)"""
        try:
            origin_station = nearest_stations['origin_station']
            dest_station = nearest_stations['dest_station']
            
            # Try to find direct train route first
            direct_train = self._get_train_route_info(
                origin_station['Station_Code'], dest_station['Station_Code']
            )
            
            if direct_train:
                return [{
                    'type': 'direct',
                    'route_info': direct_train,
                    'transfer_stations': []
                }]
            else:
                # Find train routes via transfer
                transfer_routes = self._find_train_transfer_routes(
                    origin_station, dest_station, max_transfers
                )
                return transfer_routes
                
        except Exception as e:
            print(f"Error finding train routes between stations: {e}")
            return []
    
    def _combine_truck_train_routes(self, truck_routes: Dict, train_routes: List[Dict],
                                  nearest_stations: Dict, weight_tons: float) -> List[Dict]:
        """Step 3: Combine truck routes + train routes"""
        try:
            combined_routes = []
            origin_station = nearest_stations['origin_station']
            dest_station = nearest_stations['dest_station']
            
            for train_route in train_routes:
                if train_route['type'] == 'direct':
                    # Direct train route
                    route = self._create_combined_route(
                        truck_routes, train_route['route_info'], 
                        origin_station, dest_station, weight_tons, []
                    )
                    if route:
                        combined_routes.append(route)
                else:
                    # Transfer train route
                    route = self._create_combined_route(
                        truck_routes, train_route['route_info'],
                        origin_station, dest_station, weight_tons, 
                        train_route['transfer_stations']
                    )
                    if route:
                        combined_routes.append(route)
            
            return combined_routes
            
        except Exception as e:
            print(f"Error combining truck train routes: {e}")
            return []
    
    def _find_train_transfer_routes(self, origin_station: Dict, dest_station: Dict, 
                                  max_transfers: int) -> List[Dict]:
        """Find train routes via transfer"""
        try:
            # Logic to find transfer routes (using old logic but only for train routes)
            # Find all stations reachable from origin station
            from_origin = self.train_time[
                self.train_time['Departure_Station_Code'] == origin_station['Station_Code']
            ]['Arrival_Station_Code'].unique()
            
            # Find all stations that can reach destination station
            to_dest = self.train_time[
                self.train_time['Arrival_Station_Code'] == dest_station['Station_Code']
            ]['Departure_Station_Code'].unique()
            
            # Find transfer stations
            transfer_stations = set(from_origin) & set(to_dest)
            
            routes = []
            for transfer_station in list(transfer_stations)[:max_transfers]:
                # Find train routes via transfer
                leg1 = self._get_train_route_info(origin_station['Station_Code'], transfer_station)
                leg2 = self._get_train_route_info(transfer_station, dest_station['Station_Code'])
                
                if leg1 and leg2:
                    # Calculate total information
                    total_time = leg1['time'] + leg2['time']
                    total_distance = leg1['distance'] + leg2['distance']
                    
                    routes.append({
                        'type': 'transfer',
                        'route_info': {
                            'time': total_time,
                            'distance': total_distance
                        },
                        'transfer_stations': [str(transfer_station)]
                    })
            
            return routes
            
        except Exception as e:
            print(f"Error finding train transfer routes: {e}")
            return []
    
    def _create_combined_route(self, truck_routes: Dict, train_info: Dict,
                              origin_station: Dict, dest_station: Dict, 
                              weight_tons: float, transfer_stations: List) -> Optional[Dict]:
        """Create combined truck + train route"""
        try:
            origin_to_station = truck_routes['origin_to_station']
            station_to_dest = truck_routes['station_to_dest']
            
            # Calculate total time and distance
            total_time = (
                origin_to_station['time'] + 
                station_to_dest['time'] + 
                train_info['time']
            )
            
            total_distance = (
                origin_to_station['distance'] + 
                station_to_dest['distance'] + 
                train_info['distance'] * 1000  # Convert km to meters
            )
            
            # Calculate CO2 emissions
            co2_emissions = (
                self._calculate_co2_emissions('truck', weight_tons, origin_to_station['distance']/1000) +
                self._calculate_co2_emissions('truck', weight_tons, station_to_dest['distance']/1000) +
                self._calculate_co2_emissions('train', weight_tons, train_info['distance'])
            )
            
            # Create combined geometry
            geometry = self._create_combined_geometry(
                origin_to_station, station_to_dest, 
                origin_station, dest_station, transfer_stations
            )
            
            # Create route name
            if transfer_stations:
                route_name = f"Truck + Train (transfer via {', '.join(transfer_stations)})"
                mode = 'truck_train_transfer'
            else:
                route_name = "Truck + Train"
                mode = 'truck_train'
            
            return {
                'mode': mode,
                'total_time_minutes': total_time,
                'total_distance_meters': total_distance,
                'total_distance_km': total_distance / 1000,
                'co2_emissions_grams': co2_emissions,
                'origin_port': '',
                'dest_port': '',
                'origin_station': origin_station['Station_Name'],
                'dest_station': dest_station['Station_Name'],
                'transfer_port': '',
                'transfer_station': ', '.join(str(s) for s in transfer_stations) if transfer_stations else '',
                'ship_time_hours': 0,
                'train_time_minutes': train_info['time'],
                'truck_time_minutes': origin_to_station['time'] + station_to_dest['time'],
                'truck_distance_km': (origin_to_station['distance'] + station_to_dest['distance']) / 1000,
                'geometry': geometry
            }
            
        except Exception as e:
            print(f"Error creating combined route: {e}")
            return None
    
    def _create_combined_geometry(self, origin_to_station: Dict, station_to_dest: Dict,
                                origin_station: Dict, dest_station: Dict, 
                                transfer_stations: List) -> LineString:
        """Create combined truck + train geometry"""
        try:
            # Get truck geometries
            truck_geom_1 = origin_to_station.get('geometry')
            truck_geom_2 = station_to_dest.get('geometry')
            
            if truck_geom_1 and truck_geom_2:
                # Use truck geometries from database
                truck1_coords = self._get_geometry_coords(truck_geom_1)
                truck2_coords = self._get_geometry_coords(truck_geom_2)
                
                # Create train segment
                if transfer_stations:
                    # Train route via transfer - need to find coordinates of transfer stations
                    train_coords = [(origin_station['lon'], origin_station['lat'])]
                    
                    # Add coordinates of transfer stations
                    for transfer_station in transfer_stations:
                        # Find coordinates of transfer station
                        transfer_station_info = self.station_gdf[
                            self.station_gdf['Station_Code'] == transfer_station
                        ]
                        if not transfer_station_info.empty:
                            transfer_coords = (
                                transfer_station_info['lon'].iloc[0], 
                                transfer_station_info['lat'].iloc[0]
                            )
                            train_coords.append(transfer_coords)
                    
                    # Add final destination
                    train_coords.append((dest_station['lon'], dest_station['lat']))
                else:
                    # Direct train route
                    train_coords = [
                        (origin_station['lon'], origin_station['lat']),
                        (dest_station['lon'], dest_station['lat'])
                    ]
                
                # Combine: truck1 + train + truck2
                combined_coords = truck1_coords + train_coords + truck2_coords[1:]
                
                # Ensure coordinates are tuples
                clean_coords = []
                for coord in combined_coords:
                    if hasattr(coord, 'x') and hasattr(coord, 'y'):
                        clean_coords.append((float(coord.x), float(coord.y)))
                    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                        clean_coords.append((float(coord[0]), float(coord[1])))
                
                return LineString(clean_coords)
            else:
                # Fallback: straight line
                return LineString([
                    (origin_to_station.get('start_point', Point(0, 0)).x, origin_to_station.get('start_point', Point(0, 0)).y),
                    (origin_station['lon'], origin_station['lat']),
                    (dest_station['lon'], dest_station['lat']),
                    (station_to_dest.get('end_point', Point(0, 0)).x, station_to_dest.get('end_point', Point(0, 0)).y)
                ])
                
        except Exception as e:
            print(f"Error creating combined geometry: {e}")
            # Fallback to simple straight line
            return LineString([
                (0, 0), (0, 0), (0, 0), (0, 0)
            ])
    
    def _get_truck_routes_to_ports(self, origin_point: Point, dest_point: Point, 
                                  nearest_ports: Dict) -> Optional[Dict]:
        """Step 1: Find truck routes to nearest ports"""
        try:
            origin_port = nearest_ports['origin_port']
            dest_port = nearest_ports['dest_port']
            
            origin_to_port = self._get_truck_route_info(
                origin_point, Point(origin_port['X'], origin_port['Y'])
            )
            port_to_dest = self._get_truck_route_info(
                Point(dest_port['X'], dest_port['Y']), dest_point
            )
            
            if origin_to_port and port_to_dest:
                return {
                    'origin_to_port': origin_to_port,
                    'port_to_dest': port_to_dest,
                    'origin_port': origin_port,
                    'dest_port': dest_port
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error getting truck routes to ports: {e}")
            return None
    
    def _find_ship_routes_between_ports(self, nearest_ports: Dict, weight_tons: float,
                                       max_transfers: int, show_all: bool) -> List[Dict]:
        """Step 2: Find ship routes between ports (direct or via transfer)"""
        try:
            origin_port = nearest_ports['origin_port']
            dest_port = nearest_ports['dest_port']
            
            # Try to find direct ship route first
            direct_ship = self._get_ship_route_info(
                origin_port['C02_005'], dest_port['C02_005']
            )
            
            if direct_ship:
                return [{
                    'type': 'direct',
                    'route_info': direct_ship,
                    'transfer_ports': []
                }]
            else:
                # Find ship routes via transfer
                transfer_routes = self._find_ship_transfer_routes(
                    origin_port, dest_port, max_transfers
                )
                return transfer_routes
                
        except Exception as e:
            print(f"Error finding ship routes between ports: {e}")
            return []
    
    def _combine_truck_ship_routes(self, truck_routes: Dict, ship_routes: List[Dict],
                                  nearest_ports: Dict, weight_tons: float) -> List[Dict]:
        """Step 3: Combine truck routes + ship routes"""
        try:
            combined_routes = []
            origin_port = nearest_ports['origin_port']
            dest_port = nearest_ports['dest_port']
            
            for ship_route in ship_routes:
                if ship_route['type'] == 'direct':
                    # Direct ship route
                    route = self._create_combined_ship_route(
                        truck_routes, ship_route['route_info'], 
                        origin_port, dest_port, weight_tons, []
                    )
                    if route:
                        combined_routes.append(route)
                else:
                    # Transfer ship route
                    route = self._create_combined_ship_route(
                        truck_routes, ship_route['route_info'],
                        origin_port, dest_port, weight_tons, 
                        ship_route['transfer_ports']
                    )
                    if route:
                        combined_routes.append(route)
            
            return combined_routes
            
        except Exception as e:
            print(f"Error combining truck ship routes: {e}")
            return []
    
    def _find_ship_transfer_routes(self, origin_port: Dict, dest_port: Dict, 
                                  max_transfers: int) -> List[Dict]:
        """Find ship routes via transfer"""
        try:
            # Logic to find transfer routes (using old logic but only for ship routes)
            # Find all ports reachable from origin port
            from_origin = self.ferry_time[
                self.ferry_time['Departure_Location_(National_Land_Numerical_Information_Format)'] == origin_port['C02_005']
            ]['Arrival_Location_(National_Land_Numerical_Information_Format)'].unique()
            
            # Find all ports that can reach destination port
            to_dest = self.ferry_time[
                self.ferry_time['Arrival_Location_(National_Land_Numerical_Information_Format)'] == dest_port['C02_005']
            ]['Departure_Location_(National_Land_Numerical_Information_Format)'].unique()
            
            # Find transfer ports
            transfer_ports = set(from_origin) & set(to_dest)
            
            routes = []
            for transfer_port in list(transfer_ports)[:max_transfers]:
                # Find ship routes via transfer
                leg1 = self._get_ship_route_info(origin_port['C02_005'], transfer_port)
                leg2 = self._get_ship_route_info(transfer_port, dest_port['C02_005'])
                
                if leg1 and leg2:
                    # Calculate total information
                    total_time = leg1['time'] + leg2['time']
                    total_distance = leg1['distance'] + leg2['distance']
                    
                    routes.append({
                        'type': 'transfer',
                        'route_info': {
                            'time': total_time,
                            'distance': total_distance
                        },
                        'transfer_ports': [str(transfer_port)]
                    })
            
            return routes
            
        except Exception as e:
            print(f"Error finding ship transfer routes: {e}")
            return []
    
    def _create_combined_ship_route(self, truck_routes: Dict, ship_info: Dict,
                                   origin_port: Dict, dest_port: Dict, 
                                   weight_tons: float, transfer_ports: List) -> Optional[Dict]:
        """Create combined truck + ship route"""
        try:
            origin_to_port = truck_routes['origin_to_port']
            port_to_dest = truck_routes['port_to_dest']
            
            # Calculate total time and distance
            total_time = (
                origin_to_port['time'] + 
                port_to_dest['time'] + 
                ship_info['time'] * 60  # Convert hours to minutes
            )
            
            total_distance = (
                origin_to_port['distance'] + 
                port_to_dest['distance'] + 
                ship_info['distance']
            )
            
            # Calculate CO2 emissions
            co2_emissions = (
                self._calculate_co2_emissions('truck', weight_tons, origin_to_port['distance']/1000) +
                self._calculate_co2_emissions('truck', weight_tons, port_to_dest['distance']/1000) +
                self._calculate_co2_emissions('ship', weight_tons, ship_info['distance']/1000)
            )
            
            # Create combined geometry
            geometry = self._create_combined_ship_geometry(
                origin_to_port, port_to_dest, 
                origin_port, dest_port, transfer_ports
            )
            
            # Create route name
            if transfer_ports:
                route_name = f"Truck + Ship (transfer via {', '.join(transfer_ports)})"
                mode = 'truck_ship_transfer'
            else:
                route_name = "Truck + Ship"
                mode = 'truck_ship'
            
            return {
                'mode': mode,
                'total_time_minutes': total_time,
                'total_distance_meters': total_distance,
                'total_distance_km': total_distance / 1000,
                'co2_emissions_grams': co2_emissions,
                'origin_port': origin_port['C02_005'],
                'dest_port': dest_port['C02_005'],
                'origin_station': '',
                'dest_station': '',
                'transfer_port': ', '.join(transfer_ports) if transfer_ports else '',
                'transfer_station': '',
                'ship_time_hours': ship_info['time'],
                'train_time_minutes': 0,
                'truck_time_minutes': origin_to_port['time'] + port_to_dest['time'],
                'truck_distance_km': (origin_to_port['distance'] + port_to_dest['distance']) / 1000,
                'geometry': geometry
            }
            
        except Exception as e:
            print(f"Error creating combined ship route: {e}")
            return None
    
    def _create_combined_ship_geometry(self, origin_to_port: Dict, port_to_dest: Dict,
                                      origin_port: Dict, dest_port: Dict, 
                                      transfer_ports: List) -> LineString:
        """Create combined truck + ship geometry"""
        try:
            # Get truck geometries
            truck_geom_1 = origin_to_port.get('geometry')
            truck_geom_2 = port_to_dest.get('geometry')
            
            if truck_geom_1 and truck_geom_2:
                # Use truck geometries from database
                truck1_coords = self._get_geometry_coords(truck_geom_1)
                truck2_coords = self._get_geometry_coords(truck_geom_2)
                
                # Create ship segment
                if transfer_ports:
                    # Ship route via transfer - need to find coordinates of transfer port
                    ship_coords = [(origin_port['X'], origin_port['Y'])]
                    
                    # Add coordinates of transfer ports
                    for transfer_port in transfer_ports:
                        # Find coordinates of transfer port
                        transfer_port_info = self.minato_gdf[
                            self.minato_gdf['C02_005'] == transfer_port
                        ]
                        if not transfer_port_info.empty:
                            transfer_coords = (
                                transfer_port_info['X'].iloc[0], 
                                transfer_port_info['Y'].iloc[0]
                            )
                            ship_coords.append(transfer_coords)
                    
                    # Add final destination
                    ship_coords.append((dest_port['X'], dest_port['Y']))
                else:
                    # Direct ship route
                    ship_coords = [
                        (origin_port['X'], origin_port['Y']),
                        (dest_port['X'], dest_port['Y'])
                    ]
                
                # Combine: truck1 + ship + truck2
                combined_coords = truck1_coords + ship_coords + truck2_coords[1:]
                
                # Ensure coordinates are tuples
                clean_coords = []
                for coord in combined_coords:
                    if hasattr(coord, 'x') and hasattr(coord, 'y'):
                        clean_coords.append((float(coord.x), float(coord.y)))
                    elif isinstance(coord, (tuple, list)) and len(coord) == 2:
                        clean_coords.append((float(coord[0]), float(coord[1])))
                
                return LineString(clean_coords)
            else:
                # Fallback: straight line
                return LineString([
                    (origin_to_port.get('start_point', Point(0, 0)).x, origin_to_port.get('start_point', Point(0, 0)).y),
                    (origin_port['X'], origin_port['Y']),
                    (dest_port['X'], dest_port['Y']),
                    (port_to_dest.get('end_point', Point(0, 0)).x, port_to_dest.get('end_point', Point(0, 0)).y)
                ])
                
        except Exception as e:
            print(f"Error creating combined ship geometry: {e}")
            # Fallback to simple straight line
            return LineString([
                (0, 0), (0, 0), (0, 0), (0, 0)
            ])
    
    def _get_geometry_coords(self, geometry) -> List[Tuple[float, float]]:
        """Get coordinates from geometry (handles both Shapely and GeoJSON)"""
        if isinstance(geometry, dict):
            # GeoJSON geometry
            if geometry.get('type') == 'LineString':
                return geometry.get('coordinates', [])
            elif geometry.get('type') == 'MultiLineString':
                # Flatten all LineString coordinates
                all_coords = []
                for line in geometry.get('coordinates', []):
                    all_coords.extend(line)
                return all_coords
        elif isinstance(geometry, str):
            # WKT string - try to parse
            try:
                from shapely import wkt
                geom = wkt.loads(geometry)
                if hasattr(geom, 'coords'):
                    return list(geom.coords)
                elif hasattr(geom, 'geoms'):
                    all_coords = []
                    for g in geom.geoms:
                        all_coords.extend(list(g.coords))
                    return all_coords
            except:
                pass
        else:
            # Shapely geometry
            if hasattr(geometry, 'coords'):
                return list(geometry.coords)
            elif hasattr(geometry, 'geoms'):
                # MultiLineString
                all_coords = []
                for geom in geometry.geoms:
                    all_coords.extend(list(geom.coords))
                return all_coords
        return []
    
    def _get_truck_route_info(self, start_point: Point, end_point: Point) -> Optional[Dict]:
        """Get truck route info from database or fallback"""
        # Try database first
        if self.db_pool:
            print(f"Debug: Trying database for truck route from ({start_point.x}, {start_point.y}) to ({end_point.x}, {end_point.y})")
            db_result = self._get_truck_route_info_db(start_point, end_point)
            if db_result:
                coords = self._get_geometry_coords(db_result.get('geometry', {}))
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
            'distance': distance,
            'time': time_minutes,
            'geometry': LineString([start_point, end_point])
        }
    
    def _get_ship_route_info(self, origin_port: str, dest_port: str) -> Optional[Dict]:
        """Get ship route information"""
        route = self.ferry_time[
            (self.ferry_time['Departure_Location_(National_Land_Numerical_Information_Format)'] == origin_port) &
            (self.ferry_time['Arrival_Location_(National_Land_Numerical_Information_Format)'] == dest_port)
        ]
        
        if not route.empty:
            route_time = route['Route_Time'].iloc[0]
            # Calculate distance between ports
            origin_port_data = self.minato_gdf[self.minato_gdf['C02_005'] == origin_port]
            dest_port_data = self.minato_gdf[self.minato_gdf['C02_005'] == dest_port]
            
            if not origin_port_data.empty and not dest_port_data.empty:
                distance = self._calculate_distance(
                    origin_port_data['Y'].iloc[0], origin_port_data['X'].iloc[0],
                    dest_port_data['Y'].iloc[0], dest_port_data['X'].iloc[0]
                )
                
                return {
                    'time': route_time,
                    'distance': distance
                }
        
        return None
    
    def _get_train_route_info(self, origin_station_code: str, dest_station_code: str) -> Optional[Dict]:
        """Get train route information"""
        route = self.train_time[
            (self.train_time['Departure_Station_Code'] == origin_station_code) &
            (self.train_time['Arrival_Station_Code'] == dest_station_code)
        ]
        
        if not route.empty:
            duration = route['train_Duration'].iloc[0]
            distance = route['Distance_(km)'].iloc[0] if 'Distance_(km)' in route.columns else 0
            
            # Convert duration to minutes
            if hasattr(duration, 'total_seconds'):
                time_minutes = duration.total_seconds() / 60
            else:
                time_minutes = 0
            
            return {
                'time': time_minutes,
                'distance': distance
            }
        
        return None
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate geodesic distance between two points"""
        geod = pyproj.Geod(ellps='WGS84')
        _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
        return distance
    
    def _calculate_co2_emissions(self, mode: str, weight_tons: float, distance_km: float) -> float:
        """Calculate CO2 emissions"""
        if mode in self.co2_factors:
            return weight_tons * distance_km * self.co2_factors[mode]
        return 0
    
    def _find_optimal_routes(self, routes: List[Dict]) -> Dict:
        """Find optimal routes by different criteria"""
        if not routes:
            return {}
        
        # Find minimum time route
        min_time_route = min(routes, key=lambda x: x['total_time_minutes'])
        
        # Find minimum distance route
        min_distance_route = min(routes, key=lambda x: x['total_distance_meters'])
        
        # Find minimum CO2 route
        min_co2_route = min(routes, key=lambda x: x['co2_emissions_grams'])
        
        # Create copies without geometry for optimal routes (to avoid JSON serialization issues)
        def create_route_summary(route):
            summary = {k: v for k, v in route.items() if k != 'geometry'}
            return summary
        
        return {
            'fastest': create_route_summary(min_time_route),
            'shortest': create_route_summary(min_distance_route),
            'greenest': create_route_summary(min_co2_route)
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to GeoJSON file"""
        # Convert results to GeoJSON format
        geojson_data = self._convert_to_geojson(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
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
            "properties": convert_numpy_types({
                "origin": results.get('origin', {}),
                "destination": results.get('destination', {}),
                "weight_tons": results.get('weight_tons', 10.0),
                "optimal_routes": results.get('optimal_routes', {}),
                "search_options": {
                    "criteria_used": results.get('criteria_used', 'fastest'),
                    "show_all": results.get('show_all', False),
                    "mode": results.get('mode', 'all'),
                    "enable_transfer": results.get('enable_transfer', False),
                    "max_transfers": results.get('max_transfers', 2)
                }
            })
        }
        
        # Convert each route to a GeoJSON feature
        for route in results.get('routes', []):
            if 'geometry' in route and route['geometry']:
                try:
                    geometry = route['geometry']
                    
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
                                print(f"Warning: Could not parse geometry string: {geometry[:100]}...")
                                continue
                    elif isinstance(geometry, LineString):
                        # If it's already a LineString object, use it directly
                        pass
                    elif isinstance(geometry, dict):
                        # Already GeoJSON, use as is
                        pass
                    else:
                        # Try to convert other geometry types
                        try:
                            geometry = wkt.loads(str(geometry))
                        except:
                            print(f"Warning: Could not convert geometry: {type(geometry)}")
                            continue
                    
                    # Create feature
                    if isinstance(geometry, dict):
                        # Already GeoJSON
                        feature = {
                            "type": "Feature",
                            "geometry": geometry,
                            "properties": convert_numpy_types({
                                "mode": route.get('mode', ''),
                                "total_time_minutes": route.get('total_time_minutes', 0),
                                "total_distance_meters": route.get('total_distance_meters', 0),
                                "total_distance_km": route.get('total_distance_km', 0),
                                "co2_emissions_grams": route.get('co2_emissions_grams', 0),
                                "origin_port": route.get('origin_port', ''),
                                "dest_port": route.get('dest_port', ''),
                                "origin_station": route.get('origin_station', ''),
                                "dest_station": route.get('dest_station', ''),
                                "transfer_port": route.get('transfer_port', ''),
                                "transfer_station": route.get('transfer_station', ''),
                                "ship_time_hours": route.get('ship_time_hours', 0),
                                "train_time_minutes": route.get('train_time_minutes', 0),
                                "truck_time_minutes": route.get('truck_time_minutes', 0),
                                "truck_distance_km": route.get('truck_distance_km', 0)
                            })
                        }
                    else:
                        # Shapely geometry
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": geometry.geom_type,
                                "coordinates": list(geometry.coords)
                            },
                            "properties": convert_numpy_types({
                                "mode": route.get('mode', ''),
                                "total_time_minutes": route.get('total_time_minutes', 0),
                                "total_distance_meters": route.get('total_distance_meters', 0),
                                "total_distance_km": route.get('total_distance_km', 0),
                                "co2_emissions_grams": route.get('co2_emissions_grams', 0),
                                "origin_port": route.get('origin_port', ''),
                                "dest_port": route.get('dest_port', ''),
                                "origin_station": route.get('origin_station', ''),
                                "dest_station": route.get('dest_station', ''),
                                "transfer_port": route.get('transfer_port', ''),
                                "transfer_station": route.get('transfer_station', ''),
                                "ship_time_hours": route.get('ship_time_hours', 0),
                                "train_time_minutes": route.get('train_time_minutes', 0),
                                "truck_time_minutes": route.get('truck_time_minutes', 0),
                                "truck_distance_km": route.get('truck_distance_km', 0)
                            })
                        }
                    
                    geojson["features"].append(feature)
                    
                except Exception as e:
                    print(f"Warning: Could not convert geometry for route {route.get('name', '')}: {e}")
        
        return geojson

    def _find_ship_routes_with_transfer(self, origin_point: Point, dest_point: Point,
                                      nearest_ports: Dict, weight_tons: float, 
                                      max_transfers: int) -> List[Dict]:
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
            
        origin_port = nearest_ports['origin_port']
        dest_port = nearest_ports['dest_port']
        
        if origin_port is None or dest_port is None:
            return routes
        
        # Find all ports reachable from origin port
        from_origin = self.ferry_time[
            self.ferry_time['Departure_Location_(National_Land_Numerical_Information_Format)'] == origin_port['C02_005']
        ]['Arrival_Location_(National_Land_Numerical_Information_Format)'].unique()
        
        # Find all ports that can reach destination port
        to_dest = self.ferry_time[
            self.ferry_time['Arrival_Location_(National_Land_Numerical_Information_Format)'] == dest_port['C02_005']
        ]['Departure_Location_(National_Land_Numerical_Information_Format)'].unique()
        
        # Find transfer ports
        transfer_ports = set(from_origin) & set(to_dest)
        
        for transfer_port in list(transfer_ports)[:max_transfers]:
            try:
                # Try to get truck routes and ports from main calculation first
                truck_routes = None
                ports = None
                if hasattr(self, '_last_truck_routes_ship') and hasattr(self, '_last_ports'):
                    truck_routes = self._last_truck_routes_ship
                    ports = self._last_ports
                
                if truck_routes and ports:
                    # Use existing truck routes and ports
                    origin_to_port = truck_routes['origin_to_port']
                    port_to_dest = truck_routes['port_to_dest']
                    # Use saved ports instead of parameters
                    origin_port = ports['origin_port']
                    dest_port = ports['dest_port']
                else:
                    # Calculate truck routes if not available
                    origin_to_port = self._get_truck_route_info(
                        origin_point, Point(origin_port['X'], origin_port['Y'])
                    )
                    port_to_dest = self._get_truck_route_info(
                        Point(dest_port['X'], dest_port['Y']), dest_point
                    )
                
                if origin_to_port and port_to_dest:
                    route = self._calculate_ship_route_with_transfer(
                        origin_point, dest_point, origin_port, dest_port, 
                        transfer_port, weight_tons, origin_to_port, port_to_dest
                    )
                else:
                    route = None
                if route:
                    routes.append(route)
            except Exception as e:
                print(f"Error calculating transfer route via {transfer_port}: {e}")
                continue
        
        return routes

    def _calculate_ship_route_with_transfer(self, origin_point: Point, dest_point: Point,
                                         origin_port: Dict, dest_port: Dict, 
                                         transfer_port: str, weight_tons: float,
                                         origin_to_port: Dict, port_to_dest: Dict) -> Optional[Dict]:
        """
        Calculate ship route with transfer
        """
        try:
            # Find transfer port information
            transfer_port_info = self.minato_gdf[
                self.minato_gdf['C02_005'] == transfer_port
            ]
            
            if transfer_port_info.empty:
                return None
            
            transfer_port_data = transfer_port_info.iloc[0]
            
            # Calculate ship route using OLD LOGIC (CSV data) - truck routes already passed
            leg1 = self._get_ship_route_info(origin_port['C02_005'], transfer_port)
            leg2 = self._get_ship_route_info(transfer_port, dest_port['C02_005'])
            
            if origin_to_port and port_to_dest and leg1 and leg2:
                total_time = (
                    origin_to_port['time'] + 
                    port_to_dest['time'] + 
                    (leg1['time'] + leg2['time']) * 60
                )
                
                total_distance = (
                    origin_to_port['distance'] + 
                    port_to_dest['distance'] + 
                    leg1['distance'] + leg2['distance']
                )
                
                co2_emissions = (
                    self._calculate_co2_emissions('truck', weight_tons, origin_to_port['distance']/1000) +
                    self._calculate_co2_emissions('truck', weight_tons, port_to_dest['distance']/1000) +
                    self._calculate_co2_emissions('ship', weight_tons, leg1['distance']/1000) +
                    self._calculate_co2_emissions('ship', weight_tons, leg2['distance']/1000)
                )
                
                geometry = LineString([
                    (origin_point.x, origin_point.y),
                    (origin_port['X'], origin_port['Y']),
                    (transfer_port_data['X'], transfer_port_data['Y']),
                    (dest_port['X'], dest_port['Y']),
                    (dest_point.x, dest_point.y)
                ])
                
                return {
                    'mode': 'truck_ship_transfer',
                    'total_time_minutes': total_time,
                    'total_distance_meters': total_distance,
                    'total_distance_km': total_distance / 1000,
                    'co2_emissions_grams': co2_emissions,
                    'origin_port': origin_port['C02_005'],
                    'dest_port': dest_port['C02_005'],
                    'transfer_port': transfer_port,
                    'ship_time_hours': leg1['time'] + leg2['time'],
                    'geometry': geometry
                }
        except Exception as e:
            print(f"Error calculating ship route with transfer: {e}")
        
        return None

    def _find_train_routes_with_transfer(self, origin_point: Point, dest_point: Point,
                                       nearest_stations: Dict, weight_tons: float, 
                                       max_transfers: int) -> List[Dict]:
        """
        Find train routes with transfer
        """
        routes = []
        
        if self.train_time is None:
            return routes
            
        origin_station = nearest_stations['origin_station']
        dest_station = nearest_stations['dest_station']
        
        if origin_station is None or dest_station is None:
            return routes
        
        # Find all stations reachable from origin station
        from_origin = self.train_time[
            self.train_time['Departure_Station_Code'] == origin_station['Station_Code']
        ]['Arrival_Station_Code'].unique()
        
        # Find all stations that can reach destination station
        to_dest = self.train_time[
            self.train_time['Arrival_Station_Code'] == dest_station['Station_Code']
        ]['Departure_Station_Code'].unique()
        
        # Find transfer stations
        transfer_stations = set(from_origin) & set(to_dest)

        
        for transfer_station in list(transfer_stations)[:max_transfers]:
            try:
                # Try to get truck routes and stations from main calculation first
                truck_routes = None
                stations = None
                if hasattr(self, '_last_truck_routes') and hasattr(self, '_last_stations'):
                    truck_routes = self._last_truck_routes
                    stations = self._last_stations
                
                if truck_routes and stations:
                    # Use existing truck routes and stations
                    origin_to_station = truck_routes['origin_to_station']
                    station_to_dest = truck_routes['station_to_dest']
                    # Use saved stations instead of parameters
                    origin_station = stations['origin_station']
                    dest_station = stations['dest_station']
                else:
                    # Calculate truck routes if not available
                    origin_to_station = self._get_truck_route_info(
                        origin_point, Point(origin_station['lon'], origin_station['lat'])
                    )
                    station_to_dest = self._get_truck_route_info(
                        Point(dest_station['lon'], dest_station['lat']), dest_point
                    )
                
                if origin_to_station and station_to_dest:
                    route = self._calculate_train_route_with_transfer(
                        origin_point, dest_point, origin_station, dest_station, 
                        transfer_station, weight_tons, origin_to_station, station_to_dest
                    )
                else:
                    route = None
                if route:
                    routes.append(route)
            except Exception as e:
                print(f"Error calculating train transfer route via {transfer_station}: {e}")
                continue
        
        return routes

    def _calculate_train_route_with_transfer(self, origin_point: Point, dest_point: Point,
                                           origin_station: Dict, dest_station: Dict, 
                                           transfer_station: int, weight_tons: float,
                                           origin_to_station: Dict, station_to_dest: Dict) -> Optional[Dict]:
        """
        Calculate train route with transfer
        """
        try:
            # Find transfer station information
            transfer_station_info = self.station_gdf[
                self.station_gdf['Station_Code'] == transfer_station
            ]
            
            if transfer_station_info.empty:
                return None
            
            transfer_station_data = transfer_station_info.iloc[0]
            
            # Calculate train route (truck routes already passed)
            leg1 = self._get_train_route_info(origin_station['Station_Code'], transfer_station)
            leg2 = self._get_train_route_info(transfer_station, dest_station['Station_Code'])
            if origin_to_station and station_to_dest and leg1 and leg2:
                total_time = (
                    origin_to_station['time'] + 
                    station_to_dest['time'] + 
                    leg1['time'] + leg2['time']
                )
                
                total_distance = (
                    origin_to_station['distance'] + 
                    station_to_dest['distance'] + 
                    leg1['distance'] + leg2['distance']
                )
                
                co2_emissions = (
                    self._calculate_co2_emissions('truck', weight_tons, origin_to_station['distance']/1000) +
                    self._calculate_co2_emissions('truck', weight_tons, station_to_dest['distance']/1000) +
                    self._calculate_co2_emissions('train', weight_tons, leg1['distance']/1000) +
                    self._calculate_co2_emissions('train', weight_tons, leg2['distance']/1000)
                )
                
                geometry = LineString([
                    (origin_point.x, origin_point.y),
                    (origin_station['lon'], origin_station['lat']),
                    (transfer_station_data['lon'], transfer_station_data['lat']),
                    (dest_station['lon'], dest_station['lat']),
                    (dest_point.x, dest_point.y)
                ])
                
                return {
                    'mode': 'truck_train_transfer',
                    'total_time_minutes': total_time,
                    'total_distance_meters': total_distance,
                    'total_distance_km': total_distance / 1000,
                    'co2_emissions_grams': co2_emissions,
                    'origin_station': origin_station['Station_Name'],
                    'dest_station': dest_station['Station_Name'],
                    'transfer_station': transfer_station,
                    'train_time_minutes': leg1['time'] + leg2['time'],
                    'geometry': geometry
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error calculating train route with transfer: {e}")
        
        return None

    def _get_train_route_info(self, origin_station_code: int, dest_station_code: int) -> Optional[Dict]:
        """
        Get train route information between 2 stations
        """
        if self.train_time is None:
            return None
            
        route = self.train_time[
            (self.train_time['Departure_Station_Code'] == origin_station_code) &
            (self.train_time['Arrival_Station_Code'] == dest_station_code)
        ]
        
        if not route.empty:
            # Calculate travel time from train_Duration
            duration = route['train_Duration'].iloc[0]
            if hasattr(duration, 'total_seconds'):
                time_minutes = duration.total_seconds() / 60
            else:
                time_minutes = 0
            
            # Estimate distance (can be improved by using real data)
            distance_km = time_minutes * 50  # Assume average speed 50 km/h
            
            return {
                'time': time_minutes,
                'distance': distance_km * 1000  # Convert to meters
            }
        
        return None

    def _find_single_ship_route_with_transfer(self, origin_point: Point, dest_point: Point,
                                            nearest_ports: Dict, weight_tons: float, 
                                            max_transfers: int) -> Optional[Dict]:
        """
        Find only 1 optimal ship route with transfer
        Uses similar logic as _find_ship_routes_with_transfer but returns only the best route
        """
        try:
            # Use old function to find all routes with transfer
            all_transfer_routes = self._find_ship_routes_with_transfer(
                origin_point, dest_point, nearest_ports, weight_tons, max_transfers
            )
            
            if not all_transfer_routes:
                return None
            
            # Find best route (shortest time)
            best_route = None
            best_time = float('inf')
            
            for route in all_transfer_routes:
                if route['total_time_minutes'] < best_time:
                    best_time = route['total_time_minutes']
                    best_route = route
            
            return best_route
            
        except Exception as e:
            print(f"Error finding single ship route with transfer: {e}")
            return None

    def _find_single_train_route_with_transfer(self, origin_point: Point, dest_point: Point,
                                             nearest_stations: Dict, weight_tons: float, 
                                             max_transfers: int) -> Optional[Dict]:
        """
        Find only 1 optimal train route with transfer
        Uses similar logic as _find_train_routes_with_transfer but returns only the best route
        """
        try:
            # Use old function to find all routes with transfer
            all_transfer_routes = self._find_train_routes_with_transfer(
                origin_point, dest_point, nearest_stations, weight_tons, max_transfers
            )
            
            if not all_transfer_routes:
                return None
            
            # Find best route (shortest time)
            best_route = None
            best_time = float('inf')
            
            for route in all_transfer_routes:
                if route['total_time_minutes'] < best_time:
                    best_time = route['total_time_minutes']
                    best_route = route
            
            return best_route
            
        except Exception as e:
            print(f"Error finding single train route with transfer: {e}")
            return None

    def _init_database(self):
        """Initialize database connection pool"""
        if not PSYCOPG2_AVAILABLE:
            print("Warning: psycopg2 not available, truck routes will use fallback method")
            self.db_pool = None
            return
            
        if not self.db_config:
            print("Warning: No database config provided, truck routes will use fallback method")
            self.db_pool = None
            return
            
        try:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print("Database connection initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize database connection: {e}")
            print("Truck routes will use fallback method")
            self.db_pool = None

    def _db_query_one(self, query: str, params: tuple) -> Optional[Dict]:
        """Execute a single query and return one result"""
        if not self.db_pool:
            return None
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    return None
                cols = [desc.name for desc in cur.description]
                return {c: v for c, v in zip(cols, row)}
        except Exception as e:
            print(f"Database query error: {e}")
            return None
        finally:
            self.db_pool.putconn(conn)

    def _db_query_all(self, query: str, params: tuple) -> Optional[List[Dict]]:
        """Execute a query and return all results"""
        if not self.db_pool:
            return None
        
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                if not rows:
                    return None
                cols = [desc.name for desc in cur.description]
                result = []
                for row in rows:
                    result.append({c: v for c, v in zip(cols, row)})
                return result
        except Exception as e:
            print(f"Database query error: {e}")
            return None
        finally:
            self.db_pool.putconn(conn)

    def _nearest_node_id(self, lon: float, lat: float) -> Optional[Dict]:
        """Find nearest node ID from database"""
        return self._db_query_one("SELECT * FROM nearest_node_id(%s, %s)", (lon, lat))

    def _get_node_component(self, start_node: int, end_node: int) -> Optional[List[Dict]]:
        """Get node components to check connectivity"""
        sql = "SELECT * FROM pgr_connectedComponents('SELECT gid AS id, source, target, cost_s as cost, reverse_cost FROM jpn_ways') WHERE node IN (%s, %s)"
        return self._db_query_all(sql, (start_node, end_node))

    def _route_truck_mm(self, o_lon: float, o_lat: float, d_lon: float, d_lat: float, toll_per_km: float = 30.0) -> Optional[Dict]:
        """Get truck route from database using route_truck_mm function - copied from app.py"""
        result = self._db_query_one(
            """
            SELECT geom_geojson, distance_km, travel_time_h, motorway_km, toll_estimate_yen,
                   entry_ic_name, entry_ic_lon, entry_ic_lat,
                   exit_ic_name,  exit_ic_lon,  exit_ic_lat
            FROM route_truck_mm(%s, %s, %s, %s, %s)
            """,
            (o_lon, o_lat, d_lon, d_lat, toll_per_km)
        )
        
        if not result or not result.get("geom_geojson"):
            return None
        
        return {
            "geometry": json.loads(result["geom_geojson"]),
            "distance_km": float(result["distance_km"]),
            "travel_time_h": float(result["travel_time_h"]),
            "motorway_km": float(result["motorway_km"]),
            "toll_estimate_yen": float(result["toll_estimate_yen"]) if result["toll_estimate_yen"] is not None else None,
            "entry_ic": (
                {"name": result["entry_ic_name"], "lon": result["entry_ic_lon"], "lat": result["entry_ic_lat"]}
                if result.get("entry_ic_name") is not None else None
            ),
            "exit_ic": (
                {"name": result["exit_ic_name"], "lon": result["exit_ic_lon"], "lat": result["exit_ic_lat"]}
                if result.get("exit_ic_name") is not None else None
            ),
        }

    def _get_truck_route_info_db(self, start_point: Point, end_point: Point, toll_per_km: float = 30.0) -> Optional[Dict]:
        """Get truck route info from database - using logic from app.py"""
        try:
            # Check if nodes are connected (same logic as app.py)
            print(f"Debug: Finding nearest nodes for ({start_point.x}, {start_point.y}) and ({end_point.x}, {end_point.y})")
            start_node = self._nearest_node_id(start_point.x, start_point.y)
            end_node = self._nearest_node_id(end_point.x, end_point.y)
            
            if not start_node or not end_node:
                print("Debug: Could not find nearest nodes")
                return None
            
            print(f"Debug: Found nodes - start: {start_node['nearest_node_id']}, end: {end_node['nearest_node_id']}")
            
            # Check connectivity (same logic as app.py)
            components = self._get_node_component(start_node['nearest_node_id'], end_node['nearest_node_id'])
            if not components or len(components) < 2:
                print("Debug: Nodes not connected or components not found")
                return None
            
            start_component = components[0]['component']
            end_component = components[1]['component']
            
            if start_component != end_component:
                print(f"Debug: Nodes in different components - start: {start_component}, end: {end_component}")
                return None
            
            print("Debug: Nodes are connected, getting route from database")
            
            # Get route from database using app.py logic
            route_result = self._route_truck_mm(
                start_point.x, start_point.y, 
                end_point.x, end_point.y, 
                toll_per_km
            )
            
            if not route_result:
                print("Debug: route_truck_mm returned None")
                return None
            
            print(f"Debug: Got route from database - distance: {route_result['distance_km']} km, time: {route_result['travel_time_h']} h")
            
            # Use geometry directly from database like app.py - no conversion
            geometry = route_result['geometry']
            
            # Return in the format expected by route_optimizer
            return {
                'time': route_result['travel_time_h'] * 60,  # Convert to minutes
                'distance': route_result['distance_km'] * 1000,  # Convert to meters
                'geometry': geometry  # Use Shapely geometry object
            }
            
        except Exception as e:
            print(f"Error getting truck route from database: {e}")
            import traceback
            traceback.print_exc()
            return None


