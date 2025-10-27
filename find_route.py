#!/usr/bin/env python3
"""
Enhanced script to find optimal routes with transfer capability
"""

import sys
import argparse
from route_optimizer import RouteOptimizer


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal route between two points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Default: Fastest route (automatically finds transfers if needed)
  python find_route.py 35.6762 139.6503 34.6937 135.5023
  
  # Shortest route
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --criteria shortest
  
  # Greenest route (lowest CO2)
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --criteria greenest
  
  # Show all routes
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --show-all
  
  # Customize maximum transfers
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --max-transfers 5
  
  # Save results
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --output results
        """
    )

    parser.add_argument('origin_lat', type=float, help='Origin latitude')
    parser.add_argument('origin_lon', type=float, help='Origin longitude')
    parser.add_argument('dest_lat', type=float, help='Destination latitude')
    parser.add_argument('dest_lon', type=float, help='Destination longitude')
    parser.add_argument('--weight', type=float, default=10.0, help='Cargo weight (tons)')
    parser.add_argument('--data-folder', 
                       default='sample/content/drive/MyDrive/modalshift',
                       help='Path to data folder')
    parser.add_argument('--output', help='File to save results (GeoJSON)')
    parser.add_argument('--criteria', 
                       choices=['fastest', 'shortest', 'greenest'],
                       default='fastest',
                       help='Optimization criteria (fastest: fastest, shortest: shortest, greenest: lowest CO2)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all routes instead of just optimal')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    parser.add_argument('--mode', 
                       choices=['all', 'truck_only', 'truck_ship', 'truck_train'],
                       default='all',
                       help='Route type (all: all, truck_only: truck only, truck_ship: truck+ship, truck_train: truck+train)')
    parser.add_argument('--max-transfers', type=int, default=10,
                       help='Maximum number of transfers (default: 10)')
    
    args = parser.parse_args()
    
    # Validate coordinates
    if not (-90 <= args.origin_lat <= 90) or not (-180 <= args.origin_lon <= 180):
        print("Error: Invalid origin coordinates")
        sys.exit(1)
    
    if not (-90 <= args.dest_lat <= 90) or not (-180 <= args.dest_lon <= 180):
        print("Error: Invalid destination coordinates")
        sys.exit(1)
    
    if args.weight <= 0:
        print("Error: Weight must be greater than 0")
        sys.exit(1)
    
    try:
        # Initialize optimizer
        if args.verbose:
            print('Initializing Route Optimizer...')
        
        # Database configuration (optional)
        db_config = {
            'host': 'localhost',
            'port': 5435,
            'database': 'pgrouting_japan_logistics',
            'user': 'postgres',
            'password': 'pgrouting'
        }
        
        optimizer = RouteOptimizer(args.data_folder, db_config)
        
        if args.verbose:
            print('Finding route...')
        
        # Find route with automatic transfer detection
        results = optimizer.find_route(
            args.origin_lat, args.origin_lon, 
            args.dest_lat, args.dest_lon, 
            args.weight, args.mode,
            enable_transfer=True,  # Automatically enabled
            max_transfers=args.max_transfers,
            show_all=args.show_all
        )
        
        # Display results
        print_route_results(results, args.criteria, args.verbose, args.mode, args.show_all)
        
        # Save to file if requested
        if args.output:
            # Add .geojson extension if not provided
            if not args.output.endswith('.geojson'):
                args.output += '.geojson'
            
            # Determine what to save based on criteria
            if args.show_all:
                # Save all routes
                save_results = results
            else:
                # Save only optimal route for the specified criteria
                optimal_routes = results.get('optimal_routes', {})
                if args.criteria in optimal_routes:
                    # Find the full route with geometry from the original routes list
                    optimal_route_summary = optimal_routes[args.criteria]
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
                        'optimal_routes': {args.criteria: optimal_route_summary},
                        'criteria_used': args.criteria,
                        'show_all': args.show_all,
                        'mode': args.mode,
                        'enable_transfer': True,  # Automatically enabled
                        'max_transfers': args.max_transfers
                    }
                else:
                    # Fallback to all routes if optimal route not found
                    save_results = results
            
            optimizer.save_results(save_results, args.output)
            print(f"\nResults saved to: {args.output}")
            if not args.show_all:
                print(f"Saved optimal route by criteria: {args.criteria}")
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        print(f"Please check the path: {args.data_folder}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_route_results(results, criteria='fastest', verbose=False, mode='all', show_all=False):
    """Print route search results"""
    print("=" * 60)
    print("ROUTE SEARCH RESULTS")
    print("=" * 60)
    
    origin = results['origin']
    destination = results['destination']
    weight = results['weight_tons']
    
    print(f"Origin: ({origin['lat']:.6f}, {origin['lon']:.6f})")
    print(f"Destination: ({destination['lat']:.6f}, {destination['lon']:.6f})")
    print(f"Weight: {weight} tons")
    print(f"Route type: {mode}")
    print()
    
    routes = results.get('routes', [])
    if not routes:
        print("No routes found!")
        return
    


def print_route_details(route, verbose=False):
    """Print details of a route"""
    print(f"   Time: {route['total_time_minutes']:.1f} minutes")
    print(f"   Distance: {route['total_distance_km']:.1f} km")
    print(f"   CO2 emissions: {route['co2_emissions_grams']:.1f} g")
    
    if verbose:
        if 'origin_port' in route and 'dest_port' in route:
            print(f"   Origin port: {route['origin_port']}")
            print(f"   Destination port: {route['dest_port']}")
            if 'transfer_port' in route:
                print(f"   Transfer port: {route['transfer_port']}")
            if 'ship_time_hours' in route:
                print(f"   Ship time: {route['ship_time_hours']:.1f} hours")
        
        if 'origin_station' in route and 'dest_station' in route:
            print(f"   Origin station: {route['origin_station']}")
            print(f"   Destination station: {route['dest_station']}")
            if 'transfer_station' in route:
                print(f"   Transfer station: {route['transfer_station']}")
            if 'train_time_minutes' in route:
                print(f"   Train time: {route['train_time_minutes']:.1f} minutes")


if __name__ == '__main__':
    main()
