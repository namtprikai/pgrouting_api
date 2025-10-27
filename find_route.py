#!/usr/bin/env python3
"""
Script cải tiến để tìm tuyến đường tối ưu với khả năng tìm trung chuyển
Enhanced script to find optimal routes with transfer capability
"""

import sys
import argparse
from route_optimizer import RouteOptimizer


def main():
    parser = argparse.ArgumentParser(
        description='Tìm tuyến đường tối ưu giữa hai điểm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Mặc định: Tuyến đường nhanh nhất (tự động tìm trung chuyển nếu cần)
  python find_route.py 35.6762 139.6503 34.6937 135.5023
  
  # Tuyến đường ngắn nhất
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --criteria shortest
  
  # Tuyến đường ít CO2 nhất
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --criteria greenest
  
  # Hiển thị tất cả tuyến đường
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --show-all
  
  # Tùy chỉnh số trung chuyển tối đa
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --max-transfers 5
  
  # Lưu kết quả
  python find_route.py 35.6762 139.6503 34.6937 135.5023 --output results
        """
    )
    
    parser.add_argument('origin_lat', type=float, help='Vĩ độ điểm xuất phát')
    parser.add_argument('origin_lon', type=float, help='Kinh độ điểm xuất phát')
    parser.add_argument('dest_lat', type=float, help='Vĩ độ điểm đến')
    parser.add_argument('dest_lon', type=float, help='Kinh độ điểm đến')
    parser.add_argument('--weight', type=float, default=10.0, help='Trọng lượng hàng hóa (tấn)')
    parser.add_argument('--data-folder', 
                       default='sample/content/drive/MyDrive/modalshift',
                       help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--output', help='File để lưu kết quả (GeoJSON)')
    parser.add_argument('--criteria', 
                       choices=['fastest', 'shortest', 'greenest'],
                       default='fastest',
                       help='Tiêu chí tối ưu (fastest: nhanh nhất, shortest: ngắn nhất, greenest: ít CO2 nhất)')
    parser.add_argument('--show-all', action='store_true',
                       help='Hiển thị tất cả tuyến đường thay vì chỉ tuyến tối ưu')
    parser.add_argument('--verbose', '-v', action='store_true', help='Hiển thị thông tin chi tiết')
    parser.add_argument('--mode', 
                       choices=['all', 'truck_only', 'truck_ship', 'truck_train'],
                       default='all',
                       help='Loại đường đi (all: tất cả, truck_only: chỉ xe tải, truck_ship: xe tải+tàu biển, truck_train: xe tải+tàu hỏa)')
    parser.add_argument('--max-transfers', type=int, default=10,
                       help='Số lượng trung chuyển tối đa (mặc định: 10)')
    
    args = parser.parse_args()
    
    # Validate coordinates
    if not (-90 <= args.origin_lat <= 90) or not (-180 <= args.origin_lon <= 180):
        print("Lỗi: Tọa độ điểm xuất phát không hợp lệ")
        sys.exit(1)
    
    if not (-90 <= args.dest_lat <= 90) or not (-180 <= args.dest_lon <= 180):
        print("Lỗi: Tọa độ điểm đến không hợp lệ")
        sys.exit(1)
    
    if args.weight <= 0:
        print("Lỗi: Trọng lượng phải lớn hơn 0")
        sys.exit(1)
    
    try:
        # Initialize optimizer
        if args.verbose:
            print("Đang khởi tạo Route Optimizer...")
        
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
            print("Đang tìm tuyến đường...")
        
        # Find route with automatic transfer detection
        results = optimizer.find_route(
            args.origin_lat, args.origin_lon, 
            args.dest_lat, args.dest_lon, 
            args.weight, args.mode,
            enable_transfer=True,  # Tự động bật
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
                    
                    # Debug info
                    if args.verbose:
                        print(f"Debug: Found {len(all_routes)} routes")
                        print(f"Debug: Looking for route with name='{optimal_route_summary.get('name')}' and mode='{optimal_route_summary.get('mode')}'")
                        if optimal_route_full:
                            print(f"Debug: Found full route with geometry: {optimal_route_full.get('geometry') is not None}")
                        else:
                            print("Debug: No full route found, using summary")
                    
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
                        'enable_transfer': True,  # Tự động bật
                        'max_transfers': args.max_transfers
                    }
                else:
                    # Fallback to all routes if optimal route not found
                    save_results = results
            
            optimizer.save_results(save_results, args.output)
            print(f"\nKết quả đã được lưu vào: {args.output}")
            if not args.show_all:
                print(f"Đã lưu tuyến đường tối ưu theo tiêu chí: {args.criteria}")
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu: {e}")
        print(f"Vui lòng kiểm tra đường dẫn: {args.data_folder}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_route_results(results, criteria='fastest', verbose=False, mode='all', show_all=False):
    """In kết quả tìm tuyến đường"""
    print("=" * 60)
    print("KẾT QUẢ TÌM TUYẾN ĐƯỜNG")
    print("=" * 60)
    
    origin = results['origin']
    destination = results['destination']
    weight = results['weight_tons']
    
    print(f"Điểm xuất phát: ({origin['lat']:.6f}, {origin['lon']:.6f})")
    print(f"Điểm đến: ({destination['lat']:.6f}, {destination['lon']:.6f})")
    print(f"Trọng lượng: {weight} tấn")
    print(f"Loại đường: {mode}")
    print()
    
    routes = results.get('routes', [])
    if not routes:
        print("Không tìm thấy tuyến đường nào!")
        return
    
    # Chế độ mặc định: Chỉ hiển thị tuyến đường tối ưu theo tiêu chí
    # if not show_all:
    #     optimal_routes = results.get('optimal_routes', {})
    #     if criteria in optimal_routes:
    #         route = optimal_routes[criteria]
    #         criteria_names = {
    #             'fastest': 'NHANH NHẤT',
    #             'shortest': 'NGẮN NHẤT', 
    #             'greenest': 'ÍT CO2 NHẤT'
    #         }
    #         print(f"🏆 TUYẾN ĐƯỜNG TỐI ƯU ({criteria_names.get(criteria, criteria.upper())}):")
    #         print("-" * 50)
    #         print(f"📋 {route['name']}")
    #         print_route_details(route, verbose)
            
    #         # Hiển thị so sánh với các tiêu chí khác
    #         print("\n📊 SO SÁNH VỚI CÁC TIÊU CHÍ KHÁC:")
    #         print("-" * 50)
    #         for other_criteria in ['fastest', 'shortest', 'greenest']:
    #             if other_criteria != criteria and other_criteria in optimal_routes:
    #                 other_route = optimal_routes[other_criteria]
    #                 print(f"  {other_criteria.upper()}: {other_route['name']} "
    #                       f"({other_route['total_time_minutes']:.1f} phút, "
    #                       f"{other_route['total_distance_km']:.1f} km, "
    #                       f"{other_route['co2_emissions_grams']:.1f} g CO2)")
    #     else:
    #         print(f"❌ Không tìm thấy tuyến đường tối ưu cho tiêu chí: {criteria}")
    # else:
    #     # Chế độ hiển thị tất cả tuyến đường
    #     print(f"🗺️ TẤT CẢ TUYẾN ĐƯỜNG CÓ THỂ ({len(routes)} tuyến):")
    #     print("-" * 50)
        
    #     for i, route in enumerate(routes, 1):
    #         print(f"{i}. {route['name']}")
    #         print_route_details(route, verbose)
    #         print()
        
    #     # Hiển thị tóm tắt tuyến đường tối ưu
    #     optimal_routes = results.get('optimal_routes', {})
    #     if optimal_routes:
    #         print("🏆 TUYẾN ĐƯỜNG TỐI ƯU:")
    #         print("-" * 50)
            
    #         if 'fastest' in optimal_routes:
    #             route = optimal_routes['fastest']
    #             print(f"⚡ Nhanh nhất: {route['name']} ({route['total_time_minutes']:.1f} phút)")
            
    #         if 'shortest' in optimal_routes:
    #             route = optimal_routes['shortest']
    #             print(f"📏 Ngắn nhất: {route['name']} ({route['total_distance_km']:.1f} km)")
            
    #         if 'greenest' in optimal_routes:
    #             route = optimal_routes['greenest']
    #             print(f"🌱 Ít CO2 nhất: {route['name']} ({route['co2_emissions_grams']:.1f} g)")


def print_route_details(route, verbose=False):
    """In chi tiết một tuyến đường"""
    print(f"   Thời gian: {route['total_time_minutes']:.1f} phút")
    print(f"   Khoảng cách: {route['total_distance_km']:.1f} km")
    print(f"   Phát thải CO2: {route['co2_emissions_grams']:.1f} g")
    
    if verbose:
        if 'origin_port' in route and 'dest_port' in route:
            print(f"   Cảng xuất phát: {route['origin_port']}")
            print(f"   Cảng đến: {route['dest_port']}")
            if 'transfer_port' in route:
                print(f"   Cảng trung chuyển: {route['transfer_port']}")
            if 'ship_time_hours' in route:
                print(f"   Thời gian tàu biển: {route['ship_time_hours']:.1f} giờ")
        
        if 'origin_station' in route and 'dest_station' in route:
            print(f"   Ga xuất phát: {route['origin_station']}")
            print(f"   Ga đến: {route['dest_station']}")
            if 'transfer_station' in route:
                print(f"   Ga trung chuyển: {route['transfer_station']}")
            if 'train_time_minutes' in route:
                print(f"   Thời gian tàu hỏa: {route['train_time_minutes']:.1f} phút")


if __name__ == '__main__':
    main()
