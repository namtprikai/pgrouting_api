import asyncio
import aiohttp
import time
import json

async def test_async_api():
    url = "http://localhost:8080/api/search-route/"
    
    payload = {
        "mode": ["truck_only", "truck_train", "truck_ship"],
        "origin_name": "京浜トラックターミナル",
        "origin_lat": 35.58757039336874,
        "origin_lon": 139.75480254099043,
        "destination_name": "東大阪トラックターミナル ",
        "destination_lat": 34.69742817495373,
        "destination_lon": 135.60644728466065,
        "find_station_radius_km": 100,
        "find_port_radius_km": 100,
        "departure_hour": 8,
        "weight_tons": 1000
    }

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload) as response:
                end_time = time.time()
                
                print(f"✅ Status Code: {response.status}")
                print(f"⏱️  Response Time: {end_time - start_time:.2f} seconds")
                
                if response.status == 200:
                    data = await response.json()
                    
                    # In thông tin summary thôi, không in hết 50000 dòng
                    print(f"📊 Số lượng modes: {len(data.get('results', []))}")
                    
                    for i, result in enumerate(data.get('results', [])):
                        print(f"\n--- Mode {i+1}: {result.get('mode')} ---")
                        print(f"   Thời gian: {result.get('total_time_minutes', 0)} phút")
                        print(f"   Khoảng cách: {result.get('total_distance_km', 0)} km")
                        print(f"   CO2: {result.get('total_co2_emissions_grams', 0)} grams")
                        print(f"   Thông báo: {result.get('message', '')}")
                    
                    print(f"\n🎯 Tổng số kết quả: {len(data.get('results', []))}")
                    print("✅ API hoạt động thành công!")
                    
                    # Lưu file đầy đủ nếu muốn xem
                    with open("api_response.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print("💾 Đã lưu response đầy đủ vào file: api_response.json")
                    
                else:
                    print(f"❌ Lỗi: {response.status}")
                    error_text = await response.text()
                    print(f"Chi tiết lỗi: {error_text}")
                    
        except Exception as e:
            print(f"🚨 Lỗi kết nối: {e}")

async def test_concurrent_requests():
    """Test xử lý concurrent requests"""
    print("\n" + "="*50)
    print("🧪 TEST CONCURRENT REQUESTS")
    print("="*50)
    
    url = "http://localhost:8080/api/search-route/"
    payload = {
        "mode": ["truck_only"],
        "origin_name": "京浜トラックターミナル",
        "origin_lat": 35.58757039336874,
        "origin_lon": 139.75480254099043,
        "destination_name": "東大阪トラックターミナル ",
        "destination_lat": 34.69742817495373,
        "destination_lon": 135.60644728466065,
        "find_station_radius_km": 100,
        "find_port_radius_km": 100,
        "departure_hour": 8,
        "weight_tons": 1000
    }

    async def make_request(request_id):
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                end_time = time.time()
                print(f"Request {request_id}: {response.status} - {end_time - start_time:.2f}s")
                return response.status

    # Gửi 3 requests cùng lúc
    tasks = [make_request(i) for i in range(1, 10)]
    results = await asyncio.gather(*tasks)
    
    print(f"\n🎯 Kết quả concurrent: {sum(1 for r in results if r == 200)}/{len(tasks)} thành công")

async def main():
    print("🚀 Bắt đầu test API Async")
    print("="*50)
    
    # Test single request
    await test_async_api()
    
    # Test concurrent requests  
    await test_concurrent_requests()

if __name__ == "__main__":
    asyncio.run(main())