import json
import sys
from pathlib import Path

def normalize_geojson_with_points(input_path, output_path=None):
    """Chuẩn hóa GeoJSON và thêm các Point Feature từ phần 'points' vào features"""

    # --- Load file ---
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"File không tồn tại: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Lấy features cũ ---
    features = data.get("features", [])
    props = data.get("properties", {})
    optimal_routes = props.get("optimal_routes", {})
    fastest = optimal_routes.get("fastest", {})

    # --- Nếu có phần 'points', thêm từng point vào features ---
    points = fastest.get("points", [])
    for p in points:
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                print(f"Bỏ qua point có tọa độ không hợp lệ: {p}")
                continue

            point_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "point_type": p.get("type", "unknown"),
                    "name": p.get("name", ""),
                }
            }
            features.append(point_feature)
        except Exception as e:
            print(f"Lỗi khi thêm point: {p} ({e})")
            continue

    # --- Gộp lại thành FeatureCollection chuẩn ---
    normalized = {
        "type": "FeatureCollection",
        "features": features,
        "properties": props
    }

    # --- Ghi ra file ---
    output_path = output_path or (input_path.stem + "_with_points.geojson")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"Đã chuẩn hóa và thêm {len(points)} điểm vào: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚙️  Cách dùng:")
        print("   python normalize_geojson_with_points.py <input_file> [output_file]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        normalize_geojson_with_points(input_file, output_file)
