from pathlib import Path
from app.core.config import OUTPUT_FOLDER
import json

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

def save_to_geojson(summary: dict, output_file: str):
    """
    Save merged GeoJSON from summary results to a file.
    """
    merged_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for result in summary.get("results", []):
        geojson = result.get("geojson", {})
        features = geojson.get("features", [])
        merged_geojson["features"].extend(features)

    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_geojson, f, ensure_ascii=False, indent=2)
