import json
import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat, conint
from pydantic_settings import BaseSettings
from shapely import LineString
import geopandas as gpd
import folium

from constant import *

# =========================
# 環境変数
# =========================
class Settings(BaseSettings):
    # PGHOST: str = os.getenv("PGHOST") if os.getenv("PGHOST") else "localhost"
    # PGPORT: int = os.getenv("PGPORT") if os.getenv("PGPORT") else 5432
    # PGDATABASE: str = os.getenv("PGDATABASE") if os.getenv("PGDATABASE") else "o2p"
    # PGUSER: str = os.getenv("PGUSER") if os.getenv("PGUSER") else "o2p"
    # PGPASSWORD: str = os.getenv("PGPASSWORD") if os.getenv("PGPASSWORD") else "o2p"

    PGHOST: str = "localhost"
    PGPORT: int = 5432
    PGDATABASE: str = "o2p"
    PGUSER: str = "o2p"
    PGPASSWORD: str = "o2p"

    PORT: int = 8080

    # デフォルト料金・船速（必要に応じて上書き）
    TOLL_PER_KM: float = 30.0
    SHIP_SPEED_KPH: float = 30.0

    # CO2 係数（g-CO2/トンkm）— 仮の既定値。運用で調整してください
    EF_TRUCK_G_PER_TKM: float = 120.0
    EF_TRAIN_G_PER_TKM: float = 22.0
    EF_SHIP_G_PER_TKM: float = 12.0
    PAYLOAD_TON: float = 10.0

    class Config:
        env_file = ".env"


settings = Settings()
monitoring = False

# =========================
# DB プール
# =========================
POOL: psycopg2.pool.SimpleConnectionPool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=settings.PGHOST,
    port=settings.PGPORT,
    database=settings.PGDATABASE,
    user=settings.PGUSER,
    password=settings.PGPASSWORD,
)

def sql_one(query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            if not row:
                return None
            # psycopg2 の場合は列名を得る
            cols = [desc.name for desc in cur.description]
            return {c: v for c, v in zip(cols, row)}
    finally:
        POOL.putconn(conn)

def sql_all(query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return None
            # psycopg2 の場合は列名を得る
            cols = [desc.name for desc in cur.description]
            result = []
            for row in rows:
                result.append({c: v for c, v in zip(cols, row)})
            return result
    finally:
        POOL.putconn(conn)

# =========================
# 入出力スキーマ
# =========================
class MultimodalBody(BaseModel):
    origin_name: str
    origin_lat: confloat(ge=-90, le=90)
    origin_lon: confloat(ge=-180, le=180)
    dest_name: str
    dest_lat: confloat(ge=-90, le=90)
    dest_lon: confloat(ge=-180, le=180)

    wait_train_min: Optional[conint(ge=0)] = 60
    wait_ship_min: Optional[conint(ge=0)] = 60
    toll_per_km: Optional[confloat(ge=0)] = None
    ship_speed_kph: Optional[confloat(gt=0)] = None

    payload_ton: Optional[confloat(gt=0)] = None
    ef_truck_g_per_tkm: Optional[confloat(gt=0)] = None
    ef_train_g_per_tkm: Optional[confloat(gt=0)] = None
    ef_ship_g_per_tkm: Optional[confloat(gt=0)] = None
    street: str = STREET_TYPE["TRUCK_ONLY"]


# =========================
# DB呼び出しヘルパ
# =========================

def nearest_station(lon: float, lat: float):
    return sql_one("SELECT * FROM nearest_station(%s, %s)", (lon, lat))

def nearest_port(lon: float, lat: float):
    return sql_one("SELECT * FROM nearest_port(%s, %s)", (lon, lat))

def nearest_node_id(lon: float, lat: float):
    return sql_one("SELECT * FROM nearest_node_id(%s, %s)", (lon, lat))

def get_node_component(start_node: int, end_node: int):
    sql = "SELECT * FROM pgr_connectedComponents('SELECT gid AS id, source, target, cost_s as cost, reverse_cost FROM jpn_ways') WHERE node IN (%s, %s)"
    results = sql_all(sql, (start_node, end_node))
    return results

def find_bridge_nodes(start_node: int, end_node: int):
    print(start_node, end_node, '-------')
    row = sql_one(
        """
        SELECT *
        FROM find_bridge_nodes(%s, %s)
        """,
        (start_node, end_node),
    )
    if not row:
        return None
    return row

def create_bridge_edge(start_node: int, end_node: int, cost: float, length_m: float):
    query = "SELECT create_bridge_edge(%s, %s, %s, %s)"
    params = (start_node, end_node, cost, length_m)

    conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            if not row:
                return None
            
            return row
    finally:
        POOL.putconn(conn)

def route_truck_mm(o_lon: float, o_lat: float, d_lon: float, d_lat: float, toll_per_km: float):
    row = sql_one(
        """
        SELECT geom_geojson, distance_km, travel_time_h, motorway_km, toll_estimate_yen,
               entry_ic_name, entry_ic_lon, entry_ic_lat,
               exit_ic_name,  exit_ic_lon,  exit_ic_lat
        FROM route_truck_mm(%s, %s, %s, %s, %s)
        """,
        (o_lon, o_lat, d_lon, d_lat, toll_per_km),
    )
    if not row or not row.get("geom_geojson"):
        return None
    return {
        "geometry": json.loads(row["geom_geojson"]),
        "distance_km": float(row["distance_km"]),
        "travel_time_h": float(row["travel_time_h"]),
        "motorway_km": float(row["motorway_km"]),
        "toll_estimate_yen": float(row["toll_estimate_yen"]) if row["toll_estimate_yen"] is not None else None,
        "entry_ic": (
            {"name": row["entry_ic_name"], "lon": row["entry_ic_lon"], "lat": row["entry_ic_lat"]}
            if row.get("entry_ic_name") is not None else None
        ),
        "exit_ic": (
            {"name": row["exit_ic_name"], "lon": row["exit_ic_lon"], "lat": row["exit_ic_lat"]}
            if row.get("exit_ic_name") is not None else None
        ),
    }

def route_train(s_from: int, s_to: int, wait_min: int):
    row = sql_one("SELECT * FROM route_train(%s,%s,%s)", (s_from, s_to, wait_min))
    if not row or not row.get("geom_geojson"):
        return None
    return {
        "geometry": json.loads(row["geom_geojson"]),
        "distance_km": float(row["distance_km"]),
        "travel_time_h": float(row["travel_time_h"]),
    }

def route_ship_direct(p_from: int, p_to: int, ship_speed_kph: float, wait_min: int):
    row = sql_one(
        "SELECT * FROM route_ship_direct(%s,%s,%s,%s)",
        (p_from, p_to, ship_speed_kph, wait_min),
    )
    if not row or not row.get("geom_geojson"):
        return None
    return {
        "geometry": json.loads(row["geom_geojson"]),
        "distance_km": float(row["distance_km"]),
        "travel_time_h": float(row["travel_time_h"]),
    }

def find_route(payload, O, D, params):
     # 6パターン構築
    # Seg オブジェクト： {"mode": "truck"|"train"|"ship", "geometry": GeoJSON, "distance_km": float, "time_h": float, "meta": 任意}
    def S(mode, seg):
        return {"mode": mode, "geometry": seg["geometry"], "distance_km": seg["distance_km"], "time_h": seg["travel_time_h"], "meta": seg}
        
    patterns: List[Dict[str, Any]] = []
    
    # 最近傍の駅・港
    stO = nearest_station(O["lon"], O["lat"])
    stD = nearest_station(D["lon"], D["lat"])
    ptO = nearest_port(O["lon"], O["lat"])
    ptD = nearest_port(D["lon"], D["lat"])

    # セグメント計算
    # 1) トラックのみ
    if (payload and payload.street == STREET_TYPE['TRUCK_ONLY']):
        seg_truck_OD = route_truck_mm(O["lon"], O["lat"], D["lon"], D["lat"], params['toll_per_km'])
        if seg_truck_OD:
            patterns.append({
                "key": STREET_TYPE['TRUCK_ONLY'],
                "label": "自動車（貨物トラック）のみ",
                "segs": [S("truck", seg_truck_OD)]
            })

    # 2) トラック+列車
    if (payload and payload.street == STREET_TYPE['TRUCK_TRAIN']):
        seg_truck_O_StO = route_truck_mm(O["lon"], O["lat"], stO["slon"], stO["slat"], params['toll_per_km']) if stO else None
        seg_train_StO_StD = route_train(stO["id"], stD["id"], params['wait_train']) if (stO and stD) else None
        seg_truck_StD_D = route_truck_mm(stD["slon"], stD["slat"], D["lon"], D["lat"], params['toll_per_km']) if stD else None
        
        if seg_truck_O_StO and seg_train_StO_StD and seg_truck_StD_D:
            patterns.append({
                "key": STREET_TYPE['TRUCK_TRAIN'],
                "label": "自動車 + 貨物列車",
                "segs": [S("truck", seg_truck_O_StO), S("train", seg_train_StO_StD), S("truck", seg_truck_StD_D)]
            })
    
    # 3) トラック+船
    if (payload and payload.street == STREET_TYPE['TRUCK_SHIP']):
        seg_truck_O_PtO = route_truck_mm(O["lon"], O["lat"], ptO["plon"], ptO["plat"], params['toll_per_km']) if ptO else None
        seg_ship_PtO_PtD = route_ship_direct(ptO["id"], ptD["id"], params['ship_speed'], params['wait_ship']) if (ptO and ptD) else None
        seg_truck_PtD_D = route_truck_mm(ptD["plon"], ptD["plat"], D["lon"], D["lat"], params['toll_per_km']) if ptD else None
        
        if seg_truck_O_PtO and seg_ship_PtO_PtD and seg_truck_PtD_D:
            patterns.append({
                "key": STREET_TYPE['TRUCK_SHIP'],
                "label": "自動車 + 貨物船",
                "segs": [S("truck", seg_truck_O_PtO), S("ship", seg_ship_PtO_PtD), S("truck", seg_truck_PtD_D)]
            })

    # 4) トラック + 列車 + 船（列車→船）
    if (payload and payload.street == STREET_TYPE['TRUCK_TRAIN_SHIP']):
        seg_truck_O_StO = route_truck_mm(O["lon"], O["lat"], stO["slon"], stO["slat"], params['toll_per_km']) if stO else None
        seg_train_StO_StD = route_train(stO["id"], stD["id"], params['wait_train']) if (stO and stD) else None
        seg_ship_PtO_PtD = route_ship_direct(ptO["id"], ptD["id"], params['ship_speed'], params['wait_ship']) if (ptO and ptD) else None
        seg_truck_PtD_D = route_truck_mm(ptD["plon"], ptD["plat"], D["lon"], D["lat"], params['toll_per_km']) if ptD else None
        seg_truck_StD_PtO = route_truck_mm(stD["slon"], stD["slat"], ptO["plon"], ptO["plat"], params['toll_per_km'])
        

        if stD and ptO and seg_truck_O_StO and seg_train_StO_StD and seg_ship_PtO_PtD and seg_truck_PtD_D and seg_truck_StD_PtO:
            patterns.append({
                "key": STREET_TYPE['TRUCK_TRAIN_SHIP'],
                "label": "自動車 + 貨物列車 + 貨物船（列車→船）",
                "segs": [S("truck", seg_truck_O_StO), S("train", seg_train_StO_StD), S("truck", seg_truck_StD_PtO),
                        S("ship", seg_ship_PtO_PtD), S("truck", seg_truck_PtD_D)]
            })

    # 5) トラック + 船 + 列車（船→列車）
    if (payload and payload.street == STREET_TYPE['TRUCK_SHIP_TRAIN']):
        seg_truck_O_PtO = route_truck_mm(O["lon"], O["lat"], ptO["plon"], ptO["plat"], params['toll_per_km']) if ptO else None
        seg_ship_PtO_PtD = route_ship_direct(ptO["id"], ptD["id"], params['ship_speed'], params['wait_ship']) if (ptO and ptD) else None
        seg_train_StO_StD = route_train(stO["id"], stD["id"], params['wait_train']) if (stO and stD) else None
        seg_truck_StD_D = route_truck_mm(stD["slon"], stD["slat"], D["lon"], D["lat"], params['toll_per_km']) if stD else None
        seg_truck_PtD_StO = route_truck_mm(ptD["plon"], ptD["plat"], stO["slon"], stO["slat"], params['toll_per_km'])

        if ptD and stO and seg_truck_O_PtO and seg_ship_PtO_PtD and seg_train_StO_StD and seg_truck_StD_D and seg_truck_PtD_StO:
            patterns.append({
                "key": STREET_TYPE['TRUCK_SHIP_TRAIN'],
                "label": "自動車 + 貨物船 + 貨物列車（船→列車）",
                "segs": [S("truck", seg_truck_O_PtO), S("ship", seg_ship_PtO_PtD), S("truck", seg_truck_PtD_StO),
                        S("train", seg_train_StO_StD), S("truck", seg_truck_StD_D)]
            })

    # 6) トラック + 列車 + 船 + 列車（列車→船→列車）
    if (payload and payload.street == STREET_TYPE['TRUCK_TRAIN_SHIP_TRAIN']):
        if stO and stD and ptO and ptD:
            seg_truck_O_StO2 = seg_truck_O_StO or route_truck_mm(O["lon"], O["lat"], stO["slon"], stO["slat"], params['toll_per_km'])
            seg_train_StO_StD2 = seg_train_StO_StD or route_train(stO["id"], stD["id"], params['wait_train'])
            seg_truck_StD_PtO2 = route_truck_mm(stD["slon"], stD["slat"], ptO["plon"], ptO["plat"], params['toll_per_km'])
            seg_ship_PtO_PtD2  = seg_ship_PtO_PtD or route_ship_direct(ptO["id"], ptD["id"], params['ship_speed'], params['wait_ship'])
            seg_truck_PtD_StO2 = route_truck_mm(ptD["plon"], ptD["plat"], stO["slon"], stO["slat"], params['toll_per_km'])
            seg_train_StO_StD3 = route_train(stO["id"], stD["id"], params['wait_train'])
            seg_truck_StD_D2   = seg_truck_StD_D or route_truck_mm(stD["slon"], stD["slat"], D["lon"], D["lat"], params['toll_per_km'])

            if all([seg_truck_O_StO2, seg_train_StO_StD2, seg_truck_StD_PtO2, seg_ship_PtO_PtD2, seg_truck_PtD_StO2, seg_train_StO_StD3, seg_truck_StD_D2]):
                patterns.append({
                    "key": STREET_TYPE['TRUCK_TRAIN_SHIP_TRAIN'],
                    "label": "自動車 + 貨物列車 + 貨物船 + 貨物列車",
                    "segs": [S("truck", seg_truck_O_StO2), S("train", seg_train_StO_StD2), S("truck", seg_truck_StD_PtO2),
                            S("ship", seg_ship_PtO_PtD2), S("truck", seg_truck_PtD_StO2), S("train", seg_train_StO_StD3),
                            S("truck", seg_truck_StD_D2)]
                })

    # パターンが一つも組めない場合
    if not patterns:
        raise HTTPException(status_code=404, detail="No feasible pattern could be constructed (check station/port coverage).")

    # 各パターンの合計値と CO2
    features: List[Dict[str, Any]] = []

    for pat in patterns:
        dist_km = sum(s["distance_km"] for s in pat["segs"])
        time_h  = sum(s["time_h"] for s in pat["segs"])
        co2_g_total = 0.0
        for s in pat["segs"]:
            ef = params['ef_truck'] if s["mode"] == "truck" else (params['ef_train'] if s["mode"] == "train" else params['ef_ship'])
            co2_g_total += co2_g(s["distance_km"], ef, params['ton'])

        # トラックセグメントのIC（入口/出口）
        truck_segs = [s for s in pat["segs"] if s["mode"] == "truck" and s.get("meta")]
        entry_ic = next((s["meta"].get("entry_ic") for s in truck_segs if s["meta"].get("entry_ic")), None)
        exit_ic  = next((s["meta"].get("exit_ic") for s in reversed(truck_segs) if s["meta"].get("exit_ic")), None)

        new_feature = {
            "type": "Feature",
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [s["geometry"] for s in pat["segs"]],
            },
            "properties": {
                "pattern_key": pat["key"],
                "pattern_label": pat["label"],
                "origin_name": O["name"],
                "dest_name": D["name"],
                "total_distance_km": round(dist_km, 1),
                "total_time_h": round(time_h, 2),
                "total_co2_kg": round(co2_g_total / 1000.0, 1),
                "truck_entry_ic": entry_ic,
                "truck_exit_ic": exit_ic,
                "segments": [
                    {"mode": s["mode"], "distance_km": round(s["distance_km"], 1), "time_h": round(s["time_h"], 2)}
                    for s in pat["segs"]
                ],
            },
        }

        features.append(new_feature)

        save_geojson([new_feature], path=f'{pat["key"]}.geojson')
        return features

# =========================
# CO2ヘルパ
# =========================
def co2_g(total_km: float, ef_g_per_tkm: float, ton: float) -> float:
    return total_km * ef_g_per_tkm * ton

def save_geojson(features: List[Dict[str, Any]], path: str) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def make_map(features: List[Dict[str, Any]], path: str) -> None:
    # 中心は最初のFeatureの始点に
    if not features:
        return
    first = features[0]
    coords = first["geometry"]["geometries"][0]["coordinates"]
    center = coords[0][1], coords[0][0]  # (lat, lon)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, feat in enumerate(features):
        coords = feat["geometry"]["geometries"][0]["coordinates"]
        latlons = [(lat, lon) for lon, lat in coords]
        travel_mode = feat['properties']['segments'][0]['mode']
        total_distance_km = feat['properties']['segments'][0]['distance_km']
        total_duration_h = feat['properties']['segments'][0]['time_h']
        folium.PolyLine(latlons, weight=5, opacity=0.9, color=palette[i % len(palette)],
                        tooltip=f"{travel_mode}: "
                                f"{total_distance_km}km / "
                                f"{total_duration_h}h").add_to(m)
        # start/end marker
        folium.Marker(latlons[0], tooltip="Start").add_to(m)
        folium.Marker(latlons[-1], tooltip="End",
                      icon=folium.Icon(color="red", icon="flag")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(path)

# =========================
# FastAPI
# =========================
app = FastAPI(title="Multimodal Truck/Train/Ship Router (FastAPI)")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/multimodal/route")
def multimodal_route(payload: MultimodalBody):
     # 基本パラメータ
    wait_train = int(payload.wait_train_min if payload.wait_train_min is not None else 60)
    wait_ship  = int(payload.wait_ship_min  if payload.wait_ship_min  is not None else 60)
    toll_per_km = float(payload.toll_per_km if payload.toll_per_km is not None else settings.TOLL_PER_KM)
    ship_speed  = float(payload.ship_speed_kph if payload.ship_speed_kph is not None else settings.SHIP_SPEED_KPH)

    ton      = float(payload.payload_ton if payload.payload_ton is not None else settings.PAYLOAD_TON)
    ef_truck = float(payload.ef_truck_g_per_tkm if payload.ef_truck_g_per_tkm is not None else settings.EF_TRUCK_G_PER_TKM)
    ef_train = float(payload.ef_train_g_per_tkm if payload.ef_train_g_per_tkm is not None else settings.EF_TRAIN_G_PER_TKM)
    ef_ship  = float(payload.ef_ship_g_per_tkm  if payload.ef_ship_g_per_tkm  is not None else settings.EF_SHIP_G_PER_TKM)

    street = payload.street if payload.street is not None else "TRUCK_ONLY"

    params = {
        "wait_train": wait_train,
        "wait_ship": wait_ship,
        "toll_per_km": toll_per_km,
        "ship_speed": ship_speed,
        "ton": ton,
        "ef_truck": ef_truck,
        "ef_train": ef_train,
        "ef_ship": ef_ship,
        "street": street
    }


    O = {"name": payload.origin_name, "lon": float(payload.origin_lon), "lat": float(payload.origin_lat)}
    D = {"name": payload.dest_name,   "lon": float(payload.dest_lon),   "lat": float(payload.dest_lat)}

    start_node = nearest_node_id(O["lon"], O["lat"])
    end_node = nearest_node_id(D["lon"], D["lat"])

    node_component = get_node_component(start_node['nearest_node_id'], end_node['nearest_node_id'])

    start_node_component = node_component[0]['component']
    end_node_component = node_component[1]['component']

    print(start_node['nearest_node_id'], end_node['nearest_node_id'], '====')
    print(start_node_component, end_node_component, '////////')
    if start_node_component == end_node_component:
        features = find_route(payload, O, D, params)
    else:
        # bridge_node = find_bridge_nodes(start_node['nearest_node_id'], end_node['nearest_node_id'])
        # cost = bridge_node['physical_bridge_distance_meters'] + wait_ship
        # result = create_bridge_edge(start_node['nearest_node_id'], end_node['nearest_node_id'], cost, bridge_node['physical_bridge_distance_meters'])
        # features = find_route(payload, O, D, params)
        # print(features, 'features')
        raise HTTPException(status_code=404, detail="No feasible pattern could be constructed (check station/port coverage).")

    def best_by(field: str) -> str:
        sorted_feats = sorted(features, key=lambda f: f["properties"][field])
        return sorted_feats[0]["properties"]["pattern_key"]

    summary = {
        "best_time": best_by("total_time_h"),
        "best_distance": best_by("total_distance_km"),
        "best_co2": best_by("total_co2_kg"),
        "wait_train_min": wait_train,
        "wait_ship_min": wait_ship,
        "payload_ton": ton,
        "ef_g_per_tkm": {"truck": ef_truck, "train": ef_train, "ship": ef_ship},
    }

    return {"type": "FeatureCollection", "features": features, "summary": summary}


# 直接起動用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=settings.PORT, reload=True)
