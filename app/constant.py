STREET_TYPE = {
    "TRUCK_ONLY": "TRUCK_ONLY",
    "TRUCK_SHIP": "TRUCK_SHIP",
    "TRUCK_SHIP_TRAIN": "TRUCK_SHIP_TRAIN",
    "TRUCK_TRAIN": "TRUCK_TRAIN",
    "TRUCK_TRAIN_SHIP": "TRUCK_TRAIN_SHIP",
    "TRUCK_TRAIN_SHIP_TRAIN": "TRUCK_TRAIN_SHIP_TRAIN",
}

FOLDER_DATA = "importer_data"

VEHICLES = {"truck": "Truck", "ship": "Ship", "train": "Train"}

SHIP_SPEED_DEFAULT = 50

MESSAGES = {
    "no_time_data": "時刻表データがないため、簡易試算となります",
    "ship_route_not_found": "起点・終点の近隣の港を結ぶ経路が見つかりません",,
    "truck_route_not_found": "{origin} と {destination} を結ぶトラックルートが見つかりませんでした。",
    "train_route_not_found": "{origin} と {destination} を結ぶ鉄道ルートが見つかりませんでした。",
}
