
# Links-Veda Find Routing

## Description:
API for calculating the optimal route in Japan

## Tech Stack
**Server:** Python 3.11

**Database:** Postgres 17

**Extensions:** Pgrouting, postgis

**Libs:** Pgrouting, postgis

**Tools:** osm2pgsql, osmctools, osmium-tool, osm2pgrouting

## Init data:
1. Download and move file **japan_highways.osm.bz2** to folder **importer_data**

2. Download and move file **mapconfig.xml** to folder **importer_data**

3. Download and move files **L013101物流拠点出発地到着地リスト.csv, 貨物船_位置情報（国土数値情報）.csv, 貨物船_時刻表（2024年版海上定期便ガイド）.csv, 貨物駅_位置情報.xlsx, 貨物駅_時刻表.xlsx** file to **data_file** folder

## Installation

### Make venv and install requirements
```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

### Build docker container
1. Build docker container

```bash
docker compose up --build
```
2. Import data from osm.bz2 file to database
```bash
docker exec -it pgrouting-importer-1 bash
```
```bash
osm2pgrouting -f /japan_highways.osm.bz2 -c /mapconfig.xml -d pgrouting -U postgres -W pgrouting -h db
```

3. Setting database, add table, function
```bash
docker exec -it database_pg bash -c "/scripts/init_db.sh"
```

## Import data for freigh_stations table:

1. Download and move **貨物駅_位置情報.xlsx** file to **importer_data** folder

2. Run file import_freight_stations.py in the **app** folder

```bash
python app/import_freight_stations.py
```

3. Check data in the table **freight_stations**

## Import data for ports table:

1. Download and move file **貨物船_位置情報（国土数値情報）.csv** to folder **importer_data**

2. Run file **import_ports.py** in the **app** folder

```bash
python app/import_ports.py
```

3. Check data in the table **ports**

## Running Tests

### Input
```bash
{
    "origin_name":<Name origin>,
    "origin_lat":<origin latitude>,
    "origin_lon":<origin longitude>, 
    "dest_name":<Name destination>,
    "dest_lat":<dest latitude>,
    "dest_lon":<dest longitude>,
    "mode": <mode_type>
}
```
### Exam:
```bash
{
    "origin_name":"葛西トラックターミナル",
    "origin_lat":35.64657,
    "origin_lon":139.8624, 
    "dest_name":"宮崎ターミナル",
    "dest_lat":31.913,
    "dest_lon":131.456,
    "mode": "TRUCK_TRAIN"
}
```
### Output:
```bash
{
    "time": 33.91639269291826,
    "distance": 100600022.98356158,
    "co2": 2012000988.2931478,
    "mode": "truck_train",
    "origin_port": null,
    "dest_port": null,
    "origin_station": "東京貨物ターミナル",
    "dest_station": "佐土原オフレールステーション",
    "transfer_port": null,
    "transfer_station": null
}
```
The exported GeoJSON file is located in the **/output** folder.