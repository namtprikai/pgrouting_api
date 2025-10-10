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
1. Copy file **japan_highways.osm.bz2** to folder **importer_data**

2. Copy file **mapconfig.xml** to folder **importer_data**

## Installation
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