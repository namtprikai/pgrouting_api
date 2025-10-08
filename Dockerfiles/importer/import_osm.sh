#!/bin/bash
set -e

echo "Start import data into database"
export PGPASSWORD="pgrouting"
# Waiting for database start
until pg_isready -h db -p 5432 -U pgrouting; do
  echo "Waiting for database start..."
  sleep 3
done

echo "Database is ready!"

# Create extension PostGIS
echo "Creating PostGIS extension..."
psql -h db -p 5432 -U pgrouting -d pgrouting -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -h db -p 5432 -U pgrouting -d pgrouting -c "CREATE EXTENSION IF NOT EXISTS pgrouting;"

echo "PostGIS extension created!"

# import data using command osm2pgrouting
osm2pgrouting \
  -f japan_highways.osm.bz2 \
  -d pgrouting \
  -U pgrouting \
  -W 'pgrouting' \
  -h db \
  --port 5432 \
  --conf mapconfig.xml \
  --clean

echo "Import data success!"
