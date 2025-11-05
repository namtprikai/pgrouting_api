#!/bin/bash
set -e

echo "Start import data into database"
export PGPASSWORD="pgrouting"

# Waiting for database start
until pg_isready -h database_pg -p 5432 -U postgres; do
  echo "Waiting for database start..."
  sleep 3
done

echo "Database is ready!"

# Create database if not exists
psql -h database_pg -p 5432 -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'pgrouting';" | grep -q 1 || \
psql -h database_pg -p 5432 -U postgres -c "CREATE DATABASE pgrouting;"

# Create extension PostGIS and pgRouting
echo "Creating PostGIS extension..."
psql -h database_pg -p 5432 -U postgres -d pgrouting -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -h database_pg -p 5432 -U postgres -d pgrouting -c "CREATE EXTENSION IF NOT EXISTS pgrouting;"

echo "PostGIS extension created!"

# Import data using osm2pgrouting
osm2pgrouting \
  -f /japan_highways.osm.bz2 \
  -d pgrouting \
  -U postgres \
  -W 'pgrouting' \
  -h database_pg \
  --port 5432 \
  --conf /mapconfig.xml \
  --clean \
  --no-index

echo "Import data success!"
