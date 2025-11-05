#!/bin/bash
set -e

DB_HOST=${DB_HOST:-db}
DB_PORT=${DB_PORT:-5432}
DB_USER=${DB_USER:-postgres}
DB_NAME=${DB_NAME:-pgrouting}
DB_PASSWORD=${DB_PASSWORD:-pgrouting}

echo "=== Waiting for database to start ==="
until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
  echo "Database not ready yet... retrying in 3s"
  sleep 3
done

echo "=== Database is ready! Starting initialization ==="

for sql_file in /scripts/*.sql; do
  echo "Running: $sql_file"
  PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$sql_file"
done

echo "=== All SQL scripts executed successfully! ==="