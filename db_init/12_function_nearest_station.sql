CREATE OR REPLACE FUNCTION nearest_station(lon double precision, lat double precision)
RETURNS TABLE (id bigint, name text, slon double precision, slat double precision) AS $$
  SELECT s.id, s.name, ST_X(s.geom), ST_Y(s.geom)
  FROM freight_stations s
  ORDER BY s.geom <-> ST_SetSRID(ST_Point(lon,lat),4326)
  LIMIT 1;
$$ LANGUAGE sql STABLE;