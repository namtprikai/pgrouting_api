CREATE OR REPLACE FUNCTION nearest_port(lon double precision, lat double precision)
RETURNS TABLE (id bigint, name text, plon double precision, plat double precision) AS $$
  SELECT p.id, p.name, ST_X(p.geom), ST_Y(p.geom)
  FROM ports p
  ORDER BY p.geom <-> ST_SetSRID(ST_Point(lon,lat),4326)
  LIMIT 1;
$$ LANGUAGE sql STABLE;