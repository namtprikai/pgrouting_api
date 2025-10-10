CREATE OR REPLACE FUNCTION nearest_node_id(
  in_lon double precision,
  in_lat double precision
)
RETURNS bigint AS $$
  SELECT nid
  FROM jpn_nodes
  ORDER BY geom <-> ST_SetSRID(ST_Point(in_lon, in_lat), 4326)
  LIMIT 1;
$$ LANGUAGE sql STABLE;