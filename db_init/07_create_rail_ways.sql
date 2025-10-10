-- Create Railway table
DROP TABLE IF EXISTS rail_ways;
CREATE TABLE rail_ways(
  gid serial PRIMARY KEY,
  source bigint,
  target bigint,
  geom geometry(LineString,4326),
  length_m double precision,
  oneway boolean DEFAULT true,
  cost_s double precision
);
CREATE INDEX rail_ways_gix ON rail_ways USING GIST (geom);