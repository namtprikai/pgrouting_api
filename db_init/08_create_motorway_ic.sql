DROP TABLE IF EXISTS motorway_ic;
CREATE TABLE motorway_ic (
  id   serial PRIMARY KEY,
  name text,
  lon  double precision,
  lat  double precision,
  geom geometry(Point,4326)
);