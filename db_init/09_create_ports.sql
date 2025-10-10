DROP TABLE IF EXISTS ports;
CREATE TABLE ports(
  id serial PRIMARY KEY,
  name text,
  geom geometry(Point,4326)
);

CREATE INDEX ports_gix ON ports USING GIST (geom);
