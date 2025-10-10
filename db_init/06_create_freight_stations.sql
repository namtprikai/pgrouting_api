DROP TABLE IF EXISTS freight_stations;
CREATE TABLE freight_stations(
    id serial PRIMARY KEY,
    name text,
    geom geometry(Point,4326)
);

CREATE INDEX freight_stations_gix ON freight_stations USING GIST (geom);