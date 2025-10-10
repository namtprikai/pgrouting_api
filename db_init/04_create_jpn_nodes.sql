DROP TABLE IF EXISTS jpn_nodes;
	CREATE TABLE jpn_nodes AS
	SELECT id AS nid,
		   the_geom AS geom
	FROM ways_vertices_pgr;

CREATE INDEX IF NOT EXISTS jpn_nodes_gix ON jpn_nodes USING GIST (geom);