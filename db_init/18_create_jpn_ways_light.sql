-- Create jpn_ways table
DROP TABLE IF EXISTS jpn_ways;
	CREATE TABLE jpn_ways as
	SELECT w.gid, w.source, w.target,
            w.the_geom as geom, c.tag_value as highway,
            w.name, w.oneway, w.cost as cost_s,
            w.reverse_cost, w.length_m, w.blocked,
            w.maxspeed_forward as maxspeed_forward,
            w.maxspeed_backward as maxspeed_backward,
            CASE
                WHEN c.tag_value IN ('motorway', 'motorway_link') THEN (w.cost) * 0.6
                WHEN c.tag_value IN ('trunk', 'trunk_link', 'primary', 'primary_link') THEN (w.cost)
                else (w.cost) * 1.4
            END AS cost_s
    FROM ways w inner join "configuration" c on w.tag_id = c.tag_id

CREATE INDEX IF NOT EXISTS jpn_ways_geom_idx ON jpn_ways USING GIST (geom);
CREATE INDEX IF NOT EXISTS jpn_ways_gid_idx ON jpn_ways(gid);
