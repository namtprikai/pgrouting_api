-- Create jpn_ways table
DROP TABLE IF EXISTS jpn_ways;
	CREATE TABLE jpn_ways as	
	SELECT w.gid, w.source, w.target, 
            w.the_geom as geom, c.tag_value as highway, 
            w.name, w.oneway, w.cost as cost_s, 
            w.reverse_cost, w.length_m, w.blocked, 
            w.maxspeed_forward as maxspeed_forward, 
            w.maxspeed_backward as maxspeed_backward 
    FROM ways w inner join "configuration" c on w.tag_id = c.tag_id