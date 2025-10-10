CREATE OR REPLACE FUNCTION public.route_truck_mm(o_lon double precision, o_lat double precision, d_lon double precision, d_lat double precision, toll_per_km numeric DEFAULT 30.0)
 RETURNS TABLE(geom_geojson text, distance_km numeric, travel_time_h numeric, motorway_km numeric, toll_estimate_yen numeric, entry_ic_name text, entry_ic_lon double precision, entry_ic_lat double precision, exit_ic_name text, exit_ic_lon double precision, exit_ic_lat double precision)
 LANGUAGE plpgsql
 STABLE
AS $function$
DECLARE
  src bigint := nearest_node_id(o_lon, o_lat);
  dst bigint := nearest_node_id(d_lon, d_lat);
BEGIN
  RETURN QUERY
  WITH r AS (
    SELECT * FROM pgr_bdAstar(
      $q$
      SELECT gid AS id, source, target,cost_s AS cost,reverse_cost,   
             ST_X(ST_StartPoint(geom)) AS x1, ST_Y(ST_StartPoint(geom)) AS y1,
             ST_X(ST_EndPoint(geom))   AS x2, ST_Y(ST_EndPoint(geom))   AS y2,
             length_m, highway, geom
      FROM jpn_ways
      WHERE NOT blocked
      $q$, src, dst, directed := true
    )
  ),
  seg AS (
    SELECT e.*, r.seq, r.cost AS seg_cost
    FROM r
    JOIN (
      SELECT gid AS id, source, target,
             cost_s,
             CASE 
                WHEN oneway = 'YES' THEN 1e15
                WHEN oneway = 'NO' THEN cost_s
                WHEN oneway = 'UNKNOWN' THEN cost_s
                WHEN oneway = 'REVERSIBLE' THEN cost_s
                ELSE cost_s
             END AS reverse_cost,
             length_m, highway, geom, maxspeed_forward
      FROM jpn_ways
      WHERE NOT blocked
    ) e ON e.id = r.edge
    ORDER BY r.seq
  ),
  agg AS (
    SELECT
      ST_AsGeoJSON(ST_LineMerge(ST_Union(seg.geom))) AS gj,
      SUM(seg.length_m)/1000.0 AS dist_km,
      SUM(
	      seg.length_m / (
	        COALESCE(
	          (NULLIF(seg.maxspeed_forward, null)::double precision),
	          CASE
	            WHEN seg.highway IN ('motorway','motorway_link') THEN 120.0
	            WHEN seg.highway IN ('trunk','trunk_link') THEN 100.0
	            WHEN seg.highway IN ('primary','primary_link') THEN 80.0
	            WHEN seg.highway IN ('secondary','secondary_link') THEN 60.0
	            WHEN seg.highway IN ('tertiary','tertiary_link') THEN 40.0
	            ELSE 30.0
	          END
	        ) * 1000.0
	      )
	  ) AS tt_h,
      SUM(CASE WHEN seg.highway IN ('motorway','motorway_link')
               THEN seg.length_m ELSE 0 END)/1000.0 AS mw_km
    FROM seg
  ),
  mw AS (
    SELECT * FROM seg
    WHERE highway IN ('motorway','motorway_link')
    ORDER BY seq
  ),
  entry_edge AS (SELECT * FROM mw ORDER BY seq ASC LIMIT 1),
  exit_edge  AS (SELECT * FROM mw ORDER BY seq DESC LIMIT 1),
  entry_pt AS (
    SELECT ST_StartPoint(geom) AS pt FROM entry_edge
  ),
  exit_pt AS (
    SELECT ST_EndPoint(geom) AS pt FROM exit_edge
  ),
  entry_ic AS (
    SELECT name, lon, lat
    FROM motorway_ic
    ORDER BY ST_SetSRID(ST_MakePoint(lon,lat),4326) <->
             (SELECT pt FROM entry_pt)
    LIMIT 1
  ),
  exit_ic AS (
    SELECT name, lon, lat
    FROM motorway_ic
    ORDER BY ST_SetSRID(ST_MakePoint(lon,lat),4326) <->
             (SELECT pt FROM exit_pt)
    LIMIT 1
  )
  SELECT
    agg.gj,
    COALESCE(agg.dist_km, 0)::numeric,
    COALESCE(agg.tt_h, 0)::numeric,
    COALESCE(agg.mw_km, 0)::numeric,
    COALESCE(ROUND(agg.mw_km * toll_per_km), 0)::numeric AS toll_est,
    (SELECT name FROM entry_ic), (SELECT lon FROM entry_ic), (SELECT lat FROM entry_ic),
    (SELECT name FROM exit_ic),  (SELECT lon FROM exit_ic),  (SELECT lat FROM exit_ic)
  FROM agg;
END;
$function$
;
