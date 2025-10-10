CREATE OR REPLACE FUNCTION public.route_train(s_from bigint, s_to bigint, wait_min numeric DEFAULT 60)
 RETURNS TABLE(geom_geojson text, distance_km double precision, travel_time_h double precision)
 LANGUAGE plpgsql
 STABLE
AS $function$
DECLARE
  src bigint := s_from;
  dst bigint := s_to;
BEGIN
  RETURN QUERY
  WITH r AS (
    SELECT * FROM pgr_bdAstar(
      	$q$
       	SELECT gid AS id, source, target,
             cost_s AS cost,
             CASE
		        WHEN oneway IN ('yes', '1', 'true') THEN 1e15
		     	ELSE cost_s
		     END AS reverse_cost,
             ST_X(ST_StartPoint(geom)) AS x1, ST_Y(ST_StartPoint(geom)) AS y1,
             ST_X(ST_EndPoint(geom))   AS x2, ST_Y(ST_EndPoint(geom))   AS y2,
             length_m, geom
      	FROM rail_ways
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
		        WHEN oneway IN ('yes', '1', 'true') THEN 1e15
		     	ELSE cost_s
		     END AS reverse_cost,
             length_m, geom
      FROM jpn_ways
      WHERE NOT blocked
    ) e ON e.id = r.edge
    ORDER BY r.seq
  ),
  agg AS (
    SELECT
      ST_AsGeoJSON(ST_LineMerge(ST_Union(seg.geom))) AS gj,
      SUM(seg.length_m)/1000.0 AS dist_km,
      (SUM(seg.seg_cost) + (wait_min*60))/3600.0 AS tt_h
  FROM seg)
  SELECT gj, dist_km, tt_h FROM agg;
END;
$function$
;
