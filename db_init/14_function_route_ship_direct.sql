CREATE OR REPLACE FUNCTION public.route_ship_direct(port_from bigint, port_to bigint, ship_speed_kph numeric DEFAULT 30, wait_min numeric DEFAULT 60)
 RETURNS TABLE(geom_geojson text, distance_km double precision, travel_time_h double precision)
 LANGUAGE plpgsql
 STABLE
AS $function$
DECLARE
  p1 geometry;
  p2 geometry;
  dist_m double precision;
BEGIN
  SELECT geom INTO p1 FROM ports WHERE id = port_from;
  SELECT geom INTO p2 FROM ports WHERE id = port_to;

  dist_m := ST_DistanceSphere(p1, p2); -- 大円距離(m)

  RETURN QUERY
  SELECT
    ST_AsGeoJSON(ST_MakeLine(p1, p2)) AS gj,
    dist_m/1000.0 AS dist_km,
    ( (dist_m/1000.0) / ship_speed_kph + (wait_min/60.0) ) AS tt_h;
END;
$function$
;
