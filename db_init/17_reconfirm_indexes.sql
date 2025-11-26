-- Xóa các index KHÔNG được sử dụng (0 lần scan)
DROP INDEX IF EXISTS pointsofinterest_pointsofinterest_osm_id_key1;
DROP INDEX IF EXISTS pointsofinterest_pointsofinterest_pkey;
DROP INDEX IF EXISTS pointsofinterest_pointsofinterest_the_geom_idx;
DROP INDEX IF EXISTS pointsofinterest_pointsofinterest_the_geom_idx1;
DROP INDEX IF EXISTS ports_ports_pkey;
DROP INDEX IF EXISTS ports_ports_gix;
DROP INDEX IF EXISTS ports_idx_ports_geom;
DROP INDEX IF EXISTS rail_ways_rail_ways_pkey;
DROP INDEX IF EXISTS rail_ways_rail_ways_gix;
DROP INDEX IF EXISTS configuration_configuration_pkey;
DROP INDEX IF EXISTS configuration_configuration_tag_id_key;
DROP INDEX IF EXISTS configuration_configuration_tag_id_key1;
DROP INDEX IF EXISTS ways_vertices_pgr_ways_vertices_pgr_osm_id_key;
DROP INDEX IF EXISTS ways_vertices_pgr_ways_vertices_pgr_osm_id_key1;
DROP INDEX IF EXISTS ways_vertices_pgr_ways_vertices_pgr_pkey;
DROP INDEX IF EXISTS ways_vertices_pgr_ways_vertices_pgr_the_geom_idx;
DROP INDEX IF EXISTS ways_vertices_pgr_ways_vertices_pgr_the_geom_idx1;
DROP INDEX IF EXISTS ways_ways_the_geom_idx;
DROP INDEX IF EXISTS ways_idx_ways_source;
DROP INDEX IF EXISTS ways_idx_ways_target;
DROP INDEX IF EXISTS ways_idx_ways_source_osm;
DROP INDEX IF EXISTS ways_idx_ways_target_osm;
DROP INDEX IF EXISTS jpn_ways_routing_jpn_ways_routing_id_idx;
DROP INDEX IF EXISTS jpn_ways_routing_jpn_ways_routing_src_idx;
DROP INDEX IF EXISTS node_components_idx_node_components_node_id;
DROP INDEX IF EXISTS jpn_ways_routing_jpn_ways_routing_tgt_idx;
DROP INDEX IF EXISTS node_components_idx_jpn_components_node;
DROP INDEX IF EXISTS freight_stations_freight_stations_pkey;
DROP INDEX IF EXISTS jpn_nodes_idx_jpn_nodes_nid;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_source;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_target;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_highway;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_blocked;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_source_target;
DROP INDEX IF EXISTS jpn_ways_idx_jpn_ways_not_blocked;
DROP INDEX IF EXISTS motorway_ic_motorway_ic_pkey;
DROP INDEX IF EXISTS pointsofinterest_pointsofinterest_osm_id_key;



-- Giải phóng không gian các bảng lớn sau khi xóa index thừa
VACUUM FULL jpn_ways;
VACUUM FULL ways_vertices_pgr;
VACUUM FULL jpn_nodes;
VACUUM FULL motorway_ic;
VACUUM FULL freight_stations;
VACUUM FULL ways;
VACUUM FULL pointsofinterest;
VACUUM FULL ports;
VACUUM FULL rail_ways;
VACUUM FULL configuration;
VACUUM FULL node_components;
VACUUM FULL jpn_ways_routing;