-- Create table node_components
DROP TABLE IF EXISTS node_components;
CREATE TABLE node_components AS
SELECT * FROM pgr_connectedComponents(
    'SELECT gid as id, source, target, cost_s as cost, reverse_cost FROM jpn_ways'
);