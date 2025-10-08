FROM postgis/postgis:16-3.4
RUN apt-get update && apt-get install -y \
    osm2pgsql \
    osmctools \
    osmium-tool \
    osm2pgrouting

COPY Dockerfiles/importer/import_osm.sh /import_osm.sh
RUN chmod +x /import_osm.sh
CMD ["/import_osm.sh"]