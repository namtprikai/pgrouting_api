FROM postgis/postgis:16-3.4
RUN apt-get update && apt-get install -y \
    osm2pgsql \
    osmctools \
    osmium-tool \
    osm2pgrouting

COPY importer_data/japan_highways.osm.bz2 /japan_highways.osm.bz2
COPY importer_data/mapconfig.xml /mapconfig.xml

COPY importer_data/import_osm.sh /import_osm.sh
RUN sed -i 's/\r$//' /import_osm.sh && chmod +x /import_osm.sh
CMD ["/bin/bash", "-c", "/import_osm.sh && tail -f /dev/null"]
