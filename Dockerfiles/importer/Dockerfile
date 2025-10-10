FROM postgis/postgis:16-3.4
RUN apt-get update && apt-get install -y \
    osm2pgsql \
    osmctools \
    osmium-tool \
    osm2pgrouting

# COPY importer_data/import_osm.sh /import_osm.sh
COPY importer_data/japan_highways.osm.bz2 /japan_highways.osm.bz2
COPY importer_data/mapconfig.xml /mapconfig.xml

# RUN chmod +x /import_osm.sh
# CMD ["/import_osm.sh"]
CMD ["tail", "-f", "/dev/null"]