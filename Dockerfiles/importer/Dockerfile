FROM postgis/postgis:16-3.4

ENV DEBIAN_FRONTEND=noninteractive

# Tạm disable PGDG để tránh apt kéo postgresql-17*, cài tool OSM, rồi bật lại.
RUN if [ -f /etc/apt/sources.list.d/pgdg.list ]; then \
    mv /etc/apt/sources.list.d/pgdg.list /etc/apt/sources.list.d/pgdg.list.disabled ; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    osm2pgsql \
    osmctools \
    osmium-tool \
    osm2pgrouting \
    bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && if [ -f /etc/apt/sources.list.d/pgdg.list.disabled ]; then \
    mv /etc/apt/sources.list.d/pgdg.list.disabled /etc/apt/sources.list.d/pgdg.list ; \
    fi

# Dữ liệu và cấu hình
COPY importer_data/japan_highways.osm.bz2 /japan_highways.osm.bz2
COPY importer_data/chubu-highways.osm.bz2 /chubu-highways.osm.bz2
COPY importer_data/mapconfig.xml /mapconfig.xml

ENTRYPOINT []

# Giữ container sống để bạn vào chạy lệnh thủ công  
CMD ["tail", "-f", "/dev/null"]
