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
COPY importer_data/japan-251109.osm.bz2 /japan-251109.osm.bz2
COPY importer_data/japan-251109.osm.pbf /japan-251109.osm.pbf
COPY importer_data/mapconfig.xml /mapconfig.xml
COPY data_file/L013101物流拠点出発地到着地リスト.csv data_file/L013101物流拠点出発地到着地リスト.csv
COPY data_file/貨物船_位置情報（国土数値情報）.csv data_file/貨物船_位置情報（国土数値情報）.csv
COPY data_file/貨物船_時刻表（2024年版海上定期便ガイド）.csv data_file/貨物船_時刻表（2024年版海上定期便ガイド）.csv
COPY data_file/貨物駅_位置情報.xlsx data_file/貨物駅_位置情報.xlsx
COPY data_file/貨物駅_時刻表.xlsx data_file/貨物駅_時刻表.xlsx

ENTRYPOINT []

# Giữ container sống để bạn vào chạy lệnh thủ công  
CMD ["tail", "-f", "/dev/null"]
