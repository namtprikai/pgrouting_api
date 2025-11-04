FROM postgis/postgis:16-3.4
RUN apt-get update && apt-get install -y \
    osm2pgsql \
    osmctools \
    osmium-tool \
    osm2pgrouting

# COPY importer_data/import_osm.sh /import_osm.sh
COPY importer_data/japan_highways.osm.bz2 /japan_highways.osm.bz2
COPY importer_data/mapconfig.xml /mapconfig.xml
COPY data_file/L013101物流拠点出発地到着地リスト.csv data_file/L013101物流拠点出発地到着地リスト.csv
COPY data_file/貨物船_位置情報（国土数値情報）.csv data_file/貨物船_位置情報（国土数値情報）.csv
COPY data_file/貨物船_時刻表（2024年版海上定期便ガイド）.csv data_file/貨物船_時刻表（2024年版海上定期便ガイド）.csv
COPY data_file/貨物駅_位置情報.xlsx data_file/貨物駅_位置情報.xlsx
COPY data_file/貨物駅_時刻表.xlsx data_file/貨物駅_時刻表.xlsx

# RUN chmod +x /import_osm.sh
# CMD ["/import_osm.sh"]
CMD ["tail", "-f", "/dev/null"]