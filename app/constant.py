STREET_TYPE = {
    "TRUCK_ONLY": "TRUCK_ONLY",
    "TRUCK_SHIP": "TRUCK_SHIP",
    "TRUCK_SHIP_TRAIN": "TRUCK_SHIP_TRAIN",
    "TRUCK_TRAIN": "TRUCK_TRAIN",
    "TRUCK_TRAIN_SHIP": "TRUCK_TRAIN_SHIP",
    "TRUCK_TRAIN_SHIP_TRAIN": "TRUCK_TRAIN_SHIP_TRAIN",
}

"""
TRUCK_TRAIN_SHIP_TRAIN:
Origin --- <truck_route_1> --- stO_1 --- <train_route_1> --- stD_1 --- <truck_route_2> --- origin_port --- <ship_route> --- dest_port --- <truck_route_3> --- stO_2 --- <train_route_2> --- stD_2 --- <truck_route_4> --- Destination


Base:

Steps: 
1. Get the truck path from the Source to S_Station_1

2. Get the train path from S_Station_1 to D_Station_1

3. Get the truck path from D_Station_1 to S_Port

4. Get the ship path form S_Port to D_Port

5. Get the truck path from D_Port to S_Station_2

6. Get the train path from S_Station_2 to D_Station_2

7. Get the truck path from D_Station_2 to Destination

"""

"""
Run command:

python find_route.py 35.3616 139.0225 31.513 130.527 --data-folder "C:\rikai\source_code\pgrouting_api\importer_data" --mode "truck_train_ship_train" --output "test_2" --show-all

"""
