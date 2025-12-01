from pydantic import BaseModel, confloat

class SeachRouteBody(BaseModel):
    mode: list
    origin_name: str
    origin_lat: confloat(ge=-90, le=90)
    origin_lon: confloat(ge=-180, le=180)
    destination_name: str
    destination_lat: confloat(ge=-90, le=90)
    destination_lon: confloat(ge=-180, le=180)
    max_transfers: int = 1
    show_all: bool = True
    find_station_radius_km: int = 100
    find_port_radius_km: int = 100
    departure_hour: int
    weight_tons: float