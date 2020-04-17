from math import sin, cos, sqrt, atan2, radians


def calculate_leak_position(latitude_range, longitude_range, grid_dimension):
    horizon_lat = 28.755372
    horizon_long = -88.387681

    i = (horizon_lat-latitude_range[0])*grid_dimension[0]/(latitude_range[1]-latitude_range[0])
    j = (horizon_long-longitude_range[0])*grid_dimension[1]/(longitude_range[1]-longitude_range[0])
    return round(i), round(j)


def calculate_size(lat_range, lon_range):
    height = calculate_distance((lat_range[0], lon_range[0]), (lat_range[1], lon_range[0]))
    width = calculate_distance((lat_range[0], lon_range[0]), (lat_range[0], lon_range[1]))
    return height, width


def calculate_distance(p1, p2):
    R = 6373.0

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance*1000


def calculate_grid_dimension_with_cell_size(latitude_range, longitude_range, meters_cell_size):
    meters_height, meters_width = calculate_size(latitude_range, longitude_range)
    return round(meters_height/meters_cell_size), round(meters_width/meters_cell_size)


def round_degrees(degree):
    return round(degree * 4) / 4
