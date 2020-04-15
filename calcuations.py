def calculate_leak_position(latitude_range, longitude_range, grid_dimension):
    horizon_lat = 28.755372
    horizon_long = -88.387681

    i = (horizon_lat-latitude_range[0])*grid_dimension[0]/(latitude_range[1]-latitude_range[0])
    j = (horizon_long-longitude_range[0])*grid_dimension[1]/(longitude_range[1]-longitude_range[0])
    return round(i), round(j)

