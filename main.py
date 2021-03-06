import numpy as np
import update_function
import matplotlib; matplotlib.use("TkAgg")
from animation import make_animation, get_land_map
from calcuations import *
import time
import netCDF4
from scipy.ndimage import interpolation
import shutil
import os


class CellularAutomata:
    def __init__(self, dimension, out_dir, leak_rate_per_step, leak_location, land_mask, clean_out=False):
        self.dimension = dimension
        self.grid = np.zeros(dimension)
        self.leak_rate_per_step = leak_rate_per_step
        self.leak_location_i = leak_location[0]
        self.leak_location_j = leak_location[1]
        self.land_mask = land_mask
        self.out_dir = out_dir
        if clean_out:
            print(f"Cleaning output directory {out_dir}")
            shutil.rmtree(self.out_dir, onerror=None)
            os.makedirs(self.out_dir)

    def run(self, duration_hours, step_minutes, current):
        n_steps = round(duration_hours*60//step_minutes)
        meters_height, meters_width = calculate_size(LATITUDE_RANGE, LONGITUDE_RANGE)
        print("Simulation parameters:")
        print(f"    Grid size {self.dimension[0]}x{self.dimension[1]}")
        print(f"    Simulation arena size {round(meters_height)}x{round(meters_width)} meters")
        print(f"    Cell size {round(meters_height/self.dimension[0])}x{round(meters_width/self.dimension[1])} meters")
        print(f"    Duration: {duration_hours} hours")
        print(f"    Timestep: {step_minutes} minutes")
        print(f"    Leak: {self.leak_rate_per_step} kilograms per step at [{self.leak_location_i}, {self.leak_location_j}]")
        print(f"    Simulation will take {n_steps} steps")
        time.sleep(2)
        print("Starting simulation")
        for t in range(1, n_steps+1):
            ts = time.time()
            velocity_grid = current.get_east_velocity_grid(t)
            north_velocity_grid = current.get_north_velocity_grid(t)
            self.grid[self.leak_location_i, self.leak_location_j] += self.leak_rate_per_step
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D,
                                                    np.array(velocity_grid), np.array(north_velocity_grid), self.land_mask)
            self.save_grid_to_file(t)
            tm = (time.time() - ts)
            print(f"Step {t} performed, realization time {int(tm*1000)} ms, estimated {(n_steps-t)*tm} seconds")

    def save_grid_to_file(self, t):
        data = self.grid.astype(int)
        np.savetxt(self.out_dir + 'frame{}.txt'.format(t), data, fmt='%i')


# data for currents prep (lat 18: 31 long -101:-82
# top-left corner of a map corresponds to: 31 lat, -101 long
class Currents:
    def __init__(self, latitude_range, longitude_range):
        fp = 'currents.nc'
        self.data = netCDF4.Dataset(fp)
        # index corresponding to values lat 18: 31 -> indexes 72:125
        # long -101:-82 -> indexes 16:93
        # time: 5 days -> indexes 0:5

        print(f"Searching for latitude {latitude_range} and longitude {longitude_range}")
        start_lat_index = np.where(np.array(self.data['latitude']) == latitude_range[0])[0][0]
        end_lat_index = np.where(np.array(self.data['latitude']) == latitude_range[1])[0][0]+1
        start_lon_index = np.where(np.array(self.data['longitude']) == longitude_range[0])[0][0]
        end_lon_index = np.where(np.array(self.data['longitude']) == longitude_range[1])[0][0]+1

        print(f"Latitude index ranges {start_lat_index}-{end_lat_index}")
        print(f"Longitude index ranges {start_lon_index}-{end_lon_index}")

        self.lat = self.data['latitude'][start_lat_index:end_lat_index]
        self.lon = self.data['longitude'][start_lon_index:end_lon_index]
        tmp = self.data['u'][0:DAYS, :, 72:125, 16:93]

        # flip is made to ensure that velocityEast[0][0] refers to the current int the top left corner of the map
        self.velocityEast = np.flip(self.data['u'][0:DAYS, :, start_lat_index:end_lat_index, start_lon_index:end_lon_index], 2)  # (time, depth, lat, long)
        self.velocityNorth = np.flip(self.data['v'][0:DAYS, :, start_lat_index:end_lat_index, start_lon_index:end_lon_index], 2)  # (time, depth, lat, long)

        self.masked_el = self.data['u'][0, 0, 18, -101]

        print("Scaling currents")
        self.eastVelocities = self.filter_and_scale_and_normalise(self.velocityEast)
        self.northVelocities = self.filter_and_scale_and_normalise(self.velocityNorth)

        self.time = self.data['time']

    def filter_and_scale_and_normalise(self, arr_to_preprocess, output_arr=None):         # Filter Nan values and scale to grid size
        if output_arr is None:
            output_arr = []
        for i in range(DAYS):
            curr = arr_to_preprocess[i, 0, :, :]
            curr[curr == self.masked_el] = 0.0000001
            curr[np.isnan(curr)] = 0.0000001
            curr = interpolation.zoom(curr, (N_HEIGHT / curr.shape[0], N_WIDTH / curr.shape[1]))
            curr[np.isnan(curr)] = 0.0000001
            curr = curr/2
            curr = curr/np.max(np.abs(arr_to_preprocess[0:DAYS, 0, :, :]))
            output_arr.append(curr)
        return output_arr

    def get_east_velocity_grid(self, time):
        day = int(np.floor((TIMESTEP*time)/(60*24)))
        if day >= DAYS:
            return np.zeros((N_HEIGHT, N_WIDTH)) + 0.00001
        else:
            return -self.eastVelocities[day]

    def get_north_velocity_grid(self, time):
        day = int(np.floor((TIMESTEP * time) / (60 * 24)))
        if day >= DAYS:
            return np.zeros((N_HEIGHT, N_WIDTH)) + 0.00001
        else:
            return self.northVelocities[day]


####################################################################################################################
#                                       Parameters to set
####################################################################################################################
DAYS = 10                 # how many days simulate
LEAK_PER_DAY = 7659770    # of kilograms per day

# Position and size parameters
HORIZON_LAT = 28.755372
HORIZON_LONG = -88.387681

LATITUDE_RANGE = (round_degrees(28.0), round_degrees(30.25))       # must be rounded to 0.25
LONGITUDE_RANGE = (round_degrees(-89.75), round_degrees(-86.75))    # must be rounded to 0.25

GRIDS_OUT_DIR = "out/grids/"
FRAME_IMAGES_OUT_DIR = "out/frames/"
ANIMATION_FILE = "out/animations/sim_animation.mp4"
BACKGROUND_IMG = "map_for_simulation.png"


####################################################################################################################
#                            Other parameters which should not be changed
####################################################################################################################

# size of cell [m]
CELL_SIZE = 800

# Transmission parameters
M = 0.088
D = 0.18

# Duration Parameters
DURATION = 24*DAYS         # hours
TIMESTEP = round(300/60)   # minutes

# Leakage parameters
LEAK_PER_STEP = TIMESTEP*LEAK_PER_DAY/(24*60)   # of kilograms per step

# Calculate grid size so that cell size has appropriate length
N_HEIGHT, N_WIDTH = calculate_grid_dimension_with_cell_size(LATITUDE_RANGE, LONGITUDE_RANGE, CELL_SIZE)


if __name__ == "__main__":
    # current = Currents(LATITUDE_RANGE, LONGITUDE_RANGE)
    # leak_pos = calculate_leak_position(LATITUDE_RANGE, LONGITUDE_RANGE, (N_HEIGHT, N_WIDTH))
    land_map = get_land_map(BACKGROUND_IMG, LATITUDE_RANGE, LONGITUDE_RANGE, (N_WIDTH, N_HEIGHT))
    # ca = CellularAutomata((N_HEIGHT, N_WIDTH), GRIDS_OUT_DIR, LEAK_PER_STEP, leak_pos, land_map, clean_out=True)
    # ca.run(DURATION, TIMESTEP, current)
    make_animation(GRIDS_OUT_DIR, FRAME_IMAGES_OUT_DIR, BACKGROUND_IMG, LATITUDE_RANGE, LONGITUDE_RANGE, (N_WIDTH, N_HEIGHT), land_map, ANIMATION_FILE)
