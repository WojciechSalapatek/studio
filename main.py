import numpy as np
import update_function
import matplotlib; matplotlib.use("TkAgg")
from animation import make_animation
from calcuations import *
import time
import netCDF4
from scipy.ndimage import interpolation
import shutil
import os

M = 0.098
D = 0.18
N_HEIGHT = 53*3  # lat
N_WIDTH = 77*3  # long

DAYS = 1
DURATION = 24*DAYS         # hours
TIMESTEP = round(300/60)   # minutes

LEAK_PER_DAY = 1500000                          # of kilograms per day
LEAK_PER_STEP = TIMESTEP*LEAK_PER_DAY/(24*60)   # of kilograms per step


class CellularAutomata:
    def __init__(self, dimension, out_dir, leak_rate_per_step, leak_location, clean_out=False):
        self.dimension = dimension
        self.grid = np.zeros(dimension)
        self.leak_rate_per_step = leak_rate_per_step
        self.leak_location_i = leak_location[0]
        self.leak_location_j = leak_location[1]
        self.out_dir = out_dir
        if clean_out:
            print(f"Cleaning output directory {out_dir}")
            shutil.rmtree(self.out_dir, onerror=None)
            os.makedirs(self.out_dir)

    def run(self, duration_hours, step_minutes, current):
        n_steps = round(duration_hours*60//step_minutes)

        print("Simulation parameters:")
        print(f" Grid size {self.dimension[0]}x{self.dimension[1]}")
        print(f" Duration: {duration_hours} hours")
        print(f" Timestep: {step_minutes} minutes")
        print(f" Leak: {self.leak_rate_per_step} kilograms per step at [{self.leak_location_i}, {self.leak_location_j}]")
        print(f" Simulation will take {n_steps} steps")

        for t in range(1, n_steps+1):
            ts = time.time()
            velocity_grid = current.get_east_velocity_grid(t)
            north_velocity_grid = current.get_north_velocity_grid(t)
            self.grid[self.leak_location_i, self.leak_location_j] += self.leak_rate_per_step
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D, np.array(velocity_grid),
                                                    np.amax(velocity_grid), np.array(north_velocity_grid), np.amax(north_velocity_grid))
            self.save_grid_to_file(t)
            tm = (time.time() - ts)

            print(f"Step {t} performed, realization time {int(tm*1000)} ms, estimated {(n_steps-t)*tm} seconds")

    def save_grid_to_file(self, t):
        np.savetxt(self.out_dir + 'frame{}.txt'.format(t), self.grid)


# data for currents prep (lat 18: 31 long -101:-82
# top-left corner of a map corresponds to: 31 lat, -101 long
class Currents:
    def __init__(self):
        fp = 'currents.nc'
        self.data = netCDF4.Dataset(fp)
        # index corresponding to values lat 18: 31 -> indexes 72:125
        # long -101:-82 -> indexes 16:93
        # time: 5 days -> indexes 0:5

        self.lat = self.data['latitude'][72:125]
        self.lon = self.data['longitude'][16:93]
        tmp = self.data['u'][0:DAYS, :, 72:125, 16:93]

        # flip is made to ensure that velocityEast[0][0] refers to the current int the top left corner of the map
        self.velocityEast = np.flip(self.data['u'][0:DAYS, :, 72:125, 16:93], 2)  # (time, depth, lat, long)
        self.velocityNorth = np.flip(self.data['v'][0:DAYS, :, 72:125, 16:93], 2)  # (time, depth, lat, long)

        print("scaling currents")
        self.eastVelocities = self.filter_and_scale(self.velocityEast)
        self.northVelocities = self.filter_and_scale(self.velocityNorth)

        self.time = self.data['time']

    def filter_and_scale(self, arr_to_preprocess, output_arr=None):         # Filter Nan values and scale to grid size
        if output_arr is None:
            output_arr = []
        for i in range(DAYS):
            curr = arr_to_preprocess[i, 0, :, :]
            curr[curr == curr[0, 0]] = 0
            curr = interpolation.zoom(curr, (N_HEIGHT / curr.shape[0], N_WIDTH / curr.shape[1]))
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
            return -self.northVelocities[day]


if __name__ == "__main__":
    grids_out_dir = "out/grids/"
    frame_images_out_dir = "out/frames/"
    animation_file = "out/animations/anim12.mp4"
    current = Currents()
    leak_pos = calculate_leak_position((18, 31), (-101, -82), (N_HEIGHT, N_WIDTH))
    ca = CellularAutomata((N_HEIGHT, N_WIDTH), grids_out_dir, LEAK_PER_STEP, leak_pos, clean_out=True)
    ca.run(DURATION, TIMESTEP, current)
    # make_animation(grids_out_dir, frame_images_out_dir, "map_for_simulation.png", (N_WIDTH, N_HEIGHT), animation_file)
