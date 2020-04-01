import numpy as np
import update_function
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import os
import re
import netCDF4

M = 0.098
D = 0.18
DURATION = 5*24
N_HEIGHT = 221
N_WIDTH = 181
TIMESTEP = 60
INITIAL_OIL_MASS = 5000



class CellularAutomata:
    def __init__(self, dimension):
        self.dimension = dimension
        self.grid = np.zeros(dimension)
        self.out_dir = "out/"

    def initialize_states(self, n_oil_mass):
        center_x = round(self.dimension[0]/2)
        center_y = round(self.dimension[1]/2)
        self.grid[center_x-5:center_x+5, center_y-5:center_y+5] = n_oil_mass

    def run(self, duration_hours, step_minutes, current):
        n_steps = round(duration_hours*60//step_minutes)
        print("Simulation will take {} steps".format(n_steps))
        for t in range(n_steps):
            velocity_grid = current.get_east_velocity_grid(t)
            north_velocity_grid = current.get_north_velocity_grid(t)
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D, velocity_grid,
                                                    np.amax(velocity_grid)), north_velocity_grid, np.amax(north_velocity_grid)
            self.save_grid_to_file(t)
            print(t)
        self.animation()

    def save_grid_to_file(self, t):
        data = self.grid
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(self.out_dir + 'frame{}.png'.format(t))

    def animation(self):
        key_pat = re.compile(r"^frame(.*).png$")

        def key(item):
            m = key_pat.match(item)
            return int(m.group(1))

        fig = plt.figure()
        files = []
        ims = []
        for file in os.listdir(self.out_dir):
            files.append(file)
        files.sort(key=key)
        for image in files:
            im = plt.imshow(Image.open(self.out_dir + image), animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=100)
        plt.show()


class Currents:
    def __init__(self):
        fp = 'currents.nc'
        self.data = netCDF4.Dataset(fp)
        self.velocityEast = self.data['u']  # (time, depth, lat, long)
        self.velocityNorth = self.data['v'] # (time, depth, lat, long)
        self.time = self.data['time']
        self.lat = self.data['latitude']
        self.lon = self.data['longitude']

    def get_east_velocity_grid(self, time):
        if time <= 10 :
            v = -self.velocityEast[0, 0, :, :]
        if 10 < time <= 20:
            v = -self.velocityEast[1, 0, :, :]
        if 20 < time <= 30:
            v = -self.velocityEast[2, 0, :, :]
        if 30 < time <= 40:
            v = -self.velocityEast[3, 0, :, :]
        if 40 < time <= 50:
            v = -self.velocityEast[5, 0, :, :]
        if 50 < time:
            v = -self.velocityEast[6, 0, :, :]
        if time > 60:
            v = np.zeros(221, 181)
        return v

    def get_north_velocity_grid(self, time):
        if time <= 10 :
            v = -self.velocityNorth[0, 0, :, :]
        if 10 < time <= 20:
            v = -self.velocityNorth[1, 0, :, :]
        if 20 < time <= 30:
            v = -self.velocityNorth[2, 0, :, :]
        if 30 < time <= 40:
            v = -self.velocityNorth[3, 0, :, :]
        if 40 < time <= 50:
            v = -self.velocityNorth[5, 0, :, :]
        if 50 < time:
            v = -self.velocityNorth[6, 0, :, :]
        if time > 60:
            v = np.zeros(221, 181)
        return v


if __name__ == "__main__":
    current = Currents()
    ca = CellularAutomata((N_HEIGHT, N_WIDTH))
    ca.initialize_states(INITIAL_OIL_MASS)
    ca.run(DURATION, TIMESTEP, current)


