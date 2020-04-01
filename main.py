import numpy as np
import update_function
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import os
import re
import netCDF4

M = 0.058
D = 0.18
DURATION = 24*5
N_HEIGHT = 181 #long
N_WIDTH = 221 #lat
TIMESTEP = round(300/60)
INITIAL_OIL_MASS = 50000


class CellularAutomata:
    def __init__(self, dimension):
        self.dimension = dimension
        self.grid = np.zeros(dimension)
        self.out_dir = "out/"

    def initialize_states(self, n_oil_mass):
        center_x = round(self.dimension[0]/2)
        center_y = round(self.dimension[1]/2)
        self.grid[center_x, center_y] = n_oil_mass

    def run(self, duration_hours, step_minutes, current):
        n_steps = round(duration_hours*60//step_minutes)
        print("Simulation will take {} steps".format(n_steps))
        for t in range(n_steps):
            velocity_grid = current.get_east_velocity_grid(t)
            north_velocity_grid = current.get_north_velocity_grid(t)
            a = north_velocity_grid[16, 213]
            velocity_grid[velocity_grid == a] = 0
            north_velocity_grid[north_velocity_grid == a] = 0
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D, np.array(velocity_grid),
                                                    np.amax(velocity_grid), np.array(north_velocity_grid), np.amax(north_velocity_grid))
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
        ani = animation.ArtistAnimation(fig, ims, interval=7, blit=True,
                                        repeat_delay=100)
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)  # Uncomment to save .mp4 animation
        # ani.save('anim4.mp4', writer=writer)
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
        day = round((TIMESTEP*time)/(60*24))
        if day <= 1 :
            v = -self.velocityEast[0, 0, :, :]
        elif 1 < day <= 2:
            v = -self.velocityEast[1, 0, :, :]
        elif 2 < day <= 3:
            v = -self.velocityEast[2, 0, :, :]
        elif 3 < day <= 4:
            v = -self.velocityEast[3, 0, :, :]
        elif 4 < day <= 5:
            v = -self.velocityEast[5, 0, :, :]
        elif 5 < day:
            v = -self.velocityEast[6, 0, :, :]
        else :
            v = np.zeros((181, 221))
        return v

    def get_north_velocity_grid(self, time):
        day = round((TIMESTEP * time) / (60 * 24))
        if day <= 1 :
            v = -self.velocityNorth[0, 0, :, :]
        elif 1 < day <= 2:
            v = -self.velocityNorth[1, 0, :, :]
        elif 2 < day <= 3:
            v = -self.velocityNorth[2, 0, :, :]
        elif 3 < day <= 4:
            v = -self.velocityNorth[3, 0, :, :]
        elif 4 < day <= 5:
            v = -self.velocityNorth[5, 0, :, :]
        elif 5 < day:
            v = -self.velocityNorth[6, 0, :, :]
        return v


if __name__ == "__main__":
    current = Currents()
    ca = CellularAutomata((N_HEIGHT, N_WIDTH))
    ca.initialize_states(INITIAL_OIL_MASS)
    ca.run(DURATION, TIMESTEP, current)

