import numpy as np
import update_function
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import os
import re
import netCDF4
from scipy.ndimage import interpolation

M = 0.058
D = 0.18
N_HEIGHT = 53*3  # lat
N_WIDTH = 77*3 # long
INITIAL_OIL_MASS = 50000

DAYS = 1
DURATION = 24*DAYS
TIMESTEP = round(300/60)


class CellularAutomata:
    def __init__(self, dimension):
        self.dimension = dimension
        self.grid = np.zeros(dimension)
        self.out_dir = "out/"

    def initialize_states(self, n_oil_mass):
        self.grid[20, 150] = n_oil_mass

    def run(self, duration_hours, step_minutes, current, make_animation=False):
        n_steps = round(duration_hours*60//step_minutes)
        print("Simulation parameters:")
        print(" Grid size {}x{}".format(N_HEIGHT, N_WIDTH))
        print(" Duration: {} days".format(DAYS))
        print(" Timestep: {} minutes".format(TIMESTEP))
        print(" Simulation will take {} steps".format(n_steps))
        for t in range(1, n_steps+1):
            velocity_grid = current.get_east_velocity_grid(t)
            north_velocity_grid = current.get_north_velocity_grid(t)
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D, np.array(velocity_grid),
                                                    np.amax(velocity_grid), np.array(north_velocity_grid), np.amax(north_velocity_grid))
            self.save_grid_to_file(t)
            print(t)
        if make_animation:
            self.animation()

    def save_grid_to_file(self, t):
        data = self.grid
        rescaled = (255.0 / data.max() * (data - data.min()))
        black = np.zeros((N_HEIGHT, N_WIDTH, 4))
        black[:, :, 3] = rescaled
        im = Image.fromarray(black.astype(np.uint8))
        im.save(self.out_dir + 'frame{}.png'.format(t))

    def animation(self):
        key_pat = re.compile(r"^frame(.*).png$")

        def key(item):
            m = key_pat.match(item)
            return int(m.group(1))

        img = Image.open("map_for_simulation.png")
        img = img.resize((N_WIDTH, N_HEIGHT))
        fig = plt.figure()
        files = []
        ims = []
        for file in os.listdir(self.out_dir):
            files.append(file)
        files.sort(key=key)
        for image in files:
            frame = Image.open(self.out_dir + image)
            comp = Image.alpha_composite(img, frame.convert('RGBA'))
            im = plt.imshow(comp, animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=7, blit=True,
                                        repeat_delay=100)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)  # Uncomment to save .mp4 animation
        ani.save('anim9.mp4', writer=writer)
        plt.show()

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
        tmp = self.data['u'][0:5, :, 72:125, 16:93]

        # flip is made to ensure that velocityEast[0][0] refers to the current int the top left corner of the map
        self.velocityEast = np.flip(self.data['u'][0:5, :, 72:125, 16:93], 2)  # (time, depth, lat, long)
        self.velocityNorth = np.flip(self.data['v'][0:5, :, 72:125, 16:93], 2)  # (time, depth, lat, long)

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
            return np.zeros((N_HEIGHT, N_WIDTH))
        else:
            return -self.eastVelocities[day]

    def get_north_velocity_grid(self, time):
        day = int(np.floor((TIMESTEP * time) / (60 * 24)))
        if day >= DAYS:
            return np.zeros((N_HEIGHT, N_WIDTH))
        else:
            return -self.northVelocities[day]


if __name__ == "__main__":
    current = Currents()
    ca = CellularAutomata((N_HEIGHT, N_WIDTH))
    ca.initialize_states(INITIAL_OIL_MASS)
    ca.run(DURATION, TIMESTEP, current, make_animation=False)

