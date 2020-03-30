import numpy as np
import update_function
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
import os
import re


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

    def run(self, duration_hours, step_minutes):
        n_steps = round(duration_hours*60//step_minutes)
        print("Simulation will take {} steps".format(n_steps))
        for t in range(n_steps):
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), M, D)
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


if __name__ == "__main__":
    ca = CellularAutomata((N_HEIGHT, N_WIDTH))
    ca.initialize_states(INITIAL_OIL_MASS)
    ca.run(DURATION, TIMESTEP)

