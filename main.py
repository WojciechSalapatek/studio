import numpy as np
import update_function
import matplotlib.pyplot as plt

m = 0.13
d = 0.05


class CellularAutomata:
    def __init__(self, dimension):
        self.dimension = dimension
        self.grid = np.zeros(dimension)

    def initialize_states(self, n_oil_mass):
        center_x = round(self.dimension[0]/2)
        center_y = round(self.dimension[1]/2)
        self.grid[center_x-5:center_x+5, center_y-5:center_y+5] = n_oil_mass

    def run(self, duration):
        for t in range(duration):
            self.grid = update_function.update_grid(self.grid, np.zeros(self.dimension), m, d)
            print(t)


if __name__ == "__main__":
    ca = CellularAutomata((100, 100))
    ca.initialize_states(5000)
    plt.matshow(ca.grid)
    plt.show()
    ca.run(150)
    plt.matshow(ca.grid)
    plt.show()
    ca.run(150)
    plt.matshow(ca.grid)
    plt.show()