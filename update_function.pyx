#cython.wraparound=False
#cython.boundscheck=False
cimport numpy as np

def update_grid(np.ndarray[np.float64_t, ndim=2] input_grid, np.ndarray[np.float64_t, ndim=2] output_grid,
                double m, double d):
    cdef:
        size_t i
        size_t j

    for i in range(1, input_grid.shape[0]-1):
        for j in range(1, input_grid.shape[1]-1):
            output_grid[i][j] = \
                input_grid[i][j] \
                + m*((input_grid[i-1][j] - input_grid[i][j]) + (input_grid[i+1][j] - input_grid[i][j]) + (input_grid[i][j-1] - input_grid[i][j]) + (input_grid[i][j+1]) - input_grid[i][j]) \
                + m*d*((input_grid[i-1][j-1] - input_grid[i][j]) + (input_grid[i-1][j+1] - input_grid[i][j]) + (input_grid[i+1][j-1] - input_grid[i][j]) + (input_grid[i+1][j+1] - input_grid[i][j]))

    return output_grid