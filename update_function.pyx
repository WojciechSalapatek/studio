cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def update_grid(np.ndarray[np.float64_t, ndim=2] input_grid, np.ndarray[np.float64_t, ndim=2] output_grid,
                double m, double d, np.ndarray[np.float64_t, ndim=2] west_currents_grid,
                np.ndarray[np.float64_t, ndim=2] north_currents_grid):
    cdef:
        size_t i
        size_t j
        float w
        float n
        float s
        float e
        float wn
        size_t rows
        size_t cols

    rows = input_grid.shape[0]-1
    cols = input_grid.shape[1]-1
    for i in prange(1, rows, nogil=True):
        for j in prange(1, cols):
            n = north_currents_grid[i,j] + north_currents_grid[i+1,j]
            w = west_currents_grid[i,j] + west_currents_grid[i-1,j]
            s = -north_currents_grid[i,j] - north_currents_grid[i-1,j]
            e = -west_currents_grid[i,j] - west_currents_grid[i+1,j]
            wn = sqrt(n*n + w*w)

            output_grid[i,j] = \
                input_grid[i,j] \
                + m*(((1+s)*input_grid[i-1,j] - (1-s)*input_grid[i,j]) + ((1+n)*input_grid[i+1,j] - (1-n)*input_grid[i,j]) + ((1+w)*input_grid[i,j-1] - (1-w)*input_grid[i,j]) + ((1+e)*input_grid[i,j+1]) - (1-e)*input_grid[i,j]) \
                + m*d*((input_grid[i-1,j-1] - input_grid[i,j]) + (input_grid[i-1,j+1] - input_grid[i,j]) + (input_grid[i+1,j-1] - input_grid[i,j]) + (input_grid[i+1,j+1] - input_grid[i,j]))
    return output_grid