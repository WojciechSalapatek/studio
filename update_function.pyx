#cython.wraparound=False
#cython.boundscheck=False
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def update_grid(np.ndarray[np.float64_t, ndim=2] input_grid, np.ndarray[np.float64_t, ndim=2] output_grid,
                double m, double d, np.ndarray[np.float64_t, ndim=2] west_currents_grid, np.float64_t west_max_current,
                np.ndarray[np.float64_t, ndim=2] north_currents_grid, np.float64_t north_max_current):
    cdef:
        size_t i
        size_t j
        float w
        float n
        float wn
        int rows
        int cols

    rows = input_grid.shape[0]-1
    cols = input_grid.shape[1]-1
    for i in prange(1, rows, nogil=True):
        for j in range(1, cols):
            n = ((north_currents_grid[i,j] + north_currents_grid[i,j+1])/2/north_max_current)
            w = ((west_currents_grid[i,j] + west_currents_grid[i-1,j])/2/west_max_current)
            wn = sqrt(n*n + w*w)

            output_grid[i,j] = \
                input_grid[i,j] \
                + m*(((1+w)*input_grid[i-1,j] - (1-w)*input_grid[i,j]) + (input_grid[i+1,j] - input_grid[i,j]) + ((n+1)*input_grid[i,j-1] - (n-1)*input_grid[i,j]) + (input_grid[i,j+1]) - input_grid[i,j]) \
                + m*d*((input_grid[i-1,j-1] - input_grid[i,j]) + (input_grid[i-1,j+1] - input_grid[i,j]) + (input_grid[i+1,j-1] - input_grid[i,j]) + (input_grid[i+1,j+1] - input_grid[i,j]))

    return output_grid