### Cython fasty fast stuff
### Peyton Greenside
### 9/8/15

import numpy as np
cimport numpy as np
cimport cython

# x is weights_i
@cython.boundscheck(False)
def calc_sqrt_sum_sqr_sqr_sums(np.ndarray[np.float64_t, ndim=1] x):
    # find all non-zero entries
    cdef float sum_squared_values = 0
    cdef float sum_values = 0
    cdef Py_ssize_t i
    cdef float value
    for i in range(x.shape[0]):
        value = x[i] 
        sum_squared_values += value*value
        sum_values += value
    return np.sqrt(sum_squared_values/(sum_values**2))
