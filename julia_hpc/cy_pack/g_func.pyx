import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import parallel, prange

from libc.stdlib cimport free, malloc
from os import cpu_count

cdef int __CPUS = cpu_count() - 1 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def g(float[:, :] a) -> list[float]:
    """
    Uses MemoryView from numpy array as input
    Rough equivalent of numpy:
    >>> def dot_zero(matrix: np.ndarray) -> np.ndarray:
            out = matrix.dot(matrix.T)
            np.fill_diagonal(out, 0)
            return out.sum(axis=1)
    However in this version we do not expand memory from N x M (e.g 68K x 7) to N x N (68K x 68K)
    We reduce the output while computing results thus keeping memory to the minimum N

    matrix shape : N x M
    out shape : N x 1
    """
    cdef Py_ssize_t  K = a.shape[0]
    cdef Py_ssize_t  J = a.shape[1]
    cdef Py_ssize_t  I = a.shape[0]
    cdef float* out = <float *> malloc(K * sizeof(float))
    if not out:
        raise MemoryError()
    cdef float total
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    try:
        with nogil, parallel(num_threads=__CPUS):
            for k in prange(K):
                total = 0
                for i in range(I):
                    if i == k:
                        continue
                    for j in range(J):
                        total = total + (a[k, j] * a[i, j])
                out[k] = total
        return [item for item in out[:K]]
    finally:
        free(out)