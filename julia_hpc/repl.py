import numpy as np

# arr = np.random.randn(1_000, 10).astype(np.float32)
arr = np.random.randn(100_000, 10).astype(np.float32)

def f(a):
    y = a @ a.T
    np.fill_diagonal(y, 0)
    return y.sum(axis=1)

%timeit f(arr)

def g(a: np.array) -> np.array:
    N, M = a.shape
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        total = np.float32(0)
        for ii in range(N):
            if i == ii: continue
            for j in range(M):
                total += a[i,j] * a[ii, j]
        out[i] = total
    return out

%timeit g(arr)

from numba import njit, prange

@njit
def g(a):
    N,M = a.shape
    out = np.zeros(N, dtype=np.float32)

    for i in prange(N):
        total = np.float32(0)
        for ii in range(N):
            if i == ii: continue
            for j in range(M):
                total += a[i,j] * a[ii, j]
        out[i] = total
    return out

g(arr)
%timeit g(arr)