# JULIA Come for the performance, stay for the type system

## our algorithm:

given A = Matrix shape (N, M) filled with float32

- multiply A by it's transpose
- sum along the rows

in pseudo code
```
    out = sum(diagzero(A*A'), dims=2)
```

## python

- A*A' is pretty common so numpy optimizes it with Fortran / Blas

Problem ==> A*A' => Matrix (N, N) that could blow up in memory

Since we reduce the matrix to just (N, 1) we could do things using a for loop


### unrolling a loop is a major problem since it break the advantage of numpy array programming

- Most python engineers have given up at this point and for a good reason
- Welcome to optimisation hell...


cython build: 
```batch
python setup.py build_ext --inplace
```

```python
from cy_pack.g_func import g
g(arr)
```

## Now that we have seen that Julia is superior performance wise
But wait ! There's more !


## Let's dive in the type system

