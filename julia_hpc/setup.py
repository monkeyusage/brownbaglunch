from os import name
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

OPEN_MP = ["-fopenmp"] if name != "nt" else ["/openmp"]

ext_modules = [
    Extension(
        "cy_pack.g_func",
        ["cy_pack/g_func.pyx"],
        extra_compile_args = OPEN_MP,
        extra_link_args = OPEN_MP,
    )
]

# setup(
#     ext_modules = cythonize("sparsy/numeric.pyx"),
#     include_dirs=[numpy.get_include()]
# )

setup(
    name="cy_pack.g_func",
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)