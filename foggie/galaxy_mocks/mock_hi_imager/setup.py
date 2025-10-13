from setuptools import setup
from Cython.Build import cythonize
import numpy as np

#To Compile:
#python setup.py build_ext --inplace

setup(
    ext_modules=cythonize("custom_clean.pyx"),
    include_dirs=[np.get_include()]
)