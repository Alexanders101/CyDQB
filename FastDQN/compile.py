from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

from numpy import get_include
import os

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

ext = [Extension("utils", ["utils.pyx"],
                 include_dirs=[get_include()],
                 extra_compile_args=["-O3", "-march=native"],
                 language="c++"),
       Extension("DQNModel", ["DQNModel.pyx"],
                 include_dirs=[get_include()],
                 extra_compile_args=["-O3", "-march=native"],
                 language="c++"),
       Extension("FastDQN", ["FastDQN.pyx"],
                 include_dirs=[get_include()],
                 extra_compile_args=["-O3", "-march=native"],
                 language="c++"),
       Extension("tests", ["tests.pyx"],
                 include_dirs=[get_include()],
                 extra_compile_args=["-O3", "-march=native"],
                 language="c++")
       ]

setup(ext_modules=cythonize(ext, annotate=True),
      cmdclass={'build_ext': build_ext})
