import os
from glob import glob

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from skbuild import setup

# from setuptools import Extension, setup
__version__ = "0.0.1"

# linkargs = [
#     "-Ofast",
#     "-ffast-math",
#     "-lpthread",
#     "-lgomp",
#     "-fopenmp-simd",
#     "-g",
#     "-w",
#     "-fPIC",
#     "-DNDEBUG",
#     "-DEIGEN_USE_BLAS",
#     # "-lopenblas",
# ]
# compargs = [
#     "-Ofast",
#     "-ffast-math",
#     "-lpthread",
#     "-lgomp",
#     "-fopenmp-simd",
#     "-std=c++17",
#     "-DNDEBUG",
#     "-DEIGEN_USE_BLAS",
#     # "-lopenblas",
# ]


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Pybind11Extension(
        "solver_fast",
        sorted(glob("rlassopy/*.cpp")),
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            os.environ.get("EIGEN_INCLUDE_DIR", "extern/eigen-3.4.0"),
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

# cpp_args = ["-std=c++14"]
#
# ext_modules = [
#     Extension(
#         "solver_fast",
#         sorted(glob("rlassopy/*.cpp")),
#         include_dirs=[pybind11.get_include(), "extern/eigen-3.4.0"],
#         language="c++",
#         extra_compile_args=cpp_args,
#     ),
# ]
extra_requires = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

install_requires = ["numpy", "scipy", "scikit-learn"]

setup(
    name="rlassopy",
    version=__version__,
    author="Matias Piqueras",
    author_email="matias@piqueras.se",
    url="https://github.com/matpiq/rlassopy",
    description="Rigorous Lasso in Python",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extra_requires,
    zip_safe=False,
)
