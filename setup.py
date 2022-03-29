import os
import sys
from glob import glob
from pathlib import Path

# import pybind11
# from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, find_packages, setup

# from skbuild import setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

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


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name


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


# ext_modules = [
#     Pybind11Extension(
#         "solver_fast",
#         ["rlassopy/solver_fast.cpp"],
#         include_dirs=[
#             get_pybind_include(),
#             get_pybind_include(user=True),
#             get_eigen_include(),
#             # os.environ.get("EIGEN_INCLUDE_DIR", "extern/eigen-3.4.0"),
#         ],
#         define_macros=[("VERSION_INFO", __version__)],
#     ),
# ]

cpp_args = ["-std=c++14"]

ext_modules = [
    Extension(
        "solver_fast",
        ["rlassopy/solver_fast.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
        extra_compile_args=cpp_args,
    ),
]
extra_requires = {
    "tests": ["pytest", "pytest-cov"],
    # "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

install_requires = ["numpy", "scipy", "scikit-learn", "cvxpy"]

setup(
    name="rlassopy",
    version=__version__,
    author="Matias Piqueras",
    author_email="matias@piqueras.se",
    url="https://github.com/matpiq/rlassopy",
    description="Rigorous Lasso in Python",
    long_description="",
    ext_modules=ext_modules,
    # cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extra_requires,
    include_package_data=True,
    zip_safe=False,
)
