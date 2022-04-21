import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages
from skbuild import setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

__version__ = "0.0.1"


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


ext_modules = [
    Pybind11Extension(
        "_solver_fast",
        ["rlassomodels/_solver_fast.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

extra_requires = {
    "test": ["pytest", "pytest-cov"],
}

install_requires = [
    "numpy",
    "scipy",
    "scikit-learn",
    "cvxpy",
    "patsy",
    "pandas",
    "statsmodels",
    "linearmodels",
]

setup(
    name="RlassoModels",
    version=__version__,
    author="Matias Piqueras",
    author_email="matias@piqueras.se",
    url="https://github.com/matpiq/rlassomodels",
    description="Rigorous Lasso in Python",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=install_requires,
    extras_require=extra_requires,
    packages=find_packages(),
    zip_safe=False,
)
