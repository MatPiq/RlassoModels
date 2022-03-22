from glob import glob

import pybind11

# from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

__version__ = "0.0.1"

cpp_args = ["-std=c++14"]

ext_modules = [
    Extension(
        "solver_fast",
        sorted(glob("rlassopy/*.cpp")),
        include_dirs=[pybind11.get_include(), "extern/eigen-3.4.0"],
        language="c++",
        extra_compile_args=cpp_args,
    ),
]
extra_requires = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name="rlassopy",
    version=__version__,
    author="Matias Piqueras",
    author_email="matias@piqueras.se",
    url="https://github.com/matpiq/rlassopy",
    description="Rigorous Lasso in Python",
    long_description="",
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy", "sklearn"],
    extras_require=extra_requires,
    zip_safe=False,
)
