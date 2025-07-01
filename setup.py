# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from setuptools import setup, find_packages
import os

# Common packages for all versions of FelooPy

common = [
    "gputil",
    "infix",
    "matplotlib",
    "nbformat",
    "numpy",
    "openpyxl",
    "pandas",
    "psutil",
    "py-cpuinfo",
    "tabulate",
    "win-unicode-console",
    "xlsxwriter",
    "tqdm",
    "colorama",
    "rich",
]

# Interfaces for optimization solvers or algorithms

pico = [""]

nano = pico + [
    "pymprog==1.1.2",
    "highspy==1.10.0",
]

micro = nano + [
    "gekko==1.3.0",
    "mealpy==3.0.1",
]

mini = micro + [
    "cvxpy==1.6.4",
    "ortools==9.12.4544",
]

full = mini + [
    "pydecision==4.7.5",
    "pymoo==0.6.1.3",
    "rsome==1.3.1",
]

stock = full + [
    "linopy==0.5.2",
    "mip==1.15.0",
    "niapy==2.5.2",
    "picos==2.6.0",
    "pulp==3.1.1",
    "pygad==3.4.0",
    "pyomo==6.9.1",
    "highsbox==1.10.0",
    "pyoptinterface==0.4.1",
]

# Solvers for optimization problems or algorithms

plus_gurobi = [
    "gurobipy==12.0.1",
]

plus_copt = [
    "coptpy==7.2.6",
]

plus_cplex = [
    "cplex==22.1.2.0",
    "docplex==2.29.241",
]

plus_xpress = [
    "xpress==9.5.4",
]

plus_gams = [
    "gamspy==1.8.0",
]

plus_insideoptdemo = [
    "insideopt-demo==0.3.3",
]

plus_insideopt = [
    "insideopt-seeker==0.1.21",
]

plus_jump = [
    "juliacall==0.9.24",
]

hyper = stock + plus_gurobi + plus_cplex + plus_xpress + plus_copt + plus_gams

# Might have some os-dependent issues

only_cylp = [
    "cylp==0.92.2",
]

only_linux = [
    "pymultiobjective==1.5.5",
]

mega = hyper + only_cylp + only_linux

beta = ["casadi==3.7.0", "scipy==1.15.2"]

canvas = [
    "IPython",
    "gif",
    "folium",
    "plotly",
]

extras = ["numba", "scikit-learn"]

extra_dict = {
    "pico": pico,
    "nano": nano,
    "micro": micro,
    "mini": mini,
    "full": full,
    "stock": stock,
    "hyper": hyper,
    "plus_gurobi": plus_gurobi,
    "plus_cplex": plus_cplex,
    "plus_xpress": plus_xpress,
    "plus_copt": plus_copt,
    "plus_insideopt": plus_insideopt,
    "plus_insideoptdemo": plus_insideoptdemo,
    "plus_gams": plus_gams,
    "plus_jump": plus_jump,
    "only_cylp": only_cylp,
    "only_linux": only_linux,
    "mega": mega,
    "beta": beta,
}

keywords_list = [
    "computer science",
    "data science",
    "decision making",
    "decision science",
    "industrial engineering",
    "machine learning",
    "management science",
    "mathematical modeling",
    "operations management",
    "operations research",
    "optimization",
    "simulation",
    "supply chain",
]

here = os.path.abspath(os.path.dirname(__file__))
def read_file(fname):
    path = os.path.join(here, fname)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_version():
    version_ns = {}
    version_file = os.path.join(os.path.dirname(__file__), "feloopy", "_version.py")
    with open(version_file) as f:
        exec(f.read(), version_ns)
    return version_ns["__version__"]

if __name__ == "__main__":

    setup(
        name="feloopy",
        version=get_version(),
        description="FelooPy: Efficient and feature-rich integrated decision environment",
        packages=find_packages(include=["feloopy", "feloopy.*"]),
        long_description=read_file("readme.md"),
        long_description_content_type="text/markdown",
        keywords=keywords_list,
        author="Keivan Tafakkori",
        author_email="k.tafakkori@gmail.com",
        maintainer="Keivan Tafakkori",
        maintainer_email="k.tafakkori@gmail.com",
        url="https://feloopy.github.io/",
        download_url="https://github.com/feloopy/feloopy/releases",
        license="MIT",
        entry_points={
            "console_scripts": [
                "feloopy = feloopy.cli:main",
                "flp = feloopy.cli:main",
                "fly = feloopy.cli:main",
            ]
        },
        python_requires=">=3.10",
        extras_require=extra_dict,
        install_requires=common,
    )
