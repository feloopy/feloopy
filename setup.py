# Copyright (c) 2022-2024, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from setuptools import setup, find_packages

# Common packages for all versions of FelooPy

common = [
    'gputil',
    'infix',
    'matplotlib',
    'nbformat',
    'numba',
    'numpy',
    'openpyxl',
    'pandas',
    'plotly',
    'polars',
    'psutil',
    'py-cpuinfo',
    'scikit-learn',
    'tabulate',
    'win-unicode-console',
    'xlsxwriter',
    'tqdm',
    'colorama',
]

# Interfaces for optimization solvers or algorithms

pico = [
    ''
]

nano = pico + [
    'pymprog==1.1.2',
    'highspy',
]

micro = nano + [
    'gekko==1.1.1',
    'mealpy==3.0.1',
]

mini = micro + [
    'cvxpy==1.5.0',
    'ortools==9.10.4067',
]

full = mini + [
    'pydecision==4.5.4',
    'pymoo==0.6.1.1',
    'rsome==1.2.6',
]

stock = full + [
    'linopy==0.3.8',
    'mip==1.15.0',
    'niapy==2.1.0',
    'picos==2.4.17',
    'pulp==2.8.0',
    'pygad==3.2.0',
    'pyomo==6.7.2',
    'highsbox',
    'pyoptinterface',
]

# Solvers for optimization problems or algorithms

plus_gurobi = [
    'gurobipy',
]

plus_cplex = [
    'cplex',
    'docplex',
]

plus_xpress = [
    'xpress',
]

plus_copt = [
    'coptpy',
]

plus_insideoptdemo = [
    'insideopt-demo',
]

plus_insideopt = [
    'insideopt',
]

plus_gams = [
    'gamspy',
]

plus_jump = [
    'juliacall',
]

hyper = stock + plus_gurobi + plus_cplex + plus_xpress + plus_copt + plus_gams

# Might have some os-dependent issues

only_cylp = [
    'cylp==0.92.2',
]

only_linux = [
    'pymultiobjective==1.5.4',
]

mega = hyper + only_cylp + only_linux

extra_dict = {
    'pico': pico,
    'nano': nano,
    'micro': micro, 
    'mini': mini, 
    'full': full, 
    'stock': stock,
    'hyper': hyper,
    'plus_gurobi': plus_gurobi,
    'plus_cplex': plus_cplex,
    'plus_xpress': plus_xpress,
    'plus_copt': plus_copt,
    'plus_insideopt': plus_insideopt,
    'plus_insideoptdemo': plus_insideoptdemo,
    'plus_gams': plus_gams,
    'plus_jump': plus_jump,
    'only_cylp': only_cylp,
    'only_linux': only_linux,
    'mega': mega}

keywords_list = [
    'computer science',
    'data science',
    'decision making',
    'decision science',
    'industrial engineering',
    'machine learning',
    'management science',
    'mathematical modeling',
    'operations management',
    'operations research',
    'optimization',
    'simulation',
    'supply chain',
    ]

if __name__ == '__main__':

    setup(
        
        name = 'feloopy',
        
        version = '0.3.5',
        
        description = 'FelooPy: Efficient and feature-rich integrated decision environment',
        
        packages = find_packages(include=['feloopy', 'feloopy.*']),
        
        long_description = open('./README.md', encoding="utf8").read(),
        
        long_description_content_type = 'text/markdown',
        
        keywords = keywords_list,
        
        author='Keivan Tafakkori',
        
        author_email='k.tafakkori@gmail.com',
        
        maintainer='Keivan Tafakkori',
        
        maintainer_email='k.tafakkori@gmail.com',
        
        url='https://github.com/ktafakkori/feloopy',
        
        download_url='https://github.com/ktafakkori/feloopy/releases',
        
        license='MIT',
        
        entry_points={
        
        "console_scripts": ["feloopy = feloopy.cli:main",
                            "flp = feloopy.cli:main",
                            "fly = feloopy.cli:main",
                            ]},
        
        python_requires='>=3.10',
        
        extras_require=extra_dict,

        install_requires=common,
        
    )
