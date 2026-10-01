"""Multi-objective optimisation via epsilon constraint and weighted sum methods.

Generates Pareto-optimal trade-off surfaces for problems with
conflicting objectives.

Functions:
    sol_multi -- generate Pareto set using ECM or NWSM approach.

Error:
    MultiObjectiveError
"""

from .enums import MultiObjectiveError
from .core import sol_multi

__all__ = [
    'MultiObjectiveError',
    'sol_multi',
]
