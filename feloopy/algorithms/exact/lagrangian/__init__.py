"""Lagrangian relaxation for constrained optimisation.

Relaxes difficult constraints into the objective with Lagrange multipliers,
then solves a sequence of relaxed subproblems to refine the multipliers.

Classes:
    LagrangianRelaxation -- manual and automatic Lagrangian relaxation.

Result:
    LagrangianResult

Error:
    LagrangianError
"""

from .core import LagrangianRelaxation, LagrangianResult, LagrangianError
