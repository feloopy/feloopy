# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from enum import Enum


class BendersCutStrategy(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


class BendersAcceleration(Enum):
    NONE = 'none'
    PARETO = 'pareto'
    STABILIZATION = 'stabilization'
    TRUST_REGION = 'trust_region'
    MULTI_CUT = 'multi_cut'
    HYBRID = 'hybrid'


class BendersMethod(Enum):
    AUTO = 'auto'
    CLASSICAL = 'classical'
    LOGIC = 'logic'
    COMBINATORIAL = 'combinatorial'
    GENERALIZED = 'generalized'
    L_SHAPED = 'l_shaped'
    INTEGER_L_SHAPED = 'integer_l_shaped'
    GENERALIZED_L_SHAPED = 'generalized_l_shaped'


class BendersStatus(Enum):
    UNSOLVED = 'unsolved'
    OPTIMAL = 'optimal'
    FEASIBLE = 'feasible'
    INFEASIBLE = 'infeasible'
    UNBOUNDED = 'unbounded'
    TIMEOUT = 'timeout'
    MAX_ITERATIONS = 'max_iterations'
    DIRECT = 'direct'
    ERROR = 'error'


class BendersError(Exception):
    """Raised when Benders decomposition configuration or solve fails."""
