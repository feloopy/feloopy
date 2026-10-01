from enum import Enum

import numpy as np


class SequentialError(Exception):
    """Raised when sequential decision problem configuration or simulation fails."""


class PolicyType(Enum):
    PFA = 'pfa'
    CFA = 'cfa'
    VFA = 'vfa'
    DLA = 'dla'


_VALID_METHODS = ('exact', 'heuristic', 'uncertain', 'constraint')


# Options that belong to the sequential search itself.  They are forwarded
# into the policy objects by flp.search(...) and must never reach a stage
# solver: pulp/gurobi/... reject unknown keywords outright.
_SDM_SEARCH_OPTIONS = frozenset({
    'environment', 'policy', 'transition', 'cost', 'reward', 'exogenous',
    'terminal_cost', 'feasibility', 'build_model', 'S0', 'theta0',
    'theta_bounds', 'T', 'N', 'horizon', 'nsim', 'discount',
    'discount_factor', 'seed', 'save_trace', 'compare', 'search_theta',
    'opt_solver', 'opt_interface', 'opt_solver_options', 'max_iters',
    'optimize', 'var_names', 'scenarios', 'obj_operators',
    'uncertainty_set_constraints', 'verbose', 'repeat', 'should_run',
})

# Tuning knobs that only mean something for a metaheuristic solver; an
# exact solver would choke on them the same way it chokes on 'horizon'.
_SDM_HEURISTIC_OPTIONS = frozenset({
    'epoch', 'pop_size', 'population_size', 'sol_per_pop', 'crossover_rate',
    'mutation_rate', 'max_fe', 'max_iter', 'max_iterations',
})

# Default search budget for a heuristic *stage* solve.  A stage is rebuilt
# and re-solved T x N times per simulation (and once per theta candidate
# during optimize), so feloopy's global heuristic default (100 epochs x
# 50 pop = 5000 model rebuilds, ~2 min per stage) makes an SDM run take
# hours.  5 x 10 is the budget the SDM tests/examples use; any epoch or
# pop_size the policy sets replaces it.
_SDM_DEFAULT_STAGE_EPOCH = 5
_SDM_DEFAULT_STAGE_POP_SIZE = 10


def _log_message(msg):
    print(f"[Feloopy:Sequential] {msg}", flush=True)


def _resolve_dim(dim):
    """Resolve dim to an integer size.

    dim=0 or dim=[] -> 1 (scalar)
    dim=N (int)     -> N
    dim=[N]         -> N
    dim=[N, M]      -> N*M
    """
    if dim == 0 or dim == [] or dim == ():
        return 1
    elif isinstance(dim, (list, tuple)):
        return int(np.prod(dim)) if len(dim) > 1 else int(dim[0])
    else:
        return int(dim)
