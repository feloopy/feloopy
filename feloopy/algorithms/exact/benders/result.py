# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import json
from dataclasses import dataclass, field

from .enums import BendersStatus


@dataclass
class BendersResult:
    """Structured result from Benders decomposition with runtime breakdown.

    Attributes
    ----------
    status : BendersStatus
        Termination status (OPTIMAL, FEASIBLE, INFEASIBLE, TIMEOUT, ...).
    objective : float
        Best objective value found.
    variables : dict
        Variable-name → value mapping for the best incumbent.
    iterations : int
        Number of Benders iterations completed.
    bounds : list of (float, float)
        (LB, UB) pair per iteration.
    n_optimality_cuts : int
        Count of optimality cuts added.
    n_feasibility_cuts : int
        Count of feasibility cuts added.
    n_cuts : int
        Total cuts added (optimality + feasibility).
    n_sol : int
        Number of feasible solutions found across iterations.
    runtime_total : float
        Total wall-clock time in seconds.
    runtime_master : float
        Time spent solving master problems.
    runtime_subproblem : float
        Time spent solving subproblems.
    gap_abs : float
        Absolute gap |UB - LB| at termination.
    gap_rel : float
        Relative gap |UB - LB| / |UB| at termination.
    lb_history : list of float
        Lower bound per iteration.
    ub_history : list of float
        Upper bound per iteration.
    obj_history : list of float
        Incumbent objective per iteration.
    """
    status: BendersStatus = BendersStatus.UNSOLVED
    objective: float = float('inf')
    variables: dict = field(default_factory=dict)
    iterations: int = 0
    bounds: list = field(default_factory=list)
    n_optimality_cuts: int = 0
    n_feasibility_cuts: int = 0
    n_cuts: int = 0
    n_sol: int = 0
    runtime_total: float = 0.0
    runtime_master: float = 0.0
    runtime_subproblem: float = 0.0
    gap_abs: float = float('inf')
    gap_rel: float = float('inf')
    lb_history: list = field(default_factory=list)
    ub_history: list = field(default_factory=list)
    obj_history: list = field(default_factory=list)

    def save(self, filename):
        """Export result to JSON.

        Parameters
        ----------
        filename : str
            Path to write the JSON file.
        """
        data = {
            'status': self.status.value if hasattr(self.status, 'value')
                      else str(self.status),
            'objective': self.objective,
            'iterations': self.iterations,
            'n_optimality_cuts': self.n_optimality_cuts,
            'n_feasibility_cuts': self.n_feasibility_cuts,
            'n_cuts': self.n_cuts,
            'n_sol': self.n_sol,
            'gap_abs': self.gap_abs,
            'gap_rel': self.gap_rel,
            'runtime_total': self.runtime_total,
            'runtime_master': self.runtime_master,
            'runtime_subproblem': self.runtime_subproblem,
            'lb_history': self.lb_history,
            'ub_history': self.ub_history,
            'obj_history': self.obj_history,
            'variables': {k: v for k, v in self.variables.items()
                          if v is not None},
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)


@dataclass
class BendersParams:
    """Parameters for Benders decomposition, comparable to BendersLib's BendersParams."""
    tol_abs: float = 1e-6
    """Absolute convergence tolerance: terminate when |UB - LB| <= tol_abs."""
    tol_rel: float = 1e-4
    """Relative convergence tolerance: terminate when |UB - LB| / |UB| <= tol_rel."""
    time_limit: float = float('inf')
    """Wall-clock time limit in seconds."""
    iter_limit: int = 1000000
    """Maximum number of Benders iterations."""
    theta_lb: float = 0.0
    """Lower bound for the estimator (theta) variable in the master problem."""
    cut_normalize: bool = False
    """Whether to normalize cut coefficients to improve numerical stability."""
    cut_max_norm: float = 1e5
    """Maximum L2 norm for cut coefficients when cut_normalize is True."""
    cut_tol_diff: float = 1e-8
    """Tolerance for detecting duplicate cuts."""
    tol_obj_diff: float = 1e-6
    """Tolerance for comparing subproblem objective vs theta."""
    multi_opti_cut: bool = False
    """For L-Shaped: add one optimality cut per scenario vs aggregated."""
    multi_feas_cut: bool = True
    """For L-Shaped: add feasibility cuts from all infeasible scenarios vs stop at first."""
    parallel_sub: bool = True
    """Solve subproblems in parallel (whenever >=2 are independent)."""
    parallel_threads: int = -1
    """Number of threads for parallel subproblem solving (-1 = auto)."""
    use_iis: bool = False
    """Use IIS-based no-good cuts for combinatorial Benders."""
    big_m: float = None
    """Big-M constant for combinatorial Benders. Auto-computed if None."""
    use_bnc: bool = False
    """Use Branch-and-Check: solve master once with lazy callbacks (Gurobi only)."""
    bnc_frac_sol: bool = False
    """Generate cuts at fractional solutions too (not just integer)."""
    log_freq_sec: float = 0.5
    """Minimum seconds between log messages."""
    log_freq_iter: int = 1
    """Log every N iterations."""
    log_level: str = 'INFO'
    """Logging level: DEBUG, INFO, WARNING, ERROR."""
    log_to_console: bool = True
    """Whether to print logs to stdout."""
    log_file: str = None
    """Path to log file. None disables file logging."""
