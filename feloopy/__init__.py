# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import warnings
import contextlib
from typing import Optional

from ._version import __version__

warnings.filterwarnings("ignore", module="cvxpy")
warnings.filterwarnings("ignore", module="pulp")

_gurobi_suppressed = False
_julia_configured = False


def _suppress_gurobi_banner() -> None:
    """Suppress Gurobi's 'Restricted license' banner on first use."""
    global _gurobi_suppressed
    if _gurobi_suppressed:
        return
    _gurobi_suppressed = True

    try:
        import gurobipy  # noqa: F401
    except Exception:
        return

    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                env = gurobipy.Env(empty=True)
                env.setParam("OutputFlag", 0)
                try:
                    env.setParam("LogToConsole", 0)
                except Exception:
                    pass
                env.start()
                m = gurobipy.Model("__feloopy_silent", env=env)
                m.dispose()
                env.dispose()
    except Exception:
        pass


def _suppress_stderr():
    """Context manager to suppress stderr from noisy third-party imports."""
    return contextlib.redirect_stderr(open(os.devnull, "w"))


def _configure_julia() -> None:
    """Point juliacall/juliapkg at feloopy-managed Julia (lazy, once)."""
    global _julia_configured
    if _julia_configured:
        return
    _julia_configured = True
    try:
        from .helpers.julia_server import configure_feloopy_julia
        configure_feloopy_julia()
    except Exception:
        pass


def _auto_setup_coin() -> None:
    """Deprecated no-op kept for ``ensure_dependencies(coin=True)``.

    COIN-OR solver binaries live in ``~/feloopy/Solvers`` and come from
    ``flp setup <solver>`` (or a one-time download on first explicit
    use of ``interface="bonmin"`` / ``interface="couenne"``).
    """
    return


def _ensure_coin_setup() -> None:
    _auto_setup_coin()


# Eager setup for Gurobi banner and Julia path.
# Solver binaries stay lazy (resolved in the generators on first use).
_suppress_gurobi_banner()
_configure_julia()

with _suppress_stderr():
    try:
        from .extras import *
    except ImportError:
        pass

    from .feloopy import *
    from .classes.linearization import (
        _LINEARIZATION_CACHE,
        _make_cache_key,
        _al_proxy_key,
    )
    from .helpers.registries import (
        HEURISTIC_ALGORITHMS,
        EXACT_ALGORITHMS,
        UNCERTAINTY_ALGORITHMS,
        CONSTRAINT_ALGORITHMS,
        MOO_ALGORITHMS,
        WEIGHTING_ALGORITHMS,
        RANKING_ALGORITHMS,
        SPECIAL_ALGORITHMS,
    )
    from .algorithms.exact.multilevel import MultiLevelProblem, MultiLevelError
    from .algorithms.exact.benders import (
        BendersDecomposition, BendersCutStrategy, BendersAcceleration,
        BendersMethod, BendersStatus, BendersError,
        BendersResult, BendersParams, BendersCallback, BendersContext,
        benders, benders_decomposition,
    )
    from .algorithms.exact.column_generation import ColumnGeneration, ColumnGenerationResult, ColumnGenerationError
    from .algorithms.exact.lagrangian import LagrangianRelaxation, LagrangianResult, LagrangianError
    from .algorithms.exact.branching import (
        BranchAndBound, BranchAndCut, BranchAndPrice,
        BranchingStatus, NodeStrategy, SelectionStrategy,
        BranchAndBoundResult, BranchAndCutResult, BranchAndPriceResult,
        BranchingCallback, SeparationOracle,
        Cut, CutPool, LazyConstraint, CutType, SeparationOracles,
    )
    from .algorithms.exact.sequential import (
        SequentialDecisionProblem, SequentialError, PolicyType,
        PFA, CFA, VFA, DLA,
        SimulationResult, OptimizationResult, sol_sequential,
    )
    from .helpers.solver_params import get_solver_params as get_params
    from .helpers.update_check import check_update, schedule_update_check


def ensure_dependencies(
    gurobi: bool = True,
    julia: bool = True,
    coin: bool = True,
) -> None:
    """Explicitly initialize optional heavy dependencies.

    Call once at application startup if you want eager setup.
    Otherwise they initialize lazily on first use.
    """
    if gurobi:
        _suppress_gurobi_banner()
    if julia:
        _configure_julia()
    if coin:
        _auto_setup_coin()


def clear(project_dir: Optional[str] = None, verbose: bool = False) -> None:
    """Clear feloopy API cache and Python __pycache__ directories.

    Args:
        project_dir: Root directory to search for __pycache__ folders.
                     Defaults to the caller's working directory.
        verbose: If True, print what was cleared.
    """
    from .classes.linearization import _LINEARIZATION_CACHE

    _LINEARIZATION_CACHE.clear()
    if verbose:
        print("API cache cleared.")

    import pathlib
    root = pathlib.Path(project_dir) if project_dir else pathlib.Path.cwd()
    count = 0
    for pycache in root.rglob("__pycache__"):
        import shutil
        shutil.rmtree(pycache, ignore_errors=True)
        count += 1
    if verbose:
        print(f"Cleared {count} __pycache__ directory(ies) under {root}.")


# Non-blocking, throttled PyPI release check (see helpers/update_check.py).
# Disable with FELOOPY_DISABLE_UPDATE_CHECK=1.
try:
    schedule_update_check()
except Exception:
    pass