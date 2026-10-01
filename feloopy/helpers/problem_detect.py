# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Problem classification and auto interface/solver selection.

Pure functions over a model's ``features`` dict so they can be imported
by ``feloopy.py`` (and others) without pulling in the model class.

Public API
----------
``analyze_expressions(features)``
    Scan objectives/constraints for linear / quadratic / nonlinear structure.
``compute_type(...)``
    Map feature flags to a problem-type label (LP, MILP, NLP, MINLP, ...).
``detect_problem_type(features)``
    End-to-end type detection (handles auto-linearization snapshots).
``select_interface()``
    Pick a default interface (HiGHS, then the first available binding).
``select_solver(features)``
    Pick a solver from problem type + available packages.
``INTERFACE_DEFAULT_SOLVER``
    Interface -> solver used when the problem-based pick cannot run.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "analyze_expressions",
    "compute_type",
    "detect_problem_type",
    "select_interface",
    "select_solver",
    "INTERFACE_DEFAULT_SOLVER",
    "AUTO_INTERFACE_ORDER",
    "NONLINEAR_OPS",
    "MINLP_SOLVER_ORDER",
]

NONLINEAR_OPS = [
    "sin", "cos", "tan", "log", "exp", "sqrt", "abs",
    "sinh", "cosh", "tanh", "asin", "acos", "atan", "log10",
]


MINLP_SOLVER_ORDER = ("scip", "bonmin", "couenne")

AUTO_INTERFACE_ORDER: Tuple[Tuple[str, str], ...] = (
    ("highs", "highspy"),
    ("pulp", "pulp"),
    ("pyomo", "pyomo"),
    ("scip", "pyscipopt"),
    ("mip", "mip"),
    ("linopy", "linopy"),
    ("cylp", "cylp"),
    ("picos", "picos"),
    ("ortools", "ortools"),
    ("gurobi", "gurobipy"),
    ("cplex", "cplex"),
    ("copt", "coptpy"),
    ("xpress", "xpress"),
    ("mosek", "mosek"),
    ("gams", "gamspy"),
    ("hexaly", "hexaly"),
)

# Solver an interface falls back to when the problem-based pick (HiGHS for
# linear, uno for nonlinear) cannot run under it.  Shared with ``feloopy``
# so model construction and ``sol()`` agree on the default.
INTERFACE_DEFAULT_SOLVER = {
    'highs': 'highs', 'pulp': 'cbc', 'gurobi': 'gurobi', 'cplex': 'cplex',
    'copt': 'copt', 'xpress': 'xpress', 'mosek': 'mosek', 'cvxpy': 'highs',
    'pyomo': 'cbc', 'casadi': 'ipopt', 'gekko': 'apopt', 'cylp': 'cbc',
    'pymprog': 'glpk', 'mathopt': 'highs', 'mip': 'cbc', 'linopy': 'cbc',
    'ortools': 'cbc', 'picos': 'cvxopt', 'jump': 'cbc', 'scip': 'scip',
    'bonmin': 'bonmin', 'couenne': 'couenne', 'uno': 'uno',
    'ortools_cp': 'ortools_cp', 'cplex_cp': 'cplex_cp', 'picat': 'picat',
    'rsome_ro': 'cplex', 'rsome_dro': 'cplex', 'gams': 'cplex',
    'hexaly': 'hexaly', 'insideopt': 'seeker', 'insideopt-demo': 'seeker',
}


def _solver_available(interface: str, solver: str) -> bool:
    """Whether ``solver`` can actually be executed under ``interface``.

    Auto-selection is problem-based, so it can name a solver the chosen
    interface cannot run here (PuLP dispatches ``highs`` to a ``highs.exe``
    binary that is not installed, while its bundled CBC always is).  The
    check is deliberately narrow: interfaces that pass the check keep the
    problem-based pick, and anything the check cannot answer is left for
    the generator to raise its usual, clearer error.
    """
    if interface == "pulp":
        return _pulp_solver_available(solver)
    return True


def _pulp_solver_available(solver: str) -> bool:
    """PuLP runs every solver as an external program: ask it directly."""
    try:
        from ..generators.solution.pulp_solution_generator import (
            _pulp_solver_map,
        )
    except Exception:  # pragma: no cover - generator always ships
        return True
    solver_cls = _pulp_solver_map.get(solver)
    if solver_cls is None:  # unknown to us; the generator reports it
        return True
    try:
        return bool(solver_cls(msg=0).available())
    except Exception:
        return False


def _counter(features: Dict, key: str) -> int:
    """Return the [count, size] slot's count (0 if missing)."""
    slot = features.get(key)
    if isinstance(slot, (list, tuple)) and slot:
        return int(slot[0])
    return 0


def analyze_expressions(
    features: Dict,
    objectives: Optional[List] = None,
    constraints: Optional[List] = None,
) -> Tuple[bool, bool, bool]:
    """Analyze objective and constraint expressions for structure.

    Parameters
    ----------
    features:
        Model features dict; used when ``objectives``/``constraints`` are None.
    objectives, constraints:
        Optional overrides (defaults to ``features['objectives']`` /
        ``features['constraints']``).

    Returns
    -------
    (is_linear, is_quadratic, has_nonlinear)
    """
    if objectives is None:
        objectives = features.get("objectives") or []
    if constraints is None:
        constraints = features.get("constraints") or []

    has_nonlinear = False
    is_quadratic = False

    def _check_expr(expr_str: str) -> None:
        nonlocal has_nonlinear, is_quadratic
        for op in NONLINEAR_OPS:
            if op + "(" in expr_str or op + " " in expr_str:
                has_nonlinear = True
                return
        if re.search(r"UnoExpr\(/\s*,\s*[^,]+,\s*UnoVar\(", expr_str):
            has_nonlinear = True
            return
        if re.search(r"/[a-zA-Z_]", expr_str):
            has_nonlinear = True
            return
        power_patterns = [
            r"\*\*\s*,\s*(?:UnoVar\([^)]+\)|UnoExpr\([^)]+\)),\s*(\d+)",
            r"\*\*(\d+)",
        ]
        for pattern in power_patterns:
            power_match = re.search(pattern, expr_str)
            if power_match:
                degree = int(power_match.group(1))
                if degree > 2:
                    has_nonlinear = True
                elif degree == 2:
                    is_quadratic = True
        if re.search(r"Term\([^)]+,\s*[^)]+\)", expr_str):
            is_quadratic = True

    for obj in objectives:
        _check_expr(str(obj))
        if has_nonlinear:
            return False, False, True

    for con in constraints:
        if isinstance(con, tuple) and len(con) == 3:
            lhs, _sense, rhs = con
            for expr in (lhs, rhs):
                _check_expr(str(expr))
                if has_nonlinear:
                    return False, False, True
        else:
            _check_expr(str(con))
            if has_nonlinear:
                return False, False, True

    is_linear = not has_nonlinear and not is_quadratic
    return is_linear, is_quadratic, has_nonlinear


def compute_type(
    has_mixed_integer: bool,
    has_continuous: bool,
    is_linear: bool,
    is_quadratic: bool,
    has_nonlinear: bool,
) -> str:
    """Map feature flags to a problem-type label.

    Types
    -----
    LP / IP / MILP / QP / IQP / MIQP / NLP / MINLP
    """
    if has_mixed_integer and (has_nonlinear or is_quadratic):
        return "MINLP"
    if has_nonlinear:
        return "NLP"
    if is_linear:
        if has_mixed_integer and not has_continuous:
            return "IP"
        if has_mixed_integer:
            return "MILP"
        return "LP"
    if is_quadratic:
        if has_mixed_integer and not has_continuous:
            return "IQP"
        if has_mixed_integer:
            return "MIQP"
        return "QP"
    return "NLP"


def detect_problem_type(features: Dict) -> str:
    """Detect the problem type from a model's features dict.

    When auto-linearization is active, records ``original_problem_type``
    and ``al_problem_type`` on ``features`` for the reporter, then returns
    the (linearized) type.
    """
    has_mixed_integer = bool(
        _counter(features, "integer_variable_counter")
        or _counter(features, "binary_variable_counter")
    )
    has_continuous = bool(
        _counter(features, "positive_variable_counter")
        or _counter(features, "free_variable_counter")
    )

    is_linear, is_quadratic, has_nonlinear = analyze_expressions(features)
    ptype = compute_type(
        has_mixed_integer, has_continuous, is_linear, is_quadratic, has_nonlinear
    )

    if features.get("_auto_lin_active"):
        orig_has_mixed = bool(
            features.get("_original_bvar_count", 0) > 0
            or features.get("_original_ivar_count", 0) > 0
        )
        features["original_problem_type"] = "MINLP" if orig_has_mixed else "NLP"
        features["al_problem_type"] = ptype
        return ptype

    return ptype


def select_interface() -> str:
    """Pick a default interface for ``interface='auto'``.

    HiGHS is always preferred - it is feloopy's default solver and ships
    in ``feloopy[stock]`` - but when ``highspy`` is not importable the
    first available binding from :data:`AUTO_INTERFACE_ORDER` is used, so
    ``'auto'`` resolves to something that can actually run rather than
    raising a bare ``ModuleNotFoundError``.  When nothing is installed it
    still returns ``"highs"``, so the failure names feloopy's default.
    """
    import importlib.util

    for interface, module in AUTO_INTERFACE_ORDER:
        try:
            if importlib.util.find_spec(module) is not None:
                return interface
        except (ImportError, ValueError):
            continue
    return "highs"


def select_solver(features: Dict) -> str:
    """Auto-select the best solver from model characteristics.

    Selection logic
    ---------------
    - LP / IP / MILP / QP / IQP / MIQP → HiGHS
    - NLP → uno
    - MINLP → scip > bonmin > couenne (first available)

    Raises
    ------
    RuntimeError
        If the problem is MINLP and no MINLP solver is installed.
    """
    interface = features.get("interface_name", "highs")
    if interface in MINLP_SOLVER_ORDER:
        return interface

    has_mixed_integer = bool(
        _counter(features, "integer_variable_counter")
        or _counter(features, "binary_variable_counter")
    )
    is_linear, is_quadratic, has_nonlinear = analyze_expressions(features)

    if has_mixed_integer and (has_nonlinear or is_quadratic):
        for name in MINLP_SOLVER_ORDER:
            if _minlp_solver_available(name):
                return name
        raise RuntimeError(
            "No MINLP solver available. Install one of: "
            "PySCIPOpt (pip install PySCIPOpt), "
            "bonmin (flp setup bonmin), or "
            "couenne (flp setup couenne)."
        )

    if is_linear:
        preferred = "highs"
    else:
        preferred = "uno"

    if _solver_available(interface, preferred):
        return preferred
    # HiGHS is the right pick for the problem but cannot run under this
    # interface here -> the interface's own default keeps every supported
    # interface (and therefore SDM) usable without manual configuration.
    return INTERFACE_DEFAULT_SOLVER.get(interface, preferred)


def _minlp_solver_available(name: str) -> bool:
    """Probe whether a named MINLP solver package is usable."""
    if name == "scip":
        try:
            import pyscipopt  # noqa: F401
            return True
        except ImportError:
            return False
    try:
        # Make ~/feloopy/Solvers binaries visible for the probe (no
        # downloads happen here — probing must stay side-effect free).
        from .solver_executables import inject_solver_paths
        inject_solver_paths((name,))
        from pyomo.environ import SolverFactory
        return bool(SolverFactory(name).available())
    except Exception:
        return False
