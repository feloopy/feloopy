# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Version compatibility helpers for the ``pulp`` interface.
"""

import pulp


PULP_V4 = int(str(getattr(pulp, "__version__", "0")).split(".")[0]) >= 4


def make_variable(problem, name, low_bound, up_bound, cat):
    """Create a variable belonging to ``problem`` on either major."""
    if PULP_V4:
        return problem.add_variable(
            name, lowBound=low_bound, upBound=up_bound, cat=cat)
    return pulp.LpVariable(name, low_bound, up_bound, cat)


def cbc_solver_class():
    """CBC driver: bundled ``PULP_CBC_CMD`` (3.x) or ``COIN_CMD`` (4.x).

    On 4.x this requires the ``cbc`` extra (``pip install pulp[cbc]``) or
    a ``cbc`` executable on PATH.
    """
    return getattr(pulp, "PULP_CBC_CMD", None) or pulp.COIN_CMD


def problem_variables(problem):
    """The problem's variables as a list (``variables`` is a method in 4.x)."""
    variables = problem.variables
    return variables() if callable(variables) else variables


def constraints_collection(problem):
    """The problem's constraints as a sized collection on either major."""
    constraints = problem.constraints
    return constraints() if not hasattr(constraints, "__len__") else constraints


def get_constraint(problem, label):
    """Look up a constraint by name on either major; ``None`` when absent."""
    getter = getattr(problem, "get_constraint_by_name", None)
    if getter is not None:
        # PuLP 4's rust core reserves leading '_' names, so the v4 rebuild
        # renames such labels the way PuLP's own MPS reader does (imp_ prefix).
        candidates = [label]
        if isinstance(label, str) and label.startswith("_"):
            candidates.append("imp_" + label)
        for candidate in candidates:
            try:
                found = getter(candidate)
            except Exception:
                found = None
            if found is not None:
                return found
        return None
    constraints = problem.constraints
    if hasattr(constraints, "get"):
        return constraints.get(label)
    return None
