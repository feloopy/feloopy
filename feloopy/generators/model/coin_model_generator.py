# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Pyomo model generation for the bonmin/couenne (COIN-OR) interfaces."""


def generate_model(features):
    """Generate a Pyomo ConcreteModel for Bonmin/Couenne.
    """
    try:
        from pyomo.environ import ConcreteModel
    except ImportError:
        raise ImportError(
            "The 'pyomo' package is required for the bonmin/couenne interface. "
            "Install it with: flp install pyomo  (or: pip install pyomo)"
        )

    from ...helpers._pyomo_types import register_pyomo_numeric_types
    register_pyomo_numeric_types()

    from ...helpers.solver_executables import (
        ensure_solver_on_path,
        inject_solver_paths,
        missing_solver_message,
    )
    from ...helpers.solver_provisioning import ensure_solver

    solver_name = None
    if features is not None:
        try:
            solver_name = features.get("solver_name")
        except AttributeError:
            try:
                solver_name = features["solver_name"]
            except Exception:
                pass

    if solver_name in ("bonmin", "couenne"):
        executable = ensure_solver(solver_name)
        if not executable:
            raise RuntimeError(missing_solver_message(solver_name))
        ensure_solver_on_path(executable)
    else:
        # No explicit solver chosen yet: just make installed binaries
        # visible to Pyomo (nothing is downloaded here).
        inject_solver_paths(("bonmin", "couenne"))

    model = ConcreteModel()
    return model
