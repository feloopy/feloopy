# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""``flp setup`` — install free solvers, guide commercial ones.
"""

from __future__ import annotations

import sys
from typing import List

from ..helpers.solver_executables import paths_file, rebuild_paths, solvers_home
from ..helpers.solver_guides import (
    commercial_names,
    install_guide,
    resolve_commercial,
)
from ..helpers.solver_provisioning import (
    SETUP_PIP_SOLVERS,
    SETUP_SOLVERS,
    SetupError,
    is_provisionable,
    provision_solver,
    resolve_setup_name,
    setup_status,
)

__all__ = ["cli_setup"]


def _print_status() -> None:
    from .inspectors import _print_table

    rows = setup_status()
    _print_table(
        ["solver", "type", "status", "version", "path", "note"],
        [[row["name"],
          "pip" if row.get("kind") == "pip" else "binary",
          "installed" if row["installed"] else "missing",
          row["version"] or "",
          row["exe"] or "",
          row["note"]] for row in rows],
        status_col=2,
    )
    print()
    print("Solvers folder: {}".format(solvers_home()))
    print("Path file:      {}".format(paths_file()))
    print("Free binaries:  flp setup <name>   (e.g. flp setup bonmin couenne cbc)")
    print("Free bindings:  flp setup <name>   (e.g. flp setup highspy pyscipopt cvxopt)")
    print("Commercial:     flp setup <name>   (e.g. flp setup gurobi — prints its install guide)")


def cli_setup(args) -> None:
    """Install free solvers, print guides for commercial names, show status."""
    solvers = [str(s).lower().strip() for s in (getattr(args, "solvers", None) or [])]
    force = bool(getattr(args, "force", False))
    show_list = bool(getattr(args, "list", False))

    if not solvers or show_list:
        rebuild_paths()
        _print_status()
        return

    # Interface spellings first (coin -> cbc, glpsol -> glpk, ...), then
    # split into "install this" and "print this solver's guide".
    unknown = [s for s in solvers
               if not is_provisionable(resolve_setup_name(s))
               and not resolve_commercial(s)]
    if unknown:
        sys.exit(
            "error: unknown solver(s): {}\n"
            "free (installable): {}\n"
            "commercial (install guide): {}".format(
                ", ".join(unknown),
                ", ".join(list(SETUP_SOLVERS) + list(SETUP_PIP_SOLVERS)),
                ", ".join(commercial_names()),
            )
        )

    to_install: List[str] = []
    guides: List[str] = []
    for raw in solvers:
        name = resolve_setup_name(raw)
        if resolve_commercial(raw) or resolve_commercial(name):
            canonical = resolve_commercial(raw) or resolve_commercial(name)
            if canonical not in guides:
                guides.append(canonical)
        elif name not in to_install:
            to_install.append(name)

    failed: List[str] = []
    # The verified tarballs live in <Solvers>/.cache, so the overlapping
    # dependency closure of `flp setup bonmin couenne` downloads once.
    for name in to_install:
        try:
            provision_solver(name, force=force)
        except SetupError as exc:
            print("{}: FAILED — {}".format(name, exc), flush=True)
            failed.append(name)

    for name in guides:
        print(install_guide(name))

    if to_install:
        print()
        _print_status()
    if failed:
        sys.exit(1)
