# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Optional dependencies FelooPy can drive, installable by plain name.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Tuple

from .inspectors import (
    SOLVER_INTERFACES,
    _Namespace,
    _as_json,
    _dist_version,
    _print_table,
    normalize_interface,
    probe_interface,
)

__all__ = ["Dep", "DEPS", "DEP_GROUPS", "Resolution", "resolve_install", "cli_deps"]


class Dep(NamedTuple):
    """One optional dependency FelooPy can install by plain name."""

    names: Tuple[str, ...]         # every spelling that resolves to it
    requirements: Tuple[str, ...]  # pip requirements, at preferred versions
    group: str                     # key into DEP_GROUPS
    provides: str                  # one-line description for listings
    interfaces: Tuple[str, ...]    # interfaces re-probed after installing
    stock: bool                    # ships in feloopy[stock]


class Resolution(NamedTuple):
    """What ``flp install`` does with the names it was given."""

    requirements: List[str]      # handed to pip, pins applied
    matched: List[Tuple[str, Dep]]   # (name as typed, dependency)
    skipped: List[Tuple[str, str]]   # (name as typed, why pip is not used)


#: Display order of the groups, with the install hint printed above each.
DEP_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("stock", 'Stock solver stack - pip install "feloopy[stock]"'),
    ("exact", "Exact solvers - flp install <name>"),
    ("modeling", "Modelling layers - flp install <name>"),
    ("robust", "Robust / distributionally robust - flp install <name>"),
    ("heuristic", "Heuristic frameworks - flp install <name>"),
    ("commercial", "Commercial solvers - flp install <name> (licence needed)"),
    ("data", "Optional data & notebook features - flp install <name>"),
)

#: name -> dependency.  Populated by :func:`_register` below.
DEPS: Dict[str, Dep] = {}


def _register(names, requirements, group, provides, interfaces=(), stock=False):
    dep = Dep(tuple(names), tuple(requirements), group, provides, tuple(interfaces), stock)
    for name in names:
        DEPS[name] = dep


# --- stock (the nine in pyproject's ``stock`` extra) -----------------------
_register(("pyomo",), ["pyomo==6.10.1"],
          "stock", 'interface="pyomo"', ("pyomo",), stock=True)
_register(("highs", "highspy"), ["highspy==1.15.1"],
          "stock", 'interface="highs" (default solver)', ("highs",), stock=True)
_register(("scip", "pyscipopt"), ["pyscipopt==6.2.1"],
          "stock", 'interface="scip"', ("scip",), stock=True)
_register(("uno", "unopy"), ["unopy==0.4.14"],
          "stock", 'interface="uno" (universal NLP)', ("uno",), stock=True)
_register(("pydecision",), ["pydecision==5.1.1"],
          "stock", 'interface="pydecision" (MCDM)', ("pydecision",), stock=True)
_register(("mealpy",), ["mealpy==3.0.1"],
          "stock", 'interface="mealpy"', ("mealpy",), stock=True)
_register(("pymoo",), ["pymoo==0.6.1.3"],
          "stock", 'interface="pymoo"', ("pymoo",), stock=True)
_register(("ortools", "mathopt", "ortools_cp"), ["ortools==9.15.6755"],
          "stock", 'interface="mathopt" / "ortools_cp"',
          ("mathopt", "ortools_cp"), stock=True)
_register(("cvxpy",), ["cvxpy==1.6.4"],
          "stock", 'interface="cvxpy"', ("cvxpy",), stock=True)

# --- exact solvers ---------------------------------------------------------
_register(("pyoptinterface",), ["pyoptinterface==0.4.1", "highsbox==1.10.0"],
          "exact", "needs a backend (highs / gurobi / copt / mosek)",
          ("pyoptinterface",))
_register(("jump", "juliacall"), ["juliacall==0.9.24"],
          "exact", 'interface="jump" (JuMP, runs Julia)', ("jump",))
_register(("juliapkg",), ["juliapkg"],
          "exact", "Julia depot bootstrap (ships with juliacall)")
_register(("insideopt", "insideopt-seeker"), ["insideopt-seeker==0.1.21"],
          "exact", 'interface="insideopt"', ("insideopt",))
_register(("insideopt-demo",), ["insideopt-demo==0.3.3"],
          "exact", 'interface="insideopt-demo"', ("insideopt-demo",))

# --- modelling layers ------------------------------------------------------
_register(("pulp",), ["pulp[cbc]==4.0.0; python_version >= '3.12'",
                     "pulp==3.3.2; python_version < '3.12'"],
          "modeling", 'interface="pulp"', ("pulp",))
_register(("linopy",), ["linopy==0.5.2"],
          "modeling", 'interface="linopy"', ("linopy",))
_register(("mip",), ["mip==1.15.0"],
          "modeling", 'interface="mip" (CBC / Gurobi)', ("mip",))
_register(("picos",), ["picos==2.6.0"],
          "modeling", 'interface="picos"', ("picos",))
_register(("cylp",), ["cylp==0.92.2"],
          "modeling", 'interface="cylp" (native HiGHS/CBC)', ("cylp",))
_register(("pymprog",), ["pymprog==1.1.2"],
          "modeling", 'interface="pymprog"', ("pymprog",))
_register(("gekko",), ["gekko==1.3.2"],
          "modeling", 'interface="gekko"', ("gekko",))
_register(("casadi",), ["casadi==3.7.0"],
          "modeling", 'interface="casadi" (beta)', ("casadi",))

# --- robust / DRO ----------------------------------------------------------
_register(("rsome", "rsome_ro", "rsome_dro"), ["rsome==1.3.1"],
          "robust", 'interface="rsome_ro" / "rsome_dro"',
          ("rsome_ro", "rsome_dro"))

# --- heuristic frameworks --------------------------------------------------
_register(("niapy",), ["niapy==2.5.2"],
          "heuristic", 'interface="niapy"', ("niapy",))
_register(("pygad",), ["pygad==3.4.0"],
          "heuristic", 'interface="pygad"', ("pygad",))
_register(("indago",), ["indago==0.6.0"],
          "heuristic", 'interface="indago"', ("indago",))
_register(("pymultiobjective",), ["pymultiobjective==1.5.7"],
          "heuristic", 'interface="pymultiobjective"', ("pymultiobjective",))

# --- commercial solvers ----------------------------------------------------
_register(("gurobi", "gurobipy"), ["gurobipy==12.0.1"],
          "commercial", "free size-limited licence included", ("gurobi",))
_register(("cplex", "cplex_cp", "docplex"),
          ["cplex==22.1.2.0", "docplex==2.29.241"],
          "commercial", "community edition included", ("cplex", "cplex_cp"))
_register(("copt", "coptpy"), ["coptpy==7.2.6"],
          "commercial", "free academic licence", ("copt",))
_register(("xpress",), ["xpress==9.5.4"],
          "commercial", "free community licence", ("xpress",))
_register(("mosek",), ["mosek==11.2.4"],
          "commercial", "free size-limited licence", ("mosek",))
_register(("gams", "gamspy"), ["gamspy==1.26.4"],
          "commercial", "size-limited community licence", ("gams",))
_register(("hexaly",), ["hexaly==15.0.20260914"],
          "commercial", "free trial licence", ("hexaly",))

# --- optional data & notebook features ------------------------------------
_register(("shapely",), ["shapely"], "data", "geo sampling in data_handler")
_register(("geopandas",), ["geopandas"], "data", "country/region shapes")
_register(("osmnx",), ["osmnx"], "data", "OpenStreetMap network samples")
_register(("networkx",), ["networkx"], "data", "graph distances")
_register(("cloudpickle",), ["cloudpickle"], "data", "pickling of local classes")
_register(("ipython",), ["IPython"], "data", "notebook display & rich output")
_register(("xarray",), ["xarray"], "data", "DataArray values (ships with linopy)")


_PIN = re.compile(r"==\s*[^\s;]+")
_DIST = re.compile(r"[\[=<>!~;\s]")


def _unpin(requirement: str) -> str:
    """Drop ``==version`` from a requirement, keeping extras and markers."""
    return _PIN.sub("", requirement)


def _dist_of(requirement: str) -> str:
    """Distribution name of a requirement (``pulp[cbc]==4.0.0; x`` -> ``pulp``)."""
    return _DIST.split(requirement, 1)[0]


def _interface_of(name: str):
    """Interface key for ``name`` (``None`` for empty input or ``auto``)."""
    key = str(name or "").strip().lower()
    if not key or key == "auto":  # 'auto' is a dispatch mode, not a package
        return None
    return normalize_interface(key)


def _lookup(name: str):
    """``(key, Dep)`` for a name FelooPy can install, else ``(None, None)``."""
    key = str(name or "").strip().lower()
    if key in DEPS:
        return key, DEPS[key]
    interface = _interface_of(key)
    if interface and interface in DEPS:
        return interface, DEPS[interface]
    return None, None


def resolve_install(names, latest: bool = False) -> Resolution:
    """Map plain names onto pip requirements.

    Names FelooPy knows become its preferred (pinned) requirements - or the
    unpinned ones when ``latest`` is set - while everything else passes
    through to pip untouched.  Executables (``bonmin``, ``picat``, ...) are
    reported as skipped: pip cannot install them, ``flp setup`` can.
    """
    requirements: List[str] = []
    matched: List[Tuple[str, Dep]] = []
    skipped: List[Tuple[str, str]] = []

    for name in names:
        raw = str(name)
        if not raw.strip():
            continue
        key, dep = _lookup(raw)
        if dep is not None:
            matched.append((key, dep))
            for requirement in dep.requirements:
                requirement = _unpin(requirement) if latest else requirement
                if requirement not in requirements:
                    requirements.append(requirement)
            continue

        interface = _interface_of(raw)
        if interface and SOLVER_INTERFACES[interface].kind == "exe":
            skipped.append((raw, "not a Python package - use 'flp setup {}'".format(interface)))
            continue
        if str(raw).strip().lower() == "auto":
            skipped.append((raw, "'auto' is a dispatch mode for flp.search, not a package - see 'flp deps'"))
            continue

        if raw not in requirements:
            requirements.append(raw)

    return Resolution(requirements, matched, skipped)


def _dep_status(dep: Dep):
    """``(installed, version)`` for a dependency, probing real imports."""
    for interface in dep.interfaces:
        available, version = probe_interface(interface)
        if available:
            return True, version
    for requirement in dep.requirements:
        version = _dist_version(_dist_of(requirement))
        if version:
            return True, version
    return False, None


def _group_records(show_missing: bool = False) -> List[Tuple[str, str, List[dict]]]:
    grouped: List[Tuple[str, str, List[dict]]] = []
    for key, heading in DEP_GROUPS:
        records: List[dict] = []
        seen: set = set()
        for name in sorted((n for n, d in DEPS.items() if d.group == key), key=str.lower):
            dep = DEPS[name]
            if id(dep) in seen:
                continue
            seen.add(id(dep))
            installed, version = _dep_status(dep)
            records.append({
                "name": name,
                "names": list(dep.names),
                "requirements": list(dep.requirements),
                "group": key,
                "provides": dep.provides,
                "installed": installed,
                "version": version,
                "stock": dep.stock,
            })
        if show_missing:
            records = [r for r in records if not r["installed"]]
        grouped.append((key, heading, records))
    return grouped


def cli_deps(args=None):
    """List the optional dependencies FelooPy can drive, and how to get them."""
    args = args if args is not None else _Namespace()
    show_missing = getattr(args, "missing", False)
    as_json = getattr(args, "json", False)

    grouped = _group_records(show_missing)

    if as_json:
        _as_json([record for _key, _heading, records in grouped for record in records])
        return

    columns = ["Name", "Requirement", "Installed", "Status", "Provides"]
    total = 0
    total_installed = 0
    for _key, heading, records in grouped:
        if not records:
            continue
        print("\n{}".format(heading))
        rows = [
            [record["name"],
             ", ".join(record["requirements"]),
             record["version"] or "-",
             "installed" if record["installed"] else "missing",
             record["provides"]]
            for record in records
        ]
        _print_table(columns, rows, status_col=3)
        total += len(records)
        total_installed += sum(1 for record in records if record["installed"])

    print("\n{}/{} optional dependencies installed.".format(total_installed, total))
    print("Install any of them with: flp install <name>   "
          "(--latest for the newest releases instead of the pins)")
    print("Binaries (bonmin, couenne, picat, ...) come from: flp setup <name>")
