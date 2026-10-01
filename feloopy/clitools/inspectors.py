# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Environment introspection tools for the FelooPy command line.

Implements the ``solvers``, ``algorithms``, ``params``, and ``info``
subcommands::

    feloopy solvers                    # which solver interfaces can I use?
    feloopy solvers --missing          # what am I missing?
    feloopy algorithms -c heuristic    # list metaheuristics
    feloopy algorithms -i mealpy -s pso
    feloopy algorithms -c exact --available --json
    feloopy params highs highs      # options for flp.search(interface=...,
                                      # solver=..., options={})
    feloopy info                       # environment summary

Everything here is lazy: probing uses ``importlib.util.find_spec`` (so no
solver is actually imported/initialized) and the algorithm registries are
plain data tables.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from typing import NamedTuple, Optional

from .._version import __version__

__version_display__ = f"v{__version__}"

__all__ = [
    "SOLVER_INTERFACES",
    "CATEGORY_LABELS",
    "ALGORITHM_CATEGORIES",
    "probe_interface",
    "normalize_interface",
    "solvers_for",
    "cli_solvers",
    "cli_algorithms",
    "cli_params",
    "cli_info",
]


# ---------------------------------------------------------------------------
# Solver interface probes
# ---------------------------------------------------------------------------

class InterfaceProbe(NamedTuple):
    """How to detect a FelooPy interface without importing the solver."""

    category: str   # exact | modeling | uncertainty | cp | heuristic | mcdm
    kind: str       # "module" | "exe" | "builtin"
    target: str     # importable module, executable name, or ""
    dist: str       # pip distribution used for the version lookup ("" = none)
    note: str


#: Every interface ``flp.search(..., interface=...)`` can dispatch to.
SOLVER_INTERFACES = {
    # --- exact (LP / MILP / QP / NLP / MINLP) -------------------------------
    "highs":            InterfaceProbe("exact", "module", "highspy", "highspy", "bundled, default solver"),
    "gurobi":           InterfaceProbe("exact", "module", "gurobipy", "gurobipy", "commercial (free size-limited license)"),
    "cplex":            InterfaceProbe("exact", "module", "cplex", "cplex", "commercial (community edition)"),
    "copt":             InterfaceProbe("exact", "module", "coptpy", "coptpy", "commercial (free academic license)"),
    "xpress":           InterfaceProbe("exact", "module", "xpress", "xpress", "commercial (free community license)"),
    "mosek":            InterfaceProbe("exact", "module", "mosek", "mosek", "commercial (free size-limited license)"),
    "scip":             InterfaceProbe("exact", "module", "pyscipopt", "pyscipopt", "academic/non-commercial"),
    "uno":              InterfaceProbe("exact", "module", "unopy", "unopy", "universal NLP solver"),
    "bonmin":           InterfaceProbe("exact", "exe", "bonmin", "", "MINLP (COIN-OR), binaries via flp setup"),
    "couenne":          InterfaceProbe("exact", "exe", "couenne", "", "global MINLP (COIN-OR), binaries via flp setup"),
    "hexaly":           InterfaceProbe("exact", "module", "hexaly", "hexaly", "commercial (free trial license)"),
    "gams":             InterfaceProbe("exact", "module", "gamspy", "gamspy", "GAMS modeling system"),
    "jump":             InterfaceProbe("exact", "module", "juliacall", "juliacall", "JuMP, runs through Julia"),
    "mathopt":          InterfaceProbe("exact", "module", "ortools", "ortools", "Google MathOpt (via OR-Tools)"),
    "pyoptinterface":   InterfaceProbe("exact", "module", "pyoptinterface", "pyoptinterface", "needs a backend (highs/gurobi/copt/mosek)"),
    "insideopt":        InterfaceProbe("exact", "module", "seeker", "insideopt-seeker", "commercial (free demo license)"),
    "insideopt-demo":   InterfaceProbe("exact", "module", "seekerdemo", "insideopt-demo", "commercial (InsideOpt Seeker demo)"),

    # --- modeling layers / meta-solvers ------------------------------------
    "cvxpy":            InterfaceProbe("modeling", "module", "cvxpy", "cvxpy", "dispatches to an installed CVXPY solver"),
    "pyomo":            InterfaceProbe("modeling", "module", "pyomo", "pyomo", "dispatches to an installed Pyomo solver"),
    "linopy":           InterfaceProbe("modeling", "module", "linopy", "linopy", "xarray-based modeling layer"),
    "pulp":             InterfaceProbe("modeling", "module", "pulp", "pulp", "ships with CBC"),
    "mip":              InterfaceProbe("modeling", "module", "mip", "mip", "python-mip, CBC/Gurobi"),
    "picos":            InterfaceProbe("modeling", "module", "picos", "picos", "dispatches to an installed PICOS solver"),
    "cylp":             InterfaceProbe("modeling", "module", "cylp", "cylp", "CyLP, native HiGHS/CBC"),
    "pymprog":          InterfaceProbe("modeling", "module", "pymprog", "pymprog", "GPLK-based modeling"),
    "gekko":            InterfaceProbe("modeling", "module", "gekko", "gekko", "bundled APMonitor solvers"),
    "casadi":           InterfaceProbe("modeling", "module", "casadi", "casadi", "symbolic NLP framework (beta)"),

    # --- robust / distributionally robust ----------------------------------
    "rsome_ro":         InterfaceProbe("uncertainty", "module", "rsome", "rsome", "robust optimization"),
    "rsome_dro":        InterfaceProbe("uncertainty", "module", "rsome", "rsome", "distributionally robust optimization"),

    # --- constraint programming --------------------------------------------
    "ortools_cp":       InterfaceProbe("cp", "module", "ortools", "ortools", "Google OR-Tools CP-SAT"),
    "cplex_cp":         InterfaceProbe("cp", "module", "cplex", "cplex", "CP Optimizer"),
    "picat":            InterfaceProbe("cp", "exe", "picat", "", "needs the picat executable on PATH"),

    # --- heuristic / metaheuristic frameworks ------------------------------
    "mealpy":           InterfaceProbe("heuristic", "module", "mealpy", "mealpy", "100+ metaheuristics"),
    "niapy":            InterfaceProbe("heuristic", "module", "niapy", "niapy", "nature-inspired algorithms"),
    "pymoo":            InterfaceProbe("heuristic", "module", "pymoo", "pymoo", "multi-objective evolutionary"),
    "pygad":            InterfaceProbe("heuristic", "module", "pygad", "pygad", "genetic algorithm"),
    "indago":           InterfaceProbe("heuristic", "module", "indago", "indago", "gradient-free optimizers"),
    # NOTE: the importable package is ``pyMultiobjective`` (mixed case); the
    # PyPI distribution is ``pymultiobjective``.  The target must be the module.
    "pymultiobjective": InterfaceProbe("heuristic", "module", "pyMultiobjective", "pymultiobjective", "multi-objective solvers"),
    "scipy":            InterfaceProbe("heuristic", "builtin", "", "", "bundled (scipy.optimize)"),
    "feloopy":          InterfaceProbe("heuristic", "builtin", "", "", "bundled (FelooPy native heuristics)"),

    # --- multi-criteria decision making ------------------------------------
    # NOTE: the importable package is ``pyDecision`` (mixed case); the PyPI
    # distribution is ``pydecision``.  The target must be the module.
    "pydecision":       InterfaceProbe("mcdm", "module", "pyDecision", "pydecision", "AHP/TOPSIS/ELECTRE/PROMETHEE/..."),
}

#: Display order (and labels) of the categories above.
CATEGORY_LABELS = {
    "exact": "Exact solvers (LP / MILP / QP / NLP / MINLP)",
    "modeling": "Modeling layers & meta-solvers",
    "uncertainty": "Robust & distributionally robust (RSOME)",
    "cp": "Constraint programming",
    "heuristic": "Heuristic / metaheuristic frameworks",
    "mcdm": "Multi-criteria decision making (MCDM)",
}


def _find_module(name: str) -> bool:
    """True when ``name`` is importable (never imports it)."""
    if not name:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def _dist_version(dist: str) -> Optional[str]:
    if not dist:
        return None
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def normalize_interface(name) -> Optional[str]:
    """Map registry/alias spellings onto a :data:`SOLVER_INTERFACES` key."""
    if not name:
        return None
    key = str(name).strip().lower()
    if key in SOLVER_INTERFACES:
        return key
    if key.startswith("pyoptinterface."):
        return "pyoptinterface"
    if key in ("ortools-cp", "ortoolscp"):
        return "ortools_cp"
    if key in ("cplex-cp", "cplexcp"):
        return "cplex_cp"
    if key == "auto":
        return "highs"
    return None


def probe_interface(name) -> tuple:
    """Return ``(installed: bool | None, version: str | None)`` for an interface.

    ``installed`` is ``None`` when the interface is unknown to FelooPy.
    """
    key = normalize_interface(name)
    if key is None:
        return None, None
    probe = SOLVER_INTERFACES[key]
    if probe.kind == "builtin":
        return True, None
    if probe.kind == "exe":
        from ..helpers.solver_executables import find_solver_executable
        return find_solver_executable(probe.target) is not None, None
    available = _find_module(probe.target)
    version = _dist_version(probe.dist) if available else None
    return available, version


# ---------------------------------------------------------------------------
# Algorithm registries
# ---------------------------------------------------------------------------

#: ``(registry, label, layout)`` — ``layout`` says which entry field is the
#: interface/backend and which one is the algorithm name.
ALGORITHM_CATEGORIES = (
    ("exact",       "EXACT_ALGORITHMS",       "Exact solvers",                          "interface-first"),
    ("heuristic",   "HEURISTIC_ALGORITHMS",   "Heuristic / metaheuristic",              "interface-first"),
    ("moo",         "MOO_ALGORITHMS",         "Multi-objective evolutionary",           "interface-first"),
    ("uncertainty", "UNCERTAINTY_ALGORITHMS", "Robust / distributionally robust",       "interface-first"),
    ("cp",          "CONSTRAINT_ALGORITHMS",  "Constraint programming",                 "interface-first"),
    ("weighting",   "WEIGHTING_ALGORITHMS",   "MCDM weighting methods",                 "method-first"),
    ("ranking",     "RANKING_ALGORITHMS",     "MCDM ranking methods",                   "method-first"),
    ("special",     "SPECIAL_ALGORITHMS",     "MCDM special methods",                   "method-first"),
)


def _iter_algorithms():
    """Yield one record per registered algorithm (lazy, pure data)."""
    from ..helpers import registries as _reg

    for key, attr, label, layout in ALGORITHM_CATEGORIES:
        for entry in getattr(_reg, attr, []) or []:
            if len(entry) < 2:
                continue
            first, second = entry[0], entry[1]
            if layout == "interface-first":
                interface, algorithm = first, second
            else:
                algorithm, interface = first, second
            yield {
                "category": key,
                "category_label": label,
                "interface": interface,
                "algorithm": algorithm,
            }


def _algorithm_counts() -> dict:
    counts = {}
    for record in _iter_algorithms():
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    return counts


def _canon(name) -> str:
    """Case/hyphen-insensitive spelling used to pair interfaces with solvers."""
    return str(name or "").strip().lower().replace("-", "_")


def solvers_for(interface) -> list:
    """Solver/algorithm names that pair with ``interface``.

    Read from the same registries ``feloopy algorithms`` prints, so this is
    exactly the vocabulary ``flp.search(..., solver=...)`` accepts for that
    interface — the "other half" of the pair ``flp params`` needs.  An
    interface with no registry entry pairs with itself (``mosek`` ->
    ``mosek``), and ``pyoptinterface`` collects its per-backend variants.
    """
    want = _canon(interface)
    if not want:
        return []
    names = sorted({str(r["algorithm"]) for r in _iter_algorithms()
                    if _canon(r["interface"]) == want
                    or _canon(r["interface"]).startswith(want + ".")},
                   key=str.lower)
    if not names:
        key = normalize_interface(interface) or str(interface).strip()
        if key:
            names = [key]
    return names


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_plain_table(columns, rows, status_col=None):
    cells = [["" if c is None else str(c) for c in row] for row in rows]
    widths = [len(str(col)) for col in columns]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    header = "  ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    print(header)
    print("-" * len(header))
    for row in cells:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _print_table(columns, rows, status_col=None):
    """Render ``rows`` as a table (rich when available, plain text otherwise)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        _print_plain_table(columns, rows, status_col)
        return

    width = max(100, min(200, shutil.get_terminal_size((120, 40)).columns))
    console = Console(width=width)
    table = Table(header_style="bold cyan")
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        cells = ["" if c is None else str(c) for c in row]
        if status_col is not None and status_col < len(cells):
            if cells[status_col] == "installed":
                cells[status_col] = "[green]installed[/green]"
            elif cells[status_col] == "missing":
                cells[status_col] = "[red]missing[/red]"
        table.add_row(*cells)
    console.print(table)


def _as_json(payload):
    print(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# feloopy solvers
# ---------------------------------------------------------------------------

def cli_solvers(args=None):
    """Probe every solver interface and report what is installed."""
    args = args if args is not None else _Namespace()
    category = getattr(args, "category", None)
    show_installed = getattr(args, "installed", False)
    show_missing = getattr(args, "missing", False)
    as_json = getattr(args, "json", False)

    records = []
    for name, probe in SOLVER_INTERFACES.items():
        available, version = probe_interface(name)
        records.append({
            "interface": name,
            "category": probe.category,
            "installed": bool(available),
            "version": version,
            "package": probe.dist or ("executable: " + probe.target if probe.kind == "exe" else "built-in"),
            "note": probe.note,
        })

    total = len(records)
    total_installed = sum(1 for r in records if r["installed"])

    if category:
        records = [r for r in records if r["category"] == category]
    if show_installed and not show_missing:
        records = [r for r in records if r["installed"]]
    if show_missing and not show_installed:
        records = [r for r in records if not r["installed"]]

    if as_json:
        _as_json(records)
        return

    columns = ["Interface", "Status", "Version", "Package", "Note"]
    for cat, label in CATEGORY_LABELS.items():
        group = [r for r in records if r["category"] == cat]
        if not group:
            continue
        print(f"\n{label}")
        rows = [
            [r["interface"],
             "installed" if r["installed"] else "missing",
             r["version"] or "-",
             r["package"],
             r["note"]]
            for r in group
        ]
        _print_table(columns, rows, status_col=1)

    missing = total - total_installed
    print(f"\n{total_installed}/{total} interfaces installed.")
    if missing:
        print('Stock stack:  pip install "feloopy[stock]"')
        print("Anything else: flp install <name>   (flp deps lists them all)")


# ---------------------------------------------------------------------------
# feloopy algorithms
# ---------------------------------------------------------------------------

def cli_algorithms(args=None):
    """List/search the registered algorithms."""
    args = args if args is not None else _Namespace()
    category = getattr(args, "category", None)
    interface = getattr(args, "interface", None)
    search = getattr(args, "search", None)
    available_only = getattr(args, "available", False)
    as_json = getattr(args, "json", False)

    category_labels = {key: label for key, _attr, label, _layout in ALGORITHM_CATEGORIES}
    all_records = list(_iter_algorithms())
    counts = _algorithm_counts()

    records = []
    for record in all_records:
        installed, _version = probe_interface(record["interface"])
        record = dict(record)
        record["installed"] = installed
        records.append(record)

    if category:
        records = [r for r in records if r["category"] == category]
    if interface:
        wanted = interface.strip().lower()
        records = [r for r in records if wanted in (str(r["interface"]).lower(), str(r["algorithm"]).lower())]
    if search:
        needle = search.strip().lower()
        records = [
            r for r in records
            if needle in str(r["algorithm"]).lower() or needle in str(r["interface"]).lower()
        ]
    if available_only:
        records = [r for r in records if r["installed"] is not False]

    if as_json:
        _as_json(records)
        return

    if not records:
        print("No algorithms matched.")
        return

    rows = [
        [r["category"],
         r["interface"],
         r["algorithm"],
         "" if r["installed"] is None else ("installed" if r["installed"] else "missing")]
        for r in records
    ]
    _print_table(["Category", "Interface", "Algorithm", "Backend"], rows, status_col=3)

    print(f"\n{len(records)} of {len(all_records)} algorithms shown "
          f"({sum(counts.values())} registered across {len(counts)} categories).")
    if category:
        print(f"Available categories: {', '.join(sorted(category_labels))}")


# ---------------------------------------------------------------------------
# feloopy params
# ---------------------------------------------------------------------------

def cli_params(args=None):
    """Print the options passable to ``flp.search(..., options={})``.

    Needs the full interface + solver pair: which options are passable
    depends on both halves (``cvxpy`` behaves differently behind ``cbc``
    and behind ``osqp``; ``mealpy`` behind ``pso`` and behind ``de``),
    so an interface on its own is rejected.  The printed stdout is
    exactly ``flp.get_params(interface, solver)`` for that pair — nothing
    is added to it.
    """
    args = args if args is not None else _Namespace()
    interface = getattr(args, "interface", None)
    solver = getattr(args, "solver", None)
    as_json = getattr(args, "json", False)

    if not interface or not solver:
        print("Usage: flp params <interface> <solver>", file=sys.stderr)
        print("Both halves are needed — an interface on its own does not "
              "decide the options: the passable ones depend on the pair.",
              file=sys.stderr)
        print("Run 'feloopy solvers' for interfaces and "
              "'feloopy algorithms -i <interface>' for the solvers behind it.",
              file=sys.stderr)
        sys.exit(2)

    from ..helpers.solver_params import get_solver_params

    if normalize_interface(interface) is None:
        print(f"warning: '{interface}' is not a known FelooPy interface; "
              "showing the generic options. Run 'feloopy solvers' to list interfaces.",
              file=sys.stderr)

    choices = solvers_for(interface)
    if choices and _canon(solver) not in {_canon(c) for c in choices}:
        print(f"'{solver}' does not pair with interface '{interface}'.",
              file=sys.stderr)
        shown = " ".join(choices[:20])
        more = f" … (+{len(choices) - 20} more)" if len(choices) > 20 else ""
        print(f"Solvers for '{interface}': {shown}{more}", file=sys.stderr)
        if len(choices) > 20:
            print(f"Full list: feloopy algorithms -i {interface}", file=sys.stderr)
        sys.exit(1)

    try:
        options = get_solver_params(interface, solver)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error collecting options for '{interface}' + '{solver}': {exc}",
              file=sys.stderr)
        sys.exit(1)

    if not options and not getattr(options, "_sections", None):
        print(f"No curated options for interface '{interface}' + solver '{solver}'.",
              file=sys.stderr)
        known = ", ".join(sorted(SOLVER_INTERFACES))
        print(f"Known interfaces: {known}", file=sys.stderr)
        sys.exit(1)

    if as_json:
        comments = getattr(options, "_comments", {}) or {}
        _as_json({key: {"default": value, "comment": comments.get(key, "")}
                  for key, value in options.items()})
        return

    # ``CommentedDict.__str__`` is its ``to_code()`` snippet, so stdout is
    # byte-identical to ``print(flp.get_params(interface, solver))`` — no
    # extra header/trailer lines are appended.
    print(options)


# ---------------------------------------------------------------------------
# feloopy info
# ---------------------------------------------------------------------------

def cli_info(args=None):
    """Print a one-screen summary of the current environment."""
    args = args if args is not None else _Namespace()
    as_json = getattr(args, "json", False)

    installed = sorted(
        name for name in SOLVER_INTERFACES
        if (probe_interface(name)[0] is True)
    )
    missing = sorted(name for name in SOLVER_INTERFACES if name not in installed)
    counts = _algorithm_counts()

    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB"
    except Exception:
        ram = "-"

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payload = {
        "feloopy": __version_display__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "package_dir": package_dir,
        "ram": ram,
        "interfaces_installed": len(installed),
        "interfaces_total": len(SOLVER_INTERFACES),
        "algorithms": sum(counts.values()),
        "algorithms_by_category": counts,
        "installed_interfaces": installed,
        "missing_interfaces": missing,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    if as_json:
        _as_json(payload)
        return

    rows = [
        ["FelooPy", payload["feloopy"]],
        ["Python", f'{payload["python"]}  ({payload["executable"]})'],
        ["Platform", payload["platform"]],
        ["RAM", payload["ram"]],
        ["Package dir", payload["package_dir"]],
        ["Interfaces", f'{payload["interfaces_installed"]}/{payload["interfaces_total"]} installed'],
        ["Algorithms", f'{payload["algorithms"]} registered'],
    ]
    _print_table(["", "Value"], rows)

    print("\nInstalled interfaces:")
    for name in installed:
        print(f"  - {name}")
    if missing:
        print("\nMissing interfaces:")
        print("  " + ", ".join(missing))


class _Namespace(dict):
    """Minimal stand-in so the ``cli_*`` functions work with no arguments."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
