# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Install guides for commercial solvers (``flp setup <commercial>``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = [
    "COMMERCIAL_GUIDES",
    "COMMERCIAL_ALIASES",
    "resolve_commercial",
    "is_commercial",
    "commercial_names",
    "install_guide",
]


COMMERCIAL_GUIDES: Dict[str, dict] = {
    "gurobi": {
        "title": "Gurobi",
        "summary": "commercial LP/MILP/QP/NLP optimizer (Gurobi Optimizer)",
        "pip": ["gurobipy"],
        "license": [
            "A size-limited license ships inside gurobipy: small models (up to",
            "2000 variables / 2000 constraints) run immediately, no setup needed.",
            "Academic, free full license:  https://www.gurobi.com/features/academic-named-user-license",
            "Commercial / WLS licenses:    https://www.gurobi.com/features/wls-license-management",
            "Point feloopy at a license file (optional):",
            "  set GRB_LICENSE_FILE=C:\\path\\to\\gurobi.lic        (Windows)",
            "  export GRB_LICENSE_FILE=/path/to/gurobi.lic        (macOS/Linux)",
        ],
        "verify": "python -c \"import gurobipy; gurobipy.Model()\"",
        "use": "interface=\"gurobi\", solver=\"gurobi\"   (also via pyomo / pulp / picos / ortools / rsome)",
        "docs": "https://docs.gurobi.com",
    },
    "cplex": {
        "title": "IBM CPLEX",
        "summary": "commercial LP/MILP/QP optimizer (IBM ILOG CPLEX Optimization Studio)",
        "pip": ["cplex", "docplex"],
        "license": [
            "The pip package runs IBM's Community Edition: small models (up to",
            "1000 variables / 1000 constraints) work with no extra license steps.",
            "Academic (free) and commercial licenses: https://www.ibm.com/products/ilog-cplex-optimization-studio",
            "  (students/faculty get it free through the IBM Academic Initiative).",
        ],
        "verify": "python -c \"import cplex; cplex.Cplex()\"",
        "use": "interface=\"cplex\", solver=\"cplex\"   (also via pyomo / pulp / picos / ortools / rsome)",
        "docs": "https://www.ibm.com/docs/en/icos",
    },
    "xpress": {
        "title": "FICO Xpress",
        "summary": "commercial LP/MILP optimizer (FICO Xpress Optimization Suite)",
        "pip": ["xpress"],
        "license": [
            "Register for FICO's free Xpress Community License (size-limited):",
            "  https://community.fico.com   -> search \"Xpress Community License\"",
            "Academic and commercial licenses: https://www.fico.com/en/products/fico-xpress-optimization",
            "Activate the pip package with the license key FICO sends you",
            "(steps are in the Xpress Python manual on the product page).",
        ],
        "verify": "python -c \"import xpress\"",
        "use": "interface=\"xpress\", solver=\"xpress\"   (also via pyomo / pulp / ortools)",
        "docs": "https://pypi.org/project/xpress/",
    },
    "mosek": {
        "title": "MOSEK",
        "summary": "commercial LP/SOCP/SDP optimizer (MOSEK)",
        "pip": ["mosek"],
        "license": [
            "Free 30-day trial:                 https://mosek.com/trial",
            "Free academic license (renewable): https://mosek.com/products/academic-licenses",
            "Set the received license file (optional if it lives in ~/.mosek):",
            "  set MOSEKLM_LICENSE_FILE=C:\\path\\to\\mosek.lic     (Windows)",
            "  export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic     (macOS/Linux)",
        ],
        "verify": "python -c \"import mosek\"",
        "use": "interface=\"mosek\", solver=\"mosek\"   (also via pyomo / pulp / picos / rsome)",
        "docs": "https://docs.mosek.com",
    },
    "copt": {
        "title": "COPT",
        "summary": "commercial LP/MILP/QP/SOCP optimizer (COPT, Shanghai Shanshu)",
        "pip": ["coptpy"],
        "license": [
            "Free trial and academic licenses:  https://www.shanshu.ai/copt",
            "Register, receive a copt.lic file, then point COPT at its folder:",
            "  set COPT_LICENSEDIR=C:\\path\\to\\license            (Windows)",
            "  export COPT_LICENSEDIR=/path/to/license            (macOS/Linux)",
        ],
        "verify": "python -c \"import coptpy\"",
        "use": "interface=\"copt\", solver=\"copt\"   (also via rsome)",
        "docs": "https://www.shanshu.ai/copt",
    },
    "gams": {
        "title": "GAMS",
        "summary": "commercial modeling system with bundled solvers (via the gamspy API)",
        "pip": ["gamspy"],
        "license": [
            "gamspy ships the GAMS system with a free size-limited Community",
            "License: small models run out of the box, no license file needed.",
            "Academic (free) and commercial licenses: https://www.gams.com -> Licensing",
        ],
        "verify": "python -c \"import gamspy\"",
        "use": "interface=\"gams\"   (also as a backend for interface=\"pyomo\", solver=\"gams\")",
        "docs": "https://gamspy.readthedocs.io",
    },
    "hexaly": {
        "title": "Hexaly",
        "summary": "commercial solver for LP/MILP/nonlinear problems (Hexaly)",
        "pip": ["hexaly"],
        "license": [
            "Free trial download:            https://www.hexaly.com/download",
            "Academic and commercial licenses: https://www.hexaly.com",
            "Follow Hexaly's activation steps with the license key they e-mail you.",
        ],
        "verify": "python -c \"import hexaly\"",
        "use": "interface=\"hexaly\", solver=\"hexaly\"",
        "docs": "https://www.hexaly.com",
    },
    "baron": {
        "title": "BARON",
        "summary": "global solver for nonconvex problems (BARON) - pyomo backend",
        "pip": [],
        "download": [
            "Free academic registration and downloads: https://minlp.com/baron",
            "Download the BARON executable for your OS, then either:",
            "  - drop it into {solvers} (any subfolder; feloopy finds it), or",
            "  - set FELOOPY_BARON_PATH to the executable.",
        ],
        "license": [
            "Free for academic users (register with an academic address at",
            "minlp.com/baron); commercial licenses are sold on the same page.",
        ],
        "verify": "flp setup --list     (BARON shows up once its folder is scanned)",
        "use": "interface=\"pyomo\", solver=\"baron\"",
        "docs": "https://minlp.com/baron",
    },
    "conopt": {
        "title": "CONOPT",
        "summary": "commercial NLP solver (CONOPT) - pyomo backend",
        "pip": [],
        "download": [
            "Free trial download: https://www.conopt.com (register, pick your OS),",
            "then drop the executable into {solvers} (any subfolder) or set",
            "FELOOPY_CONOPT_PATH to it.",
        ],
        "license": [
            "Free trial license from conopt.com; academic and commercial",
            "licenses are sold there too.",
            "CONOPT also ships inside GAMS - flp setup gams gives you a GAMS route.",
        ],
        "verify": "flp setup --list     (CONOPT shows up once its folder is scanned)",
        "use": "interface=\"pyomo\", solver=\"conopt\"",
        "docs": "https://www.conopt.com",
    },
    "knitro": {
        "title": "Artelys Knitro",
        "summary": "commercial nonlinear optimizer (Knitro)",
        "pip": [],
        "download": [
            "Free 30-day trial and downloads: https://www.artelys.com/app/knitro/",
            "Install the Knitro runtime for your OS, then put its bin directory on",
            "PATH or drop it into {solvers}.",
        ],
        "license": [
            "Trial license from artelys.com; academic licensing on request.",
        ],
        "verify": "flp setup --list     (once its folder is scanned)",
        "use": "no dedicated interface yet - use through interface=\"gams\" or interface=\"jump\"",
        "docs": "https://www.artelys.com/app/knitro/",
    },
    "insideopt": {
        "title": "InsideOpt Seeker",
        "summary": "commercial global/linear optimizer (InsideOpt Seeker)",
        "pip": ["insideopt-seeker"],
        "license": [
            "Seeker is licensed per machine: its client_machine.sio file goes",
            "to InsideOpt, who send the licence back - buy a real licence or",
            "request a free demo licence at info@insideopt.com.",
            "Install steps: https://insideopt.com/pages/install-insideopt-seeker",
        ],
        "verify": "python -c \"import seeker\"",
        "use": "interface=\"insideopt\", solver=\"seeker\"",
        "docs": "https://insideopt.com",
    },
    "insideopt-demo": {
        "title": "InsideOpt Seeker (demo)",
        "summary": "demo distribution of the commercial InsideOpt Seeker optimizer",
        "pip": ["insideopt-demo"],
        "license": [
            "Demo build of Seeker, installed straight from pip.",
            "Licences for Seeker (free demo or commercial) are issued by",
            "InsideOpt on request: info@insideopt.com",
            "Install steps: https://insideopt.com/pages/install-insideopt-seeker",
        ],
        "verify": "python -c \"import seekerdemo\"",
        "use": "interface=\"insideopt-demo\", solver=\"seeker\"",
        "docs": "https://insideopt.com",
    },
}

#: Spellings feloopy interfaces use that map onto a canonical guide.
COMMERCIAL_ALIASES: Dict[str, str] = {
    "gurobi": "gurobi",
    "gurobi-cmd": "gurobi",
    "gurobi_cmd": "gurobi",
    "gurobi_direct": "gurobi",
    "gurobi-direct": "gurobi",
    "gurobi_persistent": "gurobi",
    "gurobi-persistent": "gurobi",
    "cplex": "cplex",
    "cplex-py": "cplex",
    "cplex_py": "cplex",
    "cplex_direct": "cplex",
    "cplex-direct": "cplex",
    "cplex_persistent": "cplex",
    "cplex-persistent": "cplex",
    "xpress": "xpress",
    "xpress-py": "xpress",
    "xpress_py": "xpress",
    "xpress_direct": "xpress",
    "xpress-direct": "xpress",
    "xpress_persistent": "xpress",
    "xpress-persistent": "xpress",
    "mosek": "mosek",
    "mosek_direct": "mosek",
    "mosek-direct": "mosek",
    "mosek_persistent": "mosek",
    "mosek-persistent": "mosek",
    "mskfsn": "mosek",
    "copt": "copt",
    "gams": "gams",
    "gamspy": "gams",
    "hexaly": "hexaly",
    "baron": "baron",
    "conopt": "conopt",
    "knitro": "knitro",
    "insideopt": "insideopt",
    "insideopt-seeker": "insideopt",
    "seeker": "insideopt",
    "insideopt-demo": "insideopt-demo",
    "seekerdemo": "insideopt-demo",
}


def resolve_commercial(name: Optional[str]) -> Optional[str]:
    """Canonical guide name for a commercial solver spelling, or ``None``."""
    if not name:
        return None
    return COMMERCIAL_ALIASES.get(str(name).strip().lower())


def is_commercial(name: Optional[str]) -> bool:
    """True when ``name`` is a commercial solver with an install guide."""
    return resolve_commercial(name) is not None


def commercial_names() -> List[str]:
    """Canonical commercial solver names (sorted, for listings/errors)."""
    return sorted(COMMERCIAL_GUIDES)


def install_guide(name: str) -> str:
    """Render the official install guide for ``name`` ('' when unknown).

    The guide is compact but complete: the binding to install (or the
    official download page), the licensing routes, a verification
    command, and how the solver is addressed in feloopy.
    """
    key = resolve_commercial(name)
    if key is None:
        return ""
    guide = COMMERCIAL_GUIDES[key]
    rule = "-" * 72
    home = _solvers_home()
    step = 1
    out: List[str] = [rule, "{} - {}".format(guide["title"], guide["summary"]), rule]

    if guide.get("pip"):
        out.append("{}. Install the Python binding:".format(step))
        step += 1
        out.append("      pip install {}".format(" ".join(guide["pip"])))
        out.append("      or, pinned to FelooPy's preferred version:  flp install {}".format(key))
    elif guide.get("download"):
        out.append("{}. Install:".format(step))
        step += 1
        for line in guide["download"]:
            out.append("      " + line.replace("{solvers}", home))

    out.append("{}. License:".format(step))
    step += 1
    for line in guide.get("license", []):
        out.append("      " + line)

    out.append("{}. Verify:".format(step))
    step += 1
    out.append("      " + guide["verify"])

    out.append("{}. Use in feloopy:".format(step))
    step += 1
    out.append("      " + guide["use"])

    out.append("Docs: {}".format(guide["docs"]))
    out.append(rule)
    return "\n".join(out)


def _solvers_home() -> str:
    # Imported lazily: solver_executables imports this module's helpers
    # from its own error messages, so keep the module-level graph acyclic.
    from .solver_executables import solvers_home
    return str(solvers_home())
