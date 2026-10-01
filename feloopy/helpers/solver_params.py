# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""
Solver-specific options for ``flp.search(..., options={})``.

``get_solver_params(interface, solver)`` returns a fillable, commented
dictionary that can be passed via the ``options`` kwarg to ``flp.search``.

The return value is a ``CommentedDict`` – a ``dict`` subclass whose
``repr``/``str`` renders as a copy-pasteable Python snippet with
``# comments``.  It is still a plain ``dict`` so
``flp.search(..., options=flp.get_params(...))`` works reliably.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# CommentedDict – dict that prints as commented code
# ---------------------------------------------------------------------------
class CommentedDict(dict):
    """Dict that renders as a commented Python snippet.

    Example
    -------
    >>> opts = flp.get_params("highs")
    >>> print(opts)          # pretty, commented code
    >>> flp.search(model, options=dict(opts))  # works as plain dict
    """

    def __init__(self, *args, comments=None, sections=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._comments = comments or {}
        # sections: list of (title, [keys in order])
        self._sections = sections or []

    def to_code(self) -> str:
        lines = ["options = {"]
        # Build a map from key -> section for ordering
        seen = set()
        for title, keys in self._sections:
            # Informational section with no keys – just print header as comment
            if not keys:
                lines.append(f"    # ── {title} ──")
                lines.append("")
                continue
            # Filter to keys that actually exist in dict
            present = [k for k in keys if k in self]
            if not present:
                continue
            lines.append(f"    # ── {title} ──")
            for k in present:
                v = self[k]
                comment = self._comments.get(k, "")
                val_repr = repr(v)
                if comment:
                    lines.append(f'    "{k}": {val_repr},  # {comment}')
                else:
                    lines.append(f'    "{k}": {val_repr},')
                seen.add(k)
            lines.append("")
        # Any remaining keys not in sections (dynamic params)
        remaining = [k for k in self.keys() if k not in seen]
        if remaining:
            lines.append(f"    # ── Solver-specific ──")
            for k in remaining:
                v = self[k]
                comment = self._comments.get(k, "")
                val_repr = repr(v)
                if comment:
                    lines.append(f'    "{k}": {val_repr},  # {comment}')
                else:
                    lines.append(f'    "{k}": {val_repr},')
            lines.append("")
        lines.append("}")
        lines.append("# Pass as: flp.search(model, interface='...', solver='...', options=options)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_code()

    def __str__(self) -> str:
        return self.to_code()


# ---------------------------------------------------------------------------
# Common feloopy-level defaults (also usable as top-level kwargs)
# ---------------------------------------------------------------------------
_COMMON_DEFAULTS = {
    "time_limit": (None, "Time limit in seconds (float). Also as flp.search(time_limit=...)."),
    "cpu_threads": (None, "Number of CPU threads (int). Also as flp.search(cpu_threads=...)."),
    "absolute_gap": (None, "Absolute MIP gap (float). Also as flp.search(absolute_gap=...)."),
    "relative_gap": (None, "Relative MIP gap (float). Also as flp.search(relative_gap=...)."),
    "verbose": (False, "Verbose solver log (bool). Also as flp.search(verbose=True)."),
}

_DECOMPOSITION_DEFAULTS = {
    "max_iterations": (100, "Maximum decomposition iterations (int). Benders/Lagrangian/CG."),
    "tolerance": (1e-6, "Convergence tolerance (float). Benders/Lagrangian/CG."),
    "benders_method": (None, "Benders method: 'classic' | 'logic_based' | 'hybrid' (str)."),
    "cut_strategy": (None, "Benders cut strategy (str)."),
    "acceleration": (None, "Benders acceleration (str)."),
    "complicating_variables": (None, "List of complicating variable names for Benders (list[str])."),
    "relax_subproblem_integrality": (None, "Relax subproblem integrality (bool)."),
    "lag_step_size": (None, "Lagrangian step-size rule (str)."),
    "lag_volume_weight": (0.5, "Lagrangian volume weight (float)."),
    "cg_variant": (None, "Column-generation variant: 'dw' | 'dw2' | 'dcg' | 'ccg' (str)."),
    "cg_stabilization": (True, "Use dual stabilization for CG (bool)."),
    "linking_constraints": (None, "Linking constraint names for DW/DW2 (list[str])."),
    "pricing_strategy": (None, "Pricing strategy for CG (str)."),
    "relaxation_strategy": (None, "Relaxation strategy (str)."),
}

# Unified heuristic defaults
_HEURISTIC_UNIFIED_DEFAULTS = {
    "pop_size": (50, "Population / swarm size (int)."),
    "epoch": (100, "Number of generations / iterations (int)."),
    "max_evaluations": (None, "Maximum number of function evaluations (int). Overrides epoch if both given."),
    "seed": (None, "Random seed for reproducibility (int)."),
    "verbose": (False, "Show solver convergence log (bool)."),
    "penalty_coefficient": (0, "Penalty for constraint violation (float, 0=auto)."),
}

# Curated solver-specific defaults (fallback)
_CURATED_DEFAULTS = {
    "highs": {
        "presolve": ("choose", "Presolve: 'choose' | 'on' | 'off'"),
        "solver": ("choose", "LP solver: 'choose' | 'simplex' | 'ipm' | 'pdlp'"),
        "parallel": ("choose", "Parallel strategy: 'choose' | 'on' | 'off'"),
        "time_limit": (float("inf"), "Time limit (double)"),
        "threads": (0, "Thread count (0=auto)"),
        "mip_rel_gap": (1e-4, "Relative MIP gap"),
        "mip_abs_gap": (1e-6, "Absolute MIP gap"),
        "output_flag": (True, "Enable solver output (bool)"),
        "random_seed": (0, "Random seed"),
        "simplex_strategy": (1, "Simplex strategy"),
        "simplex_scale_strategy": (2, "Simplex scale strategy"),
        "ipm_optimality_tolerance": (1e-8, "IPM optimality tolerance"),
        "mip_heuristic_effort": (0.05, "MIP heuristic effort"),
        "mip_max_nodes": (10000000, "MIP max nodes"),
    },
    "gurobi": {
        "TimeLimit": (float("inf"), "Time limit (seconds)"),
        "Threads": (0, "Number of threads"),
        "MIPGap": (1e-4, "Relative MIP gap"),
        "MIPGapAbs": (1e-10, "Absolute MIP gap"),
        "OutputFlag": (1, "Enable output (0/1)"),
        "LogFile": ("", "Log file path"),
        "LogToConsole": (1, "Log to console (0/1)"),
        "Method": (-1, "Algorithm: -1=auto, 0=primal, 1=dual, 2=barrier, 3=concurrent, etc."),
        "Heuristics": (0.05, "Heuristic effort (0..1)"),
        "MIPFocus": (0, "MIP focus: 0=balanced, 1=feasibility, 2=optimality, 3=bound"),
        "Presolve": (-1, "Presolve: -1=auto, 0=off, 1=conservative, 2=aggressive"),
        "Cuts": (-1, "Cut generation: -1=auto, 0=off, 1=moderate, 2=aggressive, 3=very aggressive"),
        "Seed": (0, "Random seed"),
    },
    "copt": {
        "TimeLimit": (1e20, "Time limit"),
        "Threads": (-1, "Thread count (-1=auto)"),
        "RelGap": (1e-4, "Relative gap"),
        "AbsGap": (1e-6, "Absolute gap"),
        "Logging": (1, "Enable logging (0/1)"),
        "LogToConsole": (1, "Log to console (0/1)"),
        "FeasTol": (1e-6, "Feasibility tolerance"),
        "DualTol": (1e-7, "Dual tolerance"),
        "IntTol": (1e-6, "Integrality tolerance"),
        "LpMethod": (0, "LP method"),
        "CutLevel": (1, "Cut level"),
        "HeurLevel": (1, "Heuristic level"),
        "Presolve": (-1, "Presolve: -1=auto, 0=off"),
    },
    "cplex": {
        "timelimit": (1e75, "Time limit"),
        "threads": (0, "Thread count"),
        "mip.tolerances.mipgap": (1e-4, "Relative MIP gap"),
        "mip.tolerances.absmipgap": (1e-6, "Absolute MIP gap"),
        "mip.tolerances.integrality": (1e-5, "Integrality tolerance"),
        "emphasis.mip": (0, "MIP emphasis: 0=balanced, 1=feasibility, 2=optimality..."),
        "preprocessing.presolve": (1, "Presolve (0/1)"),
        "randomseed": (0, "Random seed"),
    },
    "xpress": {
        "MAXTIME": (0, "Time limit (negative = no limit)"),
        "THREADS": (0, "Number of threads"),
        "MIPRELSTOP": (1e-4, "Relative MIP gap"),
        "MIPABSSTOP": (1e-6, "Absolute MIP gap"),
        "PRESOLVE": (1, "Presolve 0=off, 1=on"),
        "MIPLOG": (1, "MIP log level"),
    },
    "mosek": {
        "MSK_DPAR_MIO_TOL_REL_GAP": (1e-4, "Relative MIP gap"),
        "MSK_DPAR_MIO_TOL_ABS_GAP": (1e-6, "Absolute MIP gap"),
        "MSK_IPAR_NUM_THREADS": (0, "Number of threads"),
        "MSK_DPAR_OPTIMIZER_MAX_TIME": (-1.0, "Time limit (-1=no limit)"),
        "MSK_IPAR_LOG": (1, "Log level"),
    },
    "ortools": {
        "relative_mip_gap": (1e-4, "Relative MIP gap (MPSolverParameters)"),
        "primal_tolerance": (1e-7, "Primal tolerance"),
        "dual_tolerance": (1e-7, "Dual tolerance"),
        "presolve": (1, "Presolve: 0=off, 1=on"),
        "lp_algorithm": (0, "LP algorithm: 0=primal, 1=dual, 2=barrier"),
        "scaling": (1, "Scaling: 0=off, 1=on"),
    },
    "scip": {
        "limits/time": (1e20, "Time limit (seconds)"),
        "limits/gap": (0.0, "Relative gap"),
        "limits/absgap": (0.0, "Absolute gap"),
        "display/verblevel": (4, "Verbosity: 0=off, 4=default, 5=verbose"),
        "presolving/maxrounds": (-1, "Presolving max rounds (-1=unlimited, 0=off)"),
        "lp/initalgorithm": ("s", "Initial LP algorithm: 's'=simplex, 'i'=interior"),
        "numerics/feastol": (1e-6, "Feasibility tolerance"),
        "randomization/randomseedshift": (0, "Random seed shift"),
    },
    "cvxpy": {
        "verbose": (False, "Verbose output"),
        "max_iters": (1000, "Maximum iterations"),
        "eps": (1e-4, "Convergence tolerance"),
        "warm_start": (False, "Warm start"),
    },
    "pyomo": {
        "tee": (False, "Stream solver output to stdout"),
        "keepfiles": (False, "Keep temporary solver files"),
        "load_solutions": (True, "Load solutions"),
    },
    "jump": {
        "time_limit": (None, "Time limit via MOI.TimeLimitSec()"),
        "silent": (False, "Silent mode"),
        "mip_gap": (1e-4, "Relative MIP gap via MOI.RelativeGapTolerance()"),
    },
    "pulp": {
        "timeLimit": (None, "Time limit (seconds)"),
        "threads": (0, "Thread count"),
        "gapRel": (1e-4, "Relative MIP gap"),
        "gapAbs": (1e-6, "Absolute MIP gap"),
        "msg": (False, "Verbose output"),
    },
    "mip": {
        "max_seconds": (float("inf"), "Time limit"),
        "threads": (0, "Thread count"),
        "max_nodes": (-1, "Maximum B&B nodes (-1=unlimited)"),
    },
    "picos": {
        "verbose": (0, "Verbose output"),
        "tol": (1e-6, "Tolerance"),
    },
    "ortools_cp": {
        "max_time_in_seconds": (None, "Time limit"),
        "num_search_workers": (8, "Number of workers"),
        "log_search_progress": (False, "Log progress"),
        "cp_model_presolve": (True, "Presolve"),
    },
    "hexaly": {
        "timeLimit": (None, "Time limit (seconds)"),
        "nbThreads": (0, "Number of threads"),
        "seed": (0, "Random seed"),
    },
    "rsome_ro": {
        "solver": ("copt", "Underlying solver: copt | cplex | gurobi | mosek etc."),
        "verbose": (False, "Verbose output"),
    },
    "rsome_dro": {
        "solver": ("copt", "Underlying solver"),
        "verbose": (False, "Verbose output"),
    },
    "generic": {
        "verbose": (False, "Verbose output"),
    },
}

_INTERFACE_MAP = {
    "highs": "highs",
    "pyoptinterface.highs": "highs",
    "pyoptinterface": "highs",
    "mathopt": "highs",
    "gurobi": "gurobi",
    "pyoptinterface.gurobi": "gurobi",
    "copt": "copt",
    "pyoptinterface.copt": "copt",
    "cplex": "cplex",
    "xpress": "xpress",
    "mosek": "mosek",
    "pyoptinterface.mosek": "mosek",
    "ortools": "ortools",
    "scip": "scip",
    "cvxpy": "cvxpy",
    "linopy": "cvxpy",
    "pyomo": "pyomo",
    "jump": "jump",
    "pulp": "pulp",
    "mip": "mip",
    "picos": "picos",
    "ortools_cp": "ortools_cp",
    "cplex_cp": "ortools_cp",
    "hexaly": "hexaly",
    "uno": "scip",
    "bonmin": "scip",
    "couenne": "scip",
    "casadi": "generic",
    "gekko": "generic",
    "gamspy": "generic",
    "rsome_ro": "rsome_ro",
    "rsome_dro": "rsome_dro",
    "cylp": "generic",
    "seeker": "generic",
    "insideopt": "generic",
    "insideopt-demo": "generic",
    "gams": "generic",
    "picat": "generic",
    "pymprog": "generic",
    "auto": "highs",
}

_HEURISTIC_IFACES = {"mealpy", "niapy", "indago", "pygad", "pymoo", "pymultiobjective", "feloopy", "scipy"}
_CONSTRAINT_IFACES = {"ortools_cp", "cplex_cp", "cplex-cp", "ortools-cp"}
_UNCERTAIN_IFACES = {"rsome_ro", "rsome_dro"}
_MADM_IFACES = {"pydecision"}
_META_IFACES = {"cvxpy", "linopy", "pyomo", "jump", "gams", "pulp", "mip", "picos", "mathopt", "cylp", "pyoptinterface", "auto"}

def _get_highs_params_with_defaults():
    try:
        import highspy
        h = highspy.Highs()
        opts = h.getOptions()
        data = {}
        comments = {}
        _type_map = {"kBool": "bool", "kInt": "int", "kDouble": "double", "kString": "string"}
        for attr in dir(opts):
            if attr.startswith("_"):
                continue
            try:
                status, val = h.getOptionValue(attr)
                if "kOk" not in str(status):
                    continue
                typ = h.getOptionType(attr)
                typ_str = str(typ)
                pretty = typ_str
                for k, v in _type_map.items():
                    if k in typ_str:
                        pretty = v
                        break
                data[attr] = val
                comments[attr] = f"{pretty}, default={val!r}"
            except Exception:
                continue
        if data:
            return data, comments
    except Exception:
        pass
    # fallback
    fallback = _CURATED_DEFAULTS.get("highs", {})
    data = {k: v for k, (v, _) in fallback.items()}
    comments = {k: c for k, (_, c) in fallback.items()}
    return data, comments

def _get_gurobi_params_with_defaults():
    try:
        import gurobipy as gp
        m = gp.Model()
        data = {}
        comments = {}
        for pname in dir(gp.GRB.Param):
            if pname.startswith("_"):
                continue
            try:
                info = m.getParamInfo(pname)
                if isinstance(info, tuple) and len(info) >= 6:
                    _, ptype, cur, pmin, pmax, default = info[:6]
                    ptype_name = getattr(ptype, "__name__", str(ptype))
                    data[pname] = default
                    comments[pname] = f"{ptype_name}, default={default!r}, range=[{pmin!r}, {pmax!r}]"
                elif isinstance(info, tuple) and len(info) >= 3:
                    _, ptype, cur = info[:3]
                    ptype_name = getattr(ptype, "__name__", str(ptype))
                    data[pname] = cur
                    comments[pname] = f"{ptype_name}, default={cur!r}"
            except Exception:
                continue
        try:
            m.dispose()
        except Exception:
            pass
        if data:
            return data, comments
    except Exception:
        pass
    fallback = _CURATED_DEFAULTS.get("gurobi", {})
    data = {k: v for k, (v, _) in fallback.items()}
    comments = {k: c for k, (_, c) in fallback.items()}
    return data, comments

def _get_copt_params_with_defaults():
    try:
        import coptpy
        env = coptpy.Envr()
        m = env.createModel("_flp_param_probe")
        data = {}
        comments = {}
        for pname in dir(coptpy.COPT.Param):
            if pname.startswith("_"):
                continue
            try:
                info = m.getParamInfo(pname)
                if isinstance(info, tuple) and len(info) >= 5:
                    _, cur, default, pmin, pmax = info[:5]
                    data[pname] = default
                    comments[pname] = f"default={default!r}, cur={cur!r}, range=[{pmin!r}, {pmax!r}]"
                elif isinstance(info, tuple) and len(info) >= 3:
                    _, cur, default = info[:3]
                    data[pname] = default
                    comments[pname] = f"default={default!r}, cur={cur!r}"
                elif isinstance(info, tuple) and len(info) >= 2:
                    data[pname] = info[1]
                    comments[pname] = f"default={info[1]!r}"
            except Exception:
                continue
        if data:
            return data, comments
    except Exception:
        pass
    fallback = _CURATED_DEFAULTS.get("copt", {})
    data = {k: v for k, (v, _) in fallback.items()}
    comments = {k: c for k, (_, c) in fallback.items()}
    return data, comments

def _heuristic_params_with_defaults(interface, solver):
    """Return (data, comments) for heuristic."""
    interface = interface.lower()
    solver = solver.lower() if solver else None
    # Unified defaults
    data = {}
    comments = {}
    for k, (default, comment) in _HEURISTIC_UNIFIED_DEFAULTS.items():
        data[k] = default
        comments[k] = comment

    # Algorithm-specific via introspection
    specific_data = {}
    specific_comments = {}
    try:
        if interface == "mealpy" and solver:
            from ..generators.model import mealpy_model_generator
            import inspect
            mapping = mealpy_model_generator.module_mappings.get(solver)
            if mapping:
                module_name, class_name, model_name = mapping
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                algo_class = getattr(model_class, model_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args", "pop_size", "epoch", "seed"):
                        continue
                    default = param.default if param.default is not inspect.Parameter.empty else None
                    specific_data[pname] = default
                    specific_comments[pname] = f"Mealpy {solver} param, default={default!r}"
        elif interface == "niapy" and solver:
            from ..generators.model import niapy_model_generator
            import inspect
            mapping = niapy_model_generator.module_mappings.get(solver)
            if mapping:
                module_name, class_name, model_name = mapping
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                algo_class = getattr(model_class, model_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args", "pop_size", "population_size", "NP", "epoch", "seed"):
                        continue
                    default = param.default if param.default is not inspect.Parameter.empty else None
                    specific_data[pname] = default
                    specific_comments[pname] = f"Niapy {solver} param, default={default!r}"
        elif interface == "indago" and solver:
            import indago
            name_upper = solver.upper()
            indago_name_map = {'pso': 'PSO', 'fwa': 'FWA', 'ssa': 'SSA', 'de': 'DE', 'ba': 'BA', 'efo': 'EFO', 'mrfo': 'MRFO', 'abc': 'ABC', 'nm': 'NM', 'msgd': 'MSGD', 'rs': 'RS', 'gwo': 'GWO', 'hbo': 'HBO', 'crs': 'CRS'}
            resolved = indago_name_map.get(solver.lower(), name_upper)
            if resolved in indago.optimizers_dict:
                opt = indago.optimizers_dict[resolved]()
                if hasattr(opt, 'params') and opt.params:
                    for k, v in opt.params.items():
                        specific_data[k] = v
                        specific_comments[k] = f"Indago {solver} param, default={v!r}"
        elif interface == "pygad" and solver:
            import pygad, inspect
            sig = inspect.signature(pygad.GA.__init__)
            skip = {"self", "kwargs", "args", "fitness_func", "gene_space", "num_genes", "sol_per_pop", "num_generations", "num_parents_mating", "random_seed", "verbose"}
            for pname, param in sig.parameters.items():
                if pname in skip:
                    continue
                default = param.default if param.default is not inspect.Parameter.empty else None
                specific_data[pname] = default
                specific_comments[pname] = f"PyGAD param, default={default!r}"
        elif interface == "pymoo" and solver:
            import inspect
            pymoo_map = {"ns-ga-ii": ("pymoo.algorithms.moo.nsga2", "NSGA2"), "d-ns-ga-ii": ("pymoo.algorithms.moo.dnsga2", "DNSGA2"), "ns-ga-iii": ("pymoo.algorithms.moo.nsga3", "NSGA3"), "mo-ea-d": ("pymoo.algorithms.moo.moead", "MOEAD"), "cta-ea": ("pymoo.algorithms.moo.ctaea", "CTAEA"), "r-ns-ga-ii": ("pymoo.algorithms.moo.rnsga2", "RNSGA2"), "r-ns-ga-iii": ("pymoo.algorithms.moo.rnsga3", "RNSGA3"), "u-ns-ga-iii": ("pymoo.algorithms.moo.unsga3", "UNSGA3"), "sp-ea-ii": ("pymoo.algorithms.moo.spea2", "SPEA2"), "sms-ea": ("pymoo.algorithms.moo.sms", "SMSEMOA"), "rv-ea": ("pymoo.algorithms.moo.rvea", "RVEA"), "age-mo-ea": ("pymoo.algorithms.moo.age", "AGEMOEA"), "age-mo-ea-ii": ("pymoo.algorithms.moo.age2", "AGEMOEA2"), "gde3": ("pymoo.algorithms.moo.gde3", "GDE3"), "kgb": ("pymoo.algorithms.moo.kgb", "KGB"), "cmopso": ("pymoo.algorithms.moo.cmopso", "CMOPSO"), "nsde": ("pymoo.algorithms.moo.nsde", "NSDE"), "omni": ("pymoo.algorithms.moo.omni", "OmniOptimizer"), "pinsga2": ("pymoo.algorithms.moo.pinsga2", "PINSGA2"), "nsder": ("pymoo.algorithms.moo.nsder", "NSDER")}
            if solver in pymoo_map:
                module_path, class_name = pymoo_map[solver]
                module = __import__(module_path, fromlist=[class_name])
                algo_class = getattr(module, class_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args", "sampling", "crossover", "mutation", "selection", "output", "repair", "eliminate_duplicates", "constr_handler", "pop_size", "n_gen", "seed", "verbose"):
                        continue
                    default = param.default if param.default is not inspect.Parameter.empty else None
                    specific_data[pname] = default
                    specific_comments[pname] = f"Pymoo {solver} param, default={default!r}"
        elif interface == "feloopy":
            # Feloopy internal heuristics
            feloopy_defaults = {"hco": {}, "gwo": {"w": 0.5}, "ga": {"crossover_rate": 0.8, "mutation_rate": 0.1}}
            if solver and solver.lower() in feloopy_defaults:
                for k, v in feloopy_defaults[solver.lower()].items():
                    specific_data[k] = v
                    specific_comments[k] = f"Feloopy {solver} param, default={v!r}"
    except Exception:
        pass

    # Merge unified + specific
    data.update(specific_data)
    comments.update(specific_comments)
    return data, comments


def get_solver_params(interface: str = "highs", solver: Optional[str] = None) -> CommentedDict:
    """
    Return solver-specific options passable via ``flp.search(..., options={})``.

    Returns a ``CommentedDict`` (dict subclass) that prints as a
    copy-pasteable commented snippet but works as a plain dict.

    Parameters
    ----------
    interface : str
        Felopy interface name, e.g. ``"highs"``, ``"gurobi"``, ``"copt"``,
        ``"cplex"``, ``"ortools"``, ``"scip"``, ``"cvxpy"``, ``"mealpy"``,
        ``"pydecision"``, ``"rsome_ro"`` etc.  Case-insensitive.
    solver : str, optional
        Solver / algorithm name.  For heuristic/madm interfaces this selects
        the algorithm (e.g. ``solver="pso"`` for ``interface="mealpy"``).

    Returns
    -------
    CommentedDict
        Dict of ``{param: default}`` with ``# comments`` when printed.
        Pass as ``flp.search(..., options=dict(opts))`` or ``options=opts``.

    Examples
    --------
    >>> import feloopy as flp
    >>> opts = flp.get_params(interface="highs")
    >>> print(opts)  # commented snippet
    >>> flp.search(model, interface="highs", solver="highs", options=dict(opts))
    """
    if interface is None:
        interface = "highs"
    iface = str(interface).strip().lower()
    solver_lc = str(solver).strip().lower() if solver is not None else None

    # ------------------------------------------------------------------
    # Heuristic / Metaheuristic
    # ------------------------------------------------------------------
    if iface in _HEURISTIC_IFACES:
        # If solver not given, return unified only
        if solver_lc:
            data, comments = _heuristic_params_with_defaults(iface, solver_lc)
            # Build sections: unified + algorithm-specific
            # Determine which keys are unified
            unified_keys = [k for k in _HEURISTIC_UNIFIED_DEFAULTS.keys() if k in data]
            algo_keys = [k for k in data.keys() if k not in unified_keys]
            sections = [("Unified (heuristic)", unified_keys)]
            if algo_keys:
                sections.append((f"Algorithm-specific ({solver_lc}/{iface})", algo_keys))
            return CommentedDict(data, comments=comments, sections=sections)
        else:
            data = {k: v for k, (v, _) in _HEURISTIC_UNIFIED_DEFAULTS.items()}
            comments = {k: c for k, (_, c) in _HEURISTIC_UNIFIED_DEFAULTS.items()}
            sections = [("Unified (heuristic)", list(data.keys()))]
            return CommentedDict(data, comments=comments, sections=sections)

    # ------------------------------------------------------------------
    # Constraint
    # ------------------------------------------------------------------
    if iface in _CONSTRAINT_IFACES:
        iface_norm = "ortools_cp"
        fallback = _CURATED_DEFAULTS.get(iface_norm, {})
        data = {k: v for k, (v, _) in fallback.items()}
        comments = {k: c for k, (_, c) in fallback.items()}
        sections = [("Constraint programming (via options)", list(data.keys()))]
        return CommentedDict(data, comments=comments, sections=sections)

    # ------------------------------------------------------------------
    # Uncertain (RSOME)
    # ------------------------------------------------------------------
    if iface in _UNCERTAIN_IFACES:
        fallback = _CURATED_DEFAULTS.get(iface, _CURATED_DEFAULTS.get("generic", {}))
        data = {k: v for k, (v, _) in fallback.items()}
        comments = {k: c for k, (_, c) in fallback.items()}
        sections = [("Uncertain optimization (RSOME via options)", list(data.keys()))]
        return CommentedDict(data, comments=comments, sections=sections)

    # ------------------------------------------------------------------
    # MADM (pydecision / weighting / ranking)
    # ------------------------------------------------------------------
    if iface in _MADM_IFACES or iface == "pydecision":
        # For MADM, options are rarely needed; return generic note
        # But support solver-specific weighting/ranking if requested
        if solver_lc:
            # For madm, solver is like 'topsis_method' – no specific options
            data = {}
            comments = {}
            sections = [(f"MADM {solver_lc} (no solver options required)", [])]
            return CommentedDict(data, comments=comments, sections=sections)
        data = {}
        comments = {}
        sections = [("MADM (pydecision) – no options required, method selects algorithm", [])]
        return CommentedDict(data, comments=comments, sections=sections)

    # ------------------------------------------------------------------
    # Exact / Convex / Generic (including meta-interfaces)
    # ------------------------------------------------------------------
    # Meta-interfaces: cvxpy/pyomo/jump etc delegate to an underlying solver.
    # If solver is given and is a known solver, return that solver's params
    # plus the meta-interface's own curated (if any).
    _KNOWN_SOLVERS = {"highs", "gurobi", "copt", "cplex", "xpress", "mosek", "scip", "glpk", "cbc", "clarabel", "osqp", "scs", "ecos", "cvxopt", "glop", "pdlp", "proxqp", "nag", "ipopt", "knitro", "baron", "scip"}
    if iface in _META_IFACES and solver_lc and solver_lc in _KNOWN_SOLVERS:
        # Use solver's params but keep meta-interface note
        # Build meta-interface curated if exists, plus solver-specific
        data = {}
        comments = {}
        sections = []
        sections.append(("Note: common top-level args – use flp.search(time_limit=..., cpu_threads=..., absolute_gap=..., relative_gap=..., verbose=...) – not via options", []))
        decomp_keys = []
        for k, (default, comment) in _DECOMPOSITION_DEFAULTS.items():
            data[k] = default
            comments[k] = comment
            decomp_keys.append(k)
        sections.append(("Decomposition / boost (via options)", decomp_keys))
        # Meta-interface own options (e.g. cvxpy verbose etc) if any and not generic
        meta_key = _INTERFACE_MAP.get(iface, iface)
        if meta_key in _CURATED_DEFAULTS and meta_key != "generic":
            meta_fallback = _CURATED_DEFAULTS[meta_key]
            meta_keys = []
            for k, (default, comment) in meta_fallback.items():
                if k not in data:
                    data[k] = default
                    comments[k] = comment
                    meta_keys.append(k)
            if meta_keys:
                sections.append((f"Meta-interface ({iface} via options)", meta_keys))
        # Solver-specific
        solver_keys = []
        # Map solver to curated key (handle aliases like cbc -> pulp/mip etc)
        _SOLVER_MAP = {"cbc": "pulp", "glpk": "pulp", "glpk-mi": "pulp", "highs": "highs", "gurobi": "gurobi", "copt": "copt", "cplex": "cplex", "xpress": "xpress", "mosek": "mosek", "scip": "scip", "clarabel": "cvxpy", "osqp": "cvxpy", "scs": "cvxpy", "ecos": "cvxpy", "cvxopt": "cvxpy", "ipopt": "generic", "knitro": "generic"}
        s_key = _SOLVER_MAP.get(solver_lc, solver_lc)
        # Try dynamic for known solvers
        if s_key == "highs":
            s_data, s_comments = _get_highs_params_with_defaults()
        elif s_key == "gurobi":
            s_data, s_comments = _get_gurobi_params_with_defaults()
        elif s_key == "copt":
            s_data, s_comments = _get_copt_params_with_defaults()
        elif s_key in _CURATED_DEFAULTS:
            s_data = {k: v for k, (v, _) in _CURATED_DEFAULTS[s_key].items()}
            s_comments = {k: c for k, (_, c) in _CURATED_DEFAULTS[s_key].items()}
        else:
            s_data, s_comments = {}, {}
        for k, v in s_data.items():
            if k not in data:
                data[k] = v
                comments[k] = s_comments.get(k, "")
                solver_keys.append(k)
        if solver_keys:
            sections.append((f"Solver-specific ({solver_lc} via {iface} via options)", solver_keys))
        return CommentedDict(data, comments=comments, sections=sections)

    # Build decomposition + solver-specific (common is top-level, not via options)
    data = {}
    comments = {}
    sections = []

    # Informational header for common top-level (not part of options dict)
    sections.append(("Note: common top-level args – use flp.search(time_limit=..., cpu_threads=..., absolute_gap=..., relative_gap=..., verbose=...) – not via options", []))

    # Decomposition
    decomp_keys = []
    for k, (default, comment) in _DECOMPOSITION_DEFAULTS.items():
        # Only include if default is not None? Keep all but allow None
        data[k] = default
        comments[k] = comment
        decomp_keys.append(k)
    sections.append(("Decomposition / boost (via options)", decomp_keys))

    # Solver-specific
    curated_key = _INTERFACE_MAP.get(iface, iface)
    solver_keys = []
    if curated_key == "highs":
        s_data, s_comments = _get_highs_params_with_defaults()
        for k, v in s_data.items():
            data[k] = v
            comments[k] = s_comments.get(k, "")
            solver_keys.append(k)
    elif curated_key == "gurobi":
        s_data, s_comments = _get_gurobi_params_with_defaults()
        for k, v in s_data.items():
            data[k] = v
            comments[k] = s_comments.get(k, "")
            solver_keys.append(k)
    elif curated_key == "copt":
        s_data, s_comments = _get_copt_params_with_defaults()
        for k, v in s_data.items():
            data[k] = v
            comments[k] = s_comments.get(k, "")
            solver_keys.append(k)
    elif curated_key in _CURATED_DEFAULTS:
        fallback = _CURATED_DEFAULTS[curated_key]
        for k, (default, comment) in fallback.items():
            # Avoid overwriting common/decomp if duplicate (keep first)
            if k in data and curated_key != "highs":
                # For highs, dynamic already has these keys; keep dynamic
                continue
            if k not in data:
                data[k] = default
                comments[k] = comment
                solver_keys.append(k)
            else:
                # Update comment if missing
                comments[k] = comment
                if k not in solver_keys:
                    solver_keys.append(k)
    else:
        # Unknown interface – generic tip
        fallback = _CURATED_DEFAULTS.get("generic", {})
        for k, (default, comment) in fallback.items():
            if k not in data:
                data[k] = default
                comments[k] = comment
                solver_keys.append(k)

    if solver_keys:
        sections.append((f"Solver-specific ({curated_key} via options)", solver_keys))

    return CommentedDict(data, comments=comments, sections=sections)


# Alias for convenience
get_params = get_solver_params
