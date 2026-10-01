# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""
Unified option normalization and metadata for heuristic interfaces.

Unified option names (work across mealpy, niapy, indago, pymoo):
    pop_size        : Population / swarm size
    epoch           : Number of generations / iterations
    max_evaluations : Maximum number of function evaluations
    seed            : Random seed for reproducibility
    verbose         : Show solver log (bool)
"""

import inspect

UNIFIED_OPTION_ALIASES = {
    "pop_size":        ["pop_size", "population_size", "swarm_size", "sol_per_pop", "num_agents"],
    "epoch":           ["epoch", "n_gen", "max_generations", "max_iterations", "num_generations", "max_iters"],
    "max_evaluations": ["max_evaluations", "n_eval", "maxfevals"],
    "seed":            ["seed", "random_seed", "key"],
    "verbose":         ["verbose", "show_log", "log_to"],
}

UNIFIED_OPTION_DESCRIPTIONS = {
    "pop_size":        "Population / swarm size (int). Default: 50.",
    "epoch":           "Number of generations / iterations (int). Default: 100.",
    "max_evaluations": "Maximum number of function evaluations (int). Overrides epoch if both given.",
    "seed":            "Random seed for reproducibility (int or None). Default: None.",
    "verbose":         "Show solver convergence log (bool). Default: False.",
}


def resolve_alias(key, options):
    """Resolve unified option names from any alias variant.

    For example, if the user passes ``n_gen=200``, this returns
    ``('epoch', 200)``.  If the key is already a unified name it is
    returned unchanged.
    """
    if key in UNIFIED_OPTION_ALIASES:
        return key, options[key]
    for unified, aliases in UNIFIED_OPTION_ALIASES.items():
        if key in aliases:
            return unified, options[key]
    return key, options[key]


def normalize_options(options, interface):
    """Convert unified option names to the native names expected by *interface*.

    Returns a **new** dict – the original *options* is never mutated.

    Supported unified names
    -----------------------
    pop_size, epoch, max_evaluations, seed, verbose

    Interface-specific mappings
    ---------------------------
    mealpy  : epoch → epoch  (native)
    niapy   : epoch → max_iters  (for Task)
    indago  : epoch → max_iterations; also passes max_evaluations
    pymoo   : epoch → n_gen; max_evaluations → n_eval
    """
    normalized = {}
    aliases_resolved = {}

    for key, val in options.items():
        unified, resolved_val = resolve_alias(key, options)
        if unified != key:
            aliases_resolved[key] = unified
        normalized[unified] = resolved_val

    interface = interface.lower()

    if interface == "mealpy":
        result = {}
        for k, v in normalized.items():
            if k == "epoch":
                result["epoch"] = v
            elif k == "pop_size":
                result["pop_size"] = v
            elif k == "seed":
                result["seed"] = v
            elif k == "max_evaluations":
                result["max_evaluations"] = v
            else:
                result[k] = v
        return result

    elif interface == "niapy":
        result = {}
        for k, v in normalized.items():
            if k == "epoch":
                result["epoch"] = v
            elif k == "pop_size":
                result["pop_size"] = v
            elif k == "seed":
                result["seed"] = v
            elif k == "max_evaluations":
                result["max_evaluations"] = v
            else:
                result[k] = v
        return result

    elif interface == "indago":
        result = {}
        for k, v in normalized.items():
            if k == "epoch":
                result["max_iterations"] = v
            elif k == "pop_size":
                result["pop_size"] = v
            elif k == "max_evaluations":
                result["max_evaluations"] = v
            elif k == "seed":
                result["seed"] = v
            elif k == "verbose":
                result["verbose"] = v
            else:
                result[k] = v
        return result

    elif interface == "pygad":
        result = {}
        for k, v in normalized.items():
            if k == "epoch":
                result["num_generations"] = v
            elif k == "pop_size":
                result["sol_per_pop"] = v
            elif k == "seed":
                result["random_seed"] = v
            elif k == "verbose":
                result["verbose"] = v
            elif k == "max_evaluations":
                continue
            else:
                result[k] = v
        return result

    elif interface == "pymoo":
        result = {}
        for k, v in normalized.items():
            if k == "epoch":
                result["n_gen"] = v
            elif k == "max_evaluations":
                result["n_eval"] = v
            elif k == "pop_size":
                result["pop_size"] = v
            elif k == "seed":
                result["seed"] = v
            elif k == "verbose":
                result["verbose"] = v
            else:
                result[k] = v
        return result

    else:
        return dict(normalized)


def get_epoch_value(options):
    """Extract the epoch / iteration count from options, regardless of alias.

    Returns ``None`` if no epoch-like option is present.
    """
    for unified, aliases in UNIFIED_OPTION_ALIASES.items():
        if unified == "epoch":
            for alias in aliases:
                if alias in options:
                    return options[alias]
    return None


def get_pop_size_value(options):
    """Extract the population size from options, regardless of alias.

    Returns ``None`` if no pop_size-like option is present.
    """
    for unified, aliases in UNIFIED_OPTION_ALIASES.items():
        if unified == "pop_size":
            for alias in aliases:
                if alias in options:
                    return options[alias]
    return None


def get_algo_options(solver_name, interface):
    """Return available options for a given algorithm and interface.

    Returns a dict of ``{option_name: description}`` for the unified options
    that the algorithm recognises, plus any algorithm-specific parameters
    discovered by inspecting the constructor.

    Examples
    --------
    >>> get_algo_options("pso", "mealpy")
    >>> get_algo_options("pso", "indago")
    """
    interface = interface.lower()
    unified_options = dict(UNIFIED_OPTION_DESCRIPTIONS)

    specific = {}

    if interface == "mealpy":
        try:
            from ..model import mealpy_model_generator
            mapping = mealpy_model_generator.module_mappings.get(solver_name)
            if mapping:
                module_name, class_name, model_name = mapping
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                algo_class = getattr(model_class, model_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args"):
                        continue
                    if pname in ("pop_size", "epoch", "seed"):
                        continue
                    desc = ""
                    if param.default is not inspect.Parameter.empty:
                        desc = f"Default: {param.default}"
                    specific[pname] = desc
        except Exception:
            pass

    elif interface == "niapy":
        try:
            from ..model import niapy_model_generator
            mapping = niapy_model_generator.module_mappings.get(solver_name)
            if mapping:
                module_name, class_name, model_name = mapping
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                algo_class = getattr(model_class, model_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args"):
                        continue
                    if pname in ("pop_size", "population_size", "NP", "epoch", "seed"):
                        continue
                    desc = ""
                    if param.default is not inspect.Parameter.empty:
                        desc = f"Default: {param.default}"
                    specific[pname] = desc
        except Exception:
            pass

    elif interface == "indago":
        try:
            import indago
            name_upper = solver_name.upper()
            indago_name_map = {
                'pso': 'PSO', 'fwa': 'FWA', 'ssa': 'SSA', 'de': 'DE',
                'ba': 'BA', 'efo': 'EFO', 'mrfo': 'MRFO', 'abc': 'ABC',
                'nm': 'NM', 'msgd': 'MSGD', 'rs': 'RS', 'gwo': 'GWO',
                'hbo': 'HBO', 'crs': 'CRS',
            }
            resolved = indago_name_map.get(solver_name.lower(), name_upper)
            if resolved in indago.optimizers_dict:
                opt = indago.optimizers_dict[resolved]()
                if hasattr(opt, 'params') and opt.params:
                    for k, v in opt.params.items():
                        specific[k] = f"Default: {v}"
        except Exception:
            pass

    elif interface == "pygad":
        try:
            import pygad
            sig = inspect.signature(pygad.GA.__init__)
            skip = {"self", "kwargs", "args", "fitness_func", "gene_space",
                    "num_genes", "sol_per_pop", "num_generations",
                    "num_parents_mating", "random_seed", "verbose"}
            for pname, param in sig.parameters.items():
                if pname in skip:
                    continue
                desc = ""
                if param.default is not inspect.Parameter.empty:
                    desc = f"Default: {param.default}"
                specific[pname] = desc
        except Exception:
            pass

    elif interface == "pymoo":
        try:
            pymoo_map = {
                "ns-ga-ii": ("pymoo.algorithms.moo.nsga2", "NSGA2"),
                "d-ns-ga-ii": ("pymoo.algorithms.moo.dnsga2", "DNSGA2"),
                "ns-ga-iii": ("pymoo.algorithms.moo.nsga3", "NSGA3"),
                "mo-ea-d": ("pymoo.algorithms.moo.moead", "MOEAD"),
                "cta-ea": ("pymoo.algorithms.moo.ctaea", "CTAEA"),
                "r-ns-ga-ii": ("pymoo.algorithms.moo.rnsga2", "RNSGA2"),
                "r-ns-ga-iii": ("pymoo.algorithms.moo.rnsga3", "RNSGA3"),
                "u-ns-ga-iii": ("pymoo.algorithms.moo.unsga3", "UNSGA3"),
                "sp-ea-ii": ("pymoo.algorithms.moo.spea2", "SPEA2"),
                "sms-ea": ("pymoo.algorithms.moo.sms", "SMSEMOA"),
                "rv-ea": ("pymoo.algorithms.moo.rvea", "RVEA"),
                "age-mo-ea": ("pymoo.algorithms.moo.age", "AGEMOEA"),
                "age-mo-ea-ii": ("pymoo.algorithms.moo.age2", "AGEMOEA2"),
                "gde3": ("pymoo.algorithms.moo.gde3", "GDE3"),
                "kgb": ("pymoo.algorithms.moo.kgb", "KGB"),
                "cmopso": ("pymoo.algorithms.moo.cmopso", "CMOPSO"),
                "nsde": ("pymoo.algorithms.moo.nsde", "NSDE"),
                "omni": ("pymoo.algorithms.moo.omni", "OmniOptimizer"),
                "pinsga2": ("pymoo.algorithms.moo.pinsga2", "PINSGA2"),
                "nsder": ("pymoo.algorithms.moo.nsder", "NSDER"),
            }
            if solver_name in pymoo_map:
                module_path, class_name = pymoo_map[solver_name]
                module = __import__(module_path, fromlist=[class_name])
                algo_class = getattr(module, class_name)
                sig = inspect.signature(algo_class.__init__)
                for pname, param in sig.parameters.items():
                    if pname in ("self", "kwargs", "args", "sampling",
                                 "crossover", "mutation", "selection",
                                 "output", "repair", "eliminate_duplicates",
                                 "constr_handler"):
                        continue
                    if pname in ("pop_size", "n_gen", "seed", "verbose"):
                        continue
                    desc = ""
                    if param.default is not inspect.Parameter.empty:
                        desc = f"Default: {param.default}"
                    specific[pname] = desc
        except Exception:
            pass

    result = {}
    result["--- Unified ---"] = ""
    for k, v in unified_options.items():
        result[k] = v
    if specific:
        result["--- Algorithm-specific ---"] = ""
        for k, v in specific.items():
            result[k] = v
    return result


def list_algorithms(interface):
    """Return a list of available algorithm names for the given interface."""
    interface = interface.lower()
    if interface == "mealpy":
        from ..model import mealpy_model_generator
        return sorted(mealpy_model_generator.module_mappings.keys())
    elif interface == "niapy":
        from ..model import niapy_model_generator
        return sorted(niapy_model_generator.module_mappings.keys())
    elif interface == "indago":
        import indago
        return sorted(k.lower() for k in indago.optimizers_dict.keys())
    elif interface == "pygad":
        return ["ga", "nsga2", "nsga3"]
    elif interface == "pymoo":
        return sorted([
            "ns-ga-ii", "d-ns-ga-ii", "ns-ga-iii", "mo-ea-d", "cta-ea",
            "r-ns-ga-ii", "r-ns-ga-iii", "u-ns-ga-iii", "sp-ea-ii",
            "sms-ea", "rv-ea", "age-mo-ea", "age-mo-ea-ii", "gde3",
            "kgb", "cmopso", "nsde", "omni", "pinsga2", "nsder",
        ])
    return []
