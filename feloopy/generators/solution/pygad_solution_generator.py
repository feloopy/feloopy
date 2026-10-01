# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from .option_utils import normalize_options, get_epoch_value, get_pop_size_value

import numpy as np
import timeit


PYGAD_OPTION_MAP = {
    "epoch":           "num_generations",
    "pop_size":        "sol_per_pop",
    "seed":            "random_seed",
    "verbose":         "verbose",
    "max_evaluations": None,
}

SOLVER_TO_SELECTION = {
    "ga":    "sss",
    "nsga2": "nsga2",
    "nsga3": "nsga3",
}

SOLVER_TO_TERMINATION = {
    "ga":    None,
    "nsga2": None,
    "nsga3": None,
}


def generate_solution(solver_name, model_object, fitness_function, total_features, objectives_directions, objective_number, number_of_times, show_plots, save_plots, show_log, solver_options, init_solutions=None, _on_repeat_start=None):

    import pygad

    is_multi_objective = len(objectives_directions) > 1
    norm = normalize_options(solver_options, "pygad")

    config = {}
    for unified_key, pygad_key in PYGAD_OPTION_MAP.items():
        if pygad_key is None:
            continue

        if unified_key in norm:
            config[pygad_key] = norm[unified_key]
        elif pygad_key in norm:
            config[pygad_key] = norm[pygad_key]

    for key, val in solver_options.items():
        if key.startswith("---"):
            continue
        if val is None:
            continue
        if key not in PYGAD_OPTION_MAP and key not in config:
            config[key] = val

    if "gene_space" not in config:
        config["gene_space"] = [{"low": 0, "high": 1} for _ in range(total_features[1])]
    if "num_genes" not in config:
        config["num_genes"] = total_features[1]
    if "num_generations" not in config:
        config["num_generations"] = 100
    if "sol_per_pop" not in config:
        config["sol_per_pop"] = 50
    if "num_parents_mating" not in config:
        config["num_parents_mating"] = config["sol_per_pop"] // 2

    if init_solutions is not None:
        import numpy as _np
        init_arr = _np.atleast_2d(_np.array(init_solutions, dtype=float))

        n_pop = int(config.get("sol_per_pop", 50))
        if init_arr.shape[0] < n_pop:
            init_arr = _np.vstack([init_arr, _np.random.rand(n_pop - init_arr.shape[0], init_arr.shape[1])])
        elif init_arr.shape[0] > n_pop:
            n_pop = int(init_arr.shape[0])
            config["sol_per_pop"] = n_pop
            if config.get("num_parents_mating", 0) > n_pop:
                config["num_parents_mating"] = max(2, n_pop // 2)
        config["initial_population"] = init_arr

    if is_multi_objective:
        selection_type = SOLVER_TO_SELECTION.get(solver_name, "nsga2")
        config["parent_selection_type"] = selection_type
        if selection_type == "nsga3" and "nsga3_num_divisions" not in config:
            config["nsga3_num_divisions"] = 12

        def mo_fitness_func(ga_instance, sol, solution_idx):
            raw = fitness_function(sol)
            if hasattr(raw, "item"):
                raw = [float(raw.item())]
            elif hasattr(raw, "__len__"):
                raw = [float(v) for v in raw]
            else:
                raw = [float(raw)]
            return raw

        ga_instance = pygad.GA(fitness_func=mo_fitness_func, **config)

        time_solve_begin = timeit.default_timer()
        ga_instance.run()
        time_solve_end = timeit.default_timer()

        population = ga_instance.population
        fitness = np.array(ga_instance.last_generation_fitness)

        n_solutions = len(population)
        dominated = np.zeros(n_solutions, dtype=bool)
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i == j or dominated[j]:
                    continue
                if np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                    dominated[i] = True
                    break

        pareto_mask = ~dominated
        pareto_solutions = population[pareto_mask]
        pareto_front = fitness[pareto_mask]

        return pareto_solutions, pareto_front, time_solve_begin, time_solve_end

    else:
        coeffs = get_coeffs(objectives_directions)

        def so_fitness_func(ga_instance, sol, solution_idx):
            result = fitness_function(sol)
            if hasattr(result, "item"):
                result = result.item()
            elif hasattr(result, "__len__") and len(result) == 1:
                result = float(result[0])
            result = float(result) * coeffs
            return float(result[0]) if len(result) == 1 else result

        selection_type = SOLVER_TO_SELECTION.get(solver_name, "sss")
        if "parent_selection_type" not in config:
            config["parent_selection_type"] = selection_type

        ga_instance = pygad.GA(fitness_func=so_fitness_func, **config)

        if number_of_times == 1:

            time_solve_begin = timeit.default_timer()
            ga_instance.run()
            time_solve_end = timeit.default_timer()

            best_agent, best_reward, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            best_reward = (best_reward * coeffs)[0]

        else:

            Multiplier = {"max": 1, "min": -1}
            directions = Multiplier[objectives_directions[objective_number]]
            time_solve_begin = []
            time_solve_end = []
            bestreward = [-directions * np.inf]
            best_reward_found = -directions * np.inf
            best_agent_found = None

            for i in range(number_of_times):
                if _on_repeat_start is not None:
                    _on_repeat_start()
                fresh_ga = pygad.GA(fitness_func=so_fitness_func, **config)
                time_solve_begin.append(timeit.default_timer())
                fresh_ga.run()
                time_solve_end.append(timeit.default_timer())
                best_agent, best_reward, solution_idx = fresh_ga.best_solution()
                obj_val = float(best_reward[0]) if hasattr(best_reward, "__len__") else float(best_reward)
                bestreward.append(obj_val)
                if directions * obj_val >= directions * best_reward_found:
                    best_agent_found = best_agent
                    best_reward_found = obj_val
            bestreward.pop(0)

            if show_log:
                from tabulate import tabulate as tb

                ave = [round((time_solve_end[i] - time_solve_begin[i]) * 10**6) for i in range(number_of_times)]

                print()
                print("~~~~~~~\nTIME INFO\n~~~~~~~")
                print(tb({
                    "cpt (ave)": [np.average(ave)],
                    "cpt (std)": [np.std(ave)],
                    "unit": ["micro sec"]
                }, headers="keys", tablefmt="github"))
                print("~~~~~~~")

                print("~~~~~~~\nOBJ INFO\n~~~~~~~")
                print(tb({
                    "obj": [np.max(bestreward), np.average(bestreward), np.std(bestreward), np.min(bestreward)],
                    "unit": ["max", "average", "standard deviation", "min"]
                }, headers="keys", tablefmt="github"))
                print("~~~~~~~")

            best_agent = best_agent_found
            best_reward = best_reward_found

        return best_agent, best_reward, time_solve_begin, time_solve_end


def get_coeffs(directions):
    max_indicator = 1
    min_indicator = -1
    return np.array([max_indicator if d != "min" else min_indicator for d in directions])
