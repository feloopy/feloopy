# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from .option_utils import normalize_options, get_epoch_value, get_pop_size_value


def generate_solution(solver_name, fitness_function, total_features, objectives_directions, objective_number, number_of_times, show_plots, show_log, solver_options, init_solutions=None):

    import timeit
    import numpy as np
    import indago

    lb = np.zeros(total_features[1])
    ub = np.ones(total_features[1])

    Multiplier = {'max': -1, 'min': 1}

    def eval_fn(X):
        result = fitness_function(X)
        if isinstance(result, (tuple, list)):
            obj_val = float(result[0])
        else:
            obj_val = float(result)
        return obj_val * Multiplier.get(objectives_directions[objective_number], 1)

    indago_name_map = {
        'pso': 'PSO',
        'fwa': 'FWA',
        'ssa': 'SSA',
        'de': 'DE',
        'ba': 'BA',
        'efo': 'EFO',
        'mrfo': 'MRFO',
        'abc': 'ABC',
        'nm': 'NM',
        'msgd': 'MSGD',
        'rs': 'RS',
        'gwo': 'GWO',
        'hbo': 'HBO',
        'crs': 'CRS',
    }

    optimizer_name = indago_name_map.get(solver_name, solver_name.upper())

    if optimizer_name not in indago.optimizers_dict:
        raise ValueError(f"Unknown indago optimizer: {optimizer_name}. Available: {list(indago.optimizers_dict.keys())}")

    optimizer_class = indago.optimizers_dict[optimizer_name]
    optimizer = optimizer_class()

    optimizer.evaluation_function = eval_fn
    optimizer.lb = lb
    optimizer.ub = ub
    optimizer.monitoring = 'basic' if show_log else 'none'

    norm = normalize_options(solver_options, "indago")

    has_budget = 'max_evaluations' in norm or 'max_iterations' in norm or 'epoch' in norm

    if 'pop_size' in norm:
        if optimizer_name == 'PSO':
            optimizer.params['swarm_size'] = norm['pop_size']
        else:
            optimizer.params['pop_size'] = norm['pop_size']

    if 'max_iterations' in norm:
        optimizer.max_iterations = norm['max_iterations']

    if 'max_evaluations' in norm:
        optimizer.max_evaluations = norm['max_evaluations']

    if 'verbose' in norm:
        optimizer.monitoring = 'basic' if norm['verbose'] else 'none'

    for key, val in norm.items():
        if key not in ('pop_size', 'max_iterations', 'max_evaluations', 'seed', 'verbose'):
            optimizer.params[key] = val

    if init_solutions is not None:
        import numpy as _np
        init_arr = _np.atleast_2d(_np.array(init_solutions, dtype=float))
        optimizer.X0 = init_arr

    if 'max_evaluations' not in norm and 'max_iterations' not in norm and 'epoch' not in norm:
        optimizer.max_evaluations = 50 * total_features[1] ** 2

    _direction = Multiplier.get(objectives_directions[objective_number], 1)

    if number_of_times == 1:
        time_solve_begin = timeit.default_timer()
        result = optimizer.optimize()
        time_solve_end = timeit.default_timer()
        best_agent = np.array(result.X)
        best_reward = np.array(result.f * _direction)

    else:
        time_solve_begin = []
        time_solve_end = []
        best_reward_found = -np.inf

        for i in range(number_of_times):
            opt_copy = optimizer.copy()
            opt_copy.monitoring = 'none'
            time_solve_begin.append(timeit.default_timer())
            result = opt_copy.optimize()
            time_solve_end.append(timeit.default_timer())
            real_f = result.f * _direction
            if real_f >= best_reward_found:
                best_agent_found = np.array(result.X)
                best_reward_found = real_f

        best_agent = np.array(best_agent_found)
        best_reward = np.array(best_reward_found)

    return best_agent, best_reward, time_solve_begin, time_solve_end
