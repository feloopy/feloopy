import numpy as np


def _apply_features(model_object, directions, **kwargs):
    model_object.features['directions'] = directions
    for key in ('solver_name', 'solver_options', 'debug_mode', 'time_limit',
                'thread_count', 'absolute_gap', 'relative_gap', 'log',
                'write_model_file', 'save_solver_log', 'email_address',
                'max_iterations', 'obj_operators', 'uncertainty_set_constraints'):
        if key in kwargs:
            model_object.features[key] = kwargs[key]


def _solve_and_collect(model_object, solution_generator):
    model_object.features['model_object_before_solve'] = model_object.model
    model_object.solution = solution_generator.generate_solution(model_object.features)
    healthy = model_object.healthy()
    ogr = None
    if healthy:
        try:
            ogr = model_object.get_ogr()
        except:
            pass
    return healthy, ogr


def _collect_variables(model_object):
    variables = {}
    for typ, var in model_object.features['variables'].keys():
        variables[var] = model_object.get_numpy_var(var)
    return variables


def _get_obj_values(model_object, var_list, n_objectives):
    return [model_object.get_variable(var_list[k]) for k in range(n_objectives)]


def _generate_weights(M, method):
    if method == 'random':
        nums = np.random.rand(M)
        return nums / nums.sum()
    return np.random.dirichlet(np.ones(M), size=1)[0]


def _is_pareto_dominated_vectorized(points, d):
    n = len(points)
    if n <= 1:
        return np.zeros(n, dtype=bool)
    scaled = d * points
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if np.all(scaled[i] <= scaled[j]) and np.any(scaled[i] < scaled[j]):
                dominated[i] = True
                break
    return dominated


def _deduplicate_pareto(pareto, variables):
    if len(pareto) == 0:
        return pareto, variables
    _, unique_indices = np.unique(pareto, axis=0, return_index=True)
    sorted_idx = np.sort(unique_indices)
    return pareto[sorted_idx], [variables[i] for i in sorted_idx]


def revise_pareto(dir_map, directions, pareto, variables):
    if len(pareto) == 0:
        return np.empty((0, len(directions))), []
    d = np.array([-1 * dir_map[direction] for direction in directions])
    dominated = _is_pareto_dominated_vectorized(pareto, d)
    mask = ~dominated
    pareto = pareto[mask]
    variables = [v for v, keep in zip(variables, mask) if keep]
    pareto, variables = _deduplicate_pareto(pareto, variables)
    return pareto, variables


def _generate_payoff_table(instance, M, directions, solver_opts,
                           solution_generator, progress):
    payoff = np.zeros([M, M])
    progress.update()
    for m in range(M):
        dir_label = f"{directions[m]}imize"
        model_object = instance()
        _apply_features(model_object, directions, **solver_opts)
        model_object.features['objective_being_optimized'] = m
        z = model_object.fvar('_z', dim=[range(M)])
        for k in range(M):
            model_object.con(z[k] == model_object.features['objectives'][k],
                             name=f'_payoff_{m}_{k}')
        _solve_and_collect(model_object, solution_generator)
        healthy = model_object.healthy()
        if not healthy:
            from .enums import MultiObjectiveError
            raise MultiObjectiveError(
                f"Failed to solve payoff table entry: "
                f"{directions[m]}imizing objective {m}")
        for k in range(M):
            payoff[m, k] = float(model_object.get_numpy_var('_z')[k])
        progress.add_step(
            f"Payoff {m+1}/{M}",
            f"{dir_label} obj {m+1}",
            str([round(float(payoff[m, k]), 2) for k in range(M)]),
        )
        progress.update()
    progress.set_payoff(payoff)
    progress.update()
    return payoff
