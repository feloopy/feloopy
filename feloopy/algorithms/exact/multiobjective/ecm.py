import numpy as np

from .enums import MultiObjectiveError, _CP_INTERFACES
from .utils import _apply_features, _solve_and_collect, _collect_variables


def _is_cp(solver_opts):
    return solver_opts.get('solver_name', '') in _CP_INTERFACES


def solve(instance, payoff, directions, M, solver_opts, approach_options,
          solution_generator, save_vars, progress):
    maxobj = np.amax(payoff, axis=0)
    minobj = np.amin(payoff, axis=0)
    if np.any(maxobj - minobj == 0):
        raise MultiObjectiveError(
            f"No conflict among objectives (identical payoff values).\n"
            f"Payoff table:\n{payoff}\n"
            f"Zero-range objectives: {np.where(maxobj - minobj == 0)[0].tolist()}")
    intervals = approach_options.get('intervals', 10)
    important_idx = approach_options.get('important_objective', None)
    is_cp = _is_cp(solver_opts)

    def _build_eps_model(g, intervals, important):
        model_object = instance()
        eps_directions = directions.copy()
        eps_directions.append(directions[important])
        _apply_features(model_object, eps_directions, **solver_opts)
        model_object.features['objective_being_optimized'] = M
        z = model_object.fvar('_z', [range(M)])
        model_object.obj(z[important])
        for k in range(M):
            model_object.con(z[k] == model_object.features['objectives'][k],
                             name=f"epsilon_objective_value_{k}")
        for k in range(M):
            if k != important:
                if directions[k] == 'max':
                    bound = maxobj[k] - (g / intervals) * (maxobj[k] - minobj[k])
                    model_object.con(z[k] >= (int(bound) if is_cp else bound),
                                     name=f"epsilon_max_{k}")
                else:
                    bound = minobj[k] + (g / intervals) * (maxobj[k] - minobj[k])
                    model_object.con(z[k] <= (int(bound) if is_cp else bound),
                                     name=f"epsilon_min_{k}")
        return model_object, z

    pareto, variables, ogrs = [], [], []
    objective_range = range(M) if important_idx is None else [important_idx]
    total = len(objective_range) * (intervals + 1)
    step = 0
    progress.update()
    for k in objective_range:
        for g in range(intervals + 1):
            step += 1
            model_object, z = _build_eps_model(g, intervals, k)
            _solve_and_collect(model_object, solution_generator)
            healthy = model_object.healthy()
            if not healthy:
                continue
            result = [float(model_object.get_numpy_var('_z')[kk]) for kk in range(M)]
            progress.add_step(
                f"Step {step}/{total}",
                f"obj={k+1}, eps={g}/{intervals}",
                str([round(v, 4) for v in result]),
            )
            progress.update()
            pareto.append(result)
            try:
                ogr = model_object.get_ogr()
                ogrs.append(ogr)
            except Exception:
                pass
            if save_vars:
                variables.append(_collect_variables(model_object))
    if len(pareto) == 0:
        raise MultiObjectiveError("ECM found no feasible Pareto solutions.")
    return np.array(pareto), variables, ogrs
