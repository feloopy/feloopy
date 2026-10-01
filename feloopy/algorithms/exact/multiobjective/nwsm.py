import numpy as np

from .enums import MultiObjectiveError
from .utils import (
    _apply_features, _solve_and_collect, _collect_variables, _generate_weights,
)


def solve(instance, payoff, directions, M, solver_opts, approach_options,
          solution_generator, save_vars, progress, weights):
    maxobj = np.amax(payoff, axis=0)
    minobj = np.amin(payoff, axis=0)
    if np.any(maxobj - minobj == 0):
        raise MultiObjectiveError(
            f"No conflict among objectives (identical payoff values).\n"
            f"Payoff table:\n{payoff}\n"
            f"Zero-range objectives: {np.where(maxobj - minobj == 0)[0].tolist()}")
    intervals = approach_options.get('intervals', 10)
    wm = approach_options.get('wm', 'dirichlet')
    is_cp = solver_opts.get('solver_name', '') in ('ortools_cp', 'cplex_cp', 'picat')

    def _build_nwsm_model(w):
        model_object = instance()
        nwsm_directions = directions.copy()
        nwsm_directions.append('min')
        _apply_features(model_object, nwsm_directions, **solver_opts)
        model_object.features['objective_being_optimized'] = M
        z = model_object.fvar('_z', [range(M)])
        if is_cp:
            SCALE = 10000
            ranges = maxobj - minobj
            scaled_w = w * ranges
            total = scaled_w.sum()
            if total > 0:
                scaled_w = scaled_w / total
            int_w = np.rint(scaled_w * SCALE).astype(int)
            normalized_terms = []
            for k in range(M):
                if directions[k] == 'max':
                    term = int_w[k] * (int(maxobj[k]) - z[k])
                else:
                    term = int_w[k] * (z[k] - int(minobj[k]))
                normalized_terms.append(term)
        else:
            normalized_terms = []
            for k in range(M):
                rng = maxobj[k] - minobj[k]
                if rng == 0:
                    rng = 1.0
                inv_rng = 1.0 / rng
                if directions[k] == 'max':
                    term = (maxobj[k] - z[k]) * inv_rng
                else:
                    term = (z[k] - minobj[k]) * inv_rng
                normalized_terms.append(w[k] * term)
        model_object.obj(sum(normalized_terms))
        for k in range(M):
            model_object.con(z[k] == model_object.features['objectives'][k],
                             name=f'_weighted_{k}')
        return model_object, z

    pareto, variables, ogrs = [], [], []
    if len(weights) == 0:
        progress.update()
        for g in range(intervals + 1):
            w = _generate_weights(M, wm)
            model_object, z = _build_nwsm_model(w)
            _solve_and_collect(model_object, solution_generator)
            healthy = model_object.healthy()
            if not healthy:
                continue
            result = [float(model_object.get_numpy_var('_z')[k]) for k in range(M)]
            w_str = " ".join(f"{wi:.3f}" for wi in w)
            progress.add_step(
                f"Step {g+1}/{intervals+1}",
                f"w=[{w_str}]",
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
                v = _collect_variables(model_object)
                v['_weights'] = w
                variables.append(v)
    else:
        progress.update()
        model_object, z = _build_nwsm_model(np.asarray(weights, dtype=float))
        _solve_and_collect(model_object, solution_generator)
        healthy = model_object.healthy()
        if healthy:
            result = [float(model_object.get_numpy_var('_z')[k]) for k in range(M)]
            progress.add_step(
                "Fixed weights",
                f"w={np.round(weights, 4)}",
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
        raise MultiObjectiveError("NWSM found no feasible Pareto solutions.")
    return np.array(pareto), variables, ogrs
