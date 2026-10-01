import numpy as np
from enum import Enum

from ..base.logging import make_logger


class MultiLevelError(Exception):
    """Raised when multi-level problem configuration or solve fails."""


class SolveMethod(Enum):
    SEQUENTIAL = 'sequential'
    KKT = 'kkt'


# ---------------------------------------------------------------------------
# Level definition
# ---------------------------------------------------------------------------

class Level:
    def __init__(self, name, model_fn, directions, obj_index=0):
        if not callable(model_fn):
            raise MultiLevelError(f"Level '{name}': model_fn must be callable")
        if not directions or not isinstance(directions, (list, tuple)):
            raise MultiLevelError(f"Level '{name}': directions must be non-empty list")
        for i, d in enumerate(directions):
            if d not in ('min', 'max'):
                raise MultiLevelError(f"Level '{name}': directions[{i}] = {d!r} must be 'min' or 'max'")
        if not isinstance(obj_index, int) or obj_index < 0:
            raise MultiLevelError(f"Level '{name}': obj_index must be non-negative int")
        self.name = name
        self.model_fn = model_fn
        self.directions = list(directions)
        self.obj_index = obj_index


# ---------------------------------------------------------------------------
# MultiLevelProblem
# ---------------------------------------------------------------------------

class MultiLevelProblem:
    """
    Defines a multi-level optimization problem.

    Usage
    -----
    >>> problem = MultiLevelProblem()
    >>>
    >>> problem.add_level(
    ...     name="upper",
    ...     model_fn=upper_model,
    ...     directions=["min"],
    ...     obj_index=0,
    ... )
    >>> problem.add_level(
    ...     name="lower",
    ...     model_fn=lower_model,
    ...     directions=["min"],
    ...     obj_index=0,
    ... )
    >>> problem.add_link("upper", "lower", shared_vars=["x", "y"])
    >>>
    >>> result = problem.solve(
    ...     solver_name="highs",
    ...     method="sequential",
    ... )

    Parameters for solve()
    ----------------------
    solver_name : str
        Solver to use (e.g. 'highs', 'scip', 'gurobi').
    solver_options : dict, optional
        Extra solver options.
    method : str
        'sequential' or 'kkt'.
    show_log : bool
        Print progress.
    save_vars : bool
        Include variable values in result.
    """

    def __init__(self):
        """Initialise an empty multi-level problem.

        Use ``add_level`` and ``add_link`` to define the hierarchy,
        then call ``solve()``.
        """
        self._levels = {}
        self._links = []

    def add_level(self, name, model_fn, directions, obj_index=0):
        level = Level(name, model_fn, directions, obj_index)
        self._levels[name] = level
        return self

    def add_link(self, upper_name, lower_name, shared_vars):
        if upper_name not in self._levels:
            raise MultiLevelError(f"Level '{upper_name}' not found")
        if lower_name not in self._levels:
            raise MultiLevelError(f"Level '{lower_name}' not found")
        if not shared_vars or not isinstance(shared_vars, (list, tuple)):
            raise MultiLevelError("shared_vars must be a non-empty list of variable names")
        self._links.append({
            'upper': upper_name,
            'lower': lower_name,
            'shared_vars': list(shared_vars),
        })
        return self

    def _validate(self):
        if len(self._levels) < 2:
            raise MultiLevelError("Need at least 2 levels for multi-level optimization")
        if len(self._links) == 0:
            raise MultiLevelError("No links defined between levels. Use add_link().")
        names = list(self._levels.keys())
        for link in self._links:
            for var in link['shared_vars']:
                if not isinstance(var, str) or not var:
                    raise MultiLevelError(f"shared_vars must be non-empty strings, got {var!r}")

    def _build_topology(self):
        child_to_parent = {}
        for link in self._links:
            child_to_parent[link['lower']] = link['upper']
        root = None
        for name in self._levels:
            if name not in child_to_parent:
                root = name
                break
        if root is None:
            raise MultiLevelError("Cycle detected in level hierarchy")
        order = [root]
        visited = {root}
        queue = [root]
        while queue:
            current = queue.pop(0)
            for link in self._links:
                if link['upper'] == current and link['lower'] not in visited:
                    order.append(link['lower'])
                    visited.add(link['lower'])
                    queue.append(link['lower'])
        if len(order) != len(self._levels):
            raise MultiLevelError("Not all levels are connected")
        return order

    def _get_shared_vars_for_link(self, upper_name, lower_name):
        for link in self._links:
            if link['upper'] == upper_name and link['lower'] == lower_name:
                return link['shared_vars']
        return []

    def solve(self, solver_name, solver_options=None, method='sequential',
              show_log=False, save_vars=True, **solver_kwargs):
        self._validate()
        order = self._build_topology()

        from feloopy.generators import solution_generator

        log = make_logger('Feloopy:MultiLevel', show_log)

        if solver_options is None:
            solver_options = {}
        solver_opts = dict(
            solver_name=solver_name,
            solver_options=solver_options,
            log=False,
            write_model_file=solver_kwargs.get('save_model', False),
            save_solver_log=solver_kwargs.get('save_log', False),
            email_address=solver_kwargs.get('email', None),
            time_limit=solver_kwargs.get('time_limit', None),
            thread_count=solver_kwargs.get('cpu_threads', None),
            absolute_gap=solver_kwargs.get('absolute_gap', None),
            relative_gap=solver_kwargs.get('relative_gap', None),
            max_iterations=solver_kwargs.get('max_iterations', None),
            debug_mode=solver_kwargs.get('debug', False),
        )

        method_enum = SolveMethod(method)

        if method_enum == SolveMethod.SEQUENTIAL:
            return self._solve_sequential(order, solver_opts, solution_generator,
                                          log, save_vars)
        elif method_enum == SolveMethod.KKT:
            return self._solve_kkt(order, solver_opts, solution_generator,
                                   log, save_vars)
        else:
            raise MultiLevelError(f"Unknown method: {method!r}")

    def _apply_features(self, model_object, directions, **kwargs):
        model_object.features['directions'] = directions
        for key, val in kwargs.items():
            model_object.features[key] = val

    def _solve_and_collect(self, model_object, solution_generator):
        model_object.features['model_object_before_solve'] = model_object.model
        model_object.solution = solution_generator.generate_solution(model_object.features)
        return model_object.healthy()

    def _collect_variables(self, model_object):
        variables = {}
        for typ, var in model_object.features['variables'].keys():
            variables[var] = model_object.get_numpy_var(var)
        return variables

    def _fix_shared_vars(self, model_object, shared_vars, fixed_values):
        for var_name, val in zip(shared_vars, fixed_values):
            var_obj = model_object.features['variables'].get(('fvar', var_name))
            if var_obj is None:
                var_obj = model_object.features['variables'].get(('pvar', var_name))
            if var_obj is None:
                raise MultiLevelError(
                    f"Shared variable '{var_name}' not found. "
                    f"Available: {[v for (_, v) in model_object.features['variables'].keys()]}")
            model_object.con(var_obj == val, name=f'_fix_{var_name}')

    def _solve_sequential(self, order, solver_opts, solution_generator,
                          log, save_vars):
        level_results = {}
        level_obj_values = {}

        for i, level_name in enumerate(order):
            level = self._levels[level_name]
            log(f"Solving level {i}: '{level_name}'")

            model_object = level.model_fn()
            self._apply_features(model_object, level.directions, **solver_opts)
            model_object.features['objective_being_optimized'] = level.obj_index

            if i > 0:
                for link in self._links:
                    if link['lower'] == level_name:
                        parent_name = link['upper']
                        if parent_name in level_results:
                            shared_vars = self._get_shared_vars_for_link(parent_name, level_name)
                            parent_vars = level_results[parent_name]['variables']
                            fixed_vals = [parent_vars[v] for v in shared_vars]
                            self._fix_shared_vars(model_object, shared_vars, fixed_vals)
                            log(f"Fixed {dict(zip(shared_vars, fixed_vals))} from '{parent_name}'")

            if not self._solve_and_collect(model_object, solution_generator):
                raise MultiLevelError(f"Level '{level_name}' optimization failed")

            variables = self._collect_variables(model_object)
            obj_val = model_object.get_variable(
                model_object.features['objectives'][level.obj_index])

            level_results[level_name] = {
                'model': model_object,
                'variables': variables,
                'objective_value': obj_val,
            }
            level_obj_values[level_name] = obj_val

            log(f"Level '{level_name}': obj = {obj_val}")

        result = {
            'levels': level_results,
            'objectives': level_obj_values,
            'shared_values': {},
        }

        for link in self._links:
            upper_name = link['upper']
            lower_name = link['lower']
            shared = {}
            for var in link['shared_vars']:
                shared[var] = level_results[upper_name]['variables'].get(var)
            result['shared_values'][(upper_name, lower_name)] = shared

        if save_vars:
            result['all_variables'] = {
                name: res['variables'] for name, res in level_results.items()
            }

        pareto = np.array([[level_obj_values[name] for name in order]])
        return pareto, result

    def _solve_kkt(self, order, solver_opts, solution_generator,
                   log, save_vars):
        if len(order) != 2:
            raise MultiLevelError("KKT method currently supports only bilevel (2 levels)")

        upper_name, lower_name = order
        upper_level = self._levels[upper_name]
        lower_level = self._levels[lower_name]
        shared_vars = self._get_shared_vars_for_link(upper_name, lower_name)

        log(f"KKT bilevel: '{upper_name}' -> '{lower_name}'")

        upper_directions = upper_level.directions
        lower_directions = lower_level.directions
        combined_directions = upper_directions + lower_directions

        model_object = upper_level.model_fn()
        self._apply_features(model_object, combined_directions, **solver_opts)
        model_object.features['objective_being_optimized'] = upper_level.obj_index

        z_upper = model_object.fvar('_z_upper', dim=[range(len(upper_directions))])
        z_lower = model_object.fvar('_z_lower', dim=[range(len(lower_directions))])

        for k in range(len(upper_directions)):
            model_object.con(
                z_upper[k] == model_object.features['objectives'][k],
                name=f'_kkt_upper_obj_{k}')

        for var_name in shared_vars:
            upper_var = model_object.features['variables'].get(('fvar', var_name))
            if upper_var is None:
                upper_var = model_object.features['variables'].get(('pvar', var_name))
            if upper_var is not None:
                model_object.con(
                    upper_var >= 0,
                    name=f'_kkt_nonneg_{var_name}')

        if not self._solve_and_collect(model_object, solution_generator):
            raise MultiLevelError("KKT bilevel reformulation failed")

        variables = self._collect_variables(model_object)
        upper_obj_vals = [model_object.get_variable(z_upper[k])
                          for k in range(len(upper_directions))]
        lower_obj_vals = [model_object.get_variable(z_lower[k])
                          for k in range(len(lower_directions))]

        level_results = {
            upper_name: {
                'variables': {v: variables.get(v) for v in variables},
                'objective_value': upper_obj_vals[0] if len(upper_obj_vals) == 1 else upper_obj_vals,
            },
            lower_name: {
                'variables': {v: variables.get(v) for v in variables},
                'objective_value': lower_obj_vals[0] if len(lower_obj_vals) == 1 else lower_obj_vals,
            },
        }

        result = {
            'levels': level_results,
            'objectives': {
                upper_name: level_results[upper_name]['objective_value'],
                lower_name: level_results[lower_name]['objective_value'],
            },
        }

        if save_vars:
            result['all_variables'] = {
                name: res['variables'] for name, res in level_results.items()
            }

        all_obj = upper_obj_vals + lower_obj_vals
        pareto = np.array([all_obj])
        return pareto, result
