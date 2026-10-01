# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import importlib
import itertools as it
import os
import sys
import time
import warnings
import concurrent.futures
from typing import Literal, Optional, Any
from contextlib import suppress,redirect_stdout
import timeit
import numpy as np
from .helpers._lazy import pl
from .algorithms import *
from .algorithms.exact.multilevel import MultiLevelProblem
from .classes import *
from .classes.math_api import MathAPI
from .classes.linearization import (
    _LINEARIZATION_CACHE,
    _make_cache_key,
    clear_linearization_cache as _clear_linearization_cache,
    linearization_cache_stats as _linearization_cache_stats,
    _AutomaticLinearExpression,
    _AutomaticLinearVariable,
    _AutomaticLinearConstraint,
    _AutomaticVariableArray,
)
from .algorithms.exact.automatic_capture import _AutomaticCaptureModel
from .helpers import *
from .helpers.jlcode import JLCodeHandler
from .helpers.model_utils import (
    _can_pickle,
    _constraint_violation,
    _constraint_violations,
    _evaluate_term_expression,
    _is_solver_expression,
    _salvage_value,
    _sanitize_features,
    _slot_index,
)
from .helpers.reporter import ReportEngine, left_align
from .helpers.solutions import _normalize_init_solutions
from .operators import *
from ._version import __version__, __release_month__, __release_year__
from .helpers.registries import (
    HEURISTIC_ALGORITHMS,
    EXACT_ALGORITHMS,
    UNCERTAINTY_ALGORITHMS,
    CONSTRAINT_ALGORITHMS,
    MOO_ALGORITHMS,
    WEIGHTING_ALGORITHMS,
    RANKING_ALGORITHMS,
    SPECIAL_ALGORITHMS,
)
from .algorithms.exact.sequential import (
    SDMMixin,
    _make_build_model_fn,
    sdm,
)
from .algorithms.madm import cwdea_method, la_method, lp_method


warnings.simplefilter(action='ignore', category=FutureWarning)

__author__ = ['Keivan Tafakkori']

class model(
    SDMMixin,
    MathAPI,
    JLCodeHandler,
    TensorVariableClass,
    TensorVariableCollectionClass,
    MultidimVariableClass,
    MultidimVariableCollectionClass,
    EventVariableClass,
    EventVariableCollectionClass,
    SpecialConstraintClass,
    NormalConstraintClass,
    LinearizationClass,
    ConstraintProgrammingClass,
    CvxpExpressionClass,
    ):
    
    def __init__(
        self,
        method: Literal['constraint', 'convex', 'exact', 'heuristic', 'uncertain'] = 'exact',
        name: str = 'instance',
        interface: Literal['auto', 'copt','cplex','cplex_cp','cvxpy','cylp','casadi','feloopy','gekko','gurobi','indago','linopy','mealpy','mip','niapy','ortools','ortools_cp','picat','picos','pulp','pygad','pymoo','pymprog','pymultiobjective','pyomo','rsome_dro','rsome_ro','xpress','insideopt','insideopt-demo', 'scipy' , 'gams','highs', 'jump', 'mathopt', 'pyoptinterface', 'pyoptinterface.highs', 'pyoptinterface.copt', 'pyoptinterface.mosek', 'pyoptinterface.gurobi', 'uno', 'scip', 'bonmin', 'couenne', 'hexaly', 'mosek'] = 'auto',
        agent: Optional[Any] = None,
        no_scenarios: Optional[int] = None,
        no_agents: Optional[int] = None,
        scenario_ids: Optional[list] = None,
        constraint_ids: Optional[list] = None,
        validate: bool = True,
        auto_linearize: bool = False,
        ):
        
        """
        Creates and initializes the mathematical modeling environment.

        Parameters
        ----------
        name
            Name of the mathematical model.
        method
            Desired solution method.
        interface
            Desired solver interface.
        agent
            Search agent in heuristic optimization.
        no_scenarios
            Number of scenarios in uncertainty handling.
        no_agents
            Number of search agents in heuristic optimization.
        scenario_ids
            Indices of scenarios in uncertainty handling.
        constraint_ids
            Indices of constraints to be considered.
        auto_linearize
            If True, automatically linearize nonlinear expressions for LP/MILP solvers.
            If False (default), variables are raw solver expressions.
        """
        
        if validate: 

            validate_string(
                label="method",
                list_of_allowed_values=['constraint', 'convex', 'exact', 'heuristic', 'uncertain'],
                input_string=method,
                required=True)

            validate_string(
                label="interface",
                list_of_allowed_values=['auto', 'copt','cplex','cplex_cp','cvxpy','scipy','cylp','casadi','feloopy','gekko','gurobi','indago','linopy','mealpy','mip','niapy','ortools','ortools_cp','picat','picos','pulp','pygad','pymoo','pymprog','pymultiobjective','pyomo','rsome_dro','rsome_ro','xpress','insideopt','insideopt-demo', 'gams','highs', 'jump', 'mathopt', 'pyoptinterface', 'pyoptinterface.highs', 'pyoptinterface.copt', 'pyoptinterface.mosek', 'pyoptinterface.gurobi', 'uno', 'scip', 'bonmin', 'couenne', 'hexaly', 'mosek'], 
                input_string=interface,
                required=True)
            
            validate_integer(
                label="no_scenarios", 
                min_value=1, 
                max_value=None, 
                input_integer=no_scenarios, 
                required=False)

            validate_integer(
                label="no_agents", 
                min_value=1, 
                max_value=None, 
                input_integer=no_agents, 
                required=False)
            
            validate_existence(
                label="agent", 
                input_value=agent, 
                condition=True if method=="heuristic" else False)

        self.name = name
        self.method = method
        self.interface = interface
        self.solver = interface
        self.key_vars = []
        self.sensitivity_analyzed = False
        self.inputdata = None
        self.output_decimals = 4
        self.show_tensor = False
        self.show_detailed_tensors = False
        self.data = {}
        self.debug = False
        self.method_was = None
        self.approach = None
        self.em = self
        self.solver_options = {}
        self.solutions = []
        self.mgt = 0
        self.cpt = 0
        self.should_benchmark = False
        self.dataset_size = None
        self.track_history = False
        self.agent = agent
        self.no_scenarios = no_scenarios
        self.no_agents = no_agents
        self.scenario_ids = scenario_ids
        self.constraint_ids = constraint_ids
        self.decoder = None
        self._search_decoder = None
        self._agent_replacement = None

        if self.method in ["constraint", "convex", "uncertain"]:
            self.method_was = self.method
            self.method = "exact"
        else:
            self.method_was = None
        
        self.features = {
            'solution_method': self.method,
            'model_name': self.name,
            'interface_name': self.interface,
            'no_scenarios': self.no_scenarios,
            'no_agents': self.no_agents,
            'scenario_ids': self.scenario_ids,
            'constraint_ids': self.constraint_ids,
            'solver_name': None,
            'constraints': [],
            'constraint_labels': [],
            'objectives': [],
            'objective_labels': [],
            'directions': [],
            'positive_variable_counter': [0, 0],
            'integer_variable_counter': [0, 0],
            'binary_variable_counter': [0, 0],
            'free_variable_counter': [0, 0],
            'event_variable_counter': [0, 0],
            'sequential_variable_counter': [0,0],
            'dependent_variable_counter': [0,0],
            'total_variable_counter': [0, 0],
            'objective_counter': [0, 0],
            'constraint_counter': [0, 0],
            'objective_being_optimized': 0,
            'solver_options': {},
            '_al_cache': {},
            '_terms': {},
            '_deferred_terms': {},
            '_term_order': [],
        }

        if self.features['interface_name'] == 'auto':
            self.features['_auto_interface'] = True
            self.features['interface_name'] = self._auto_select_interface()

        if self.method == 'exact':

            from .generators import model_generator
            self.model = model_generator.generate_model(self.features)
            
            from collections import defaultdict

            self.features.update(
                {
                'model_object': self.model,
                'model_object_before_solve': self.model,
                'variables': {},
                'dimensions': {},
                'variable_type': dict(),
                'variable_bound': dict(),
                'variable_dim': dict(),
                }
            )
            
            self.link_to_interface = self.lti = self._ = self.model

            if auto_linearize:
                from .classes.linearization import needs_auto_linearization
                self.features['auto_linearize'] = needs_auto_linearization(
                    self.features['interface_name'], self.features['interface_name']
                )
            else:
                self.features['auto_linearize'] = False
                    
        if self.method == 'heuristic':
            
            self.features.update(
                {
                'agent_status': self.agent[0],
                'variable_spread': self.agent[2] if self.agent[0] != 'idle' else dict(),
                'variable_type': dict() if self.agent[0] == 'idle' else None,
                'variable_bound': dict() if self.agent[0] == 'idle' else None,
                'variable_dim': dict() if self.agent[0] == 'idle' else None,
                'pop_size': 1 if self.agent[0] == 'idle' else len(self.agent[1]),
                'penalty_coefficient': 0 if self.agent[0] == 'idle' else self.agent[3],
                'vectorized': self.interface in ['feloopy', 'pymoo'],
                }
            )
            if self.agent[0] != 'idle':
                self.agent = self.agent[1].copy()
                
        self.grad_counter=0

        self.binary_variable = self.binary = self.bool = self.add_bool = self.add_binary = self.add_binary_variable = self.boolean_variable = self.add_boolean_variable = self.bvar
        self.positive_variable = self.positive = self.add_positive = self.add_positive_variable = self.pvar
        self.integer_variable = self.integer = self.add_integer = self.add_integer_variable = self.ivar
        self.free_variable = self.free = self.float = self.add_free = self.add_float = self.real = self.add_real = self.add_free_variable = self.fvar
        self.sequential_variable = self.sequence = self.sequential = self.add_sequence = self.add_sequential = self.add_sequential_variable = self.permutation_variable = self.add_permutation_variable = self.svar
        self.positive_tensor_variable = self.positive_tensor = self.add_positive_tensor = self.add_positive_tensor_variable = self.ptvar
        self.binary_tensor_variable = self.binary_tensor = self.add_binary_tensor = self.add_binary_tensor_variable = self.add_boolean_tensor_variable = self.boolean_tensor_variable = self.btvar
        self.integer_tensor_variable = self.integer_tensor = self.add_integer_tensor = self.add_integer_tensor_variable = self.itvar
        self.free_tensor_variable = self.free_tensor = self.float_tensor = self.add_free_tensor = self.add_float_tensor = self.add_free_tensor_variable = self.ftvar
        self.random_variable = self.add_random_variable = self.rvar
        self.random_tensor_variable = self.add_random_tensor_variable = self.rtvar
        self.dependent_variable = self.array = self.add_array = self.add_dependent_variable = self.dvar
        self.objective = self.reward = self.hypothesis = self.fitness = self.goal = self.add_objective = self.loss = self.gain = self.obj
        self.constraint = self.equation = self.add_constraint = self.add_equation = self.st = self.subject_to = self.cb = self.computed_by = self.penalize = self.pen = self.eq = self.con
        self.solve = self.implement = self.run = self.optimize = self.sol
        self.get_obj = self.get_objective
        self.get_stat = self.get_status
        self.get_tensor = self.get_numpy_var
        self.get_var = self.value = self.get = self.get_variable
        self.add_term = self.report_term = self.computed = self.computed_term = self.term
        self.PI = self.pi = np.pi
        self._impl = None

    def __getstate__(self):
        """Detach live solver objects so this instance can cross a process
        boundary (see parallel_search). The plain results (status, objective,
        solution values) are kept as-is; unpicklable formulation pieces
        (solver expressions in ``objectives`` / ``constraints``) are replaced
        by their string form, which is what reports print anyway."""
        state = self.__dict__.copy()
        detached = False
        handled = ('features', 'model', 'link_to_interface', 'lti', '_')
        for attr in ('model', 'link_to_interface', 'lti', '_'):
            value = state.get(attr)
            if value is not None and value is not self and not _can_pickle(value):
                state[attr] = None
                detached = True
        features = state.get('features')
        if isinstance(features, dict):
            features, features_changed = _sanitize_features(
                features, state.get('solutions'))
            state['features'] = features
            detached = detached or features_changed
        for attr, value in list(state.items()):
            if attr in handled or value is None or value is self:
                continue
            if _can_pickle(value):
                continue
            state[attr] = _salvage_value(value, stringify=False)
            detached = True
        if detached:
            state['_detached'] = True
        return state

    def _detached_result(self, indicator, variable=None):
        if indicator == 'status':
            return getattr(self, 'status', None)
        if indicator == 'objective':
            return getattr(self, 'obj_val', None)
        if indicator == 'variable':
            solutions = getattr(self, 'solutions', None)
            if isinstance(solutions, dict):
                names = []
                if isinstance(variable, str):
                    names.append(variable)
                elif isinstance(variable, (tuple, list)) and variable:
                    names.extend(variable)
                    names.append(variable[-1])
                for name in names:
                    if name in solutions:
                        return solutions[name]
        return None

    def _result(self, indicator, variable=None):
        if getattr(self, '_detached', False):
            return self._detached_result(indicator, variable)
        from .generators import result_generator
        return result_generator.get(self.features, self.model, self.solution, indicator, variable)

    def set_agent(self, solutions):
        if self._impl is None:
            return
        norm = _normalize_init_solutions(solutions, self._impl.VariablesBound, self._impl.VariablesSpread)
        if norm is None:
            return
        mo = self._impl.ModelObject
        iface = self._impl.interface_name
        if iface == 'feloopy' and mo is not None and hasattr(mo, 'pi'):
            arr = np.asarray(mo.pi)
            f = mo.f
            n = min(norm.shape[0], arr.shape[0])
            arr[:n, :f] = norm[:n, :f]
            mo.pi = arr
        elif iface == 'mealpy' and mo is not None:
            try:
                from mealpy.utils.agent import Agent
                mo.pop = [mo.generate_agent(pos) for pos in norm]
            except Exception:
                pass
        if norm.shape[0] == 1 and self.agent is not None:
            pop_size = np.asarray(self.agent).shape[0]
            if pop_size > 1:
                norm = np.tile(norm, (pop_size, 1))
        self._agent_replacement = norm

    def get_agent(self):
        if self._impl is None:
            return None
        if self._agent_replacement is not None:
            return self._agent_replacement.copy()
        mo = self._impl.ModelObject
        iface = self._impl.interface_name
        if iface == 'feloopy' and mo is not None and hasattr(mo, 'pi'):
            arr = np.asarray(mo.pi)
            f = mo.f
            return arr[:, :f].copy()
        if iface == 'mealpy' and mo is not None and hasattr(mo, 'pop') and mo.pop is not None:
            try:
                return np.array([a.solution for a in mo.pop])
            except Exception:
                pass
        if self.agent is not None:
            return np.atleast_2d(np.array(self.agent)).copy()
        return None

    def __getitem__(self, agent):
        agent_status = self.features['agent_status']
        vectorized = self.features['vectorized']
        interface_name = self.features['interface_name']
        if agent_status == 'idle':
            return self
        elif agent_status == 'feasibility_check':
            return self._feasibility_check()
        else:
            return self._get_result(vectorized, interface_name)

    def _feasibility_check(self, tol=None) -> str:
        
        """
        Perform a feasibility check based on the model's features.

        Parameters
        ----------
        tol : float, optional
            Absolute tolerance on the penalty norm. Violations smaller than
            ``tol`` are treated as feasible (floating-point noise).
            Defaults to ``features.get('feasibility_tol', 1e-6)``.

        Returns
        -------
        str
            The feasibility status:
            - 'feasible (unconstrained)' if the penalty coefficient is 0.
            - 'infeasible (constrained)' if the penalty exceeds the tolerance.
            - 'feasible (constrained)' otherwise.
        """
        
        if self.features['penalty_coefficient'] == 0:
            return 'feasible (unconstrained)'
        else:
            if tol is None:
                tol = self.features.get('feasibility_tol', 1e-6)
            p = self.penalty
            if isinstance(p, np.ndarray):
                violated = np.any(p > tol)
            else:
                violated = p > tol
            return 'infeasible (constrained)' if violated else 'feasible (constrained)'

    def sets(self,*args):
        
        if len(args)==1:
            return args[0]
        else:
            return it.product(*args)
    
    def fix_ifneeded(self, dims):
        return fix_dims(dims)
    
    def _get_result(self, vectorized: bool, interface_name: str):
        """
        Retrieve the optimization result based on the specified parameters.

        Parameters
        ----------
        vectorized : bool
            A boolean indicating whether the result should be vectorized.
        interface_name : str
            The name of the solver interface.

        Returns
        -------
        ConditionalObject
            The optimization result:
            - If vectorized is True and the interface is 'feloopy', returns the search agent.
            - If vectorized is True and the interface is not 'feloopy', returns the singular result.
            - If vectorized is False, returns the response.
        """
        
        if vectorized:
            return self.agent if interface_name == 'feloopy' else self.sing_result
        else:
            return self.response

    def vstart(self, variable, input_value):

        from .generators import init_generator
        init_generator.generate_init(self.features,variable,input_value,fix=False)

    def tstart(self, name, input_tensor):

        input_tensor = np.array(input_tensor)

        from .generators import init_generator
        for i,j in self.features['variables'].keys():
            if j==name:
                if self.features['dimensions'][j]==0:
                    init_generator.generate_init(self.features,self.features['variables'][(i,j)],input_tensor,fix=False)

                elif len(self.features['dimensions'][j])==1:

                    for k in fix_dims(self.features['dimensions'][j])[0]:
                        init_generator.generate_init(self.features,self.features['variables'][(i,j)][k],input_tensor[k],fix=False)
                else:
                    for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                        init_generator.generate_init(self.features,self.features['variables'][(i,j)][k],input_tensor[k],fix=False)
                            
    def vfix(self, variables, values):
        
        from .generators import init_generator
        init_generator.generate_init(self.features,variables,values,fix=True)

    def tfix(self, name, input_tensor):
        
        input_tensor = np.array(input_tensor)
        
        from .generators import init_generator
        for i,j in self.features['variables'].keys():
            if j==name:
                if self.features['dimensions'][j]==0:
                    init_generator.generate_init(self.features,self.features['variables'][(i,j)],input_tensor,fix=True)
                    
                elif len(self.features['dimensions'][j])==1:
                
                    for k in fix_dims(self.features['dimensions'][j])[0]:
                        init_generator.generate_init(self.features,self.features['variables'][(i,j)][k],input_tensor[k],fix=True)
                else:
                    for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                        init_generator.generate_init(self.features,self.features['variables'][(i,j)][k],input_tensor[k],fix=True)

    def _apply_exact_init(self, init):
        if self.features.get('agent_status') not in (None, 'idle'):
            return
        from .generators import init_generator
        import numpy as _np
        from .operators.fix_operators import fix_dims as _fix_dims

        def _dim_size(dims):
            if isinstance(dims, (int, _np.integer)):
                return int(dims)
            if isinstance(dims, (list, tuple)):
                size = 1
                for s in dims:
                    size *= _dim_size(s)
                return size
            return len(dims)

        if init is None:
            # values staged by vstart/tstart are flushed by sol()
            return

        if not init_generator.is_supported(self.features):
            init_generator.warn_unsupported(self.features, init)
            return

        if isinstance(init, dict):
            init_dicts = [init]
        elif isinstance(init, list) and len(init) > 0 and isinstance(init[0], dict):
            init_dicts = init
        else:
            arr = _np.asarray(init, dtype=float).flatten()
            d = {}
            idx = 0
            for key in self.features['variables'].keys():
                name = key[1] if isinstance(key, tuple) and len(key) > 1 else key
                dim = self.features['dimensions'].get(name, 0)
                if not dim:
                    if idx >= arr.size:
                        raise ValueError(
                            "flat init has %d values but the model has more "
                            "variables than that" % arr.size)
                    d[name] = arr[idx]
                    idx += 1
                else:
                    size = _dim_size(_fix_dims(dim))
                    if idx + size > arr.size:
                        raise ValueError(
                            "flat init has %d values but variable %r alone "
                            "needs %d of them" % (arr.size, name, size))
                    d[name] = arr[idx:idx + size]
                    idx += size
            init_dicts = [d]

        # staged here and applied by flush_init() just before the solve
        init_generator.stage_init_values(self.features, init_dicts)

    def grad(self, value):
        if self.features['agent_status'] != 'idle':            
            self.total_features = int(len(self.agent[:,:-2][0])/2)            
            self.agent[0,self.total_features:2*self.total_features][self.grad_counter] =value
            self.grad_counter+=1
        else:
            pass
    
    def grads(self, values):
        if self.features['agent_status'] != 'idle':
            self.total_features = int(len(self.agent[:,:-2][0])/2)
            self.agent[0,self.total_features:2*self.total_features] =np.reshape(np.array(values),[len(values),])
            self.grad_counter+=len(values)
        else:
            pass

    def maximize(self, expression=0, label=None):
        """Shorthand for obj(expression, direction='max')."""
        self.obj(expression, direction='max', label=label)

    def minimize(self, expression=0, label=None):
        """Shorthand for obj(expression, direction='min')."""
        self.obj(expression, direction='min', label=label)

    def obj(self, expression=0, direction=None, label=None):
        """Register an objective function (features + optional SDM stage cost).

        Supports two syntaxes:

        1. Standard: m.obj(x + y, direction='max')
        2. SDM expression with m.state / m.decision / m.exogenous:
            m.obj(45 * m.min(m.state[0] + m.decision[0], m.exogenous[0])
                  - 30 * m.decision[0], direction='max')
        3. SDM callable: m.obj(lambda s,d,w,t: -(45*min(s[0]+d[0],w[0]) - 30*d[0]), 'min')
        """
        from .classes.linearization import LinearizationProxy
        if isinstance(expression, LinearizationProxy):
            expression = expression._native

        if self.features.get('interface_name') == 'gams':
            try:
                self.features['of'].append(self.features['of'][-1] + 1)
            except:
                self.features['of'] = [0]
            z = self.fvar(f"of{self.features['of'][-1]}")
            self.features['objective_variable'] = z
            expression = expression == self.features['objective_variable']

        if self.features.get('solution_method') == 'exact' or (
                self.features.get('solution_method') == 'heuristic'
                and self.features.get('agent_status') == 'idle'):
            self.features['directions'].append(direction)
            self.features['objectives'].append(expression)
            self.features['objective_labels'].append(label)
            self.features['objective_counter'][0] += 1
            self.features['objective_counter'][1] += 1
        elif self.features.get('solution_method') == 'heuristic' and \
                self.features.get('agent_status') != 'idle':
            self.features['directions'].append(direction)
            if self.features.get('interface_name') == 'mealpy':
                self.features['objectives'].append(float(expression))
            else:
                self.features['objectives'].append(expression)
            self.features['objective_counter'][0] += 1

        if callable(expression) or hasattr(expression, '_tree'):
            self._register_sdm_cost(expression, direction)
        elif not hasattr(self, '_sdm_components'):
            self._sdm_components = {}

    def _auto_select_interface(self):
        """Pick default interface for ``interface='auto'`` (see helpers.problem_detect)."""
        from .helpers.problem_detect import select_interface
        return select_interface()

    def _auto_select_solver(self):
        """Auto-select the best solver based on model characteristics.

        Thin wrapper over ``helpers.problem_detect.select_solver``.
        """
        from .helpers.problem_detect import select_solver
        return select_solver(self.features)

    def _detect_problem_type(self):
        """Detect the problem type (see helpers.problem_detect)."""
        from .helpers.problem_detect import detect_problem_type
        return detect_problem_type(self.features)

    def _compute_type(self, has_mixed_integer, has_continuous, is_linear, is_quadratic, has_nonlinear):
        """Map feature flags to a problem-type label (see helpers.problem_detect)."""
        from .helpers.problem_detect import compute_type
        return compute_type(has_mixed_integer, has_continuous, is_linear, is_quadratic, has_nonlinear)

    def _analyze_expressions(self):
        """Analyze expressions for structure (see helpers.problem_detect)."""
        from .helpers.problem_detect import analyze_expressions
        return analyze_expressions(self.features)

    def sol(self, directions=None, solver=None, solver_options=dict(), obj_id=0, email=None, debug=False, time_limit=None, cpu_threads=None, absolute_gap=None, relative_gap=None, show_log=False, save_log=False, save_model=False, max_iterations=None, obj_operators=[], uncertainty_set_constraints=[], callback=None, auto_linearize=False):
        """
        Solve Command Definition
        ~~~~~~~~~~~~~~~~~~~~~~~~
        To define solver and its settings to solve the problem.

        Args:
            directions (list, optional): please set the optimization directions of the objectives, if not provided before. Defaults to None.
            solver_name (_type_, optional): please set the solver_name. Defaults to None.
            solver_options (dict, optional): please set the solver options using a dictionary with solver specific keys. Defaults to None.
            obj_id (int, optional): please provide the objective id (number) that you wish to optimize. Defaults to 0.
            email (_type_, optional): please provide your email address if you wish to use cloud solvers (e.g., NEOS server). Defaults to None.
            debug (bool, optional): please state if the model should be checked for feasibility or logical bugs. Defaults to False.
            time_limit (seconds, optional): please state if the model should be solved under a specific timelimit. Defaults to None.
            cpu_threads (int, optional): please state if the solver should use a specific number of cpu threads. Defaults to None.
            absolute_gap (value, optional): please state an abolute gap to find the optimal objective value. Defaults to None.
            relative_gap (%, optional): please state a releative gap (%) to find the optimal objective value. Defaults to None.
            obj_operators (list, optional): uncertainty operators for rsome (e.g., ['max'] for RO minmax, ['sup'] for DRO minsup). Defaults to [].
            uncertainty_set_constraints (list, optional): constraints defining the uncertainty set for RO minmax/maxmin. Defaults to [].
            callback (_type_, optional): callback function. Defaults to None.
            auto_linearize (bool, optional): if True, automatically linearize nonlinear expressions for LP/MILP solvers. If False (default), use raw solver variables. Defaults to False.
        """

        _mgt_start = timeit.default_timer()

        solver_options = dict(solver_options or {})

        if self.no_agents!=None:
            if self.features['interface_name'] in ['mealpy' , 'feloopy', 'niapy', 'pygad']:
                solver_options['pop_size'] = self.no_agents

        if len(self.features['objectives']) !=0 and directions is not None and len(directions)!=len(self.features['objectives']):
            raise MultiObjectivityError("The number of directions and the provided objectives do not match.")

        self.features['objective_being_optimized'] = obj_id
        self.features['solver_options'].update(self.solver_options)
        self.features['solver_options'].update(solver_options)
        self.features['debug_mode'] = debug
        self.features['time_limit'] = time_limit
        self.features['thread_count'] = cpu_threads
        self.features['absolute_gap'] = absolute_gap
        self.features['relative_gap'] = relative_gap
        self.features['log'] = show_log
        self.features['write_model_file'] = save_model
        self.features['save_solver_log'] = save_log
        self.features['email_address'] = email
        self.features['max_iterations'] = max_iterations
        self.features['obj_operators'] = obj_operators
        self.features['uncertainty_set_constraints'] = uncertainty_set_constraints
        self.features['callback'] = callback

        _auto_op = self.features.get('_auto_obj_operators')
        if _auto_op and len(obj_operators) == 0:
            self.features['obj_operators'] = [_auto_op]

        _is_cp = self.features.get('interface_name', '') in ['ortools_cp', 'cplex_cp', 'picat']

        if _is_cp:
            solver = self.features['interface_name']
        elif solver is None and self.features['solution_method'] == 'exact':
            solver = self._auto_select_solver()

        self.features['solver_name'] = solver

        from .classes.linearization import needs_auto_linearization
        if auto_linearize:
            self.features['auto_linearize'] = True

        if self.features['solution_method'] == 'exact':
            if self.features.get('_auto_lin_active'):
                self.features['_original_obj_count'] = self.features['objective_counter'][0]
                self._refresh_snapshot_for_sol()
                self.features['_original_variable_count'] = self.features['total_variable_counter'][1]
            self.features['problem_type'] = self._detect_problem_type()
            problem_type = self.features['problem_type']
            has_int = self.features['integer_variable_counter'][0] > 0
            has_bin = self.features['binary_variable_counter'][0] > 0
            has_mixed = has_int or has_bin

            if not _is_cp:
                if problem_type == 'MINLP' and not self.features.get('_auto_lin_active') and solver not in ['scip', 'bonmin', 'couenne']:
                    raise RuntimeError(
                        "MINLP problems require scip, bonmin, or couenne solver. "
                        "Install with: pip install PySCIPOpt (recommended), "
                        "or run: flp setup bonmin"
                    )
                elif solver == 'uno' and has_mixed:
                    raise RuntimeError(
                        "Uno is a continuous NLP solver and cannot handle integer/binary variables. "
                        "Use scip/bonmin/couenne for MINLP, or HiGHS for MILP/IQP/MIQP."
                    )

        try:
            if type(obj_id) != str and directions != None:

                if self.features['directions'][obj_id] == None:

                    self.features['directions'][obj_id] = directions[obj_id]
                for i in range(len(self.features['objectives'])):
                    if i != obj_id:
                        del self.features['directions'][i]
                        del directions[i]
                        del self.features['objectives'][i]
                obj_id = 0

                self.features['objective_counter'] = [1, 1]

            else:

                for i in range(len(self.features['directions'])):

                    self.features['directions'][i] = directions[i]
        except:
            pass

        if self.features['solution_method'] == 'exact':
            for _i, _d in enumerate(self.features['directions']):
                if _d is None:
                    _obj_label = self.features['objective_labels'][_i] or f"objective {_i}"
                    raise DirectionError(
                        f"Direction not specified for {_obj_label}. "
                        f"Please provide a direction via one of:\n"
                        f"  1) direction parameter in m.obj(..., direction='min') or m.obj(..., direction='max')\n"
                        f"  2) m.minimize(...) or m.maximize(...) instead of m.obj(...)\n"
                        f"  3) directions parameter in flp.search(..., directions=['min'])"
                    )

        self.mgt = timeit.default_timer() - _mgt_start

        match self.features['solution_method']:

            case 'exact':

                self.features['model_object_before_solve'] = self.model

                from .generators import init_generator

                init_generator.flush_init(self.features)

                from .generators import solution_generator

                if self.features['constraint_ids'] is not None:
                    original_constraints = self.features['constraints']
                    original_labels = self.features['constraint_labels']
                    _ids = set(self.features['constraint_ids'])

                    for _bbase, _bentry in (self.features.get('constraint_batches') or {}).items():
                        if _bbase in _ids:
                            for _bidx, _blbl in _bentry.get('elements', ()):
                                _ids.add(_blbl)
                    filtered_constraints = []
                    filtered_labels = []
                    for constraint, label in zip(original_constraints, original_labels):
                        if label in _ids:
                            filtered_constraints.append(constraint)
                            filtered_labels.append(label)
                    self.features['constraints'] = filtered_constraints
                    self.features['constraint_labels'] = filtered_labels

                try:
                    if len(self.features['objectives'])==0:
                        self.obj()
                        self.features['objective_counter'][1] = 0
                        self.features['directions'] = ["nan"]
                        self.features['solver_name'] = directions
        
                    self.solution = solution_generator.generate_solution(
                        self.features)
                
                except Exception as _sol_err:

                    if len(self.features['objectives'])==0:
                        self.obj()
                        self.features['objective_counter'][1] = 0
                        self.features['directions'] = ["min"]
                        self.features['solver_name'] = directions
                        self.solution = solution_generator.generate_solution(self.features)
                    else:
                        raise _sol_err
                finally:

                    if self.features['constraint_ids'] is not None:
                        self.features['constraints'] = original_constraints
                        self.features['constraint_labels'] = original_labels
                    
                try:
                    self.obj_val = self.get_objective()
                except Exception:
                    pass
                try:
                    self.status = self.get_status()
                except Exception:
                    pass
                try:
                    self.cpt = self.get_time()
                except Exception:
                    pass

                if self.features['debug_mode'] and not self.healthy():
                    self.features['debug_info'] = self.find_iis()

                if not hasattr(self, 'solutions') or not self.solutions or self.solutions == []:
                    self.solutions = {}
                    for typ, var in self.em.features.get('variables', {}).keys():
                        if typ == 'evar':
                            continue
                        val = None
                        try:
                            val = self.em.get_numpy_var(var)
                        except Exception:
                            pass
                        if val is None:
                            try:
                                _var_obj = self.em.features['variables'][(typ, var)]
                                if hasattr(_var_obj, 'solution_value'):
                                    val = _var_obj.solution_value
                                elif hasattr(_var_obj, 'get_value'):
                                    val = _var_obj.get_value()
                            except Exception:
                                pass
                        if isinstance(val, np.ndarray):
                            if not np.all(np.isnan(val)):
                                self.solutions[var] = val
                        elif val is not None:
                            self.solutions[var] = val

            case 'heuristic':

                if self.features['agent_status'] == 'idle':

                    "Do nothing"

                    self.current_min = [-np.inf for i in range(len(directions))]
                    self.current_max = [ np.inf for i in range(len(directions))]
                    self.current_ave = [ np.inf for i in range(len(directions))]
                    self.current_std = [ np.inf for i in range(len(directions))]
                    
                    if len(self.current_min)==1 and self.features["agent_status"] != 'feasibility_check':
                        self.current_min = self.current_min[0]
                        self.current_max = self.current_max[0]
                        self.current_ave = np.inf
                        self.current_std = np.inf

                else:
                    
                    if self.features['penalty_coefficient'] == 0 and len(self.features['constraints']) != 0:
                        raise ValueError(f"'penalty_coefficient' must be greater than zero for constrained environments.")
                    
                    if self.features['vectorized']:    
                        if self.features['interface_name']=='feloopy':
                            self.penalty = np.zeros(np.shape(self.agent)[0])
                            if self.features['penalty_coefficient'] != 0 and len(self.features['constraints']) >= 1:
                                _pop_n = np.shape(self.agent)[0]
                                _con_cols = []
                                for _ci in self.features['constraints']:
                                    _c = np.reshape(_ci, [_pop_n, 1]) if np.ndim(_ci) < 2 else np.asarray(_ci)
                                    _con_cols.append(_constraint_violation(_c))
                                _con_cols.append(np.zeros(shape=(_pop_n, 1)))
                                _con_arr = np.concatenate(_con_cols, axis=1)
                                self.penalty = np.sqrt(np.sum(_con_arr ** 2, axis=1))
                                self.agent[np.where(self.penalty == 0), -2] = 1
                                self.agent[np.where(self.penalty > 0), -2] = -1
                            else:
                                self.agent[:, -2] = 2

                            if type(obj_id) != str:
                                term = np.reshape(self.features['objectives'][obj_id], [self.agent.shape[0],])
                                _pen_val = self.features['penalty_coefficient'] * (self.penalty)**2
                                if directions[obj_id] == 'max':
                                    self.agent[:, -1] = term - _pen_val
                                if directions[obj_id] == 'min':
                                    self.agent[:, -1] = term + _pen_val
                                _feas = self.penalty == 0
                                _infeas = ~_feas
                                if np.any(_feas) and np.any(_infeas):
                                    if directions[obj_id] == 'max':
                                        _worst_feas = np.min(self.agent[_feas, -1])
                                        self.agent[_infeas, -1] = _worst_feas - self.penalty[_infeas] - 1
                                    else:
                                        _worst_feas = np.max(self.agent[_feas, -1])
                                        self.agent[_infeas, -1] = _worst_feas + self.penalty[_infeas] + 1

                                if self.features["agent_status"] !=  'feasibility_check': 
                                    self.current_min = np.min(self.agent[:, -1])
                                    self.current_max = np.max(self.agent[:, -1])
                                    self.current_ave = np.mean(self.agent[:, -1])
                                    self.current_std = np.std(self.agent[:, -1])

                            else:
                                self.agent[:, -1] = 0
                                total_obj = self.features['objective_counter'][0]
                                self.features['objectives'] = np.array(self.features['objectives']).T
                                for i in range(self.features['objective_counter'][0]):
                                    if directions[i] == 'max':
                                        self.agent[:, -2-total_obj+i] = self.features['objectives'][:,i] - self.features['penalty_coefficient'] * (self.penalty)**2
                                    if directions[i] == 'min':
                                        self.agent[:, -2-total_obj+i] = self.features['objectives'][:,i] + self.features['penalty_coefficient'] * (self.penalty)**2
                                
                                if self.features["agent_status"] !=  'feasibility_check': 
                                    self.current_min = np.min(self.agent[:, -2-total_obj:-2], axis = 0)
                                    self.current_max = np.max(self.agent[:, -2-total_obj:-2], axis = 0)
                                    self.current_ave = np.mean(self.agent[:, -2-total_obj:-2], axis =0)
                                    self.current_std = np.std(self.agent[:, -2-total_obj:-2], axis = 0)

                        else:

                            self.penalty = np.zeros(np.shape(self.agent)[0])

                            if self.features['penalty_coefficient'] != 0 and len(self.features['constraints']) >= 1:
                                _pop_n = np.shape(self.agent)[0]
                                _con_cols = []
                                for _ci in self.features['constraints']:
                                    _c = np.reshape(_ci, [_pop_n, 1]) if np.ndim(_ci) < 2 else np.asarray(_ci)
                                    _con_cols.append(_constraint_violation(_c))
                                _con_cols.append(np.zeros(shape=(_pop_n, 1)))
                                _con_arr = np.concatenate(_con_cols, axis=1)
                                self.penalty = np.sqrt(np.sum(_con_arr ** 2, axis=1))

                            if type(obj_id) != str:

                                _pen_val = self.features['penalty_coefficient'] * (self.penalty)**2
                                if directions[obj_id] == 'max':
                                    self.sing_result = np.reshape(self.features['objectives'][obj_id], [self.agent.shape[0],]) - _pen_val
                                if directions[obj_id] == 'min':
                                    self.sing_result = np.reshape(self.features['objectives'][obj_id], [self.agent.shape[0],]) + _pen_val
                                _feas = self.penalty == 0
                                _infeas = ~_feas
                                if np.any(_feas) and np.any(_infeas):
                                    if directions[obj_id] == 'max':
                                        _worst_feas = np.min(self.sing_result[_feas])
                                        self.sing_result[_infeas] = _worst_feas - self.penalty[_infeas] - 1
                                    else:
                                        _worst_feas = np.max(self.sing_result[_feas])
                                        self.sing_result[_infeas] = _worst_feas + self.penalty[_infeas] + 1
                                
                                if self.features["agent_status"] !=  'feasibility_check': 
                                    self.current_min = np.min(self.sing_result)
                                    self.current_max = np.max(self.sing_result)
                                
                            else:

                                total_obj = self.features['objective_counter'][0]
                                self.sing_result = []
                            
                                n_objs = int(self.features['objective_counter'][0])

                                for i in range(n_objs):
                                    obj = np.array(self.features['objectives'][i])
                                    pen = np.array(self.features['penalty_coefficient']) * (np.array(self.penalty) ** 2)
                                    if obj.ndim == 2 and obj.shape[1] == 1 and pen.ndim == 1:
                                        pen = pen[:, np.newaxis]
                                    elif obj.ndim == 1 and pen.ndim == 2 and pen.shape[1] == 1:
                                        pen = pen.ravel()
                                    if directions[i] == 'max':
                                        result_i = obj - pen
                                    else:
                                        result_i = obj + pen
                                    self.sing_result.append(result_i)
                                if self.features["agent_status"] !=  'feasibility_check': 
                                    self.current_min = np.atleast_1d(np.min(np.array(self.sing_result), axis = 0))
                                    self.current_max = np.atleast_1d(np.max(np.array(self.sing_result), axis = 0))

                    else:

                        self.penalty = 0
                        
                        if len(self.features['constraints']) >= 1:
                            _raw = np.concatenate(
                                [_constraint_violations(_c)
                                 for _c in self.features['constraints']])
                            if _raw.size:
                                self.penalty = float(np.sqrt(np.sum(_raw ** 2)))

                        if type(obj_id) != str:

                            if directions[obj_id] == 'max':
                                self.response = self.features['objectives'][obj_id] - \
                                    self.features['penalty_coefficient'] * \
                                    (self.penalty-0)**2

                            if directions[obj_id] == 'min':
                                self.response = self.features['objectives'][obj_id] + \
                                    self.features['penalty_coefficient'] * \
                                    (self.penalty-0)**2
                            if self.features["agent_status"] !=  'feasibility_check': 
                                self.current_min = np.min(self.response)
                                self.current_max = np.max(self.response)

                        else:

                            total_obj = self.features['objective_counter'][0]

                            self.response = [None for i in range(total_obj)]

                            for i in range(total_obj):

                                if directions[i] == 'max':

                                    self.response[i] = self.features['objectives'][i] - \
                                        self.features['penalty_coefficient'] * \
                                        (self.penalty)**2

                                if directions[i] == 'min':

                                    self.response[i] = self.features['objectives'][i] + \
                                        self.features['penalty_coefficient'] * \
                                        (self.penalty)**2
                            if self.features["agent_status"] !=  'feasibility_check': 
                                self.current_min = np.atleast_1d(np.min(np.array(self.response), axis = 0))
                                self.current_max = np.atleast_1d(np.max(np.array(self.response), axis = 0))

    def healthy(self):
        def _haystack(status):
            parts = [str(status)]
            if hasattr(status, 'value'):
                parts.append(str(status.value))
            return " ".join(parts).lower()

        try:
            status = _haystack(self.get_status())
            return ('optimal' in status or 'feasible' in status or 'succ' in status) and 'infeasible' not in status and 'unsucc' not in status and 'not optimal' not in status
        except:
            try:
                status = _haystack(self.get_status())
                return ('feasible' in status or 'optimal' in status) and 'infeasible' not in status
            except:
                return False

    def build_incremental(self, directions=None, obj_index=0,
                          solver_name=None, solver_options=None,
                          time_limit=None, thread_count=None,
                          absolute_gap=None, relative_gap=None,
                          log=False, debug=False):
        """Build a solver-independent incremental model.

        Returns an ``IncrementalModel`` that wraps this model's native
        solver object and provides incremental operations (add constraints,
        fix/unfix variables, re-solve) without rebuilding from scratch.

        Parameters
        ----------
        directions : list of str, optional
            Optimization directions, e.g. ``['min']``.
        obj_index : int
            Which objective to optimize (default 0).
        solver_name : str, optional
            Solver engine name.
        solver_options : dict, optional
            Solver-specific options.
        time_limit : float, optional
            Time limit in seconds.
        thread_count : int, optional
            CPU threads.
        absolute_gap : float, optional
            Absolute MIP gap.
        relative_gap : float, optional
            Relative MIP gap.
        log : bool
            Show solver output.
        debug : bool
            Debug mode.

        Returns
        -------
        IncrementalModel
            Solver-independent incremental model wrapper.
        """
        from .classes.incremental import IncrementalModel
        return IncrementalModel(
            self, directions=directions, obj_index=obj_index,
            solver_name=solver_name, solver_options=solver_options,
            time_limit=time_limit, thread_count=thread_count,
            absolute_gap=absolute_gap, relative_gap=relative_gap,
            log=log, debug=debug)

    # Get values

    def get_variable(self, variable_with_index):
        from .generators import result_generator
        from .classes.linearization import LinearizationProxy
        if isinstance(variable_with_index, LinearizationProxy):
            variable_with_index = variable_with_index.native
        if getattr(self, '_detached', False):
            return self._detached_result('variable', variable_with_index)
        if hasattr(self, 'model') and self.model is not None:
            return result_generator.get(self.features, self.model, self.solution, 'variable', variable_with_index)
        if hasattr(self, '_impl') and self._impl is not None:
            _name = variable_with_index if isinstance(variable_with_index, str) else variable_with_index[0]
            return self._impl.get_numpy_var(_name)
        if hasattr(self, 'model') and getattr(self, 'solution', None) is not None:
            return result_generator.get(self.features, self.features, self.solution, 'variable', variable_with_index)
        return None

    def get_rc(self, variable_with_index):
        from .classes.linearization import LinearizationProxy
        if isinstance(variable_with_index, LinearizationProxy):
            variable_with_index = variable_with_index.native
        return self._result('rc', variable_with_index)

    def _gather_constraint_batch(self, indicator, label):
        """Return an index tensor of results for a constraint batch, or None.
        """
        registry = (getattr(self, 'features', None) or {}).get('constraint_batches') or {}
        entry = registry.get(label)
        if not entry:
            return None
        elements = entry.get('elements') or []
        if not elements:
            return None
        if len(elements) == 1:
            return self._result(indicator, elements[0][1])
        vals = []
        for _idx, _lbl in elements:
            try:
                v = self._result(indicator, _lbl)
            except Exception:
                v = None
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        idx_list = [e[0] for e in elements]
        _has_idx = all(isinstance(i, tuple) for i in idx_list)
        if _has_idx:
            _rank = len(idx_list[0])
            if all(len(i) == _rank for i in idx_list):
                _shape = tuple(max(i[d] for i in idx_list) + 1
                               for d in range(_rank))
                arr = np.full(_shape, np.nan, dtype=float)
                for i, v in zip(idx_list, vals):
                    arr[i] = v
                return arr
        return np.array(vals, dtype=float)

    def get_dual(self, constraint_label_with_index, tensor=True):
        if tensor and isinstance(constraint_label_with_index, str):
            _batch = self._gather_constraint_batch('dual', constraint_label_with_index)
            if _batch is not None:
                return _batch
        return self._result('dual', constraint_label_with_index)

    def get_iis(self):
        """Format IIS / infeasibility explanation as a string."""
        from .classes.iis import IISFinder
        return IISFinder(self).get_iis()

    def find_iis(self, callback=None):
        """Return constraint labels in an IIS (or violated set)."""
        from .classes.iis import IISFinder
        return IISFinder(self).find_iis(callback)

    def get_slack(self, constraint_label_with_index, tensor=True):
        if tensor and isinstance(constraint_label_with_index, str):
            _batch = self._gather_constraint_batch('slack', constraint_label_with_index)
            if _batch is not None:
                return _batch
        return self._result('slack', constraint_label_with_index)

    def get_objective(self):
        return self._result('objective', None)

    def get_status(self):
        return self._result('status', None)

    def get_time(self):
        return self._result('time', None)

    def get_bound(self):
        try:
            return self._result('bound', None)
        except:
            return None

    def get_ogr(self):
        try:
            return self._result('ogr', None)
        except:
            return None

    def get_density(self):
        import numbers
        import numpy as np

        def _count(val, path):
            if isinstance(val, np.ndarray):
                return int(np.count_nonzero(val))
            if isinstance(val, numbers.Number) or isinstance(val, np.generic):
                return int(val != 0)
            if isinstance(val, dict):
                total = 0
                for subkey, subval in val.items():
                    total += _count(subval, path + [subkey])
                return total
            try:
                import xarray
                if isinstance(val, xarray.DataArray):
                    return int(np.count_nonzero(val.values))
            except ImportError:
                pass
            raise TypeError(
                f"Unsupported type at {'/'.join(path)}: {type(val).__name__}; "
                "expected ndarray, dict, or numeric scalar."
            )

        if not hasattr(self, "solutions") or not self.solutions:
            return "n/a"

        total_nonzeros = 0
        solutions = self.solutions
        if isinstance(solutions, list):
            for sol in solutions:
                if isinstance(sol, dict):
                    for key, value in sol.items():
                        total_nonzeros += _count(value, [key])
            total_nonzeros = total_nonzeros // len(solutions) if solutions else 0
        else:
            for key, value in solutions.items():
                total_nonzeros += _count(value, [key])
        return total_nonzeros

    def get_start(self, invterval_variable):

        if self.features['interface_name'] == 'cplex_cp':
            return self.solution[0].get_var_solution(invterval_variable).get_start()
        
        if self.features['interface_name'] == 'ortools_cp':
            solver = self.solution[0][1]
            return solver.Value(invterval_variable.StartExpr())

        if self.features['interface_name'] == 'picat':
            if isinstance(invterval_variable, dict) and 'start' in invterval_variable:
                start_var = invterval_variable['start']
                return self.get_variable(start_var) if not isinstance(start_var, (int, float)) else start_var
            return None

    def get_interval(self, invterval_variable):

        if self.features['interface_name'] == 'cplex_cp':
            return self.solution[0].get_var_solution(invterval_variable)
        
        if self.features['interface_name'] == 'ortools_cp':
            solver = self.solution[0][1]
            return {
                'start': solver.Value(invterval_variable.StartExpr()),
                'end': solver.Value(invterval_variable.EndExpr()),
                'size': solver.Value(invterval_variable.SizeExpr()),
            }

        if self.features['interface_name'] == 'picat':
            if isinstance(invterval_variable, dict) and 'start' in invterval_variable:
                start_var = invterval_variable['start']
                end_var = invterval_variable['end']
                size_val = invterval_variable['size']
                start_val = self.get_variable(start_var) if not isinstance(start_var, (int, float)) else start_var
                end_val = self.get_variable(end_var) if not isinstance(end_var, (int, float)) else end_var
                return {
                    'start': start_val,
                    'end': end_val,
                    'size': size_val,
                }
            return None

    def get_end(self, invterval_variable):

        if self.features['interface_name'] == 'cplex_cp':
            return self.solution[0].get_var_solution(invterval_variable).get_end()
        if self.features['interface_name'] == 'ortools_cp':
            solver = self.solution[0][1]
            return solver.Value(invterval_variable.EndExpr())
        if self.features['interface_name'] == 'picat':
            if isinstance(invterval_variable, dict) and 'end' in invterval_variable:
                end_var = invterval_variable['end']
                return self.get_variable(end_var) if not isinstance(end_var, (int, float)) else end_var
            return None

    def dis_time(self):

        hour = round((self.get_time()), 3) % (24 * 3600) // 3600
        min = round((self.get_time()), 3) % (24 * 3600) % 3600 // 60
        sec = round((self.get_time()), 3) % (24 * 3600) % 3600 % 60

        print(f"cpu time [{self.features['interface_name']}]: ", self.get_time(
        )*10**6, '(microseconds)', "%02d:%02d:%02d" % (hour, min, sec), '(h, m, s)')

    def append_full_report(self, filename="result.txt", dir='./results/texts/', **kwargs):
        path = dir + filename
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.isfile(path):
            open(path, 'w').close()
        with open(path, 'a', encoding='utf-8') as f:
            with redirect_stdout(f):
                self.full_report(**kwargs)
                
    def append_report(self, filename="result.txt", dir='./results/texts/', **kwargs):
        path = dir + filename
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.isfile(path):
            open(path, 'w').close()
        with open(path, 'a', encoding='utf-8') as f:
            with redirect_stdout(f):
                self.report(**kwargs)
                               
    def write_full_report(self, filename="result.txt", dir='./results/texts/', **kwargs):
        path = dir + filename
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.isfile(path):
            open(path, 'w').close()
        with open(path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                self.full_report(**kwargs)

    def write_report(self, filename="result.txt", dir='./results/texts/', **kwargs):
        path = dir + filename
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.isfile(path):
            open(path, 'w').close()
        with open(path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                self.report(**kwargs)
            
    def full_report(self, show_elements=False):
        
        self.report(full=True, 
                    skip_system_information=False, 
                    show_elements=show_elements)
            
    def clean_report(self, **kwargs):

        clear_console()
        self.report(**kwargs)
        
    def report(self, style=1, skip_system_information=True, show_elements=True, width=78, skip=False, full=False, save=None, copy_to_clipboard=False, show_tensors=None, diagnostics=False):
        if show_tensors is not None:
            show_elements = not show_tensors
        self.number_of_objectives = self.features['objective_counter'][0]
        self.directions = self.features.get('directions', [])
        if self.features.get('solver_name'):
            self.solver = self.features['solver_name']
        if not hasattr(self, 'objective_values') or self.objective_values is None:
            try:
                self.objective_values = np.array([[self.get_obj()]])
            except:
                self.objective_values = np.array([[None]])
        if not hasattr(self, 'solutions') or not self.solutions:
            self.solutions = {}
            for i, j in self.features.get('variables', {}).keys():
                if i != 'evar':
                    try:
                        val = self.get_numpy_var(j)
                        if isinstance(val, np.ndarray):
                            if not np.all(np.isnan(val)):
                                self.solutions[j] = val
                        elif val is not None:
                            self.solutions[j] = val
                    except:
                        pass
        ReportEngine(self).report_search(style=style, skip_system_information=skip_system_information, show_elements=show_elements, width=width, skip=skip, full=full, save=save, copy_to_clipboard=copy_to_clipboard, diagnostics=diagnostics)
        return self

    def get_numpy_var(self, var_name, dual=False, slack=False, reduced_cost=False):

        if self.features["interface_name"]=="jump":
            return self.get(var_name)

        if getattr(self, '_impl', None) is not None and not (dual or slack or reduced_cost):
            return self._impl.get_numpy_var(var_name)

        _is_rsome = self.features.get('interface_name', '') in ['rsome_ro', 'rsome_dro']
        _interface = self.features.get('interface_name', '')
        _is_exact_capable = _interface in ['highs', 'ortools', 'gurobi', 'cplex', 'pyomo', 'linopy', 'hexaly', 'mosek', 'picat', 'uno', 'scip', 'bonmin', 'couenne', 'gams', 'casadi', 'pymprog'] or 'pyoptinterface' in _interface
        _exact = (self.features.get('solution_method') == 'exact' or _interface == 'picat') and _is_exact_capable
        _getter = self.get_variable if _exact else self.get

        def _get_var(i, j, k=None):
            if _exact:
                if k is not None:
                    idx_str = f"{j}[{k}]"
                else:
                    idx_str = j
                return self.get_variable(idx_str)
            elif _is_rsome:
                var_obj = self.features['variables'][(i,j)]
                if isinstance(var_obj, dict):
                    if k is not None:
                        val = var_obj[k].get()
                        return val.item() if hasattr(val, 'item') else val
                    else:
                        return {kk: (vv.get().item() if hasattr(vv.get(), 'item') else vv.get()) for kk, vv in var_obj.items()}
                full_val = var_obj.get()
                if k is not None:
                    return full_val[k]
                else:
                    return full_val
            else:
                if k is not None:
                    try:
                        return _getter(self.features['variables'][(i,j)][k])
                    except:
                        return _getter(self.features['variables'][(i,j)])[k]
                else:
                    return _getter(self.features['variables'][(i,j)])

        if not dual and not slack:
            for i,j in self.features['variables'].keys():
                if j==var_name:
                    if self.features['dimensions'][j]==0:
                        output = _get_var(i, j)
                    elif len(self.features['dimensions'][j])==1 or isinstance(self.features['dimensions'][j],set):
                        if isinstance(self.features['dimensions'][j],list):
                            output = np.zeros(shape=len(fix_dims(self.features['dimensions'][j])[0]))
                            for k in fix_dims(self.features['dimensions'][j])[0]:
                                val = _get_var(i, j, k)
                                if val is not None:
                                    output[k] = val
                        else:
                            output = {}
                            for k in self.features['dimensions'][j]:
                                val = _get_var(i, j, k)
                                if val is not None:
                                    output[k] = val
                    else:
                        output = np.zeros(shape=tuple([len(dim) for dim in fix_dims(self.features['dimensions'][j])]))
                        for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                            val = _get_var(i, j, k)
                            if val is not None:
                                output[k] = val

        if reduced_cost:
            
    
            for i,j in self.features['variables'].keys():
                if j==var_name:
                    if self.features['dimensions'][j]==0:
                        output = self.get_rc(self.features['variables'][(i,j)])
                    elif len(self.features['dimensions'][j])==1:
                        output = np.zeros(shape=len(fix_dims(self.features['dimensions'][j])[0]))
                        for k in fix_dims(self.features['dimensions'][j])[0]:
                            try:
                                output[k] = self.get_rc(self.features['variables'][(i,j)][k])
                            except:
                                output[k] = self.get_rc(self.features['variables'][(i,j)])[k]
                    else:
                        output = np.zeros(shape=tuple([len(dim) for dim in fix_dims(self.features['dimensions'][j])]))
                        for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                            try:
                                output[k] = self.get_rc(self.features['variables'][(i,j)][k])
                            except:
                                output[k] =  self.get_rc(self.features['variables'][(i,j)])[k]            
        if dual:
            
            output = []
            for i in self.features['constraint_labels']:
                if var_name in i:
                    output.append(self.get_dual(i, tensor=False))
            output = np.array(output)
        
        if slack:

            output = []
            for i in self.features['constraint_labels']:
                if var_name in i:
                    output.append(self.get_slack(i, tensor=False))
            output = np.array(output)
                        
        return output

    def get_var_at(self, var_name, index=None):
        """Value of a single element of ``var_name`` (``None`` *index*: scalars).
        """
        if self.features["interface_name"] == "jump":
            out = self.get(var_name)
            if index is None:
                return out
            try:
                return out[index]
            except Exception:
                return None

        _is_rsome = self.features.get('interface_name', '') in ['rsome_ro', 'rsome_dro']
        _interface = self.features.get('interface_name', '')
        _is_exact_capable = _interface in ['highs', 'ortools', 'gurobi', 'cplex', 'pyomo', 'linopy', 'hexaly', 'mosek', 'picat', 'uno', 'scip', 'bonmin', 'couenne', 'gams', 'casadi', 'pymprog'] or 'pyoptinterface' in _interface
        _exact = (self.features.get('solution_method') == 'exact' or _interface == 'picat') and _is_exact_capable
        _getter = self.get_variable if _exact else self.get

        if _exact:
            # exact interfaces index straight into the result lookup
            if index is None:
                return self.get_variable(var_name)
            return self.get_variable(f"{var_name}[{index}]")

        for i, j in self.features['variables'].keys():
            if j != var_name:
                continue
            var_obj = self.features['variables'][(i, j)]
            if _is_rsome:
                if isinstance(var_obj, dict):
                    if index is None:
                        return {kk: (vv.get().item()
                                     if hasattr(vv.get(), 'item')
                                     else vv.get())
                                for kk, vv in var_obj.items()}
                    if index not in var_obj:
                        return None
                    val = var_obj[index].get()
                    return val.item() if hasattr(val, 'item') else val
                full_val = var_obj.get()
                if index is None:
                    return full_val
                try:
                    return full_val[index]
                except Exception:
                    return None
            if index is None:
                return _getter(var_obj)
            try:
                return _getter(var_obj[index])
            except Exception:
                try:
                    return _getter(var_obj)[index]
                except Exception:
                    return None
        return None

    def decision_information_print(self,status, show_tensors, show_detailed_tensors, box_width=88):
        
        if show_detailed_tensors: show_tensors=True
        
        _interface = self.features.get('interface_name', '')
        _is_exact_capable = _interface in ['highs', 'ortools', 'gurobi', 'cplex', 'pyomo', 'linopy', 'hexaly', 'mosek', 'picat', 'uno', 'scip', 'bonmin', 'couenne', 'gams', 'casadi', 'pymprog'] or 'pyoptinterface' in _interface
        _exact = (self.features.get('solution_method') == 'exact' or _interface == 'picat') and _is_exact_capable

        if not show_tensors:

            for i,j in self.features['variables'].keys():
                if i!='evar':
                    if _exact:
                        if self.features['dimensions'][j] == 0:
                            val = self.get_variable(j)
                            if val not in [0, None]:
                                print(f"{j} = {val}")
                        elif len(self.features['dimensions'][j])==1:
                            if type(self.features['dimensions'][j])==set:
                                for k in self.features['dimensions'][j]:
                                    val = self.get_variable(f"{j}[{k}]")
                                    if val not in [0, None]:
                                        if "[" in str(k): index = k
                                        else: index = f"[{k}]"
                                        print(f"{j}{index} = {val}".replace("(", "").replace(")", ""))
                            else:  
                                for k in fix_dims(self.features['dimensions'][j])[0]:
                                    val = self.get_variable(f"{j}[{k}]")
                                    if val not in [0, None]:
                                        print(f"{j}[{k}] = {val}")
                        else:
                            if type(self.features['dimensions'][j])==set:
                                for k in self.features['dimensions'][j]:
                                    val = self.get_variable(f"{j}[{k}]")
                                    if val not in [0, None]:
                                        if "[" in str(k): index = k
                                        else: index = f"[{k}]"
                                        print(f"{j}{index} = {val}".replace("(", "").replace(")", ""))
                            else:
                                for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                                    val = self.get_variable(f"{j}[{k}]")
                                    if val not in [0, None]:
                                        print(f"{j}[{k}] = {val}".replace("(", "").replace(")", ""))
                    else:
                        if self.features['dimensions'][j] == 0:
                            if self.get(self.features['variables'][(i,j)]) not in [0, None]:
                                print(f"{j} = {self.get(self.features['variables'][(i,j)])}")
                        elif len(self.features['dimensions'][j])==1:
                            if type(self.features['dimensions'][j])==set:
                                for k in self.features['dimensions'][j]:
                                    if self.get(self.features['variables'][(i,j)][k]) not in [0, None]:
                                        if "[" in str(k): index = k
                                        else: index = f"[{k}]"
                                        print(f"{j}{index} = {self.get(self.features['variables'][(i,j)][k])}".replace("(", "").replace(")", ""))
                            else:  
                                try:
                                    for k in fix_dims(self.features['dimensions'][j])[0]:
                                        if self.get(self.features['variables'][(i,j)][k]) not in [0, None]:
                                            print(f"{j}[{k}] = {self.get(self.features['variables'][(i,j)][k])}")
                                except:
                                    for k in fix_dims(self.features['dimensions'][j])[0]:
                                        if self.get(self.features['variables'][(i,j)])[k] not in [0, None]:
                                            print(f"{j}[{k}] = {self.get(self.features['variables'][(i,j)])[k]}")
                        else:
                            if type(self.features['dimensions'][j])==set:
                                for k in self.features['dimensions'][j]:
                                    if self.get(self.features['variables'][(i,j)][k]) not in [0, None]:
                                        if "[" in str(k): index = k
                                        else: index = f"[{k}]"
                                        print(f"{j}{index} = {self.get(self.features['variables'][(i,j)][k])}".replace("(", "").replace(")", ""))
                            else:
                                try:
                                    for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                                        if self.get(self.features['variables'][(i,j)][k]) not in [0, None]:
                                            print(f"{j}[{k}] = {self.get(self.features['variables'][(i,j)][k])}".replace("(", "").replace(")", ""))
                                except:
                                    for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                                        if self.get(self.features['variables'][(i,j)])[k] not in [0, None]:
                                            print(f"{j}[{k}] = {self.get(self.features['variables'][(i,j)])[k]}".replace("(", "").replace(")", ""))

                else:

                    if self.features['dimensions'][j] == 0:
                            iv = self.get_interval(self.features['variables'][(i,j)])
                            if iv is not None:
                                print(f"{j} = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")

                    elif len(self.features['dimensions'][j])==1:                    
                        for k in fix_dims(self.features['dimensions'][j])[0]:
                            iv = self.get_interval(self.features['variables'][(i,j)][k])
                            if iv is not None:
                                print(f"{j}[{k}] = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")

                    else:                    
                        for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                            iv = self.get_interval(self.features['variables'][(i,j)][k])
                            if iv is not None:
                                print(f"{j}[{k}] = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")
                    
        else:
            
            if show_detailed_tensors: np.set_printoptions(threshold=np.inf)
            
            for i,j in self.features['variables'].keys():
                
                if i!='evar':
                    
                    numpy_var = self.get_numpy_var(j) 

                    if type(numpy_var)==np.ndarray:

                        numpy_str = np.array2string(numpy_var, separator=', ', prefix='  ', style=str)
                        rows = numpy_str.split('\n')
                        first_row_len = len(rows[0])
                        for idx, row in enumerate(rows):
                            if idx == 0:
                                print(f"{j} = {row}")
                            else:
                                print(" "*(len(f"{j} =")-1)+row)
                    else:
                        print(f"{j} = {numpy_var}")
                        
                else:

                    if self.features['dimensions'][j] == 0:
                            iv = self.get_interval(self.features['variables'][(i,j)])
                            if iv is not None:
                                print(f"{j} = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")

                    elif len(self.features['dimensions'][j])==1:                    
                        for k in fix_dims(self.features['dimensions'][j])[0]:
                            iv = self.get_interval(self.features['variables'][(i,j)][k])
                            if iv is not None:
                                print(f"{j}[{k}] = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")

                    else:                    
                        for k in it.product(*tuple(fix_dims(self.features['dimensions'][j]))):
                            iv = self.get_interval(self.features['variables'][(i,j)][k])
                            if iv is not None:
                                print(f"{j}[{k}] = start: {iv['start']}, end: {iv['end']}, size: {iv['size']}")
                            
    # Methods to work with input and output data.

    def set_local_parameters(self, dataset):

        if type(dataset)==dict:
            for key, value in dataset.items():
                locals()[key] = value
        else:
            for key, value in dataset.data.items():
                locals()[key] = value      

    def set_global_parameters(self, dataset):

        if type(dataset)==dict:
            for key, value in dataset.items():
                globals()[key] = value
        else:
            for key, value in dataset.data.items():
                globals()[key] = value
    
    def decode(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0](*args[1:], **kwargs)
        decoder_fn = getattr(self, '_search_decoder', None) or kwargs.pop('decoder', None)
        if decoder_fn is None:
            return None
        return decoder_fn(*args, **kwargs)

    def rget(self, param, *index_sets, mode='auto'):
        from collections import OrderedDict
        ndim = len(index_sets)
        if isinstance(mode, str):
            mode = [mode] * ndim
        else:
            assert len(mode) == ndim, "Mode list must match number of dimensions"
        remapped_indices = []
        for dim_indices, m in zip(index_sets, mode):
            dim_indices = np.asarray(dim_indices)
            if m == 'auto':
                is_normal = (
                    np.issubdtype(dim_indices.dtype, np.integer) and
                    np.array_equal(np.sort(np.unique(dim_indices)), np.arange(len(dim_indices)))
                )
                m = 'raw' if is_normal else 'unique'

            if m == 'unique':
                unique_vals = list(OrderedDict.fromkeys(dim_indices))
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                remapped = np.array([val_to_idx[v] for v in dim_indices])
            elif m == 'strict':
                remapped = np.arange(len(dim_indices))
            elif m == 'raw':
                remapped = dim_indices
            else:
                raise ValueError(f"Invalid mode: {m}")
            remapped_indices.append(remapped)
        if isinstance(param, np.ndarray):
            return param[tuple(remapped_indices)]
        elif isinstance(param, dict):
            keys = list(zip(*remapped_indices))
            return [param[k] for k in keys]
        else:
            raise TypeError("param must be either numpy.ndarray or dict")

    def term(self, name, value):
        _order = self.features.setdefault('_term_order', [])
        if name not in _order:
            _order.append(name)
        if callable(value):
            self.features['_deferred_terms'][name] = value
            return value
        if _is_solver_expression(value):
            _expr = value
            _em = self
            self.features['_deferred_terms'][name] = (
                lambda _m, _expr=_expr, _em=_em:
                    _evaluate_term_expression(_expr, _em))
            return value
        self.features['_terms'][name] = np.asarray(value).item() if (isinstance(value, np.ndarray) and value.size == 1) else np.asarray(value)
        return value

class Implement:

    def __init__(self, ModelFunction, init_solutions=None):
        '''
        Creates and returns an implementor for the representor model.
        '''

        self.model_data = ModelFunction(['idle'])
        self.ModelFunction = ModelFunction
        self.features = self.model_data.features
        self.interface_name = self.model_data.features['interface_name']
        self.solution_method = self.model_data.features['solution_method']
        self.model_name = self.model_data.features['model_name']
        self.solver_name = self.model_data.features['solver_name']
        self.model_constraints = self.model_data.features['constraints']
        self.model_objectives = self.model_data.features['objectives']
        self.objectives_directions = self.model_data.features['directions']
        self.pos_var_counter = self.model_data.features['positive_variable_counter']
        self.bin_var_counter = self.model_data.features['binary_variable_counter']
        self.int_var_counter = self.model_data.features['integer_variable_counter']
        self.free_var_counter = self.model_data.features['free_variable_counter']
        self.tot_counter = self.model_data.features['total_variable_counter']
        self.con_counter = self.model_data.features['constraint_counter']
        self.obj_counter = self.model_data.features['objective_counter']
        self.AlgOptions = self.model_data.features['solver_options']
        self.VariablesSpread = self.model_data.features['variable_spread']
        self.VariablesBound = self.model_data.features['variable_bound']        
        self.VariablesType = self.model_data.features['variable_type']
        self.ObjectiveBeingOptimized = self.model_data.features['objective_being_optimized']
        self.VariablesDim = self.model_data.features['variable_dim']
        if self.ObjectiveBeingOptimized == 'all':
            self.ObjectiveBeingOptimized = 0
        self.decoder = getattr(getattr(self, "model_data", None), "decoder", None)
        self.status = 'Not solved'
        self.response = None
        self.start = 0
        self.end = 0
        self.BestAgent = None
        self.BestReward = None
        self.AgentProperties = [None, None, None, None]
        if init_solutions is not None:
            self.init_solutions = _normalize_init_solutions(
                init_solutions, self.VariablesBound, self.VariablesSpread)
        else:
            self.init_solutions = None
        self.get_objective = self.get_obj
        self.get_var = self.get_variable = self.get
        self.search = self.solve = self.optimize = self.run = self.sol
        self.get_tensor = self.get_numpy_var
        self.PI = self.pi = np.pi
        
        match self.interface_name:

            case 'mealpy':

                from .generators.model import mealpy_model_generator
                self.ModelObject = mealpy_model_generator.generate_model(
                    self.solver_name, self.AlgOptions)

            case 'niapy':

                from .generators.model import niapy_model_generator
                self.ModelObject = niapy_model_generator.generate_model(
                    self.solver_name, self.AlgOptions)

            case 'scipy':

                self.ModelObject = None

            case 'pygad':

                self.ModelObject = None

            case 'pymultiobjective':

                self.ModelObject = None

            case 'pymoo':

                self.ModelObject = None

            case 'indago':

                self.ModelObject = None

            case 'feloopy':
                
                self.LB = np.concatenate([
                    np.full(self.VariablesSpread[key][1] - self.VariablesSpread[key][0], self.VariablesBound[key][0])
                    for key in self.VariablesBound.keys()
                ])
                self.UB = np.concatenate([
                    np.full(self.VariablesSpread[key][1] - self.VariablesSpread[key][0], self.VariablesBound[key][1])
                    for key in self.VariablesBound.keys()
                ])
                from .generators.model import feloopy_model_generator
                self.ModelObject = feloopy_model_generator.generate_model(
                    self.tot_counter[1], self.objectives_directions, self.solver_name, self.AlgOptions, self.LB, self.UB)

    def remove_infeasible_solutions(self):

        self.BestAgent = np.delete(self.BestAgent, self.remove, axis=0)
        self.BestReward = np.delete(self.BestReward, self.remove, axis=0)

    def sol(self, penalty_coefficient=0, number_of_times=1, show_plots=False, save_plots=False, show_log=False):

        self.penalty_coefficient = penalty_coefficient

        match self.interface_name:

            case 'mealpy':

                from .generators.solution import mealpy_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end = mealpy_solution_generator.generate_solution(
                    self.ModelObject, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots,show_log, self.AlgOptions, init_solutions=self.init_solutions)

            case 'scipy':

                from .generators.solution import scipy_solution_generator
                self.BestAgent, self.BestReward,self.start, self.end = scipy_solution_generator.generate_solution(self.solver_name, self.AlgOptions, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots, show_log, init_solutions=self.init_solutions)

            case 'niapy':

                from .generators.solution import niapy_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end = niapy_solution_generator.generate_solution(
                    self.ModelObject, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots,show_log, self.AlgOptions, init_solutions=self.init_solutions)

            case 'pygad':

                from .generators.solution import pygad_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end = pygad_solution_generator.generate_solution(
                    self.solver_name, self.ModelObject, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots,show_log, self.AlgOptions, init_solutions=self.init_solutions)

            case 'pymultiobjective':

                from .generators.solution import pymultiobjective_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end = pymultiobjective_solution_generator.generate_solution(
                    self.solver_name, self.AlgOptions, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots,show_log, init_solutions=self.init_solutions)
                self.remove = []

                for i in range(np.shape(self.BestReward)[0]):

                    if 'infeasible' in self.Check_Fitness(self.BestAgent[i]):

                        self.remove.append(i)

                # see the pymoo branch: never delete the last row left
                if len(self.remove) != 0 and len(self.remove) < np.shape(self.BestAgent)[0]:
                    self.remove_infeasible_solutions()

            case 'pymoo':

                from .generators.solution import pymoo_solution_generator
                self.BestAgent, self.BestReward,self.start, self.end = pymoo_solution_generator.generate_solution(self.solver_name, self.AlgOptions, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, save_plots, show_log, init_solutions=self.init_solutions)
                self.remove = []

                for i in range(np.shape(self.BestReward)[0]):

                    if 'infeasible' in self.Check_Fitness(np.array([self.BestAgent[i]])):

                        self.remove.append(i)

                if len(self.remove) != 0 and len(self.remove) < np.shape(self.BestAgent)[0]:
                    self.remove_infeasible_solutions()

            case 'indago':

                from .generators.solution import indago_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end = indago_solution_generator.generate_solution(
                    self.solver_name, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, show_log, self.AlgOptions, init_solutions=self.init_solutions)
            
            case 'feloopy':

                from .generators.solution import feloopy_solution_generator
                self.BestAgent, self.BestReward, self.start, self.end, self.status = feloopy_solution_generator.generate_solution(
                    self.ModelObject, self.Fitness, self.tot_counter, self.objectives_directions, self.ObjectiveBeingOptimized, number_of_times, show_plots, show_log, init_solutions=self.init_solutions)

    def dis_plots(self, ideal_pareto: Optional[np.ndarray] = [], step: Optional[tuple] = (0.1,)):

        """
        Calculates selected Pareto front metrics and displays the results in a tabulated format.

        :param ideal_pareto: An array of shape (n_samples, n_objectives) containing the ideal Pareto front. Default is None.
        """

        obtained_pareto = self.BestReward

        try:
            from pyMultiobjective.util import graphs
        except:
            ""
        ObjectivesDirections = [-1 if direction =='max' else 1 for direction in self.objectives_directions]
        def f1(X): return ObjectivesDirections[0]*self.Fitness(np.array(X))[0]
        def f2(X): return ObjectivesDirections[1]*self.Fitness(np.array(X))[1]
        def f3(X): return ObjectivesDirections[2]*self.Fitness(np.array(X))[2]
        def f4(X): return ObjectivesDirections[3]*self.Fitness(np.array(X))[3]
        def f5(X): return ObjectivesDirections[4]*self.Fitness(np.array(X))[4]
        def f6(X): return ObjectivesDirections[5]*self.Fitness(np.array(X))[5]
        my_list_of_functions = [f1, f2, f3, f4, f5, f6]
        parameters = dict()
        list_of_functions = []
        for i in range(len(ObjectivesDirections)): list_of_functions.append(my_list_of_functions[i])
        
        solution = np.concatenate((self.BestAgent, self.BestReward*ObjectivesDirections), axis=1)

   
        parameters = {
        'min_values': (0,)*self.tot_counter[1],
        'max_values': (1,)*self.tot_counter[1],
        'step': step*self.tot_counter[1],
        'solution': solution, 
        'show_pf': True,
        'show_pts': True,
        'show_sol': True,
        'pf_min': True, 
        'custom_pf': ideal_pareto*ObjectivesDirections if type(ideal_pareto) == np.ndarray else [],
        'view': 'browser'
        }
        graphs.plot_mooa_function(list_of_functions = list_of_functions, **parameters)

        parameters = {
            'min_values': (0,)*self.tot_counter[1],
            'max_values': (1,)*self.tot_counter[1],
            'step': step*self.tot_counter[1],
            'solution': solution, 
            'show_pf': True,
            'pf_min': True,  
            'custom_pf': ideal_pareto*ObjectivesDirections if type(ideal_pareto) == np.ndarray else [],
            'view': 'browser'
        }
        graphs.parallel_plot(list_of_functions = list_of_functions, **parameters)

    def dis_status(self):
        print('status:', self.get_status())

    def get_pop(self):
        """
        Returns the final population from a heuristic. Each row is one solution.

        For the 'feloopy' interface, returns the full population matrix.
        For other interfaces, returns the best agent reshaped as a single-row matrix.
        """
        if hasattr(self, 'ModelObject') and hasattr(self.ModelObject, 'pi'):
            if not isinstance(self.ModelObject.pi, (int, float)):
                pi = np.array(self.ModelObject.pi)
                f = self.ModelObject.f
                return pi[:, :f].copy()
        return np.atleast_2d(np.array(self.BestAgent)).copy()

    def get_status(self):

        if len(self.objectives_directions)==1:

            if self.interface_name == 'feloopy':

                try:
                    code = self.status[0]
                    code = int(float(code))
                    if code == 1:
                        return 'feasible (constrained)'
                    elif code == 2:
                        return 'feasible (unconstrained)'
                    elif code == -1:
                        return self.Check_Fitness(np.array([self.BestAgent]))
                except Exception:
                    pass

            return self.Check_Fitness(self.BestAgent)

        else:

            status = []

            for i in range(np.shape(self.BestReward)[0]):
                if self.interface_name in ['feloopy', 'pymoo']:
                    status.append(self.Check_Fitness(np.array([self.BestAgent[i]])))
                else:
                    status.append(self.Check_Fitness(self.BestAgent[i]))

            return status

    def Check_Fitness(self, X):

        self.AgentProperties[0] = 'feasibility_check'
        self.AgentProperties[1] = X
        self.AgentProperties[2] = self.VariablesSpread
        self.AgentProperties[3] = self.penalty_coefficient

        return self.ModelFunction(self.AgentProperties)

    def Fitness(self, X):

        self.AgentProperties[0] = 'active'
        self.AgentProperties[1] = X
        self.AgentProperties[2] = self.VariablesSpread
        self.AgentProperties[3] = self.penalty_coefficient

        return self.ModelFunction(self.AgentProperties)

    def evaluate(self, show_fig=True, save_fig=False, file_name=None, dpi=800, fig_size=(18, 4), opt=None, opt_features=None, pareto=None, abs_tol=0.001, rel_tol=0.001):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=fig_size)

        m = self.ModelObject.epsiode

        no_epochs = self.AlgOptions['epoch']
        no_episodes = self.AlgOptions['episode']

        max_epoch_time = []
        for epoch in range(0, no_epochs):
            episode_time = []
            for episode in range(0, no_episodes):
                episode_time.append(m[episode]['epoch_time'][epoch])
            max_epoch_time.append(np.max(episode_time))
        max_epoch_time = np.array(max_epoch_time)

        min_epoch_time = []
        for epoch in range(0, no_epochs):
            episode_time = []
            for episode in range(0, no_episodes):
                episode_time.append(m[episode]['epoch_time'][epoch])
            min_epoch_time.append(np.min(episode_time))
        min_epoch_time = np.array(min_epoch_time)

        ave_epoch_time = []
        for epoch in range(0, no_epochs):
            episode_time = []
            for episode in range(0, no_episodes):
                episode_time.append(m[episode]['epoch_time'][epoch])
            ave_epoch_time.append(np.average(episode_time))
        ave_epoch_time = np.array(ave_epoch_time)

        std_epoch_time = []
        for epoch in range(0, no_epochs):
            episode_time = []
            for episode in range(0, no_episodes):
                episode_time.append(m[episode]['epoch_time'][epoch])
            std_epoch_time.append(np.std(episode_time))
        std_epoch_time = np.array(std_epoch_time)

        axs = fig.add_subplot(1, 5, 5)
        x = np.arange(no_epochs)
        axs.plot(x, max_epoch_time, 'blue', alpha=0.4)
        axs.plot(x, ave_epoch_time, 'blue', alpha=0.8)
        axs.plot(x, min_epoch_time, 'blue', alpha=0.4)
        axs.fill_between(x, ave_epoch_time - std_epoch_time,
                         ave_epoch_time + std_epoch_time, color='blue', alpha=0.3)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Time (second)')
        axs.set_xlim(-0.5, no_epochs-1+0.5)

        max_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
            max_epoch_obj.append(np.max(max_episode_obj))
        max_epoch_obj = np.array(max_epoch_obj)

        min_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
            min_epoch_obj.append(np.min(max_episode_obj))
        min_epoch_obj = np.array(min_epoch_obj)

        ave_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
            ave_epoch_obj.append(np.average(max_episode_obj))
        ave_epoch_obj = np.array(ave_epoch_obj)

        std_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
            std_epoch_obj.append(np.std(max_episode_obj))
        std_epoch_obj = np.array(std_epoch_obj)

        axs = fig.add_subplot(1, 5, 4)
        x = np.arange(no_epochs)
        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'max':
            axs.plot(x, max_epoch_obj, 'green', alpha=0.4)
            axs.plot(x, ave_epoch_obj, 'green', alpha=0.8)
            axs.plot(x, min_epoch_obj, 'green', alpha=0.4)
            axs.fill_between(x, ave_epoch_obj - std_epoch_obj,
                             ave_epoch_obj + std_epoch_obj, color='green', alpha=0.3)
        else:
            axs.plot(x, max_epoch_obj, 'red', alpha=0.4)
            axs.plot(x, ave_epoch_obj, 'red', alpha=0.8)
            axs.plot(x, min_epoch_obj, 'red', alpha=0.4)

            axs.fill_between(x, ave_epoch_obj - std_epoch_obj,
                             ave_epoch_obj + std_epoch_obj, color='red', alpha=0.3)
        axs.set_xlabel('Epoch')
        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'max':
            axs.set_ylabel('Maximum reward')
        else:
            axs.set_ylabel('Maximum loss')
        axs.set_xlim(-0.5, no_epochs-1+0.5)

        max_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(np.average(
                    m[episode]['epoch_solutions'][epoch][:, -1]))
            max_epoch_obj.append(np.max(max_episode_obj))
        max_epoch_obj = np.array(max_epoch_obj)

        min_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(np.average(
                    m[episode]['epoch_solutions'][epoch][:, -1]))
            min_epoch_obj.append(np.min(max_episode_obj))
        min_epoch_obj = np.array(min_epoch_obj)

        ave_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(np.average(
                    m[episode]['epoch_solutions'][epoch][:, -1]))
            ave_epoch_obj.append(np.average(max_episode_obj))
        ave_epoch_obj = np.array(ave_epoch_obj)

        std_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(np.average(
                    m[episode]['epoch_solutions'][epoch][:, -1]))
            std_epoch_obj.append(np.std(max_episode_obj))
        std_epoch_obj = np.array(std_epoch_obj)

        axs = fig.add_subplot(1, 5, 3)
        x = np.arange(no_epochs)
        axs.plot(x, max_epoch_obj, 'orange', alpha=0.4)
        axs.plot(x, ave_epoch_obj, 'orange', alpha=0.8)
        axs.plot(x, min_epoch_obj, 'orange', alpha=0.4)
        axs.fill_between(x, ave_epoch_obj - std_epoch_obj,
                         ave_epoch_obj + std_epoch_obj, color='orange', alpha=0.3)
        axs.set_xlabel('Epoch')
        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'max':
            axs.set_ylabel('Average reward')
        else:
            axs.set_ylabel('Average loss')
        axs.set_xlim(-0.5, no_epochs-1+0.5)

        max_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
            max_epoch_obj.append(np.max(max_episode_obj))
        max_epoch_obj = np.array(max_epoch_obj)

        min_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
            min_epoch_obj.append(np.min(max_episode_obj))
        min_epoch_obj = np.array(min_epoch_obj)

        ave_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
            ave_epoch_obj.append(np.average(max_episode_obj))
        ave_epoch_obj = np.array(ave_epoch_obj)

        std_epoch_obj = []
        for epoch in range(0, no_epochs):
            max_episode_obj = []
            for episode in range(0, no_episodes):
                max_episode_obj.append(
                    np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
            std_epoch_obj.append(np.std(max_episode_obj))
        std_epoch_obj = np.array(std_epoch_obj)

        axs = fig.add_subplot(1, 5, 2)
        x = np.arange(no_epochs)
        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'max':
            axs.plot(x, max_epoch_obj, 'red', alpha=0.4)
            axs.plot(x, ave_epoch_obj, 'red', alpha=0.8)
            axs.plot(x, min_epoch_obj, 'red', alpha=0.4)
            axs.fill_between(x, ave_epoch_obj - std_epoch_obj,
                             ave_epoch_obj + std_epoch_obj, color='red', alpha=0.3)
        else:
            axs.plot(x, max_epoch_obj, 'green', alpha=0.4)
            axs.plot(x, ave_epoch_obj, 'green', alpha=0.8)
            axs.plot(x, min_epoch_obj, 'green', alpha=0.4)
            axs.fill_between(x, ave_epoch_obj - std_epoch_obj,
                             ave_epoch_obj + std_epoch_obj, color='green', alpha=0.3)
        axs.set_xlabel('Epoch')
        axs.set_xlim(-0.5, no_epochs-1+0.5)
        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'max':
            axs.set_ylabel('Minimum reward')
        else:
            axs.set_ylabel('Minimum loss')

        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'min':
            best_min_min = np.inf
            best_min_min_t = []
            best_sol_t = []
            best_per_episode = []
            no_features = self.tot_counter[1]
            for epoch in range(0, no_epochs):
                best_min = []
                best_sol = []
                for episode in range(0, no_episodes):
                    best_min.append(
                        np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
                    best_sol.append(m[episode]['epoch_solutions'][epoch][np.argmin(
                        m[episode]['epoch_solutions'][epoch][:, -1]), :])
                    best_track = np.min(best_min)
                    for x in best_sol:
                        if x[-1] == best_track:
                            best_sol_found = x[:no_features]
                if best_track <= best_min_min:
                    best_min_min = best_track
                    best_min_min_t.append(best_track)
                    if no_features == 1:
                        best_sol_t.append(best_sol_found[0])
                    if no_features == 2:
                        best_sol_t.append(
                            [best_sol_found[0], best_sol_found[1]])
                    else:
                        best_sol_t.append(
                            [best_sol_found[0], best_sol_found[1], best_sol_found[2]])
                else:
                    best_min_min_t.append(best_min_min)
                    best_sol_t.append(best_sol_t[-1])

                    if epoch == no_epochs-1:
                        best_per_episode.append(best_track)

            best_min_min_t = np.array(best_min_min_t)
        else:
            best_min_min = -np.inf
            best_min_min_t = []
            best_sol_t = []
            best_per_episode = []
            no_features = self.tot_counter[1]
            for epoch in range(0, no_epochs):
                best_min = []
                best_sol = []
                for episode in range(0, no_episodes):
                    best_min.append(
                        np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
                    best_sol.append(m[episode]['epoch_solutions'][epoch][np.argmax(
                        m[episode]['epoch_solutions'][epoch][:, -1]), :])
                    best_track = np.max(best_min)
                    for x in best_sol:
                        if x[-1] == best_track:
                            best_sol_found = x[:no_features]
                if best_track >= best_min_min:
                    best_min_min = best_track
                    best_min_min_t.append(best_track)
                    if no_features == 1:
                        best_sol_t.append(best_sol_found[0])
                    if no_features == 2:
                        best_sol_t.append(
                            [best_sol_found[0], best_sol_found[1]])
                    else:
                        best_sol_t.append(
                            [best_sol_found[0], best_sol_found[1], best_sol_found[2]])
                else:
                    best_min_min_t.append(best_min_min)
                    best_sol_t.append(best_sol_t[-1])

                    if epoch == no_epochs-1:
                        best_per_episode.append(best_track)

            best_min_min_t = np.array(best_min_min_t)

        if no_features == 1:
            axs = fig.add_subplot(1, 5, 1)
            axs.plot(np.arange(no_epochs), best_sol_t, c='black', lw=1)
            if opt_features != None:
                axs.scatter(np.arange(no_epochs),
                            opt_features[0], c='black', marker='*', lw=1)

            axs.set_ylim(-0.5, 1.5)
            axs.set_xlim(-0.5, no_epochs-1+0.5)
            axs.set_xlabel('Epoch')
            axs.set_ylabel('Feature')

        if no_features == 2:
            axs = fig.add_subplot(1, 5, 1)
            from matplotlib.patches import Rectangle
            for i in range(0, no_epochs):
                hg = 0.1+i/(no_epochs)
                axs.scatter(best_sol_t[i][0], best_sol_t[i]
                            [1], c='black', lw=1, alpha=hg)
            if opt_features != None:
                axs.scatter(opt_features[0], opt_features[1],
                            c='black', marker='*', lw=1)

            axs.add_patch(Rectangle((0, 0), 1, 1, fill=None, alpha=1))

            axs.set_ylim(-0.5, 1.5)
            axs.set_xlim(-0.5, 1.5)
            axs.set_xlabel('Feature 1')
            axs.set_ylabel('Feature 2')

        if no_features == 3:
            axs = fig.add_subplot(1, 5, 1, projection='3d')
            for i in range(0, no_epochs):
                hg = 0.1+i/(no_epochs)
                axs.scatter(best_sol_t[i][0], best_sol_t[i][1],
                            best_sol_t[i][2], lw=1, alpha=hg, color='black')
            if opt_features != None:
                axs.scatter(opt_features[0], opt_features[1],
                            opt_features[2], c='red', marker='*', lw=1)
            axs.set_xlabel('Feature 1')
            axs.set_ylabel('Feature 2')
            axs.set_zlabel('Feature 3')
            axs.set_ylim(-0.5, 1.5)
            axs.set_xlim(-0.5, 1.5)
            axs.set_zlim(-0.5, 1.5)

            axs.view_init(azim=30)

        if no_features <= 2:
            plt.subplots_adjust(left=0.071, bottom=0.217,
                                right=0.943, top=0.886, wspace=0.35, hspace=0.207)
        else:
            plt.subplots_adjust(left=0.03, bottom=0.252,
                                right=0.945, top=0.886, wspace=0.421, hspace=0.22)

        if save_fig:
            if file_name == None:
                plt.savefig('evaluation_results.png', dpi=dpi)
            else:
                plt.savefig(file_name, dpi=dpi)

        if show_fig:
            plt.show()

        obj = []
        time = []
        for episode in range(0, no_episodes):
            obj.append(m[episode]['best_single'][0][-1])
            time.append(m[episode]['episode_time'][0])

        opt = np.array([opt])
        if opt != 0:
            accuracy = (1-np.abs(opt-best_min_min_t)/opt)*100
        else:
            opt = opt + 1
            best_min_min_t = best_min_min_t+1
            accuracy = (1-np.abs(opt-best_min_min_t)/opt)*100
            accuracy[np.where(accuracy < 0)] = 0

        from math import isclose

        opt = np.array([opt])
        prob_per_epoch = []

        findbest = np.zeros(shape=(no_episodes, no_epochs))

        if self.objectives_directions[self.ObjectiveBeingOptimized] == 'min':
            for episode in range(0, no_episodes):
                episode_tracker = []
                best = np.inf
                for epoch in range(0, no_epochs):
                    if np.min(m[episode]['epoch_solutions'][epoch][:, -1]) <= best:
                        best = np.min(
                            m[episode]['epoch_solutions'][epoch][:, -1])
                        episode_tracker.append(
                            np.min(m[episode]['epoch_solutions'][epoch][:, -1]))
                    else:
                        episode_tracker.append(best)
                for epoch in range(0, no_epochs):
                    if opt == 0:
                        if isclose(episode_tracker[epoch], opt, abs_tol=abs_tol):
                            findbest[episode, epoch] = 1
                    else:
                        if isclose(episode_tracker[epoch], opt, rel_tol=rel_tol):
                            findbest[episode, epoch] = 1
        else:
            for episode in range(0, no_episodes):
                episode_tracker = []
                best = -np.inf
                for epoch in range(0, no_epochs):
                    if np.max(m[episode]['epoch_solutions'][epoch][:, -1]) >= best:
                        best = np.max(
                            m[episode]['epoch_solutions'][epoch][:, -1])
                        episode_tracker.append(
                            np.max(m[episode]['epoch_solutions'][epoch][:, -1]))
                    else:
                        episode_tracker.append(best)
                for epoch in range(0, no_epochs):
                    if opt == 0:
                        if isclose(episode_tracker[epoch], opt, abs_tol=abs_tol, rel_tol=rel_tol):
                            findbest[episode, epoch] = 1
                    else:
                        if isclose(episode_tracker[epoch], opt, abs_tol=abs_tol, rel_tol=rel_tol):
                            findbest[episode, epoch] = 1

        # abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        prob_per_epoch = [sum(findbest[episode, epoch] for episode in range(
            0, no_episodes))/no_episodes for epoch in range(0, no_epochs)]

        return [obj, time, accuracy, prob_per_epoch]

    def get(self, *args):
        if self.obj_counter[0] == 1:
            match self.interface_name:
                case _ if self.interface_name in ['mealpy', 'niapy', 'pygad', 'scipy', 'indago', 'pymoo']:
                    for i in args:
                        _agent = self.BestAgent
                        if _agent.ndim == 2:
                            _agent = _agent[0]
                        if len(i) >= 2:
                            match self.VariablesType[i[0]]:
                                case 'pvar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]
                                        return var(*i[1])
                                case 'fvar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]
                                        return var(*i[1])
                                case 'bvar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return np.int64(np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))[0]
                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return np.int64(self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)])
                                        return var(*i[1])
                                case 'ivar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]
                                        return var(*i[1])
                                case 'svar':
                                    return np.int64(np.argsort(_agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]])[i[1]].astype(np.int64))

                        else:
                            match self.VariablesType[i[0]]:
                                case 'pvar':
                                    return (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                case 'fvar':
                                    return (self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                case 'bvar':
                                    return np.int64(np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))[0]
                                case 'ivar':
                                    return np.int64(np.round(self.VariablesBound[i[0]][0] + _agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))[0]
                                case 'svar':
                                    return np.int64(np.argsort(_agent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]]).astype(np.int64))
                case 'feloopy':
                    for i in args:
                        if len(i) >= 2:
                            match self.VariablesType[i[0]]:
                                case 'pvar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]
                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]
                                        return var(*i[1])
                                case 'fvar':

                                    if self.VariablesDim[i[0]] == 0:
                                        return (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]

                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]

                                        return var(*i[1])

                                case 'bvar':
                                
                                    if self.VariablesDim[i[0]] == 0:
                                        return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))[0]

                                    else:

                                        def var(*args):
                                            self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return np.int64(self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)])

                                        return var(*i[1])
                                case 'ivar':
                                    if self.VariablesDim[i[0]] == 0:
                                        return np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))[0]

                                    else:
                                        def var(*args):
                                            self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                                self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                            return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]
                                        return var(*i[1])

                                case 'svar':
                                    return np.argsort(self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]])[i[1]].astype(np.int64)

                        else:
                            match self.VariablesType[i[0]]:

                                case 'pvar':
                                    return (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                case 'fvar':
                                    return (self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                case 'bvar':
                                    return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                                case 'ivar':
                                    return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                                case 'svar':
                                    return np.int64(np.argsort(self.BestAgent[self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]]).astype(np.int64))
        else:

            for i in args:
                if len(i) >= 2:

                    match self.VariablesType[i[0]]:

                        case 'pvar':

                            if self.VariablesDim[i[0]] == 0:
                                return (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))

                            else:
                                def var(*args):
                                    self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                    return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]

                                return var(*i[1])

                        case 'fvar':
                            if self.VariablesDim[i[0]] == 0:
                                return (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))

                            else:
                                def var(*args):
                                    self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                    return self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)]

                                return var(*i[1])

                        case 'bvar':
                            if self.VariablesDim[i[0]] == 0:
                                return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))

                            else:
                                def var(*args):
                                    self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                    return np.int64(self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)])

                                return var(*i[1])
                        case 'ivar':
                            if self.VariablesDim[i[0]] == 0:
                                return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                            else:
                                def var(*args):
                                    self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                        self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                    return np.int64(self.NewAgentProperties[_slot_index(self.VariablesDim[i[0]], args)])
                                return var(*i[1])

                        case 'svar':

                            return np.int64(np.argsort(self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]])[i[1]].astype(np.int64))

                else:

                    match self.VariablesType[i[0]]:
                        case 'pvar':
                            return (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                        case 'fvar':
                            return (self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                        case 'bvar':
                            return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                        case 'ivar':
                            return np.int64(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                        case 'svar':
                            return np.int64(np.argsort(self.BestAgent[:, self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]]).astype(np.int64))

    def dis_indicators(self, ideal_pareto: Optional[np.ndarray] = [], ideal_point: Optional[np.array] = [], step: Optional[tuple] = (0.1,), epsilon: float = 0.01, p: float = 2.0, n_clusters: int = 5, save_path: Optional[str] = None, show_log: Optional[bool] = False):

        """
        Calculates selected Pareto front metrics and displays the results in a tabulated format.

        :param ideal_pareto: An array of shape (n_samples, n_objectives) containing the ideal Pareto front. Default is None.
        :param epsilon: A float value for the epsilon value used in the epsilon metric. Default is 0.01.
        :param p: A float value for the power parameter used in the weighted generational distance and weighted inverted generational distance metrics. Default is 2.0.
        :param n_clusters: An integer value for the number of clusters used in the knee point distance metric. Default is 5.
        :param save_path: A string value for the path where the results should be saved. Default is None.
        """

        self.get_indicators(ideal_pareto, ideal_point, step, epsilon, p, n_clusters, save_path, show_log = True)

    def get_indicators(self, ideal_pareto: Optional[np.ndarray] = [], ideal_point: Optional[np.array] = [], step: Optional[tuple] = (0.2,), epsilon: float = 0.01, p: float = 2.0, n_clusters: int = 5, save_path: Optional[str] = None, show_log: Optional[bool] = False, normalize_hv: Optional[bool] = False, bypass_limit=False):

        """
        Calculates selected Pareto front metrics and displays the results in a tabulated format.

        :param ideal_pareto: An array of shape (n_samples, n_objectives) containing the ideal Pareto front. Default is None.
        :param epsilon: A float value for the epsilon value used in the epsilon metric. Default is 0.01.
        :param p: A float value for the power parameter used in the weighted generational distance and weighted inverted generational distance metrics. Default is 2.0.
        :param n_clusters: An integer value for the number of clusters used in the knee point distance metric. Default is 5.
        :param save_path: A string value for the path where the results should be saved. Default is None.
        """
        if len(self.get_obj())!=0:

            obtained_pareto = self.BestReward
            try:
                from pyMultiobjective.util import indicators
            except:
                ""

            ObjectivesDirections = [-1 if direction =='max' else 1 for direction in self.objectives_directions]

            def f1(X): return ObjectivesDirections[0]*self.Fitness(np.array(X))[0]
            def f2(X): return ObjectivesDirections[1]*self.Fitness(np.array(X))[1]
            def f3(X): return ObjectivesDirections[2]*self.Fitness(np.array(X))[2]
            def f4(X): return ObjectivesDirections[3]*self.Fitness(np.array(X))[3]
            def f5(X): return ObjectivesDirections[4]*self.Fitness(np.array(X))[4]
            def f6(X): return ObjectivesDirections[5]*self.Fitness(np.array(X))[5]

            list_of_functions = [f1, f2, f3, f4, f5, f6]

            solution = np.concatenate((self.BestAgent, self.BestReward*ObjectivesDirections), axis=1)
            self.calculated_indicators = dict()

            #Does not require the ideal_pareto
            parameters = {
                'solution': solution,
                'n_objs': len(ObjectivesDirections),
                'ref_point': ideal_point,
                'normalize': normalize_hv
            }
            hypervolume = indicators.hv_indicator(**parameters)
            self.calculated_indicators['hv'] = hypervolume

            parameters = {
                'min_values': (0,)*self.tot_counter[1],
                'max_values': (1,)*self.tot_counter[1],
                'step': step*self.tot_counter[1],
                'solution': solution,
                'pf_min': True,
                'custom_pf': ideal_pareto*ObjectivesDirections if type(ideal_pareto) == np.ndarray else []
            }

            sp = indicators.sp_indicator(list_of_functions=list_of_functions, **parameters)
            self.calculated_indicators['sp'] = sp

            #Computationally efficient only if ideal_pareto exists
            if self.tot_counter[1]<=3 or ideal_pareto != [] or bypass_limit:
                gd = indicators.gd_indicator(list_of_functions=list_of_functions, **parameters)
                gdp = indicators.gd_plus_indicator(list_of_functions=list_of_functions, **parameters)
                igd = indicators.igd_indicator(list_of_functions=list_of_functions, **parameters)
                igdp = indicators.igd_plus_indicator(list_of_functions=list_of_functions, **parameters)
                ms = indicators.ms_indicator(list_of_functions=list_of_functions, **parameters)
                self.calculated_indicators['gd'] = gd
                self.calculated_indicators['gdp'] = gdp
                self.calculated_indicators['igd'] = igd
                self.calculated_indicators['igdp'] = igdp
                self.calculated_indicators['ms'] = ms

            return self.calculated_indicators

    def dis_time(self):

        hour = round(((self.end-self.start)), 3) % (24 * 3600) // 3600
        min = round(((self.end-self.start)), 3) % (24 * 3600) % 3600 // 60
        sec = round(((self.end-self.start)), 3) % (24 * 3600) % 3600 % 60

        print(f"cpu time [{self.interface_name}]: ", (self.end-self.start)*10 **
              6, '(microseconds)', "%02d:%02d:%02d" % (hour, min, sec), '(h, m, s)')
  
    def get_time(self):
        """

        Used to get solution time in seconds.
        
        """

        if isinstance(self.start, list):
            return sum(e - s for s, e in zip(self.start, self.end))
        return self.end-self.start

    def get_obj(self):
        return self.BestReward

    def dis(self, input):
        if len(input) >= 2:
            print(input[0]+str(input[1])+': ', self.get(input))
        else:
            print(str(input[0])+': ', self.get(input))

    def dis_obj(self):
        print('objective: ', self.BestReward)

    def get_bound(self, *args):

        for i in args:

            if len(i) >= 2:
            
                match self.VariablesType[i[0]]:

                    case 'pvar':

                        if self.VariablesDim[i[0]] == 0:
                            UB = np.max((self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                            LB = np.min((self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                            return [LB,UB]

                        else:
                            def var(*args):
                                self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                    self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                return self.NewAgentProperties[:,_slot_index(self.VariablesDim[i[0]], args)]
                            return [np.min(var(*i[1])),np.max(var(*i[1]))]

                    case 'fvar':

                        if self.VariablesDim[i[0]] == 0:
                            LB = np.min(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                            UB = np.max(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                            return [LB,UB]

                        else:
                            def var(*args):
                                self.NewAgentProperties = (self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                    self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                return self.NewAgentProperties[:,_slot_index(self.VariablesDim[i[0]], args)]
                            return [np.min(var(*i[1])),np.max(var(*i[1]))]

                    case 'bvar':
                        if self.VariablesDim[i[0]] == 0:
                            LB = np.int64(np.min(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                            UB = np.int64(np.max(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                            return [LB,UB]

                        else:
                            def var(*args):
                                self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                    self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                return np.int64(self.NewAgentProperties[:,_slot_index(self.VariablesDim[i[0]], args)])
                            return [np.min(var(*i[1])),np.max(var(*i[1]))]
                        
                    case 'ivar':
                        if self.VariablesDim[i[0]] == 0:
                            LB = np.int64(np.min(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                            UB = np.int64(np.min(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                            return [LB,UB]

                        else:
                            def var(*args):
                                self.NewAgentProperties = np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (
                                    self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                                return np.int64(self.NewAgentProperties[:,_slot_index(self.VariablesDim[i[0]], args)])
                            return [np.min(var(*i[1])),np.max(var(*i[1]))]
                        
                    case 'svar':
                        return np.int64(np.argsort(self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]])[i[1]])

            else:

                match self.VariablesType[i[0]]:

                    case 'pvar':
                        UB = np.max((self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                        LB = np.min((self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0])))
                        return [LB,UB]

                    case 'fvar':
                        UB = np.max(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                        LB = np.min(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))
                        return [LB,UB]

                    case 'bvar':
                        UB = np.int64(np.max(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                        LB = np.int64(np.min(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                        return [LB,UB]

                    case 'ivar':
                        UB = np.int64(np.max(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                        LB= np.int64(np.min(np.round(self.VariablesBound[i[0]][0] + self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]] * (self.VariablesBound[i[0]][1] - self.VariablesBound[i[0]][0]))))
                        return [UB,LB]

                    case 'svar':
                        return np.int64(np.argsort(self.BestAgent[:,self.VariablesSpread[i[0]][0]:self.VariablesSpread[i[0]][1]]))
                    
    def get_payoff(self):

        payoff=[]
        for i in range(len(self.objectives_directions)):
            if self.objectives_directions[i]=='max':
                ind =np.argmax(self.get_obj()[:, i])
                val = self.get_obj()[ind, :]
            elif self.objectives_directions[i] =='min':
                ind = np.argmin(self.get_obj()[:, i])
                val = self.get_obj()[ind, :]
            payoff.append(val)
        return np.array(payoff)

    def clean_report(self,**kwargs):

        clear_console()
        self.report(**kwargs)
        
    def report(self, style=1, skip_system_information=True, show_elements=True, width=78, skip=False, full=False, save=None, copy_to_clipboard=False, show_tensors=None, diagnostics=False):
        if show_tensors is not None:
            show_elements = not show_tensors
        self.em = self.model_data
        self.solver = self.solver_name
        self.method = self.model_data.features.get('solution_method', 'heuristic')
        self.number_of_objectives = self.model_data.features['objective_counter'][0]
        self.solutions = getattr(self, 'solutions', {})
        self.inputdata = getattr(self, 'inputdata', None)
        self.sensitivity_analyzed = getattr(self, 'sensitivity_analyzed', False)
        self.output_decimals = getattr(self, 'output_decimals', 4)
        self.debug = getattr(self, 'debug', False)
        if not hasattr(self, 'key_vars'):
            self.key_vars = []
        ReportEngine(self).report_search(style=style, skip_system_information=skip_system_information, show_elements=show_elements, width=width, skip=skip, full=full, save=save, copy_to_clipboard=copy_to_clipboard, diagnostics=diagnostics)
        return self

    def get_numpy_var(self, var_name):
        output = []
        for i in self.VariablesDim.keys():
            if i == var_name:
                if self.VariablesDim[i] == 0:
                    output = self.get([i, (0,)])
                elif isinstance(self.VariablesDim[i], (set, frozenset)):
                    output = {
                        k: self.get([i, k if isinstance(k, tuple) else (k,)])
                        for k in self.VariablesDim[i]
                    }
                    return output
                elif len(self.VariablesDim[i]) == 1:
                    for k in fix_dims(self.VariablesDim[i])[0]:
                        output.append(self.get([i, (k,)]))
                else:
                    for k in it.product(*tuple(fix_dims(self.VariablesDim[i]))):
                        output.append(self.get([i, (*k,)]))
                    output = np.array(output).reshape([len(element) if not isinstance(element, int) else element for element in fix_dims(self.VariablesDim[i])])
        return output if isinstance(output, dict) else np.array(output)

    def healthy(self):
        try:
            status = self.get_status()
        except Exception:
            return False
        if status is None:
            return False
        if isinstance(status, list):
            if len(status) == 0:
                return False
            status = status[0]
        if not isinstance(status, str):
            try:
                status = str(status)
            except Exception:
                return False
        status = status.lower()
        has_feasible = 'optimal' in status or 'feasible' in status or 'succ' in status
        has_infeasible = 'infeasible' in status or 'unsucc' in status or 'not optimal' in status
        return has_feasible and not has_infeasible
            
    def decision_information_print(self, status, show_tensors, show_detailed_tensors, box_width=88):
        
        if show_detailed_tensors: show_tensors=True
        
        if not show_tensors:
        
            if type(status) == str:
                for i in self.VariablesDim.keys():
                    if self.VariablesDim[i] == 0:
                        if self.get([i, (0,)]) != 0:
                            print(f"{i} = {self.get([i, (0,)])}")

                    elif len(self.VariablesDim[i]) == 1:
                        for k in fix_dims(self.VariablesDim[i])[0]:
                            if self.get([i, (k,)]) != 0:
                                print(f"{i}[{k}] = {self.get([i, (k,)])}")
                    else:
                        for k in it.product(*tuple(fix_dims(self.VariablesDim[i]))):
                            if self.get([i, (*k,)]) != 0:
                                print(f"{i}[{k}] = {self.get([i, (*k,)])}".replace("(", "").replace(")", ""))
            else:
                for i in self.VariablesDim.keys():
                    if self.VariablesDim[i] == 0:
                        if self.get_bound([i, (0,)])!=[0,0]:
                            print(f"{i} = {self.get_bound([i, (0,)])}")
                    elif len(self.VariablesDim[i]) == 1:
                        for k in fix_dims(self.VariablesDim[i])[0]:
                            if self.get_bound([i, (k,)])!= [0,0]:
                                print(f"{i}[{k}] = {self.get_bound([i, (k,)])}")
                    else:
                        for k in it.product(*tuple(fix_dims(self.VariablesDim[i]))):
                            if self.get_bound([i, (*k,)]) != [0,0]:
                                print(f"{i}[{k}] = {self.get_bound([i, (*k,)])}".replace("(", "").replace(")", ""))
    
        else:
        
            if show_detailed_tensors: np.set_printoptions(threshold=np.inf)
            
            for i in self.VariablesDim.keys():
                
                if type(status) == str:
    
                    numpy_var = self.get_numpy_var(i) 

                    if type(numpy_var)==np.ndarray:

                        numpy_str = np.array2string(numpy_var, separator=', ', prefix='  ', style=str)
                        rows = numpy_str.split('\n')
                        first_row_len = len(rows[0])
                        for idx, row in enumerate(rows):
                            if idx == 0:
                                print(f"{i} = {row}")
                            else:
                                print(" "*(len(f"{i} =")-1)+row)
                    else:
                        print(f"{i} = {numpy_var}")
                                        
construct = make_model = implementor = implement = Implement

class MADM:

    def __init__(self, method, name='madm_problem', interface='pydecision'):
        
        """
        Initializes an instance of MADM.

        Parameters:
        method (str): The solution method to use (e.g. 'ahp', 'topsis', 'auto', 'mix').
            When 'auto', the method is detected from the inputs provided.
            When 'mix', multiple algorithms are run and results are ensemble-ranked.
        name (str): The name of the problem.
        interface (str): The solver interface (default 'pydecision').

        Returns:
        None
        """
        
        self.model_name = name
        self.interface_name = 'pyDecision.algorithm' if interface == 'pydecision' else interface
        self.madam_method = method

        if self.interface_name == 'pyDecision.algorithm':
            _NO_SUFFIX = {
                'auto', 'mix', 'bwm', 'bwm_s', 'fuzzy_bwm',
                'electre_i', 'electre_i_s', 'electre_i_v',
                'electre_ii', 'electre_iii', 'electre_iv',
                'electre_tri_b', 'electre_tri_c', 'electre_tri_nb', 'electre_tri_nc',
                'promethee_i', 'promethee_ii', 'promethee_iii',
                'promethee_iv', 'promethee_v', 'promethee_vi',
                'promethee_gaia', 'ec_promethee',
                'e_i', 'e_i_s', 'e_i_v', 'e_ii', 'e_iii', 'e_iv',
                'e_tri_b', 'e_tri_c', 'e_tri_nb', 'e_tri_nc',
            }
            if "_method" not in method and method not in _NO_SUFFIX:
                self.madam_method = method + "_method"

            self.loaded_module = None

        if self.interface_name == 'feloopy':
            if "_method" not in method and 'auto' not in method and 'mix' not in method and 'electre' not in method and 'promethee' not in method:
                self.madam_method = method + "_method"

        if method == 'auto':
            self.madam_method = 'auto'

        if method == 'mix':
            self.madam_method = 'mix'

        self.solver_options = dict()

        self.get_tensor = self.get_numpy_var

        self.interface = interface
        self.solver = method
        self.name = self.model_name
        self.method = 'madm'
        self.em = self
        self.inputdata = None
        self.sensitivity_analyzed = False
        self.number_of_objectives = 0
        self.solutions = {}
        self.key_vars = []
        self.approach = None
        self.dataset_size = None
        self.should_benchmark = False
        self.ben_results = None
        self.cpt = 0
        self.mgt = 0
        self.track_history = False
        self.objective_values = None
        self.directions = None
        self.utility_functions = None

        self.features = {
            'weights_found': False,
            'ranks_found': False,
            'inconsistency_found': False,
            'dpr_found': False,
            'dmr_found': False,
            'rpc_found': False,
            'rmc_found': False,
            'concordance_found': False,
            'discordance_found': False,
            'dominance_found': False,
            'kernel_found': False,
            'dominated_found': False,
            'global_concordance_found': False,
            'credibility_found': False,
            'dominance_s_found': False,
            'dominance_w_found': False,
            'd_rank_found': False,
            'a_rank_found': False,
            'n_rank_found': False,
            'p_rank_found': False,
            'classification_found': False,
            'selection_found': False,
            'mix_found': False,
            'constraint_labels': [],
            'solver_options': {},
        }
        
    def healthy(self):
        return True

    def add_criteria_set(self, index='', bound=None, step=1, to_list=False):
        """
        Adds a criteria set.

        Parameters:
        index (str): The index of the criteria set.
        bound (tuple): The range of the criteria set.
        step (int): The step size for the criteria set.
        to_list (bool): Whether to return the criteria set as a list or a set.

        Returns:
        set or list: The criteria set.
        """
        if bound is None and not index:
            raise ValueError('Either bound or index must be provided.')

        start, end = bound if bound else (0, len(index))
        criteria_set = [f'{index}{i}' for i in range(start, end, step)]

        return set(criteria_set) if not to_list else list(criteria_set)

    def add_alternatives_set(self, index='', bound=None, step=1, to_list=False):
        """
        Adds an alternatives set.

        Parameters:
        index (str): The index of the alternatives set.
        bound (tuple): The range of the alternatives set.
        step (int): The step size for the alternatives set.
        to_list (bool): Whether to return the alternatives set as a list or a set.

        Returns:
        set or list: The alternatives set.
        """
        if bound is None and not index:
            raise ValueError('Either bound or index must be provided.')

        start, end = bound if bound else (0, len(index))
        alternatives_set = [f'{index}{i}' for i in range(start, end, step)]

        self.features['number_of_alternatives'] = len(alternatives_set)
        
        return set(alternatives_set) if not to_list else list(alternatives_set)

    def add_uf(self, functions):
        self.features['uf_defined'] = True
        self.utility_functions = functions

    def add_dm(self, data):

        self.features['dm_defined'] = True
        self.decision_matrix = np.array(data, dtype=float)

        if self.features.get('uf_defined', False) and self.utility_functions is not None:
            for j, fn in enumerate(self.utility_functions):
                if fn is not None:
                    self.decision_matrix[:, j] = np.array([fn(x) for x in self.decision_matrix[:, j]], dtype=float)

        if self.madam_method != 'electre_tri_b' and 'cpp_tri' not in self.madam_method:
            self.solver_options['dataset'] = self.decision_matrix
        elif 'cpp_tri' in self.madam_method:
        
            self.solver_options['decision_matrix'] = self.decision_matrix
        else:
            self.solver_options['performance_matrix'] = self.decision_matrix

    def add_profiles(self, data):

        self.features['profiles_defined'] = True
        self.profiles_data = np.array(data)
        self.solver_options['profiles'] = self.profiles_data

    def add_fcim(self, data):

        self.features['cim_defined'] = True
        self.influence_matrix = np.array(data)
        self.solver_options['dataset'] = self.influence_matrix

    def add_cim(self, data):

        self.features['cim_defined'] = True
        self.influence_matrix = np.array(data)
        self.solver_options['dataset'] = self.influence_matrix

    def add_bocv(self, data):
        
        self.features['bocv_defined'] = True
        self.best_to_others = np.array(data)
        self.solver_options['mic'] = self.best_to_others

    def add_owcv(self, data):

        self.features['owcv_defined'] = True
        self.others_to_worst = np.array(data)
        self.solver_options['lic'] = self.others_to_worst

    def add_fbocv(self, data):
        
        self.features['fbocv_defined'] = True
        self.best_to_others = data
        self.solver_options['mic'] = self.best_to_others

    def add_fowcv(self, data):

        self.features['fowcv_defined'] = True
        self.others_to_worst = data
        self.solver_options['lic'] = self.others_to_worst

    def add_fim(self, data):
        self.features['fim_defined'] = True
        self.fuzzy_influence_matrix = np.array(data)
        self.solver_options['dataset'] = self.fuzzy_influence_matrix

    def add_fdm(self, data):

        self.features['fdm_defined'] = True
        self.fuzzy_decision_matrix = np.array(data)
        self.solver_options['dataset'] = self.fuzzy_decision_matrix

    def add_wv_lb(self,data):
        self.features['wv_lb_defined'] = True
        self.solver_options['W_lower'] = np.array(data).tolist()

    def add_wv_ub(self,data):
        self.features['wv_ub_defined'] = True
        self.solver_options['W_upper'] = np.array(data)

    def add_wv(self,data):

        self.features['wv_defined'] = True
        self.weights = np.array(data)
        if self.madam_method not in ['electre_i','electre_i_s', 'electre_i_v', 'electre_ii', 'electre_iii', 'electre_tri_b', 'promethee_i','promethee_ii','promethee_iii', 'promethee_iv', 'promethee_v', 'promethee_gaia']:
            self.solver_options['weights'] = self.weights
        else:
            self.solver_options['W'] = self.weights

    def add_fwv(self,data):

        self.features['fwv_defined'] = True
        self.fuzzy_weights = data
        self.solver_options['weights'] = self.fuzzy_weights

    def add_bt(self, data):

        self.features['b_threshold_defined'] = True
        self.b_threshold = np.array(data).tolist()
        self.solver_options['B'] = self.b_threshold

    def add_grades(self, data):
        self.solver_options['grades'] = np.array(data)

    def add_lbt(self,data):

        self.features['lb_threshold_defined'] = True
        self.lb_threshold =  np.array(data) 
        if self.madam_method not in ['spotis_method']:
            self.solver_options['lower'] = self.lb_threshold
        if self.madam_method in ['spotis_method']:
            self.solver_options['s_min'] = self.lb_threshold
        
    def add_ubt(self,data):

        self.features['ub_threshold_defined'] = True
        self.ub_threshold =  np.array(data) 
        if self.madam_method not in ['spotis_method']:
            self.solver_options['upper'] = self.ub_threshold
        if self.madam_method in ['spotis_method']:
            self.solver_options['s_max'] = self.ub_threshold

    def add_qt(self,data):

        self.features['q_threshold_defined'] = True
        self.q_threshold =  np.array(data) 
        self.solver_options['Q'] = self.q_threshold

    def add_pt(self,data):

        self.features['p_threshold_defined'] = True
        self.p_threshold =  np.array(data) 
        self.solver_options['P'] = self.p_threshold

    def add_st(self,data):

        self.features['s_threshold_defined'] = True
        self.s_threshold =  np.array(data) 
        self.solver_options['S'] = self.s_threshold

    def add_vt(self,data):

        self.features['v_threshold_defined'] = True
        self.v_threshold =  np.array(data) 
        self.solver_options['V'] = self.v_threshold

    def add_uf(self,data):

        self.features['uf_defined'] = True
        self.utility_functions = data 
        if self.madam_method not in ['promethee_i', 'promethee_ii', 'promethee_iii', 'promethee_iv', 'promethee_v', 'promethee_vi', 'promethee_gaia']:
            self.solver_options['utility_functions'] = self.utility_functions
        else:
            self.solver_options['F'] = self.utility_functions

    def add_cr(self,data):

        self.features['cr_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['criteria_rank'] = self.criteria_pairwise_comparison_matrix

    def add_cp(self,data):

        self.features['cpm_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['criteria_priority'] = self.criteria_pairwise_comparison_matrix

    def add_er(self,data):

        self.features['er_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['experts_rank'] = self.criteria_pairwise_comparison_matrix

    def add_erc(self,data):

        self.features['erc_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['experts_rank_criteria'] = self.criteria_pairwise_comparison_matrix

    def add_era(self,data):

        self.features['era_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['experts_rank_alternatives'] = self.criteria_pairwise_comparison_matrix

    def add_cpcm(self,data):

        self.features['cpm_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['dataset'] = self.criteria_pairwise_comparison_matrix

    def add_pcm(self, data):
        self.add_cpcm(data)

    def add_ppfcpcm(self,data):

        self.features['ppfcpm_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['comparison_matrix'] = self.criteria_pairwise_comparison_matrix
        
    def add_ppfcpcm(self,data):

        self.features['ppfcpm_defined'] = True
        self.criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['comparison_matrix'] = self.criteria_pairwise_comparison_matrix

    def add_fcpcm(self,data):

        self.features['fcpcm_defined'] = True
        self.fuzzy_criteria_pairwise_comparison_matrix = np.array(data)
        self.solver_options['dataset'] = self.fuzzy_criteria_pairwise_comparison_matrix

    def add_apcm(self,data):

        self.features['apcm_defined'] = True
        self.alternatives_pairwise_comparison_matrix = np.array(data)
        self.solver_options['dataset'] = self.alternatives_pairwise_comparison_matrix

    def add_fapcm(self,data):

        self.features['fapm_defined'] = True
        self.fuzzy_alternatives_pairwise_comparison_matrix = np.array(data)
        self.solver_options['dataset'] = self.fuzzy_alternatives_pairwise_comparison_matrix

    def add_con_max_criteria(self,data):

        self.solver_options['criteria'] = data

    def add_con_cost_budget(self,cost, budget):

        self.solver_options['cost'] = cost
        self.solver_options['budget'] = budget

    def add_con_forbid_selection(self,selections):
        self.solver_options['forbidden'] = selections

    def _detect_method(self):
        """Auto-detect the best MADM method from the inputs provided."""
        has_pcm = self.features.get('cpm_defined', False)
        has_dm = self.features.get('dm_defined', False)
        has_wv = self.features.get('wv_defined', False)
        has_bocv = self.features.get('bocv_defined', False)
        has_owcv = self.features.get('owcv_defined', False)
        has_cim = self.features.get('cim_defined', False)
        has_profiles = self.features.get('profiles_defined', False)
        has_fuzzy = self.features.get('fcpcm_defined', False) or self.features.get('fapm_defined', False)

        if has_bocv or has_owcv:
            return 'bw_method'
        if has_pcm and has_fuzzy:
            return 'fuzzy_ahp_method'
        if has_pcm:
            return 'ahp_method'
        if has_cim:
            return 'dematel_method'
        if has_profiles and has_dm:
            return 'electre_tri_b'
        if has_dm and has_wv:
            return 'topsis_method'
        if has_dm:
            return 'mix'
        return 'topsis_method'

    def _mix_sol(self, directions=[], show_graph=None, show_log=None):
        """Run multiple algorithms procedurally based on user inputs and ensemble the results."""

        has_pcm = self.features.get('cpm_defined', False)
        has_dm = self.features.get('dm_defined', False)
        has_wv = self.features.get('wv_defined', False)
        has_bocv = self.features.get('bocv_defined', False)
        has_owcv = self.features.get('owcv_defined', False)
        has_cim = self.features.get('cim_defined', False)
        has_profiles = self.features.get('profiles_defined', False)
        has_fcpcm = self.features.get('fcpcm_defined', False) or self.features.get('fapm_defined', False)

        n_c = self._ds.shape[1] if self._ds is not None else 0
        if not directions or directions is None:
            directions = [1] * n_c

        _OBJECTIVE_WEIGHTING = [
            'entropy_method', 'critic_method', 'cilos_method',
            'idocriw_method', 'merec_method', 'seca_method', 'mpsi_method',
        ]
        _COMPARATIVE_WEIGHTING = [
            'ahp_method', 'anp_method', 'fucom_method', 'roc_method',
            'rrw_method', 'rsw_method',
        ]
        _FUZZY_WEIGHTING = ['fuzzy_ahp_method', 'ppf_ahp_method', 'fuzzy_bw_method']
        _RANKING = [
            'topsis_method', 'vikor_method', 'aras_method', 'copras_method',
            'edas_method', 'codas_method', 'moora_method', 'mabac_method',
            'waspas_method', 'marcos_method', 'cradis_method', 'mairca_method',
            'cocoso_method', 'saw_method', 'ocra_method',
            'gra_method', 'todim_method', 'spotis_method',
            'multimoora_method', 'moosra_method',
            'borda_method', 'copeland_method', 'psi_method', 'piv_method',
            'rov_method', 'regime_method', 'oreste_method',
        ]
        _STANDALONE = ['dematel_method', 'wings_method']

        _steps = []
        if has_cim:
            _steps.append(('standalone', _STANDALONE))
        if has_pcm and has_fcpcm:
            _steps.append(('weighting', _FUZZY_WEIGHTING))
        elif has_pcm:
            _steps.append(('weighting', _COMPARATIVE_WEIGHTING))
        if has_bocv or has_owcv:
            _steps.append(('weighting', ['bw_method', 'simplified_bw_method']))
        if has_dm and not has_wv:
            _steps.append(('weighting', _OBJECTIVE_WEIGHTING))
        _steps.append(('ranking', _RANKING))

        all_ranks = {}
        all_scores = {}
        all_weights = {}
        collected_weights = []

        for stage, method_list in _steps:
            for method_name in method_list:
                saved = dict(self.solver_options)
                try:
                    if self.interface_name == 'pyDecision.algorithm':
                        self.loaded_module = importlib.import_module(self.interface_name)
                        self.madam_algorithm = getattr(self.loaded_module, method_name)

                    local_opts = dict(self.solver_options)
                    if 'criterion_type' not in local_opts and directions:
                        local_opts['criterion_type'] = directions

                    if stage == 'ranking' and collected_weights and 'weights' not in local_opts:
                        local_opts['weights'] = np.mean(collected_weights, axis=0).tolist()

                    import inspect as _inspect
                    sig = _inspect.signature(self.madam_algorithm)
                    accepted = set(sig.parameters.keys())
                    filtered = {k: v for k, v in local_opts.items() if k in accepted}

                    _devnull = open(os.devnull, 'w')
                    _old_stdout = sys.stdout
                    _old_stderr = sys.stderr
                    sys.stdout = _devnull
                    sys.stderr = _devnull
                    try:
                        start = time.time()
                        result = self.madam_algorithm(**filtered)
                        finish = time.time()
                    finally:
                        sys.stdout = _old_stdout
                        sys.stderr = _old_stderr
                        _devnull.close()

                    scores = self._extract_alternative_scores(result)

                    if scores is not None:
                        scores_arr = np.array(scores, dtype=float)
                        if np.any(np.isnan(scores_arr)) or np.any(np.isinf(scores_arr)):
                            self.solver_options = saved
                            continue

                        if stage == 'weighting':
                            n_c = self._ds.shape[1] if self._ds is not None else 0
                            w_arr = scores_arr.flatten()
                            if len(w_arr) == n_c:
                                collected_weights.append(w_arr.tolist())
                                all_weights[method_name] = w_arr.tolist()
                        else:
                            all_scores[method_name] = scores_arr
                            all_ranks[method_name] = np.argsort(scores_arr)[::-1]

                    self.solver_options = saved
                except Exception:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    self.solver_options = saved
                    continue

        if not all_ranks:
            raise RuntimeError("mix: no algorithm produced results. Check inputs.")

        n_alt = len(next(iter(all_ranks.values())))
        n_methods = len(all_ranks)

        rank_matrix = np.zeros((n_methods, n_alt), dtype=float)
        for i, (mname, r) in enumerate(all_ranks.items()):
            rank_matrix[i, :] = np.array(r, dtype=float)[:n_alt]

        avg_ranks = np.mean(rank_matrix, axis=0)
        final_order = np.argsort(avg_ranks)

        self.mix_ranks = rank_matrix
        self.mix_method_names = list(all_ranks.keys())
        self.mix_avg_ranks = avg_ranks
        self.mix_final_order = final_order
        self.mix_weights = all_weights

        self.ranks = final_order
        self.features['ranks_found'] = True
        self.features['mix_found'] = True
        self.features['number_of_alternatives'] = n_alt
        if self._ds is not None:
            self.features['number_of_criteria'] = self._ds.shape[1]

    def _extract_ranks_from_scores(self, scores):
        if scores is None:
            return None
        scores = np.array(scores, dtype=float).flatten()
        return np.argsort(scores)[::-1]

    def _extract_alternative_scores(self, result):
        """Extract a 1D score array (n_alternatives,) from any pyDecision result tuple."""
        if result is None:
            return None

        if isinstance(result, np.ndarray) and result.ndim == 1:
            return result
        if isinstance(result, (list, tuple)):
            for item in result:
                arr = np.array(item, dtype=float)
                if arr.ndim == 1 and len(arr) >= 2:
                    return arr
        if isinstance(result, (list, tuple)):
            parts = []
            for item in result:
                arr = np.array(item, dtype=float)
                if arr.ndim == 0:
                    parts.append(float(arr))
                elif arr.ndim == 1 and len(arr) >= 2:
                    return arr
            if len(parts) >= 2:
                return np.array(parts, dtype=float)
        return None

    def sol(self, directions = [], solver_options=dict(), show_graph=None, show_log=None):

        self._ranks_final = False

        if self.madam_method == 'auto':
            self.madam_method = self._detect_method()
            self.solver = self.madam_method

        if self.madam_method == 'mix':
            self.solver_options.update(solver_options)
            self._ds = self.solver_options.get('dataset', self.solver_options.get('performance_matrix', self.solver_options.get('_original_dm', self.solver_options.get('data'))))
            self._mix_sol(directions=directions, show_graph=show_graph, show_log=show_log)
            self.cpt = 0
            return

        if self.madam_method in ['promethee_ii', 'promethee_iv', 'promethee_vi']:
            self.solver_options['sort'] = False

        if self.madam_method in ['waspas_method']:
            if 'lambda_value' not in self.solver_options.keys():
                self.solver_options['lambda_value'] = 0.5

        _PROMETHEE_METHODS = {
            'promethee_i', 'promethee_ii', 'promethee_iii', 'promethee_iv',
            'promethee_v', 'promethee_vi', 'promethee_gaia', 'ec_promethee',
        }
        if self.madam_method in _PROMETHEE_METHODS:
            for key, default in [('Q', 0), ('S', 0), ('P', 1), ('F', 'usual')]:
                if key not in self.solver_options:
                    self.solver_options[key] = default

        if self.madam_method == 'flowsort_method':
            for key, default in [('W', None), ('Q', 0), ('S', 0), ('P', 1), ('F', 'usual')]:
                if key not in self.solver_options:
                    if key == 'W' and 'weights' in self.solver_options:
                        self.solver_options['W'] = self.solver_options.pop('weights')
                    elif key != 'W':
                        self.solver_options[key] = default

        if self.interface_name == 'pyDecision.algorithm':
            if self.loaded_module is None:
                self.loaded_module = importlib.import_module(self.interface_name)
            self.madam_algorithm = getattr(self.loaded_module, self.madam_method)
        else:

            if self.madam_method == 'cwdea_method':
                self.madam_algorithm = cwdea_method 

            if self.madam_method == 'lp_method':
                self.madam_algorithm = lp_method 

            if self.madam_method == 'la_method':
                self.madam_algorithm = la_method 

        self.solver_options.update(solver_options)

        _NO_CRITERION_TYPE = {
            'ahp_method', 'anp_method', 'bw_method', 'fuzzy_ahp_method', 'ppf_ahp_method',
            'fuzzy_bw_method', 'simplified_bw_method', 'fucom_method', 'fuzzy_fucom_method',
            'roc_method', 'rrw_method', 'rsw_method',
            'dematel_method', 'fuzzy_dematel_method', 'wings_method', 'opa_method',
            'electre_i', 'electre_i_s', 'electre_i_v', 'electre_ii',
            'electre_iii', 'electre_iv', 'electre_tri_b', 'electre_tri_c',
            'electre_tri_nb', 'electre_tri_nc', 'cpp_tri_method', 'flowsort_method',
            'promethee_i', 'promethee_ii', 'promethee_iii', 'promethee_iv',
            'promethee_v', 'promethee_vi', 'promethee_gaia',
            'utadis_i_method', 'utadis_ii_method', 'utadis_iii_method',
        }

        _NO_WEIGHTS = {
            'mabac_method', 'psi_method', 'borda_method', 'copeland_method',
            'multimoora_method', 'promethee_vi', 'rancom_method',
            'entropy_method', 'critic_method', 'cilos_method', 'idocriw_method',
            'merec_method', 'seca_method', 'mpsi_method',
        }

        _NO_GRAPH_VERBOSE = {
            'fuzzy_dematel_method',
            'lara_method',
        }

        _USE_PERFORMANCE_MATRIX = {
            'lmaw_method', 'electre_tri_b', 'electre_tri_c',
            'electre_tri_nc', 'cpp_tri_method',
        }

        _SMART_RENAME = {
            'utility_functions': 'grades',
        }

        _PROMETHEE_VI_REMAP = {
            'weights': 'W_lower',
        }

        _USES_CRITERIA_RANK = {'fucom_method', 'roc_method', 'rrw_method', 'rsw_method',
                               'fuzzy_fucom_method'}

        if 'criteria_rank' in self.solver_options and self.madam_method not in _USES_CRITERIA_RANK:
            self.solver_options.pop('criteria_rank', None)

        try:
            if len(directions)!=0 and self.madam_method not in _NO_CRITERION_TYPE:
            
                self.solver_options['criterion_type'] = directions
                self.criteria_directions = directions
        except:
            pass

        if 'dataset' in self.solver_options and self.madam_method in _USE_PERFORMANCE_MATRIX:
            self.solver_options['performance_matrix'] = self.solver_options.pop('dataset')

        if self.madam_method == 'electre_tri_nb' and 'dataset' in self.solver_options:
            self.solver_options['perf_matrix'] = self.solver_options.pop('dataset')

        if 'weights' in self.solver_options and self.madam_method in _NO_WEIGHTS:
            self.solver_options.pop('weights', None)

        if self.madam_method == 'rancom_method' and 'dataset' in self.solver_options:
            ds = self.solver_options['dataset']
            if hasattr(ds, 'ndim') and ds.ndim > 1:
                self.solver_options['data'] = ds[0].tolist() if hasattr(ds[0], 'tolist') else list(ds[0])
                self.solver_options['_original_dm'] = ds
            else:
                self.solver_options['data'] = ds if isinstance(ds, list) else ds.tolist()
            self.solver_options.pop('dataset', None)

        if self.madam_method in ('lara_method', 'sabina_method') and 'criterion_type' in self.solver_options:
            self.solver_options['criteria_type'] = self.solver_options.pop('criterion_type')

        if self.madam_method == 'sabina_method' and 'labels' not in self.solver_options:
            ds = self.solver_options.get('X', self.solver_options.get('dataset'))
            if ds is not None:
                n_alt = ds.shape[0] if hasattr(ds, 'shape') else len(ds)
                self.solver_options['labels'] = [f'a{i+1}' for i in range(n_alt)]

        if self.madam_method == 'lara_method' and 'weights' in self.solver_options:
            self.solver_options['W'] = self.solver_options.pop('weights')

        if self.madam_method == 'flowsort_method' and 'weights' in self.solver_options:
            self.solver_options['W'] = self.solver_options.pop('weights')

        if self.madam_method == 'electre_tri_nb':
            if 'Q' in self.solver_options:
                self.solver_options['q'] = self.solver_options.pop('Q')
            if 'P' in self.solver_options:
                self.solver_options['p'] = self.solver_options.pop('P')
            if 'V' in self.solver_options:
                self.solver_options['v'] = self.solver_options.pop('V')
            if 'weights' in self.solver_options:
                self.solver_options['w'] = self.solver_options.pop('weights')

        _NEEDS_W = {
            'electre_i', 'electre_i_s', 'electre_i_v', 'electre_ii', 'electre_iii',
            'electre_tri_b', 'electre_tri_c', 'electre_tri_nc',
            'promethee_i', 'promethee_ii', 'promethee_iii', 'promethee_iv',
            'promethee_v', 'promethee_vi', 'promethee_gaia',
            'flowsort_method', 'lara_method',
        }
        if self.madam_method in _NEEDS_W and 'weights' in self.solver_options:
            self.solver_options['W'] = self.solver_options.pop('weights')

        if self.madam_method in ('sabina_method',) and 'dataset' in self.solver_options:
            self.solver_options['X'] = self.solver_options.pop('dataset')

        if self.madam_method in ('utadis_i_method', 'utadis_ii_method', 'utadis_iii_method') and 'dataset' in self.solver_options:
            self.solver_options['X'] = self.solver_options.pop('dataset')

        if self.madam_method in ('smart_method', 'odo_ovo_method'):
            if self.madam_method == 'smart_method' and 'utility_functions' in self.solver_options:
                self.solver_options['grades'] = self.solver_options.pop('utility_functions')
            if 'criterion_type' in self.solver_options:
                ct = self.solver_options['criterion_type']
                self.solver_options['criterion_type'] = [
                    'max' if c in ('beneficial', 'max') else 'min' for c in ct
                ]

        if self.madam_method == 'electre_tri_b' and 'profiles' in self.solver_options:
            profiles = self.solver_options.pop('profiles')
            if hasattr(profiles, 'tolist'):
                profiles = profiles.tolist()
            elif isinstance(profiles, np.ndarray):
                profiles = [list(row) for row in profiles]
            self.solver_options['B'] = profiles

        if self.madam_method == 'bw_method':
            self.solver_options['verbose'] = False

        self._ds = self.solver_options.get('dataset', self.solver_options.get('performance_matrix', self.solver_options.get('_original_dm', self.solver_options.get('data'))))

        if self.madam_method == 'spotis_method' and self._ds is not None:
            if 's_min' not in self.solver_options:
                self.solver_options['s_min'] = np.min(self._ds, axis=0)
            if 's_max' not in self.solver_options:
                self.solver_options['s_max'] = np.max(self._ds, axis=0)

        if self.madam_method == 'flowsort_method' and self._ds is not None:
            if 'S' not in self.solver_options:
                self.solver_options['S'] = np.ones(self._ds.shape[1]) * 0.5

        _FUZZY_METHODS = {
            'fuzzy_aras_method', 'fuzzy_copras_method', 'fuzzy_edas_method',
            'fuzzy_moora_method', 'fuzzy_ocra_method', 'fuzzy_topsis_method',
            'fuzzy_vikor_method', 'fuzzy_waspas_method', 'fuzzy_merec_method',
        }
        if self.madam_method in _FUZZY_METHODS and 'weights' in self.solver_options:
            w = self.solver_options['weights']
            w_arr = np.array(w) if not isinstance(w, np.ndarray) else w
            if w_arr.ndim == 2:
                self.solver_options['weights'] = w_arr.reshape(1, *w_arr.shape)
            elif isinstance(w, list) and len(w) > 0 and isinstance(w[0], list) and len(w[0]) > 0 and not isinstance(w[0][0], (list, np.ndarray, tuple)):
                self.solver_options['weights'] = [w]

        self._ds = self.solver_options.get('dataset', self.solver_options.get('performance_matrix', self.solver_options.get('_original_dm', self.solver_options.get('data'))))

        _EXPAND_SCALAR_QPV = {
            'electre_i_s', 'electre_i_v', 'electre_ii', 'electre_iii', 'electre_iv',
            'promethee_i', 'promethee_ii', 'promethee_iii', 'promethee_iv',
            'promethee_v', 'promethee_vi', 'promethee_gaia', 'ec_promethee',
        }
        if self.madam_method in _EXPAND_SCALAR_QPV and self._ds is not None:
            n_crit = self._ds.shape[1]
            for key in ('Q', 'P', 'V', 'S'):
                val = self.solver_options.get(key)
                if val is not None and not isinstance(val, (list, np.ndarray)):
                    self.solver_options[key] = np.full(n_crit, float(val))
            if 'F' in self.solver_options:
                val = self.solver_options['F']
                if isinstance(val, str):
                    self.solver_options['F'] = [val] * n_crit

        self.auxiliary_solver_options = dict()
        if self.madam_method not in _NO_GRAPH_VERBOSE:
            if show_graph is not None:
                self.auxiliary_solver_options['graph'] = show_graph
            if show_log is not None:
                self.auxiliary_solver_options['verbose'] = show_log

        _devnull = open(os.devnull, 'w')
        _old_stdout = sys.stdout
        _old_stderr = sys.stderr
        sys.stdout = _devnull
        sys.stderr = _devnull
        try:
            self.start = time.time()
            self.result =  self.madam_algorithm(**{**self.solver_options, **self.auxiliary_solver_options})
            self.finish = time.time()
        except TypeError:
            import inspect as _inspect
            sig = _inspect.signature(self.madam_algorithm)
            accepted = set(sig.parameters.keys())
            all_opts = {**self.solver_options, **self.auxiliary_solver_options}
            filtered = {k: v for k, v in all_opts.items() if k in accepted}
            self.start = time.time()
            self.result = self.madam_algorithm(**filtered)
            self.finish = time.time()
        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
            _devnull.close()

        self.status = 'feasible (solved)'
        self.cpt = self.finish - self.start

        if  self.madam_method in np.array(WEIGHTING_ALGORITHMS)[:,0]:

            self.features['weights_found'] = True
            
            try:
                self.features['number_of_criteria'] = len(self.result[0])
            except:
                self.features['number_of_criteria'] = len(self.result)
                self.weights = self.result

            if self.madam_method in ['simplified_bw_method']:
                self.features['inconsistency_found'] = True
                self.weights = self.result[1]
                self.inconsistency = self.result[0]

            if self.madam_method in ['lp_method']:

                self.features['inconsistency_found'] = True
                self.weights = self.result[0]
                self.inconsistency = self.result[1]
                
            if self.madam_method in ['ahp_method', 'ppf_ahp_method']:

                self.features['inconsistency_found'] = True
                self.weights = self.result[0]
                self.inconsistency = self.result[1]

            if self.madam_method == 'anp_method':
                _anp = np.array(self.result, dtype=float)
                self.weights = _anp[:, 0] if _anp.ndim == 2 else _anp

            if self.madam_method in ['fuzzy_ahp_method']:

                self.fuzzy_weights = self.result[0]
                self.weights = self.result[2]
                self.inconsistency = self.result[3]

            if self.madam_method in ['fuzzy_bw_method']:

                self.features['number_of_criteria'] = len(self.result[2])
                self.fuzzy_weights = self.result[2]
                self.weights = self.result[3]
                self.inconsistency = self.result[1]
                self.epsilon = self.result[0]
                self.features['inconsistency_found'] = True

            if self.madam_method in ['fuzzy_fucom_method']:
                self.weights = self.result[1]
                self.fuzzy_weights = self.result[0]

        if  self.madam_method in np.array(RANKING_ALGORITHMS)[:,0]:

            self.features['ranks_found'] = True
            self.features['number_of_criteria'] = self._ds.shape[1]
            self.features['number_of_alternatives'] = self._ds.shape[0]
            self.ranks = self.result

            if self.madam_method in ['rancom_method']:
                self.features['weights_found'] = True
                self.weights = np.array(self.result.weights)
                self._ds = np.array(self._ds) if not isinstance(self._ds, np.ndarray) else self._ds

            if self.madam_method in ['promethee_vi']:
                self.features['ranks_found'] = True

            if self.madam_method in ['multimoora_method', 'waspas_method', 'fuzzy_waspas_method']:
                self.ranks = self.result[2]

            if self.madam_method in ['vikor_method', 'fuzzy_vikor_method']:
                self.ranks = self.result[2]

            if self.madam_method in ['lara_method']:
                self.ranks = self.result[0]
                self.lara_scores = self.result[1]
                self.lara_graph = self.result[2]

            if self.madam_method not in ['promethee_vi']:
                self.ranks = self.get_ranks()
            else:
                self.ranks = self.result[1]
                self.middle_ranks = self.get_ranks()
                self.ranks = self.result[2]
                self.upper_ranks =  self.get_ranks()
                self.ranks = self.result[0]
                self.lower_ranks =  self.get_ranks()
  
                self.ranks = np.array([self.lower_ranks, self.middle_ranks, self.upper_ranks ]).T

        if self.madam_method in np.array(SPECIAL_ALGORITHMS)[:,0]:

            if self.madam_method == 'opa_method':

                self.expert_weights = self.result[0]
                self.criteria_weights = self.result[1]
                self.alternative_weights = self.result[2]

                self.weights = {'experts': self.expert_weights, 
                                'criteria': self.criteria_weights,
                                'alternatives': self.alternative_weights}

            if self.madam_method == 'cwdea_method':

                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['weights_found'] = True
                self.features['ranks_found'] = True
                self.ranks, self.weights = self.result
                self.ranks = self.get_ranks()
                self._ranks_final = True

            if 'dematel' in self.madam_method:
                
                self.features['number_of_criteria'] = self._ds.shape[1]

                self.features['weights_found'] = True
                self.features['dpr_found'] = True
                self.features['dmr_found'] = True
                self.D_plus_R, self.D_minus_R, self.weights = self.result
                self.weights = self.result[2]

            if  self.madam_method in ['electre_i', 'electre_i_v']:
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.concordance, self.discordance, self.dominance, self.kernel, self.dominated = self.result
                def _parse_kernel_item(s):
                    s = str(s)
                    digits = ''.join(filter(str.isdigit, s))
                    return int(digits) - 1 if digits else s
                self.kernel = [_parse_kernel_item(s) for s in self.kernel]
                self.dominated = [_parse_kernel_item(s) for s in self.dominated]
                self.features['concordance_found'] = True
                self.features['discordance_found'] = True
                self.features['dominance_found'] = True
                self.features['kernel_found'] = True
                self.features['dominated_found'] = True

            if self.madam_method == 'electre_i_s':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.global_concordance, self.discordance, self.kernel, self.credibility, self.dominated = self.result
                def _parse_kernel_item(s):
                    s = str(s)
                    digits = ''.join(filter(str.isdigit, s))
                    return int(digits) - 1 if digits else s
                self.kernel = [_parse_kernel_item(s) for s in self.kernel]
                self.dominated = [_parse_kernel_item(s) for s in self.dominated]
                self.features['global_concordance_found'] = True
                self.features['discordance_found'] = True
                self.features['kernel_found'] = True
                self.features['credibility_found'] = True
                self.features['dominated_found'] = True

            if self.madam_method == 'electre_ii':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.concordance, self.discordance, self.dominance_s, self.dominance_w, self.rank_D, self.rank_A, self.rank_N, self.rank_P = self.result
                self.features['concordance_found'] = True
                self.features['discordance_found'] = True
                self.features['dominance_s_found'] = True
                self.features['dominance_w_found'] = True
                self.features['d_rank_found'] = True
                self.features['a_rank_found'] = True
                self.features['n_rank_found'] = True
                self.features['p_rank_found'] = True

                self.rank_D = [[int(''.join(filter(str.isdigit, s)))-1 for s in sd] for sd in self.rank_D]
                self.rank_A = [[int(''.join(filter(str.isdigit, s)))-1 for s in sd] for sd in self.rank_A]
                self.rank_N = [int(''.join(filter(str.isdigit, s)))-1 if isinstance(s, str) else s for s in self.rank_N]
                self.rank_P = [int(''.join(filter(str.isdigit, s)))-1 if isinstance(s, str) else s for s in self.rank_P]

            if self.madam_method == 'electre_iii':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.global_concordance, self.credibility, self.rank_D, self.rank_A, self.rank_N, self.rank_P = self.result
                self.features['global_concordance_found'] = True
                self.features['credibility_found'] = True
                self.features['d_rank_found'] = True
                self.features['a_rank_found'] = True
                self.features['n_rank_found'] = True
                self.features['p_rank_found'] = True

                def _parse_electre_rank(item):
                    item = str(item)
                    if ';' not in item:
                        return int(item[1:])
                    return [int(sub_item[1:]) for sub_item in item.split('; ')]

                self.rank_D = [_parse_electre_rank(item) for item in self.rank_D]
                self.rank_A = [_parse_electre_rank(item) for item in self.rank_A]

            if self.madam_method == 'electre_iv':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.credibility, self.rank_D, self.rank_A, self.rank_N, self.rank_P = self.result
                self.features['credibility_found'] = True
                self.features['d_rank_found'] = True
                self.features['a_rank_found'] = True
                self.features['n_rank_found'] = True
                self.features['p_rank_found'] = True

                def _parse_electre_rank(item):
                    item = str(item)
                    if ';' not in item:
                        return int(item[1:])
                    return [int(sub_item[1:]) for sub_item in item.split('; ')]

                self.rank_D = [_parse_electre_rank(item) for item in self.rank_D]
                self.rank_A = [_parse_electre_rank(item) for item in self.rank_A]

            if self.madam_method == 'electre_tri_b':
                self.features['number_of_criteria'] = self.solver_options['performance_matrix'].shape[1]
                self.features['number_of_alternatives'] = self.solver_options['performance_matrix'].shape[0]
                self.classification = self.result
                self.features['classification_found'] = True

            if 'cpp_tri' in self.madam_method:
                self.features['number_of_criteria'] = self.solver_options['decision_matrix'].shape[1]
                self.features['number_of_alternatives'] = self.solver_options['decision_matrix'].shape[0]
                self.classification = self.result
                self.features['classification_found'] = True
                
            if self.madam_method == 'promethee_v':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.selection = self.result
                self.features['selection_found'] = True

            if self.madam_method in ['promethee_i', 'promethee_iii']:
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.rank_P = self.result
                self.features['p_rank_found'] = True
            
            if self.madam_method in ['wings_method']:
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['weights_found'] = True
                self.features['rpc_found'] = True
                self.features['rmc_found'] = True

                self.R_plus_C, self.R_minus_C, self.weights = self.result

            if self.madam_method == 'anp_method':
                self.features['weights_found'] = True
                self.features['number_of_criteria'] = len(self.result)
                self.weights = self.result

            if self.madam_method == 'mpsi_method':
                self.features['weights_found'] = True
                self.features['number_of_criteria'] = len(self.result)
                self.weights = self.result

            if self.madam_method == 'lara_method':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['ranks_found'] = True
                self.ranks = self.result[0]

            if self.madam_method == 'rafsi_method':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['ranks_found'] = True
                self.ranks = self.result

            if self.madam_method == 'odo_ovo_method':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['ranks_found'] = True
                self.ranks = self.result

            if self.madam_method in ('electre_tri_c', 'electre_tri_nb', 'electre_tri_nc'):
                _ds = self.solver_options.get('performance_matrix', self.solver_options.get('perf_matrix', self.solver_options.get('dataset')))
                self.features['number_of_criteria'] = _ds.shape[1]
                self.features['number_of_alternatives'] = _ds.shape[0]
                self.classification = self.result[0]
                self.credibility = self.result[1]
                self.features['classification_found'] = True
                self.features['credibility_found'] = True

            if self.madam_method == 'flowsort_method':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.classification = self.result
                self.features['classification_found'] = True

            if self.madam_method == 'ec_promethee':
                self.features['number_of_criteria'] = self._ds.shape[1]
                self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['ranks_found'] = True
                self.ranks = self.result[1]

            if self.madam_method in ('utadis_i_method', 'utadis_ii_method', 'utadis_iii_method'):
                self.classification = self.result
                self.features['classification_found'] = True
                if isinstance(self.result, dict) and 'w' in self.result:
                    self.features['weights_found'] = True
                    self.weights = self.result['w']

            if self.madam_method == 'sabina_method':
                self.ranks, self.sabina_scores, self.sabina_res = self.result
                self._ds = self.solver_options.get('X', self.solver_options.get('dataset'))
                if self._ds is not None:
                    self.features['number_of_criteria'] = self._ds.shape[1]
                    self.features['number_of_alternatives'] = self._ds.shape[0]
                self.features['weights_found'] = True
                self.features['ranks_found'] = True
                self.weights = self.sabina_res.get('weights', None)
            
    def get_numpy_var(self, input):

        if input == 'frv':
            return self.ranks
        
        if input == 'rv':
            return self.ranks

        if input == 'wv':
        
            return self.weights

        if input == 'fwv':
            return self.fuzzy_weights

        if input == 'dmrv':
            return self.D_minus_R

        if input == 'dprv':
            return self.D_plus_R

        if input == 'rmcv':
            return self.R_minus_C

        if input == 'rpcv':
            return self.R_plus_C

        if input in ['dominated']:
            return self.dominated
        
        if input in ['concordance', 'cmm']:
            return self.concordance

        if input in ['discordance', 'dcm']:
            return self.discordance

        if input in ['kernel']:
            return self.kernel
        
        if input in ['dominance', 'dmm']:
            return self.dominance
        
        if input in ['dominance_s', 'dmm_s']:
            return self.dominance_s

        if input in ['dominance_w', 'dmm_w']:
            return self.dominance_w
         
        if input in ['global_concordance', 'gcm']:
            return self.global_concordance

        if input in ['credibility', 'crm']:
            return self.credibility
    
        if input in ['rank_d', 'd_rv']:
            return self.rank_D

        if input in ['rank_a', 'a_rv']:
            return self.rank_A

        if input in ['rank_p', 'p_rv']:
            return self.rank_P
        
        if input in ['classification', 'class', 'c']:
            return self.classification     
          
    def get_ranks_base(self):
        
        try:
            return np.array(self.ranks)[:,1]
        except: 
            return np.array(self.ranks)
        
    def get_ranks(self):

        if getattr(self, '_ranks_final', False):
            return self.ranks

        if self.madam_method not in ['la_method']:

            if self.madam_method not in ['vikor_method', 'fuzzy_vikor_method']:

                try:
                    if self.madam_method not in ['borda_method', 'cradis_method', 'mairca_method', 'oreste_method', 'piv_method', 'spotis_method']:

                        return np.argsort(np.array(self.ranks)[:,1])[::-1]

                    else:
                        return np.argsort(np.array(self.ranks)[:,1])

                except:
                    if self.madam_method not in ['borda_method','cradis_method', 'mairca_method', 'oreste_method', 'piv_method', 'spotis_method']:
                        return np.argsort(np.array(self.ranks))[::-1]
                    else:
                        return np.argsort(np.array(self.ranks))

            else:

                arr = np.array(self.ranks)
                if arr.ndim == 2:
                    return np.int64(arr[:,0]) - 1
                return np.int64(arr)
            
        else:
            return self.ranks
        
    def get_status(self):
        return self.status

    def get_inconsistency(self):
        return self.inconsistency

    def get_weights(self):

        return self.weights
        
    def get_fuzzy_weights(self):

        return self.fuzzy_weights
        
    def get_crisp_weights(self):

        return self.crisp_weights
        
    def get_normalized_weights(self):
        
        return self.normalized_weights

    def clean_report(self,**kwargs):

        clear_console()
        self.report(**kwargs)
        
    def report(self, style=1, width=78, save=None, show_elements=False, key_vars=[], hidden_variables=False, show_tensors=None):
        if show_tensors is None:
            show_tensors = False

        ReportEngine(self).report_madm(style=style, save=save, width=width, show_tensors=show_tensors)
        return self

    def display_as_tensor(self, name, numpy_var, detailed):
        if detailed:
            np.set_printoptions(threshold=np.inf)

        if isinstance(numpy_var, np.ndarray):
            tensor_str = np.array2string(numpy_var, separator=', ', prefix='│ ') #Style argument deprecated.
            rows = tensor_str.split('\n')
            first_row_len = len(rows[0])
            for k, row in enumerate(rows):
                if k == 0:
                    left_align(f"{name} = {row}")
                else:
                    left_align(" " * (len(f"{name} =") - 1) + row)
        else:
            left_align(f"{name} = {numpy_var}")

madm = MADM

class search(model,Implement):
    """Solve a decision model and analyze its results.

    Parameters
    ----------
    environment : callable
        A function ``environment(m) -> m`` that receives a ``model`` instance
        and returns it after adding variables, constraints, and objectives.
        This is the *only* required argument in practice.
    name : str, optional
        A descriptive name for the model (default ``"model_name"``).  Used in
        reports and output files.
    method : str or None, optional
        Solution method.  One of:

        - ``"exact"`` — deterministic LP/MILP/QP/NLP/MINLP (default).
        - ``"heuristic"`` — metaheuristic solvers (mealpy, niapy, etc.).
        - ``"convex"`` — convex optimization via cvxpy or linopy.
        - ``"constraint"`` — constraint programming (OR-Tools CP-SAT, etc.).
        - ``"uncertain"`` — robust/stochastic optimization (RSOME DRO, etc.).
        - ``"madm"`` — multi-criteria decision making (TOPSIS, VIKOR, etc.).
        - ``None`` — automatically inferred from ``solver`` using the
          algorithm registries.  Falls back to ``"exact"`` when the solver
          name is not found.
    approach : str, optional
        Multi-objective scalarization approach.  One of:

        - ``"nwsm"`` — Normalized Weighted Sum Method (default).
        - ``"ecm"`` — Epsilon-Constraint Method.

        Ignored when ``directions`` has a single element.
    interface : str, optional
        Solver interface / modeling layer.  One of:

        - ``"auto"`` (default) “ automatically select based on problem type.
        - ``"highs"`` — HiGHS (LP, MILP, QP, MIQP).
        - ``"ortools"`` — Google OR-Tools.
        - ``"scip"`` — SCIP (MILP, MINLP, NLP).
        - ``"uno"`` — Uno (NLP).
        - ``"bonmin"`` / ``"couenne"`` — MINLP.
        - ``"cvxpy"`` — CVXPY.
        - ``"linopy"`` — Linopy (xarray-based).
        - ``"pyomo"`` — Pyomo.
        - ``"gurobi"`` — Gurobi.
        - ``"cplex"`` — IBM CPLEX.
        - ``"xpress"`` — FICO Xpress.
        - ``"gamspy"`` — GAMSPY.
        - ``"rsome_dro"`` / ``"rsome_ro"`` — RSOME robust/DRO.
        - ``"mealpy"`` / ``"niapy"`` / ``"pygad"`` — Metaheuristics.
        - ``"gekko"`` — GEKKO.
        - ``"jump"`` — JuMP (Julia).
        - ``"ortools_cp"`` / ``"cplex_cp"`` — Constraint programming.
        - ``"hexaly"`` — Hexaly.
        - ``"casadi"`` — CasADi.
        - ``"pymprog"`` — pymprog.
        - ``"picos"`` — PICOS.
        - ``"copt"`` — COPT.
        - ``"mathopt"`` — MathOpt.
        - ``"cylp"`` — CyLP.
    directions : list of str, optional
        Optimization direction for each objective.  Each element is ``"min"``
        or ``"max"``.  Required for all methods except ``"madm"`` and
        ``"constraint"``.  Examples: ``["min"]``, ``["max", "min"]``.
    solver : str, optional
        Underlying solver engine (e.g. ``"highs"``, ``"scip"``, ``"gurobi"``).
        When ``interface="auto"`` the solver is selected automatically.
        You only need to set this to override the default solver for an
        interface (e.g. ``interface="ortools"``, ``solver="scip"``).
    program : str, optional
        Shorthand that sets both ``interface`` and ``solver`` at once.
        One of: ``"lp"``, ``"ip"``, ``"milp"``, ``"qp"``, ``"iqp"``,
        ``"miqp"``, ``"nlp"``, ``"minlp"``.
    dataset : data_toolkit, optional
        A ``data_toolkit`` instance whose stored parameters are used as
        dynamic inputs to the model function.  When provided, the model
        function receives the dataset so that ``dt.store()`` values are
        accessible for sensitivity analysis and reporting.
    key_params : list of str, optional
        Names of parameters (stored in ``dataset``) to vary in the
        sensitivity analysis.  Ignored when ``sensitivity`` is given.
    scenarios : list of list, optional
        Scenario values for each parameter in ``key_params``.
        ``scenarios[i]`` is a list of values for ``key_params[i]``.
        Ignored when ``sensitivity`` is given.
    sensitivity : dict, optional
        Convenience shorthand for ``key_params`` and ``scenarios``.
        Maps parameter names to lists of scenario values::

            sensitivity={"a": [1, 2, 3], "b": [10, 20]}

        is equivalent to::

            key_params=["a", "b"], scenarios=[[1, 2, 3], [10, 20]]

    control_scenario : int, optional
        Index of the control (base) scenario for sensitivity similarity
        computation (default ``0``).
    constraint_ids : list of str, optional
        If given, only constraints with these labels are included in the
        formulation.  Useful for Benders decomposition and constraint
        filtering.
    benchmark : str or list, optional
        Run algorithmic benchmarks after solving.  Set to ``"all"`` to
        benchmark against all available algorithms, or pass a list of
        algorithm names (e.g. ``["scip", "ortools"]``).
    decoder : callable, optional
        Custom decoder function applied to heuristic solutions.
    repeat : int, optional
        Number of times to repeat heuristic runs (default ``1``).
    verbose : bool, optional
        If ``True``, suppress progress spinners and print solver output
        directly (default ``False``).
    progress : bool, optional
        If ``True``, show a live progress bar during heuristic optimization
        (default ``False``).
    should_run : bool, optional
        If ``False``, build the model but do *not* solve it (default
        ``True``).  Useful for inspecting the formulation before solving.
    memorize : bool, optional
        If ``True`` (default), store all ``dt.store()`` values in the
        dataset's internal dictionary so they are accessible after the
        model function returns.
    report : bool, optional
        If ``True``, automatically print the full report after solving
        (default ``False``).
    options : dict, optional
        Solver-specific options passed through to the underlying solver.
        Common keys: ``"penalty_coefficient"`` (for heuristic methods).
    email : str, optional
        Email address for solver license verification (e.g. Gurobi, CPLEX).
    time_limit : float, optional
        Solver time limit in seconds.
    cpu_threads : int, optional
        Number of CPU threads the solver may use.
    absolute_gap : float, optional
        Absolute optimality gap tolerance for MIP solvers.
    relative_gap : float, optional
        Relative optimality gap tolerance for MIP solvers.
    track_history : bool, optional
        If ``True``, record per-epoch objective and trajectory history
        during heuristic optimization (default ``False``).
    debug : bool, optional
        If ``True``, run constraint feasibility checking and diagnostic
        output after solving (default ``False``).
    *args, **kwargs
        Additional positional and keyword arguments forwarded to the
        ``environment`` function.

    Returns
    -------
    search
        A ``search`` instance.  Call ``.report()`` for a formatted summary,
        or inspect ``.solutions``, ``.get_obj()``, ``.get_numpy_var(name)``,
        and ``.healthy()`` directly.

    Raises
    ------
    RuntimeError
        If ``directions`` is ``None`` for a non-MADM method, or if the
        problem type is incompatible with the selected solver.
    ValueError
        If ``key_params`` and ``scenarios`` have mismatched lengths.

    Examples
    --------
    **Linear program (single objective)**

    >>> import feloopy as flp
    >>> def my_model(m):
    ...     x = m.pvar(name="x", bound=[0, 10])
    ...     y = m.pvar(name="y", bound=[0, 10])
    ...     m.con(x + y <= 12, name="budget")
    ...     m.obj(3*x + 5*y)
    ...     return m
    >>> r = flp.search(my_model, directions=["max"])
    >>> r.report()

    **Integer program**

    >>> def ip_model(m):
    ...     x = m.bvar(name="x")          # binary (0 or 1)
    ...     y = m.ivar(name="y", bound=[0, 5])  # integer 0..5
    ...     m.con(x + y <= 3, name="c1")
    ...     m.obj(2*x + 3*y)
    ...     return m
    >>> r = flp.search(ip_model, directions=["max"])

    **Multi-objective**

    >>> def mo_model(m):
    ...     x = m.pvar(name="x", bound=[0, 10])
    ...     m.obj(x, direction="max")        # maximize x
    ...     m.obj(10 - x, direction="min")   # minimize 10-x
    ...     return m
    >>> r = flp.search(mo_model, directions=["max", "min"])

    **Sensitivity analysis**

    >>> dt = flp.data_toolkit(key=0)
    >>> a = dt.store("a", 10)
    >>> def sen_model(m):
    ...     x = m.bvar(name="x")
    ...     m.obj(a * x, direction="max")
    ...     m.minimize(x)
    ...     return m
    >>> r = flp.search(sen_model, dataset=dt,
    ...               sensitivity={"a": [5, 10, 15, 20]})
    >>> r.report_sensitivity()

    **Using dataset for dynamic parameters**

    >>> dt = flp.data_toolkit(key=0)
    >>> demand = dt.store("demand", [10, 20, 15])
    >>> def inventory_model(m):
    ...     x = m.pvar(name="x", dim=[3], bound=[0, 50])
    ...     m.con(x <= demand, name="supply")
    ...     m.obj(sum(x))
    ...     return m
    >>> r = flp.search(inventory_model, dataset=dt, directions=["min"])

    **Inspecting results without printing**

    >>> r = flp.search(my_model, directions=["max"], verbose=True)
    >>> r.solutions          # dict of variable values
    >>> r.get_obj()          # objective value
    >>> r.get_numpy_var("x") # variable as NumPy array
    >>> r.healthy()          # True if feasible

    **Deferred solving**

    >>> r = flp.search(my_model, directions=["max"], should_run=False)
    >>> # inspect r.em (the model object) without solving
    >>> r.run(verbose=True)  # solve manually when ready
    """

    def __init__(
        self,
        environment=None,
        name="Untitled",
        method=None,
        approach = "nwsm",
        interface="auto",
        directions=None,
        solver=None,
        program=None,
        dataset = None,
        key_params = [],
        key_vars = [],
        scenarios = [],
        constraint_ids=None,
        benchmark=None,
        decoder=None,
        repeat=1,
        verbose=False,
        progress=False,
        should_run=True,
        memorize=True,
        report=False,
        control_scenario=0,
        sensitivity=None,
        options={},
        email=None,
        time_limit=None,
        cpu_threads=None,
        absolute_gap=None,
        relative_gap=None,
        track_history=None,
        debug=False,
        no_scenarios=None,
        cache_key=None,
        linearize=None,
        boost=False,
        decompose=False,
        init=None,
        *args, **kwargs
    ):

        if "auto_linearize" in kwargs:
            linearize = kwargs.pop("auto_linearize")

        if method is None and solver is not None:
            from feloopy.helpers.registries import (
                HEURISTIC_ALGORITHMS, EXACT_ALGORITHMS,
                UNCERTAINTY_ALGORITHMS, CONSTRAINT_ALGORITHMS,
                WEIGHTING_ALGORITHMS, RANKING_ALGORITHMS, SPECIAL_ALGORITHMS,
            )
            _solver_method_lookup = {}
            for _reg, _mth in [
                (HEURISTIC_ALGORITHMS, "heuristic"),
                (EXACT_ALGORITHMS, "exact"),
                (UNCERTAINTY_ALGORITHMS, "uncertain"),
                (CONSTRAINT_ALGORITHMS, "constraint"),
            ]:
                for _pair in _reg:
                    _solver_method_lookup[_pair[1]] = _mth
            for _reg in [WEIGHTING_ALGORITHMS, RANKING_ALGORITHMS, SPECIAL_ALGORITHMS]:
                for _pair in _reg:
                    _solver_method_lookup[_pair[0]] = "madm"
            if solver in _solver_method_lookup:
                method = _solver_method_lookup[solver]
            else:
                from feloopy.helpers.solver_params import _HEURISTIC_IFACES
                method = "heuristic" if str(interface).lower() in _HEURISTIC_IFACES else "exact"

        if method is not None and method != "madm" and directions is not None:
            validate_existence(
                    label="directions", 
                    input_value=directions, 
                    condition=True if method!="constraint" else False)

        if method == "madm" and interface == "auto":
            interface = "pydecision"

        _MADM_METHODS = [
            'add_pcm', 'add_cpcm', 'add_dm', 'add_wv', 'add_fwv',
            'add_bocv', 'add_owcv', 'add_fbocv', 'add_fowcv',
            'add_cim', 'add_fcim', 'add_fim', 'add_fdm',
            'add_profiles', 'add_cr', 'add_cp', 'add_apcm', 'add_fapcm',
            'add_uf', 'add_bt', 'add_lbt', 'add_ubt',
            'add_qt', 'add_pt', 'add_st', 'add_vt',
            'add_grades', 'add_con_max_criteria', 'add_con_cost_budget',
            'add_con_forbid_selection', 'add_ppfcpcm', 'add_fcpcm',
            'add_er', 'add_erc', 'add_era', 'add_wv_lb', 'add_wv_ub',
        ]

        if method is None and environment is not None:
            try:
                import inspect as _auto_inspect
                _src = _auto_inspect.getsource(environment)
                _has_madm = any(f'.{m}(' in _src or f' {m}(' in _src for m in _MADM_METHODS)
                if _has_madm:
                    method = "madm"
                    if interface == "auto":
                        interface = "pydecision"
                    if solver is None:
                        solver = "auto"
            except Exception:
                pass

        if method is None:
            method = "exact"

        if program is not None:
            program = program.lower()
            program_map = {
                'lp': ('highs', 'highs'), 'ip': ('highs', 'highs'), 'milp': ('highs', 'highs'),
                'qp': ('highs', 'highs'), 'iqp': ('highs', 'highs'), 'miqp': ('highs', 'highs'),
                'nlp': ('uno', 'uno'), 'minlp': ('scip', 'scip'),
            }
            if program in program_map:
                interface, solver = program_map[program]
        elif solver is None:
            if method == "madm":
                solver = "auto"
            else:
                if interface == 'auto':
                    interface = self._auto_select_interface()
                from .helpers.problem_detect import INTERFACE_DEFAULT_SOLVER
                solver = INTERFACE_DEFAULT_SOLVER.get(interface, 'highs')

        self.key_params = key_params
        self.key_vars = key_vars
        self.scenarios = scenarios
        if sensitivity is not None and isinstance(sensitivity, dict):
            self.key_params = list(sensitivity.keys())
            self.scenarios = [sensitivity[k] for k in self.key_params]
        if len(self.key_params) != 0 and len(self.scenarios) == 0 and dataset is not None:
            self.scenarios = self._generate_scenarios(dataset, self.key_params)
        self.sensitivity_descriptions = {}
        if dataset is not None and hasattr(dataset, 'descriptions'):
            self.sensitivity_descriptions = {k: dataset.descriptions.get(k, '') for k in self.key_params}
        self.email = email
        self.cpu_threads = cpu_threads
        self.time_limit = time_limit
        self.absolute_gap = absolute_gap
        self.relative_gap = relative_gap
        self.args = args
        self.kwargs = {k: v for k, v in kwargs.items() if k not in ['callback', 'obj_operators', 'uncertainty_set_constraints']}
        self.obj_operators = kwargs.get('obj_operators', [])
        self.uncertainty_set_constraints = kwargs.get('uncertainty_set_constraints', [])
        self.environment = environment
        if name == "Untitled" and environment is not None and hasattr(environment, '__name__'):
            self.name = environment.__name__
        else:
            self.name = name
        self.method = method
        self.approach = approach
        self.interface = interface
        self.directions = directions
        self.solver = solver
        self.program = program
        self.verbose = verbose
        self.should_run = should_run
        self.memorize = memorize
        self.control_scenario = control_scenario
        self.sensitivity_analyzed = False
        self.options = options
        self.should_benchmark = True if (type(benchmark)==str and benchmark=='all') or (type(benchmark)==list and len(benchmark)>=1) else False
        self.inputdata = dataset
        self.data = {}
        self.repeat = repeat
        self.progress = progress
        self.mgt = 0
        self.decoder = decoder
        self._decoder_args_spec = None
        if decoder is not None:
            import inspect as _inspect
            try:
                sig = _inspect.signature(decoder)
                self._decoder_args_spec = [p.name for p in sig.parameters.values()]
            except (ValueError, TypeError):
                self._decoder_args_spec = None
        if track_history is None and method == "heuristic":
            self.track_history = True
        elif track_history is None:
            self.track_history = False
        else:
            self.track_history = track_history
        self.constraint_ids = constraint_ids
        self.no_scenarios = no_scenarios
        self.debug = debug
        self.auto_linearize = linearize
        self.boost = boost
        self.decompose = decompose
        self.init = init

        self._search_kwargs = dict(
            environment=environment,
            name=name,
            method=method,
            approach=approach,
            interface=interface,
            directions=directions,
            solver=solver,
            program=program,
            dataset=dataset,
            key_params=key_params,
            key_vars=key_vars,
            scenarios=scenarios,
            constraint_ids=None,
            benchmark=benchmark,
            decoder=decoder,
            repeat=repeat,
            verbose=verbose,
            progress=progress,
            should_run=should_run,
            memorize=memorize,
            report=False,
            control_scenario=control_scenario,
            options=options,
            email=email,
            time_limit=time_limit,
            cpu_threads=cpu_threads,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            track_history=track_history,
            debug=debug,
            auto_linearize=linearize,
            boost=boost,
            init=init,
            *args, **kwargs
        )

        if self.method!= "madm":
            if self.directions is not None:
                self.number_of_objectives = len(self.directions)
            else:
                self.number_of_objectives = 0

        auto_key = _make_cache_key(environment, args, kwargs, interface, solver, method)
        self._cache_key = cache_key if cache_key is not None else auto_key
        _use_cache = self.method not in ("heuristic", "madm", "sequential")

        cached = _LINEARIZATION_CACHE.get(self._cache_key) if _use_cache else None

        start = timeit.default_timer()
        self.create_env(environment, verbose=True if self.should_benchmark else self.verbose)
        end = timeit.default_timer()
        self.mgt+=end-start

        if cached is not None and 'features' in cached:
            self.mgt = cached.get('mgt', 0)
            if self.directions is None and self.method != "madm":
                self.directions = self.em.features.get('directions', [])

        func_code = b''
        func_name = ''
        func_consts = ()
        if environment is not None and hasattr(environment, '__code__'):
            func_code = environment.__code__.co_code
            func_name = environment.__name__
            func_consts = environment.__code__.co_consts

        features_to_cache = {}
        features = getattr(self.em, 'features', {})
        for k, v in features.items():
            if k in ('variables', 'constraints', 'objectives'):
                features_to_cache[k] = v
            elif k.startswith('_') or k in ('interface_name', 'method', 'name'):
                features_to_cache[k] = v
            elif isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
                features_to_cache[k] = v

        if _use_cache:
            features_to_cache, _ = _sanitize_features(features_to_cache)
            _LINEARIZATION_CACHE[self._cache_key] = {
                'mgt': self.mgt,
                'features': features_to_cache,
                'func_code': func_code,
                'func_name': func_name,
                'func_consts': func_consts,
                'method': method,
                'timestamp': time.time(),
            }

        if self.method != "madm":
            if self.em is not None and hasattr(self.em, 'features'):
                _user_dirs = self.directions
                _feat_dirs = self.em.features.get('directions', [])
                if _user_dirs is not None:
                    for _i in range(min(len(_user_dirs), len(_feat_dirs))):
                        if _feat_dirs[_i] is None and _user_dirs[_i] is not None:
                            _feat_dirs[_i] = _user_dirs[_i]
                    self.directions = _feat_dirs
                else:
                    self.directions = _feat_dirs
            self.number_of_objectives = len(self.directions) if self.directions else 0

        self.dataset_size = None
        if self.inputdata:
            if type(self.inputdata)!=dict:
                self.dataset_size = self.inputdata.size
                self.big_m_value = self.inputdata.possible_big_m
                self.epsilon_value = self.inputdata.possible_epsilon

        if self.should_run:
            if self.should_benchmark: 
                self.benchmark_results = self.benchmark(algorithms=benchmark, repeat=self.repeat)
            
            #run_with_progress(self.run, show_log= self.progress, verbose=self.verbose)
            start = timeit.default_timer()
            self.run(verbose=True if self.should_benchmark else self.verbose, show_solver_log=False if self.should_benchmark else None)
            end = timeit.default_timer()
            self.mgt+=end-start

            if self.solver is None and hasattr(self, 'em') and hasattr(self.em, 'features'):
                self.solver = self.em.features.get('solver_name', self.solver)
        
        if len(self.key_params)!=0 and len(self.scenarios)!=0:
            start_progress(message="Analyzing...", spinner="dots",
                           color="cyan", show_elapsed=True)
            try:
                self.sensitivity(dataset, self.key_params, self.scenarios,
                                 environment, control_scenario)
                end_progress(success_message="Analyzed",
                             show_elapsed=False)
            except Exception:
                end_progress(success_message=None,
                             failure_message="Analysis failed",
                             success=False, show_elapsed=False)
                raise

        if report:
            self.report()

        if self.track_history:
            self.best_epoch_objective = []
            self.best_overall_objective = []
            self.best_epoch_trajectory = []
            self.best_overall_trajectory = []

    def _build_linearized_capture(self):
        """Run user code on a temporary auto-linearized model and extract
        the linearized LP as an ``_AutomaticCaptureModel``.
        """
        temp_em = model(
            method='exact', interface='highs', auto_linearize=True)
        self.environment(temp_em, *self.args, **self.kwargs)

        n_col = temp_em.model.getNumCol()

        col_to_key = {}
        for (kind, name), native_val in temp_em.features.get('variables', {}).items():
            if isinstance(native_val, dict):
                for idx, hv in native_val.items():
                    col_to_key[hv.index] = (kind, name, idx)
            else:
                col_to_key[native_val.index] = (kind, name, None)

        col_lower = {}
        col_upper = {}
        for col_j in range(n_col):
            _, cost_j, lb_j, ub_j, _ = temp_em.model.getCol(col_j)
            col_lower[col_j] = lb_j
            col_upper[col_j] = ub_j

        captured = _AutomaticCaptureModel()
        temp_var_dim = temp_em.features.get('variable_dim', {})

        seen_vars = {}
        for col_j in range(n_col):
            key = col_to_key.get(col_j)
            if key is None:
                key = ('fvar', '_aux_%d' % col_j, None)
            kind, name, idx = key
            lb = col_lower.get(col_j, None)
            ub = col_upper.get(col_j, None)
            if lb is not None and lb <= -1e20:
                lb = None
            if ub is not None and ub >= 1e20:
                ub = None

            var_key = (kind, name)
            if var_key not in seen_vars:
                dim = temp_var_dim.get(name, 0)
                seen_vars[var_key] = {'dim': dim, 'bound': [lb, ub], 'values': _AutomaticVariableArray()}
            var_idx = idx if idx is not None else None
            auto_var = _AutomaticLinearVariable((kind, name, var_idx), captured)
            seen_vars[var_key]['values'][var_idx] = auto_var

        captured.variables = seen_vars

        for constraint in temp_em.features.get('constraints', []):
            try:
                idxs, vals = constraint.unique_elements()
                row_lb, row_ub = constraint.bounds
            except Exception:
                continue

            terms = {}
            for col_j, coeff in zip(idxs, vals):
                key = col_to_key.get(int(col_j))
                if key is None:
                    key = ('fvar', '_aux_%d' % int(col_j), None)
                terms[key] = terms.get(key, 0.0) + float(coeff)

            lb_finite = row_lb > -1e20
            ub_finite = row_ub < 1e20

            if lb_finite and ub_finite:
                captured.constraints.append(
                    (_AutomaticLinearConstraint(
                        _AutomaticLinearExpression(-float(row_lb), dict(terms)), '>='), None))
                captured.constraints.append(
                    (_AutomaticLinearConstraint(
                        _AutomaticLinearExpression(-float(row_ub), dict(terms)), '<='), None))
            elif ub_finite:
                captured.constraints.append(
                    (_AutomaticLinearConstraint(
                        _AutomaticLinearExpression(-float(row_ub), dict(terms)), '<='), None))
            elif lb_finite:
                captured.constraints.append(
                    (_AutomaticLinearConstraint(
                        _AutomaticLinearExpression(-float(row_lb), dict(terms)), '>='), None))

        obj_expr = temp_em.features['objectives'][0]
        direction = (temp_em.features.get('directions') or ['min'])[0]
        obj_terms = {}
        try:
            for col_j, coeff in zip(obj_expr.idxs, obj_expr.vals):
                key = col_to_key.get(int(col_j))
                if key is None:
                    key = ('fvar', '_aux_%d' % int(col_j), None)
                obj_terms[key] = float(coeff)
        except Exception:
            pass
        obj_const = getattr(obj_expr, 'constant', None) or 0.0
        captured.objectives.append(
            (_AutomaticLinearExpression(float(obj_const), obj_terms),
             direction or (self.directions[0] if self.directions else 'min'), None))

        return captured

    def _run_automatic_benders(self, show_solver_log):
        """Solve via automatic Benders decomposition.

        Delegates to ``benders.automatic.run_automatic_benders``.
        """
        from .algorithms.exact.benders.automatic import run_automatic_benders
        run_automatic_benders(self, show_solver_log)

    def _select_boost_method(self):
        """Auto-select decomposition method based on problem structure.

        Uses a scoring system that evaluates multiple structural features
        to determine which decomposition technique will provide the most
        computational advantage for the given problem.
        """
        _valid = {'benders', 'lagrangian', 'cg', 'ccg', 'dcg', 'dw', 'dw2', 'branching'}
        if isinstance(self.boost, str):
            _m = self.boost.lower().strip()
            if _m not in _valid:
                raise ValueError(
                    f"boost={self.boost!r} is not valid. "
                    f"Choose from: {', '.join(sorted(_valid))} or True for auto-select")
            return _m
        if self.boost is True:
            _cg_variant = (self.options or {}).get('cg_variant')
            if _cg_variant and _cg_variant.lower().strip() in _valid:
                return _cg_variant.lower().strip()

        n_bvar = 0
        n_ivar = 0
        n_fvar = 0
        n_pvar = 0
        n_rvar = 0
        n_total = 0
        has_large_indexed = False
        max_dim = 0
        indexed_dims = []

        col_to_kind = {}
        variable_dim = self.em.features.get('variable_dim', {})
        for (kind, name) in self.em.features.get('variables', {}):
            dim = variable_dim.get(name, 0)
            count = 1
            effective_dim = None
            if isinstance(dim, (int, float)) and dim > 0:
                try:
                    dims = fix_dims([range(int(dim))] if isinstance(dim, (int, float)) else dim)
                    count = 1
                    for d in dims:
                        count *= len(d)
                except Exception:
                    count = int(dim)
                effective_dim = int(dim)
            elif isinstance(dim, (list, tuple)) and dim:
                try:
                    dims = fix_dims(dim)
                    count = 1
                    for d in dims:
                        count *= len(d)
                except Exception:
                    count = 1
                    for d in dim:
                        if hasattr(d, '__len__'):
                            count *= len(d)
                effective_dim = max((len(d) if hasattr(d, '__len__') else int(d) for d in dim), default=0)

            var_obj = self.em.features['variables'][(kind, name)]
            if isinstance(var_obj, dict):
                for idx, v in var_obj.items():
                    if hasattr(v, 'index'):
                        col_to_kind[v.index] = kind
                    elif hasattr(v, '_col'):
                        col_to_kind[v._col] = kind
            elif hasattr(var_obj, 'index'):
                col_to_kind[var_obj.index] = kind
            elif hasattr(var_obj, '_col'):
                col_to_kind[var_obj._col] = kind

            if effective_dim is not None and effective_dim >= 4:
                has_large_indexed = True
                max_dim = max(max_dim, effective_dim)
                indexed_dims.append(effective_dim)

            if kind == 'bvar':
                n_bvar += count
            elif kind == 'ivar':
                n_ivar += count
            elif kind == 'fvar':
                n_fvar += count
            elif kind == 'pvar':
                n_pvar += count
            elif kind == 'rvar':
                n_rvar += count
            n_total += count

        n_total_int = n_bvar + n_ivar
        n_total_cont = n_fvar + n_pvar + n_rvar
        int_ratio = n_total_int / n_total if n_total > 0 else 0.0
        cont_ratio = n_total_cont / n_total if n_total > 0 else 0.0

        n_constraints = len(self.em.features.get('constraints', []))
        constraint_density = n_constraints / n_total if n_total > 0 else 0.0

        n_linking = 0
        n_integer_only = 0
        n_continuous_only = 0
        linking_vars_per_constraint = []
        _dw_terms = []       # column indices per constraint (dw graph)
        _dw_terms_ok = True  # False when a key cannot map to a column
        _features_vars = self.em.features.get('variables', {})

        def _dw_col(key):
            """Map a terms key ``(kind, name, index)`` to its column index.

            Constraints are stored either as sparse column indices or as
            expression term keys; both spellings have to land in the same
            key space before connected components mean anything.
            """
            if isinstance(key, int) and not isinstance(key, bool):
                return key
            if not (isinstance(key, tuple) and len(key) >= 2):
                return None
            var_obj = _features_vars.get((key[0], key[1]))
            if isinstance(var_obj, dict):
                idx = key[2] if len(key) >= 3 else None
                try:
                    var_obj = var_obj.get(idx)
                except TypeError:
                    return None
            elif len(key) >= 3 and key[2] is not None:
                var_obj = None
            if var_obj is None:
                return None
            if hasattr(var_obj, 'index'):
                return var_obj.index
            if hasattr(var_obj, '_col'):
                return var_obj._col
            return None

        for constraint in self.em.features.get('constraints', []):
            var_types = set()
            cobj = constraint[0] if isinstance(constraint, tuple) else constraint
            if hasattr(cobj, 'expression') and hasattr(cobj.expression, 'terms'):
                term_keys = list(cobj.expression.terms)
                for key in term_keys:
                    if isinstance(key, tuple) and len(key) >= 1:
                        var_types.add(key[0])
                if _dw_terms_ok:
                    _cols = [_dw_col(key) for key in term_keys]
                    if None in _cols:
                        _dw_terms_ok = False
                    else:
                        _dw_terms.append(_cols)
            elif hasattr(cobj, 'idxs'):
                for col_j in cobj.idxs:
                    kind = col_to_kind.get(int(col_j))
                    if kind:
                        var_types.add(kind)
                if _dw_terms_ok:
                    _dw_terms.append([int(cj) for cj in cobj.idxs])
            elif hasattr(cobj, 'expression') and hasattr(cobj.expression, 'idxs'):
                for col_j in cobj.expression.idxs:
                    kind = col_to_kind.get(int(col_j))
                    if kind:
                        var_types.add(kind)
                if _dw_terms_ok:
                    _dw_terms.append([int(cj) for cj in cobj.expression.idxs])

            has_int = any(t in ('ivar', 'bvar') for t in var_types)
            has_cont = any(t in ('fvar', 'pvar', 'rvar') for t in var_types)
            n_linking_vars = len([t for t in var_types if t in ('ivar', 'bvar')]) if has_cont else 0

            if has_int and has_cont:
                n_linking += 1
                linking_vars_per_constraint.append(n_linking_vars)
            elif has_int and not has_cont:
                n_integer_only += 1
            elif has_cont and not has_int:
                n_continuous_only += 1

        linking_ratio = n_linking / n_constraints if n_constraints > 0 else 0.0
        avg_linking_vars = (sum(linking_vars_per_constraint) / len(linking_vars_per_constraint)
                           if linking_vars_per_constraint else 0.0)
        integer_only_ratio = n_integer_only / n_constraints if n_constraints > 0 else 0.0

        avg_indexed_dim = (sum(indexed_dims) / len(indexed_dims)
                          if indexed_dims else 0.0)
        total_indexed_vars = sum(1 for d in indexed_dims if d >= 4)

        scores = {
            'benders': 0.0,
            'lagrangian': 0.0,
            'cg': 0.0,
            'ccg': 0.0,
            'dcg': 0.0,
            'dw': 0.0,
            'dw2': 0.0,
            'branching': 0.0,
        }

        if n_total == 0:
            return 'benders'

        # For very small problems, decomposition overhead exceeds direct solve.
        # HiGHS solves <100 variables in microseconds.
        _SMALL_THRESHOLD = 100
        if n_total <= _SMALL_THRESHOLD:
            return 'direct'

        n_dw_blocks = 0
        if _dw_terms_ok and _dw_terms:
            _dw_adj = {}
            for _vl in _dw_terms:
                if len(_vl) > 1:
                    _head = _vl[0]
                    _lst = _dw_adj.setdefault(_head, set())
                    for _v in _vl[1:]:
                        _lst.add(_v)
                        _dw_adj.setdefault(_v, set()).add(_head)
                elif _vl:
                    _dw_adj.setdefault(_vl[0], set())
            _dw_seen = set()
            for _v0 in _dw_adj:
                if _v0 in _dw_seen:
                    continue
                n_dw_blocks += 1
                _stack = [_v0]
                while _stack:
                    _u = _stack.pop()
                    if _u in _dw_seen:
                        continue
                    _dw_seen.add(_u)
                    _stack.extend(w for w in _dw_adj[_u] if w not in _dw_seen)

        # Benders: good when few integer variables, many continuous,
        # and clear first-stage/second-stage structure exists.
        if n_total_int > 0 and n_total_int <= 3 and n_continuous_only > 0:
            scores['benders'] += 12.0
        elif n_total_int > 0 and n_total_int <= 10 and n_continuous_only > 0:
            scores['benders'] += 8.0
        elif n_total_int > 0 and n_total_int <= 50 and n_continuous_only > 0:
            scores['benders'] += 4.0
        elif n_total_int > 50 and n_continuous_only > 0:
            scores['benders'] += 1.0

        # Benders needs continuous variables for a meaningful subproblem
        if n_total_cont > 0 and n_total_int > 0:
            if n_total_cont > 3 * n_total_int:
                scores['benders'] += 6.0
            if n_integer_only > n_linking and n_integer_only > n_continuous_only:
                scores['benders'] += 3.0

        # Benders bonus: many continuous-only constraints indicate a rich
        # subproblem that benefits from LP-based decomposition
        if n_continuous_only > 0 and n_linking > 0:
            subproblem_richness = n_continuous_only / max(n_linking, 1)
            if subproblem_richness >= 2.0:
                scores['benders'] += 4.0
            elif subproblem_richness >= 1.0:
                scores['benders'] += 2.0

        # Benders bonus: high linking constraint ratio means fixing
        # integer vars strongly affects the subproblem
        if n_linking > 0 and n_constraints > 0:
            linking_ratio_eff = n_linking / n_constraints
            if linking_ratio_eff >= 0.5:
                scores['benders'] += 3.0

        # Lagrangian: good when many coupling constraints exist that,
        # when relaxed, decompose the problem into easier subproblems.
        _has_linking = n_linking > 0
        if n_total_int >= 5 and _has_linking and linking_ratio >= 0.2:
            scores['lagrangian'] += 6.0
        if int_ratio >= 0.4 and _has_linking and linking_ratio >= 0.3:
            scores['lagrangian'] += 4.0
        if int_ratio >= 0.6 and n_total_int >= 10 and _has_linking:
            scores['lagrangian'] += 3.0
        if linking_ratio >= 0.3 and int_ratio >= 0.3:
            scores['lagrangian'] += 3.0

        # CG: only useful for problems with large indexed dimensions
        # that suggest many potential columns (cutting stock, routing).
        if has_large_indexed and cont_ratio >= 0.5 and avg_indexed_dim >= 4:
            scores['cg'] += 6.0
        if has_large_indexed and total_indexed_vars >= 3 and avg_indexed_dim >= 6:
            scores['cg'] += 4.0
        if avg_indexed_dim >= 8:
            scores['cg'] += 3.0

        # CCG: only useful when there are explicit scenario-like structures
        # (indexed dimensions suggesting multiple scenarios/cases).
        if has_large_indexed and _has_linking and avg_indexed_dim >= 4:
            scores['ccg'] += 7.0
        if has_large_indexed and total_indexed_vars >= 3 and _has_linking:
            scores['ccg'] += 5.0
        if avg_indexed_dim >= 8 and _has_linking:
            scores['ccg'] += 3.0

        # DCG: only useful for large problems with independent subsystems
        if int_ratio >= 0.5 and n_total > 50 and n_total_cont > 10:
            scores['dcg'] += 5.0
        if int_ratio >= 0.6 and n_total > 100 and n_total_cont > 20:
            scores['dcg'] += 4.0
        if n_total_int >= 30 and n_total_cont >= 30:
            scores['dcg'] += 3.0

        # DW / DW2: the constraint-variable graph already splits into
        # many independent blocks — Dantzig-Wolfe prices each block as
        # columns and couples them in a small master (single vs
        # per-block convexity).  Prefer dw: same result, smaller master.
        if n_dw_blocks >= 4:
            scores['dw'] += 6.0
            scores['dw2'] += 5.0
        if n_dw_blocks >= 8 and n_total > 300:
            scores['dw'] += 3.0
            scores['dw2'] += 3.0

        # Branching: good when many integer variables with moderate structure
        # that doesn't clearly favor another decomposition method.
        if n_total_int >= 10 and int_ratio >= 0.3:
            scores['branching'] += 4.0
        if n_total_int >= 20 and int_ratio >= 0.5:
            scores['branching'] += 5.0
        if n_total_int >= 50 and constraint_density >= 2.0:
            scores['branching'] += 6.0
        if n_total_int >= 10 and n_total_cont == 0:
            scores['branching'] += 8.0

        best_method = max(scores, key=scores.get)
        best_score = scores[best_method]

        # Require a minimum score to justify decomposition overhead.
        # A score of 3.0 indicates moderate structural benefit;
        # below that, direct solve is typically faster.
        if best_score < 3.0:
            return 'direct'

        return best_method

    def _run_automatic_lagrangian(self, show_solver_log):
        """Solve via automatic Lagrangian relaxation.

        Automatically selects step-size strategy based on problem structure:
        - Adaptive step size for general problems
        - Polyak's step size when optimal value estimate is available
        - Volume algorithm for problems with good primal solutions
        """
        from .algorithms.exact.lagrangian import LagrangianRelaxation

        if self.auto_linearize:
            captured = self._build_linearized_capture()
        else:
            captured = _AutomaticCaptureModel()
            try:
                self.environment(captured, *self.args, **self.kwargs)
            except (TypeError, ValueError) as error:
                from .helpers.error import ConstantConstraintError
                if isinstance(error, ConstantConstraintError):
                    raise  # infeasible model, not an unsupported expression
                raise ValueError(
                    "boost='lagrangian' supports linear MILP expressions") from error

        _if = self.interface if isinstance(self.interface, str) else (
            self.interface[0] if isinstance(self.interface, (list, tuple)) else 'highs')
        _sv = self.solver if isinstance(self.solver, str) else (
            self.solver[0] if isinstance(self.solver, (list, tuple)) else 'highs')

        n_constraints = len(captured.constraints)
        n_variables = len(captured.variables)
        n_integer = sum(1 for (kind, _) in captured.variables if kind in ('ivar', 'bvar'))

        _INCREMENTAL_INTERFACES = {'highs', 'gurobi', 'cplex'}
        if _if in _INCREMENTAL_INTERFACES:
            relaxed_solver, multiplier_update, initial, relaxable = (
                LagrangianRelaxation.from_automatic_incremental(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                    step_size=self.options.get('lag_step_size', 1.0),
                ))
        else:
            relaxed_solver, multiplier_update, initial, relaxable = (
                LagrangianRelaxation.from_automatic(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                    step_size=self.options.get('lag_step_size', 1.0),
                ))

        obj_expression, obj_direction, obj_label = captured.objectives[0]
        obj_terms = dict(obj_expression.terms)
        obj_constant = obj_expression.constant

        def _resolve_scalar(val, index):
            """Resolve a variable value for a captured ``(kind, name, index)``
            key, tolerating dict / list / ndarray containers.
            """
            if index is None:
                if isinstance(val, dict):
                    return next(iter(val.values()), None)
                if isinstance(val, (int, float)):
                    return float(val)
                try:
                    arr = np.asarray(val)
                    if arr.ndim == 0:
                        return float(arr)
                    if arr.size == 1:
                        return float(arr.reshape(-1)[0])
                except (TypeError, ValueError):
                    return None
                return None
            if isinstance(val, dict):
                return val.get(index, None)
            if isinstance(val, (int, float)):
                return float(val)
            try:
                arr = val if isinstance(val, np.ndarray) else np.asarray(val)
                resolved = arr[index]
                if np.ndim(resolved) != 0:
                    return None
                return float(resolved)
            except (IndexError, TypeError, ValueError):
                return None

        def primal_upper_bound(solution):
            """Evaluate primal objective from relaxed solution for bound tracking."""
            if not isinstance(solution, dict):
                return None
            total = obj_constant
            for key, coefficient in obj_terms.items():
                kind, name, index = key
                val = solution.get(name, None)
                if val is None:
                    continue
                resolved = _resolve_scalar(val, index)
                if resolved is not None:
                    total = total + coefficient * resolved
            try:
                obj_val = float(total)
            except (TypeError, ValueError):
                return None
            for cobj, clabel in captured.constraints:
                cval = cobj.expression.constant
                for ckey, ccoeff in cobj.expression.terms.items():
                    ck, cname, cindex = ckey
                    cvar = solution.get(cname, None)
                    if cvar is None:
                        continue
                    resolved = _resolve_scalar(cvar, cindex)
                    if resolved is not None:
                        cval = cval + ccoeff * resolved
                try:
                    cval = float(cval)
                except (TypeError, ValueError):
                    continue
                sense = cobj.sense
                if sense == '<=' and cval > 1e-6:
                    return None
                elif sense == '>=' and cval < -1e-6:
                    return None
                elif sense == '==' and abs(cval) > 1e-6:
                    return None
            return obj_val

        direction = self.directions[0] if self.directions else 'min'

        step_strategy = self.options.get('lag_step_strategy', None)
        if step_strategy is None:
            if n_integer >= 20 and n_constraints >= 10:
                step_strategy = 'volume'
            elif n_integer >= 10:
                step_strategy = 'adaptive'
            else:
                step_strategy = 'adaptive'

        lr = LagrangianRelaxation(
            relaxed_solver=relaxed_solver,
            multiplier_update=multiplier_update,
            initial_multipliers=initial,
            upper_bound=primal_upper_bound,
            direction=direction,
            multiplier_senses={
                idx: captured.constraints[idx][0].sense
                for idx in relaxable},
        )
        self._lagrangian_result = lr.solve(
            max_iterations=self.options.get('max_iterations', 100),
            tolerance=self.options.get('tolerance', 1e-6),
            show_log=bool(show_solver_log),
            step_strategy=step_strategy,
            polyak_estimate=self.options.get('lag_polyak_estimate', None),
            volume_weight=self.options.get('lag_volume_weight', 0.5),
        )

    def _run_automatic_branching(self, show_solver_log):
        """Solve via automatic branching (BaB / BaC / BaP).

        Analyses model structure and selects the best branching variant.
        """
        from .algorithms.exact.branching.automatic import run_automatic_branching

        def _model_fn(m):
            self.environment(m, *self.args, **self.kwargs)
            return m

        if self.auto_linearize:
            captured = self._build_linearized_capture()
        else:
            captured = _AutomaticCaptureModel()
            try:
                self.environment(captured, *self.args, **self.kwargs)
            except (TypeError, ValueError) as error:
                from .helpers.error import ConstantConstraintError
                if isinstance(error, ConstantConstraintError):
                    raise  # infeasible model, not an unsupported expression
                raise ValueError(
                    "boost='branching' supports linear MILP expressions") from error

        result = run_automatic_branching(
            model_fn=_model_fn,
            directions=self.directions,
            interface=self.interface if isinstance(self.interface, str) else (
                self.interface[0] if isinstance(self.interface, (list, tuple)) else 'highs'),
            solver=self.solver if isinstance(self.solver, str) else (
                self.solver[0] if isinstance(self.solver, (list, tuple)) else 'highs'),
            captured=captured,
            em=self.em,
            show_log=show_solver_log,
            cpu_threads=self.cpu_threads,
            max_time=self.time_limit if self.time_limit else 60.0,
        )
        self._branching_result = result

    def _run_automatic_column_generation(self, show_solver_log, variant=None):
        """Solve via automatic column generation.

        Automatically configures dual stabilization and anti-cycling
        based on problem structure.
        """
        from .algorithms.exact.column_generation import ColumnGeneration

        if self.auto_linearize:
            captured = self._build_linearized_capture()
        else:
            captured = _AutomaticCaptureModel()
            try:
                self.environment(captured, *self.args, **self.kwargs)
            except (TypeError, ValueError) as error:
                from .helpers.error import ConstantConstraintError
                if isinstance(error, ConstantConstraintError):
                    raise  # infeasible model, not an unsupported expression
                raise ValueError(
                    "boost='cg' supports linear MILP expressions") from error

        _if = self.interface if isinstance(self.interface, str) else (
            self.interface[0] if isinstance(self.interface, (list, tuple)) else 'highs')
        _sv = self.solver if isinstance(self.solver, str) else (
            self.solver[0] if isinstance(self.solver, (list, tuple)) else 'highs')

        n_constraints = len(captured.constraints)
        n_variables = len(captured.variables)

        use_stabilization = self.options.get('cg_stabilization', True)
        if use_stabilization is None:
            use_stabilization = n_constraints > 5

        _INCREMENTAL_INTERFACES = {'highs', 'gurobi', 'cplex', 'copt'}
        use_incremental = _if in _INCREMENTAL_INTERFACES

        if variant == 'dw':
            master_builder, modified_master_builder, pricing_oracle, initial_solutions = (
                ColumnGeneration.from_automatic_dw(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                    linking_constraints=self.options.get('linking_constraints'),
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                master_solver=master_builder,
                pricing_oracle=pricing_oracle,
                modified_master_solver=modified_master_builder,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=False,
                initial_columns=initial_solutions,
            )
        elif variant == 'dw2':
            master_builder, modified_master_builder, pricing_oracle, initial_solutions = (
                ColumnGeneration.from_automatic_dw2(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                    linking_constraints=self.options.get('linking_constraints'),
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                master_solver=master_builder,
                pricing_oracle=pricing_oracle,
                modified_master_solver=modified_master_builder,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=False,
                initial_columns=initial_solutions,
            )
        elif variant == 'dcg':
            master_builder, master_pricing, sub_pricing = (
                ColumnGeneration.from_automatic_dcg(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    sub_interface=_if, sub_solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                master_solver=master_builder,
                pricing_oracle=master_pricing,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=use_stabilization,
                sub_pricing=sub_pricing,
            )
            self._dcg_master_result = self._cg_result
            self._dcg_sub_result = self._cg_result
        elif variant == 'ccg':
            master_builder, pricing_oracle = (
                ColumnGeneration.from_automatic_ccg(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                master_solver=master_builder,
                pricing_oracle=pricing_oracle,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=use_stabilization,
            )
            self._ccg_result = self._cg_result
        elif use_incremental:
            master_inc, add_column_fn, pricing_oracle = (
                ColumnGeneration.from_automatic_incremental(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                pricing_oracle=pricing_oracle,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=use_stabilization,
                incremental_master=master_inc,
                add_column_fn=add_column_fn,
            )
        else:
            master_builder, pricing_oracle = (
                ColumnGeneration.from_automatic(
                    captured=captured, em=self.em,
                    interface=_if, solver=_sv,
                    directions=self.directions, method=self.method,
                    time_limit=self.time_limit, cpu_threads=self.cpu_threads,
                    absolute_gap=self.absolute_gap, relative_gap=self.relative_gap,
                ))
            cg = ColumnGeneration()
            self._cg_result = cg.solve(
                master_solver=master_builder,
                pricing_oracle=pricing_oracle,
                max_iterations=self.options.get('max_iterations', 100),
                tolerance=self.options.get('tolerance', 1e-6),
                show_log=bool(show_solver_log),
                use_stabilization=use_stabilization,
            )

    def create_env(self, environment, verbose):

        _show_progress = (not verbose) and self.method != "sequential"
        if _show_progress:
            start_progress(message="Generating...", spinner="dots")
        try:
            return self._create_env_impl(environment, verbose)
        except BaseException:
            # stop the spinner on failure too, otherwise its thread keeps
            # clearing/spinning forever in notebooks and hides the error
            if _show_progress:
                end_progress(success=False, failure_message="Generation failed")
            raise

    def _create_env_impl(self, environment, verbose):

        _pc = self.options.get("penalty_coefficient",0)
        self.penalty_coefficient = 0 if _pc is None else _pc
    
        if self.method in ["exact", "convex", "constraint", "uncertain"]:
            def _try_generate(iface):
                em = model(method=self.method,name=self.name,interface=iface,constraint_ids=self.constraint_ids,no_scenarios=self.no_scenarios,auto_linearize=self.auto_linearize)
                # batch-removal analysis: con() drops every constraint
                # registered under an excluded batch id while rebuilding
                _excl = getattr(self, '_exclude_batches', None)
                if _excl:
                    em.features['_exclude_batches'] = tuple(_excl)
                em = self.environment(em, *self.args, **self.kwargs) or em
                return em

            if self.solver is None and self.interface in ('auto', 'highs'):
                if self.interface == 'auto':
                    self.interface = self._auto_select_interface()
                has_nonlinear_expr = False
                has_quadratic_expr = False
                has_int_or_bin = False
                source_ok = False
                try:
                    import inspect
                    src = inspect.getsource(self.environment)
                    nonlinear_ops = ['sin(', 'cos(', 'tan(', 'log(', 'exp(', 'sqrt(', 'abs(', 'math.sin', 'math.cos', 'math.tan', 'math.log', 'math.exp', 'math.sqrt', 'math.fabs', 'np.sin', 'np.cos', 'np.tan', 'np.log', 'np.exp', 'np.sqrt', 'np.abs']
                    has_nonlinear_expr = any(op in src for op in nonlinear_ops)
                    has_quadratic_expr = '**' in src
                    has_int_or_bin = 'bvar(' in src or 'ivar(' in src or 'binary(' in src or 'integer(' in src
                    source_ok = True
                except Exception:
                    pass

                if not source_ok:
                    try:
                        import pyscipopt
                        self.interface = 'scip'
                        self.solver = 'scip'
                    except ImportError:
                        try:
                            from .helpers.solver_executables import inject_solver_paths
                            inject_solver_paths(("bonmin",))
                            from pyomo.environ import SolverFactory
                            if SolverFactory('bonmin').available():
                                self.interface = 'bonmin'
                                self.solver = 'bonmin'
                            else:
                                self.interface = 'highs'
                                self.solver = 'highs'
                        except Exception:
                            self.interface = 'highs'
                            self.solver = 'highs'
                    self.em = _try_generate(self.interface)
                elif has_nonlinear_expr or has_quadratic_expr:
                    if has_int_or_bin:
                        try:
                            import pyscipopt
                            self.interface = 'scip'
                            self.solver = 'scip'
                        except ImportError:
                            self.interface = 'bonmin'
                            self.solver = 'bonmin'
                    else:
                        self.interface = 'uno'
                        self.solver = 'uno'
                    self.em = _try_generate(self.interface)
                else:
                    try:
                        self.em = _try_generate('highs')
                    except Exception:
                        if has_int_or_bin:
                            try:
                                import pyscipopt
                                self.interface = 'scip'
                                self.solver = 'scip'
                            except ImportError:
                                self.interface = 'bonmin'
                                self.solver = 'bonmin'
                        else:
                            self.interface = 'uno'
                            self.solver = 'uno'
                        self.em = _try_generate(self.interface)
            else:
                self.em = _try_generate(self.interface)

            self.em._apply_exact_init(self.init)

            if self.interface =="jump" and self.inputdata:
                self.em.jlcode_data(self.inputdata)

        if self.method in ["heuristic"]:

            _probe = None
            if self.directions is None and self.environment is not None:
                _probe = model(method=self.method, name=self.name, interface=self.interface, agent=['idle'], no_agents=self._get_pop_size(), auto_linearize=self.auto_linearize)
                _probe._search_decoder = self.decoder
                _probe = self.environment(_probe, *self.args, **self.kwargs)
                if _probe is not None and _probe.features.get('directions'):
                    self.directions = [
                        _d if _d in ('min', 'max') else 'min'
                        for _d in _probe.features['directions']
                    ]
                    if self.number_of_objectives is None or self.number_of_objectives == 0:
                        self.number_of_objectives = len(self.directions)

            self._inferred_bounds = {}
            if self.environment is not None and _probe is not None:
                _probe_bounds = _probe.features
                _auto_vars = _probe_bounds.get('_auto_bound_vars', [])
                if _auto_vars:
                    from .operators.heuristic_operators import infer_bounds_from_constraints
                    self._inferred_bounds = infer_bounds_from_constraints(
                        self.environment, self.args, self.kwargs,
                        _probe_bounds, self.interface,
                        self._get_pop_size(), self.auto_linearize
                    )
                    for _vn, _vb in self._inferred_bounds.items():
                        _probe.features['variable_bound'][_vn] = _vb

            if self.directions is not None and len(self.directions)==1:

                if self.track_history:
                    self._epoch_count = self._get_epoch()
                    self._pop_size = self._get_pop_size()
                    self._hist_idx = 0
                    self.lb_record = []
                    self.ub_record = []
                    if self.interface == "feloopy":
                        self.ave_record = []
                        self.std_record = []

                if self.penalty_coefficient == 0:
                    self._penalty_needs_init = True
                else:
                    self._penalty_needs_init = False
                self._gen_count = 0
                self._infeasible_counts = []
                self._adaptive_boosted = False
                self._initial_penalty = 0

                self.em = None

                def instance(X):
                    
                    lm = model(method=self.method, name=self.name, interface=self.interface, agent=X, no_agents=self._get_pop_size(), auto_linearize=self.auto_linearize)
                    lm._search_decoder = self.decoder
                    if self.em is not None:
                        lm._impl = self.em
                    if self._inferred_bounds:
                        lm.features['_inferred_bounds'] = self._inferred_bounds
                    lm = environment(lm, *self.args, **self.kwargs)
                    if self._penalty_needs_init and lm.features['agent_status'] == 'idle':
                        if self.directions is None and lm.features['directions']:
                            self.directions = lm.features['directions']
                            if hasattr(self, 'em') and self.em is not None and hasattr(self.em, 'objectives_directions'):
                                self.em.objectives_directions = self.directions
                        if lm.features['constraint_counter'][0] > 0:
                            _obj_arr = np.asarray(lm.features['objectives'][0])
                            _obj_val = float(np.abs(_obj_arr).mean()) if lm.features['objective_counter'][0] > 0 else 1.0
                            _n_con = lm.features['constraint_counter'][0]
                            _var_range = 0.0
                            for _vb in lm.features['variable_bound'].values():
                                _var_range = max(_var_range, abs(_vb[1] - _vb[0]))
                            _var_range = max(_var_range, 1.0)
                            self.penalty_coefficient = max(10.0, 10.0 * max(1.0, _obj_val) * max(1.0, _n_con * 0.5) * max(1.0, _var_range / max(1.0, _obj_val)))
                            self._initial_penalty = self.penalty_coefficient
                            lm.features['penalty_coefficient'] = self.penalty_coefficient
                        self._penalty_needs_init = False
                    _replacement = getattr(lm, '_agent_replacement', None)
                    if _replacement is not None:
                        lm.agent = _replacement
                        lm.features['agent'] = _replacement
                        lm._agent_replacement = None
                    lm.sol(directions=self.directions,solver=self.solver,show_log=self.verbose, solver_options=self.options, auto_linearize=self.auto_linearize)
                    if self.track_history and hasattr(self, 'lb_record') and lm.features["agent_status"] not in ('feasibility_check', 'idle'):
                        self.lb_record.append(lm.current_min)
                        self.ub_record.append(lm.current_max)
                        if self.interface == "feloopy":
                            self.ave_record.append(lm.current_ave)
                            self.std_record.append(lm.current_std)
                        self._hist_idx += 1
                    self._gen_count += 1
                    if self._gen_count >= 3 and hasattr(lm, 'penalty') and isinstance(lm.penalty, np.ndarray) and lm.penalty.ndim == 1 and lm.penalty.shape[0] > 1:
                        _n_infeasible = int(np.sum(lm.penalty > 0))
                        _n_total = lm.penalty.shape[0]
                        self._infeasible_counts.append(_n_infeasible)
                        _check_window = 5
                        if len(self._infeasible_counts) >= _check_window:
                            _recent = self._infeasible_counts[-_check_window:]
                            _infeasible_frac = np.mean(_recent) / max(1, _n_total)
                            _cap = max(1e7, self._initial_penalty * 10000)
                            if _infeasible_frac > 0.8:
                                _factor = 1.0 + 2.0 * min(1.0, (_infeasible_frac - 0.8) / 0.2)
                                self.penalty_coefficient = min(self.penalty_coefficient * _factor, _cap)
                                lm.features['penalty_coefficient'] = self.penalty_coefficient
                            elif _infeasible_frac < 0.1 and self.penalty_coefficient > self._initial_penalty:
                                self.penalty_coefficient = max(self.penalty_coefficient * 0.9, self._initial_penalty)
                                lm.features['penalty_coefficient'] = self.penalty_coefficient
                    return lm[X]
                
                self.em = Implement(instance, init_solutions=self.init)

            else:

                if self.track_history:
                    self._epoch_count = self._get_epoch()
                    self._pop_size = self._get_pop_size()
                    self._hist_idx = 0
                    self.lb_record = []
                    self.ub_record = []
                    if self.interface == "feloopy":
                        self.ave_record = []
                        self.std_record = []

                if self.penalty_coefficient == 0:
                    self._penalty_needs_init = True
                else:
                    self._penalty_needs_init = False
                self._gen_count = 0
                self._infeasible_counts = []
                self._adaptive_boosted = False
                self._initial_penalty = 0

                self.em = None

                def instance(X):
                    m = model(method=self.method, name=self.name, interface=self.interface, agent=X, no_agents=self._get_pop_size(), auto_linearize=self.auto_linearize)
                    m._search_decoder = self.decoder
                    if self.em is not None:
                        m._impl = self.em
                    if self._inferred_bounds:
                        m.features['_inferred_bounds'] = self._inferred_bounds
                    m = self.environment(m, *self.args, **self.kwargs) or m
                    if self._penalty_needs_init and m.features['agent_status'] == 'idle':
                        if self.directions is None and m.features['directions']:
                            self.directions = m.features['directions']
                            if hasattr(self, 'em') and self.em is not None and hasattr(self.em, 'objectives_directions'):
                                self.em.objectives_directions = self.directions
                        if m.features['constraint_counter'][0] > 0:
                            _obj_arr = np.asarray(m.features['objectives'][0])
                            _obj_val = float(np.abs(_obj_arr).mean()) if m.features['objective_counter'][0] > 0 else 1.0
                            _n_con = m.features['constraint_counter'][0]
                            _var_range = 0.0
                            for _vb in m.features['variable_bound'].values():
                                _var_range = max(_var_range, abs(_vb[1] - _vb[0]))
                            _var_range = max(_var_range, 1.0)
                            self.penalty_coefficient = max(10.0, 10.0 * max(1.0, _obj_val) * max(1.0, _n_con * 0.5) * max(1.0, _var_range / max(1.0, _obj_val)))
                            self._initial_penalty = self.penalty_coefficient
                            m.features['penalty_coefficient'] = self.penalty_coefficient
                        self._penalty_needs_init = False
                    _replacement = getattr(m, '_agent_replacement', None)
                    if _replacement is not None:
                        m.agent = _replacement
                        m.features['agent'] = _replacement
                        m._agent_replacement = None
                    m.sol(self.directions, self.solver, self.options, obj_id='all', auto_linearize=self.auto_linearize)
                    if self.track_history and hasattr(self, 'lb_record') and m.features["agent_status"] not in ('feasibility_check', 'idle'):
                        self.lb_record.append(m.current_min)
                        self.ub_record.append(m.current_max)
                        if self.interface == "feloopy":
                            self.ave_record.append(m.current_ave)
                            self.std_record.append(m.current_std)
                        self._hist_idx += 1
                    self._gen_count += 1
                    if self._gen_count >= 3 and hasattr(m, 'penalty') and isinstance(m.penalty, np.ndarray) and m.penalty.ndim == 1 and m.penalty.shape[0] > 1:
                        _n_infeasible = int(np.sum(m.penalty > 0))
                        _n_total = m.penalty.shape[0]
                        self._infeasible_counts.append(_n_infeasible)
                        _check_window = 5
                        if len(self._infeasible_counts) >= _check_window:
                            _recent = self._infeasible_counts[-_check_window:]
                            _infeasible_frac = np.mean(_recent) / max(1, _n_total)
                            _cap = max(1e7, self._initial_penalty * 10000)
                            if _infeasible_frac > 0.8:
                                _factor = 1.0 + 2.0 * min(1.0, (_infeasible_frac - 0.8) / 0.2)
                                self.penalty_coefficient = min(self.penalty_coefficient * _factor, _cap)
                                m.features['penalty_coefficient'] = self.penalty_coefficient
                            elif _infeasible_frac < 0.1 and self.penalty_coefficient > self._initial_penalty:
                                self.penalty_coefficient = max(self.penalty_coefficient * 0.9, self._initial_penalty)
                                m.features['penalty_coefficient'] = self.penalty_coefficient
                    return m[X]
                self.em = implement(instance, init_solutions=self.init)

        if self.method in ["madm"]:
            self.em = madm(self.solver,self.name, self.interface)
            self.em = self.environment(self.em, *self.args, **self.kwargs) or self.em

        if self.method in ["sequential"]:
            from .algorithms.exact.sequential import (
                SequentialDecisionProblem, PFA, CFA, VFA, DLA, sdm_model)
            self._sdp = SequentialDecisionProblem()

            # --- Detect SDM components from environment function ---
            _env = self.kwargs.get('environment', self.environment)
            _sdm = {}
            if _env is not None:
                try:
                    _probe = sdm_model(name='_probe')
                    _env(_probe)
                    _sdm = _probe._get_sdm_components()
                except Exception:
                    pass

            # --- Extract kwargs with environment-registered defaults ---
            policy_type = self.kwargs.get('policy', None) or _sdm.get('policy') or 'pfa'
            S0 = self.kwargs.get('S0', None) or _sdm.get('S0')
            T = self.kwargs.get('T', None) or _sdm.get('T') or self.options.get('horizon', None) or 10
            N = self.kwargs.get('N', None) or _sdm.get('N') or self.options.get('nsim', None) or 100
            discount = (self.kwargs.get('discount', None)
                        or self.options.get('discount_factor', None)
                        or _sdm.get('discount') or 1.0)
            seed = self.kwargs.get('seed', None)
            save_trace = self.kwargs.get('save_trace', True)
            theta0 = self.kwargs.get('theta0', None) or _sdm.get('theta0')
            theta_bounds = (self.kwargs.get('theta_bounds', None)
                            or _sdm.get('theta_bounds'))
            var_names = self.kwargs.get('var_names', None)
            horizon = self.kwargs.get('horizon', None) or self.options.get('horizon', 1)
            scenarios = self.kwargs.get('scenarios', None)
            obj_operators = self.kwargs.get('obj_operators', None)
            uncertainty_set_constraints = self.kwargs.get('uncertainty_set_constraints', None)

            # --- Register transition/cost/exogenous on SDP ---
            _tr = self.kwargs.get('transition', _sdm.get('transition', None))
            _co = self.kwargs.get('cost', _sdm.get('cost', None))
            _rw = self.kwargs.get('reward', _sdm.get('reward', None))
            _ex = self.kwargs.get('exogenous', _sdm.get('exogenous', None))
            _tc = self.kwargs.get('terminal_cost', None)
            _fc = self.kwargs.get('feasibility', None)

            if self.directions is None and 'direction' in _sdm:
                self.directions = [_sdm['direction']]

            if 'direction' in _sdm:
                self._sdp._direction = _sdm['direction']

            if _tr is not None:
                self._sdp.transition(_tr)
            if _co is not None:
                self._sdp.cost(_co)
            if _rw is not None:
                self._sdp.reward(_rw)
            if _ex is not None:
                self._sdp.exogenous(_ex)
            if _tc is not None:
                self._sdp.terminal_cost(_tc)
            if _fc is not None:
                self._sdp.feasibility(_fc)

            # --- Transfer variable-based API attributes ---
            _sdp_vars = _sdm.get('sdp_vars', {})
            if _sdp_vars:
                self._sdp._sdp_vars = _sdp_vars
                _s0 = _sdm.get('S0', None)
                if isinstance(_s0, dict):
                    self._sdp._S0 = _s0

            # --- Build policy ---
            # --- Stage solver wiring: an auto-built stage model must use an
            # --- interface that can build one, and a solver name is only
            # --- kept when it belongs to that same interface (a heuristic
            # --- algorithm name must never reach an exact solver).
            try:
                from .helpers.solver_params import _HEURISTIC_IFACES
            except Exception:  # pragma: no cover
                _HEURISTIC_IFACES = frozenset()
            _has_build = self.kwargs.get('build_model', None) is not None
            _stage_iface = self.interface or 'highs'
            if _stage_iface in _HEURISTIC_IFACES and not _has_build:
                _stage_iface = 'highs'
            _stage_solver = (self.solver
                             if self.interface == _stage_iface else None)
            _stage_method = ('heuristic'
                             if self.interface in _HEURISTIC_IFACES
                             else 'exact')
            _tpar_info = {k: v for k, v in _sdp_vars.items()
                          if v.get('type') == 'tpar'} if _sdp_vars else {}
            _compiled_policy = _sdm.get('policy', None)
            if isinstance(policy_type, (PFA, CFA, VFA, DLA)):
                self._sdp_policy = policy_type
            elif callable(policy_type) and not isinstance(policy_type, type):
                self._sdp_policy = PFA(policy_type, theta0=theta0,
                                       name='pfa_sequential',
                                       tpar_info=_tpar_info)
            elif policy_type == 'pfa':
                _pf = _compiled_policy or self.kwargs.get('policy_fn', None) or _env
                self._sdp_policy = PFA(_pf, theta0=theta0,
                                       name='pfa_sequential',
                                       tpar_info=_tpar_info)
            elif policy_type == 'cfa':
                _build = self.kwargs.get('build_model', None)
                if _build is None:
                    _build = _make_build_model_fn(
                        _sdp_vars, _sdm, interface=_stage_iface)
                self._sdp_policy = CFA(
                    _build, theta0=theta0,
                    var_names=var_names,
                    method=_stage_method,
                    interface=_stage_iface,
                    solver=_stage_solver,
                    solver_options=self.options,
                    obj_operators=obj_operators,
                    uncertainty_set_constraints=uncertainty_set_constraints,
                    name='cfa_sequential')
            elif policy_type == 'vfa':
                _build = self.kwargs.get('build_model', None)
                if _build is None:
                    _build = _make_build_model_fn(
                        _sdp_vars, _sdm, include_value=True,
                        interface=_stage_iface)
                self._sdp_policy = VFA(
                    _build, None, theta0=theta0,
                    var_names=var_names,
                    method=_stage_method,
                    interface=_stage_iface,
                    solver=_stage_solver,
                    solver_options=self.options,
                    obj_operators=obj_operators,
                    uncertainty_set_constraints=uncertainty_set_constraints,
                    name='vfa_sequential')
            elif policy_type == 'dla':
                _build = self.kwargs.get('build_model', None)
                if _build is None:
                    _build = _make_build_model_fn(
                        _sdp_vars, _sdm, interface=_stage_iface)
                self._sdp_policy = DLA(
                    _build, horizon=horizon,
                    scenarios=scenarios, var_names=var_names,
                    method=_stage_method,
                    interface=_stage_iface,
                    solver=_stage_solver,
                    solver_options=self.options,
                    obj_operators=obj_operators,
                    uncertainty_set_constraints=uncertainty_set_constraints,
                    name='dla_sequential')
            else:
                raise ValueError(f"Unknown sequential policy: {policy_type!r}")

            self._sdp.environment(_env or self.environment)
            self._sdp_T = T
            self._sdp_N = N
            self._sdp_discount = discount
            self._sdp_seed = seed
            self._sdp_save_trace = save_trace
            self._sdp_S0 = S0
            self._sdp_theta_bounds = theta_bounds

            self.em = model(method='exact', name=self.name, auto_linearize=self.auto_linearize)
            self.em.features['directions'] = self.directions or ['min']
            self.directions = self.directions or ['min']
            self.number_of_objectives = len(self.directions)
            self.solutions = {}

        if not verbose and self.method != "sequential":
            end_progress(success_message="Generated")

    def healthy(self):
        if self.method == "sequential":
            return True
        return self.em.healthy()

    def get_status(self):
        return self.em.get_status()

    def get_bound(self):
        return self.em.get_bound()

    def get_variable(self, variable_with_index):
        return self.em.get_variable(variable_with_index)

    @staticmethod
    def get_algo_options(solver, interface):
        """Return available options for a heuristic algorithm.

        Args:
            solver: Algorithm name (e.g. ``'pso'``, ``'de'``, ``'ns-ga-ii'``).
            interface: Heuristic interface (``'mealpy'``, ``'niapy'``,
                ``'indago'``, ``'pymoo'``).

        Returns:
            dict: ``{option_name: description}`` for unified and
            algorithm-specific parameters.

        Examples
        --------
        >>> flp.search.get_algo_options("pso", "mealpy")
        >>> flp.search.get_algo_options("pso", "indago")
        """
        from .generators.solution.option_utils import get_algo_options
        return get_algo_options(solver, interface)

    @staticmethod
    def get_params(interface="highs", solver=None):
        """Return solver-specific options passable via ``flp.search(..., options={})``.

        Args:
            interface: Felopy interface name, e.g. ``"highs"``, ``"gurobi"``,
                ``"copt"``, ``"cplex"``, ``"ortools"``, ``"scip"`` etc.
            solver: Solver name (optional).  For heuristic interfaces
                (mealpy/niapy/indago/pymoo) this selects the algorithm.

        Returns:
            dict: ``{param_name: description}``.  Keys can be copied into
            ``options`` for :meth:`search`.

        Examples
        --------
        >>> flp.search.get_params("highs", "highs")
        >>> flp.search.get_params("gurobi", "gurobi")
        >>> flp.search.get_params("mealpy", "pso")   # heuristic
        """
        from .helpers.solver_params import get_solver_params
        return get_solver_params(interface, solver)

    @staticmethod
    def list_algorithms(interface):
        """Return a list of available algorithm names for the given interface.

        Args:
            interface: Heuristic interface (``'mealpy'``, ``'niapy'``,
                ``'indago'``, ``'pymoo'``).

        Returns:
            list: Sorted list of algorithm name strings.
        """
        from .generators.solution.option_utils import list_algorithms
        return list_algorithms(interface)

    def _get_epoch(self):
        from .generators.solution.option_utils import get_epoch_value
        return get_epoch_value(self.options) or 100

    def _get_pop_size(self):
        from .generators.solution.option_utils import get_pop_size_value
        return get_pop_size_value(self.options) or 50

    def _reset_repeat_state(self):
        if self.track_history:
            self.lb_record = []
            self.ub_record = []
            if self.interface == "feloopy":
                self.ave_record = []
                self.std_record = []
            self._hist_idx = 0
            self._gen_count = 0
            self._infeasible_counts = []
            self._adaptive_boosted = False
            self._penalty_needs_init = (self.penalty_coefficient == 0)

    def _save_history_snapshot(self):
        if not self.track_history:
            return None
        snap = {
            'lb_record': list(self.lb_record),
            'ub_record': list(self.ub_record),
            '_hist_idx': self._hist_idx,
            '_gen_count': self._gen_count,
            '_infeasible_counts': list(self._infeasible_counts),
            '_adaptive_boosted': self._adaptive_boosted,
            '_penalty_needs_init': self._penalty_needs_init,
        }
        if self.interface == "feloopy":
            snap['ave_record'] = list(self.ave_record)
            snap['std_record'] = list(self.std_record)
        return snap

    def _restore_history_snapshot(self, snap):
        if snap is None:
            return
        self.lb_record = snap['lb_record']
        self.ub_record = snap['ub_record']
        self._hist_idx = snap['_hist_idx']
        self._gen_count = snap['_gen_count']
        self._infeasible_counts = snap['_infeasible_counts']
        self._adaptive_boosted = snap['_adaptive_boosted']
        self._penalty_needs_init = snap['_penalty_needs_init']
        if self.interface == "feloopy":
            self.ave_record = snap['ave_record']
            self.std_record = snap['std_record']
    
    def run(self, verbose, show_solver_log=None):

        if show_solver_log is None:
            show_solver_log = verbose

        _show_progress = ((not verbose) and self.method != "sequential"
                          and not self.boost)
        if _show_progress:
            start_progress(message="Searching...", spinner="dots", color="cyan")

        try:
            return self._run_impl(verbose, show_solver_log)
        except BaseException:
            # stop the spinner on failure too, otherwise its thread keeps
            # clearing/spinning forever in notebooks and hides the error
            if _show_progress:
                end_progress(success=False, failure_message="Search failed")
            raise

    def _run_impl(self, verbose, show_solver_log=None):

        import sys
        import threading
        import time

        solving_complete = False
        
        if self.method!="madm":
            if  self.number_of_objectives==1:
                if self.method in ["exact", "convex", "constraint", "uncertain"]:
                    if self.boost:
                        _boost_overhead_start = timeit.default_timer()
                        if self.auto_linearize and self.boost is True:
                            _boost_needs_direct_solve = True
                        else:
                            _boost_method = self._select_boost_method()
                            if _boost_method == 'direct':
                                _boost_needs_direct_solve = True
                            else:
                                try:
                                    if _boost_method == 'benders':
                                        self._run_automatic_benders(verbose)
                                    elif _boost_method == 'lagrangian':
                                        self._run_automatic_lagrangian(verbose)
                                    elif _boost_method == 'cg':
                                        self._run_automatic_column_generation(verbose)
                                    elif _boost_method == 'ccg':
                                        self._run_automatic_column_generation(verbose, variant='ccg')
                                    elif _boost_method == 'dcg':
                                        self._run_automatic_column_generation(verbose, variant='dcg')
                                    elif _boost_method == 'dw':
                                        self._run_automatic_column_generation(verbose, variant='dw')
                                    elif _boost_method == 'dw2':
                                        self._run_automatic_column_generation(verbose, variant='dw2')
                                    elif _boost_method == 'branching':
                                        self._run_automatic_branching(verbose)
                                    else:
                                        _boost_method = 'direct'
                                    if _boost_method == 'direct':
                                        _boost_needs_direct_solve = True
                                    else:
                                        _boost_needs_direct_solve = False
                                except Exception as _decomp_exc:
                                    _boost_needs_direct_solve = True
                                    for _res_attr in ('_benders_result',
                                                      '_lagrangian_result',
                                                      '_cg_result',
                                                      '_branching_result'):
                                        if hasattr(self, _res_attr):
                                            delattr(self, _res_attr)
                                    self._boost_fallback_reason = (
                                        "decomposition failed: %s: %s"
                                        % (type(_decomp_exc).__name__,
                                           _decomp_exc))
                            if hasattr(self, '_benders_result'):
                                _bst = getattr(
                                    self._benders_result, 'status', None)
                                if (_bst == 'direct' or
                                        getattr(_bst, 'value', None) == 'direct'):
                                    _boost_needs_direct_solve = True
                        self.mgt += timeit.default_timer() - _boost_overhead_start
                    solve_options = self.options
                    if self.boost:
                        solve_options = {
                            key: value for key, value in self.options.items()
                            if key not in ('max_iterations', 'tolerance', 'cg_variant',
                                           'pricing_strategy', 'relaxation_strategy', 'lag_step_size',
                                           'benders_method', 'logic_cut_callback',
                                           'relax_subproblem_integrality',
                                           'cut_strategy', 'acceleration',
                                           'complicating_variables')
                        }
                    _sol_solver = self.solver
                    if self.boost and isinstance(self.solver, (list, tuple)):
                        _sol_solver = self.solver[0]
                    if not self.boost or _boost_needs_direct_solve:
                        self.em.sol(directions=self.directions, solver=_sol_solver, debug=self.debug, show_log=False if self.boost else show_solver_log, email=self.email, time_limit=self.time_limit, cpu_threads=self.cpu_threads,absolute_gap=self.absolute_gap, relative_gap=self.relative_gap, solver_options=solve_options, obj_operators=self.obj_operators, uncertainty_set_constraints=self.uncertainty_set_constraints, auto_linearize=self.auto_linearize)
                    if self.boost and hasattr(self, '_benders_result') and getattr(self._benders_result, 'status', None) == 'logic_based_direct':
                        self._benders_result.objective = float(self.em.get_obj())
                        self._benders_result.status = 'optimal' if self.em.healthy() else self._benders_result.status
                        self._benders_result.variables = {
                            name: self.em.get_numpy_var(name)
                            for kind, name in self.em.features.get('variables', {})
                            if kind in ('ivar', 'bvar')
                        }
                    if self.boost and hasattr(self, '_benders_result') and getattr(self._benders_result, 'status', None) != 'logic_based_direct':
                        obj_val = getattr(self._benders_result, 'objective', None)
                        if obj_val is not None:
                            if self.directions and self.directions[0] == 'max':
                                obj_val = -obj_val
                            self.objective_values = np.array([[obj_val]])
                        bounds = getattr(self._benders_result, 'bounds', [])
                        if bounds:
                            last_bound = bounds[-1] if bounds else (None, None)
                            self._decomposition_lower_bound = last_bound[0] if last_bound[0] is not None else None
                            self._decomposition_upper_bound = last_bound[1] if last_bound[1] is not None else None
                        self._decomposition_iterations = getattr(self._benders_result, 'iterations', 0)
                        _bst = getattr(self._benders_result, 'status', 'unknown')
                        # status may be a plain Enum — report its value,
                        # not 'BendersStatus.INFEASIBLE'.
                        self._decomposition_status = str(getattr(_bst, 'value', _bst))
                    if self.boost and hasattr(self, '_lagrangian_result'):
                        obj_val = getattr(self._lagrangian_result, 'objective', None)
                        if obj_val is not None:
                            self.objective_values = np.array([[obj_val]])
                        self._decomposition_lower_bound = getattr(self._lagrangian_result, 'lower_bound', None)
                        self._decomposition_upper_bound = getattr(self._lagrangian_result, 'upper_bound', None)
                        self._decomposition_iterations = getattr(self._lagrangian_result, 'iterations', 0)
                        self._decomposition_status = getattr(self._lagrangian_result, 'status', 'unknown')
                    if self.boost and hasattr(self, '_cg_result'):
                        obj_val = getattr(self._cg_result, 'objective', None)
                        if obj_val is not None:
                            self.objective_values = np.array([[obj_val]])
                        self._decomposition_iterations = getattr(self._cg_result, 'iterations', 0)
                        self._decomposition_status = getattr(self._cg_result, 'status', 'unknown')
                    if self.boost and hasattr(self, '_branching_result'):
                        obj_val = getattr(self._branching_result, 'objective', None)
                        if obj_val is not None and not np.isinf(obj_val):
                            self.objective_values = np.array([[obj_val]])
                        self._decomposition_iterations = getattr(self._branching_result, 'iterations', 0)
                        _bst = getattr(self._branching_result, 'status', 'unknown')
                        self._decomposition_status = str(getattr(_bst, 'value', _bst))
                        bb = getattr(self._branching_result, 'best_bound', None)
                        if bb is not None:
                            self._decomposition_lower_bound = bb
                    # --- Fix CPT for decomposition methods ---
                    # When using boost, CPT should come from the
                    # decomposition result's runtime, not from em.get_time().
                    if self.boost and not _boost_needs_direct_solve:
                        _decomp_runtime = None
                        if hasattr(self, '_benders_result'):
                            _decomp_runtime = getattr(self._benders_result, 'runtime_total', None)
                        elif hasattr(self, '_lagrangian_result'):
                            _decomp_runtime = getattr(self._lagrangian_result, 'runtime_total', None)
                        elif hasattr(self, '_cg_result'):
                            _decomp_runtime = getattr(self._cg_result, 'runtime_total', None)
                        elif hasattr(self, '_branching_result'):
                            _decomp_runtime = getattr(self._branching_result, 'runtime_total', None)
                        if _decomp_runtime is not None:
                            self.cpt = _decomp_runtime

                    if self.boost and not _boost_needs_direct_solve:
                        import time as _time
                        _decomp_obj = None
                        _decomp_status = None
                        _decomp_time = None
                        if hasattr(self, '_benders_result'):
                            _decomp_obj = getattr(self._benders_result, 'objective', None)
                            _bst = getattr(self._benders_result, 'status', 'unknown')
                            _decomp_status = str(getattr(_bst, 'value', _bst))
                            _decomp_time = getattr(self._benders_result, 'runtime_total', None)
                        elif hasattr(self, '_lagrangian_result'):
                            _decomp_obj = getattr(self._lagrangian_result, 'objective', None)
                            _decomp_status = getattr(self._lagrangian_result, 'status', 'unknown')
                            _decomp_time = getattr(self._lagrangian_result, 'runtime_total', None)
                        elif hasattr(self, '_cg_result'):
                            _decomp_obj = getattr(self._cg_result, 'objective', None)
                            _decomp_status = getattr(self._cg_result, 'status', 'unknown')
                            _decomp_time = getattr(self._cg_result, 'runtime_total', None)
                        elif hasattr(self, '_branching_result'):
                            _decomp_obj = getattr(self._branching_result, 'objective', None)
                            _bst = getattr(self._branching_result, 'status', 'unknown')
                            _decomp_status = str(getattr(_bst, 'value', _bst))
                            _decomp_time = getattr(self._branching_result, 'runtime_total', None)
                        try:
                            _decomp_valid = (_decomp_obj is not None
                                             and _decomp_obj != 0.0
                                             and not np.isinf(float(_decomp_obj))
                                             and not np.isnan(float(_decomp_obj)))
                        except (TypeError, ValueError):
                            _decomp_valid = False
                        _need_fallback = False
                        _fallback_reason = None
                        if not _decomp_valid:
                            _need_fallback = True
                            _fallback_reason = (
                                f"decomposition objective invalid "
                                f"(status={_decomp_status}, obj={_decomp_obj!r})")
                        else:
                            # Always solve direct and compare quality + timing
                            _t0_direct = _time.perf_counter()
                            self.em.sol(
                                directions=self.directions, solver=_sol_solver,
                                debug=self.debug, show_log=False,
                                email=self.email, time_limit=self.time_limit,
                                cpu_threads=self.cpu_threads,
                                absolute_gap=self.absolute_gap,
                                relative_gap=self.relative_gap,
                                solver_options=solve_options,
                                obj_operators=self.obj_operators,
                                uncertainty_set_constraints=self.uncertainty_set_constraints,
                                auto_linearize=self.auto_linearize)
                            _direct_obj = float(self.em.get_obj())
                            _t1_direct = _time.perf_counter()
                            _direct_time = _t1_direct - _t0_direct
                            self._direct_solved = True

                            _is_min = self.directions and self.directions[0] == 'min'

                            # Flip Benders/Lagrangian objective sign for max
                            # so the comparison with direct solve is fair.
                            if not _is_min and _decomp_obj is not None:
                                _decomp_obj = -_decomp_obj

                            # Check quality: decomposition must be at least as good
                            if _is_min:
                                if _decomp_obj > _direct_obj + 1e-6:
                                    _need_fallback = True
                                    _fallback_reason = (
                                        f"worse quality: decomp {_decomp_obj!r} "
                                        f"vs direct {_direct_obj!r} (min)")
                            else:
                                if _decomp_obj < _direct_obj - 1e-6:
                                    _need_fallback = True
                                    _fallback_reason = (
                                        f"worse quality: decomp {_decomp_obj!r} "
                                        f"vs direct {_direct_obj!r} (max)")

                            # Check timing: if decomposition is significantly
                            # slower than direct solve, prefer direct
                            if (not _need_fallback and _decomp_time is not None
                                    and _direct_time > 0):
                                _speedup = _direct_time / max(_decomp_time, 1e-9)
                                # If decomposition is >2x slower and not
                                # significantly better in quality, prefer direct
                                if _speedup < 0.5:
                                    _quality_diff = abs(_decomp_obj - _direct_obj)
                                    if _quality_diff < 1e-4 * max(abs(_direct_obj), 1.0):
                                        _need_fallback = True
                                        _fallback_reason = (
                                            f"slower than direct "
                                            f"({_decomp_time:.3f}s decomp vs "
                                            f"{_direct_time:.3f}s direct, "
                                            f"equal quality)")
                        # Exposed for diagnostics/tests: which of the three
                        # safety-net checks (invalid / quality / timing)
                        # triggered the fallback, or None if it passed.
                        self._boost_fallback_reason = _fallback_reason
                        if _need_fallback:
                            # Decomposition failed or was worse; use direct solve
                            if not hasattr(self, '_direct_solved') or not self._direct_solved:
                                self.em.sol(
                                    directions=self.directions, solver=_sol_solver,
                                    debug=self.debug, show_log=show_solver_log,
                                    email=self.email, time_limit=self.time_limit,
                                    cpu_threads=self.cpu_threads,
                                    absolute_gap=self.absolute_gap,
                                    relative_gap=self.relative_gap,
                                    solver_options=solve_options,
                                    obj_operators=self.obj_operators,
                                    uncertainty_set_constraints=self.uncertainty_set_constraints,
                                    auto_linearize=self.auto_linearize)
                                self._direct_solved = True
                            # Update decomposition result with direct solve
                            # objective so get_obj() returns the best value.
                            _direct_obj = float(self.em.get_obj())
                            if hasattr(self, '_cg_result') and self._cg_result is not None:
                                self._cg_result.objective = _direct_obj
                                self._cg_result.status = 'fallback_direct'
                            if hasattr(self, '_lagrangian_result') and self._lagrangian_result is not None:
                                self._lagrangian_result.objective = _direct_obj
                                self._lagrangian_result.status = 'fallback_direct'
                                # keep bounds consistent with the direct
                                # optimum (avoids an LB=inf UB=inf display)
                                try:
                                    self._lagrangian_result.lower_bound = float(_direct_obj)
                                    self._lagrangian_result.upper_bound = float(_direct_obj)
                                except Exception:
                                    pass
                            if hasattr(self, '_benders_result') and self._benders_result is not None:
                                # Benders fallback: update objective/status and
                                # also fix bounds to direct's optimal (avoid LB84 UB0 display)
                                try:
                                    self._benders_result.objective = _direct_obj
                                    self._benders_result.status = 'fallback_direct'
                                    # Keep bounds consistent with direct optimum
                                    if hasattr(self._benders_result, 'bounds') and self._benders_result.bounds:
                                        self._benders_result.bounds = [(float(_direct_obj), float(_direct_obj))]
                                    if hasattr(self._benders_result, 'lb_history'):
                                        self._benders_result.lb_history = [float(_direct_obj)]
                                    if hasattr(self._benders_result, 'ub_history'):
                                        self._benders_result.ub_history = [float(_direct_obj)]
                                except Exception:
                                    pass
                    if self.solver == 'bonmin' and not self.em.healthy():
                        self.interface = 'couenne'
                        self.solver = 'couenne'
                        def _regen(iface):
                            em = model(method=self.method,name=self.name,interface=iface,constraint_ids=self.constraint_ids,no_scenarios=self.no_scenarios,auto_linearize=self.auto_linearize)
                            _excl = getattr(self, '_exclude_batches', None)
                            if _excl:
                                em.features['_exclude_batches'] = tuple(_excl)
                            em = self.environment(em, *self.args, **self.kwargs) or em
                            return em
                        self.em = _regen('couenne')
                        self.em.features['solver_name'] = 'couenne'
                        self.em.sol(directions=self.directions, solver='couenne', debug=self.debug, show_log=show_solver_log, email=self.email, time_limit=self.time_limit, cpu_threads=self.cpu_threads,absolute_gap=self.absolute_gap, relative_gap=self.relative_gap, solver_options=self.options, auto_linearize=self.auto_linearize)
                        
                elif self.method == "heuristic":
                    
                    if self.repeat > 1:
                        Multiplier = {'max': 1, 'min': -1}
                        _dir = Multiplier.get(self.directions[0] if self.directions else 'max', 1)
                        _best_reward = -_dir * np.inf
                        _best_snap = None
                        for _ri in range(self.repeat):
                            self._reset_repeat_state()
                            self.em.sol(penalty_coefficient=self.penalty_coefficient, number_of_times=1, show_log=show_solver_log)
                            _snap = self._save_history_snapshot()
                            _r = float(np.asarray(self.em.BestReward).flat[0])
                            if _dir * _r >= _dir * _best_reward:
                                _best_reward = _r
                                _best_snap = _snap
                                _best_agent = self.em.BestAgent
                                _best_time_start = self.em.start
                                _best_time_end = self.em.end
                        self.em.BestAgent = _best_agent
                        self.em.BestReward = _best_reward
                        self.em.start = _best_time_start
                        self.em.end = _best_time_end
                        self._restore_history_snapshot(_best_snap)
                    else:
                        self.em.sol(penalty_coefficient=self.penalty_coefficient, number_of_times=1, show_log=show_solver_log)
                    
                    if self.track_history:

                        _ep = self._epoch_count
                        _n_used = self._hist_idx
                        lb = np.array(self.lb_record, dtype=float)
                        ub = np.array(self.ub_record, dtype=float)
                        del self.lb_record, self.ub_record

                        if lb.ndim == 2 and lb.shape[1] == 1:
                            lb = lb.ravel()
                            ub = ub.ravel()

                        if self.interface in ['mealpy', 'niapy', 'pygad']: 

                            _ps = self._pop_size or 1
                            # records are appended once per evaluated agent,
                            # but idle / feasibility-check batches are skipped
                            # and the population size may be misreported: never
                            # reshape more rows than were actually recorded
                            if _n_used <= 0:
                                lb = np.zeros(1)
                                ub = np.zeros(1)
                                _n_used = 1
                            if _n_used < _ps:
                                _ps = _n_used
                            n_ep = _n_used // _ps if _ps else _ep - 1
                            n_ep = max(n_ep, 1)
                            ub = ub[:n_ep * _ps].reshape(n_ep, _ps)
                            ep_max = np.max(ub, axis=1)
                            ep_min = np.min(ub, axis=1)
                            ep_ave = np.mean(ub, axis=1)
                            ep_std = np.std(ub, axis=1)

                            self.best_max = np.maximum.accumulate(ep_max)
                            self.best_min = np.minimum.accumulate(ep_min)
                            self.middle = (self.best_max + self.best_min) / 2
                            self.range = self.best_max - self.best_min
                            self.average = ep_ave
                            self.std = ep_std

                        else:
                            ave = np.array(self.ave_record, dtype=float) if self.interface == "feloopy" else ub
                            std = np.array(self.std_record, dtype=float) if self.interface == "feloopy" else np.zeros_like(ub)
                            if self.interface == "feloopy":
                                del self.ave_record, self.std_record
                            if ave.ndim == 2 and ave.shape[1] == 1:
                                ave = ave.ravel()
                                std = std.ravel()

                            self.best_max = np.maximum.accumulate(ub)
                            self.best_min = np.minimum.accumulate(lb)
                            self.middle = (self.best_max + self.best_min) / 2
                            self.range = self.best_max - self.best_min
                            self.average = ave
                            self.std = std

                        self.final_min = float(self.best_min[-1])
                        self.final_max = float(self.best_max[-1])
                        self.lb_for_min = self.best_max - self.range[-1]
                        self.ub_for_max = self.best_min + self.range[-1]
                        self.stagnation = float(np.sum(self.ub_for_max - self.best_max <= 1e-6)) / _ep

                        _improve = np.diff(self.best_max)
                        self.convergence_rate = float(np.sum(_improve > 0)) / len(_improve) if len(_improve) > 0 else 0.0
                        self.improvement_pct = float((self.best_max[-1] - self.best_max[0]) / (abs(self.best_max[0]) + 1e-12)) if len(self.best_max) > 1 else 0.0
                        _plateau = 0
                        _max_plateau = 0
                        for v in _improve:
                            if v <= 1e-12:
                                _plateau += 1
                                _max_plateau = max(_max_plateau, _plateau)
                            else:
                                _plateau = 0
                        self.max_plateau = _max_plateau
                        self.population_diversity = float(np.mean(self.range)) if len(self.range) > 0 else 0.0
                    
                elif self.method == "sequential":
                    _do_optimize = self.kwargs.get('optimize', False)
                    if _do_optimize:
                        _opt_opts = self.options
                        self._opt_result = self._sdp.optimize(
                            self._sdp_policy, self._sdp_S0, self._sdp_T,
                            N_eval=self._sdp_N, seed=self._sdp_seed,
                            max_iters=self.kwargs.get('max_iters', 50),
                            theta_bounds=self._sdp_theta_bounds,
                            show_log=show_solver_log,
                            solver=self.kwargs.get('opt_solver', 'ga'),
                            interface=self.kwargs.get('opt_interface', 'feloopy'),
                            solver_options=_opt_opts,
                        )
                        self.result = self._sdp.simulate(
                            self._sdp_policy, self._sdp_S0, self._sdp_T,
                            N=self._sdp_N, seed=self._sdp_seed,
                            discount=self._sdp_discount, show_log=False,
                            save_trace=self._sdp_save_trace)
                        self.solutions = {
                            'total_costs': self.result.total_costs,
                            'discounted_costs': self.result.discounted_costs,
                            'stats': self.result.stats,
                            'summary': self.result.summary(),
                            'optimization': self._opt_result.summary(),
                            'opt_result': self._opt_result,
                        }
                        if self.result.traces is not None:
                            self.solutions['traces'] = self.result.traces_as_dicts
                    else:
                        self.result = self._sdp.simulate(
                            self._sdp_policy, self._sdp_S0, self._sdp_T,
                            N=self._sdp_N, seed=self._sdp_seed,
                            discount=self._sdp_discount, show_log=show_solver_log,
                            save_trace=self._sdp_save_trace)
                        self.solutions = {
                            'total_costs': self.result.total_costs,
                            'discounted_costs': self.result.discounted_costs,
                            'stats': self.result.stats,
                            'summary': self.result.summary(),
                        }
                        if self.result.traces is not None:
                            self.solutions['traces'] = self.result.traces_as_dicts

            else:

                if self.method in ["exact", "convex", "constraint", "uncertain"]:

                    if self.approach == "multilevel" and isinstance(self.options.get('problem'), MultiLevelProblem):
                        problem = self.options['problem']
                        self.time_solve_begin = timeit.default_timer()
                        self.result = problem.solve(
                            solver_name=self.solver,
                            solver_options={},
                            method=self.options.get('method', 'sequential'),
                            show_log=show_solver_log,
                            save_vars=True,
                            email=self.email,
                            time_limit=self.time_limit,
                            cpu_threads=self.cpu_threads,
                            absolute_gap=self.absolute_gap,
                            relative_gap=self.relative_gap,
                        )
                        self.time_solve_end = timeit.default_timer()
                    elif self.directions is None or len(self.directions) <= 1:
                        self.em.sol(directions=self.directions, solver=self.solver, debug=self.debug, show_log=show_solver_log, email=self.email, 
                            time_limit=self.time_limit, cpu_threads=self.cpu_threads, absolute_gap=self.absolute_gap, 
                            relative_gap=self.relative_gap, solver_options=self.options, obj_operators=self.obj_operators,
                            uncertainty_set_constraints=self.uncertainty_set_constraints, auto_linearize=self.auto_linearize)
                    else:
                        try:
                            from .extras.algorithms.exact.multiobjective import sol_multi
                        except:
                            from .algorithms.exact.multiobjective import sol_multi
                            
                        def instance():
                            self.em = model(method=self.method, name=self.name, interface=self.interface, no_scenarios=self.no_scenarios, auto_linearize=self.auto_linearize)
                            self.em = self.environment(self.em, *self.args, **self.kwargs) or self.em
                            return self.em
                        
                        self.time_solve_begin = timeit.default_timer()
                        self.result = sol_multi(instance=instance,
                                                directions=self.directions.copy(),
                                                objective_id=self.approach,
                                                solver_name=self.solver,
                                                save_vars=True,
                                                approach_options=self.options,
                                                show_log=show_solver_log, 
                                                email=self.email, 
                                                time_limit=self.time_limit, 
                                                cpu_threads=self.cpu_threads,
                                                absolute_gap=self.absolute_gap, 
                                                relative_gap=self.relative_gap,
                                                obj_operators=self.obj_operators,
                                                uncertainty_set_constraints=self.uncertainty_set_constraints
                                                )
                        self.time_solve_end = timeit.default_timer()

                        if self.em.features.get('problem_type') is None:
                            self.em.features['problem_type'] = self.em._detect_problem_type()

            if self.method in ["heuristic"] and self.number_of_objectives > 1:

                    if self.repeat > 1:
                        _best_reward = -np.inf
                        _best_snap = None
                        for _ri in range(self.repeat):
                            self._reset_repeat_state()
                            self.em.solve(show_log=show_solver_log, penalty_coefficient=self.penalty_coefficient, number_of_times=1)
                            _snap = self._save_history_snapshot()
                            _r = float(np.mean(self.em.BestReward)) if hasattr(self.em.BestReward, '__len__') else float(self.em.BestReward)
                            if _r >= _best_reward:
                                _best_reward = _r
                                _best_snap = _snap
                                _best_agent = self.em.BestAgent
                                _best_time_start = self.em.start
                                _best_time_end = self.em.end
                        self.em.BestAgent = _best_agent
                        self.em.BestReward = _best_agent if _best_reward == -np.inf else self.em.BestReward
                        self.em.start = _best_time_start
                        self.em.end = _best_time_end
                        self._restore_history_snapshot(_best_snap)
                    else:
                        self.em.solve(show_log=show_solver_log, penalty_coefficient=self.penalty_coefficient, number_of_times=1)

                    if self.track_history:

                        _ep = self._epoch_count
                        _nobj = self.number_of_objectives
                        _n_used = self._hist_idx
                        lb = np.array(self.lb_record, dtype=float)
                        ub = np.array(self.ub_record, dtype=float)
                        del self.lb_record, self.ub_record
                        ave = np.array(self.ave_record, dtype=float) if self.interface == "feloopy" else ub
                        std = np.array(self.std_record, dtype=float) if self.interface == "feloopy" else np.zeros_like(ub)
                        if self.interface == "feloopy":
                            del self.ave_record, self.std_record

                        self.best_max = np.maximum.accumulate(ub, axis=0)
                        self.best_min = np.minimum.accumulate(lb, axis=0)
                        self.middle = (self.best_max + self.best_min) / 2
                        self.range = self.best_max - self.best_min
                        self.average = ave
                        self.std = std
                        self.lb_for_min = self.best_max - self.range[-1:]
                        self.ub_for_max = self.best_min + self.range[-1:]

                        self.final_min = self.best_min[-1].copy()
                        self.final_max = self.best_max[-1].copy()

                        self.stagnation = float(np.mean(np.all(self.ub_for_max - self.best_max <= 1e-6, axis=1))) / _ep

                        _improve = np.diff(self.best_max, axis=0)
                        self.convergence_rate = float(np.mean(np.any(_improve > 0, axis=1))) if len(_improve) > 0 else 0.0
                        self.max_plateau = 0
                        _plateau = 0
                        for row in _improve:
                            if np.all(row <= 1e-12):
                                _plateau += 1
                                self.max_plateau = max(self.max_plateau, _plateau)
                            else:
                                _plateau = 0
                        self.population_diversity = float(np.mean(self.range)) if len(self.range) > 0 else 0.0

        if self.method == "madm":
            self.time_solve_begin = timeit.default_timer()
            self.em.sol(directions=self.directions, solver_options=self.options)
            self.time_solve_end = timeit.default_timer()
        
        if self.method in ["heuristic"]: self.cpt = self.em.get_time()

        is_multilevel = (self.approach == "multilevel" and
                         isinstance(self.options.get('problem'), MultiLevelProblem))

        if self.method in ("sequential", "heuristic") or is_multilevel or self.em.healthy() or self.number_of_objectives != 1:
            if self.method not in ["madm"]:
                if is_multilevel:
                    self.solutions = self.result[1].get('all_variables', {})
                    self.objective_values = self.result[0]
                    self.num_objective_values = self.objective_values.shape[0]
                    self.cpt = self.time_solve_end - self.time_solve_begin
                elif self.number_of_objectives > 1:
                    
                    if self.method not in ["heuristic"]:
                        self.solutions = self.result[4]
                        self.payoff = self.result[1]
                        self.conflict = self.result[2]
                        self.conflict_metric = self.result[3]
                        self.ogrs = self.result[5] if len(self.result) > 5 else []
                    else:
                        from .algorithms.exact.multiobjective.utils import \
                            _deduplicate_pareto
                        _reward = np.asarray(self.em.get_obj())
                        _agent = np.asarray(self.em.BestAgent)
                        if (_reward.ndim > 1 and _agent.ndim > 1
                                and len(_reward) > 0
                                and len(_reward) == len(_agent)):
                            _reward, _agent = _deduplicate_pareto(
                                _reward, list(_agent))
                            if len(_reward) > 0:
                                self.em.BestReward = _reward
                                self.em.BestAgent = np.asarray(_agent)

                        num_pareto = self.em.get_obj().shape[0]
                        self.solutions = [ {} for i in range(num_pareto)]
                        for i in range(num_pareto):
                            for j in self.em.VariablesDim.keys():
                                if self.em.VariablesType[j] in ["pvar", "fvar"]:
                                    self.solutions[i][j] = self.em.VariablesBound[j][0] + self.em.BestAgent[i,self.em.VariablesSpread[j][0]:self.em.VariablesSpread[j][1]] * (self.em.VariablesBound[j][1] - self.em.VariablesBound[j][0])
                                elif self.em.VariablesType[j] in ["bvar", "ivar"]:
                                    self.solutions[i][j] = np.int64(np.round(np.array(self.em.VariablesBound[j][0] + self.em.BestAgent[i,self.em.VariablesSpread[j][0]:self.em.VariablesSpread[j][1]] * (self.em.VariablesBound[j][1] - self.em.VariablesBound[j][0]))))
                                elif self.em.VariablesType[j] in ["svar"]:
                                    self.solutions[i][j] = np.argsort(self.em.VariablesBound[j][0] + self.em.BestAgent[i,self.em.VariablesSpread[j][0]:self.em.VariablesSpread[j][1]] * (self.em.VariablesBound[j][1] - self.em.VariablesBound[j][0]))
                
                                if self.em.VariablesDim[j] == 0:
                                    try:
                                        self.solutions[i][j] = self.solutions[i][j][0]
                                    except:
                                        pass
                                elif len(self.em.VariablesDim[j]) == 1:
                                    pass
                                else:
                                    self.solutions[i][j] = np.array(self.solutions[i][j]).reshape([len(element) if not isinstance(element, int) else element for element in fix_dims(self.em.VariablesDim[j])])
                                
                            for decoder in (getattr(self, "decoder", None), getattr(getattr(self, "em", None), "decoder", None)):
                                if decoder:
                                    _, output_features = get_in_out(decoder, self.solutions[i])
                                    self.solutions[i].update(to_plain_python(output_features))


                else:
                    if self.method != "sequential":
                        self.solutions = {}
                        if self.method not in ["heuristic"]:
                            for typ, var in self.em.features['variables'].keys():
                                if typ == 'evar':
                                    continue
                                val = self.em.get_numpy_var(var)
                                if isinstance(val, np.ndarray):
                                    if not np.all(np.isnan(val)):
                                        self.solutions[var] = val
                                elif val is not None:
                                    self.solutions[var] = val
                        else:
                            for j in self.em.VariablesDim.keys():
                                self.solutions[j] = self.em.get_numpy_var(j)
                            if self.decoder is not None and self._decoder_args_spec is not None:
                                _decoder_input = {}
                                for _arg in self._decoder_args_spec:
                                    if _arg in self.solutions:
                                        _decoder_input[_arg] = self.solutions[_arg]
                                if _decoder_input:
                                    try:
                                        _dec_result = to_plain_python(self.decoder(**_decoder_input))
                                        if isinstance(_dec_result, dict):
                                            self.solutions.update(_dec_result)
                                        elif _dec_result is not None:
                                            _values = (_dec_result
                                                       if isinstance(_dec_result, tuple)
                                                       else (_dec_result,))
                                            try:
                                                _names = get_return_names(self.decoder)
                                            except Exception:
                                                _names = []
                                            if _names and len(_names) == len(_values):
                                                # never overwrite a real solution
                                                # entry with a decoded one
                                                self.solutions.update(
                                                    {n: v for n, v in zip(_names, _values)
                                                     if n not in self.solutions})
                                            else:
                                                self.solutions['decoded'] = _dec_result
                                    except Exception:
                                        pass
                            for decoder in (getattr(getattr(self, "em", None), "decoder", None),):
                                if decoder and decoder is not self.decoder:
                                    try:
                                        _, output_features = get_in_out(decoder, self.solutions)
                                        self.solutions.update(to_plain_python(output_features))
                                    except Exception:
                                        pass

            if self.method in ["heuristic"] and hasattr(self.em, 'BestAgent') and self.em.BestAgent is not None:
                try:
                    _best = np.atleast_2d(self.em.BestAgent)
                    if _best.ndim == 3:
                        _best = _best[0]

                    _pts = (self.solutions if isinstance(self.solutions, list)
                            else None)
                    _runs = (min(len(_best), len(_pts)) if _pts is not None
                             else 1)
                    for _pi in range(_runs):
                        try:
                            _row = (_best[_pi:_pi + 1] if _pts is not None
                                    else _best)
                            _term_agent = ['active', _row, self.em.VariablesSpread, self.em.penalty_coefficient]
                            _term_model = model(method=self.method, name=self.name, interface=self.interface, agent=_term_agent, no_agents=len(_row), auto_linearize=self.auto_linearize)
                            _term_model._search_decoder = self.decoder
                            if self.em is not None:
                                _term_model._impl = self.em
                            _term_model = self.environment(_term_model, *self.args, **self.kwargs)
                            _dst = (_pts[_pi] if _pts is not None
                                    else self.solutions)
                            if not isinstance(_dst, dict):
                                continue
                            if hasattr(_term_model, 'features') and _term_model.features.get('_terms'):
                                _dst.update(_term_model.features['_terms'])
                            if hasattr(_term_model, 'features') and _term_model.features.get('_deferred_terms'):
                                for _dname, _dfn in _term_model.features['_deferred_terms'].items():
                                    try:
                                        _dval = _dfn(_term_model)
                                        _dst[_dname] = np.asarray(_dval).item() if (isinstance(_dval, np.ndarray) and _dval.size == 1) else np.asarray(_dval)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    pass

            elif (self.method not in ["heuristic", "sequential", "madm"]
                  and self.em is not None
                  and not (self.em.features.get('_deferred_terms')
                           or self.em.features.get('_terms'))):
                pass

            elif self.method not in ["heuristic", "sequential", "madm"] and self.em is not None:
                try:
                    _var_bound = self.em.features['variable_bound']
                    _var_dim = self.em.features['variable_dim']
                    _deferred = self.em.features.get('_deferred_terms', {})
                    _spread = {}
                    _idx = 0
                    for var in _var_dim:
                        _dim = _var_dim[var]
                        # same counting rules as
                        # operators.update_operators.count_variable so the
                        # rebuilt model's slices match this spread
                        if _dim == 0:
                            _size = 1
                        elif isinstance(_dim, (int, np.integer)):
                            _size = int(_dim)
                        elif isinstance(_dim, (set, frozenset)):
                            _size = len(_dim)
                        else:
                            _size = 1
                            for d in _dim:
                                _size *= len(d) if not isinstance(d, (int, np.integer)) else 1
                        _spread[var] = (_idx, _idx + _size)
                        _idx += _size
                    _vtype = self.em.features.get('variable_type') or {}

                    def _agent_from(_sol):
                        """Normalized decision vector for one solution dict."""
                        _a = np.zeros(_idx)
                        for var in _var_bound:
                            if (not isinstance(_sol, dict) or var not in _sol
                                    or var not in _spread):
                                continue
                            _lb, _ub = _var_bound[var]
                            _start, _end = _spread[var]

                            if _lb is None:
                                _lb = 0 if _vtype.get(var) == 'pvar' else -1e6
                            if _ub is None:
                                _ub = 1e6
                            _val = _sol[var]
                            try:
                                if isinstance(_val, dict):
                                    # set-dim variable: slots follow set order
                                    _flat = np.asarray(
                                        [_val[k] for k in _var_dim[var] if k in _val],
                                        dtype=float)
                                else:
                                    _flat = np.asarray(_val, dtype=float).ravel()
                            except Exception:
                                continue
                            if _flat.size == 0:
                                continue
                            _denom = float(_ub) - float(_lb)
                            if _denom == 0:
                                continue
                            if _flat.size == _end - _start:
                                _a[_start:_end] = (_flat - float(_lb)) / _denom
                            else:
                                # scalar or shape mismatch: broadcast one value
                                _a[_start:_end] = (float(_flat.flat[0]) - float(_lb)) / _denom
                        return _a

                    def _terms_into(_sol):
                        """Rebuild a term model for one solution and store the
                        m.term() values on it."""
                        if not isinstance(_sol, dict):
                            return
                        _tm = model(method='heuristic', name=self.name, interface='feloopy',
                                    agent=['active', np.atleast_2d(_agent_from(_sol)),
                                           _spread, 0],
                                    no_agents=1, auto_linearize=self.auto_linearize)
                        if self.em is not None:
                            # mirror the heuristic branch: deferred callables
                            # (m.term value=fn) read the solved values through _impl
                            _tm._impl = self.em
                        _tm = self.environment(_tm, *self.args, **self.kwargs)
                        _tf = getattr(_tm, 'features', None) or {}

                        _done = set(_tf.get('_terms') or {})
                        if _done:
                            _sol.update(_tf['_terms'])
                        _dfr_tm = _tf.get('_deferred_terms') or {}
                        for _dname, _dfn in _deferred.items():
                            if _dname in _done:
                                continue
                            try:
                                _dval = None
                                _fn = _dfr_tm.get(_dname)
                                if _fn is not None:
                                    try:
                                        _dval = _fn(_tm)
                                    except Exception:
                                        _dval = None
                                if _dval is None:
                                    _dval = _dfn(_tm)
                                _sol[_dname] = np.asarray(_dval).item() if (isinstance(_dval, np.ndarray) and _dval.size == 1) else np.asarray(_dval)
                            except Exception:
                                pass

                    # a multi-objective run holds one dict per Pareto point:
                    # each point gets its own term values, so m.get('y')
                    # returns one entry per point instead of nothing
                    _pts = (self.solutions if isinstance(self.solutions, list)
                            else [self.solutions])
                    for _pt in _pts:
                        try:
                            _terms_into(_pt)
                        except Exception:
                            pass
                except Exception:
                    pass

            elif self.method != "sequential":
                # a sequential run keeps the simulation results written
                # above (total_costs, stats, traces, ...): the MADM term
                # sweep below has nothing to read from the placeholder
                # model and would wipe them out
                values_list = [
                        'rv', 
                        'wv', 
                        'fwv', 
                        'dmrv', 
                        'dprv',
                        'rmcv',
                        'rpcv',
                        'dominated',
                        'concordance',
                        'discordance',
                        'kernel',
                        'dominance',
                        'dominance_s',
                        'dominance_w',
                        'global_concordance',
                        'credibility',
                        'rank_d',
                        'rank_a',
                        'classification'
                    ]
                self.solutions = {}
                for key in values_list:
                    try:
                        self.solutions[key] = self.em.get_numpy_var(key)
                    except:
                        pass
            
            if self.memorize and self.method!="madm":
                if self.number_of_objectives == 1:
                    self.data.update(self.solutions)
                else:
                    self.data.update({"pareto": self.solutions})

            if self.method != 'madm':
                if self.method == "sequential":
                    self.objective_values = np.array(
                        [[self.result.stats['mean_cost']]])
                    self.num_objective_values = 1
                    self.cpt = 0
                elif self.number_of_objectives == 1:
                    self.objective_values = np.array([[self.em.get_obj()]])
                    self.num_objective_values = 1
                    # For boost methods, CPT was already set from the
                    # decomposition result's runtime_total. Only use
                    # em.get_time() for non-boost exact solves, or when
                    # boost was requested but no decomposition ran
                    # (e.g. a simple LP with boost=True).
                    if not getattr(self, 'boost', False) or not hasattr(self, 'cpt'):
                        self.cpt = self.em.get_time()
                else:
                    if self.method == "heuristic":
                        self.objective_values = self.em.get_obj()
                        self.num_objective_values = self.objective_values.shape[0]
                        self.cpt = self.em.get_time()
                        if self.number_of_objectives > 1:
                            try:
                                _dir_mult = np.array([1 if d == 'max' else -1 for d in self.directions])
                                _flipped = np.atleast_2d(self.objective_values * _dir_mult)
                                if _flipped.shape[0] >= 2:
                                    corr = np.corrcoef(_flipped.T)
                                    _M = corr.shape[0]
                                    if _M > 1:
                                        _mask = ~np.eye(_M, dtype=bool)
                                        _mean_corr = np.mean(corr[_mask])
                                        self.conflict_metric = 0.0 if np.isnan(_mean_corr) else float((1 - _mean_corr) / 2)
                                    else:
                                        self.conflict_metric = 0.0
                                else:
                                    # fewer than two non-dominated points: correlation
                                    # over a single observation has zero degrees of
                                    # freedom, and np.corrcoef would warn (degrees of
                                    # freedom / divide by zero / invalid multiply)
                                    # before handing back the same NaN matrix anyway
                                    corr = np.full((_flipped.shape[1], _flipped.shape[1]), np.nan)
                                    self.conflict_metric = 0.0
                                self.conflict = corr
                                self.payoff = self.get_payoff()
                            except Exception:
                                self.conflict_metric = 0.0
                    else:
                        self.objective_values = self.result[0]
                        self.num_objective_values = self.objective_values.shape[0]
                        self.cpt = self.time_solve_end - self.time_solve_begin
            else:
                self.cpt = self.time_solve_end - self.time_solve_begin

            try:
                self.mgt = self.mgt - float(self.cpt)
            except (TypeError, ValueError, AttributeError):
                self.mgt = self.mgt

            if self.memorize:
                if self.method != 'madm':
                    self.data["obj"] = self.objective_values
                    self.data["cpt"] = self.cpt
                    self.data["mgt"] = self.mgt
                    self.data["healthy"] = (True if self.method == "sequential"
                                            else self.em.healthy())
                else:
                    self.data["cpt"] = self.time_solve_end - self.time_solve_begin
                    self.data["mgt"] = self.mgt
                    self.data["healthy"] = self.em.healthy()               

        else:
            self.cpt=0
            
        solving_complete = True

        if not verbose and self.method != "sequential":
            end_progress(success_message="Searched")

    def get(self, input=None):

        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be a string, got {type(input).__name__}")
        if hasattr(self, "solutions"):
            if isinstance(self.solutions, list):
                
                return [
                    item.get(input)
                    for item in self.solutions
                    if isinstance(item, dict) and item.get(input) is not None
                ]
            elif isinstance(self.solutions, dict):
                return self.solutions.get(input)
            else:
                raise TypeError(f"'solutions' must be a list or dict, got {type(self.solutions).__name__}")
        elif hasattr(self, "em") and hasattr(self.em, "get_tensor"):
            return self.em.get_tensor(input)
        else:
            raise AttributeError("'self' has neither 'solutions' nor a valid 'em.get_tensor' method")

    def get_obj(self):
        if getattr(self, 'boost', False) and hasattr(self, '_benders_result'):
            if getattr(self._benders_result, 'status', None) == 'direct':
                return self.em.get_obj()
            obj = getattr(self._benders_result, 'objective', None)
            if obj is not None and self.directions and self.directions[0] == 'max':
                return -obj
            return obj
        if getattr(self, 'boost', False) and hasattr(self, '_lagrangian_result'):
            return getattr(self._lagrangian_result, 'objective', None)
        if getattr(self, 'boost', False) and hasattr(self, '_cg_result'):
            return getattr(self._cg_result, 'objective', None)
        if getattr(self, 'boost', False) and hasattr(self, 'em'):
            return self.em.get_obj()
        if self.number_of_objectives==1:
            obj = self.em.get_obj()
            if hasattr(obj, 'ndim') and obj.ndim > 0:
                obj = obj.item() if obj.size == 1 else obj
            return obj
        else:
            if self.method=='heuristic':
                return self.em.get_obj()
            else:
                return self.result[0]

    def get_time(self):
        """
        Returns timing information in seconds.

        Returns
        -------
        dict
            {'mgt': float, 'cpt': float}
            mgt: model generation time in seconds
            cpt: computation (solve) time in seconds
        """
        return {'mgt': self.mgt, 'cpt': self.cpt}

    def get_pop(self):
        """
        Returns the final population from a heuristic run as a 2D numpy array
        in [0,1] normalized space.  Each row is one solution.

        Compatible with the `init` parameter of `search()`.

        For the 'feloopy' interface, returns the full population matrix.
        For other interfaces, returns the best agent reshaped as a single-row matrix.
        """
        return self.em.get_pop()

    def get_convergence(self):
        """
        Returns convergence history from a heuristic or decomposition-based run.

        For Benders/Lagrangian/column-generation (boost=True), returns
        bounds-per-iteration data.  For heuristic runs, returns
        population-level statistics.

        Returns
        -------
        dict - see keys below.
        """
        from .algorithms.exact.benders import BendersResult as _BR
        benders_res = getattr(self, '_benders_result', None)
        if isinstance(benders_res, _BR):
            lb = list(benders_res.lb_history) if benders_res.lb_history else []
            ub = list(benders_res.ub_history) if benders_res.ub_history else []
            # Benders stores everything in internal (min) sign.
            # Flip for max so convergence matches the user-facing objective.
            _is_min = self.directions and self.directions[0] == 'min'
            if not _is_min:
                lb = [-v for v in lb]
                ub = [-v for v in ub]
            gap_abs = [abs(u - l) for u, l in zip(ub, lb)]
            gap_rel = [
                abs(u - l) / abs(u) if abs(u) > 1e-15 else 0.0
                for u, l in zip(ub, lb)
            ]
            status_str = benders_res.status
            if hasattr(status_str, 'value'):
                status_str = status_str.value
            obj = benders_res.objective
            if not _is_min and obj is not None:
                obj = -obj
            return {
                'lb_history': lb,
                'ub_history': ub,
                'obj_history': list(benders_res.obj_history) if benders_res.obj_history else [],
                'gap_abs_history': gap_abs,
                'gap_rel_history': gap_rel,
                'iterations': benders_res.iterations,
                'status': status_str,
                'n_optimality_cuts': benders_res.n_optimality_cuts,
                'n_feasibility_cuts': benders_res.n_feasibility_cuts,
                'n_cuts': benders_res.n_cuts,
                'n_sol': benders_res.n_sol,
                'runtime_total': benders_res.runtime_total,
                'objective': obj,
            }
        # Column-generation / Lagrangian / branching boost runs: their
        # DecompositionResult carries per-iteration history and bounds.
        # Previously only _benders_result was handled, so these runs fell
        # through to the heuristic branch and returned None.
        for _attr in ('_cg_result', '_lagrangian_result', '_branching_result'):
            _res = getattr(self, _attr, None)
            if _res is None or not hasattr(_res, 'history'):
                continue
            _is_min = not (self.directions and self.directions[0] == 'max')
            _hist = list(getattr(_res, 'history', None) or [])
            _lb, _ub, _obj = [], [], []
            for _h in _hist:
                if not isinstance(_h, dict):
                    continue
                if _h.get('lower_bound') is not None:
                    _lb.append(float(_h['lower_bound']))
                if _h.get('upper_bound') is not None:
                    _ub.append(float(_h['upper_bound']))
                if _h.get('best_bound') is not None:
                    # branch-and-bound records one direction-dependent bound
                    (_lb if _is_min else _ub).append(float(_h['best_bound']))
                if _h.get('objective') is not None:
                    try:
                        _obj.append(float(_h['objective']))
                    except (TypeError, ValueError):
                        pass
            _fin_lb = getattr(_res, 'lower_bound', None)
            _fin_ub = getattr(_res, 'upper_bound', None)
            _bb = getattr(_res, 'best_bound', None)
            if _bb is not None:
                if _is_min and (_fin_lb is None or np.isneginf(_fin_lb)):
                    _fin_lb = float(_bb)
                elif not _is_min and (_fin_ub is None or np.isposinf(_fin_ub)):
                    _fin_ub = float(_bb)
            _gap_abs, _gap_rel = [], []
            for _l, _u in zip(_lb, _ub):
                if not (np.isfinite(_l) and np.isfinite(_u)):
                    _gap_abs.append(float('inf'))
                    _gap_rel.append(float('inf'))
                    continue
                _d = abs(_u - _l)
                _gap_abs.append(_d)
                _gap_rel.append(_d / abs(_u) if abs(_u) > 1e-15 else 0.0)
            _st = getattr(_res, 'status', None)
            if hasattr(_st, 'value'):
                _st = _st.value
            return {
                'lb_history': _lb,
                'ub_history': _ub,
                'obj_history': _obj,
                'gap_abs_history': _gap_abs,
                'gap_rel_history': _gap_rel,
                'iterations': getattr(_res, 'iterations', None),
                'status': _st,
                'lower_bound': _fin_lb,
                'upper_bound': _fin_ub,
                'runtime_total': getattr(_res, 'runtime_total', None),
                'objective': getattr(_res, 'objective', None),
                'n_columns_added': getattr(_res, 'n_columns_added', None),
            }
        is_min = self.directions and self.directions[0] == 'min'
        best_min = getattr(self, 'best_min', None)
        best_max = getattr(self, 'best_max', None)
        if best_min is None and best_max is None:
            return None
        return {
            'best_so_far': np.array(best_min) if is_min else np.array(best_max),
            'population_min': np.array(best_min) if best_min is not None else None,
            'population_max': np.array(best_max) if best_max is not None else None,
            'average': np.array(getattr(self, 'average', None)),
            'std': np.array(getattr(self, 'std', None)),
            'stagnation': getattr(self, 'stagnation', None),
            'convergence_rate': getattr(self, 'convergence_rate', None),
            'population_diversity': getattr(self, 'population_diversity', None),
            'directions': self.directions,
        }

    def get_ogr(self):
        """
        Returns optimality gap ratio from the solver.

        For multi-objective, returns the average OGR across all solves.
        For single-objective, returns the OGR from the single solve.

        Returns
        -------
        float or None
            OGR = |obj - bound| / max(|obj|, |bound|), or None if not available.
        """
        if self.number_of_objectives > 1 and hasattr(self, 'ogrs') and self.ogrs:
            vals = [v for v in self.ogrs if v is not None]
            return sum(vals) / len(vals) if vals else None
        return self.em.get_ogr()

    def get_dual(self, input, tensor=True):
        return self.em.get_dual(input, tensor=tensor)

    def get_slack(self, input, tensor=True):
        return self.em.get_slack(input, tensor=tensor)

    def get_iis(self):
        return self.em.get_iis()

    def _extract_solutions_for_sensitivity(self):
        healthy = self.em.healthy()
        if self.number_of_objectives == 1:
            if self.method != "heuristic":
                _out = {var: self.em.get_numpy_var(var) if healthy else None
                        for typ, var in self.em.features['variables'].keys()}
            else:
                _out = {j: self.em.get_numpy_var(j) if healthy else None
                        for j in self.em.VariablesDim.keys()}
            # m.term() values are collected into self.solutions by the
            # term-model rebuild but are never registered as variables;
            # carry them into the recorded scenario dict so the per-term
            # sensitivity columns have values to diff against the control.
            if healthy and isinstance(getattr(self, 'solutions', None), dict):
                try:
                    _tf = self.em.features or {}
                    _imm = _tf.get('_terms') or {}
                    _names = list(_imm.keys())
                    _names += [k for k in (_tf.get('_deferred_terms') or {})
                               if k not in _imm]
                    for _tn in _names:
                        if _tn in self.solutions:
                            _out[_tn] = self.solutions[_tn]
                except Exception:
                    pass
            return _out
        else:
            if self.method != "heuristic":
                return self.result[3] if healthy else None
            else:
                if not isinstance(self.solutions, list) or not self.solutions:
                    return {}
                return {i: dict(sol) for i, sol in enumerate(self.solutions)}

    def _extract_objectives_for_sensitivity(self):
        healthy = self.em.healthy()
        if self.number_of_objectives == 1:
            return np.array([[self.em.get_obj()]])[0][0] if healthy else None
        else:
            if self.method != "heuristic":
                return self.result[0] if healthy else np.array([[None] * len(self.directions)])
            else:
                return self.em.get_obj() if healthy else None

    def _extraction_time_for_sensitivity(self):
        if self.number_of_objectives == 1:
            return self.em.get_time()
        else:
            if self.method != "heuristic":
                return self.time_solve_end - self.time_solve_begin
            else:
                return self.em.get_time()

    @staticmethod
    def clear_linearization_cache(key=None, environment=None, method='exact'):
        """Clear the linearization cache.

        Parameters
        ----------
        key : str, optional
            If provided, only clear the cache entry for this key.
            If None, clear all cached entries.
        environment : callable, optional
            If provided, clear the cache entry for this environment function.
        method : str, optional
            Method to match (default 'exact').
        """
        _clear_linearization_cache(key=key, environment=environment, method=method)

    @staticmethod
    def cache_stats():
        """Return cache statistics: size, hits, misses, hit rate."""
        return _linearization_cache_stats()

    @staticmethod
    def _generate_scenarios(dataset, parameter_names, n_scenarios=5):
        import numpy as np
        scenarios = []
        for pname in parameter_names:
            base_val = dataset.data.get(pname)
            desc = dataset.descriptions.get(pname, '').lower() if hasattr(dataset, 'descriptions') else ''
            type_t = dataset.type_params.get(pname, '') if hasattr(dataset, 'type_params') else ''
            min_v = dataset.minimum_params.get(pname, None) if hasattr(dataset, 'minimum_params') else None
            max_v = dataset.maximum_params.get(pname, None) if hasattr(dataset, 'maximum_params') else None
            std_v = dataset.std_params.get(pname, None) if hasattr(dataset, 'std_params') else None

            if base_val is None:
                scenarios.append([base_val])
                continue

            is_array = isinstance(base_val, (list, tuple)) or (hasattr(base_val, 'shape') and hasattr(base_val, 'ndim') and base_val.ndim > 0)

            if is_array:
                arr = np.asarray(base_val, dtype=float)
                
                is_coordinates = False
                if arr.ndim == 2 and arr.shape[1] == 2:
                    is_coordinates = True
                elif arr.ndim == 1 and arr.size > 0:
                    flat = np.asarray(base_val, dtype=object).flatten()
                    if len(flat) > 0 and hasattr(flat[0], '__len__') and len(flat[0]) == 2:
                        is_coordinates = True
                
                if is_coordinates:
                    if arr.ndim == 1:
                        arr_2d = np.array([list(x) if hasattr(x, '__len__') else [x, 0] for x in base_val], dtype=float)
                    else:
                        arr_2d = arr.copy()
                    
                    centroid = np.mean(arr_2d, axis=0)
                    
                    # Detect if these are lat/lon coordinates
                    _pname_lower_c = pname.lower().strip()
                    _is_geo_coord = any(kw in _pname_lower_c for kw in (
                        'lat', 'lon', 'lng', 'coordinate', 'coord', 'location',
                        'position', 'geo', 'point'))
                    if _is_geo_coord or (arr_2d.shape[0] > 0 and
                        np.all(arr_2d[:, 0] >= -90) and np.all(arr_2d[:, 0] <= 90) and
                        np.all(arr_2d[:, 1] >= -180) and np.all(arr_2d[:, 1] <= 180)):
                        _is_geo_coord = True
                    
                    arr_scenarios = []
                    
                    tight = centroid + (arr_2d - centroid) * 0.25
                    arr_scenarios.append(tight)
                    
                    shrink = centroid + (arr_2d - centroid) * 0.5
                    arr_scenarios.append(shrink)
                    
                    arr_scenarios.append(arr_2d)
                    
                    expand = centroid + (arr_2d - centroid) * 1.5
                    arr_scenarios.append(expand)
                    
                    wide = centroid + (arr_2d - centroid) * 2.0
                    arr_scenarios.append(wide)
                    
                    if _is_geo_coord:
                        for _si in range(len(arr_scenarios)):
                            arr_scenarios[_si] = np.column_stack([
                                np.clip(arr_scenarios[_si][:, 0], -90, 90),
                                np.clip(arr_scenarios[_si][:, 1], -180, 180)])
                    
                    if isinstance(base_val, list):
                        arr_scenarios = [s.tolist() for s in arr_scenarios]
                    elif isinstance(base_val, tuple):
                        arr_scenarios = [tuple(map(tuple, s.tolist())) for s in arr_scenarios]
                    
                    scenarios.append(arr_scenarios)
                    continue
                arr = np.asarray(base_val, dtype=float)
                category = 'default'
                _cat_rules = [
                    (['demand', 'requirement', 'd_req'], 'demand'),
                    (['cost', 'price', 'fee', 'expense', 'setup', 'ordering'], 'cost'),
                    (['hold', 'inventory', 'storage'], 'holding_cost'),
                    (['cap', 'capacity', 'limit', 'maximum'], 'capacity'),
                    (['time', 'duration', 'lead', 'period', 'horizon'], 'time'),
                    (['prob', 'probability', 'chance', 'likelihood'], 'probability'),
                    (['weight', 'wt'], 'weight'),
                    (['revenue', 'return', 'income', 'profit'], 'revenue'),
                    (['budget', 'fund', 'resource'], 'budget'),
                    (['loss', 'penalty', 'shortage'], 'penalty'),
                    (['qty', 'quantity', 'amount', 'volume'], 'quantity'),
                    (['rate', 'ratio', 'speed'], 'rate'),
                ]
                for keywords, cat in _cat_rules:
                    for kw in keywords:
                        if kw in desc or pname.lower() == kw or pname.lower().endswith('_' + kw):
                            category = cat
                            break
                    if category != 'default':
                        break

                arr_mean = float(np.mean(arr))
                arr_std = float(np.std(arr)) if arr.size > 1 else 0.0
                is_positive = 'ℝ⁺' in str(type_t) or 'ℕ' in str(type_t) or 'ℤ⁺' in str(type_t)
                # Detect ratio: probability category, type in [0,1], or values all in [0,1]
                _pname_lower_arr = pname.lower().strip()
                _name_implies_ratio = any(kw in _pname_lower_arr for kw in (
                    'ratio', 'rate', 'probability', 'prob', 'fraction', 'proportion',
                    'share', 'utilization', 'util', 'efficiency', 'yield', 'dropout',
                    'learning_rate', 'lr'))
                is_ratio = (category == 'probability' or '[0, 1]' in str(type_t)
                            or _name_implies_ratio
                            or (0 <= arr.min() and arr.max() <= 1 and arr_mean > 0))
                is_integer = 'ℤ' in str(type_t) or 'ℕ' in str(type_t) or 'ℤ⁺' in str(type_t)

                if is_ratio:
                    mean_shifts = [-0.3, -0.15, 0.0, 0.15, 0.3]
                elif arr_std > 0:
                    mean_shifts = [-2.0, -1.0, 0.0, 1.0, 2.0]
                else:
                    mean_shifts = [-0.5, -0.25, 0.0, 0.25, 0.5]

                arr_scenarios = []
                for shift in mean_shifts:
                    if is_ratio:
                        new_mean = arr_mean + shift
                        new_mean = max(0.0, min(1.0, new_mean))
                    elif arr_std > 0:
                        new_mean = arr_mean + shift * arr_std
                    else:
                        new_mean = arr_mean * (1 + shift) if arr_mean != 0 else shift

                    if arr_mean != 0:
                        scale = new_mean / arr_mean
                    else:
                        scale = 1.0 if new_mean == 0 else (new_mean if arr_mean == 0 else 1.0)

                    scaled = arr * scale

                    if is_positive or is_ratio:
                        scaled = np.maximum(0, scaled)
                    if is_ratio:
                        scaled = np.minimum(1, scaled)
                    if is_integer:
                        scaled = np.round(scaled).astype(int)

                    if isinstance(base_val, list):
                        arr_scenarios.append(scaled.tolist())
                    elif isinstance(base_val, tuple):
                        arr_scenarios.append(tuple(scaled.tolist()))
                    else:
                        arr_scenarios.append(scaled)

                base_idx = mean_shifts.index(0.0) if 0.0 in mean_shifts else len(mean_shifts) // 2
                arr_scenarios.insert(0, arr_scenarios.pop(base_idx))
                scenarios.append(arr_scenarios)
                continue

            if not isinstance(base_val, (int, float, np.integer, np.floating)):
                scenarios.append([base_val])
                continue

            base = float(base_val)
            is_integer = isinstance(base_val, (int, np.integer)) or 'ℤ' in str(type_t) or 'ℕ' in str(type_t)
            is_positive = 'ℝ⁺' in str(type_t) or 'ℕ' in str(type_t) or 'ℤ⁺' in str(type_t)
            is_binary = '𝔹' in str(type_t)
            min_val = float(min_v) if min_v is not None and not isinstance(min_v, str) else None
            max_val = float(max_v) if max_v is not None and not isinstance(max_v, str) else None

            # Detect ratio parameter: value in [0,1] with type indicating bounded ratio
            is_ratio = ('[0, 1]' in str(type_t) or 'ℝ⁺ ∩ [0, 1]' in str(type_t)
                        or (0 <= base <= 1 and pname.lower() in ('ratio', 'rate', 'probability',
                            'prob', 'fraction', 'proportion', 'share', 'utilization', 'util',
                            'efficiency', 'yield', 'dropout', 'learning_rate', 'lr')))
            # Also detect by value range when type is positive real and values sit in [0,1]
            if not is_ratio and is_positive and 0 <= base <= 1:
                if min_val is not None and max_val is not None and 0 <= min_val and max_val <= 1:
                    is_ratio = True
                elif not is_integer:
                    is_ratio = True

            # Detect latitude/longitude parameters
            _pname_lower = pname.lower().strip()
            _is_lat = any(kw in _pname_lower for kw in ('lat', 'latitude'))
            _is_lon = any(kw in _pname_lower for kw in ('lon', 'lng', 'longitude'))
            _is_coord = _is_lat or _is_lon or any(kw in _pname_lower for kw in ('x_coord', 'y_coord', 'pos_x', 'pos_y', 'position'))

            if is_binary:
                scenarios.append([0, 1])
                continue

            category = 'default'
            _cat_rules = [
                (['demand', 'requirement', 'd_req'], 'demand'),
                (['cost', 'price', 'fee', 'expense', 'setup', 'ordering'], 'cost'),
                (['hold', 'inventory', 'storage'], 'holding_cost'),
                (['cap', 'capacity', 'limit', 'maximum'], 'capacity'),
                (['time', 'duration', 'lead', 'period', 'horizon'], 'time'),
                (['prob', 'probability', 'chance', 'likelihood'], 'probability'),
                (['weight', 'wt'], 'weight'),
                (['revenue', 'return', 'income', 'profit'], 'revenue'),
                (['budget', 'fund', 'resource'], 'budget'),
                (['loss', 'penalty', 'shortage'], 'penalty'),
                (['qty', 'quantity', 'amount', 'volume'], 'quantity'),
                (['rate', 'ratio', 'speed'], 'rate'),
            ]
            for keywords, cat in _cat_rules:
                for kw in keywords:
                    if kw in desc or pname.lower() == kw or pname.lower().endswith('_' + kw):
                        category = cat
                        break
                if category != 'default':
                    break

            std_val = float(std_v) if std_v is not None and not isinstance(std_v, str) else 0.0

            # --- Latitude: perturb in degrees with geographic bounds ---
            if _is_lat:
                lat_min = max(-90.0, min_val if min_val is not None else -90.0)
                lat_max = min(90.0, max_val if max_val is not None else 90.0)
                delta = max(0.5, (lat_max - lat_min) * 0.1)
                vals = [base - 2*delta, base - delta, base, base + delta, base + 2*delta]
                vals = [max(lat_min, min(lat_max, v)) for v in vals]
                vals = sorted(set(round(v, 4) for v in vals))
                if base in vals:
                    vals.remove(base)
                    vals.insert(len(vals) // 2, base)
                scenarios.append(vals)
                continue

            # --- Longitude: perturb in degrees with geographic bounds ---
            if _is_lon:
                lon_min = max(-180.0, min_val if min_val is not None else -180.0)
                lon_max = min(180.0, max_val if max_val is not None else 180.0)
                delta = max(0.5, (lon_max - lon_min) * 0.1)
                vals = [base - 2*delta, base - delta, base, base + delta, base + 2*delta]
                vals = [max(lon_min, min(lon_max, v)) for v in vals]
                vals = sorted(set(round(v, 4) for v in vals))
                if base in vals:
                    vals.remove(base)
                    vals.insert(len(vals) // 2, base)
                scenarios.append(vals)
                continue

            # --- Ratio [0,1]: use absolute offsets that stay within [0,1] ---
            if is_ratio:
                step = 0.05
                if base > 0.5:
                    vals = [max(0.0, base - 4*step), max(0.0, base - 2*step),
                            base,
                            min(1.0, base + 2*step), min(1.0, base + 4*step)]
                elif base < 0.5:
                    vals = [max(0.0, base - 4*step), max(0.0, base - 2*step),
                            base,
                            min(1.0, base + 2*step), min(1.0, base + 4*step)]
                else:
                    vals = [0.1, 0.25, 0.5, 0.75, 0.9]
                vals = sorted(set(round(v, 4) for v in vals))
                if base in vals:
                    vals.remove(base)
                    vals.insert(len(vals) // 2, base)
                scenarios.append(vals)
                continue

            if category == 'probability':
                vals = [0.05, 0.25, 0.5, 0.75, 0.95]
            elif category in ('cost', 'holding_cost', 'penalty', 'revenue'):
                if base == 0:
                    vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
                else:
                    pcts = [-0.5, -0.25, 0.0, 0.25, 0.5]
                    vals = [base * (1 + p) for p in pcts]
            elif category in ('demand', 'quantity', 'capacity', 'budget'):
                if base == 0:
                    vals = [0.5, 0.75, 1.0, 1.25, 1.5]
                else:
                    pcts = [0.5, 0.75, 1.0, 1.25, 1.5]
                    vals = [base * p for p in pcts]
            elif category == 'time':
                if std_val > 0:
                    vals = [max(0, base - 2 * std_val), max(0, base - std_val), base, base + std_val, base + 2 * std_val]
                elif base == 0:
                    vals = [0.5, 1.0, 2.0, 3.0, 5.0]
                else:
                    pcts = [0.5, 0.75, 1.0, 1.25, 1.5]
                    vals = [base * p for p in pcts]
            elif category == 'weight':
                vals = [0.1, 0.25, 0.5, 0.75, 1.0]
            elif category == 'rate':
                if base == 0:
                    vals = [0.01, 0.05, 0.1, 0.2, 0.5]
                else:
                    pcts = [0.5, 0.75, 1.0, 1.25, 1.5]
                    vals = [base * p for p in pcts]
            else:
                if std_val > 0:
                    vals = [base - 2 * std_val, base - std_val, base, base + std_val, base + 2 * std_val]
                elif base == 0:
                    vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
                elif min_val is not None and max_val is not None and max_val > min_val:
                    span = max_val - min_val
                    vals = [min_val, min_val + 0.25 * span, (min_val + max_val) / 2, max_val - 0.25 * span, max_val]
                elif is_integer and abs(base) <= 10:
                    vals = [base - 2, base - 1, base, base + 1, base + 2]
                else:
                    pcts = [-0.5, -0.25, 0.0, 0.25, 0.5]
                    vals = [base * (1 + p) for p in pcts]

            if is_positive:
                vals = [max(0, v) for v in vals]
            if is_integer:
                vals = [int(round(v)) for v in vals]

            vals = sorted(set(vals))
            if base in vals:
                vals.remove(base)
                # keep the baseline scenario exactly type-compatible with the
                # original value: `base = float(base_val)` above would reinsert
                # 3.0 for an int param, and models using it as an array index
                # (e.g. alpha_bar[i-(alpha+1), ...]) then crash with a float
                # index error during the scenario rebuild
                _reinsert = int(round(base)) if is_integer else base
                vals.insert(len(vals) // 2, _reinsert)

            scenarios.append(vals)

        return scenarios

    def sensitivity(self, dataset, parameter_names, parameter_values, environment=None, control_scenario=0):

        from .operators.metrics import compute_similarity
        import copy

        self.sensitivity_parameter_names = parameter_names
        self.sensitivity_parameter_values = parameter_values

        if not self.em.healthy():
            return None

        if len(parameter_values) != len(parameter_names):
            raise ValueError(
                "Number of parameter names and values do not match. "
                "Use key_params=['a','b'] and scenarios=[[values_a],[values_b]]."
            )

        self.sensitivity_analyzed = True
        if environment is None:
            environment = self.environment

        result_dataset = data_toolkit(key=0, measure=False)
        data_keys = [
            "sensitivity_values",
            "sensitivity_of_health_to",
            "sensitivity_of_cpt_to",
            "sensitivity_of_objectives_to",
            "sensitivity_of_solutions_to",
            "sensitivity_of_ogr_to",
        ]
        if self.key_vars:
            data_keys.append("sensitivity_of_vars_to")

        self.sensitivity_begin_timer = timeit.default_timer()

        _orig_objective_values = copy.deepcopy(self.objective_values) if self.objective_values is not None else None
        _orig_solutions = copy.deepcopy(self.solutions) if self.solutions is not None else None
        _orig_em = self.em
        _orig_cpt = self.cpt
        _orig_mgt = self.mgt
        _orig_data = copy.deepcopy(self.data) if hasattr(self, 'data') and self.data is not None else None

        # Boost/decomposition results from the search run must not leak into
        # scenario re-solves: _run_impl copies objective values from an
        # existing _cg/_benders/..._result, which would record the previous
        # run's objective as if it belonged to the scenario.  Snapshot them
        # so get_convergence() keeps returning the search history afterwards.
        _MISSING = object()
        _boost_attrs = ('_benders_result', '_cg_result', '_lagrangian_result',
                        '_branching_result')
        _orig_boost_attrs = {a: getattr(self, a, _MISSING) for a in _boost_attrs}

        def _clear_stale_results():
            for _a in _boost_attrs:
                try:
                    delattr(self, _a)
                except AttributeError:
                    pass
            self.objective_values = None

        from .operators.data_handler import rewrap_scenario as _rewrap

        for pname, pvalues in zip(parameter_names, parameter_values):
            update_progress(f"Analyzing the impact of {pname}")
            for key in data_keys:
                result_dataset.store(f"{key}_{pname}", [])

            original_value = copy.deepcopy(dataset.data[pname])

            for pvalue in pvalues:
                dataset.data[pname] = _rewrap(original_value, pvalue)
                result_dataset.data[f"sensitivity_values_{pname}"].append(pvalue)

                _scenario_failed = False
                try:
                    with suppress_output():
                        _clear_stale_results()
                        self.create_env(environment, verbose=True)
                        self.run(verbose=True)
                except Exception:
                    _scenario_failed = True

                # tri-state health: None = build/run raised (FAIL),
                # False = solver reports infeasible (INFEAS), True = healthy.
                if _scenario_failed:
                    _health = None
                else:
                    try:
                        _health = self.em.healthy()
                    except Exception:
                        _health = None
                _usable = _health is True
                result_dataset.data[f"sensitivity_of_health_to_{pname}"].append(_health)
                result_dataset.data[f"sensitivity_of_cpt_to_{pname}"].append(
                    self._extraction_time_for_sensitivity() if _usable else None)
                if not _usable:
                    result_dataset.data[f"sensitivity_of_objectives_to_{pname}"].append(None)
                    result_dataset.data[f"sensitivity_of_solutions_to_{pname}"].append({})
                    result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(None)
                    if self.key_vars:
                        result_dataset.data[f"sensitivity_of_vars_to_{pname}"].append(
                            {vk: None for vk in self.key_vars})
                else:
                    result_dataset.data[f"sensitivity_of_objectives_to_{pname}"].append(
                        self._extract_objectives_for_sensitivity())
                    result_dataset.data[f"sensitivity_of_solutions_to_{pname}"].append(
                        self._extract_solutions_for_sensitivity())
                    try:
                        ogr_val = self.em.get_ogr()
                        result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(ogr_val)
                    except Exception:
                        result_dataset.data[f"sensitivity_of_ogr_to_{pname}"].append(None)
                    if self.key_vars:
                        _var_vals = {}
                        for vk in self.key_vars:
                            try:
                                _var_vals[vk] = self.em.get_numpy_var(vk)
                            except Exception:
                                try:
                                    _var_vals[vk] = self.solutions.get(vk) if isinstance(self.solutions, dict) else None
                                except Exception:
                                    _var_vals[vk] = None
                        result_dataset.data[f"sensitivity_of_vars_to_{pname}"].append(_var_vals)

            dataset.data[pname] = original_value

        self.sensitivity_end_timer = timeit.default_timer()

        self.objective_values = _orig_objective_values
        self.solutions = _orig_solutions
        self.em = _orig_em
        self.cpt = _orig_cpt
        self.mgt = _orig_mgt
        if _orig_data is not None:
            self.data = _orig_data
        for _a in _boost_attrs:
            _v = _orig_boost_attrs.get(_a, _MISSING)
            if _v is _MISSING:
                try:
                    delattr(self, _a)
                except AttributeError:
                    pass
            else:
                setattr(self, _a, _v)

        if self.number_of_objectives == 1:
            for pname in parameter_names:
                result_dataset.data[f"sensitivity_of_similarity_to_{pname}"] = compute_similarity(
                    result_dataset.data[f"sensitivity_of_solutions_to_{pname}"],
                    control_scenario_id=control_scenario,
                )
        else:
            for pname in parameter_names:
                _sols_raw = result_dataset.data[f"sensitivity_of_solutions_to_{pname}"]
                _sols_flat = []
                for _sol in _sols_raw:
                    if isinstance(_sol, dict):
                        if self.method == "heuristic":
                            _flat = {}
                            for _pareto_idx, _var_dict in _sol.items():
                                if isinstance(_var_dict, dict):
                                    _flat.update(_var_dict)
                            _sols_flat.append(_flat)
                        else:
                            _sols_flat.append(_sol)
                    else:
                        _sols_flat.append(_sol if isinstance(_sol, dict) else {})
                result_dataset.data[f"sensitivity_of_similarity_to_{pname}"] = compute_similarity(
                    _sols_flat,
                    control_scenario_id=control_scenario,
                )

        self.sensitivity_data = copy.deepcopy(result_dataset.data)
        _control_indices = {}
        for pname, pvalues in zip(parameter_names, parameter_values):
            _original_value = dataset.data.get(pname)
            _ctrl_idx = control_scenario
            for _ci, _cv in enumerate(pvalues):
                try:
                    if _cv == _original_value or (hasattr(_cv, '__float__') and hasattr(_original_value, '__float__') and abs(float(_cv) - float(_original_value)) < 1e-10):
                        _ctrl_idx = _ci
                        break
                except Exception:
                    pass
            _control_indices[pname] = _ctrl_idx
        self.sensitivity_control_indices = _control_indices
        return self.sensitivity_data
        
    def is_value_unreliable(self, data, bounds, features, vartype):
        if 'variables' not in features:
            return False

        var_names = []
        for key in features['variables']:
            if isinstance(key, tuple) and len(key) == 2 and key[0] == vartype:
                var_names.append(key[1])

        if not var_names:
            return False

        def flatten_value(v):
            if isinstance(v, (list, tuple)):
                for item in v:
                    yield item
            elif hasattr(v, "flatten"):
                try:
                    for item in v.flatten():
                        yield item
                except Exception:
                    yield v
            else:
                yield v

        if vartype == "bvar":
            condition = lambda val: (self.epsilon_value < val < 1 - self.epsilon_value) or val < 0 or val > 1
        elif vartype in ("ivar", "pvar", "fvar"):
            condition = lambda val: val < bounds[0] or val > bounds[1]
        else:
            return False

        def check_data(item):
            if not isinstance(item, dict):
                return False
            for k, v in item.items():
                if k in var_names and k not in ["_z"]:
                    for val in flatten_value(v):
                        if condition(val):
                            return True
            return False

        try:
            if isinstance(data, dict):
                return check_data(data)
            elif isinstance(data, list):
                return any(check_data(d) for d in data)
            return False
        except Exception:
            return False

    def is_value_impresice(self, data, bounds, features, vartype):
        if 'variables' not in features:
            return False

        var_names = []
        for key in features['variables']:
            if isinstance(key, tuple) and len(key) == 2 and key[0] == vartype:
                var_names.append(key[1])

        if not var_names:
            return False

        def flatten_value(v):
            if isinstance(v, (list, tuple)):
                for item in v:
                    yield item
            elif hasattr(v, "flatten"):
                try:
                    for item in v.flatten():
                        yield item
                except Exception:
                    yield v
            else:
                yield v

        if vartype == "bvar":
            def check_data(item):
                if not isinstance(item, dict):
                    return False
                for k, v in item.items():
                    if k in var_names:
                        for val in flatten_value(v):
                            if val not in (0, 1):
                                return True
                return False
        elif vartype == "ivar":
            def check_data(item):
                if not isinstance(item, dict):
                    return False
                for k, v in item.items():
                    if k in var_names:
                        for val in flatten_value(v):
                            if abs(val - round(val)) > self.epsilon_value:
                                return True
                return False
        else:
            return False

        try:
            if isinstance(data, dict):
                return check_data(data)
            elif isinstance(data, list):
                return any(check_data(d) for d in data)
            return False
        except Exception:
            return False

    def benchmark(self, environment=None, algorithms=None, repeat=1, show_report=False):

        if environment is None:
            environment = self.environment        
        
        if algorithms is None or algorithms == "all":
            if self.method=="exact":
                algorithms=EXACT_ALGORITHMS
            if self.method=="heuristic":
                algorithms=HEURISTIC_ALGORITHMS
        
        _CRASH_ISOLATED = {
            ('ortools', 'gurobi'),
        }

        _safe = [a for a in algorithms if tuple(a) not in _CRASH_ISOLATED]
        _crash = [a for a in algorithms if tuple(a) in _CRASH_ISOLATED]

        columns = ['time_ave', 'time_std', 'time_min', 'time_max', 'obj_ave', 'obj_std', 'obj_min', 'obj_max', 'interface', 'solver']
        df_rows = []
        counter = 0
        _orig_interface = self.interface
        _orig_solver = self.solver
        _orig_em = self.em

        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_stdout_fd = os.dup(1)
        _saved_stderr_fd = os.dup(2)
        _saved_sys_stdout = sys.stdout
        _saved_sys_stderr = sys.stderr
        _devnull_file = os.fdopen(_devnull_fd, 'w', closefd=False)

        import warnings as _warnings

        for interface, solver in _safe:
            if counter > 0:
                end_progress(success_message="")
            start_progress(message=f"Benchmarking {interface}-{solver}...", spinner="dots", color="cyan")
            self.interface = interface
            self.solver = solver

            os.dup2(_devnull_fd, 1)
            os.dup2(_devnull_fd, 2)
            sys.stdout = _devnull_file
            sys.stderr = _devnull_file
            _warnings.filterwarnings("ignore")
            try:
                row = {'interface': interface, 'solver': solver}
                for _ in range(repeat):
                    self.create_env(environment, verbose=True)
                    with suppress(Exception):
                        self.run(verbose=True, show_solver_log=False)
                    obj = self.em.get_obj()
                    t = self.em.get_time()
                    row['time_ave'] = t
                    row['obj_ave'] = obj
                df_rows.append(row)
            except Exception:
                df_rows.append({'interface': interface, 'solver': solver})
            finally:
                os.dup2(_saved_stdout_fd, 1)
                os.dup2(_saved_stderr_fd, 2)
                sys.stdout = _saved_sys_stdout
                sys.stderr = _saved_sys_stderr
                _warnings.resetwarnings()
            counter += 1

        os.close(_saved_stdout_fd)
        os.close(_saved_stderr_fd)
        os.close(_devnull_fd)
        self.interface = _orig_interface
        self.solver = _orig_solver
        self.em = _orig_em

        if self.track_history:
            self.lb_record = []
            self.ub_record = []
            self.ave_record = []
            self.std_record = []
            self._hist_idx = 0
            self._gen_count = 0
            self._infeasible_counts = []
            self._adaptive_boosted = False
            self._initial_penalty = 0
            if self.penalty_coefficient == 0:
                self._penalty_needs_init = True

        if _crash:
            import subprocess as _sp, pickle as _pkl, tempfile as _tmp, json as _json
            try:
                import cloudpickle as _cpkl
            except ImportError:
                _cpkl = _pkl

            _runner_code = (
                "import sys,os,json,pickle,warnings\n"
                "os.environ['MPLBACKEND']='Agg'\n"
                "warnings.filterwarnings('ignore')\n"
                "from contextlib import suppress\n"
                "_real=sys.__stdout__\n"
                "_devnull=open(os.devnull,'w')\n"
                "sys.stdout=_devnull\n"
                "sys.stderr=_devnull\n"
                "import feloopy as flp\n"
                "_data=pickle.loads(open(sys.argv[1],'rb').read())\n"
                "_env_fn=_data['env_fn']\n"
                "_kw=_data['kw']\n"
                "_algos=_data['algorithms']\n"
                "_repeat=_data.get('repeat',1)\n"
                "for _iface,_sol in _algos:\n"
                "    try:\n"
                "        m=flp.search(_env_fn,**_kw)\n"
                "        m.interface=_iface\n"
                "        m.solver=_sol\n"
                "        m.create_env(_env_fn,verbose=True)\n"
                "        for _r in range(_repeat):\n"
                "            with suppress(Exception):\n"
                "                m.run(verbose=True,show_solver_log=False)\n"
                "        print(json.dumps({'s':'ok','i':_iface,'v':_sol,'obj':m.em.get_obj(),'t':m.em.get_time()}),file=_real,flush=True)\n"
                "    except Exception as _e:\n"
                "        print(json.dumps({'s':'err','i':_iface,'v':_sol}),file=_real,flush=True)\n"
            )
            _runner_path = os.path.join(_tmp.gettempdir(), '_feloopy_bench_crash.py')
            with open(_runner_path, 'w') as _f:
                _f.write(_runner_code)

            _kw = dict(
                directions=self.directions,
                verbose=False,
                key_params=getattr(self, 'key_params', []),
                scenarios=getattr(self, 'scenarios', []),
                dataset=getattr(self, 'inputdata', None),
                options=self.options,
                method=self.method,
                name=self.name,
            )
            _payload_path = os.path.join(_tmp.gettempdir(), '_feloopy_bench_crash_payload.pkl')
            with open(_payload_path, 'wb') as _f:
                _cpkl.dump({
                    'env_fn': environment,
                    'kw': _kw,
                    'algorithms': _crash,
                    'repeat': repeat,
                }, _f)

            for ci, (iface, sol) in enumerate(_crash):
                if counter > 0:
                    end_progress(success_message="")
                start_progress(message=f"Benchmarking {iface}-{sol}...", spinner="dots", color="cyan")

            _proc = _sp.Popen(
                [sys.executable, _runner_path, _payload_path],
                stdout=_sp.PIPE, stderr=_sp.DEVNULL, text=True,
                creationflags=getattr(_sp, 'CREATE_NO_WINDOW', 0)
            )
            _crash_results = {}
            for line in _proc.stdout:
                try:
                    r = _json.loads(line.strip())
                    _crash_results[(r.get('i'), r.get('v'))] = r
                except Exception:
                    pass
            _proc.wait()

            for interface, solver in _crash:
                r = _crash_results.get((interface, solver), {'s': 'err'})
                row = {'interface': interface, 'solver': solver}
                if r.get('s') == 'ok':
                    row['time_ave'] = r.get('t')
                    row['obj_ave'] = r.get('obj')
                df_rows.append(row)
                counter += 1

            try:
                os.remove(_runner_path)
            except OSError:
                pass
            try:
                os.remove(_payload_path)
            except OSError:
                pass
        
        end_progress(success_message="Benchmarked")
        
        df = pl.DataFrame(df_rows)
        for col in ['time_ave', 'obj_ave']:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        df_sorted = df.drop_nulls(subset=['time_ave', 'obj_ave']).sort('time_ave')
        
        if show_report:
            print(df_sorted)

        self.ben_results = df_sorted

        return self.ben_results

    def report_decision(self, style=1, key_vars=[], show_elements=False, width=90, skip=False, hidden_variables=False):
        """Print decision variable values in a formatted table.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        key_vars : list of str, optional
            Variable names to display.  If empty, all variables are shown.
        show_elements : bool, optional
            If ``True``, display individual array elements (default ``False``).
        width : int, optional
            Table width in characters (default 90).
        skip : bool, optional
            If ``True``, skip printing (default ``False``).
        hidden_variables : bool, optional
            If ``True``, hide auto-linearization auxiliary variables (default ``False``).
        """
        ReportEngine(self).report_decision(style=style, key_vars=key_vars, show_elements=show_elements, width=width, skip=skip, hidden_variables=hidden_variables)

    def report_specs(self, style=1, width=90, skip_system_information=False):
        """Print system and solver specifications.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        skip_system_information : bool, optional
            If ``True``, omit OS/Python version details (default ``False``).
        """
        ReportEngine(self).report_specs(style=style, width=width, skip_system_information=skip_system_information)

    def report_model(self, style=1, width=90):
        """Print model statistics: variable counts by type, constraint count, etc.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        """
        ReportEngine(self).report_model(style=style, width=width)

    def report_formulation(self, style=1, width=90):
        """Print the mathematical formulation (variables, constraints, objectives).

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        """
        ReportEngine(self).report_formulation(style=style, width=width)

    def report_lp_insights(self, style=1, width=90):
        """Print LP-specific insights: duals, slacks, and reduced costs.

        Only available for LP/MILP problems solved via exact methods.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        """
        ReportEngine(self).report_lp_insights(style=style, width=width)

    def report_debug(self, style=1):
        """Print debug information including constraint feasibility.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        """
        ReportEngine(self).report_debug(style=style)

    def get_diagnostics(self):
        """Diagnostics results as a plain dict (nothing is printed).

        The programmatic counterpart of the ``Diagnostics`` box, with the
        same three sections and the same scope:

        ``batches`` : list of dict
            Constraint-batch impact records: ``name`` and ``n`` of the batch,
            the objective with and without it (``obj_full``/``obj_without``)
            plus their difference and percent (``obj_delta``/``obj_pct``),
            the same split per objective for multi-objective runs
            (``obj_full_means``/``obj_without_means``/``obj_full_pct_k``/
            ``obj_pct_k``), the solve times (``time_full``/``time_without``/
            ``time_pct``) and ``status`` (``OK``/``INFEAS``/``FAIL``).
            Internal ``_``-prefixed batches never reach this list.  Empty
            until a sensitivity sweep has run.
        ``lp_analysis`` : tuple or None
            ``None`` when the model is not a plain LP or has nothing worth
            reporting, otherwise ``(meaningful, n_constraints, at_lower,
            at_upper)``: ``meaningful`` lists ``(constraint, slack, dual)``
            for every constraint with a non-zero dual, the two other entries
            are ``(name, value)`` pairs of variables sitting on a bound.
        ``infeasibility`` : list or None
            ``None`` when the question does not apply (healthy, heuristic or
            multi-objective runs), otherwise the ``con:``/``batch:`` lines
            behind an infeasible model -- an empty list when nothing could be
            identified.  Probed once and cached, exactly like the box.

        Returns
        -------
        dict
            Always the three keys above; ``None`` marks a section that does
            not apply rather than a section that found nothing.
        """
        eng = ReportEngine(self)
        # A default report deferred the constraint-batch pass; this is the
        # diagnostics accessor, so run it now (it stays silent).
        try:
            eng._flush_pending_batch_impact(self, progress=False)
        except Exception:
            pass
        _batches = list(getattr(self, 'sensitivity_batch_data', None) or [])
        try:
            _lp = eng._collect_lp_analysis()
        except Exception:
            _lp = None
        if getattr(self, '_diagnostics_iis', None) is None:
            try:
                _iis = eng._resolve_infeasibility(self)
            except Exception:
                _iis = None
        else:
            _iis = self._diagnostics_iis
        return {'batches': _batches,
                'lp_analysis': _lp,
                'infeasibility': _iis}

    def report_performance(self, style=1, width=90, skip=False):
        """Print performance metrics: MGT, CPT, and PVR.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        skip : bool, optional
            If ``True``, skip printing (default ``False``).
        """
        ReportEngine(self).report_metrics(style=style, width=width, skip=skip)

    def report_data(self, style=1, width=90):
        """Print stored parameter data with statistics (type, size, min, max, avg, std).

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        """
        ReportEngine(self).report_data(style=style, width=width)

    def report_benchmark(self, width=90, style=1):
        """Print benchmark comparison results.

        Requires ``benchmark`` to have been set when creating the ``search``.

        Parameters
        ----------
        width : int, optional
            Table width in characters (default 90).
        style : int, optional
            Box-drawing style (default 1).
        """
        ReportEngine(self).report_benchmark(width=width, style=style)

    def report_sensitivity(self, width=90, style=1, skip=False, show_elements=False, hidden_variables=False):
        """Print sensitivity analysis results.

        Requires ``sensitivity`` (or ``key_params``/``scenarios``) to have been
        set when creating the ``search``.  For each parameter and scenario,
        shows health status, computation time, and objective values.

        Parameters
        ----------
        width : int, optional
            Table width in characters (default 90).
        style : int, optional
            Box-drawing style (default 1).
        skip : bool, optional
            If ``True``, skip printing (default ``False``).
        show_elements : bool, optional
            If ``True``, display individual variable elements (default ``False``).
        hidden_variables : bool, optional
            If ``True``, show hidden variables such as internal linearization
            and SOS2 helper variables (default ``False``).
        """
        ReportEngine(self).report_sensitivity(width=width, style=style, skip=skip, show_elements=show_elements, hidden_variables=hidden_variables)

    def get_impact(self, parameter_name):
        """Impact of one swept parameter as a plain dict (nothing is printed).

        The programmatic counterpart of that parameter's column of
        :meth:`report_sensitivity`.  The aggregate metrics come from the
        report's own ``ReportEngine.compute_impact``, so the printed footer
        and the queried numbers can never disagree.

        Parameters
        ----------
        parameter_name : str
            Name of a swept parameter; the swept set is
            ``m.sensitivity_parameter_names``.

        Returns
        -------
        dict
            ``parameter``, ``control_index``, ``values``, ``action``
            (one of ``raise``/``lower``/``none``/``mixed`` -- the word shown
            beside the parameter marker), ``metrics`` (Sen, Dec, Asy, Unc,
            Tim, Imp and the ``direction_effect`` glyph) and ``scenarios``
            (one dict per scenario with ``value``, ``health``,
            ``objective``, ``objective_means``, ``objective_pct``,
            ``cpt``, ``cpt_pct`` and ``solution``).

        Raises
        ------
        ValueError
            When ``parameter_name`` was never swept.
        """
        return ReportEngine(self).get_impact(parameter_name)

    def report_binding_constraints(self, style=1, width=90):
        """Print constraints with zero slack (binding) and their dual values.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        width : int, optional
            Table width in characters (default 90).
        """
        ReportEngine(self).report_binding_constraints(style=style, width=width)

    def report_status(self, style=1):
        """Print a compact one-line status (healthy / infeasible / error).

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        """
        ReportEngine(self).report_status(style=style)

    def clean_report(self,**kwargs):

        clear_console()
        self.report(**kwargs)
        
    def report(self, style=1, skip_system_information=True, show_elements=True, width=78, skip=False, full=False, save=None, copy_to_clipboard=False, hidden_variables=False, show_info=False, show_tensors=None, diagnostics=False):
        """Print the full formatted report.

        Combines specs, data, metrics, objectives, and decisions into a
        single boxed output.  Use ``skip_system_information=True`` (default)
        for a shorter output.

        Parameters
        ----------
        style : int, optional
            Box-drawing style (default 1).
        skip_system_information : bool, optional
            If ``True``, omit OS/Python details (default ``True``).
        show_elements : bool, optional
            If ``True``, show individual array elements (default ``False``).
        width : int, optional
            Table width in characters (default 78).
        skip : bool, optional
            If ``True``, skip printing (default ``False``).
        full : bool, optional
            If ``True``, include all sections (default ``False``).
        save : str, optional
            File path to save the report as text.
        copy_to_clipboard : bool, optional
            If ``True``, copy the report text to clipboard (default ``False``).
        hidden_variables : bool, optional
            If ``True``, hide auto-linearization auxiliary variables (default ``False``).
        show_tensors : bool, optional
            If ``True``, pack tensors instead of showing individual elements.
            Overrides ``show_elements`` with inverted logic (default ``None``).
        diagnostics : bool, optional
            If ``True``, run the Diagnostics section: constraint-batch impact
            (one extra solve per batch) and the LP analysis -- a dual/slack
            query per constraint plus the variables sitting on a bound
            (default ``False``).  An infeasible model is always explained
            through its conflicting constraints either way.
        markdown : bool, optional
            If ``True``, output the report in Markdown format (default ``False``).
        """
        if show_tensors is not None:
            show_elements = not show_tensors
        if self.method == "madm":
            self.em.mgt = self.mgt
            self.em.cpt = self.cpt
            self.em.report(style=style, save=save, width=width, show_elements=show_elements, hidden_variables=hidden_variables)
        else:
            ReportEngine(self).report_search(style=style, skip_system_information=skip_system_information, show_elements=show_elements, width=width, skip=skip, full=full, save=save, copy_to_clipboard=copy_to_clipboard, hidden_variables=hidden_variables, show_info=show_info, diagnostics=diagnostics)
        return self

    def save_io(self,name,extra=None):
        """Export inputs and outputs to a JSON file.

        Saves to ``results/data/<name>.json``.

        Parameters
        ----------
        name : str
            File name (without extension).
        extra : dict, optional
            Additional data to include in the export.
        """
        dt = data_toolkit(key=0)
        if type(self.inputdata)==dict:
            dt.data["inputs"] = self.inputdata
        else:
            dt.data["inputs"] = self.inputdata.data
        dt.data["outputs"] = self.solutions
        if extra: dt.data["extra"] = extra
        dt.save(name=name)

    def get_density(self):
        import numbers
        import numpy as np

        def _count(val, path):
            if isinstance(val, np.ndarray):
                return int(np.count_nonzero(val))
            if isinstance(val, numbers.Number) or isinstance(val, np.generic):
                return int(val != 0)
            if isinstance(val, dict):
                total = 0
                for subkey, subval in val.items():
                    total += _count(subval, path + [subkey])
                return total
            try:
                import xarray
                if isinstance(val, xarray.DataArray):
                    return int(np.count_nonzero(val.values))
            except ImportError:
                pass
            raise TypeError(
                f"Unsupported type at {'/'.join(path)}: {type(val).__name__}; "
                "expected ndarray, dict, or numeric scalar."
            )

        if not hasattr(self, "solutions") or not self.solutions:
            return "n/a"

        total_nonzeros = 0
        solutions = self.solutions
        if isinstance(solutions, list):
            for sol in solutions:
                if isinstance(sol, dict):
                    for key, value in sol.items():
                        total_nonzeros += _count(value, [key])
            total_nonzeros = total_nonzeros // len(solutions) if solutions else 0
        else:
            for key, value in solutions.items():
                total_nonzeros += _count(value, [key])
        return total_nonzeros

class parallel_search:
    """Run multiple optimization searches in parallel.

    Parameters
    ----------
    configurations : list[dict]
        Keyword-argument dicts, each passed to ``flp.search()``.  The model
        function is ``search``'s first positional parameter, so pass it as
        ``environment``: ``dict(environment=my_model, solver="highs")``.
    parallelization_method : str, ``"thread"`` or ``"process"``
        Backend for parallelism (default ``"thread"``).

        ``"process"`` requires the configurations to be picklable (define the
        model function at module level and keep the usual
        ``if __name__ == "__main__":`` guard).  Results come back detached
        from the solver: status, objective, solution values, ``get()`` and
        ``report()`` all work; live-solver operations (IIS, duals, re-solve)
        are not available.  Worker spawn problems (e.g. a missing
        ``if __name__ == "__main__":`` guard) are retried once with threads.
    max_workers : int, optional
        Maximum number of concurrent workers (default: number of
        configurations, at least 1).
    timeout : float, optional
        Overall deadline in seconds for collecting all results.  Searches
        still outstanding when it expires are recorded in ``errors`` as
        abandoned and ``run()`` returns without waiting for them; work
        already in flight is not killed.
    progress : bool
        Print an aggregate progress counter (default ``False``).  The
        per-search spinner is always suppressed here because spinner state
        is process-wide.

    Attributes
    ----------
    results : list
        One entry per configuration, ``None`` for failed searches.  Running
        the searches on first access.
    errors : dict[int, str]
        ``{index: traceback}`` for failed searches.  Running the searches on
        first access.

    Examples
    --------
    >>> configs = [
    ...     dict(environment=my_model, solver="highs"),
    ...     dict(environment=my_model, solver="scip"),
    ... ]
    >>> ps = flp.parallel_search(configs, parallelization_method="thread")
    >>> ps[0]  # result for first configuration
    >>> ps.results  # list of all results
    >>> ps.errors   # dict of {index: traceback} for failed searches
    """

    def __init__(self, configurations, parallelization_method="thread",
                 max_workers=None, timeout=None, progress=False):
        if parallelization_method not in ("thread", "process"):
            raise ValueError(
                "parallelization_method must be 'thread' or 'process', "
                f"got {parallelization_method!r}")
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout!r}")
        self.configurations = [dict(c) for c in configurations]
        self.method = parallelization_method
        if max_workers is None:
            max_workers = max(1, len(self.configurations))
        elif max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers!r}")
        self.max_workers = max_workers
        self.timeout = timeout
        self.progress = progress

        self._total = len(self.configurations)
        self._results = [None] * self._total
        self._errors = {}
        self._completed = 0
        self._done = False

    def _reset(self):
        self._results = [None] * self._total
        self._errors = {}
        self._completed = 0

    def _ensure_done(self):
        if not self._done:
            self.run()

    def __getitem__(self, index):
        self._ensure_done()
        return self._results[index]

    def __len__(self):
        return self._total

    def __iter__(self):
        self._ensure_done()
        return iter(self._results)

    def __repr__(self):
        status = "done" if self._done else "pending"
        return (
            f"parallel_search(configs={self._total}, "
            f"method='{self.method}', workers={self.max_workers}, "
            f"errors={len(self._errors)}, status={status})"
        )

    @property
    def results(self):
        self._ensure_done()
        return self._results

    @property
    def errors(self):
        self._ensure_done()
        return self._errors

    @staticmethod
    def _is_spawn_error(exc):
        text = str(exc)
        return any(s in text for s in (
            "freeze_support", "bootstrapping phase",
            "can't start new process", "'fork'"))

    def run(self):
        if self._done:
            return self._results

        self._reset()
        if not self._total:
            self._done = True
            return self._results

        executor_class = (
            concurrent.futures.ProcessPoolExecutor
            if self.method == "process"
            else concurrent.futures.ThreadPoolExecutor)

        try:
            self._execute(executor_class)
        except RuntimeError as e:
            if self.method == "process" and self._is_spawn_error(e):
                self._reset()
                self._execute(concurrent.futures.ThreadPoolExecutor)
            else:
                raise

        self._done = True
        return self._results

    def _execute(self, executor_class):
        executor = executor_class(max_workers=self.max_workers)
        timed_out = False
        try:
            if self.method == "process":
                self._preflight_configs()

            pending = {}
            for idx, config in enumerate(self.configurations):
                if idx in self._errors:
                    continue
                pending[executor.submit(self._run_single_search, config)] = idx

            try:
                for future in concurrent.futures.as_completed(
                        list(pending), timeout=self.timeout):
                    self._collect(pending.pop(future), future)
            except concurrent.futures.TimeoutError:
                timed_out = True

            if timed_out:
                for future, idx in pending.items():
                    future.cancel()
                    self._errors[idx] = (
                        f"timeout: no result within {self.timeout}s "
                        f"(search abandoned)")
                    self._completed += 1
                self._report_progress()
        finally:
            executor.shutdown(wait=not timed_out, cancel_futures=True)
            if self.progress and self._total:
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _preflight_configs(self):
        """Reject unpicklable configurations before the pool breaks."""
        import pickle
        for idx, config in enumerate(self.configurations):
            try:
                pickle.dumps((self._run_single_search, config))
            except Exception as e:
                self._errors[idx] = (
                    f"configuration {idx} is not picklable for "
                    f"parallelization_method='process': "
                    f"{type(e).__name__}: {e}")
                self._completed += 1

    def _collect(self, idx, future):
        try:
            self._results[idx] = future.result()
        except Exception:
            self._errors[idx] = self._format_error()
        self._completed += 1
        self._report_progress()

    def _format_error(self):
        import traceback
        text = traceback.format_exc().rstrip()
        if self.method == "process":
            low = text.lower()
            if "pickle" in low:
                text += (
                    "\n[parallel_search] the result could not be pickled on "
                    "its way back from the worker process; use "
                    "parallelization_method='thread' to skip the pickle.")
            elif "bootstrap" in low or "freeze_support" in low:
                text += (
                    "\n[parallel_search] the worker process could not "
                    "start; guard the calling script with "
                    "if __name__ == '__main__': or use "
                    "parallelization_method='thread'.")
        return text

    def _report_progress(self):
        if not self.progress or not self._total:
            return
        pct = self._completed / self._total * 100
        sys.stdout.write(
            f"\rProgress: {self._completed}/{self._total} ({pct:.0f}%)")
        sys.stdout.flush()

    @staticmethod
    def _run_single_search(config):
        with suppress_progress():
            return search(**config)

from .generators.model_generator import (
    feloop_model,
    copt_model,
    cplex_cp_model,
    cplex_model,
    cylp_model,
    cvxpy_model,
    gekko_model,
    gurobi_model,
    gams_model,
    linopy_model,
    mip_model,
    ortools_cp_model,
    ortools_model,
    picos_model,
    pulp_model,
    pyomo_model,
    pymprog_model,
    rsome_dro_model,
    rsome_ro_model,
    seeker_model,
    xpress_model,
    hexaly_model,
    mosek_model,
)