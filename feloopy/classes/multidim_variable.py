"""
Multi-dimensional variables module

This module facilitates the creation of various types of multi-dimensional variables.

Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

from typing import List, Optional, Union
import itertools as it
import numpy as np
from ..operators.fix_operators import fix_dims
from ..operators.update_operators import update_variable_features, _assert_unique_variable_name

    
class MultidimVariable:
    """Specifies the variable type."""

class MultidimVariableClass:
    """Class that provides methods to create multi-dimensional variables."""
    
    def fvar(
        self,
        name: str,
        dim: List[int] = 0,
        bound: List[Optional[float]] = [None, None]
    ) -> MultidimVariable:
        
        """
        Creates and returns a free variable.

        Parameters
        ----------
        name : str
            Name.
        dim : List[int], optional
            Dimensions. Default is 0 (scalar). N = vector of N.
        bound : List[Optional[float]], optional
            Lower and upper bounds. Default is [None, None].

        Returns
        -------
        MultidimVariable or _VarProxy
        """

        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _resolve_dim
            fdim = _resolve_dim(dim)
            bound = list(bound)
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'dp', 'dim': fdim, 'bounds': bound,
                'var_type': 'cont', 'proxy': proxy
            }
            return proxy

        bound = list(bound)
        dim = self.fix_ifneeded(dim)
        self.features = update_variable_features(name, dim, bound, 'free_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator
            self.features['variables'][("fvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'fvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim

            if self.features['interface_name'] in ['rsome_ro', 'rsome_dro', 'cvxpy']:
                if dim == 0:
                    if bound[0] is not None:
                        self.con(self.features['variables'][("fvar", name)] >= bound[0])
                    if bound[1] is not None:
                        self.con(self.features['variables'][("fvar", name)] <= bound[1])
                elif len(dim) == 1:
                    for i in dim[0]:
                        if bound[0] is not None:
                            self.con(self.features['variables'][("fvar", name)][i] >= bound[0])
                        if bound[1] is not None:
                            self.con(self.features['variables'][("fvar", name)][i] <= bound[1])
                else:
                    for i in it.product(*tuple(dim)):
                        if bound[0] is not None:
                            self.con(self.features['variables'][("fvar", name)][i] >= bound[0])
                        if bound[1] is not None:
                            self.con(self.features['variables'][("fvar", name)][i] <= bound[1])

            native_var = self.features['variables'][("fvar", name)]

            if self.features.get('auto_linearize'):
                from ..classes.linearization import LinearizationProxy
                if isinstance(native_var, dict):
                    return {k: LinearizationProxy(v, self, f"{name}[{k}]", lb=bound[0], ub=bound[1]) for k, v in native_var.items()}
                return LinearizationProxy(native_var, self, name, lb=bound[0], ub=bound[1])

            return native_var

        elif self.features['solution_method'] == 'heuristic':
            if any(b is None for b in bound):
                if self.features['agent_status'] == 'idle':
                    _override = self.features.get('_auto_bound_overrides', {}).get(name)
                    if _override:
                        if bound[0] is None: bound[0] = _override[0]
                        if bound[1] is None: bound[1] = _override[1]
                    else:
                        _ib = self.features.get('_inferred_bounds', {}).get(name)
                        if _ib:
                            if bound[0] is None: bound[0] = _ib[0]
                            if bound[1] is None: bound[1] = _ib[1]
                        else:
                            if bound[0] is None: bound[0] = -1e6
                            if bound[1] is None: bound[1] =  1e6
                    self.features.setdefault('_auto_bound_vars', []).append(name)
                else:
                    _ib = self.features.get('_inferred_bounds', {}).get(name)
                    if _ib:
                        if bound[0] is None: bound[0] = _ib[0]
                        if bound[1] is None: bound[1] = _ib[1]
                    else:
                        if bound[0] is None: bound[0] = -1e6
                        if bound[1] is None: bound[1] =  1e6
            
            from ..operators.heuristic_operators import generate_heuristic_variable
            return generate_heuristic_variable(
                self.features, 'fvar', name, dim, bound, self.agent, self.no_agents
            )
        
    def pvar(
        self,
        name: str,
        dim: List[int] = 0,
        bound: List[Optional[float]] = [0, None]
    ) -> MultidimVariable:
        """
        Creates and returns a positive variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        dim : List[int], optional
            Dimensions of this variable. Default: 0 (scalar). N = vector of N.
        bound : List[Optional[float]], optional
            Bounds of this variable. Default: [0, None].

        Returns
        -------
        MultidimVariable or _VarProxy
        """

        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _resolve_dim
            fdim = _resolve_dim(dim)
            bound = list(bound)
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'dp', 'dim': fdim, 'bounds': bound or [[0, None]],
                'var_type': 'cont', 'proxy': proxy
            }
            return proxy

        bound = list(bound)
        dim = self.fix_ifneeded(dim)
        self.features = update_variable_features(name, dim, bound, 'positive_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator
            self.features['variables'][("pvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'pvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim

            if self.features['interface_name'] in ['rsome_ro', 'rsome_dro', 'cvxpy']:
                if dim == 0:
                    self.con(self.features['variables'][("pvar", name)] >= 0)
                    if bound[1] is not None:
                        self.con(self.features['variables'][("pvar", name)] <= bound[1])
                elif len(dim) == 1:
                    for i in dim[0]:
                        self.con(self.features['variables'][("pvar", name)][i] >= 0)
                        if bound[1] is not None:
                            self.con(self.features['variables'][("pvar", name)][i] <= bound[1])
                else:
                    for i in it.product(*tuple(dim)):
                        self.con(self.features['variables'][("pvar", name)][i] >= 0)
                        if bound[1] is not None:
                            self.con(self.features['variables'][("pvar", name)][i] <= bound[1])

            native_var = self.features['variables'][("pvar", name)]

            if self.features.get('auto_linearize'):
                from ..classes.linearization import LinearizationProxy
                if isinstance(native_var, dict):
                    return {k: LinearizationProxy(v, self, f"{name}[{k}]", lb=bound[0], ub=bound[1]) for k, v in native_var.items()}
                return LinearizationProxy(native_var, self, name, lb=bound[0], ub=bound[1])

            return native_var

        elif self.features['solution_method'] == 'heuristic':
            
            from ..operators.heuristic_operators import generate_heuristic_variable
            
            if any(b is None for b in bound):
                if self.features['agent_status'] == 'idle':
                    _override = self.features.get('_auto_bound_overrides', {}).get(name)
                    if _override:
                        if bound[0] is None: bound[0] = _override[0]
                        if bound[1] is None: bound[1] = _override[1]
                    else:
                        _ib = self.features.get('_inferred_bounds', {}).get(name)
                        if _ib:
                            if bound[0] is None: bound[0] = _ib[0]
                            if bound[1] is None: bound[1] = _ib[1]
                        else:
                            if bound[0] is None: bound[0] = 0
                            if bound[1] is None: bound[1] = 1e6
                    self.features.setdefault('_auto_bound_vars', []).append(name)
                else:
                    _ib = self.features.get('_inferred_bounds', {}).get(name)
                    if _ib:
                        if bound[0] is None: bound[0] = _ib[0]
                        if bound[1] is None: bound[1] = _ib[1]
                    else:
                        if bound[0] is None: bound[0] = 0
                        if bound[1] is None: bound[1] = 1e6
            
            return generate_heuristic_variable(
                self.features, 'pvar', name, dim, bound, self.agent, self.no_agents
            )
    
    def ivar(
        self,
        name: str,
        dim: List[int] = 0,
        bound: List[Optional[float]] = [0, None]
    ) -> MultidimVariable:
        """
        Creates and returns an integer variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        dim : List[int], optional
            Dimensions of this variable. Default: 0 (scalar). N = vector of N.
        bound : List[Optional[float]], optional
            Bounds of this variable. Default: [0, None].

        Returns
        -------
        MultidimVariable or _VarProxy
        """

        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _resolve_dim
            fdim = _resolve_dim(dim)
            bound = list(bound)
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'dp', 'dim': fdim, 'bounds': bound,
                'var_type': 'int', 'proxy': proxy
            }
            return proxy

        bound = list(bound)
        dim = self.fix_ifneeded(dim)
        self.features = update_variable_features(name, dim, bound, 'integer_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator
            self.features['variables'][("ivar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'ivar', name, bound, dim
            )
            self.features['dimensions'][name] = dim

            if self.features['interface_name'] in ['rsome_ro', 'rsome_dro', 'cvxpy']:
                if dim == 0:
                    self.con(self.features['variables'][("ivar", name)] >= 0)
                    if bound[1] is not None:
                        self.con(self.features['variables'][("ivar", name)] <= bound[1])
                elif len(dim) == 1:
                    for i in dim[0]:
                        self.con(self.features['variables'][("ivar", name)][i] >= 0)
                        if bound[1] is not None:
                            self.con(self.features['variables'][("ivar", name)][i] <= bound[1])
                else:
                    for i in it.product(*tuple(dim)):
                        self.con(self.features['variables'][("ivar", name)][i] >= 0)
                        if bound[1] is not None:
                            self.con(self.features['variables'][("ivar", name)][i] <= bound[1])

            native_var = self.features['variables'][("ivar", name)]

            if self.features.get('auto_linearize'):
                from ..classes.linearization import LinearizationProxy
                if isinstance(native_var, dict):
                    return {k: LinearizationProxy(v, self, f"{name}[{k}]", lb=bound[0], ub=bound[1]) for k, v in native_var.items()}
                return LinearizationProxy(native_var, self, name, lb=bound[0], ub=bound[1])

            return native_var

        elif self.features['solution_method'] == 'heuristic':

            if any(b is None for b in bound):
                if self.features['agent_status'] == 'idle':
                    _override = self.features.get('_auto_bound_overrides', {}).get(name)
                    if _override:
                        if bound[0] is None: bound[0] = _override[0]
                        if bound[1] is None: bound[1] = _override[1]
                    else:
                        _ib = self.features.get('_inferred_bounds', {}).get(name)
                        if _ib:
                            if bound[0] is None: bound[0] = _ib[0]
                            if bound[1] is None: bound[1] = _ib[1]
                        else:
                            if bound[0] is None: bound[0] = 0
                            if bound[1] is None: bound[1] = 1e6
                    self.features.setdefault('_auto_bound_vars', []).append(name)
                else:
                    _ib = self.features.get('_inferred_bounds', {}).get(name)
                    if _ib:
                        if bound[0] is None: bound[0] = _ib[0]
                        if bound[1] is None: bound[1] = _ib[1]
                    else:
                        if bound[0] is None: bound[0] = 0
                        if bound[1] is None: bound[1] = 1e6

            from ..operators.heuristic_operators import generate_heuristic_variable         
            return generate_heuristic_variable(
                self.features, 'ivar', name, dim, bound, self.agent, self.no_agents
            )

    def bvar(
        self,
        name: str,
        dim: List[int] = 0,
        bound: List[Optional[float]] = [0, 1]
    ) -> MultidimVariable:
        """
        Creates and returns a binary variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        dim : List[int], optional
            Dimensions of this variable. Default: 0 (scalar). N = vector of N.
        bound : List[Optional[float]], optional
            Bounds of this variable. Default: [0, 1].

        Returns
        -------
        MultidimVariable or _VarProxy
        """

        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _resolve_dim
            fdim = _resolve_dim(dim)
            bound = list(bound)
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'dp', 'dim': fdim, 'bounds': bound,
                'var_type': 'bin', 'proxy': proxy
            }
            return proxy

        bound = list(bound)
        dim = self.fix_ifneeded(dim)
        self.features = update_variable_features(name, dim, bound, 'binary_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator
            self.features['variables'][("bvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'bvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim

            if self.features['interface_name'] in ['cvxpy']:
                if dim == 0:
                    self.con(self.features['variables'][("bvar", name)] >= 0)
                    self.con(self.features['variables'][("bvar", name)] <= 1)
                elif len(dim) == 1:
                    for i in dim[0]:
                        self.con(self.features['variables'][("bvar", name)][i] >= 0)
                        self.con(self.features['variables'][("bvar", name)][i] <= 1)
                else:
                    for i in it.product(*tuple(dim)):
                        self.con(self.features['variables'][("bvar", name)][i] >= 0)
                        self.con(self.features['variables'][("bvar", name)][i] <= 1)
            native_var = self.features['variables'][("bvar", name)]

            if self.features.get('auto_linearize'):
                from ..classes.linearization import LinearizationProxy
                if isinstance(native_var, dict):
                    return {k: LinearizationProxy(v, self, f"{name}[{k}]", lb=bound[0], ub=bound[1]) for k, v in native_var.items()}
                return LinearizationProxy(native_var, self, name, lb=bound[0], ub=bound[1])

            return native_var

        elif self.features['solution_method'] == 'heuristic':
            from ..operators.heuristic_operators import generate_heuristic_variable
            return generate_heuristic_variable(
                self.features, 'bvar', name, dim, bound, self.agent, self.no_agents
            )
            
    def svar(
        self,
        name: str,
        initial=None,
        length: int = 1
    ) -> MultidimVariable:
        """
        Creates and returns a state variable.

        When solution_method == 'sequential', creates an SDM state variable
        whose shape and initial value come from a data_toolkit parameter or
        array. Otherwise creates a sequence variable (permutation).

        Parameters
        ----------
        name : str
            Name of this variable.
        initial : array-like, NativeArray, DataRef, or numeric, optional
            Initial value for sequential mode. Dimension is inferred from shape.
        length : int, optional
            Length for sequence variable (non-sequential mode). Default: 1.

        Returns
        -------
        MultidimVariable or _VarProxy
        """
        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy
            if initial is None:
                raise ValueError("svar() in sequential mode requires 'initial'")
            val = initial
            if hasattr(val, '_arr'):
                val = val._arr
            elif hasattr(val, 'value'):
                val = val.value
            val = np.atleast_1d(np.asarray(val, dtype=float))
            dim = val.size
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'state', 'dim': dim, 'initial': val.copy(), 'proxy': proxy
            }
            if '_S0' not in self._sdm_components:
                self._sdm_components['_S0'] = {}
            self._sdm_components['_S0'][name] = val.copy()
            return proxy

        dim = fix_dims([length])
        self.features = update_variable_features(name, dim, [0, 1], 'sequential_variable_counter', self.features)

        if self.features['solution_method'] == 'heuristic':
            from ..operators.heuristic_operators import generate_heuristic_variable
            return generate_heuristic_variable(
                self.features, 'svar', name, dim, [0, 1], self.agent, self.no_agents
            )

    def xvar(
        self,
        name: str,
        dim: int = 0,
        inform=None,
    ):
        """Define an exogenous information variable with its generation function.

        When solution_method == 'sequential', creates an SDM exogenous variable.
        dim=0 -> scalar. dim=[N] -> N-vector.

        Parameters
        ----------
        name : str
            Variable name.
        dim : int
            Dimension. 0=scalar (default), N=vector of N.
        inform : _Proxy, callable, or None
            How to generate values each period.

        Returns
        -------
        _VarProxy
        """
        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _replay, _resolve_dim
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            fdim = _resolve_dim(dim)
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'exo', 'dim': fdim, 'proxy': proxy
            }
            if inform is not None:
                if callable(inform) and not hasattr(inform, '_tree'):
                    self._sdm_components['_exogenous_fn'] = inform
                elif hasattr(inform, '_tree'):
                    tree = inform._tree
                    def _compiled(t, rng):
                        ctx = {'rng': rng, 't': t}
                        result = _replay(tree, ctx)
                        val = result._backing if isinstance(result, _VarProxy) else np.asarray(result)
                        return {name: np.atleast_1d(val).flatten()}
                    self._sdm_components['_exogenous_fn'] = _compiled
                    self._sdm_components['_inform_target'] = name
            return proxy
        raise ValueError("xvar() is only available in sequential mode")

    def ppar(
        self,
        name: str,
        dim=0,
        initial=0.0,
        bound=None,
    ):
        """Define a tunable parameter (individual, named).

        When solution_method == 'sequential', creates an SDM parameter.
        dim=0 -> scalar. dim=[N] -> N-vector. dim=[[N,M]] -> NxM matrix.

        Parameters
        ----------
        name : str
            Parameter name.
        dim : int or list
            Dimension. 0=scalar (default).
        initial : float or array-like
            Initial value.
        bound : list of [lb, ub] or None
            Bounds for each dimension.

        Returns
        -------
        _VarProxy
        """
        if self.features.get('solution_method') == 'sequential':
            from ..algorithms.exact.sequential import _VarProxy, _resolve_dim
            fdim = _resolve_dim(dim)
            init_val = np.atleast_1d(np.asarray(initial, dtype=float)).flatten()
            if init_val.size == 1 and fdim > 1:
                init_val = np.full(fdim, init_val[0])
            if not hasattr(self, '_sdm_components'):
                self._sdm_components = {}
            if '_sdp_vars' not in self._sdm_components:
                self._sdm_components['_sdp_vars'] = {}
            _assert_unique_variable_name(
                name, self.features, sdp_types=self._sdm_components['_sdp_vars'])
            proxy = _VarProxy(name)
            self._sdm_components['_sdp_vars'][name] = {
                'type': 'tpar', 'dim': fdim, 'initial': init_val,
                'bounds': bound, 'proxy': proxy
            }
            if '_theta0' not in self._sdm_components:
                self._sdm_components['_theta0'] = {}
            self._sdm_components['_theta0'][name] = init_val.copy()
            if bound is not None:
                if '_theta_bounds' not in self._sdm_components:
                    self._sdm_components['_theta_bounds'] = {}
                self._sdm_components['_theta_bounds'][name] = bound
            return proxy
        raise ValueError("ppar() is only available in sequential mode")

    def rvar(
        self,
        name: str,
        dim: List[int] = 0
    ) -> MultidimVariable:
        """
        Creates and returns a random variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        dim : List[int], optional
            Dimensions of this variable. Default: 0.

        Returns
        -------
        MultidimVariable
            A random variable.
        """
        dim = self.fix_ifneeded(dim)
        from ..generators import variable_generator
        return variable_generator.generate_variable(
            self.features['interface_name'], self.model, 'rvar', name, [None, None], dim
        )
        
    def dvar(
        self,
        name: str,
        dim: Union[int, List[Union[int, range]]] = 0
    ) -> np.ndarray:
        """
        Creates and returns a dependent variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        dim : Union[int, List[Union[int, range]]], optional
            Dimensions of this variable. Default: 0.

        Returns
        -------
        np.ndarray
            A dependent variable.

        """
        dim = self.fix_ifneeded(dim)

        if self.no_agents is not None:
            default_pop = self.no_agents
        else:
            default_pop = 100

        if self.features['solution_method'] == 'exact':
            return np.zeros([len(dims) for dims in dim])
        
        elif self.features['solution_method'] == 'heuristic':
            if self.features['agent_status'] == 'idle':
                if self.features['vectorized']:
                    if dim == 0:
                        return 0
                    else:
                        return np.random.rand(*tuple([default_pop] + [len(dims) for dims in dim]))
                else:
                    if dim == 0:
                        return 0
                    else:
                        return np.zeros([len(dims) for dims in dim])
            else:
                if self.features['vectorized']:
                    if dim == 0:
                        return np.zeros(self.features['pop_size'])
                    else:
                        return np.zeros([self.features['pop_size']] + [len(dims) for dims in dim])
                else:
                    if dim == 0:
                        return 0
                    else:
                        return np.zeros([len(dims) for dims in dim])
