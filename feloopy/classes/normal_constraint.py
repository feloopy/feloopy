"""
Normal constraint module

Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

import re
from typing import List, Optional, Any

import numpy as np
from numpy import reshape, shape

_BATCH_NAME_RE = re.compile(r'^(.+?)\[(\d+(?:\s*,\s*\d+)*)\]$')


def parse_batch_name(name):
    """Split a constraint id into (base, index_tuple).

    ``"cost[3,2]"`` -> ``("cost", (3, 2))``; plain ids like ``"cost"`` or
    non-string values are returned unchanged with index ``None``.  The base
    is the batch id shared by every constraint declared under it, so
    ``get_dual("cost")`` can return a tensor indexed by ``i``/``j`` and
    sensitivity analysis can add/remove the whole batch at once.
    """
    if isinstance(name, str):
        m = _BATCH_NAME_RE.match(name.strip())
        if m:
            return m.group(1), tuple(int(x) for x in m.group(2).split(','))
        return name, None
    return name, None


def check_constraint_type(constraint):

    if isinstance(constraint, list) and len(constraint)!=0:

        if isinstance(constraint[0], list) and isinstance(constraint[0][1], str):
            return 'list with sense'
        
        elif len(constraint)>=2 and isinstance(constraint[1], str):
            
            return 'single list with sense'
        
        else:

            return 'list without sense'

    elif type(constraint)== dict:
        if isinstance(list(constraint.values())[0], list):
            return 'single dict with sense'
        else:
            
            return 'dict without sense'
    elif isinstance(constraint, list) and len(constraint)==0:
        return 'pass'
    elif isinstance(constraint, str):
        return 'evaluation string'
    else:
        return 'classic'

def check_sense(sense):

    if sense in ['<=', 'le', 'leq', '=l=']:
        return '<='
    elif sense in ['>=', 'ge', 'geq', '=g=']:
        return '>='
    elif sense in ['==', 'eq', '=e=']:
        return '=='
    elif sense in ['<', 'lt']:
        return '<'
    elif sense in ['>', 'gt']:
        return '>'
    elif sense in ['!=', 'neq']:
        return '!='
    
def generate_constraint(lhs, sense, rhs, epsilon):
    
    match sense:
        case '<=':
            return lhs <= rhs
        case '>=':
            return lhs >= rhs        
        case '==':
            return lhs == rhs   
        case '<':
            return lhs <= rhs - epsilon
        case '>':
            return lhs >= rhs + epsilon
        case '!=':
            return [lhs >= rhs + epsilon, lhs <= rhs - epsilon]       
        
class Expression:
    pass

class NormalConstraintClass:
    
    def enforce_gt(self, lhs, rhs, epsilon=1e-6, name=None):
        self.con([lhs, '>', rhs, epsilon], name)
 
    def enforce_lt(self, lhs, rhs, epsilon=1e-6, name=None):
        self.con([lhs, '<', rhs, epsilon], name)

    def enforce_geq(self, lhs, rhs, name=None):
        self.con([lhs, '>=', rhs], name)
        
    def enforce_leq(self, lhs, rhs, epsilon=1e-6, name=None):
        self.con([lhs, '<=', rhs], name)

    def enforce_eq(self, lhs, rhs, epsilon=1e-6, name=None):
        self.con([lhs, '==', rhs], name)

    def enforce_neq(self, lhs, rhs, epsilon=1e-6, name=None):
        self.con([lhs, '!=', rhs, epsilon], name)
                             
    def con(self, expression, name=None):
        """
        Constraint Definition
        ~~~~~~~~~~~~~~~~~~~~~
        To define a constraint.

        Args:
            expression (formula): what are the terms of this constraint?
            name (str, optional): what is the name of this constraint?. Defaults to None.
        """

        if not self.features.get('_auto_lin_generating', False) and not self.features.get('_lin_generating', False):
            self.features['_user_con_count'] = self.features.get('_user_con_count', 0) + 1
                    
        def add_special_constraint(element):
            relation, lower_bound, upper_bound = element[1], element[0], element[2]

            epsilon = element[3] if len(element) == 4 else 0.000001

            _is_cp = self.features.get('interface_name', '') in ['ortools_cp', 'cplex_cp', 'picat']

            if _is_cp and relation == '!=':
                const = [lower_bound != upper_bound]
            elif _is_cp and relation == '>':
                const = [lower_bound > upper_bound]
            elif _is_cp and relation == '<':
                const = [lower_bound < upper_bound]
            else:
                const = generate_constraint(lower_bound,relation,upper_bound,epsilon)
            if not isinstance(const, list):
                const = [const]
            return const

        if self.features.get('constraint_ids') is not None and name is not None:
            _ids = self.features['constraint_ids']
            if isinstance(name, list):
                if not any(n in _ids or (isinstance(n, str) and parse_batch_name(n)[0] in _ids)
                           for n in name):
                    return
            else:
                if name not in _ids and parse_batch_name(name)[0] not in _ids:
                    return

        # ---- constraint batch support -----------------------------------
        # Batches group constraints created under one id: repeated calls
        # (`for i in I: m.con(e, name="ci")`), list forms (labels ci0,
        # ci1, ...) and indexed names (`name="c[i, j]"`) are registered in
        # features['constraint_batches'] so get_dual/get_slack("c") can
        # return an index tensor and sensitivity analysis can add/remove
        # the batch as a whole via features['_exclude_batches'].
        self.features.setdefault('constraint_batches', {})
        _excluded = tuple(self.features.get('_exclude_batches') or ())
        if _excluded:
            if isinstance(expression, dict):
                _kept = {k: v for k, v in expression.items()
                         if not (isinstance(k, str) and parse_batch_name(k)[0] in _excluded)}
                if len(_kept) != len(expression):
                    if not _kept:
                        return
                    expression = _kept
            elif isinstance(name, list):
                _keep_i = [i for i, n in enumerate(name)
                           if not (isinstance(n, str) and parse_batch_name(n)[0] in _excluded)]
                if not _keep_i:
                    return
                if len(_keep_i) != len(name):
                    if isinstance(expression, list) and len(expression) == len(name):
                        expression = [expression[i] for i in _keep_i]
                        name = [name[i] for i in _keep_i]
                    else:
                        return
            elif isinstance(name, str) and parse_batch_name(name)[0] in _excluded:
                return

        # A repeated plain id would create duplicate labels and break
        # label->row lookups; later occurrences get a "#k" suffix while
        # staying grouped under the original batch id (_group_name).
        _group_name = name
        if isinstance(name, str):
            _gbase, _gidx = parse_batch_name(name)
            if _gidx is None and _gbase in self.features['constraint_batches']:
                _existing_labels = set(x for x in self.features['constraint_labels']
                                       if x is not None)
                _occ = 1
                while (f"{_gbase}#{_occ}" in self.features['constraint_batches']
                       or f"{_gbase}#{_occ}" in _existing_labels):
                    _occ += 1
                name = f"{_gbase}#{_occ}"

        _lbl0 = len(self.features['constraint_labels'])

        match self.features['solution_method']:

            case 'exact':

                if 'insideopt' in self.features['interface_name']:
                    self.features['constraint_labels'].append(name)
                    self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                    self.features['constraints'].append(expression)
                    self.features['constraint_counter'][1] = len(self.features['constraints'])

                elif 'pyoptinterface' in self.features['interface_name']:
                    match check_constraint_type(expression):

                        case 'list with sense':
                            const_list = []
                            for element in expression:
                                const = add_special_constraint(element)
                                if not isinstance(const, list):
                                    const = [const]
                                const_list += const
                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(expression))]
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraints'] += const_list
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'single list with sense':
                            if len(expression)==3: name = [name]
                            const = add_special_constraint(expression)
                            if not isinstance(const, list):
                                const = [const]
                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(const))]
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraints'] += const
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'list without sense':
                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(expression))]
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraints'] += list(expression)
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'dict with sense':
                            for key, value in expression.items():
                                const = add_special_constraint(value)
                                if not isinstance(const, list):
                                    const = [const]
                                self.features['constraint_labels'].append(key)
                                self.features['constraints'] += const
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'single dict with sense':
                            for key, value in expression.items():
                                const = add_special_constraint(value)
                                if not isinstance(const, list):
                                    const = [const]
                                self.features['constraint_labels'].append(key)
                                self.features['constraints'] += const
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'pass':
                            return

                        case 'classic':
                            self.features['constraint_labels'].append(name)
                            self.features['constraint_counter'][0] = len(set([x for x in self.features['constraint_labels'] if x is not None]))
                            self.features['constraints'].append(expression)
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                else:           
                    match check_constraint_type(expression):

                        case 'list with sense':
                            const_list = []
                            for element in expression:
                                const = add_special_constraint(element)
                                const_list+=const
                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(expression))]
                            self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                            self.features['constraints'] += const_list
                            self.features['constraint_counter'][1] = len(self.features['constraints'])
                    
                        case 'single list with sense': 
                
                            if len(expression)==3: name = [name]
                            const = add_special_constraint(expression)
                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(const))]
                            self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                            self.features['constraints'] += const
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'list without sense':

                            if isinstance(name, list): self.features['constraint_labels'] += name
                            else: self.features['constraint_labels'] += [str(name)+str(i) if name else None for i in range(len(expression))]
                            self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                            self.features['constraints'] += list(expression)
                            self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'dict with sense':

                            for key, value in expression.items():
        
                                const = add_special_constraint(value)
                                self.features['constraint_labels'].append(key)
                                self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                                self.features['constraints']+=const
                                self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'single dict with sense':
                    
                            for key, value in expression.items():
                                const = add_special_constraint(value)
                                self.features['constraint_labels'].append(key)
                                self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                                self.features['constraints']+=const
                                self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'dict without sense':

                            self.features['constraint_labels']+=list(expression.keys())
                            self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                            self.features['constraints']+=list(expression.values())
                            self.features['constraint_counter'][1] = len(self.features['constraints'])
                        
                        case 'classic':
                        
                            self.features['constraint_labels'].append(name)
                            self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                            self.features['constraints'].append(expression)
                            self.features['constraint_counter'][1] = len(self.features['constraints'])
                            if self.features.get('_objective_already_set') and hasattr(self, 'model') and hasattr(self.model, 'st'):
                                self.model.st(expression)

                        case 'evaluation string':
                            
                            if self.features['interface_name']=="jump":
                                self.features['constraint_labels'].append(name)
                                self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                                self.features['constraints'].append(expression)
                                self.features['constraint_counter'][1] = len(self.features['constraints'])

                        case 'pass':
                            pass

            case 'heuristic':

                if self.features['agent_status'] == 'idle':
                    self.features['constraint_labels'].append(name)
                    self.features['constraint_counter'][0] = len(set(self.features['constraint_labels']))
                    self.features['constraints'].append(expression)
                    self.features['constraint_counter'][1] = len(self.features['constraints'])
                else:
                    if self.features['vectorized']:
                        _pop_n = shape(self.agent)[0]

                        def _as_col(e):
                            # mirror the exact-path constant filter: vacuous
                            # values add no row, a constant False is fatal
                            if e is None or isinstance(e, (bool, np.bool_)):
                                if e is not None and not bool(e):
                                    from ..helpers.error import ConstantConstraintError
                                    raise ConstantConstraintError(
                                        "A constraint contains no variables and "
                                        "evaluates to False, so the model can "
                                        "never be satisfied. Check the data "
                                        "feeding it.")
                                return None
                            if isinstance(e, (int, float, np.integer, np.floating)):
                                return None  # bare number: placeholder, not a constraint
                            # vectorised expressions arrive as (pop, 1)
                            # columns, and a whole-array con() (x <= cap)
                            # as a (pop, k) matrix or taller — rows are
                            # agents by construction, and the penalty
                            # consumer reads ndim>=2 entries as per-agent
                            # columns (feloopy.py sol()). Reshaping a
                            # (pop, 1) one costs a wrapper frame and a
                            # view per element, six figures per solve
                            if isinstance(e, np.ndarray) and e.ndim >= 2 and e.shape[0] == _pop_n:
                                # strict comparisons (<, >) hand back booleans
                                # where True already means satisfied; store a
                                # 0/1 violated mask so a con() that mixes them
                                # with signed residuals cannot promote True to
                                # 1.0 when the columns are concatenated below
                                if e.dtype == bool:
                                    e = np.logical_not(e).astype(float)
                                if e.ndim == 2:
                                    return e
                                return np.reshape(e, [_pop_n, -1])
                            if np.asarray(e).dtype == bool:
                                e = np.logical_not(np.asarray(e)).astype(float)
                            return reshape(e, [_pop_n, 1])

                        if isinstance(expression, (list, tuple)):
                            # one con() call becomes one (pop, k) entry:
                            # stack the per-agent expressions as columns —
                            # the penalty consumer reads ndim>=2 entries as
                            # (pop, k) matrices (feloopy.py sol())
                            _cols = [c for c in (_as_col(e) for e in expression)
                                     if c is not None]
                            if _cols:
                                self.features['constraints'].append(
                                    _cols[0] if len(_cols) == 1
                                    else np.concatenate(_cols, axis=1))
                        else:
                            _c = _as_col(expression)
                            if _c is not None:
                                self.features['constraints'].append(_c)
                    else:
                        self.features['constraints'].append(expression)

        self._register_constraint_batch(
            _group_name, self.features['constraint_labels'][_lbl0:])

    def _register_constraint_batch(self, name, new_labels):
        """Record labels appended by the last con() call as a constraint batch.

        """
        registry = self.features.setdefault('constraint_batches', {})
        if not new_labels:
            return
        _pairs = []
        if isinstance(name, list):
            for i, lbl in enumerate(new_labels):
                if not isinstance(lbl, str):
                    continue
                _n = name[i] if i < len(name) else lbl
                base, idx = parse_batch_name(_n if isinstance(_n, str) else lbl)
                if isinstance(base, str):
                    _pairs.append((base, idx, lbl))
        elif isinstance(name, str):
            base, idx = parse_batch_name(name)
            if not isinstance(base, str):
                return
            if len(new_labels) == 1:
                if isinstance(new_labels[0], str):
                    _pairs.append((base, idx, new_labels[0]))
            else:
                # list-form auto labels (ci0, ci1, ...): positional group
                for lbl in new_labels:
                    if isinstance(lbl, str):
                        _pairs.append((base, None, lbl))
        else:
            # name None: dict ids become single-element batches
            for lbl in new_labels:
                if not isinstance(lbl, str):
                    continue
                base, idx = parse_batch_name(lbl)
                if isinstance(base, str):
                    _pairs.append((base, idx, lbl))
        for base, idx, lbl in _pairs:
            registry.setdefault(base, {'elements': []})
            entry = registry[base]['elements']
            if not any(existing[1] == lbl for existing in entry):
                entry.append((idx, lbl))