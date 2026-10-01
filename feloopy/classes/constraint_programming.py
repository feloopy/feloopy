"""
ConstraintProgramming Module

This module defines a class, `ConstraintProgrammingClass`, that is used in constraint programming.

Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

from .event_variable import EventVariable
from typing import Optional, Union

try:
    from ..generators.picat_expression import PicatVar, PicatExpr
except ImportError:
    PicatVar = None
    PicatExpr = None

class ConstraintFeature:
    pass

class ConstraintProgrammingClass:
    
    def cp_get_event_start(self, event_variable: EventVariable, absent_value: Optional[ConstraintFeature] = None) -> ConstraintFeature:
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_of(event_variable, absent_value)
        elif self.features['interface_name'] == 'ortools_cp':
            return event_variable.StartExpr()

    def cp_get_event_end(self, event_variable: EventVariable, absent_value: Optional[ConstraintFeature] = None) -> ConstraintFeature:
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_of(event_variable, absent_value)
        elif self.features['interface_name'] == 'ortools_cp':
            return event_variable.EndExpr()
    
    def cp_get_event_length(self, event_variable: EventVariable, absent_value: Optional[ConstraintFeature] = None) -> ConstraintFeature:
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.length_of(event_variable, absent_value)
        elif self.features['interface_name'] == 'ortools_cp':
            return event_variable.EndExpr() - event_variable.StartExpr()

    def cp_get_event_size(self, event_variable: EventVariable, absent_value: Optional[ConstraintFeature] = None) -> ConstraintFeature:
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.size_of(event_variable, absent_value)
        elif self.features['interface_name'] == 'ortools_cp':
            return event_variable.SizeExpr()

    def cp_get_presence(self, event_variable: EventVariable):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.presence_of(event_variable)
        elif self.features['interface_name'] == 'ortools_cp':
            literals = event_variable.presence_literals()
            return literals[0] if literals else None
        
    def cp_event_start_exactly_before_start(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_at_start(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay == second_event.StartExpr())

    def cp_event_start_exactly_before_end(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_at_end(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay == second_event.EndExpr())

    def cp_event_end_exactly_before_start(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_at_start(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.EndExpr() + delay == second_event.StartExpr())

    def cp_event_end_exactly_before_end(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_at_end(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.EndExpr() + delay == second_event.EndExpr())

    def cp_event_start_before_start(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_before_start(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay <= second_event.StartExpr())

    def cp_event_start_before_end(self, first_event, second_event, delay: Union[int, float] = 0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_before_end(first_event, second_event, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay <= second_event.EndExpr())

    def cp_event_end_before_start(self, event_one, event_two, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_before_start(event_one, event_two, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(event_one.EndExpr() + delay <= event_two.StartExpr())

    def cp_event_end_before_end(self, event_one, event_two, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_before_end(event_one, event_two, delay)
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(event_one.EndExpr() + delay <= event_two.EndExpr())
    
    def cp_forbid_event_start(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_start(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_start_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_start_high_{domain[1]}')
                self.model.Add(event.StartExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.StartExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])

    def cp_forbid_event_end(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_end(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_end_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_end_high_{domain[1]}')
                self.model.Add(event.EndExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.EndExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])
            
    def cp_forbid_event_overlap(self, event_variables, transition_matrix=None):
        if self.features['interface_name'] == 'cplex_cp':
            if transition_matrix is None:
                return self.model.no_overlap(event_variables)
            else:
                return self.model.no_overlap(event_variables, transition_matrix)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddNoOverlap(event_variables)
        
    def cp_forbid_event_extent(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_extent(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_extent_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_extent_high_{domain[1]}')
                self.model.Add(event.EndExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.StartExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])
            
    def cp_event_overlap_length(self, event1, event2, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.overlap_length(event1, event2, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            _max = 10000
            overlap_start = self.model.NewIntVar(0, _max, "overlap_start")
            overlap_end = self.model.NewIntVar(0, _max, "overlap_end")
            self.model.AddMaxEquality(overlap_start, [event1.StartExpr(), event2.StartExpr()])
            self.model.AddMinEquality(overlap_end, [event1.EndExpr(), event2.EndExpr()])
            result = self.model.NewIntVar(0, _max, "overlap_length")
            self.model.AddMaxEquality(result, [overlap_end - overlap_start, 0])
            return result

    def cp_start_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.StartExpr())

    def cp_end_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.EndExpr())

    def cp_size_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.size_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.SizeExpr())

    def cp_length_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.length_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.EndExpr() - event.StartExpr())

    def cp_span(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.span(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            intervals = function(event)
            for i in intervals:
                self.model.Add(event.StartExpr() <= i.StartExpr())
                self.model.Add(event.EndExpr() >= i.EndExpr())

    def cp_always_equal(self, state_function, input1, input2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.always_equal(state_function, input1, input2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(input1 == input2)

    def cp_alternative(self, event, array, cardinality=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.alternative(event, array, cardinality)
        if self.features['interface_name'] == 'ortools_cp':
            bools = [self.model.NewBoolVar(f"alt_{i}") for i in range(len(array))]
            self.model.AddExactlyOne(bools)
            for b, alt in zip(bools, array):
                self.model.Add(event.StartExpr() == alt.StartExpr()).OnlyEnforceIf(b)
                self.model.Add(event.EndExpr() == alt.EndExpr()).OnlyEnforceIf(b)
                self.model.Add(event.SizeExpr() == alt.SizeExpr()).OnlyEnforceIf(b)

    def cp_all_dist_above(self, exprs, value):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.all_min_distance(exprs, value)
        if self.features['interface_name'] == 'ortools_cp':
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    b = self.model.NewBoolVar(f'all_dist_{i}_{j}')
                    self.model.Add(exprs[i] - exprs[j] >= value).OnlyEnforceIf(b)
                    self.model.Add(exprs[j] - exprs[i] >= value).OnlyEnforceIf(b.Not())

    def cp_if_then(self, condition, consequence):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.if_then(condition, consequence)
        if self.features['interface_name'] == 'ortools_cp':
            from ortools.sat.python import cp_model as _cm
            if isinstance(consequence, (_cm.BoolVarL, _cm.NotBoolVar)):
                self.model.AddImplication(condition, consequence)
            else:
                self.model.Add(consequence).OnlyEnforceIf(condition)

    def cp_synchronize(self, event, array):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.synchronize(event, array)
        if self.features['interface_name'] == 'ortools_cp':
            for other in array:
                self.model.Add(event.StartExpr() == other.StartExpr())

    def cp_control_resource(self, *args, function='pulse'):
        if self.features['interface_name'] == 'cplex_cp':
            if function == 'pulse':
                return self.model.pulse(*args)
            if function == 'step':
                return self.model.step(*args)
            if function == 'start':
                return self.model.step_at_start(*args)
            if function == 'end':
                return self.model.step_at_end(*args)
        if self.features['interface_name'] == 'ortools_cp':
            if function == 'cumulative':
                intervals, demands, capacity = args
                self.model.AddCumulative(intervals, demands, capacity)

    def cp_circuit(self, arcs):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.circuit(arcs)
        if self.features['interface_name'] == 'ortools_cp':
            from ortools.sat.python.cp_model_helper import IntVar
            has_successor_var = any(isinstance(a[1], IntVar) for a in arcs)
            if has_successor_var:
                n = len(arcs)
                all_arcs = []
                for i, successor_var, _ in arcs:
                    self.model.Add(successor_var != i)
                    for j in range(n):
                        arc = self.model.NewBoolVar(f'circuit_arc_{i}_{j}')
                        self.model.Add(successor_var == j).OnlyEnforceIf(arc)
                        self.model.Add(successor_var != j).OnlyEnforceIf(arc.Not())
                        all_arcs.append((i, j, arc))
                return self.model.AddCircuit(all_arcs)
            else:
                return self.model.AddCircuit(arcs)

    def cp_allowed_assignments(self, variables, tuples):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.allowed_assignments(variables, tuples)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAllowedAssignments(variables, tuples)

    def cp_inverse(self, variables, inverse_variables):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.inverse(variables, inverse_variables)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddInverse(variables, inverse_variables)

    def cp_logical_and(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_and(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolAnd([expr1, expr2])

    def cp_logical_or(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_or(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolOr([expr1, expr2])

    def cp_logical_not(self, expr):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_not(expr)
        if self.features['interface_name'] == 'ortools_cp':
            return expr.Not()

    def cp_less_than(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.less_than(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(expr1 < expr2)

    def cp_greater_than(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.greater_than(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(expr1 > expr2)
        
    def cp_add_exactly_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.exactly_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddExactlyOne(variable_list)

    def cp_add_at_least_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.at_least_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAtLeastOne(variable_list)

    def cp_add_at_most_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.at_most_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAtMostOne(variable_list)

    def cp_add_all_different(self, expressions):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.all_diff(expressions)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAllDifferent(expressions)
        if self.features['interface_name'] == 'picat':
            if isinstance(expressions, dict):
                var_list = list(expressions.values())
            elif isinstance(expressions, list):
                var_list = expressions
            else:
                var_list = [expressions]
            exprs = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in var_list)
            self.model.add_constraint_string(f"all_different([{exprs}])")

    def cp_add_cumulative(self, intervals, demands, capacity):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.cumulative(intervals, demands, capacity)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddCumulative(intervals, demands, capacity)
        if self.features['interface_name'] == 'picat':
            if isinstance(intervals, list):
                starts_str = ', '.join(s._picat_expr if hasattr(s, '_picat_expr') else str(s) for s in intervals)
            else:
                starts_str = intervals._picat_expr if hasattr(intervals, '_picat_expr') else str(intervals)
            if isinstance(demands, list):
                dur_str = ', '.join(str(d) for d in demands)
                res_str = ', '.join('1' for _ in demands)
            else:
                dur_str = str(demands)
                res_str = '1'
            self.model.add_constraint_string(f"cumulative([{starts_str}], [{dur_str}], [{res_str}], {capacity})")

    def cp_add_element(self, index_var, expressions, target):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.element(index_var, expressions, target)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddElement(index_var, expressions, target)
        if self.features['interface_name'] == 'picat':
            idx = index_var._picat_expr if hasattr(index_var, '_picat_expr') else str(index_var)
            exprs = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in expressions) if isinstance(expressions, list) else str(expressions)
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            self.model.add_constraint_string(f"element({idx}, [{exprs}], {tgt})")

    def cp_add_automaton(self, variables, start_state, transition_tuples, final_states):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.automaton(variables, start_state, transition_tuples, final_states)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAutomaton(variables, start_state, final_states, transition_tuples)

    def cp_add_no_overlap_2d(self, x_intervals, y_intervals):
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddNoOverlap2D(x_intervals, y_intervals)
        if self.features['interface_name'] == 'picat':
            for xi, yi in zip(x_intervals, y_intervals):
                x_start = xi.get('start') if isinstance(xi, dict) else None
                x_end = xi.get('end') if isinstance(xi, dict) else None
                y_start = yi.get('start') if isinstance(yi, dict) else None
                y_end = yi.get('end') if isinstance(yi, dict) else None
                if x_start and x_end and y_start and y_end:
                    xs = x_start._picat_expr if hasattr(x_start, '_picat_expr') else str(x_start)
                    xe = x_end._picat_expr if hasattr(x_end, '_picat_expr') else str(x_end)
                    ys = y_start._picat_expr if hasattr(y_start, '_picat_expr') else str(y_start)
                    ye = y_end._picat_expr if hasattr(y_end, '_picat_expr') else str(y_end)
                    self.model.add_constraint_string(f"({xs} #>= {ye} #\\/ {xe} #=< {ys})")

    def cp_add_multiple_circuit(self, starts, ends, arcs):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.circuit(arcs)
        if self.features['interface_name'] == 'ortools_cp':
            from ortools.sat.python.cp_model_helper import IntVar
            has_successor_var = any(isinstance(a[1], IntVar) for a in arcs)
            if has_successor_var:
                n = len(arcs)
                all_arcs = []
                for i, successor_var, used_var in arcs:
                    self.model.Add(successor_var != i)
                    for j in range(n):
                        if i == j:
                            continue
                        arc = self.model.NewBoolVar(f'mcircuit_arc_{i}_{j}')
                        self.model.Add(successor_var == j).OnlyEnforceIf(arc)
                        self.model.Add(successor_var != j).OnlyEnforceIf(arc.Not())
                        all_arcs.append((i, j, arc))
                return self.model.AddMultipleCircuit(all_arcs)
            else:
                return self.model.AddMultipleCircuit(arcs)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables) if isinstance(variables, list) else str(variables)
            self.model.add_constraint_string(f"circuit([{vars_str}])")

    def cp_add_reservoir(self, times, levels, min_level, max_level):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.reservoir(times, levels, min_level, max_level)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddReservoirConstraint(times, levels, min_level, max_level)

    def cp_add_reservoir_with_active(self, times, levels, active, min_level, max_level):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.reservoir(times, levels, min_level, max_level, active)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddReservoirConstraintWithActive(times, levels, active, min_level, max_level)
        if self.features['interface_name'] == 'picat':
            times_str = ', '.join(t._picat_expr if hasattr(t, '_picat_expr') else str(t) for t in times)
            levels_str = ', '.join(str(l) for l in levels)
            self.model.add_constraint_string(f"cumulative([{times_str}], [{levels_str}], 1, {max_level})")

    def cp_add_max_equality(self, target, expressions):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.maximum(target, expressions)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddMaxEquality(target, expressions)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            if isinstance(expressions, list):
                exprs = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in expressions)
            else:
                exprs = expressions._picat_expr if hasattr(expressions, '_picat_expr') else str(expressions)
            self.model.add_constraint_string(f"max([{exprs}]) #= {tgt}")

    def cp_add_min_equality(self, target, expressions):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.minimum(target, expressions)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddMinEquality(target, expressions)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            if isinstance(expressions, list):
                exprs = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in expressions)
            else:
                exprs = expressions._picat_expr if hasattr(expressions, '_picat_expr') else str(expressions)
            self.model.add_constraint_string(f"min([{exprs}]) #= {tgt}")

    def cp_add_modulo_equality(self, target, expr, modulus):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.modulo(target, expr, modulus)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddModuloEquality(target, expr, modulus)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            e = expr._picat_expr if hasattr(expr, '_picat_expr') else str(expr)
            self.model.add_constraint_string(f"({e} mod {modulus}) #= {tgt}")

    def cp_add_multiplication_equality(self, target, expressions):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.prod(target, expressions)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddMultiplicationEquality(target, expressions)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            if isinstance(expressions, list):
                exprs = '*'.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in expressions)
            else:
                exprs = expressions._picat_expr if hasattr(expressions, '_picat_expr') else str(expressions)
            self.model.add_constraint_string(f"({exprs}) #= {tgt}")

    def cp_add_division_equality(self, target, expr_dividend, expr_divisor):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.division(target, expr_dividend, expr_divisor)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddDivisionEquality(target, expr_dividend, expr_divisor)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            d = expr_dividend._picat_expr if hasattr(expr_dividend, '_picat_expr') else str(expr_dividend)
            r = expr_divisor._picat_expr if hasattr(expr_divisor, '_picat_expr') else str(expr_divisor)
            self.model.add_constraint_string(f"({d}//{r}) #= {tgt}")

    def cp_add_implication(self, boolvar, consequence):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.if_then(boolvar, consequence)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddImplication(boolvar, consequence)
        if self.features['interface_name'] == 'picat':
            b = boolvar._picat_expr if hasattr(boolvar, '_picat_expr') else str(boolvar)
            c = consequence._picat_expr if hasattr(consequence, '_picat_expr') else str(consequence)
            self.model.add_constraint_string(f"({b} #=> {c})")

    def cp_add_bool_and(self, boolvars):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_and(boolvars)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolAnd(boolvars)
        if self.features['interface_name'] == 'picat':
            exprs = [v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in boolvars]
            if len(exprs) == 1:
                self.model.add_constraint_string(f"({exprs[0]})")
            else:
                joined = ' #/\\ '.join(f'({e})' for e in exprs)
                self.model.add_constraint_string(f"({joined})")

    def cp_add_bool_or(self, boolvars):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_or(boolvars)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolOr(boolvars)
        if self.features['interface_name'] == 'picat':
            exprs = [v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in boolvars]
            if len(exprs) == 1:
                self.model.add_constraint_string(f"({exprs[0]})")
            else:
                joined = ' #\\/ '.join(f'({e})' for e in exprs)
                self.model.add_constraint_string(f"({joined})")

    def cp_add_bool_xor(self, boolvars):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_xor(boolvars[0], boolvars[1])
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolXOr(boolvars)
        if self.features['interface_name'] == 'picat':
            a = boolvars[0]._picat_expr if hasattr(boolvars[0], '_picat_expr') else str(boolvars[0])
            b = boolvars[1]._picat_expr if hasattr(boolvars[1], '_picat_expr') else str(boolvars[1])
            self.model.add_constraint_string(f"({a} #^ {b})")

    def cp_add_forbidden_assignments(self, variables, tuples):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbidden_assignments(variables, tuples)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddForbiddenAssignments(variables, tuples)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            tuples_str = ', '.join('{' + ','.join(str(e) for e in t) + '}' for t in tuples)
            self.model.add_constraint_string(f"table_notin({{{vars_str}}}, [{tuples_str}])")

    def cp_add_no_overlap(self, intervals):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.no_overlap(intervals)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddNoOverlap(intervals)
        if self.features['interface_name'] == 'picat':
            starts, durations = [], []
            for iv in intervals:
                if isinstance(iv, dict):
                    starts.append(iv.get('start', iv.get('_start', '')))
                    durations.append(iv.get('size', iv.get('duration', '')))
                else:
                    starts.append(str(iv))
                    durations.append('1')
            s_str = ', '.join(s._picat_expr if hasattr(s, '_picat_expr') else str(s) for s in starts)
            d_str = ', '.join(d._picat_expr if hasattr(d, '_picat_expr') else str(d) for d in durations)
            self.model.add_constraint_string(f"serialized([{s_str}], [{d_str}])")

    def cp_add_abs_equality(self, target, expr):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.abs(target, expr)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAbsEquality(target, expr)
        if self.features['interface_name'] == 'picat':
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            e = expr._picat_expr if hasattr(expr, '_picat_expr') else str(expr)
            self.model.add_constraint_string(f"abs({e}) #= {tgt}")

    def cp_add_linear_constraint(self, expressions, domain):
        if self.features['interface_name'] == 'cplex_cp':
            total = sum(expressions) if isinstance(expressions, list) else expressions
            return self.model.add((total >= domain[0]) & (total <= domain[1]))
        if self.features['interface_name'] == 'ortools_cp':
            total = sum(expressions) if isinstance(expressions, list) else expressions
            return self.model.AddLinearConstraint(total, domain[0], domain[1])
        if self.features['interface_name'] == 'picat':
            if isinstance(expressions, list):
                exprs = '+'.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in expressions)
            else:
                exprs = expressions._picat_expr if hasattr(expressions, '_picat_expr') else str(expressions)
            self.model.add_constraint_string(f"({exprs}) #>= {domain[0]}")
            self.model.add_constraint_string(f"({exprs}) #<= {domain[1]}")

    def cp_add_hint(self, variable, value):
        if self.features['interface_name'] == 'cplex_cp':
            return None
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddHint(variable, value)
        if self.features['interface_name'] == 'picat':
            return None  # Picat does not support hints directly

    def cp_add_assumption(self, boolvar):
        if self.features['interface_name'] == 'cplex_cp':
            return None
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAssumption(boolvar)
        if self.features['interface_name'] == 'picat':
            return None  # Picat does not support assumptions directly

    def cp_add_count(self, variables, value, target):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.count(variables, value, target)
        if self.features['interface_name'] == 'ortools_cp':
            bvars = [self.model.NewBoolVar(f'count_{i}') for i in range(len(variables))]
            for i, v in enumerate(variables):
                self.model.Add(v == value).OnlyEnforceIf(bvars[i])
                self.model.Add(v != value).OnlyEnforceIf(bvars[i].Not())
            self.model.Add(sum(bvars) == target)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            tgt = target._picat_expr if hasattr(target, '_picat_expr') else str(target)
            self.model.add_constraint_string(f"count({value}, [{vars_str}], #=, {tgt})")

    def cp_add_reservoir(self, times, levels, min_level, max_level):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.reservoir(times, levels, min_level, max_level)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddReservoirConstraint(times, levels, min_level, max_level)
        if self.features['interface_name'] == 'picat':
            times_str = ', '.join(t._picat_expr if hasattr(t, '_picat_expr') else str(t) for t in times)
            levels_str = ', '.join(str(l) for l in levels)
            self.model.add_constraint_string(f"cumulative([{times_str}], [{levels_str}], 1, {max_level})")
            if min_level > 0:
                self.model.add_constraint_string(f"cumulative([{times_str}], [{levels_str}], 1, {max_level})")

    def cp_add_circuit(self, variables):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.circuit(variables)
        if self.features['interface_name'] == 'ortools_cp':
            n = len(variables)
            arcs = []
            for i, v in enumerate(variables):
                for j in range(n):
                    if i != j:
                        b = self.model.NewBoolVar(f'circuit_{i}_{j}')
                        self.model.Add(v == j).OnlyEnforceIf(b)
                        self.model.Add(v != j).OnlyEnforceIf(b.Not())
                        arcs.append((i, j, b))
            self.model.AddCircuit(arcs)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            self.model.add_constraint_string(f"circuit([{vars_str}])")

    def cp_add_inverse(self, variables, inverse_variables):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.inverse(variables, inverse_variables)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddInverse(variables, inverse_variables)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            inv_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in inverse_variables)
            self.model.add_constraint_string(f"assignment([{vars_str}], [{inv_str}])")

    def cp_add_allowed_assignments(self, variables, tuples):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.allowed_assignments(variables, tuples)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAllowedAssignments(variables, tuples)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            tuples_str = ', '.join('{' + ','.join(str(e) for e in t) + '}' for t in tuples)
            self.model.add_constraint_string(f"table_in({{{vars_str}}}, [{tuples_str}])")

    def cp_add_automaton(self, variables, start_state, transition_tuples, final_states):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.automaton(variables, start_state, transition_tuples, final_states)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAutomaton(variables, start_state, final_states, transition_tuples)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variables)
            trans_str = ', '.join(f"({t[0]},{t[1]},{t[2]})" for t in transition_tuples)
            final_str = ', '.join(str(s) for s in final_states)
            self.model.add_constraint_string(f"regular([{vars_str}], {len(final_states)}, {len(set(t[1] for t in transition_tuples))}, [{trans_str}], {start_state}, [{final_str}])")

    def cp_logical_and(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_and(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolAnd([expr1, expr2])
        if self.features['interface_name'] == 'picat':
            a = expr1._picat_expr if hasattr(expr1, '_picat_expr') else str(expr1)
            b = expr2._picat_expr if hasattr(expr2, '_picat_expr') else str(expr2)
            self.model.add_constraint_string(f"({a} #/\\ {b})")

    def cp_logical_or(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_or(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddBoolOr([expr1, expr2])
        if self.features['interface_name'] == 'picat':
            a = expr1._picat_expr if hasattr(expr1, '_picat_expr') else str(expr1)
            b = expr2._picat_expr if hasattr(expr2, '_picat_expr') else str(expr2)
            self.model.add_constraint_string(f"({a} #\\/ {b})")

    def cp_logical_not(self, expr):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.logical_not(expr)
        if self.features['interface_name'] == 'ortools_cp':
            return expr.Not()
        if self.features['interface_name'] == 'picat':
            e = expr._picat_expr if hasattr(expr, '_picat_expr') else str(expr)
            self.model.add_constraint_string(f"(#~ {e})")

    def cp_less_than(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.less_than(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(expr1 < expr2)
        if self.features['interface_name'] == 'picat':
            a = expr1._picat_expr if hasattr(expr1, '_picat_expr') else str(expr1)
            b = expr2._picat_expr if hasattr(expr2, '_picat_expr') else str(expr2)
            self.model.add_constraint_string(f"({a} #< {b})")

    def cp_greater_than(self, expr1, expr2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.greater_than(expr1, expr2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(expr1 > expr2)
        if self.features['interface_name'] == 'picat':
            a = expr1._picat_expr if hasattr(expr1, '_picat_expr') else str(expr1)
            b = expr2._picat_expr if hasattr(expr2, '_picat_expr') else str(expr2)
            self.model.add_constraint_string(f"({a} #> {b})")

    def cp_add_exactly_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.exactly_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddExactlyOne(variable_list)
        if self.features['interface_name'] == 'picat':
            vars_str = ', '.join(v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variable_list)
            self.model.add_constraint_string(f"all_different([{vars_str}])")
            self.model.add_constraint_string(f"({vars_str}) #>= 1")

    def cp_add_at_least_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.at_least_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAtLeastOne(variable_list)
        if self.features['interface_name'] == 'picat':
            exprs = [v._picat_expr if hasattr(v, '_picat_expr') else str(v) for v in variable_list]
            joined = ' #\\/ '.join(f'({e})' for e in exprs)
            self.model.add_constraint_string(f"({joined})")

    def cp_add_at_most_one(self, variable_list):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.at_most_one(variable_list)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddAtMostOne(variable_list)
        if self.features['interface_name'] == 'picat':
            for i in range(len(variable_list)):
                for j in range(i + 1, len(variable_list)):
                    a = variable_list[i]._picat_expr if hasattr(variable_list[i], '_picat_expr') else str(variable_list[i])
                    b = variable_list[j]._picat_expr if hasattr(variable_list[j], '_picat_expr') else str(variable_list[j])
                    self.model.add_constraint_string(f"({a} + {b} #<= 1)")

    def cp_if_then(self, condition, consequence):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.if_then(condition, consequence)
        if self.features['interface_name'] == 'ortools_cp':
            from ortools.sat.python import cp_model as _cm
            if isinstance(consequence, (_cm.BoolVarL, _cm.NotBoolVar)):
                self.model.AddImplication(condition, consequence)
            else:
                self.model.Add(consequence).OnlyEnforceIf(condition)
        if self.features['interface_name'] == 'picat':
            c = condition._picat_expr if hasattr(condition, '_picat_expr') else str(condition)
            e = consequence._picat_expr if hasattr(consequence, '_picat_expr') else str(consequence)
            self.model.add_constraint_string(f"({c} #=> {e})")

    def cp_synchronize(self, event, array):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.synchronize(event, array)
        if self.features['interface_name'] == 'ortools_cp':
            for other in array:
                self.model.Add(event.StartExpr() == other.StartExpr())
        if self.features['interface_name'] == 'picat':
            ev_start = event.get('start', event) if isinstance(event, dict) else event
            s1 = ev_start._picat_expr if hasattr(ev_start, '_picat_expr') else str(ev_start)
            for other in array:
                o_start = other.get('start', other) if isinstance(other, dict) else other
                s2 = o_start._picat_expr if hasattr(o_start, '_picat_expr') else str(o_start)
                self.model.add_constraint_string(f"({s1} #= {s2})")

    def cp_span(self, event, sub_events):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.span(event, sub_events)
        if self.features['interface_name'] == 'ortools_cp':
            for i in sub_events:
                self.model.Add(event.StartExpr() <= i.StartExpr())
                self.model.Add(event.EndExpr() >= i.EndExpr())
        if self.features['interface_name'] == 'picat':
            ev_start = event.get('start') if isinstance(event, dict) else None
            ev_end = event.get('end') if isinstance(event, dict) else None
            if ev_start and ev_end:
                s1 = ev_start._picat_expr if hasattr(ev_start, '_picat_expr') else str(ev_start)
                e1 = ev_end._picat_expr if hasattr(ev_end, '_picat_expr') else str(ev_end)
                for sub in sub_events:
                    sub_s = sub.get('start') if isinstance(sub, dict) else None
                    sub_e = sub.get('end') if isinstance(sub, dict) else None
                    if sub_s and sub_e:
                        s2 = sub_s._picat_expr if hasattr(sub_s, '_picat_expr') else str(sub_s)
                        e2 = sub_e._picat_expr if hasattr(sub_e, '_picat_expr') else str(sub_e)
                        self.model.add_constraint_string(f"({s1} #=< {s2})")
                        self.model.add_constraint_string(f"({e1} #>= {e2})")

    def cp_always_equal(self, state_function, input1, input2):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.always_equal(state_function, input1, input2)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.Add(input1 == input2)
        if self.features['interface_name'] == 'picat':
            a = input1._picat_expr if hasattr(input1, '_picat_expr') else str(input1)
            b = input2._picat_expr if hasattr(input2, '_picat_expr') else str(input2)
            self.model.add_constraint_string(f"({a} #= {b})")

    def cp_alternative(self, event, array, cardinality=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.alternative(event, array, cardinality)
        if self.features['interface_name'] == 'ortools_cp':
            bools = [self.model.NewBoolVar(f"alt_{i}") for i in range(len(array))]
            self.model.AddExactlyOne(bools)
            for b, alt in zip(bools, array):
                self.model.Add(event.StartExpr() == alt.StartExpr()).OnlyEnforceIf(b)
                self.model.Add(event.EndExpr() == alt.EndExpr()).OnlyEnforceIf(b)
                self.model.Add(event.SizeExpr() == alt.SizeExpr()).OnlyEnforceIf(b)
        if self.features['interface_name'] == 'picat':
            ev_start = event.get('start') if isinstance(event, dict) else None
            if ev_start:
                s1 = ev_start._picat_expr if hasattr(ev_start, '_picat_expr') else str(ev_start)
                for alt in array:
                    alt_s = alt.get('start') if isinstance(alt, dict) else None
                    if alt_s:
                        s2 = alt_s._picat_expr if hasattr(alt_s, '_picat_expr') else str(alt_s)
                        self.model.add_constraint_string(f"({s1} #= {s2})")

    def cp_all_dist_above(self, exprs, value):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.all_min_distance(exprs, value)
        if self.features['interface_name'] == 'ortools_cp':
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    b = self.model.NewBoolVar(f'all_dist_{i}_{j}')
                    self.model.Add(exprs[i] - exprs[j] >= value).OnlyEnforceIf(b)
                    self.model.Add(exprs[j] - exprs[i] >= value).OnlyEnforceIf(b.Not())
        if self.features['interface_name'] == 'picat':
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    a = exprs[i]._picat_expr if hasattr(exprs[i], '_picat_expr') else str(exprs[i])
                    b = exprs[j]._picat_expr if hasattr(exprs[j], '_picat_expr') else str(exprs[j])
                    self.model.add_constraint_string(f"abs({a} - {b}) #>= {value}")

    def cp_get_event_start(self, event_variable, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_of(event_variable, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return event_variable.StartExpr()
        if self.features['interface_name'] == 'picat':
            if isinstance(event_variable, dict):
                return event_variable.get('start')
            return event_variable

    def cp_get_event_end(self, event_variable, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_of(event_variable, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return event_variable.EndExpr()
        if self.features['interface_name'] == 'picat':
            if isinstance(event_variable, dict):
                return event_variable.get('end')
            return event_variable

    def cp_get_event_length(self, event_variable, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.length_of(event_variable, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return event_variable.EndExpr() - event_variable.StartExpr()
        if self.features['interface_name'] == 'picat':
            if isinstance(event_variable, dict):
                s = event_variable.get('start')
                e = event_variable.get('end')
                if s and e:
                    return PicatExpr(f"({e._picat_expr} - {s._picat_expr})")
            return None

    def cp_get_event_size(self, event_variable, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.size_of(event_variable, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return event_variable.SizeExpr()
        if self.features['interface_name'] == 'picat':
            if isinstance(event_variable, dict):
                return event_variable.get('size')
            return None

    def cp_get_presence(self, event_variable):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.presence_of(event_variable)
        if self.features['interface_name'] == 'ortools_cp':
            literals = event_variable.presence_literals()
            return literals[0] if literals else None
        if self.features['interface_name'] == 'picat':
            return None

    def cp_event_start_exactly_before_start(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_at_start(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay == second_event.StartExpr())
        if self.features['interface_name'] == 'picat':
            s1 = first_event.get('start') if isinstance(first_event, dict) else first_event
            s2 = second_event.get('start') if isinstance(second_event, dict) else second_event
            a = s1._picat_expr if hasattr(s1, '_picat_expr') else str(s1)
            b = s2._picat_expr if hasattr(s2, '_picat_expr') else str(s2)
            self.model.add_constraint_string(f"({a} + {delay} #= {b})")

    def cp_event_start_exactly_before_end(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_at_end(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay == second_event.EndExpr())
        if self.features['interface_name'] == 'picat':
            s1 = first_event.get('start') if isinstance(first_event, dict) else first_event
            e2 = second_event.get('end') if isinstance(second_event, dict) else second_event
            a = s1._picat_expr if hasattr(s1, '_picat_expr') else str(s1)
            b = e2._picat_expr if hasattr(e2, '_picat_expr') else str(e2)
            self.model.add_constraint_string(f"({a} + {delay} #= {b})")

    def cp_event_end_exactly_before_start(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_at_start(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.EndExpr() + delay == second_event.StartExpr())
        if self.features['interface_name'] == 'picat':
            e1 = first_event.get('end') if isinstance(first_event, dict) else first_event
            s2 = second_event.get('start') if isinstance(second_event, dict) else second_event
            a = e1._picat_expr if hasattr(e1, '_picat_expr') else str(e1)
            b = s2._picat_expr if hasattr(s2, '_picat_expr') else str(s2)
            self.model.add_constraint_string(f"({a} + {delay} #= {b})")

    def cp_event_end_exactly_before_end(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_at_end(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.EndExpr() + delay == second_event.EndExpr())
        if self.features['interface_name'] == 'picat':
            e1 = first_event.get('end') if isinstance(first_event, dict) else first_event
            e2 = second_event.get('end') if isinstance(second_event, dict) else second_event
            a = e1._picat_expr if hasattr(e1, '_picat_expr') else str(e1)
            b = e2._picat_expr if hasattr(e2, '_picat_expr') else str(e2)
            self.model.add_constraint_string(f"({a} + {delay} #= {b})")

    def cp_event_start_before_start(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_before_start(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay <= second_event.StartExpr())
        if self.features['interface_name'] == 'picat':
            s1 = first_event.get('start') if isinstance(first_event, dict) else first_event
            s2 = second_event.get('start') if isinstance(second_event, dict) else second_event
            a = s1._picat_expr if hasattr(s1, '_picat_expr') else str(s1)
            b = s2._picat_expr if hasattr(s2, '_picat_expr') else str(s2)
            self.model.add_constraint_string(f"({a} + {delay} #=< {b})")

    def cp_event_start_before_end(self, first_event, second_event, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_before_end(first_event, second_event, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(first_event.StartExpr() + delay <= second_event.EndExpr())
        if self.features['interface_name'] == 'picat':
            s1 = first_event.get('start') if isinstance(first_event, dict) else first_event
            e2 = second_event.get('end') if isinstance(second_event, dict) else second_event
            a = s1._picat_expr if hasattr(s1, '_picat_expr') else str(s1)
            b = e2._picat_expr if hasattr(e2, '_picat_expr') else str(e2)
            self.model.add_constraint_string(f"({a} + {delay} #=< {b})")

    def cp_event_end_before_start(self, event_one, event_two, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_before_start(event_one, event_two, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(event_one.EndExpr() + delay <= event_two.StartExpr())
        if self.features['interface_name'] == 'picat':
            e1 = event_one.get('end') if isinstance(event_one, dict) else event_one
            s2 = event_two.get('start') if isinstance(event_two, dict) else event_two
            a = e1._picat_expr if hasattr(e1, '_picat_expr') else str(e1)
            b = s2._picat_expr if hasattr(s2, '_picat_expr') else str(s2)
            self.model.add_constraint_string(f"({a} + {delay} #=< {b})")

    def cp_event_end_before_end(self, event_one, event_two, delay=0):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_before_end(event_one, event_two, delay)
        if self.features['interface_name'] == 'ortools_cp':
            self.model.Add(event_one.EndExpr() + delay <= event_two.EndExpr())
        if self.features['interface_name'] == 'picat':
            e1 = event_one.get('end') if isinstance(event_one, dict) else event_one
            e2 = event_two.get('end') if isinstance(event_two, dict) else event_two
            a = e1._picat_expr if hasattr(e1, '_picat_expr') else str(e1)
            b = e2._picat_expr if hasattr(e2, '_picat_expr') else str(e2)
            self.model.add_constraint_string(f"({a} + {delay} #=< {b})")

    def cp_forbid_event_start(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_start(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_start_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_start_high_{domain[1]}')
                self.model.Add(event.StartExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.StartExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])
        if self.features['interface_name'] == 'picat':
            s = event.get('start') if isinstance(event, dict) else event
            s_expr = s._picat_expr if hasattr(s, '_picat_expr') else str(s)
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                self.model.add_constraint_string(f"({s_expr} #< {domain[0]} #\\/ {s_expr} #> {domain[1]})")

    def cp_forbid_event_end(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_end(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_end_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_end_high_{domain[1]}')
                self.model.Add(event.EndExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.EndExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])
        if self.features['interface_name'] == 'picat':
            e = event.get('end') if isinstance(event, dict) else event
            e_expr = e._picat_expr if hasattr(e, '_picat_expr') else str(e)
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                self.model.add_constraint_string(f"({e_expr} #< {domain[0]} #\\/ {e_expr} #> {domain[1]})")

    def cp_forbid_event_overlap(self, event_variables, transition_matrix=None):
        if self.features['interface_name'] == 'cplex_cp':
            if transition_matrix is None:
                return self.model.no_overlap(event_variables)
            else:
                return self.model.no_overlap(event_variables, transition_matrix)
        if self.features['interface_name'] == 'ortools_cp':
            return self.model.AddNoOverlap(event_variables)
        if self.features['interface_name'] == 'picat':
            self.cp_add_no_overlap(event_variables)

    def cp_forbid_event_extent(self, event, function):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.forbid_extent(event, function)
        if self.features['interface_name'] == 'ortools_cp':
            forbidden_domains = function(event)
            for domain in forbidden_domains:
                b1 = self.model.NewBoolVar(f'forbid_extent_low_{domain[0]}')
                b2 = self.model.NewBoolVar(f'forbid_extent_high_{domain[1]}')
                self.model.Add(event.EndExpr() < domain[0]).OnlyEnforceIf(b1)
                self.model.Add(event.StartExpr() > domain[1]).OnlyEnforceIf(b2)
                self.model.AddBoolOr([b1, b2])
        if self.features['interface_name'] == 'picat':
            s = event.get('start') if isinstance(event, dict) else None
            e = event.get('end') if isinstance(event, dict) else None
            if s and e:
                s_expr = s._picat_expr if hasattr(s, '_picat_expr') else str(s)
                e_expr = e._picat_expr if hasattr(e, '_picat_expr') else str(e)
                forbidden_domains = function(event)
                for domain in forbidden_domains:
                    self.model.add_constraint_string(f"({e_expr} #< {domain[0]} #\\/ {s_expr} #> {domain[1]})")

    def cp_event_overlap_length(self, event1, event2, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.overlap_length(event1, event2, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            _max = 10000
            overlap_start = self.model.NewIntVar(0, _max, "overlap_start")
            overlap_end = self.model.NewIntVar(0, _max, "overlap_end")
            self.model.AddMaxEquality(overlap_start, [event1.StartExpr(), event2.StartExpr()])
            self.model.AddMinEquality(overlap_end, [event1.EndExpr(), event2.EndExpr()])
            result = self.model.NewIntVar(0, _max, "overlap_length")
            self.model.AddMaxEquality(result, [overlap_end - overlap_start, 0])
            return result
        if self.features['interface_name'] == 'picat':
            return None

    def cp_start_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.start_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.StartExpr())
        if self.features['interface_name'] == 'picat':
            s = event.get('start') if isinstance(event, dict) else event
            return function(s)

    def cp_end_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.end_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.EndExpr())
        if self.features['interface_name'] == 'picat':
            e = event.get('end') if isinstance(event, dict) else event
            return function(e)

    def cp_size_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.size_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.SizeExpr())
        if self.features['interface_name'] == 'picat':
            sz = event.get('size') if isinstance(event, dict) else event
            return function(sz)

    def cp_length_eval(self, event, function, absent_value=None):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.length_eval(event, function, absent_value)
        if self.features['interface_name'] == 'ortools_cp':
            return function(event.EndExpr() - event.StartExpr())
        if self.features['interface_name'] == 'picat':
            e = event.get('end') if isinstance(event, dict) else None
            s = event.get('start') if isinstance(event, dict) else None
            if s and e:
                return function(PicatExpr(f"({e._picat_expr} - {s._picat_expr})"))
            return None
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.if_then(boolvar, constraints)
        if self.features['interface_name'] == 'ortools_cp':
            if not isinstance(constraints, list):
                constraints = [constraints]
            for c in constraints:
                if not hasattr(c, 'OnlyEnforceIf'):
                    c = self.model.Add(c)
                c.OnlyEnforceIf(boolvar)
        if self.features['interface_name'] == 'picat':
            pass  # Picat does not support conditional enforcement directly

    def cp_add_int_var_from_domain(self, domain, name):
        if self.features['interface_name'] == 'cplex_cp':
            return self.model.integer_var(domain=domain, name=name)
        if self.features['interface_name'] == 'ortools_cp':
            from ortools.sat.python import cp_model as _cm
            if isinstance(domain, _cm.IntVar):
                return domain
            from ortools.util.python.sorted_interval_list import Domain
            if isinstance(domain, Domain):
                return self.model.NewIntVarFromDomain(domain, name)
            if isinstance(domain, set):
                sorted_vals = sorted(domain)
                intervals = []
                start = sorted_vals[0]
                prev = sorted_vals[0]
                for v in sorted_vals[1:]:
                    if v == prev + 1:
                        prev = v
                    else:
                        intervals.append([start, prev])
                        start = v
                        prev = v
                intervals.append([start, prev])
                return self.model.NewIntVarFromDomain(Domain.from_intervals(intervals), name)
            if isinstance(domain, list):
                if len(domain) > 0 and isinstance(domain[0], (list, tuple)):
                    return self.model.NewIntVarFromDomain(Domain.from_intervals(domain), name)
                return self.model.NewIntVarFromDomain(Domain.from_values(domain), name)
            return self.model.NewIntVarFromDomain(domain, name)
        if self.features['interface_name'] == 'picat':
            if isinstance(domain, set):
                sorted_vals = sorted(domain)
                min_v, max_v = sorted_vals[0], sorted_vals[-1]
            elif isinstance(domain, list):
                if len(domain) > 0 and isinstance(domain[0], (list, tuple)):
                    min_v = domain[0][0]
                    max_v = domain[-1][1]
                else:
                    min_v = min(domain)
                    max_v = max(domain)
            elif hasattr(domain, 'lb') and hasattr(domain, 'ub'):
                min_v = domain.lb
                max_v = domain.ub
            else:
                min_v = domain
                max_v = domain
            var = PicatVar(name, min_v, max_v, self.model, 'ivar')
            self.model.add_variable(var)
            return var

    def _picat_var_str(self, v):
        if hasattr(v, '_picat_expr'):
            return v._picat_expr
        return str(v)

    def _picat_list_str(self, lst):
        return ', '.join(self._picat_var_str(v) for v in lst)

    def cp_add_global_cardinality(self, variables, values, counts):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(variables)
            vals_str = ', '.join(str(v) for v in values)
            cnts_str = self._picat_list_str(counts) if isinstance(counts, list) else self._picat_var_str(counts)
            self.model.add_constraint_string(f"global_cardinality([{vars_str}], [{vals_str}], [{cnts_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.AddGlobalCardinalityWithCounts(variables, values, counts)

    def cp_add_scalar_product(self, coeffs, variables, target):
        if self.features['interface_name'] == 'picat':
            coeffs_str = ', '.join(str(c) for c in coeffs)
            vars_str = self._picat_list_str(variables)
            tgt = self._picat_var_str(target)
            self.model.add_constraint_string(f"scalar_product([{coeffs_str}], [{vars_str}], #=, {tgt})")
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.Add(sum(c * v for c, v in zip(coeffs, variables)) == target)

    def cp_add_nvalue(self, variables, target):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(variables)
            tgt = self._picat_var_str(target)
            self.model.add_constraint_string(f"nvalue({tgt}, [{vars_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            pass

    def cp_add_all_equal(self, expressions):
        if self.features['interface_name'] == 'picat':
            if isinstance(expressions, dict):
                var_list = list(expressions.values())
            elif isinstance(expressions, list):
                var_list = expressions
            else:
                var_list = [expressions]
            if len(var_list) >= 2:
                first = self._picat_var_str(var_list[0])
                for v in var_list[1:]:
                    self.model.add_constraint_string(f"({first} #= {self._picat_var_str(v)})")
        elif self.features['interface_name'] == 'ortools_cp':
            if isinstance(expressions, list) and len(expressions) > 1:
                for v in expressions[1:]:
                    self.model.Add(expressions[0] == v)

    def cp_add_neqs(self, expressions):
        if self.features['interface_name'] == 'picat':
            if isinstance(expressions, dict):
                var_list = list(expressions.values())
            elif isinstance(expressions, list):
                var_list = expressions
            else:
                var_list = [expressions]
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    a = self._picat_var_str(var_list[i])
                    b = self._picat_var_str(var_list[j])
                    self.model.add_constraint_string(f"({a} #\\= {b})")
        elif self.features['interface_name'] == 'ortools_cp':
            if isinstance(expressions, list) and len(expressions) >= 2:
                for i in range(len(expressions)):
                    for j in range(i + 1, len(expressions)):
                        self.model.Add(expressions[i] != expressions[j])

    def cp_add_lex_le(self, left, right):
        if self.features['interface_name'] == 'picat':
            left_str = self._picat_list_str(left)
            right_str = self._picat_list_str(right)
            self.model.add_constraint_string(f"lex_le([{left_str}], [{right_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            for l, r in zip(left, right):
                self.model.Add(l <= r)

    def cp_add_lex_lt(self, left, right):
        if self.features['interface_name'] == 'picat':
            left_str = self._picat_list_str(left)
            right_str = self._picat_list_str(right)
            self.model.add_constraint_string(f"lex_lt([{left_str}], [{right_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            n = len(left)
            for i in range(n):
                b = self.model.NewBoolVar(f'lex_lt_{i}')
                self.model.Add(left[i] < right[i]).OnlyEnforceIf(b)
                for j in range(i):
                    self.model.Add(left[j] == right[j]).OnlyEnforceIf(b)

    def cp_add_increasing(self, expressions):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(expressions)
            self.model.add_constraint_string(f"increasing([{vars_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            for i in range(len(expressions) - 1):
                self.model.Add(expressions[i] <= expressions[i + 1])

    def cp_add_increasing_strict(self, expressions):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(expressions)
            self.model.add_constraint_string(f"increasing_strict([{vars_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            for i in range(len(expressions) - 1):
                self.model.Add(expressions[i] < expressions[i + 1])

    def cp_add_decreasing(self, expressions):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(expressions)
            self.model.add_constraint_string(f"decreasing([{vars_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            for i in range(len(expressions) - 1):
                self.model.Add(expressions[i] >= expressions[i + 1])

    def cp_add_decreasing_strict(self, expressions):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(expressions)
            self.model.add_constraint_string(f"decreasing_strict([{vars_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            for i in range(len(expressions) - 1):
                self.model.Add(expressions[i] > expressions[i + 1])

    def cp_add_bin_packing(self, variables, bin_ids, bin_capacities):
        if self.features['interface_name'] == 'picat':
            vars_str = self._picat_list_str(variables)
            bins_str = self._picat_list_str(bin_ids)
            self.model.add_constraint_string(f"bin_packing([{vars_str}], [{bins_str}], {bin_capacities})")
        elif self.features['interface_name'] == 'ortools_cp':
            pass

    def cp_add_sliding_sum(self, variables, seq_length, min_sum, max_sum):
        if self.features['interface_name'] == 'picat':
            for i in range(len(variables) - seq_length + 1):
                window = variables[i:i + seq_length]
                vars_str = self._picat_list_str(window)
                self.model.add_constraint_string(f"(sum([{vars_str}]) #>= {min_sum})")
                self.model.add_constraint_string(f"(sum([{vars_str}]) #<= {max_sum})")
        elif self.features['interface_name'] == 'ortools_cp':
            for i in range(len(variables) - seq_length + 1):
                window = variables[i:i + seq_length]
                self.model.Add(sum(window) >= min_sum)
                self.model.Add(sum(window) <= max_sum)

    def cp_add_knapsack(self, weights, profits, capacity):
        if self.features['interface_name'] == 'picat':
            w_str = self._picat_list_str(weights)
            cap = self._picat_var_str(capacity)
            self.model.add_constraint_string(f"(sum([{w_str}]) #=< {cap})")
        elif self.features['interface_name'] == 'ortools_cp':
            pass

    def cp_add_diffn(self, x_starts, x_durations, x_ends, y_starts, y_durations, y_ends):
        if self.features['interface_name'] == 'picat':
            rects = []
            for i in range(len(x_starts)):
                xs = self._picat_var_str(x_starts[i])
                xd = self._picat_var_str(x_durations[i])
                xe = self._picat_var_str(x_ends[i])
                ys = self._picat_var_str(y_starts[i])
                yd = self._picat_var_str(y_durations[i])
                ye = self._picat_var_str(y_ends[i])
                rects.append(f"rect({xs},{xd},{xe},{ys},{yd},{ye})")
            self.model.add_constraint_string(f"diffn([{', '.join(rects)}])")
        elif self.features['interface_name'] == 'ortools_cp':
            self.model.AddNoOverlap2D(
                [self.model.NewIntervalVar(s, d, e, f'diffn_x_{i}') for i, (s, d, e) in enumerate(zip(x_starts, x_durations, x_ends))],
                [self.model.NewIntervalVar(s, d, e, f'diffn_y_{i}') for i, (s, d, e) in enumerate(zip(y_starts, y_durations, y_ends))]
            )

    def cp_add_disjunctive_tasks(self, start_vars, duration_vars, end_vars=None):
        if self.features['interface_name'] == 'picat':
            s_str = self._picat_list_str(start_vars)
            d_str = self._picat_list_str(duration_vars)
            self.model.add_constraint_string(f"serialized([{s_str}], [{d_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            intervals = [self.model.NewIntervalVar(s, d, e, f'disj_{i}')
                         for i, (s, d, e) in enumerate(zip(start_vars, duration_vars, end_vars if end_vars else [None]*len(start_vars)))]
            self.model.AddNoOverlap(intervals)

    def cp_add_seq_precede_chain(self, values, sequence):
        if self.features['interface_name'] == 'picat':
            seq_str = self._picat_list_str(sequence)
            vals_str = ', '.join(str(v) for v in values)
            self.model.add_constraint_string(f"seq_precede_chain([{vals_str}], [{seq_str}])")
        elif self.features['interface_name'] == 'ortools_cp':
            pass

    def cp_add_network_flow(self, arcs, demand, supply, flow_vars=None):
        if self.features['interface_name'] == 'picat':
            pass
        elif self.features['interface_name'] == 'ortools_cp':
            pass
