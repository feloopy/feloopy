# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from typing import List, Optional
from itertools import product as sets 
from ..operators.update_operators import update_variable_features

try:
    from ..generators.picat_expression import PicatVar
except ImportError:
    PicatVar = None

class EventVariable:
    """Placeholder class for event variables."""

class EventVariableClass:
    """
    Class that provides methods to create event variables.
    """

    def _create_cp_interval_var(self, name, start, size, end, optional=False):
        """Helper to create CP interval variables with proper None handling."""
        model = self.model
        
        # Handle size
        if size is None:
            raise ValueError(f"Interval variable '{name}' requires a size")
        
        # Handle start
        if start is None:
            start_var = model.NewIntVar(0, 10000, f"{name}_start")
            start = start_var
        elif isinstance(start, int):
            start_var = model.NewIntVar(start, start, f"{name}_start")
            start = start_var
            
        # Handle end
        if end is None:
            end_var = model.NewIntVar(0, 10000, f"{name}_end")
            end = end_var
        elif isinstance(end, int):
            end_var = model.NewIntVar(end, end, f"{name}_end")
            end = end_var
        
        # Create interval variable
        if optional:
            return model.NewOptionalIntervalVar(start, size, end, True, name)
        else:
            return model.NewIntervalVar(start, size, end, name)

    def evar(
        self,
        name: str,
        event: List[Optional[float]] = [None, None, None],
        dim: List[int] = 0,
        optional: bool = False
    ) -> EventVariable:
        """
        Creates and returns an event (interval) variable.

        Parameters
        ----------
        name : str
            Name of this variable.
        event : List[Optional[float]], optional
            [size, start, end]. Default: [None, None, None].
        dim : List[int], optional
            Dimensions of this variable. Default: 0.
        optional : bool, optional
            Flag indicating whether the variable is optional. Default: False.

        Returns
        -------
        EventVariable
            An event (interval) variable.
        """

        dim = self.fix_ifneeded(dim)

        self.features = update_variable_features(name, dim, None, 'event_variable_counter', self.features)

        if len(event) == 1:
            event = [event[0], None, None]

        if dim == 0:

            if self.features['interface_name'] == 'cplex_cp':
                self.features['variables'][("evar", name)] = self.model.interval_var(start=event[1], size=event[0], end=event[2], name=name, optional=optional)
                self.features['dimensions'][name] = dim
                return self.features['variables'][("evar", name)]
                
            elif self.features['interface_name'] == 'ortools_cp':
                self.features['variables'][("evar", name)] = self._create_cp_interval_var(name, event[1], event[0], event[2], optional)
                self.features['dimensions'][name] = dim
                return self.features['variables'][("evar", name)]

            elif self.features['interface_name'] == 'picat':
                lb = event[1] if event[1] is not None else 0
                ub = event[2] if event[2] is not None else 10000
                if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
                    lb = int(lb)
                    ub = int(ub)
                    start_var_name = f"{name}_start"
                    end_var_name = f"{name}_end"
                    start_var = PicatVar(start_var_name, lb, ub, self.model, 'ivar')
                    end_var = PicatVar(end_var_name, lb, ub, self.model, 'ivar')
                    self.model.add_variable(start_var)
                    self.model.add_variable(end_var)
                    self.model.add_constraint_string(f"({end_var._picat_expr} - {start_var._picat_expr}) #= {event[0]}")
                    self.features['variables'][("evar", name)] = {'start': start_var, 'end': end_var, 'size': event[0]}
                    self.features['dimensions'][name] = dim
                    return self.features['variables'][("evar", name)]
        else:

            if self.features['interface_name'] == 'cplex_cp':

                if len(dim) == 1:
                    self.features['variables'][("evar", name)] = {key: self.model.interval_var(start=event[1], size=event[0], end=event[2], name=f"{name}{key}", optional=optional) for key in dim[0]}
                    self.features['dimensions'][name] = dim
                    return self.features['variables'][("evar", name)] 
                    
                else:
                    self.features['variables'][("evar", name)]  = {key: self.model.interval_var(start=event[1], size=event[0], end=event[2], name=f"{name}{key}", optional=optional) for key in sets(*dim)}
                    self.features['dimensions'][name] = dim
                    return  self.features['variables'][("evar", name)] 

            elif self.features['interface_name'] == 'ortools_cp':

                if len(dim) == 1:
                    self.features['variables'][("evar", name)] = {key: self._create_cp_interval_var(f"{name}{key}", event[1], event[0], event[2], optional) for key in dim[0]}
                    self.features['dimensions'][name] = dim
                    return self.features['variables'][("evar", name)]
                    
                else:
                    self.features['variables'][("evar", name)]  = {key: self._create_cp_interval_var(f"{name}{key}", event[1], event[0], event[2], optional) for key in sets(*dim)}
                    self.features['dimensions'][name] = dim
                    return self.features['variables'][("evar", name)]

            elif self.features['interface_name'] == 'picat':
                lb = event[1] if event[1] is not None else 0
                ub = event[2] if event[2] is not None else 10000
                result = {}
                if len(dim) == 1:
                    for key in dim[0]:
                        sname = f"{name}{key}_start"
                        ename = f"{name}{key}_end"
                        sv = PicatVar(sname, lb, ub, self.model, 'ivar')
                        ev = PicatVar(ename, lb, ub, self.model, 'ivar')
                        self.model.add_variable(sv)
                        self.model.add_variable(ev)
                        self.model.add_constraint_string(f"({ev._picat_expr} - {sv._picat_expr}) #= {event[0]}")
                        result[key] = {'start': sv, 'end': ev, 'size': event[0]}
                else:
                    for key in sets(*dim):
                        sname = f"{name}{key}_start"
                        ename = f"{name}{key}_end"
                        sv = PicatVar(sname, lb, ub, self.model, 'ivar')
                        ev = PicatVar(ename, lb, ub, self.model, 'ivar')
                        self.model.add_variable(sv)
                        self.model.add_variable(ev)
                        self.model.add_constraint_string(f"({ev._picat_expr} - {sv._picat_expr}) #= {event[0]}")
                        result[key] = {'start': sv, 'end': ev, 'size': event[0]}
                self.features['variables'][("evar", name)] = result
                self.features['dimensions'][name] = dim
                return result
