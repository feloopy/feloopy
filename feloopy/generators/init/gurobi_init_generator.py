# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from gurobipy import GRB

def set_init_value(features, variable, value, fix):
    if fix:
        variable.lb = value
        variable.ub = value
    else:
        variable.Start = value

def set_branch_priority(features, variable, priority):
    variable.BranchPriority = priority

def set_var_hint(variable, hint_val, hint_pri=0):
    variable.VarHintVal = hint_val
    variable.VarHintPri = hint_pri

def set_pwlobj(variable, x_vals, y_vals):
    variable.PWLObj = list(zip(x_vals, y_vals))
