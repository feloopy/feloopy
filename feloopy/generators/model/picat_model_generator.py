# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


class PicatModel:
    """Container for a Picat model, accumulating variables, constraints, and objectives."""

    def __init__(self, features):
        self.features = features
        self.variables = []
        self.variable_map = {}
        self.name_map = {}
        self.constraints = []
        self.constraint_strings = []
        self.objectives = []
        self.directions = []
        self.output_vars = []
        self.search_options = []
        self.intervals = []
        self.pragmas = []

    def add_variable(self, var):
        self.variables.append(var)
        self.variable_map[var.name] = var

    def register_name(self, original_name, safe_name):
        self.name_map[original_name] = safe_name

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def add_constraint_string(self, constraint_str):
        self.constraint_strings.append(constraint_str)

    def add_objective(self, expr, direction):
        if isinstance(expr, (int, float)):
            expr_str = str(expr)
        elif hasattr(expr, '_picat_expr'):
            expr_str = expr._picat_expr
        else:
            expr_str = str(expr)
        self.objectives.append((expr_str, direction))
        self.directions.append(direction)

    def add_output_var(self, name):
        if name not in self.output_vars:
            self.output_vars.append(name)

    def add_interval(self, interval_info):
        self.intervals.append(interval_info)

    def add_pragma(self, pragma_str):
        self.pragmas.append(pragma_str)


def generate_model(features):
    return PicatModel(features)
