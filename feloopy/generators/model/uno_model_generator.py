# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


class UnoModel:
    """Wrapper for Uno model that collects problem data before building the unopy.Model."""

    def __init__(self):
        self.number_variables = 0
        self.variables_lower_bounds = []
        self.variables_upper_bounds = []
        self.objective = None
        self.objective_direction = None
        self.objective_gradient = None
        self.constraints = []
        self.constraints_lower_bounds = []
        self.constraints_upper_bounds = []
        self.jacobian_row_indices = []
        self.jacobian_column_indices = []
        self.jacobian = None
        self.number_constraints = 0
        self.number_jacobian_nonzeros = 0
        self.initial_primal_iterate = None
        self.lagrangian_hessian = None
        self.number_hessian_nonzeros = 0
        self.hessian_row_indices = []
        self.hessian_column_indices = []
        self.lagrangian_sign_convention = None


def generate_model(features):
    model = UnoModel()
    return model
