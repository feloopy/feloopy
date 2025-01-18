# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import gurobipy as gurobi_interface

def generate_model(features):

    return gurobi_interface.Model(features['model_name'])
