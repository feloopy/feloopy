# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import mip as mip_interface

mip_solver_map = {
    'cbc': 'CBC',
    'cplex': 'CPLEX',
    'glpk': 'GLPK',
    'gurobi': 'GRB',
}

def generate_model(features):
    solver_name = features.get('solver_name', 'cbc')
    mapped = mip_solver_map.get(solver_name, 'CBC')
    return mip_interface.Model(features['model_name'], solver_name=mapped)
