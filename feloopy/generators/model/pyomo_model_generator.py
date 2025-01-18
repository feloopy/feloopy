# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyomo.environ as pyomo_interface

def generate_model(features):

    return pyomo_interface.ConcreteModel(name=features['model_name'])
