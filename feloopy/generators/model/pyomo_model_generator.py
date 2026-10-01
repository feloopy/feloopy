# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pyomo.environ as pyomo_interface

from ...helpers._pyomo_types import register_pyomo_numeric_types

register_pyomo_numeric_types()


def generate_model(features):

    return pyomo_interface.ConcreteModel(name=features['model_name'])
