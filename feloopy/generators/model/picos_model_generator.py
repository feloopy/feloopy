# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import picos as picos_interface

def generate_model(features):

    return picos_interface.Problem(name=features['model_name'])
