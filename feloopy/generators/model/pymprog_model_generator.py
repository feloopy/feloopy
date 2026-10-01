# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import pymprog as pymprog_interface

def generate_model(features):
    try:
        pymprog_interface.end()
    except Exception:
        pass
    pymprog_interface.begin(features['model_name'])
    return pymprog_interface.model._prob_
