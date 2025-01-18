# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ortools.sat.python import cp_model as ortools_interface

def generate_model(features):
    return ortools_interface.CpModel()
