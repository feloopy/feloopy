# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import itertools as it
from .jump_expression import JumpVar, jl_safe_name

sets = it.product


def generate_variable(model_object, variable_type, variable_name, variable_bound, variable_dim=0):

    if isinstance(variable_dim, set):
        variable_dim = [variable_dim]

    jl_name = jl_safe_name(variable_name)

    match variable_type:

        case 'pvar' | 'ptvar':
            if variable_dim == 0:
                decl = f"\n@variable(jlmodel, {jl_name} >= 0)"
            else:
                index_ranges = ', '.join([f'i{n} in {str(list(dim))}' for n, dim in enumerate(variable_dim, 1)])
                decl = f"\n@variable(jlmodel, {jl_name}[{index_ranges}] >= 0)"
            return JumpVar(jl_name, decl)

        case 'bvar' | 'btvar':
            if variable_dim == 0:
                decl = f"\n@variable(jlmodel, {jl_name}, Bin)"
            else:
                index_ranges = ', '.join([f'i{n} in {str(list(dim))}' for n, dim in enumerate(variable_dim, 1)])
                decl = f"\n@variable(jlmodel, {jl_name}[{index_ranges}], Bin)"
            return JumpVar(jl_name, decl)

        case 'ivar' | 'itvar':
            if variable_dim == 0:
                decl = f"\n@variable(jlmodel, {jl_name}, Int)"
            else:
                index_ranges = ', '.join([f'i{n} in {str(list(dim))}' for n, dim in enumerate(variable_dim, 1)])
                decl = f"\n@variable(jlmodel, {jl_name}[{index_ranges}], Int)"
            return JumpVar(jl_name, decl)

        case 'fvar' | 'ftvar':
            if variable_dim == 0:
                decl = f"\n@variable(jlmodel, {jl_name})"
            else:
                index_ranges = ', '.join([f'i{n} in {str(list(dim))}' for n, dim in enumerate(variable_dim, 1)])
                decl = f"\n@variable(jlmodel, {jl_name}[{index_ranges}])"
            return JumpVar(jl_name, decl)
