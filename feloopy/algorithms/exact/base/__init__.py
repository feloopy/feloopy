# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Shared utilities for decomposition algorithms."""

from .logging import make_logger
from .result import DecompositionResult
from .model import (
    create_model,
    solve_model,
    relax_integrality,
    get_var_value,
    get_obj,
    get_status,
    get_dual,
    build_var_col_map,
    capture_original_bounds,
    fix_variable_bound,
    unfix_variable_bound,
)
