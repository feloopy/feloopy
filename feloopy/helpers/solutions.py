# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np


def _normalize_init_solutions(init, variables_bound, variables_spread):
    """Convert init solutions from original variable space to [0,1] normalized space.

    Formats:
      - list[dict]:   each dict is {var_name: value} -> multiple population rows
      - dict:         {var_name: value} -> single population row
      - list / 1D np.array: single solution in flat variable order
      - 2D np.array:  population matrix.  If all values in [0,1] already,
                      treated as pre-normalized; otherwise normalized.

    Returns np.ndarray (n_solutions, n_vars) in [0,1], or None.
    """
    if init is None:
        return None

    n_vars = sum(variables_spread[k][1] - variables_spread[k][0] for k in variables_spread)
    var_order = list(variables_spread.keys())

    def _flatten_value(val):
        """Flatten one variable's value to a 1-D array in the order the
        heuristic decoder reads the flat agent slice.

        - ``dict`` values (set-declared variables, e.g. ``pvar(dim={...})``,
          which exact solvers return as ``{index_tuple: value}``) keep their
          insertion order, which is the ``list(set)`` order the decoder uses
          as its key map.
        - arrays of any rank are raveled in C order, matching the decoder's
          ``np.reshape(agent[spread], dim)``.
        """
        if isinstance(val, dict):
            parts = [np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in val.values()]
            return np.concatenate(parts) if parts else np.zeros(0)
        return np.atleast_1d(np.asarray(val, dtype=float)).ravel()

    def _place_dict_into_row(d, result, row_idx):
        for key, val in d.items():
            if key not in variables_spread:
                continue
            lb, ub = variables_bound[key]
            start, end = variables_spread[key]
            span = end - start
            flat = _flatten_value(val)
            for j in range(span):
                v = flat[j] if j < flat.size else (lb + ub) / 2.0
                result[row_idx, start + j] = (v - lb) / (ub - lb) if ub > lb else 0.5

    def _normalize_array(arr):
        result = np.full((arr.shape[0], n_vars), 0.5)
        for i in range(arr.shape[0]):
            col = 0
            for key in var_order:
                lb, ub = variables_bound[key]
                start, end = variables_spread[key]
                span = end - start
                for j in range(span):
                    v = arr[i, col + j] if col + j < arr.shape[1] else (lb + ub) / 2.0
                    result[i, start + j] = (v - lb) / (ub - lb) if ub > lb else 0.5
                col += span
        return result

    if isinstance(init, dict):
        result = np.random.rand(1, n_vars)
        _place_dict_into_row(init, result, 0)
        return result

    if isinstance(init, list) and len(init) > 0 and isinstance(init[0], dict):
        result = np.random.rand(len(init), n_vars)
        for i, d in enumerate(init):
            _place_dict_into_row(d, result, i)
        return result

    arr = np.asarray(init, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        return _normalize_array(arr)

    if arr.ndim == 2:
        all_in_01 = np.all((arr >= 0) & (arr <= 1))
        if all_in_01 and n_vars == arr.shape[1]:
            return arr.copy()
        return _normalize_array(arr)

    return None
