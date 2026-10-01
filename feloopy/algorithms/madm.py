# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np

from ..operators.set_operators import sets
from ..operators.math_operators import normalize_array_sum

_LP_SATISFY_RATIOS = np.array([
    0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45,
    1.49, 1.51, 1.48, 1.56, 1.57, 1.59,
])

_MAX_TIE_BREAKS = 25
_TIE_REL_TOL = 1e-6
_TIE_ABS_TOL = 1e-9

def _as_matrix(dataset, positive=False, name='dataset'):
    dataset = np.asarray(dataset, dtype=float)
    if dataset.ndim != 2:
        raise ValueError(f"{name} expects a 2-D array, got shape {dataset.shape}.")
    if positive and np.any(dataset <= 0):
        raise ValueError(f"{name} expects strictly positive entries.")
    return dataset


def _build_lp(dataset, interface_name):
    num_criteria = dataset.shape[0]
    import feloopy as flp
    m = flp.model('exact', 'linear programming method', interface_name)
    I = m.set('', [0, num_criteria])
    J = m.set('', [0, num_criteria])
    z = m.pvar('z', [I, J])
    x = m.fvar('x', [I])
    y = m.fvar('y', [I, J])

    for i, j in sets(I, J):
        if i != j:
            m.con(x[i] - x[j] - y[i, j] == np.log(dataset[i, j]))

    for i, j in sets(I, J):
        if i < j:
            m.con(z[i, j] >= y[i, j])
            m.con(z[i, j] >= y[j, i])

    m.con(x[0] == 0)

    for i, j in sets(I, J):
        if i < j and dataset[i, j] > 1:
            m.con(x[i] - x[j] >= 0)
        if i < j and all(dataset[i, k] > dataset[j, k] for k in range(num_criteria) if k != i != j) \
                and any(dataset[i, q] > dataset[j, q] for q in range(num_criteria)):
            m.con(x[i] - x[j] >= 0)

    return m, z, x, I, J


def lp_method(dataset, interface_name='pulp', solver_name='cbc'):
    """Criteria weights from a pairwise comparison matrix by linear programming.

    Returns
    -------
    (weights, inconsistency) : tuple
        ``weights`` sums to 1; ``inconsistency`` is the consistency ratio, or
        an explanatory string when the matrix has more criteria than the
        tabulated random indices cover.
    """
    dataset = _as_matrix(dataset, positive=True, name='lp_method')
    if dataset.shape[0] != dataset.shape[1]:
        raise ValueError(
            f"lp_method expects a square pairwise comparison matrix, got shape {dataset.shape}.")
    num_criteria = dataset.shape[0]
    pairs = [(i, j) for i in range(num_criteria - 1) for j in range(i + 1, num_criteria)]

    m, z, x, I, J = _build_lp(dataset, interface_name)
    m.obj(sum(z[i, j] for i, j in pairs))
    m.sol(['min'], solver_name)
    z_star = m.get_obj()
    if z_star is None:
        raise RuntimeError("lp_method could not solve the first-stage problem.")

    m, z, x, I, J = _build_lp(dataset, interface_name)
    m.obj(m.lin_max([z[i, j] for i, j in sets(I, J) if i < j], 'pvar', ub_max=10e9))
    m.con(sum(z[i, j] for i, j in pairs) <= z_star)
    m.sol(['min'], solver_name)
    z_star = m.get_obj()

    try:
        inconsistency = (2 * z_star / (num_criteria * (num_criteria - 1))) / _LP_SATISFY_RATIOS[num_criteria]
    except Exception:
        inconsistency = f'N/A. Num Criteria > {len(_LP_SATISFY_RATIOS)}'

    weights = normalize_array_sum(np.exp(m.get_numpy_var('x')))
    return weights, inconsistency


def la_method(dataset, criterion_type, weights, interface_name='pulp', solver_name='cbc'):
    """Rank alternatives by a linear-assignment problem over weighted ranks.

    Each alternative gets a rank position (0 = best) under every criterion;
    the assignment problem then selects one alternative per position so that
    the total weight of matching ranks is maximal.

    Returns
    -------
    list[int]
        Alternative indices ordered best first.
    """
    dataset = _as_matrix(dataset, name='la_method')
    num_rows, num_columns = dataset.shape

    if weights is None:
        raise ValueError("la_method requires criteria weights.")
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != num_columns:
        raise ValueError(
            f"la_method got {weights.size} weights for {num_columns} criteria.")

    if criterion_type is None:
        criterion_type = ['max'] * num_columns
    elif isinstance(criterion_type, str):
        criterion_type = [criterion_type] * num_columns
    if len(criterion_type) != num_columns:
        raise ValueError(
            f"la_method got {len(criterion_type)} criterion directions for {num_columns} criteria.")

    rank_positions = np.empty((num_rows, num_columns), dtype=np.int64)
    for i in range(num_columns):
        criterion_ = criterion_type[i]
        if criterion_ == 'max':
            order = np.argsort(dataset[:, i], kind='quicksort')[::-1]
        elif criterion_ == 'min':
            order = np.argsort(dataset[:, i], kind='quicksort')
        else:
            raise ValueError("Invalid criterion_type. Use 'max' or 'min'.")
        positions = np.empty(num_rows, dtype=np.int64)
        positions[order] = np.arange(num_rows)
        rank_positions[:, i] = positions

    h = np.zeros((num_rows, num_rows), dtype=float)
    for j in range(num_rows):
        for i in range(num_columns):
            h[j, rank_positions[j, i]] += weights[i]

    import feloopy as flp
    m = flp.model('exact', 'common weight data envelopment analysis', interface_name)
    I = m.set('', [0, num_rows])
    J = m.set('', [0, num_rows])
    x = m.pvar('x', [I, J])

    m.obj(sum(h[i, j] * x[i, j] for i, j in sets(I, J)))

    for j in J:
        m.con(sum(x[i, j] for i in I) == 1)

    for i in I:
        m.con(sum(x[i, j] for j in J) == 1)

    m.sol(['max'], solver_name)
    assignment = m.get_numpy_var('x')

    ranks_list = []
    for position in range(assignment.shape[1]):
        for alternative in range(assignment.shape[0]):
            if assignment[alternative, position] > 0.5:
                ranks_list.append(alternative)
    return ranks_list


def _duplicates(ranks):
    return [i for i, item in enumerate(ranks) if np.count_nonzero(ranks == item) > 1]


def cwdea_method(dataset, interface_name='pulp', solver_name='cbc'):
    """Common-weight DEA: efficiency scores used as both ranks and weights.

    Minimises the largest additive shortfall over a single common set of
    criterion weights.  When two DMUs end up with the same score (or a weight
    collapses to zero) the objective is perturbed and the model re-solved so
    that ties separate.  The perturbed solves are pinned to the primary
    optimum, so they can only pick a different point on the optimal face and
    can never make the scores worse.

    Returns
    -------
    (ranks, weights) : tuple
        ``ranks`` holds one efficiency score per row of ``dataset`` (higher is
        better); ``weights`` holds one weight per column.
    """
    dataset = _as_matrix(dataset, name='cwdea_method')
    num_dmus, num_criteria = dataset.shape

    def _solve(perturbation, tied, z_star=None):
        import feloopy as flp
        m = flp.model('exact', 'common weight data envelopment analysis', interface_name)
        I = m.set('', [0, num_dmus])
        J = m.set('', [0, num_criteria])
        z = m.fvar('z')
        d = m.pvar('r', [I])
        w = m.pvar('w', [J], bound=[0.000001, None])

        if perturbation:
            m.obj(z - perturbation * sum(d[i] for i in tied))
        else:
            m.obj(z)

        for i in I:
            m.con(sum(dataset[i, j] * w[j] for j in J) + d[i] == 1)

        for i in I:
            m.con(z >= d[i])

        if perturbation and z_star is not None:
            m.con(z <= z_star * (1 + _TIE_REL_TOL) + _TIE_ABS_TOL)

        m.sol(['min'], solver_name)
        if m.em.status != 'Optimal':
            raise RuntimeError(
                f"cwdea_method solver returned status '{m.em.status}' instead of 'Optimal'.")
        return 1 - m.get_numpy_var('r'), m.get_numpy_var('w')

    ranks, weights = _solve(0.0, [])
    z_star = 1.0 - float(np.min(ranks))
    zero_weight_indices = [i for i, item in enumerate(weights) if np.abs(item) < 10e-6]
    duplicate_indices = _duplicates(ranks)

    if len(duplicate_indices) == 0 and len(zero_weight_indices) == 0:
        return ranks, weights

    k = 0.01
    for _ in range(_MAX_TIE_BREAKS):
        if len(duplicate_indices) == 0:
            break
        ranks, weights = _solve(k, duplicate_indices, z_star)
        duplicate_indices = _duplicates(ranks)
        k += 0.01

    return ranks, weights
