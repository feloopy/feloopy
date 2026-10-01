# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Automatic branching method selection.

Analyses model structure and selects between BaB, BaC, and BaP.
Called from ``feloopy.py`` when ``boost='branching'``.
"""

from ..base.logging import make_logger


def _detect_variant(captured):
    """Analyse captured model structure to choose the best branching variant.

    Parameters
    ----------
    captured : _AutomaticCaptureModel
        Structure produced by the automatic capture pass.

    Returns
    -------
    str
        ``'bab'``, ``'bac'``, or ``'bap'``.
    """
    n_integer = 0
    n_binary = 0
    n_continuous = 0
    n_constraints = len(captured.constraints)

    for (kind, _name) in captured.variables:
        if kind == 'ivar':
            n_integer += 1
        elif kind == 'bvar':
            n_binary += 1
        elif kind in ('pvar', 'fvar'):
            n_continuous += 1

    total_discrete = n_integer + n_binary

    if total_discrete == 0:
        return 'bab'

    if n_constraints > 2 * total_discrete:
        return 'bac'

    if n_continuous > total_discrete and n_constraints < 3 * total_discrete:
        return 'bap'

    return 'bac'


def run_automatic_branching(model_fn, directions, interface, solver,
                            captured=None, em=None, show_log=False, **kwargs):
    """Select and run the best branching variant.

    Parameters
    ----------
    model_fn : callable
        ``(m) -> m`` model factory.
    directions : list of str
        ``['min']`` or ``['max']``.
    interface : str
        Solver interface name.
    solver : str
        Solver name.
    captured : _AutomaticCaptureModel, optional
        Pre-built capture. When omitted it is derived from ``model_fn``.
    em : optional
        Environment model forwarded to the column-generation callbacks.
    show_log : bool
        Whether to print logs.

    Returns
    -------
    result
        BranchAndBoundResult / BranchAndCutResult / BranchAndPriceResult.
    """
    log = make_logger('Branching', show_log)

    if captured is None:
        from ..automatic_capture import _capture_from_model_fn
        captured = _capture_from_model_fn(model_fn)

    variant = _detect_variant(captured)
    log(f"Auto-selected variant: {variant}")

    if variant == 'bac':
        from .branch_and_cut import BranchAndCut
        bac = BranchAndCut.from_automatic(
            model_fn, directions, interface, solver, show_log=show_log)
        return bac.solve(
            interface=interface, solver=solver, directions=directions,
            show_log=show_log, **kwargs)

    if variant == 'bap':
        from .branch_and_price import BranchAndPrice
        bp = BranchAndPrice.from_automatic(
            model_fn, directions, interface, solver, show_log=show_log,
            captured=captured, em=em,
            cpu_threads=kwargs.get('cpu_threads'))
        return bp.solve(
            interface=interface, solver=solver, directions=directions,
            show_log=show_log, **kwargs)

    from .core import BranchAndBound
    bab = BranchAndBound.from_automatic(
        model_fn, directions, interface, solver, show_log=show_log)
    return bab.solve(
        interface=interface, solver=solver, directions=directions,
        show_log=show_log, **kwargs)
