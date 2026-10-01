import numpy as np


class MultiObjectiveError(Exception):
    """Raised when multi-objective optimisation configuration or solve fails."""


_DIR_MAP = {'max': -1, 'min': 1}

_VALID_DIRECTIONS = ('min', 'max')
_VALID_APPROACHES = ('ecm', 'nwsm')
_VALID_WEIGHT_METHODS = ('dirichlet', 'random')

_CP_INTERFACES = ('ortools_cp', 'cplex_cp', 'picat')


def _validate_directions(directions):
    if not directions or not isinstance(directions, (list, tuple)):
        raise MultiObjectiveError(
            "directions must be a non-empty list of 'min' or 'max' strings")
    for i, d in enumerate(directions):
        if d not in _VALID_DIRECTIONS:
            raise MultiObjectiveError(
                f"directions[{i}] = {d!r} must be 'min' or 'max'")


def _validate_approach(approach):
    if approach not in _VALID_APPROACHES:
        raise MultiObjectiveError(
            f"approach = {approach!r} must be one of {_VALID_APPROACHES}")


def _validate_weights(weights, M):
    if len(weights) == 0:
        return
    if len(weights) != M:
        raise MultiObjectiveError(
            f"weights length ({len(weights)}) must match number of objectives ({M})")
    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-6:
        raise MultiObjectiveError(f"weights must sum to 1.0, got {wsum:.6f}")
    if any(w < 0 for w in weights):
        raise MultiObjectiveError("weights must be non-negative")
