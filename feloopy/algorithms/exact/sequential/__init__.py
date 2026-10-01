"""Sequential decision making under uncertainty.

Implements Powell's unified framework for sequential decisions with
four policy classes: PFA, CFA, VFA, and DLA.

Classes:
    SequentialDecisionProblem -- simulation-based sequential framework.
    PFA, CFA, VFA, DLA        -- Powell's four policy classes.
    SDMMixin                   -- mixin providing SDM behaviour to models.
    sdm_model                  -- feloopy model extended for SDM.

Results:
    SimulationResult, OptimizationResult

Convenience:
    sol_sequential -- high-level one-call entry point.
    sdm            -- flp.sdm(...) shortcut for flp.search(..., method='sequential').
"""

from .enums import (
    SequentialError,
    PolicyType,
    _resolve_dim,
)
from .proxy import (
    _SDMProxy,
    _VarProxy,
    _SelectProxy,
    _Proxy,
    _RngProxy,
    _replay,
    pmax,
    pmin,
    StateDict,
    _FieldContext,
)
from .trace import Trace
from .sdm_mixin import SDMMixin
from .sdm_model import sdm_model
from .policies import PFA, CFA, VFA, DLA
from .build_model import _make_build_model_fn
from .result import SimulationResult, OptimizationResult
from .core import SequentialDecisionProblem
from .solvers import sol_sequential, sdm

__all__ = [
    'SequentialError',
    'PolicyType',
    'SequentialDecisionProblem',
    'PFA',
    'CFA',
    'VFA',
    'DLA',
    'SimulationResult',
    'OptimizationResult',
    'sol_sequential',
    'sdm',
    'sdm_model',
    'SDMMixin',
    'Trace',
    '_SDMProxy',
    '_VarProxy',
    '_SelectProxy',
    '_Proxy',
    '_RngProxy',
    '_replay',
    'pmax',
    'pmin',
    'StateDict',
    '_FieldContext',
    '_resolve_dim',
    '_make_build_model_fn',
]
