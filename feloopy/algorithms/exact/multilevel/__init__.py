"""Multi-level (bilevel) optimisation.

Models hierarchical decision-making where an upper-level problem
anticipates the lower-level's optimal response.

Classes:
    MultiLevelProblem -- defines and solves multi-level problems.

Error:
    MultiLevelError
"""

from .core import MultiLevelProblem, MultiLevelError
