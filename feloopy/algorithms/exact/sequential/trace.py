import numpy as np


class Trace:
    __slots__ = ('states', 'decisions', 'objectives', 'costs', 'feasible',
                 'exogenous', 'timestamps')

    def __init__(self):
        self.states = []
        self.decisions = []
        self.objectives = []
        self.costs = []
        self.feasible = []
        self.exogenous = []
        self.timestamps = []

    def record(self, state, decision, objective, cost, feasible,
               exogenous=None, t=None):
        if isinstance(state, dict):
            self.states.append({k: np.asarray(v, dtype=float).copy()
                               for k, v in state.items()})
        else:
            self.states.append(np.asarray(state, dtype=float).copy())
        if isinstance(decision, dict):
            self.decisions.append({k: np.asarray(v, dtype=float).copy()
                                  for k, v in decision.items()})
        elif decision is not None:
            self.decisions.append(np.asarray(decision, dtype=float).copy())
        else:
            self.decisions.append(None)
        self.objectives.append(float(objective))
        self.costs.append(float(cost))
        self.feasible.append(bool(feasible))
        if exogenous is not None:
            if isinstance(exogenous, dict):
                self.exogenous.append({k: np.asarray(v, dtype=float).copy()
                                       for k, v in exogenous.items()})
            else:
                self.exogenous.append(np.asarray(exogenous, dtype=float).copy())
        self.timestamps.append(t)

    def to_dict(self):
        return {
            'states': np.array(self.states),
            'decisions': self.decisions,
            'objectives': np.array(self.objectives),
            'costs': np.array(self.costs),
            'feasible': np.array(self.feasible),
            'exogenous': (np.array(self.exogenous)
                          if self.exogenous else np.array([])),
            'timestamps': np.array(self.timestamps),
        }
