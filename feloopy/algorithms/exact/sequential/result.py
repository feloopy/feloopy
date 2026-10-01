import numpy as np


class SimulationResult:
    """Container for simulation output with convenience properties."""

    def __init__(self, total_costs, discounted_costs, n_infeasible,
                 traces, T, N, discount, policy_name, policy_type):
        self.total_costs = total_costs
        self.discounted_costs = discounted_costs
        self.n_infeasible = n_infeasible
        self.traces = traces
        self.T = T
        self.N = N
        self.discount = discount
        self.policy_name = policy_name
        self.policy_type = policy_type

    @property
    def stats(self):
        mc = self.total_costs
        dc = self.discounted_costs
        inf = self.n_infeasible
        total_decisions = self.T * self.N
        return {
            'policy_name': self.policy_name,
            'policy_type': self.policy_type,
            'T': self.T,
            'N': self.N,
            'discount': self.discount,
            'mean_cost': float(np.mean(mc)),
            'std_cost': float(np.std(mc)),
            'min_cost': float(np.min(mc)),
            'max_cost': float(np.max(mc)),
            'median_cost': float(np.median(mc)),
            'mean_discounted': float(np.mean(dc)),
            'std_discounted': float(np.std(dc)),
            'percentile_5': float(np.percentile(mc, 5)),
            'percentile_95': float(np.percentile(mc, 95)),
            'total_infeasible': int(np.sum(inf)),
            'infeasible_rate': float(np.sum(inf) / max(1, total_decisions)),
            'mean_infeasible_per_sim': float(np.mean(inf)),
        }

    @property
    def traces_as_dicts(self):
        if self.traces is None:
            return None
        return [t.to_dict() for t in self.traces]

    def summary(self):
        s = self.stats
        lines = [
            f"Policy: {s['policy_name']} ({s['policy_type']})",
            f"Horizon T={s['T']},  Simulations N={s['N']},  "
            f"Discount={s['discount']}",
            f"  Total cost:   mean={s['mean_cost']:.4f}  "
            f"std={s['std_cost']:.4f}  "
            f"[{s['min_cost']:.4f}, {s['max_cost']:.4f}]",
            f"  Discounted:   mean={s['mean_discounted']:.4f}  "
            f"std={s['std_discounted']:.4f}",
            f"  5th pct={s['percentile_5']:.4f}  "
            f"95th pct={s['percentile_95']:.4f}",
            f"  Feasibility:  {s['total_infeasible']} violations "
            f"({s['infeasible_rate']*100:.1f}%)",
        ]
        return "\n".join(lines)


class OptimizationResult:
    """Container for policy optimisation output."""

    def __init__(self, best_theta, best_stats, init_theta, init_stats,
                 convergence, n_evaluations, solver, direction='min'):
        self.best_theta = best_theta
        self.best_stats = best_stats
        self.init_theta = init_theta
        self.init_stats = init_stats
        self.convergence = convergence
        self.n_evaluations = n_evaluations
        self.solver = solver
        self.direction = direction

    def summary(self):
        b = self.best_stats
        i = self.init_stats
        improvement = 0.0
        if i.get('mean_cost', 0) != 0:
            if self.direction == 'max':
                improvement = (b['mean_cost'] - i['mean_cost']) / i['mean_cost'] \
                    * 100
            else:
                improvement = (i['mean_cost'] - b['mean_cost']) / i['mean_cost'] \
                    * 100
        lines = [
            f"Optimisation result ({self.solver})",
            f"  Evaluations: {self.n_evaluations}",
            f"  Initial:  theta={np.round(self.init_theta, 4)}, "
            f"cost={i.get('mean_cost', 0):.4f}",
            f"  Optimised: theta={np.round(self.best_theta, 4)}, "
            f"cost={b.get('mean_cost', 0):.4f}",
            f"  Improvement: {improvement:+.1f}%",
        ]
        return "\n".join(lines)
