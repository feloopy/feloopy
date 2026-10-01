# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from .option_utils import normalize_options, get_epoch_value


def generate_solution(solver_name, AlgOptions, Fitness, ToTalVariableCounter, ObjectivesDirections, ObjectiveBeingOptimized, number_of_times, show_plots, save_plots, show_log, init_solutions=None):

    import timeit
    from pymoo.core.problem import Problem

    pymoo_name_map = {
        "nsga2": "ns-ga-ii", "dnsga2": "d-ns-ga-ii", "nsga3": "ns-ga-iii",
        "moead": "mo-ea-d", "ctaea": "cta-ea", "rnsga2": "r-ns-ga-ii",
        "rnsga3": "r-ns-ga-iii", "unsga3": "u-ns-ga-iii", "spea2": "sp-ea-ii",
        "smsemoa": "sms-ea", "rvea": "rv-ea", "agemoa": "age-mo-ea",
        "agemoea": "age-mo-ea", "agemoa2": "age-mo-ea-ii", "agemoea2": "age-mo-ea-ii",
        "gde3": "gde3", "kgb": "kgb", "cmopso": "cmopso", "nsde": "nsde",
        "omni": "omni", "pinsga2": "pinsga2", "nsder": "nsder",
    }
    solver_name = pymoo_name_map.get(solver_name.lower(), solver_name)
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions

    import numpy as np

    if show_log:
        verbose=True
    
    else:
        verbose=False

    norm = normalize_options(AlgOptions, "pymoo")
    AlgOptions = dict(AlgOptions)

    termination_keys = set()
    if 'epoch' in AlgOptions and 'n_gen' not in AlgOptions:
        AlgOptions['n_gen'] = AlgOptions.pop('epoch')
    if 'max_evaluations' in AlgOptions and 'n_eval' not in AlgOptions:
        AlgOptions['n_eval'] = AlgOptions.pop('max_evaluations')

    for tk in ('n_gen', 'n_eval', 'time', 'design_space_tolerance_period', 'objective_space_tolerance_period'):
        if tk in AlgOptions:
            termination_keys.add(tk)

    algo_options = {k: v for k, v in AlgOptions.items() if k not in termination_keys}

    ObjectivesDirections = [-1 if direction =='max' else 1 for direction in ObjectivesDirections]

    n_obj = len(ObjectivesDirections)

    class MyProblem(Problem):

        def __init__(self):
            super().__init__(n_var=ToTalVariableCounter[1], n_obj=n_obj, xl=np.array([0,]*ToTalVariableCounter[1]), xu=np.array([1,]*ToTalVariableCounter[1]))
        
        def _evaluate(self, x, out, *args, **kwargs):

            f = Fitness(np.array(x))

            if n_obj == 1:
                if isinstance(f, (list, tuple)):
                    f = np.array(f).flatten()
                out["F"] = ObjectivesDirections[0] * np.atleast_2d(f).reshape(-1, 1)
            else:
                out["F"] = np.column_stack([ObjectivesDirections[i]*np.array(f[i]).flatten() for i in range(n_obj)])

    problem = MyProblem()

    pop_size = AlgOptions.get("pop_size", 100)

    needs_ref_dirs = solver_name in ("ns-ga-iii", "cta-ea", "rv-ea", "u-ns-ga-iii", "mo-ea-d")

    if needs_ref_dirs and "ref_dirs" not in AlgOptions:
        if n_obj <= 2:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=pop_size - 1)
        else:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
        AlgOptions["ref_dirs"] = ref_dirs

    derives_pop_from_ref = solver_name in ("cta-ea", "mo-ea-d", "r-ns-ga-iii")
    if derives_pop_from_ref and "pop_size" in AlgOptions:
        del AlgOptions["pop_size"]

    if solver_name == "r-ns-ga-ii" and "ref_points" not in AlgOptions:
        AlgOptions["ref_points"] = np.array([[0.0] * n_obj, [1.0] * n_obj])

    if solver_name == "r-ns-ga-iii":
        if "ref_points" not in AlgOptions:
            AlgOptions["ref_points"] = np.array([[0.0] * n_obj, [1.0] * n_obj])
        if "pop_per_ref_point" not in AlgOptions:
            AlgOptions["pop_per_ref_point"] = pop_size

    match solver_name:

        case "ns-ga-ii":

            from pymoo.algorithms.moo.nsga2 import NSGA2
            algorithm = NSGA2(**algo_options)

        case "d-ns-ga-ii":

            from pymoo.algorithms.moo.dnsga2 import DNSGA2
            algorithm = DNSGA2(**algo_options)

        case "ns-ga-iii":

            from pymoo.algorithms.moo.nsga3 import NSGA3
            algorithm = NSGA3(**algo_options)

        case "mo-ea-d":

            from pymoo.algorithms.moo.moead import MOEAD
            algorithm = MOEAD(**algo_options)

        case "cta-ea":

            from pymoo.algorithms.moo.ctaea import CTAEA
            algorithm = CTAEA(**algo_options)

        case "r-ns-ga-ii":

            from pymoo.algorithms.moo.rnsga2 import RNSGA2
            algorithm = RNSGA2(**algo_options)

        case "r-ns-ga-iii":

            from pymoo.algorithms.moo.rnsga3 import RNSGA3
            algorithm = RNSGA3(**algo_options)

        case "u-ns-ga-iii":
    
            from pymoo.algorithms.moo.unsga3 import UNSGA3
            algorithm = UNSGA3(**algo_options)

        case "sp-ea-ii":

            from pymoo.algorithms.moo.spea2 import SPEA2
            algorithm = SPEA2(**algo_options)
        
        case "sms-ea":

            from pymoo.algorithms.moo.sms import SMSEMOA
            algorithm = SMSEMOA(**algo_options)
        
        case "rv-ea":

            from pymoo.algorithms.moo.rvea import RVEA
            algorithm = RVEA(**algo_options)

        case "age-mo-ea":

            from pymoo.algorithms.moo.age import AGEMOEA
            algorithm = AGEMOEA(**algo_options)

        case "age-mo-ea-ii":

            from pymoo.algorithms.moo.age2 import AGEMOEA2
            algorithm = AGEMOEA2(**algo_options)

        case "gde3":

            from pymoo.algorithms.moo.gde3 import GDE3
            algorithm = GDE3(**algo_options)

        case "kgb":

            from pymoo.algorithms.moo.kgb import KGB
            algorithm = KGB(**algo_options)

        case "cmopso":

            from pymoo.algorithms.moo.cmopso import CMOPSO
            algorithm = CMOPSO(**algo_options)

        case "nsde":

            from pymoo.algorithms.moo.nsde import NSDE
            algorithm = NSDE(**algo_options)

        case "omni":

            from pymoo.algorithms.moo.omni import OmniOptimizer
            algorithm = OmniOptimizer(**algo_options)

        case "pinsga2":

            from pymoo.algorithms.moo.pinsga2 import PINSGA2
            algorithm = PINSGA2(**algo_options)

        case "nsder":

            needs_ref_dirs = True
            if "ref_dirs" not in AlgOptions:
                if n_obj <= 2:
                    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=pop_size - 1)
                else:
                    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
                AlgOptions["ref_dirs"] = ref_dirs
            from pymoo.algorithms.moo.nsder import NSDER
            algorithm = NSDER(**algo_options)

    from pymoo.termination.default import DefaultMultiObjectiveTermination

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=1000,
        n_max_evals=100000
    )

    epoch_val = get_epoch_value(AlgOptions)
    if AlgOptions.get('n_gen', None)!=None:
        termination = get_termination("n_gen", AlgOptions['n_gen'])
    elif epoch_val is not None:
        termination = get_termination("n_gen", epoch_val)

    if AlgOptions.get('n_eval', None)!=None:
        termination = get_termination("n_eval", AlgOptions['n_eval'])

    if AlgOptions.get('time', None)!=None:
        termination = get_termination("time", AlgOptions['time'])

    if AlgOptions.get('design_space_tolerance_period', None)!=None:

        from pymoo.termination.xtol import DesignSpaceTermination
        from pymoo.termination.robust import RobustTermination
        termination = RobustTermination(DesignSpaceTermination(**algo_options['design_space_tolerance_period'][0]), period=AlgOptions['design_space_tolerance_period'][1])

    if AlgOptions.get('objective_space_tolerance_period', None): 

        from pymoo.termination.ftol import MultiObjectiveSpaceTermination
        from pymoo.termination.robust import RobustTermination
        termination = RobustTermination(MultiObjectiveSpaceTermination(**algo_options['objective_space_tolerance_period'][0]), period=AlgOptions['objective_space_tolerance_period'][1])

    time_solve_begin = timeit.default_timer()
    if init_solutions is not None:
        from pymoo.core.initialization import Initialization
        from pymoo.core.population import Population
        from pymoo.core.sampling import Sampling

        init_arr = np.atleast_2d(np.array(init_solutions, dtype=float))

        class SeededSampling(Sampling):
            def __init__(self, seeds):
                super().__init__()
                self.seeds = np.atleast_2d(np.asarray(seeds, dtype=float))

            def do(self, problem, n_samples, **kwargs):
                pop = np.random.random((n_samples, problem.n_var))
                n_seed = min(len(self.seeds), n_samples)
                if n_seed:
                    n_col = min(self.seeds.shape[1], problem.n_var)
                    pop[:n_seed, :n_col] = self.seeds[:n_seed, :n_col]
                return Population.new("X", pop)

        algorithm.initialization = Initialization(SeededSampling(init_arr))
    res = minimize(problem, algorithm, termination=termination, verbose=verbose)
    time_solve_end = timeit.default_timer()
    pareto_front = ObjectivesDirections*res.F
    pareto_solutions = np.atleast_2d(res.X)

    return pareto_solutions, pareto_front, time_solve_begin, time_solve_end
