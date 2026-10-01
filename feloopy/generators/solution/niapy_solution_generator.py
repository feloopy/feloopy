# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np
import timeit
from tabulate import tabulate as tb
from niapy.problems import Problem
from niapy.task import Task
from .option_utils import get_epoch_value

def generate_solution(model_object, fitness_function, total_features, objectives_directions, objective_number, number_of_times, show_plots, save_plots,show_log, solver_options, init_solutions=None, _on_repeat_start=None):

    class MyProblem(Problem):
        def __init__(self, dimension, lower=0, upper=1, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            if objectives_directions[objective_number]=='min':
                f = fitness_function(np.array(x))
            else:
                f = -1*fitness_function(np.array(x))
            return float(np.asarray(f).flat[0])

    epoch = get_epoch_value(solver_options)
    task = Task(problem=MyProblem(total_features[1]), max_iters=epoch if epoch else 100)

    if init_solutions is not None:
        import numpy as _np
        init_arr = _np.atleast_2d(_np.array(init_solutions, dtype=float))

        _base_init = getattr(model_object, "initialization_function", None)

        def _seeded_init(task, population_size, rng, **kwargs):
            if _base_init is None:
                pop = rng.uniform(task.lower, task.upper, (population_size, task.dimension))
                fpop = _np.apply_along_axis(task.eval, 1, pop)
            else:
                pop, fpop = _base_init(task=task, population_size=population_size,
                                       rng=rng, **kwargs)

            n_seed = len(init_arr)
            if n_seed > population_size:
                n_seed = population_size
            if n_seed > len(pop):
                n_seed = len(pop)
            for i in range(n_seed):
                row = _np.clip(init_arr[i], task.lower, task.upper)
                fit = float(task.eval(row))
                if hasattr(pop[i], "x"):
                    
                    pop[i].x = row
                    pop[i].f = fit
                else:
                    pop[i] = row
                fpop[i] = fit
            return pop, fpop

        model_object.initialization_function = _seeded_init

    if number_of_times == 1:

        
        
        time_solve_begin = timeit.default_timer()
        best_agent, best_reward = model_object.run(task)
        time_solve_end = timeit.default_timer()
        if objectives_directions[objective_number]!='min':
            best_reward *= -1
            
    else:

        Multiplier = {'max': 1, 'min': -1}
        directions = Multiplier[objectives_directions[objective_number]]
        time_solve_begin = []
        time_solve_end = []
        bestreward = [-directions*np.inf]
        best_reward_found = -directions*np.inf

        for i in range(number_of_times):
            if _on_repeat_start is not None:
                _on_repeat_start()
            time_solve_begin.append(timeit.default_timer())
            best_agent, best_reward = model_object.run(task)
            time_solve_end.append(timeit.default_timer())
            Result = [best_agent, best_reward]
            Result[1] = np.asarray(Result[1])
            Result[1] = Result[1].item()
            bestreward.append(best_reward)
            if directions*(Result[1]) >= directions*(best_reward_found):
                best_agent_found = Result[0]
                best_reward_found = Result[1]
        bestreward.pop(0)

        if show_log:
            print()
            hour = []
            min = []
            sec = []
            ave = []
            for i in range(number_of_times):
                tothour = round(
                    (time_solve_end[i] - time_solve_begin[i]), 3) % (24 * 3600) // 3600
                totmin = round(
                    (time_solve_end[i] - time_solve_begin[i]), 3) % (24 * 3600) % 3600 // 60
                totsec = round(
                    (time_solve_end[i] - time_solve_begin[i]), 3) % (24 * 3600) % 3600 % 60
                hour.append(tothour)
                min.append(totmin)
                sec.append(totsec)
                ave.append(round((time_solve_end[i]-time_solve_begin[i])*10**6))

            print("~~~~~~~\nTIME INFO\n~~~~~~~")
            print(tb({
                "cpt (ave)": [np.average(ave), "%02d:%02d:%02d" % (np.average(hour), np.average(min), np.average(sec))],
                "cpt (std)": [np.std(ave), "%02d:%02d:%02d" % (np.std(hour), np.std(min), np.std(sec))],
                "unit": ["micro sec", "h:m:s"]
            }, headers="keys", tablefmt="github"))
            print("~~~~~~~")

            print("~~~~~~~\nOBJ INFO\n~~~~~~~")
            print(tb({
                "obj": [np.max(bestreward), np.average(bestreward), np.std(bestreward), np.min(bestreward)],
                "unit": ["max", "average", "standard deviation", "min"]
            }, headers="keys", tablefmt="github"))
            print("~~~~~~~")

        best_agent = best_agent_found
        best_reward = best_reward_found

    return best_agent, best_reward, time_solve_begin, time_solve_end
