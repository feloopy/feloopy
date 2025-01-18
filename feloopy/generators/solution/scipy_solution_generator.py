# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def generate_solution(solver_name, AlgOptions, Fitness, ToTalVariableCounter, ObjectivesDirections, ObjectiveBeingOptimized, number_of_times, show_plots, save_plots, show_log):

    import timeit
    from scipy.optimize import minimize
    import numpy as np

    if show_log:
        verbose=True
    else:
        verbose=False
    
    ObjectivesDirections = [-1 if direction =='max' else 1 for direction in ObjectivesDirections]

    def problem(x):
        return ObjectivesDirections[0]*Fitness(np.array(x))

    time_solve_begin = timeit.default_timer()
    res = minimize(problem, [0.5 for _ in range(ToTalVariableCounter[1])] , method=solver_name, bounds=[(0,1) for _ in range(ToTalVariableCounter[1])])
    time_solve_end = timeit.default_timer()

    return res.x, ObjectivesDirections[0]*res.fun, time_solve_begin, time_solve_end

