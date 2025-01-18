# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def generate_model(total_variables, directions, solver_name, solver_options, lb, ub):
    match solver_name:
        case 'hco':
            try:
                from ...extras.algorithms.heuristic.HCO import HCO
            except ImportError:
                from ...algorithms.heuristic.HCO import HCO
            model_object = HCO(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get('pop_size', 50), e=solver_options.get('elitism_number', 3), ac=solver_options.get('archive_cap', 100), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'gwo':
            try:
                from ...extras.algorithms.heuristic.GWO import GWO
            except ImportError:
                from ...algorithms.heuristic.GWO import GWO
            model_object = GWO(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get(
                'pop_size', 50), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'ga':
            try:
                from ...extras.algorithms.heuristic.GA import GA
            except ImportError:
                from ...algorithms.heuristic.GA import GA
            model_object = GA(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get('pop_size', 50), sc=solver_options.get('selection', 1), mu=solver_options.get('mutation_rate', 0.02), cr=solver_options.get('crossover_rate', 0.7), sfl=solver_options.get('survival_lb', 0.4), sfu=solver_options.get('survival_ub', 0.6), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'dgwo':
            try:
                from ...extras.algorithms.heuristic.DGWO import GWO
            except ImportError:
                from ...algorithms.heuristic.DGWO import GWO
            model_object = GWO(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get(
                'pop_size', 50), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'dga':
            try:
                from ...extras.algorithms.heuristic.DGA import GA
            except ImportError:
                from ...algorithms.heuristic.DGA import GA
            model_object = GA(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get('pop_size', 50), sc=solver_options.get('selection', 1), mu=solver_options.get('mutation_rate', 0.02), cr=solver_options.get('crossover_rate', 0.7), sfl=solver_options.get('survival_lb', 0.4), sfu=solver_options.get('survival_ub', 0.6), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))


        case 'de':
            try:
                from ...extras.algorithms.heuristic.DE import DE
            except ImportError:
                from ...algorithms.heuristic.DE import DE
            model_object = DE(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get('pop_size', 50), mu=solver_options.get('mutation_rate', 0.02),
                              cr=solver_options.get('crossover_rate', 0.7), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'sa':
            try:
                from ...extras.algorithms.heuristic.SA import SA
            except ImportError:
                from ...algorithms.heuristic.SA import SA
            model_object = SA(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=1, cc=solver_options.get('cooling_cycles', 10), mt=solver_options.get(
                'maximum_temperature', 1000),  ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'bo':
            try:
                from ...extras.algorithms.heuristic.BO import BO
            except ImportError:
                from ...algorithms.heuristic.BO import BO
            model_object = BO(f=total_variables, d=directions, s=solver_options.get(
                'epoch', 100), t=10, rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))
            
        case 'ts':
            try:
                from ...extras.algorithms.heuristic.TS import TS
            except ImportError:
                from ...algorithms.heuristic.TS import TS
            model_object = TS(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=1, c=solver_options.get(
                'tabu_list_size', 10), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'pso':
            try:
                from ...extras.algorithms.heuristic.PSO import PSO
            except ImportError:
                from ...algorithms.heuristic.PSO import PSO
            model_object = PSO(f=total_variables, d=directions, s=solver_options.get('epoch', 100), t=solver_options.get('pop_size', 50), w=solver_options.get('velocity_weight', 0.8), c1=solver_options.get(
                'p_best_weight', 0.1), c2=solver_options.get('g_best_weight', 0.1), ac=solver_options.get('archive_cap', 50), rep=solver_options.get('episode', 1), ben=solver_options.get('benchmark', False))

        case 'adam':
            try:
                from ...extras.algorithms.heuristic.gradient.adam import ADAM
            except ImportError:
                from ...algorithms.heuristic.gradient.adam import ADAM
            model_object = ADAM(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), b1=solver_options.get(
                'beta_1', 0.9), b2=solver_options.get('beta_2', 0.999), ben=solver_options.get('benchmark', False))

        case 'nadam':
            try:
                from ...extras.algorithms.heuristic.gradient.nadam import NADAM
            except ImportError:
                from ...algorithms.heuristic.gradient.nadam import NADAM
            model_object = NADAM(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), b1=solver_options.get(
                'beta_1', 0.9), b2=solver_options.get('beta_2', 0.999), ben=solver_options.get('benchmark', False))

        case 'adamax':
            try:
                from ...extras.algorithms.heuristic.gradient.adamax import ADAMAX
            except ImportError:
                from ...algorithms.heuristic.gradient.adamax import ADAMAX
            model_object = ADAMAX(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), b1=solver_options.get(
                'beta_1', 0.9), b2=solver_options.get('beta_2', 0.999), ben=solver_options.get('benchmark', False))

        case 'adadelta':
            try:
                from ...extras.algorithms.heuristic.gradient.adadelta import ADADELTA
            except ImportError:
                from ...algorithms.heuristic.gradient.adadelta import ADADELTA
            model_object = ADADELTA(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), rh=solver_options.get(
                'rho', 0.95), ben=solver_options.get('benchmark', False))
            
        case 'adamw':
            try:
                from ...extras.algorithms.heuristic.gradient.adam import ADAM
            except ImportError:
                from ...algorithms.heuristic.gradient.adam import ADAM
            model_object = ADAM(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), b1=solver_options.get(
                'beta_1', 0.9), b2=solver_options.get('beta_2', 0.999), wd=solver_options.get('weight_decay', 0.01), ben=solver_options.get('benchmark', False))

        case 'rmsprop':
            try:
                from ...extras.algorithms.heuristic.gradient.rmsprop import RMSPROP
            except ImportError:
                from ...algorithms.heuristic.gradient.rmsprop import RMSPROP
            model_object = RMSPROP(f=total_variables, 
                                   d=directions, 
                                   lb=lb, 
                                   ub=ub, 
                                   s=solver_options.get('epoch', 100), 
                                   t=1, 
                                   rep=solver_options.get('episode', 1), 
                                   lr=solver_options.get('learning_rate', 0.01), 
                                   m=solver_options.get('momentum', 0.9), 
                                   dr=solver_options.get('decay_rate', 0.9), 
                                   ben=solver_options.get('benchmark', False))

        case 'adagrad':
            try:
                from ...extras.algorithms.heuristic.gradient.adagrad import ADAGRAD
            except ImportError:
                from ...algorithms.heuristic.gradient.adagrad import ADAGRAD
            model_object = ADAGRAD(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), ben=solver_options.get('benchmark', False))

        case 'sgd':
            try:
                from ...extras.algorithms.heuristic.gradient.sgd import SGD
            except ImportError:
                from ...algorithms.heuristic.gradient.sgd import SGD
            model_object = SGD(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), m=solver_options.get(
                'momentum', 0.9), ben=solver_options.get('benchmark', False))

        case 'nesterov':
            try:
                from ...extras.algorithms.heuristic.gradient.sgd import SGD
            except ImportError:
                from ...algorithms.heuristic.gradient.sgd import SGD
            model_object = SGD(f=total_variables, d=directions, lb=lb, ub=ub, s=solver_options.get(
                'epoch', 100), t=1, rep=solver_options.get('episode', 1), lr=solver_options.get('learning_rate', 0.01), m=solver_options.get(
                'momentum', 0.9), ben=solver_options.get('benchmark', False))


    return model_object