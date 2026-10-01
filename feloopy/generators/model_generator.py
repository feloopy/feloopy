# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from ..feloopy import model

def generate_model(features):

    match features['interface_name']:

        case 'pulp':

            from .model import pulp_model_generator
            model_object = pulp_model_generator.generate_model(features)

        case 'casadi':

            from .model import casadi_model_generator
            model_object = casadi_model_generator.generate_model(features)
            
        case 'pyomo':

            from .model import pyomo_model_generator
            model_object = pyomo_model_generator.generate_model(features)

        case 'insideopt-demo':

            from .model import seeker_model_generator
            model_object = seeker_model_generator.generate_demo_model(features)

        case 'insideopt':

            from .model import seeker_model_generator
            model_object = seeker_model_generator.generate_model(features)

        case 'gams':

            from .model import gamspy_model_generator
            model_object = gamspy_model_generator.generate_model(features)

        case 'ortools':

            from .model import ortools_model_generator
            model_object = ortools_model_generator.generate_model(features)

        case 'highs':

            from .model import highs_model_generator
            model_object = highs_model_generator.generate_model(features)

        case 'jump':

            from .model import jump_model_generator
            model_object = jump_model_generator.generate_model(features)
            
        case 'ortools_cp':

            from .model import ortools_cp_model_generator
            model_object = ortools_cp_model_generator.generate_model(features)

        case 'gekko':

            from .model import gekko_model_generator
            model_object = gekko_model_generator.generate_model(features)

        case 'mathopt':

            from .model import mathopt_model_generator
            model_object = mathopt_model_generator.generate_model(features)

        case name if 'pyoptinterface' in name:

            from .model import pyoptinterface_model_generator
            model_object = pyoptinterface_model_generator.generate_model(features)

        case 'picos':

            from .model import picos_model_generator
            model_object = picos_model_generator.generate_model(features)

        case 'cvxpy':

            from .model import cvxpy_model_generator
            model_object = cvxpy_model_generator.generate_model(features)

        case 'cylp':

            from .model import cylp_model_generator
            model_object = cylp_model_generator.generate_model(features)

        case 'pymprog':

            from .model import pymprog_model_generator
            model_object = pymprog_model_generator.generate_model(features)

        case 'cplex':

            from .model import cplex_model_generator
            model_object = cplex_model_generator.generate_model(features)

        case 'cplex_cp':

            from .model import cplex_cp_model_generator
            model_object = cplex_cp_model_generator.generate_model(features)

        case 'gurobi':

            from .model import gurobi_model_generator
            model_object = gurobi_model_generator.generate_model(features)

        case 'copt':

            from .model import copt_model_generator
            model_object = copt_model_generator.generate_model(features)

        case 'xpress':

            from .model import xpress_model_generator
            model_object = xpress_model_generator.generate_model(features)

        case 'mip':

            from .model import mip_model_generator
            model_object = mip_model_generator.generate_model(features)

        case 'linopy':

            from .model import linopy_model_generator
            model_object = linopy_model_generator.generate_model(features)

        case 'rsome_ro':

            from .model import rsome_ro_model_generator
            model_object = rsome_ro_model_generator.generate_model(features)

        case 'rsome_dro':

            from .model import rsome_dro_model_generator
            model_object = rsome_dro_model_generator.generate_model(features)

        case 'uno':

            from .model import uno_model_generator
            model_object = uno_model_generator.generate_model(features)

        case 'bonmin' | 'couenne':

            from .model import coin_model_generator
            model_object = coin_model_generator.generate_model(features)

        case 'scip':

            from .model import scip_model_generator
            model_object = scip_model_generator.generate_model(features)

        case 'hexaly':

            from .model import hexaly_model_generator
            model_object = hexaly_model_generator.generate_model(features)

        case 'mosek':

            from .model import mosek_model_generator
            model_object = mosek_model_generator.generate_model(features)

        case 'picat':

            from .model import picat_model_generator
            model_object = picat_model_generator.generate_model(features)

    return model_object


class feloop_model(model):
    def __init__(self,name=None, agent=None):
        if agent==None:
            super().__init__('exact', name, 'feloopy')
        else:
            super().__init__('heuristic', name, 'feloopy', agent=agent)

class copt_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'copt')

class cplex_cp_model(model):
    def __init__(self,name='x'):
        super().__init__('constraint', name, 'cplex_cp')

class cplex_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'cplex')

class cylp_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'cylp')

class cvxpy_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'cvxpy')

class gekko_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'gekko')

class gurobi_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'gurobi')

class gams_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'gams')

class linopy_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'linopy')

class mip_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'mip')

class ortools_cp_model(model):
    def __init__(self,name='x'):
        super().__init__(name=name, method='constraint', interface='ortools_cp')

class ortools_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'ortools')

class picos_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'picos')

class pulp_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'pulp')

class pyomo_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'pyomo')

class pymprog_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'pymprog')

class rsome_dro_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'rsome_dro')

class rsome_ro_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'rsome_ro')

class seeker_model(model):
    def __init__(self,name='x'):
        super().__init__(name=name, method='exact', interface='insideopt')

class xpress_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'xpress')

class hexaly_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'hexaly')

class mosek_model(model):
    def __init__(self,name='x'):
        super().__init__('exact', name, 'mosek')
