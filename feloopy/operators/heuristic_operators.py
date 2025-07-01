# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.



from ..helpers.empty import *
from ..helpers.error import *
import numpy as np
import math as mt
import inspect
import ast


class NumpyVariable(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __le__(self, other):       
        return self - other

    def __ge__(self, other):
        return other - self

    def __sub__(self, other):
        return NumpyVariable(super().__sub__(other))
    
def generate_heuristic_variable(features, type, name, variable_dim, variable_bound, agent, no_agents):

    if no_agents==None:
        no_agents=100

    if features['agent_status'] == 'idle':

        if features['vectorized']:

            if variable_dim == 0:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(no_agents,1)*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(no_agents,1)*(variable_bound[1]-variable_bound[0])).astype(int))
                if type == 'svar':
                    raise VariableDimError("Dimension is set to be 0 or not defined for a sequential variable.")
                
            else:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0])).astype(int) )
                if type == 'svar':          
                    return NumpyVariable(np.argsort(np.random.rand(*tuple([no_agents]+[len(dims) for dims in variable_dim])), axis=1))

        else:
            
           

            if variable_dim == 0:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand()*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand()*(variable_bound[1]-variable_bound[0])).astype(int))
                if type == 'svar':
                    raise VariableDimError("Dimension is set to be 0 or not defined for a sequential variable.")
            
            else:

                if type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + np.random.rand(*tuple([len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0]))
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + np.random.rand(*tuple([len(dims) for dims in variable_dim]))*(variable_bound[1]-variable_bound[0])).astype(int) )
                if type == 'svar':          
                    return NumpyVariable(np.argsort(np.random.rand(*tuple([len(dims) for dims in variable_dim]))))
    else:

        spread = features['variable_spread'][name]

        if features['vectorized']:
            if variable_dim == 0:
                if type == 'bvar' or type == 'ivar':
                    return NumpyVariable(np.round(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))
                
                elif type == 'pvar' or type == 'fvar':
                    return NumpyVariable(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]))
                else:
                    return NumpyVariable(np.argsort(agent[:, spread[0]:spread[1]]))
            else:

                if type == 'bvar' or type == 'ivar':

                    var = NumpyVariable(np.round(variable_bound[0] + agent[:, spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))

                    return NumpyVariable(np.reshape(var, [var.shape[0]]+[len(dims) for dims in variable_dim]))

                elif type == 'pvar' or type == 'fvar':

                    var = variable_bound[0] + agent[:, spread[0]:spread[1]
                                                    ] * (variable_bound[1] - variable_bound[0])

                    return NumpyVariable(np.reshape(var, [var.shape[0]]+[len(dims) for dims in variable_dim]))

                else:

                    return NumpyVariable(np.argsort(agent[:, spread[0]:spread[1]]))
        else:

            if variable_dim == 0:

                if type == 'bvar' or type == 'ivar':
                    
                    return (np.round(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])).astype(int))

                elif type == 'pvar' or type == 'fvar':

                    return (variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]))

                else:

                    return (np.argsort(agent[spread[0]:spread[1]]))

            else:

                if type == 'bvar' or type == 'ivar':
                    
                    return NumpyVariable(np.reshape(np.round(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0])), [len(dims) for dims in variable_dim]).astype(int))

                elif type == 'pvar' or type == 'fvar':

                    return NumpyVariable(np.reshape(variable_bound[0] + agent[spread[0]:spread[1]] * (variable_bound[1] - variable_bound[0]), [len(dims) for dims in variable_dim]))

                else:

                    return NumpyVariable(np.argsort(agent[spread[0]:spread[1]]))
                
def get_return_names(fn):
    src = inspect.getsource(fn)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    ret = stmt.value
                    if isinstance(ret, ast.Tuple):
                        return [ast.unparse(el).strip() for el in ret.elts]
                    else:
                        return [ast.unparse(ret).strip()]
    return []

def get_in_out(fn, /, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        inputs = args[0]
    elif not args and kwargs:
        inputs = kwargs
    else:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments

    result = fn(**inputs)

    if isinstance(result, dict):
        outputs = result
    else:
        names = get_return_names(fn)
        values = result if isinstance(result, tuple) else (result,)
        if len(names) != len(values):
            raise ValueError(
                f"{fn.__name__!r} returned {len(values)} value(s) "
                f"but {len(names)} name(s) were found: {names!r}"
            )
        outputs = dict(zip(names, values))

    return inputs, outputs
