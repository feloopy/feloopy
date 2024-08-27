# Copyright (c) 2022-2024, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np
import pandas as pd
import itertools as it
import numpy as np
import pandas as pd
import polars as pl
import os
import json

try:
    from ..extras.operators.data_handler import *
except:

    class FileManager:
        pass

class DataToolkit(FileManager):

    def __init__(self, key=None, memorize=True):
        
        self.data = dict()
        self.history = dict()
        self.seed= key
        self.random = np.random.default_rng(key)
        self.criteria_directions = dict()
        self.tstart = self.vstart = self.start
        self.lfe = self.load_from_excel
        self.memorize=memorize
        self.gaussian = self.normal
        
        self.store = self.__keep

    def sets(self,*args):
        return it.product(*args)
    
    def __fix_dims(self, dim, is_range=True):
        if dim == 0:
            pass
        elif not isinstance(dim, set):
            if len(dim) >= 1:
                if is_range:
                    dim = [range(d) if not isinstance(d, range) else d for d in dim]
                else:
                    dim = [len(d) if not isinstance(d, int) else d for d in dim]

        return dim
    
    def __keep(self, name, value, neglect=False):
        if self.memorize and neglect==False:
            self.data[name]=value
            return self.data[name]
        elif neglect:
            return value
        else:
            return value
    
    ## Sets
     
    def _convert_to_set(self, input_set):
        if isinstance(input_set, set):
            return input_set
        elif isinstance(input_set, range):
            return set(input_set)
        elif isinstance(input_set, list):
            return set(input_set)
        else:
            raise TypeError("Unsupported set type")

    def _json_serializer(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": obj.shape,
                "data": obj.tolist()
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                "__type__": "dataframe",
                "index": obj.index.tolist(),
                "columns": obj.columns.tolist(),
                "data": obj.to_dict(orient='records')
            }
        elif isinstance(obj, pd.Series):
            return {
                "__type__": "series",
                "index": obj.index.tolist(),
                "data": obj.to_dict()
            }
        elif isinstance(obj, set):
            return {
                "__type__": "set",
                "data": list(obj)
            }
        elif isinstance(obj, range):
            return {
                "__type__": "range",
                "start": obj.start,
                "stop": obj.stop,
                "step": obj.step
            }
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _json_decoder(self, dct):
        # numpy arrays
        if "shape" in dct and "data" in dct:
            return np.array(dct["data"]).reshape(dct["shape"])
        # pandas DataFrames
        elif "columns" in dct and "index" in dct and "data" in dct:
            return pd.DataFrame(data=dct["data"], index=dct["index"], columns=dct["columns"])
        # pandas Series
        elif "index" in dct and "data" in dct:
            try:
                return pd.Series(data=dct["data"], index=dct["index"])
            except Exception:
                pass  
        # sets
        elif isinstance(dct, dict) and "data" in dct:
            if "__type__" in dct and dct["__type__"] == "set":
                return set(dct["data"])
        # range
        elif "start" in dct and "stop" in dct and "step" in dct:
            return range(dct["start"], dct["stop"], dct["step"])
        return dct

    def set(
        self,
        name,
        bound=None,
        step=1,
        callback=None,
        to_list=False,
        to_range=False,
        named_indices=False,
        size=None,
        init=None,
        axis=0,
        neglect=False):
        
        if size is not None:
            if to_range:
                result = range(size)
            else:
                result = set(range(size))
        elif init is not None:
            
            if type(init)==int:
                if to_range:
                    result = range(init)
                else:
                    result = set(range(init))
            elif type(init)==np.ndarray:
                if to_range:
                    result = range(np.shape(init)[axis])
                else:
                    result = set(range(np.shape(init)[axis]))

            elif type(init)==list:
                if to_range:
                    result = range(len(list))
                else:
                    result = set(range(len(list)))
            else:
                try:
                    result = set(init)
                except:
                    result = init 

            if callback:
                result =  set(item for item in result if callback(item))

        else:   
            if callback:
                named_indices = False
               
            if to_range:
                result = range(bound[0], bound[1] + 1, step)
            elif named_indices:
                result = {f"{name.lower()}{i}" for i in range(bound[0], bound[1] + 1, step) if not callback or callback(i)}
            else:
                if callback:
                    result =  set(item for item in range(bound[0], bound[1] + 1, step) if callback(item))
                else:
                    result =  set(range(bound[0], bound[1] + 1, step))
        if to_list:
            result = list(result)
        
        return self.__keep(name, result, neglect)
    
    def alias(self, name, init, neglect=False):
        result=init
        return self.__keep(name, result, neglect)

    def union(self, name, *sets, neglect=False):
        converted_sets = [self._convert_to_set(s) for s in sets]
        result=  set().union(*converted_sets)
        return self.__keep(name, result, neglect)

    def intersection(self, name, *sets, neglect=False):
        converted_sets = [self._convert_to_set(s) for s in sets]
        result = set().intersection(*converted_sets)
        return self.__keep(name, result, neglect)

    def difference(self,name, *sets, neglect=False):
        converted_sets = [self._convert_to_set(s) for s in sets]
        result = converted_sets[0]
        for s in converted_sets[1:]:
            result = result.difference(s)
        return self.__keep(name, result, neglect)

    def symmetric_difference(self,name, *sets, neglect=False):
        converted_sets = [self._convert_to_set(s) for s in sets]
        result = converted_sets[0]
        for s in converted_sets[1:]:
            result = result.symmetric_difference(s)
        return self.__keep(name, result, neglect)

    ## Parameters

    def _sample_list_or_array(self, name, init, size, replace=False, sort_result=False, return_indices=False, axis=None):
        if isinstance(init, (list, range)):
            init = np.array(init)

        if axis is None:
            sampled_indices = self.random.choice(init.size, size=size, replace=replace)
        else:
            axis = np.atleast_1d(axis)
            axis_size = init.shape[axis[0]] if len(axis) == 1 else np.prod([init.shape[i] for i in axis])
            sampled_indices = self.random.choice(axis_size, size=size, replace=replace)

        if return_indices:
            if sort_result:
                sampled_indices.sort()
            self.data[name] = sampled_indices
        else:
            sampled_data = init.flat[sampled_indices] if axis is None else np.take(init, sampled_indices, axis=axis)
            if sort_result:
                sampled_data.sort()
            self.data[name] = sampled_data

        return self.data[name]

    def _sample_pandas_dataframe(self, name, init, size, replace=False, sort_result=False, return_indices=False, axis=None):
        axis = 0 if axis is None else axis 

        if axis not in [0, 1]:
            raise ValueError("Invalid axis for Pandas DataFrame sampling. Supported axes: 0 (rows), 1 (columns)")

        sampled_indices = self.random.choice(init.shape[axis], size=size, replace=replace)

        if return_indices:
            self.data[name] = sampled_indices
        else:
            if axis == 0:
                sampled_data = init.iloc[sampled_indices, :]
            else:
                sampled_data = init.iloc[:, sampled_indices]

            if sort_result:
                sampled_data = sampled_data.sort_index(axis=axis)

            self.data[name] = sampled_data

        return self.data[name]
       
    def zeros(self, name, dim=0, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = np.zeros(1)
        else:
            if type(dim)==set:
                result = {key: 0 for key in dim}
            else:
                result = np.zeros(dim)
        return self.__keep(name, result, neglect)

    def ones(self, name, dim=0, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = np.ones(1)
        else:
            if type(dim)==set:
                result = {key: 0 for key in dim}
            else:
                result = np.ones(tuple(dim))
        return self.__keep(name, result, neglect)

    def ones_per_column(self, name, dim=0, min_ones=1, max_ones=1, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = np.ones(1)
        else:
            rows, cols = dim
            if type(rows)!=int:
                rows = len(rows)
            if type(cols)!=int:
                cols = len(cols)
            result = np.zeros((rows, cols))
            for col in range(cols):
                num_ones = self.random.integers(min_ones, max_ones + 1)
                rows_with_ones = self.random.choice(rows, num_ones, replace=False)
                result[rows_with_ones, col] = 1
        if type(dim)==set:
            result = {key: result[key] for key in dim}

        return self.__keep(name, result, neglect)
    
    def ones_per_row(self, name, dim=0, min_ones=1, max_ones=1, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = np.ones(1)
        else:
            rows, cols = dim
            if type(rows)!=int:
                rows = len(rows)
            if type(cols)!=int:
                cols = len(cols)
            result = np.zeros((rows, cols))
            for row in range(rows):
                num_ones = self.random.integers(min_ones, max_ones + 1)
                cols_with_ones = self.random.choice(cols, num_ones, replace=False)
                result[row, cols_with_ones] = 1
        if type(dim)==set:
            result = {key: result[key] for key in dim}

        return self.__keep(name, result, neglect)

    def permutation(self, name, dim=0, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if len(dim)!=2:
            raise ValueError("Only 2D matrices.")
        if dim[0]!=dim[1]:
            raise ValueError("Only box matrices (i.e., n=m)")

        identity_matrix = np.eye(dim[0])
        self.random.shuffle(identity_matrix)
        result = identity_matrix
        if type(dim)==set:
            result = {key: result[key] for key in dim}     
            
        return self.__keep(name, result, neglect)
    
    def uniformint(self, name, dim=0, bound=[1, 10], result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.integers(low=bound[0], high=bound[1] + 1)
        else:
            if type(dim)==set:
                result = {key: self.random.integers(low=bound[0], high=bound[1] + 1) for key in dim}
            else:
                result = self.random.integers(low=bound[0], high=bound[1] + 1, size=dim)
        return self.__keep(name, result, neglect)
    
    def bernoulli(self, name, dim=0, p=0.5, result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.choice([0, 1], p=[1-p, p])
        else:
            if type(dim)==set:
                result = {key: self.random.choice([0, 1], p=[1-p, p]) for key in dim}
            else:
                result = self.random.choice([0, 1], p=[1-p, p], size=dim)
        return self.__keep(name, result, neglect)
    
    def binomial(self, name, dim=0, n=None, p=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.binomial(n, p)
        else:
            if type(dim)==set:
                result = {key: self.random.binomial(n, p) for key in dim}
            else:
                result = self.random.binomial(n, p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def poisson(self, name, dim=0, lam=1, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.poisson(lam)
        else:
            if type(dim)==set:
                result = {key: self.random.poisson(lam) for key in dim}
            else:
                result = self.random.poisson(lam, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def geometric(self, name, dim=0, p=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.geometric(p)
        else:
            if type(dim)==set:
                result = {key: self.random.geometric(p) for key in dim}
            else: 
                result = self.random.geometric(p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def negative_binomial(self, name, dim=0, r=None, p=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.negative_binomial(r, p)
        else:
            if type(dim)==set:
                result = {key: self.random.negative_binomial(r, p) for key in dim}
            else:
                result = self.random.negative_binomial(r, p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def hypergeometric(self, name, dim=0, N=None, m=None, n=None, result=None, neglect=False):
        nbad = m
        ngood = N - m
        nsamples = n

        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.hypergeometric(ngood, nbad, nsamples)
        else:
            if type(dim)==set:
                result = {key: self.random.hypergeometric(ngood, nbad, nsamples) for key in dim}
            else:
                result = self.random.hypergeometric(ngood, nbad, nsamples, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def uniform(self, name, dim=0, bound=[0, 1], result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.uniform(low=bound[0], high=bound[1])
        else:
            if type(dim)==set:
                result = {key: self.random.uniform(low=bound[0], high=bound[1]) for key in dim}
            else:
                result = self.random.uniform(low=bound[0], high=bound[1], size=dim)
        return self.__keep(name, result, neglect)

    def weight(self, name, dim=0, bound=[0, 1], result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            data = self.random.uniform(low=bound[0], high=bound[1])
            result = data / data.sum() if data.sum() != 0 else data
        else:
            if type(dim)==set:
                dim = len(dim)
            data = self.random.uniform(low=bound[0], high=bound[1], size=dim)
            result = data / data.sum(axis=-1, keepdims=True) if data.sum() != 0 else data
        return self.__keep(name, result, neglect)

    def normal(self, name, dim=0, mu=0, sigma=1, result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.normal(mu, sigma)
        else:
            if type(dim)==set:
                result = {key: self.random.normal(mu, sigma) for key in dim}
            else:
                result = self.random.normal(mu, sigma, size=dim)
        return self.__keep(name, result, neglect)

    def standard_normal(self, name, dim=0, result=None, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.normal(0, 1)
        else:
            if type(dim)==set:
                result = {key: self.random.normal(0, 1) for key in dim}
            else:
                result = self.random.normal(0, 1, size=dim)
        return self.__keep(name, result, neglect)

    def exponential(self, name, dim=0, lam=1.0, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.exponential(scale=1/lam)
        else:
            if type(dim)==set:
                result = {key: self.random.exponential(scale=1/lam) for key in dim}
            else:
                result = self.random.exponential(scale=1/lam, size=dim)
        return self.__keep(name, result, neglect)

    def gamma(self, name, dim=0, alpha=1, lam=1, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.gamma(shape=alpha, scale=1/lam)
        else:
            if type(dim)==set:
                result = {key: self.random.gamma(shape=alpha, scale=1/lam) for key in dim}
            else:
                result = self.random.gamma(shape=alpha, scale=1/lam, size=dim)
        return self.__keep(name, result, neglect)

    def erlang(self, name, dim=0, alpha=1, lam=1, result=None, neglect=False):
        alpha = int(alpha)
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.gamma(shape=alpha, scale=1/lam)
        else:
            if type(dim)==set:
                result = {key: self.random.gamma(shape=alpha, scale=1/lam) for key in dim}
            else:
                result = self.random.gamma(shape=alpha, scale=1/lam, size=dim)
        return self.__keep(name, result, neglect)

    def beta(self, name, dim=0, a=1, b=1, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.beta(a, b, size=None)
        else:
            if type(dim)==set:
                result = {key: self.random.beta(a, b)  for key in dim}
            else:
                result = self.random.beta(a, b, size=dim)
        return self.__keep(name, result, neglect)

    def weibull(self, name, dim=0, alpha=None, beta=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = alpha * self.random.weibull(a=beta)
        else:
            if type(dim)==set:
                result = {key: alpha * self.random.weibull(a=beta) for key in dim}
            else: 
                result = alpha * self.random.weibull(a=beta, size=dim)
        return self.__keep(name, result, neglect)

    def cauchy(self, name, dim=0, alpha=None, beta=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.standard_cauchy()
        else:
            if type(dim)==set:
                result = {key: self.random.standard_cauchy() for key in dim}
            else:
                result = self.random.standard_cauchy(size=dim)
        return self.__keep(name, result, neglect)

    def dirichlet(self, name, dim=0, k=None, alpha=None, result=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if alpha is None:
            if k is not None:
                alpha = np.ones(k)
            elif isinstance(dim, list):
                alpha = np.ones(len(dim[-1]))
        if dim == 0 or len(dim) == 1:
            result = self.random.dirichlet(alpha)
        else:
            if type(dim)==set:
                result = {key: self.random.dirichlet(alpha) for key in dim}
            else:
                result = self.random.dirichlet(alpha, size=dim)
        return self.__keep(name, result, neglect)

    def colors(self, name, dim=0, neglect=False, with_names=True):
        import matplotlib.colors as mcolors
        self.colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        dim = self.__fix_dims(dim,is_range=False)
        dim = [range(i) for i in dim]
        if dim == 0:    
            if with_names:
                result = self.random.choice(list(self.colors.keys()))
            else:
                result = '#{:06x}'.format(self.random.integers(0, 0xFFFFFF))
        else:
            if len(dim) == 1:
                if with_names:
                    result = {key: self.random.choice(list(self.colors.keys())) for key in dim[0]}
                else:
                    result = {key: '#{:06x}'.format(self.random.integers(0, 0xFFFFFF)) for key in dim[0]}
            else:
                if with_names:
                    result = {key: self.random.choice(list(self.colors.keys())) for key in it.product(*dim)}
                else:
                    result = {key: '#{:06x}'.format(self.random.integers(0, 0xFFFFFF)) for key in it.product(*dim)}
        return self.__keep(name, result, neglect)
    
    def start(self, name, value, direction=None):
        self.criteria_directions[name] = direction
        self.data[name] = [value]

    def update(self, names: list, criteria: list, values: list, compared_with: list):
        xcounter = 0
        for name in names:
            counter = 0
            for criterion in criteria:
                if self.criteria_directions[criterion] == "max":
                    if compared_with[counter] >= self.data[criterion][-1]:
                        self.data[name] = values[xcounter]
                        self.data[criterion].append(compared_with[counter])
                elif self.criteria_directions[criterion] == "min":
                    if compared_with[counter] <= self.data[criterion][-1]:
                        self.data[name] = values[xcounter]
                        self.data[criterion].append(compared_with[counter])
                else:
                    self.data[name] = values[xcounter]
                    self.data[criterion].append(compared_with[counter])
                counter += 1
            xcounter += 1

    def sample(self, name, init, size, replace=False, sort_result=False, reset_index=False, return_indices=False, axis=None, neglect=False):

        type_is= type(init)
        if type_is ==set:
            init = list(init)
            
        if isinstance(init, (list, set,range, np.ndarray)):
            sample =  self._sample_list_or_array(name, init, size, replace, sort_result, return_indices, axis)
        elif isinstance(init, pd.DataFrame):
            sample = self._sample_pandas_dataframe(name, init, size, replace, sort_result, return_indices, axis)
        else:
            raise ValueError("Unsupported data type for sampling. Supported types: set, list, range, numpy.ndarray, pandas.DataFrame")

        if reset_index:
            sample = sample.reset_index(drop=True)

        self.__keep(name, sample, neglect)
        
        if type_is ==set:
            return set(sample)
        elif type_is in [list, range]:
            return list(sample)
        elif return_indices:
            return set(sample)
        else:
            return sample

    def set_local_parameters(self):

        for key, value in self.data.items():
            locals()[key] = value      

    def set_global_parameters(self):

        for key, value in self.data.items():
            globals()[key] = value

    def load_from_excel(
        self, name: str, dim: list, labels: list, appearance: list, file_name: str, neglect=False
    ):
        
        dim = self.__fix_dims(dim, is_range=True)

        if len(appearance) == 2:
            if (
                (appearance[0] == 1 and appearance[1] == 1)
                or (appearance[0] == 1 and appearance[1] == 0)
                or (appearance[0] == 0 and appearance[1] == 0)
                or (appearance[0] == 0 and appearance[1] == 1)
            ):
                result = pd.read_excel(
                    file_name, index_col=0, sheet_name=name
                ).to_numpy()
            else:
                parameter = pd.read_excel(
                    file_name,
                    header=[i for i in range(appearance[1])],
                    index_col=[i for i in range(appearance[0])],
                    sheet_name=name,
                )
                created_par = np.zeros(shape=([len(i) for i in dim]))
                for keys in it.product(*dim):
                    try:
                        created_par[keys] = parameter.loc[
                            tuple(
                                [labels[i] + str(keys[i]) for i in range(appearance[0])]
                            ),
                            tuple(
                                [
                                    labels[i] + str(keys[i])
                                    for i in range(appearance[0], len(labels))
                                ]
                            ),
                        ]
                    except:
                        created_par[keys] = None
                result = created_par
        else:
            par = pd.read_excel(file_name, index_col=0, sheet_name=name).to_numpy()
            result = par.reshape(
                par.shape[0],
            )
        
        if dim==0:
            result=result[0][0]
        
        elif len(dim)==1:
            result=np.reshape(result,[len(dim[0]),])
        
        else:
            pass

        return self.__keep(name, result, neglect)

    def save(self, name, format="json"):
        directory = 'results/data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, name+"."+format)
        
        if format == "json":
            try:
                with open(file_path, 'w') as file:
                    json.dump(self.data, file, default=self._json_serializer, indent=4)
                print(f"Data successfully exported to {file_path}")
            except TypeError as e:
                print(f"Serialization error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    def load(self,name, format="json", neglect=False):
        name = name+"."+format
        final_dir = './data/final'
        results_dir = 'results/datasets'
        file_path_final = os.path.join(final_dir, name)
        file_path_results = os.path.join(results_dir, name)
        
        if os.path.exists(file_path_final):
            file_path = file_path_final
        elif os.path.exists(file_path_results):
            file_path = file_path_results
            print(f"Data file found in {results_dir}. For consistency, please move it to {final_dir}.")
        else:
            raise FileNotFoundError(f"File not found. Please place the data file in {final_dir} or {results_dir}.")
        
        if "json" in name:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file, object_hook=self._json_decoder)
                print(f"Data successfully imported from {file_path}")
                data = data
            except Exception as e:
                print(f"An error occurred: {e}")
                data= None

        if name.endswith(".json"):
            name = name[:-5]
    
        return self.__keep(name, data, neglect)


data_toolkit = data_utils = data_manager = DataToolkit
