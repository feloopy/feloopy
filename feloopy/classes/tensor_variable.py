"""
TensorVariable Module

This module defines a class, `TensorVariableClass`, that facilitates the creation of various types
of tensor variables, such as free float-valued, positive float-valued, positive integer-valued, 
binary-valued, and random float-valued tensor variables. These tensor variables are used for 
matrix/tensor-wise operations in the specified mathematical models.

Supported tensor variables:

    - ftvar
    - ptvar
    - itvar
    - btvar
    - rtvar

Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
See the file LICENSE file for licensing details.
"""

from typing import List, Union, Optional
from ..operators.fix_operators import fix_dims
from ..operators.update_operators import update_variable_features

class TensorVariable:
    """Wrapper for tensor variable dicts that supports matmul (@) and indexing."""

    __array_priority__ = 10000

    def __init__(self, data, model=None):
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_model', model)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __matmul__(self, other):
        if isinstance(other, TensorVariable):
            other = other._data
        if isinstance(other, dict):
            return sum(self._data[k] * other[k] for k in self._data if k in other)
        import numpy as np
        if isinstance(other, np.ndarray):
            other_flat = other.flatten()
            return sum(self._data[k] * other_flat[k] for k in self._data if k < len(other_flat))
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, TensorVariable):
            other = other._data
        import numpy as np
        if isinstance(other, np.ndarray):
            other_flat = other.flatten()
            return sum(other_flat[k] * self._data[k] for k in self._data if k < len(other_flat))
        if isinstance(other, (list, tuple)):
            other_flat = np.array(other).flatten()
            return sum(other_flat[k] * self._data[k] for k in self._data if k < len(other_flat))
        return NotImplemented

    def __le__(self, other):
        import numpy as np
        if isinstance(other, np.ndarray):
            other_flat = other.flatten()
            return sum(self._data[k] for k in self._data) <= float(other_flat.sum()) if len(self._data) > 0 else True
        return sum(self._data[k] for k in self._data) <= other

    def __ge__(self, other):
        import numpy as np
        if isinstance(other, np.ndarray):
            other_flat = other.flatten()
            return sum(self._data[k] for k in self._data) >= float(other_flat.sum()) if len(self._data) > 0 else True
        return sum(self._data[k] for k in self._data) >= other

    def __add__(self, other):
        if isinstance(other, TensorVariable):
            other = other._data
        if isinstance(other, dict):
            return TensorVariable({k: self._data[k] + other[k] for k in self._data if k in other}, self._model)
        return TensorVariable({k: self._data[k] + other for k in self._data}, self._model)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, TensorVariable):
            other = other._data
        if isinstance(other, dict):
            return TensorVariable({k: self._data[k] - other[k] for k in self._data if k in other}, self._model)
        return TensorVariable({k: self._data[k] - other for k in self._data}, self._model)

    def __mul__(self, other):
        if isinstance(other, TensorVariable):
            other = other._data
        if isinstance(other, dict):
            return TensorVariable({k: self._data[k] * other[k] for k in self._data if k in other}, self._model)
        return TensorVariable({k: self._data[k] * other for k in self._data}, self._model)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"TensorVariable({self._data})"

    def __float__(self):
        return float(sum(self._data.values()))

class TensorVariableClass:
    """Class that provides methods to create tensor variables."""

    def ftvar(
        self,
        name: str,
        shape: Optional[Union[int, List[Union[int, range]]]] = 0,
        bound: Optional[List[float]] = [None, None]
    ) -> TensorVariable:
        """
        Create a free float-valued tensor variable.

        Parameters
        ----------
        name : str
            The name of the tensor variable.
        shape : Optional[Union[int, List[Union[int, range]]]], optional
            The shape of the tensor variable, represented as a list containing integers or ranges. Default is 0.
        bound : Optional[List[float]], optional
            The lower and upper bounds of the tensor variable. Default is [None, None].

        Returns
        -------
        TensorVariable
            A tensor variable that accepts matrix/tensor-wise operations.
        """
        bound = list(bound) if bound is not None else [None, None]
        dim = fix_dims(shape)
        self.features = update_variable_features(name, dim, bound, 'free_variable_counter', self.features)
        
        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator
            self.features['variables'][("ftvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'ftvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim
            return TensorVariable(self.features['variables'][("ftvar", name)], self)

        raise ValueError(f"Error: TensorVariable '{name}' cannot be created.")
    
    def ptvar(
        self,
        name: str,
        shape: Optional[Union[int, List[Union[int, range]]]] = 0,
        bound: Optional[List[float]] = [0, None]
    ) -> TensorVariable:
        """
        Create a positive float-valued tensor variable.

        Parameters
        ----------
        name : str
            The name of the tensor variable.
        shape : Optional[Union[int, List[Union[int, range]]]], optional
            The shape of the tensor variable, represented as a list containing integers or ranges. Default is 0.
        bound : Optional[List[float]], optional
            The lower and upper bounds of the tensor variable. Default is [0, None].

        Returns
        -------
        TensorVariable
            A tensor variable that accepts matrix/tensor-wise operations.
        """
        bound = list(bound) if bound is not None else [0, None]
        dim = fix_dims(shape)
        self.features = update_variable_features(name, dim, bound, 'positive_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator

            self.features['variables'][("ptvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'ptvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim

            if self.features['interface_name'] in ['rsome_ro', 'rsome_dro', 'cvxpy']:
                self.con(self.features['variables'][("ptvar", name)] >= 0)

                if bound and bound[1] is not None:
                    self.con(self.features['variables'][("ptvar", name)] <= bound[1])

            return TensorVariable(self.features['variables'][("ptvar", name)], self)
    
        raise ValueError(f"Error: TensorVariable '{name}' cannot be created.")
    
    def itvar(
        self,
        name: str,
        shape: Optional[Union[int, List[Union[int, range]]]] = 0,
        bound: Optional[List[float]] = [0, None]
    ) -> TensorVariable:
        """
        Create a positive integer-valued tensor variable.

        Parameters
        ----------
        name : str
            The name of the tensor variable.
        shape : Optional[Union[int, List[Union[int, range]]]], optional
            The shape of the tensor variable, represented as a list containing integers or ranges. Default is 0.
        bound : Optional[List[float]], optional
            The lower and upper bounds of the tensor variable. Default is [0, None].

        Returns
        -------
        TensorVariable
            A tensor variable that accepts matrix/tensor-wise operations.
        """

        bound = list(bound) if bound is not None else [0, None]
        dim = fix_dims(shape)
        self.features = update_variable_features(name, dim, bound, 'integer_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator

            self.features['variables'][("itvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'itvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim
            return TensorVariable(self.features['variables'][("itvar", name)], self)

        raise ValueError(f"Error: TensorVariable '{name}' cannot be created.")
    
    def btvar(
        self,
        name: str,
        shape: Optional[Union[int, List[Union[int, range]]]] = 0,
        bound: Optional[List[float]] = [0, 1]
    ) -> TensorVariable:
        """
        Create a binary-valued tensor variable.

        Parameters
        ----------
        name : str
            The name of the tensor variable.
        shape : Optional[Union[int, List[Union[int, range]]]], optional
            The shape of the tensor variable, represented as a list containing integers or ranges. Default is 0.
        bound : Optional[List[float]], optional
            The lower and upper bounds of the tensor variable. Default is [0, 1].

        Returns
        -------
        TensorVariable
            A tensor variable that accepts matrix/tensor-wise operations.
        """

        bound = list(bound) if bound is not None else [0, 1]
        dim = fix_dims(shape)
        self.features = update_variable_features(name, dim, bound, 'binary_variable_counter', self.features)

        if self.features['solution_method'] == 'exact':
            from ..generators import variable_generator

            self.features['variables'][("btvar", name)] = variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'btvar', name, bound, dim
            )
            self.features['dimensions'][name] = dim
            return TensorVariable(self.features['variables'][("btvar", name)], self)

        raise ValueError(f"Error: TensorVariable '{name}' cannot be created.")

    def rtvar(
        self,
        name: str,
        shape: Optional[Union[int, List[Union[int, range]]]] = 0
    ) -> TensorVariable:
        """
        Create a random float-valued tensor variable.

        Parameters
        ----------
        name : str
            The name of the tensor variable.
        shape : Optional[Union[int, List[Union[int, range]]]], optional
            The shape of the tensor variable, represented as a list containing integers or ranges. Default is 0.

        Returns
        -------
        TensorVariable
            A tensor variable that accepts matrix/tensor-wise operations.
        """

        dim = fix_dims(shape)
   
        if self.features['solution_method'] == 'exact':
   
            from ..generators import variable_generator
            return variable_generator.generate_variable(
                self.features['interface_name'], self.model, 'rtvar', name, [None, None], dim
            )

        raise ValueError(f"Error: TensorVariable '{name}' cannot be created.")
