# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import numpy as np
from ..helpers._lazy import pl
import itertools as it
import os
import json

from .arraybase import _ArrayBase, _nd_getitem

try:
    from ..extras.operators.data_handler import *
except:

    class FileManager:
        pass


# Parsed workbook grids, keyed by (path, mtime): one openpyxl parse per
# file revision instead of one per sheet.
_EXCEL_GRIDS = {}


def _sheet_grid(ws):
    """Values-only grid for ``ws`` as a rectangular object ndarray.

    Empty cells become ``None``.  Trailing blank rows/columns (left-over
    formatting) are dropped; blank *leading* rows/columns are kept, because
    they are positional information for the read.
    """
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    while rows and all(v is None for v in rows[-1]):
        rows.pop()
    width = max(
        (max((j for j, v in enumerate(r) if v is not None), default=-1) for r in rows),
        default=-1,
    )
    if not rows or width < 0:
        return np.empty((0, 0), dtype=object)
    grid = np.empty((len(rows), width + 1), dtype=object)
    for i, r in enumerate(rows):
        for j in range(width + 1):
            grid[i, j] = r[j] if j < len(r) else None
    return grid


def _invalidate_grids(file_name):
    """Drop every cached parse of ``file_name`` (call after writing to it)."""
    try:
        path = os.path.abspath(file_name)
    except (OSError, ValueError):
        return
    for key in [k for k in _EXCEL_GRIDS if k[0] == path]:
        del _EXCEL_GRIDS[key]


def _infer_sheet_layout(name, grid):
    """Derive ``(dim, labels, appearance)`` for a bare ``lfe(name=...)`` read.

    The sheet is classified the way ``save_to_excel`` lays sheets out and the
    way the previous pandas loader picked them apart:

    * one populated cell (the rest blank) -> a scalar, i.e. ``dim=0``;
    * leading rows without a number are header rows (``nc``);
    * leading columns below them without a number are label columns (``nr``);
    * label columns present -> ``appearance=[nr, nc]``: one dimension per
      label column, then one per header row that labels the value region,
      each sized by its distinct labels;
    * no label columns but a single row of values -> wide layout: the header
      rows name the value columns, i.e. ``appearance=[0, nc]`` (this is how
      ``save_to_excel`` writes 1-D arrays by default);
    * no label columns and many rows -> long format: the first ``ncols - 1``
      columns are indices and the last one holds the values, i.e.
      ``appearance=[k, 0]`` -- the header row then reads as data and is
      skipped by the loader's exact label matching.

    ``ValueError`` is raised when the layout cannot be recognised; pass
    ``dim`` explicitly in that case.
    """

    def _is_num(v):
        return (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and not (isinstance(v, float) and np.isnan(v))
        )

    def _present(v):
        return (
            v is not None
            and v != ""
            and not (isinstance(v, float) and np.isnan(v))
        )

    def _prefix(v):
        s = str(v)
        core = s.rstrip("0123456789")
        return core if core else s

    def _uniq(values):
        out = []
        for v in values:
            if _present(v) and v not in out:
                out.append(v)
        return out

    nrows, ncols = grid.shape
    if sum(1 for v in grid.ravel() if _present(v)) <= 1:
        # a single populated cell: read it the way dim=0 always has
        return 0, None, None

    # header rows: leading rows holding no numeric cell
    nc = 0
    while nc < nrows and not any(_is_num(v) for v in grid[nc]):
        nc += 1
    # label columns: leading columns below the headers that carry labels
    # but no numbers
    nr = 0
    while nr < ncols:
        col = grid[nc:, nr]
        if any(_is_num(v) for v in col) or not any(_present(v) for v in col):
            break
        nr += 1

    if nr == 0:
        data_rows = nrows - nc
        if data_rows <= 1:
            # wide: one row of values, the header rows name the columns
            col_dim_rows = [
                i for i in range(nc) if any(_present(v) for v in grid[i])
            ]
            if not col_dim_rows:
                raise ValueError(
                    f"Sheet '{name}': cannot infer the layout (no label "
                    "columns and no header row); pass dim, labels and "
                    "appearance explicitly"
                )
            labels, sizes = [], []
            for i_h in col_dim_rows:
                row = grid[i_h]
                labels.append(_prefix(next(v for v in row if _present(v))))
                sizes.append(len(_uniq(row)))
            return sizes, labels, [0, nc]

        # long format: the last column holds the values, the ones before it
        # are indices
        k = ncols - 1
        if k < 1:
            raise ValueError(
                f"Sheet '{name}': cannot infer the layout (a single column "
                "of values with no index); pass dim explicitly"
            )
        labels = []
        for j in range(k):
            first = next(
                (grid[i, j] for i in range(nc) if _present(grid[i, j])), None
            )
            labels.append(_prefix(first) if first is not None else "")
        if not any(labels) and "_" in name:
            # no header names to go by: the sheet name suffix does
            # (ase_ijm -> i, j, m), like the old pandas loader assumed
            suffix = name.split("_", 1)[1]
            if suffix.isalpha() and len(suffix) == k:
                labels = list(suffix)
        sizes = [len(_uniq(grid[nc:, j])) for j in range(k)]
        if 0 in sizes:
            raise ValueError(
                f"Sheet '{name}': cannot infer the layout (index column "
                "without values); pass dim, labels and appearance explicitly"
            )
        return sizes, labels, [k, 0]

    # label columns (and maybe header rows) label the value region
    labels, sizes = [], []
    for j in range(nr):
        first = next((v for v in grid[nc:, j] if _present(v)), None)
        labels.append(_prefix(first) if first is not None else "")
        sizes.append(len(_uniq(grid[nc:, j])))
    value_region = grid[:nc, nr:]
    col_dim_rows = [
        i for i in range(nc) if any(_present(v) for v in value_region[i])
    ]
    for i_h in col_dim_rows:
        row = value_region[i_h]
        labels.append(_prefix(next(v for v in row if _present(v))))
        sizes.append(len(_uniq(row)))
    return sizes, labels, [nr, nc]


class NativeArray(_ArrayBase):
    """A stored data parameter.

    A view-based ndarray skin over the shared ``_ArrayBase`` (the same base
    ``NumpyVariable`` uses).  Element access still hands back Python
    scalars -- ``p[0]`` is an ``int``, not an ``np.int64``, so it works as
    an index, a json value and inside ``isinstance`` probes -- while every
    other manipulation (arithmetic, reductions such as
    ``.sum(axis=(1, 2))``, hashing, ``in`` membership, ``int()``/``round()``
    coercion and comparisons) comes from the base, identically for data and
    decision variables.  ``_arr`` survives as the legacy accessor for the
    call sites that unwrap data before handing it to numpy.
    """

    # data compared with data keeps numpy's boolean answers; see _ArrayBase
    _is_data = True

    @property
    def _arr(self):
        # plain ndarray view over the same buffer: stats, reports and svar
        # initialization unwrap with it, and recursion like
        # ``flatten(data._arr)`` must reach the ndarray branch, not loop
        return self.view(np.ndarray)

    def __getitem__(self, key):
        v = _nd_getitem(self, key)
        if isinstance(v, np.generic):
            # Python scalars: np.int64 has no .index, defeats integer
            # indexing and is not json serializable
            return v.item()
        return v

    def __array__(self, dtype=None, copy=None):
        # numpy 2 calls __array__(dtype, copy=False) from np.asarray(); a
        # view needs no copy, and the dtype branch below already produces a
        # new array when conversion is requested
        base = self.view(np.ndarray)
        if dtype is not None and np.dtype(dtype) != base.dtype:
            return base.astype(dtype)
        if copy:
            return base.copy()
        return base

    def __repr__(self):
        # report boxes render data with plain numpy formatting
        return repr(self._arr)

    def __str__(self):
        return str(self._arr)

class DataRef:
    __slots__ = ('_data', '_key')
    def __init__(self, data, key):
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_key', key)
    def _resolve(self):
        return self._data[self._key]
    def __float__(self):
        return float(self._resolve())
    def __int__(self):
        v = self._resolve()
        if isinstance(v, _ArrayBase):
            return v.__int__()
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v)
        return v
    def __index__(self):
        v = self._resolve()
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v)
        if isinstance(v, _ArrayBase):
            # a size-1 integer array indexes like its element; anything else
            # raises numpy's own TypeError
            return v.__index__()
        raise TypeError(f"cannot use DataRef wrapping {type(v).__name__} as index")
    def __add__(self, other):
        return self._resolve() + other
    def __radd__(self, other):
        return other + self._resolve()
    def __sub__(self, other):
        return self._resolve() - other
    def __rsub__(self, other):
        return other - self._resolve()
    def __mul__(self, other):
        return self._resolve() * other
    def __rmul__(self, other):
        return other * self._resolve()
    def __truediv__(self, other):
        return self._resolve() / other
    def __rtruediv__(self, other):
        return other / self._resolve()
    def __floordiv__(self, other):
        return self._resolve() // other
    def __rfloordiv__(self, other):
        return other // self._resolve()
    def __mod__(self, other):
        return self._resolve() % other
    def __rmod__(self, other):
        return other % self._resolve()
    def __pow__(self, other):
        return self._resolve() ** other
    def __rpow__(self, other):
        return other ** self._resolve()
    def __neg__(self):
        return -self._resolve()
    def __pos__(self):
        return +self._resolve()
    def __abs__(self):
        return abs(self._resolve())
    def __round__(self, n=None):
        return round(self._resolve(), n) if n is not None else round(self._resolve())
    def __floor__(self):
        import math
        return math.floor(self._resolve())
    def __ceil__(self):
        import math
        return math.ceil(self._resolve())
    def __trunc__(self):
        import math
        return math.trunc(self._resolve())
    def __eq__(self, other):
        return self._resolve() == other
    def __ne__(self, other):
        return self._resolve() != other
    def __lt__(self, other):
        return self._resolve() < other
    def __le__(self, other):
        return self._resolve() <= other
    def __gt__(self, other):
        return self._resolve() > other
    def __ge__(self, other):
        return self._resolve() >= other
    def __hash__(self):
        return hash(self._resolve())
    def __repr__(self):
        return repr(self._resolve())
    def __str__(self):
        return str(self._resolve())
    def __bool__(self):
        return bool(self._resolve())
    def __len__(self):
        return len(self._resolve())
    def __contains__(self, item):
        return item in self._resolve()
    def __iter__(self):
        return iter(self._resolve())
    def __getitem__(self, key):
        return self._resolve()[key]
    def __setitem__(self, key, value):
        self._resolve()[key] = value
    def __array__(self, dtype=None, copy=None):
        # numpy 2 passes copy= to __array__; resolving already hands back a
        # fresh view/convert of the underlying data, so only an explicit
        # copy request needs an actual copy
        v = self._resolve()
        if copy:
            return np.array(v, dtype=dtype)
        return np.asarray(v, dtype=dtype)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        resolved = tuple(x._resolve() if isinstance(x, DataRef) else x for x in inputs)
        if 'out' in kwargs:
            outs = kwargs['out']
            kwargs['out'] = tuple(x._resolve() if isinstance(x, DataRef) else x for x in outs)
        return getattr(ufunc, method)(*resolved, **kwargs)
    def __array_function__(self, func, types, args, kwargs):
        resolved_args = tuple(x._resolve() if isinstance(x, DataRef) else x for x in args)
        return func(*resolved_args, **kwargs)


def rewrap_scenario(original, value):
    """Return *value* wrapped in the same container type as *original*.

    DataToolkit stores numeric arrays as NativeArray (whose element access
    converts numpy scalars to Python ints/floats) and sometimes as plain
    lists.  Scenario generation produces raw numpy arrays, so swapping one
    in directly would change how the model's data references behave
    (``np.int64`` has no ``.index`` and breaks highspy expressions, float
    scenario values break integer indexing, ...).  Mirroring the original
    container type keeps rebuilt models consistent with the first build.
    """
    if isinstance(original, NativeArray):
        if isinstance(value, NativeArray):
            return value
        if isinstance(value, (list, tuple, np.ndarray)):
            return NativeArray(np.asarray(value))
        return value
    if isinstance(original, list):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value
    if isinstance(original, tuple):
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value
    return value


class DataToolkit(FileManager):

    def __init__(self, key=None, memorize=True, measure=True):
        
        self.data = dict()
        self.seed= key
        self.random = np.random.default_rng(key)
        self.lfe = self.load_from_excel
        self.lse = self.save_to_excel
        self.memorize=memorize
        self.gaussian = self.normal
        self.store = self.param =self.par = self.__keep
        self.measure = measure
        self.size = 0
        self.max_among_all_params = float('-inf')
        self.min_among_all_params = float('+inf')
        self.minimum_params = {}
        self.maximum_params = {}
        self.average_params = {}
        self.size_params = {}
        self.type_params = {}
        self.possible_epsilon = 1e-16
        self.possible_big_m = 1e16
        self.std_params = {}
        self.descriptions = {}

    def sets(self, *args, name=None):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, set):
                result = arg
            elif isinstance(arg, (list, range)):
                result = set(arg)
            elif isinstance(arg, str):
                result = {arg}
            else:
                result = {arg}
        elif len(args) == 0:
            result = set()
        else:
            result = set(it.product(*args))
        if name is not None:
            self.data[name] = result
        return result
    
    def __fix_dims(self, dim, is_range=True):
        if dim == 0:
            pass
        elif isinstance(dim, set):
            pass
        elif isinstance(dim, (list, tuple)):
            if len(dim) >= 1:
                if isinstance(dim[0], str):
                    dim = set(dim)
                elif is_range:
                    dim = [range(d) if not isinstance(d, range) else d for d in dim]
                else:
                    dim = [len(d) if not isinstance(d, int) else d for d in dim]
        return dim

    def __calculate_total_size(self, data):
        if isinstance(data, DataRef):
            return self.__calculate_total_size(data._resolve())
        elif isinstance(data, NativeArray):
            return data.size
        elif isinstance(data, (list, tuple)):
            return sum(self.__calculate_total_size(item) for item in data)
        elif isinstance(data,set):
            return 0
        elif isinstance(data, dict):
            return sum(self.__calculate_total_size(key) + self.__calculate_total_size(value) for key, value in data.items())
        elif isinstance(data, np.ndarray):
            return data.size
        elif isinstance(data, pl.DataFrame):
            return data.shape[0] * data.shape[1]
        elif isinstance(data, pl.Series):
            return data.shape[0]
        elif hasattr(data, '__len__') and not isinstance(data, (str, bytes)):
            return sum(self.__calculate_total_size(item) for item in data)
        else:
            return 1

    def __calculate_stats(self, data):
        def flatten(data):
            if isinstance(data, DataRef):
                yield from flatten(data._resolve())
            elif isinstance(data, NativeArray):
                yield from flatten(data._arr)
            elif isinstance(data, (list, tuple)):
                for item in data:
                    yield from flatten(item)
            elif isinstance(data, set):
                for item in data:
                    yield from flatten(item)
            elif isinstance(data, dict):
                for key, value in data.items():
                    yield from flatten(key)
                    yield from flatten(value)
            elif isinstance(data, np.ndarray):
                for item in data.flatten():
                    yield from flatten(item)
            elif isinstance(data, pl.Series):
                for item in data.to_numpy().flatten():
                    yield from flatten(item)
            elif isinstance(data, pl.DataFrame):
                for item in data.to_numpy().flatten():
                    yield from flatten(item)
            elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                for item in data:
                    yield from flatten(item)
            elif isinstance(data, (int, float, str, bytes, complex, bool)):
                yield data
            elif isinstance(data, np.generic):
                yield data.item()
            else:
                yield data

        values = list(flatten(data))

        if not values:
            return float('inf'), float('-inf'), float('nan'), float('nan')

        if not all(isinstance(x, (int, float)) or isinstance(x, type(values[0])) for x in values):
            return '-', len(values), None, None, None, None
        

        type_checks = {
            'ℝ': all(isinstance(x, (int, float)) for x in values) and any(x > 0 for x in values) and any(x < 0 for x in values), 
            'ℝ⁺': all(isinstance(x, (int, float)) and x >= 0 for x in values) and any(x > 0 for x in values) and any(x > 1 for x in values),
            'ℝ⁻': all(isinstance(x, (int, float)) and x <= 0 for x in values) and any(x < 0 for x in values) and any(x < -1 for x in values), 
            'ℤ': all(isinstance(x, int) for x in values) and any(x > 0 for x in values) and any(x < 0 for x in values),
            'ℤ⁺': all(isinstance(x, int) and x >= 0 for x in values) and any(x > 0 for x in values), 
            'ℕ': all(isinstance(x, int) and x > 0 for x in values),
            'ℤ₀': all(isinstance(x, int) and x == 0 for x in values), 
            'ℤ⁻': all(isinstance(x, int) and x < 0 for x in values),  
            '𝔹': all(isinstance(x, int) and x in [0, 1] for x in values) and any(x == 1 for x in values),  
            'ℝ⁺ ∩ [0, 1]': all(isinstance(x, (int,float)) and 0 <= x <= 1 for x in values) and any(0 < x < 1 for x in values), 
            'ℝ⁻ ∩ [-1, 0]': all(isinstance(x, (int,float)) and -1 <= x <= 0 for x in values) and any(-1 < x < 0 for x in values),  
            'Σ': all(isinstance(x, str) for x in values), 
            'B': all(isinstance(x, bytes) for x in values), 
            'ℂ': all(isinstance(x, complex) for x in values), 
            '∅': all(x is None for x in values), 
            '-': not all(isinstance(x, (int, float,str,bytes,complex)) for x in values), 
        }

        identified_types = [name for name, check in type_checks.items() if check]
        data_type = '/'.join([t.strip() for t in identified_types]) if identified_types else 'Other'
        data_type+="    "

        if 'ℝ' in data_type or 'ℤ' in data_type: 
            numeric_values = [x for x in values if isinstance(x, (int, float))]
            if numeric_values:
                min_value = min(numeric_values)
                max_value = max(numeric_values)
                mean_value = sum(numeric_values) / len(numeric_values)
                std_deviation = (sum((x - mean_value) ** 2 for x in numeric_values) / len(numeric_values)) ** 0.5
            else:
                min_value = max_value = mean_value = std_deviation = "-       "
        else:
            numeric_values = [x for x in values if isinstance(x, (int, float))]
            min_value = min(numeric_values) if numeric_values else "-       "
            max_value = max(numeric_values) if numeric_values else "-       "
            mean_value = std_deviation = "-       "

        data_size = len(values)
        return data_type, data_size, min_value, max_value, mean_value, std_deviation

    @staticmethod
    def _infer_description(name):
        n = name.lower().strip()
        _rules = [
            (['demand', 'd_req', 'requirement'], 'Demand parameter'),
            (['cost', 'price', 'fee', 'expense', 'charge'], 'Cost parameter'),
            (['setup', 'ordering', 'order_cost'], 'Setup/ordering cost parameter'),
            (['hold', 'inventory', 'storage'], 'Holding cost parameter'),
            (['cap', 'capacity', 'limit', 'max_cap', 'maximum'], 'Capacity parameter'),
            (['time', 'duration', 'lead', 'period', 'horizon'], 'Time parameter'),
            (['prob', 'probability', 'chance', 'likelihood'], 'Probability parameter'),
            (['weight', 'wt'], 'Weight parameter'),
            (['revenue', 'return', 'income', 'profit'], 'Revenue parameter'),
            (['budget', 'fund', 'resource'], 'Budget parameter'),
            (['fixed'], 'Fixed cost parameter'),
            (['variable'], 'Variable cost parameter'),
            (['loss', 'penalty', 'fine', 'shortage'], 'Loss/penalty parameter'),
            (['qty', 'quantity', 'amount', 'volume'], 'Quantity parameter'),
            (['salvage', 'scrap', 'residual'], 'Salvage parameter'),
            (['rate', 'ratio', 'ratio'], 'Rate parameter'),
            (['speed', 'velocity'], 'Speed parameter'),
            (['distance', 'dist', 'length', 'width', 'height'], 'Distance/dimension parameter'),
            (['temperature', 'temp'], 'Temperature parameter'),
            (['energy', 'power', 'fuel'], 'Energy parameter'),
            (['capacity_factor', 'utilization', 'util'], 'Utilization parameter'),
        ]
        for keywords, desc in _rules:
            for kw in keywords:
                if kw in n or n == kw:
                    return desc
        if len(n) == 1:
            return f"Parameter {name}"
        return f"Parameter {name}"

    def __keep(self, name, value, neglect=False, description=None):
        _scalar_types = (int, float, np.integer, np.floating)
        if name in self.data:
            existing = self.data[name]
            if isinstance(existing, DataRef):
                if description is not None:
                    self.descriptions[name] = description
                return existing
            if isinstance(existing, _scalar_types):
                if description is not None:
                    self.descriptions[name] = description
                return DataRef(self.data, name)
            if description is not None:
                self.descriptions[name] = description
            return existing
        if isinstance(value, np.ndarray) and value.dtype.kind in ('i', 'f'):
            value = NativeArray(value)
        elif isinstance(value, np.ndarray) and value.dtype == object:
            # object arrays of numeric cells (slices of mixed tables, arrays
            # built from python objects) behave exactly like numeric data
            # once converted; genuinely non-numeric objects stay untouched.
            # Left as object, every arithmetic on them (and therefore the
            # constraint residuals built from them) stays object too.
            try:
                value = NativeArray(value.astype(float))
            except (TypeError, ValueError):
                pass
        if self.measure == True:
            raw = value._arr if isinstance(value, NativeArray) else value
            self.type_params[name], self.size_params[name], self.minimum_params[name],self.maximum_params[name],self.average_params[name],self.std_params[name] = self.__calculate_stats(raw)
            try:
                self.max_among_all_params = max(self.maximum_params[name],self.max_among_all_params)
                self.min_among_all_params = min(self.minimum_params[name],self.min_among_all_params)
            except:
                pass
            try:
                if self.max_among_all_params:
                    self.possible_epsilon  = 1/self.max_among_all_params
            except:
                pass
            self.possible_big_m = self.max_among_all_params
            try:
                self.size+=self.__calculate_total_size(raw)
            except:
                print("warning: exception for {name} in size calculation. Ignoring real size.")
                self.size+=1
        if description is not None:
            self.descriptions[name] = description
        elif name not in self.descriptions:
            self.descriptions[name] = self._infer_description(name)
        if self.memorize and neglect==False:
            self.data[name]=value
            if isinstance(value, _scalar_types):
                return DataRef(self.data, name)
            if isinstance(value, (list, tuple)) or isinstance(value, NativeArray):
                return DataRef(self.data, name)
            return value
        elif neglect:
            return value
        else:
            return value
    
    # === Sets

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
        elif isinstance(obj, pl.DataFrame):
            return {
                "__type__": "dataframe",
                "columns": obj.columns,
                "data": obj.to_dicts()
            }
        elif isinstance(obj, pl.Series):
            return {
                "__type__": "series",
                "name": obj.name,
                "data": obj.to_list()
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
        if "shape" in dct and "data" in dct:
            return np.array(dct["data"]).reshape(dct["shape"])
        elif "columns" in dct and "data" in dct:
            return pl.DataFrame(dct["data"])
        elif "name" in dct and "data" in dct:
            try:
                return pl.Series(name=dct["name"], values=dct["data"])
            except Exception:
                pass  
        elif isinstance(dct, dict) and "data" in dct:
            if "__type__" in dct and dct["__type__"] == "set":
                return set(dct["data"])
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
    
    def array(self, input):
        return np.array(input)

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

    def display(self, input=None):
        from pprint import pprint
        if input:
            pprint(input, width=90, indent=0, sort_dicts=True)
        else:
            pprint(self.data, width=90, indent=0, sort_dicts=True)
    
    # === Parameters

    def _sample_list_or_array(self, name, init, size, replace=False, sort_result=False, return_indices=False, axis=None):
        
        if isinstance(init, (list, range)):
            init = np.array(init)

        if axis is None:
            sampled_indices = self.random.choice(init.size, size=size, replace=replace)
        else:
            axis = np.atleast_1d(axis)
            
            for ax in axis:
                axis_size = init.shape[ax]
                sampled_indices = self.random.choice(axis_size, size=size, replace=replace)
            
                init = np.take(init, sampled_indices, axis=ax)
        
        if return_indices:
            if sort_result:
                sampled_indices.sort()
            result = sampled_indices
        else:
            result = init
            if sort_result:
                result.sort()

        return result

    def _sample_polars_dataframe(self, name, init, size, replace=False, sort_result=False, return_indices=False, axis=None):
        axis = 0 if axis is None else axis 

        if axis not in [0, 1]:
            raise ValueError("Invalid axis for Polars DataFrame sampling. Supported axes: 0 (rows), 1 (columns)")

        if axis == 0:
            n_rows = init.shape[0]
            sampled_indices = self.random.choice(n_rows, size=size, replace=replace)
        else:
            n_cols = init.shape[1]
            sampled_indices = self.random.choice(n_cols, size=size, replace=replace)

        if return_indices:
            sampled_data = sampled_indices
        else:
            if axis == 0:
                sampled_data = init[sampled_indices, :]
            else:
                sampled_data = init[:, sampled_indices]

            if sort_result:
                sampled_data = sampled_data.sort(init.columns[0] if axis == 1 else init.columns)
                
        return sampled_data

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
    
    def uniformint(self, name, dim=0, bound=[1, 10], neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.integers(low=bound[0], high=bound[1] + 1)
        else:
            if type(dim)==set:
                result = {key: self.random.integers(low=bound[0], high=bound[1] + 1) for key in dim}
            else:
                result = self.random.integers(low=bound[0], high=bound[1] + 1, size=dim)
        return self.__keep(name, result, neglect)
    
    def bernoulli(self, name, dim=0, p=0.5, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.choice([0, 1], p=[1-p, p])
        else:
            if type(dim)==set:
                result = {key: self.random.choice([0, 1], p=[1-p, p]) for key in dim}
            else:
                result = self.random.choice([0, 1], p=[1-p, p], size=dim)
        return self.__keep(name, result, neglect)
    
    def binomial(self, name, dim=0, n=None, p=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.binomial(n, p)
        else:
            if type(dim)==set:
                result = {key: self.random.binomial(n, p) for key in dim}
            else:
                result = self.random.binomial(n, p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def poisson(self, name, dim=0, lam=1, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.poisson(lam)
        else:
            if type(dim)==set:
                result = {key: self.random.poisson(lam) for key in dim}
            else:
                result = self.random.poisson(lam, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def geometric(self, name, dim=0, p=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.geometric(p)
        else:
            if type(dim)==set:
                result = {key: self.random.geometric(p) for key in dim}
            else: 
                result = self.random.geometric(p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def negative_binomial(self, name, dim=0, r=None, p=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.negative_binomial(r, p)
        else:
            if type(dim)==set:
                result = {key: self.random.negative_binomial(r, p) for key in dim}
            else:
                result = self.random.negative_binomial(r, p, size=tuple(dim))
        return self.__keep(name, result, neglect)

    def hypergeometric(self, name, dim=0, N=None, m=None, n=None, neglect=False):
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

    def uniform(self, name, dim=0, bound=[0, 1], neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.uniform(low=bound[0], high=bound[1])
        else:
            if type(dim)==set:
                result = {key: self.random.uniform(low=bound[0], high=bound[1]) for key in dim}
            else:
                result = self.random.uniform(low=bound[0], high=bound[1], size=dim)
        return self.__keep(name, result, neglect)

    def rsum(self, name, total: float, dim=0, bound=[0, 1], neglect=False):
        if total <= 0:
            raise ValueError(f"Target sum must be positive. Got: {total}")

        dim = self.__fix_dims(dim, is_range=False)
        if isinstance(dim, set):
            dim = len(dim)

        if dim == 0:
            val = self.random.uniform(low=bound[0], high=bound[1])
            return self.__keep(name, total if not neglect else val, neglect)

        try:
            data = self.random.uniform(low=bound[0], high=bound[1], size=dim)
            s = data.sum()
            if s == 0:
                raise ValueError("Generated random values sum to zero, cannot normalize")
            scaled = data * (total / s)
        except Exception as e:
            raise RuntimeError(f"Failed to generate scaled random values: {e}")

        return self.__keep(name, scaled, neglect)

    def normal(self, name, dim=0, mu=0, sigma=1, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.normal(mu, sigma)
        else:
            if type(dim)==set:
                result = {key: self.random.normal(mu, sigma) for key in dim}
            else:
                result = self.random.normal(mu, sigma, size=dim)
        return self.__keep(name, result, neglect)

    def standard_normal(self, name, dim=0, neglect=False):
        dim = self.__fix_dims(dim,is_range=False)
        if dim == 0:
            result = self.random.normal(0, 1)
        else:
            if type(dim)==set:
                result = {key: self.random.normal(0, 1) for key in dim}
            else:
                result = self.random.normal(0, 1, size=dim)
        return self.__keep(name, result, neglect)

    def exponential(self, name, dim=0, lam=1.0, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.exponential(scale=1/lam)
        else:
            if type(dim)==set:
                result = {key: self.random.exponential(scale=1/lam) for key in dim}
            else:
                result = self.random.exponential(scale=1/lam, size=dim)
        return self.__keep(name, result, neglect)

    def gamma(self, name, dim=0, alpha=1, lam=1, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.gamma(shape=alpha, scale=1/lam)
        else:
            if type(dim)==set:
                result = {key: self.random.gamma(shape=alpha, scale=1/lam) for key in dim}
            else:
                result = self.random.gamma(shape=alpha, scale=1/lam, size=dim)
        return self.__keep(name, result, neglect)

    def erlang(self, name, dim=0, alpha=1, lam=1, neglect=False):
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

    def beta(self, name, dim=0, a=1, b=1, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.beta(a, b, size=None)
        else:
            if type(dim)==set:
                result = {key: self.random.beta(a, b)  for key in dim}
            else:
                result = self.random.beta(a, b, size=dim)
        return self.__keep(name, result, neglect)

    def weibull(self, name, dim=0, alpha=None, beta=None, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = alpha * self.random.weibull(a=beta)
        else:
            if type(dim)==set:
                result = {key: alpha * self.random.weibull(a=beta) for key in dim}
            else: 
                result = alpha * self.random.weibull(a=beta, size=dim)
        return self.__keep(name, result, neglect)

    def cauchy(self, name, dim=0, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if dim == 0:
            result = self.random.standard_cauchy()
        else:
            if type(dim)==set:
                result = {key: self.random.standard_cauchy() for key in dim}
            else:
                result = self.random.standard_cauchy(size=dim)
        return self.__keep(name, result, neglect)

    def dirichlet(self, name, dim=0, k=None, alpha=None, neglect=False):
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
        colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        dim = self.__fix_dims(dim,is_range=False)
        dim = [range(i) for i in dim]
        if dim == 0:    
            if with_names:
                result = self.random.choice(list(colors_dict.keys()))
            else:
                result = '#{:06x}'.format(self.random.integers(0, 0xFFFFFF))
        else:
            if len(dim) == 1:
                if with_names:
                    result = {key: self.random.choice(list(colors_dict.keys())) for key in dim[0]}
                else:
                    result = {key: '#{:06x}'.format(self.random.integers(0, 0xFFFFFF)) for key in dim[0]}
            else:
                if with_names:
                    result = {key: self.random.choice(list(colors_dict.keys())) for key in it.product(*dim)}
                else:
                    result = {key: '#{:06x}'.format(self.random.integers(0, 0xFFFFFF)) for key in it.product(*dim)}
        return self.__keep(name, result, neglect)
    
    def distance(self, name, dim=0, bound=[0, 1], symmetric=True, as_int=False, neglect=False):
        dim = self.__fix_dims(dim, is_range=False)
        if as_int:
            mat = self.random.integers(low=bound[0], high=bound[1] + 1, size=dim)
        else:
            mat = self.random.uniform(low=bound[0], high=bound[1], size=dim)

        if symmetric:
            mat = (mat + mat.T) / 2
            if as_int:
                mat = np.round(mat).astype(int)
        np.fill_diagonal(mat, 0)
        return self.__keep(name, mat, neglect)
    
    def points(
        self,
        name,
        n=0,
        dim=2,
        bound=None,
        geo=False,
        as_int=False,
        country=None,
        city=None,
        custom_polygon=None,
        mode="euclidean",
        max_travel_time=None,
        max_tries=5,
        return_distances=False,
        neglect=False
    ):
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required. Install via `pip install numpy`.")

        if not isinstance(n, int) or n < 0:
            raise ValueError(f"`n` must be a nonnegative integer; got {n}.")
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(f"`dim` must be a nonnegative integer; got {dim}.")

        allowed_modes = {"walk", "drive", "ship", "train", "air", "euclidean", "manhattan"}
        if mode not in allowed_modes:
            raise ValueError(f"`mode` must be one of {allowed_modes}; got '{mode}'.")

        speed_map = {
            "walk": 5.0,
            "drive": 60.0,
            "ship": 30.0,
            "train": 80.0,
            "air": 800.0
        }

        def _haversine_matrix(latlons):
            R = 6371.0
            lat = np.radians(latlons[:, 0])[:, None]
            lon = np.radians(latlons[:, 1])[:, None]
            dlat = lat - lat.T
            dlon = lon - lon.T
            a = (np.sin(dlat / 2))**2 + np.cos(lat) * np.cos(lat.T) * (np.sin(dlon / 2))**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        def _sample_in_shape(n_pts, shape, as_int_flag):
            try:
                from shapely.geometry import Point
            except ImportError:
                raise ImportError("Shapely is required. Install via `pip install shapely`.")
            rng = self.random
            minx, miny, maxx, maxy = shape.bounds
            pts = []
            batch = max(n_pts * 3, 100)
            while len(pts) < n_pts:
                xs = rng.uniform(minx, maxx, size=batch)
                ys = rng.uniform(miny, maxy, size=batch)
                for x, y in zip(xs, ys):
                    p = Point(x, y)
                    if shape.contains(p):
                        pts.append((y, x))
                        if len(pts) == n_pts:
                            break
            arr = np.array(pts, dtype=float)
            if as_int_flag:
                arr = np.round(arr).astype(int)
            return arr

        if not hasattr(self, "_graph_cache"):
            self._graph_cache = {}
        if not hasattr(self, "_countries_gdf"):
            self._countries_gdf = None

        cache_dir = "osm_cache"
        os.makedirs(cache_dir, exist_ok=True)

        def _network_samples_and_distances(n_pts, city_name, net_mode, as_int_flag):
            try:
                import osmnx as ox
            except ImportError:
                raise ImportError("OSMnx is required. Install via `pip install osmnx`.")
            try:
                import networkx as nx
            except ImportError:
                raise ImportError("NetworkX is required. Install via `pip install networkx`.")

            safe_city = city_name.replace(" ", "_").replace(",", "")
            filename = f"{safe_city}_{net_mode}.graphml"
            filepath = os.path.join(cache_dir, filename)

            if (city_name, net_mode) in self._graph_cache:
                G = self._graph_cache[(city_name, net_mode)]
            elif os.path.isfile(filepath):
                try:
                    G = ox.load_graphml(filepath)
                except Exception:
                    G = ox.graph_from_place(city_name, network_type=net_mode if net_mode in ("walk", "drive") else "all",
                                            custom_filter='["route"="ferry"]' if net_mode == "ship" else None)
                    ox.save_graphml(G, filepath)
                self._graph_cache[(city_name, net_mode)] = G
            else:
                if net_mode in ("walk", "drive"):
                    G = ox.graph_from_place(city_name, network_type=net_mode)
                else:
                    G = ox.graph_from_place(
                        city_name,
                        network_type="all",
                        custom_filter='["route"="ferry"]'
                    )
                    if len(G.nodes) == 0:
                        raise ValueError(f"No ferry/ship routes found in '{city_name}'.")
                ox.save_graphml(G, filepath)
                self._graph_cache[(city_name, net_mode)] = G

            rng = self.random
            all_nodes = list(G.nodes)
            if n_pts > len(all_nodes):
                raise ValueError(f"Requested {n_pts} points, but graph has only {len(all_nodes)} nodes.")
            chosen_nodes = rng.choice(all_nodes, size=n_pts, replace=False)

            latlons = []
            for node in chosen_nodes:
                data = G.nodes[node]
                lat = data.get("y")
                lon = data.get("x")
                if lat is None or lon is None:
                    raise RuntimeError(f"Node {node} lacks 'x'/'y'.")
                latlons.append((lat, lon))
            pts_arr = np.array(latlons, dtype=float)
            if as_int_flag:
                pts_arr = np.round(pts_arr).astype(int)

            dist_mat = np.zeros((n_pts, n_pts), dtype=float)
            for i, src in enumerate(chosen_nodes):
                lengths = nx.single_source_dijkstra_path_length(G, src, weight="length")
                for j, tgt in enumerate(chosen_nodes):
                    dist_mat[i, j] = lengths.get(tgt, np.inf)
            return pts_arr, dist_mat / 1000.0

        def _geo_samples_and_haversine(n_pts, as_int_flag):
            if city is not None:
                try:
                    import osmnx as ox
                except ImportError:
                    raise ImportError("OSMnx is required. Install via `pip install osmnx`.")
                try:
                    gdf_c = ox.geocode_to_gdf(city)
                except Exception as e:
                    raise ValueError(f"OSMnx could not geocode '{city}': {e}")
                if gdf_c.empty:
                    raise ValueError(f"City '{city}' not found by OSMnx.")
                shape = gdf_c.unary_union
                return _sample_in_shape(n_pts, shape, as_int_flag)
            if country is not None:
                try:
                    import geopandas as gpd
                except ImportError:
                    raise ImportError("GeoPandas is required. Install via `pip install geopandas`.")
                if self._countries_gdf is None:
                    try:
                        self._countries_gdf = gpd.read_file(
                            "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
                        )
                    except Exception as e:
                        raise RuntimeError(f"Failed to load country boundaries. Details: {e}")
                world = self._countries_gdf
                cols = world.columns
                for candidate in ("ADMIN", "admin", "NAME", "name", "Country", "country"):
                    if candidate in cols:
                        name_col = candidate
                        break
                else:
                    raise ValueError(f"No country-name column found. Columns: {cols.tolist()}")
                matches = world[world[name_col].str.lower() == country.lower()]
                if matches.empty:
                    sample_names = ", ".join(sorted(world[name_col].unique())[:10]) + ", …"
                    raise ValueError(f"Country '{country}' not found. Examples: {sample_names}")
                shape = matches.iloc[0].geometry
                return _sample_in_shape(n_pts, shape, as_int_flag)
            if custom_polygon is not None:
                try:
                    from shapely.geometry import Polygon, MultiPolygon
                except ImportError:
                    raise ImportError("Shapely is required. Install via `pip install shapely`.")
                if dim != 2:
                    raise ValueError("custom_polygon requires dim=2.")
                if not isinstance(custom_polygon, (Polygon, MultiPolygon)):
                    raise ValueError("custom_polygon must be a Shapely Polygon or MultiPolygon.")
                return _sample_in_shape(n_pts, custom_polygon, as_int_flag)
            if geo:
                if bound is None:
                    bound_list = [[-90, 90], [-180, 180]]
                else:
                    if (
                        isinstance(bound, (list, tuple))
                        and len(bound) == 2
                        and all(np.isscalar(x) for x in bound)
                    ):
                        bound_list = [list(bound) for _ in range(2)]
                    elif (
                        isinstance(bound, (list, tuple))
                        and len(bound) == 2
                        and all(isinstance(axis, (list, tuple)) and len(axis) == 2 for axis in bound)
                    ):
                        bound_list = [list(axis) for axis in bound]
                    else:
                        raise ValueError("`bound` for geo sampling must be [lat_min, lat_max], [lon_min, lon_max].")
                lat_min, lat_max = bound_list[0]
                lon_min, lon_max = bound_list[1]
                if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                    raise ValueError(f"Latitude bounds must lie in [-90,90]. Got [{lat_min},{lat_max}].")
                if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                    raise ValueError(f"Longitude bounds must lie in [-180,180]. Got [{lon_min},{lon_max}].")
                lats = self.random.uniform(lat_min, lat_max, size=n_pts)
                lons = self.random.uniform(lon_min, lon_max, size=n_pts)
                pts_arr = np.column_stack((lats, lons))
                if as_int_flag:
                    pts_arr = np.round(pts_arr).astype(int)
                return pts_arr
            raise ValueError("Geo sampling requires city, country, custom_polygon, or geo=True.")

        if mode in ("walk", "drive", "ship"):
            if dim != 2:
                raise ValueError(f"mode='{mode}' only valid when dim=2.")
            speed = speed_map[mode]
            for attempt in range(max_tries):
                if city:
                    pts_arr, dist_mat = _network_samples_and_distances(n, city, mode, as_int)
                else:
                    pts_arr = _geo_samples_and_haversine(n, as_int)
                    dist_mat = _haversine_matrix(pts_arr)
                if max_travel_time is None:
                    break
                travel_time = dist_mat / speed
                if np.all(np.isfinite(travel_time) & (travel_time <= max_travel_time)):
                    break
            else:
                raise ValueError(
                    f"Could not generate {n} points within {max_travel_time}h by {mode} after {max_tries} tries."
                )
            stored = self.__keep(name, pts_arr, neglect)
            dist_mat = self.__keep(name+"_dist", dist_mat, neglect)
            if return_distances:
                return stored, dist_mat
            return stored

        if mode in ("train", "air"):
            if dim != 2:
                raise ValueError(f"mode='{mode}' requires dim=2.")
            speed = speed_map[mode]
            for attempt in range(max_tries):
                pts_arr = _geo_samples_and_haversine(n, as_int)
                if max_travel_time is not None:
                    dist_mat = _haversine_matrix(pts_arr)
                    travel_time = dist_mat / speed
                    if np.all(np.isfinite(travel_time) & (travel_time <= max_travel_time)):
                        break
                else:
                    break
            else:
                raise ValueError(
                    f"Could not generate {n} points within {max_travel_time}h by {mode} after {max_tries} tries."
                )
            stored = self.__keep(name, pts_arr, neglect)
            if return_distances:
                dist_mat = _haversine_matrix(pts_arr)
                dist_mat = self.__keep(name+"_dist", dist_mat, neglect)
                return stored, dist_mat
            return stored

        if mode in ("euclidean", "manhattan"):
            if bound is None:
                bound_list = [[0, 1]] * dim
            else:
                if (
                    isinstance(bound, (list, tuple))
                    and len(bound) == 2
                    and all(np.isscalar(x) for x in bound)
                ):
                    bound_list = [list(bound) for _ in range(dim)]
                elif (
                    isinstance(bound, (list, tuple))
                    and len(bound) == dim
                    and all(isinstance(axis, (list, tuple)) and len(axis) == 2 for axis in bound)
                ):
                    bound_list = [list(axis) for axis in bound]
                else:
                    raise ValueError(f"`bound` must be [low, high] or list of length {dim} of [low_i, high_i].")
            for i, (low_i, high_i) in enumerate(bound_list):
                if not (np.isscalar(low_i) and np.isscalar(high_i)):
                    raise ValueError(f"`bound[{i}]` values must be numeric; got {bound_list[i]}.")
                if low_i > high_i:
                    raise ValueError(f"`bound[{i}] = [{low_i},{high_i}]` invalid: min > max.")
            for attempt in range(max_tries):
                pts_arr = np.empty((n, dim), dtype=float)
                for i in range(dim):
                    low_i, high_i = bound_list[i]
                    pts_arr[:, i] = self.random.uniform(low_i, high_i, size=n)
                if as_int:
                    pts_arr = np.round(pts_arr).astype(int)
                if max_travel_time is not None:
                    if mode == "euclidean":
                        diff = pts_arr[:, None, :] - pts_arr[None, :, :]
                        dist_mat = np.sqrt((diff**2).sum(axis=2))
                    else:
                        diff = pts_arr[:, None, :] - pts_arr[None, :, :]
                        dist_mat = np.abs(diff).sum(axis=2)
                    travel_time = dist_mat
                    if np.all(np.isfinite(travel_time) & (travel_time <= max_travel_time)):
                        break
                else:
                    break
            else:
                raise ValueError(
                    f"Could not generate {n} points within {max_travel_time}h by {mode} after {max_tries} tries."
                )
            stored = self.__keep(name, pts_arr, neglect)
            if return_distances:
                if mode == "euclidean":
                    diff = pts_arr[:, None, :] - pts_arr[None, :, :]
                    dist_mat = np.sqrt((diff**2).sum(axis=2))
                else:
                    diff = pts_arr[:, None, :] - pts_arr[None, :, :]
                    dist_mat = np.abs(diff).sum(axis=2)
                    dist_mat = self.__keep(name+"_dist", dist_mat, neglect)
                return stored, dist_mat
            return stored

        raise RuntimeError(f"Unhandled mode '{mode}'.")

    def sample(self, name, init, size, replace=False, sort_result=False, reset_index=False, return_indices=False, axis=None, neglect=False):

        type_is= type(init)
        if type_is ==set:
            init = list(init)
            
        if isinstance(init, (list, set,range, np.ndarray)):
            sample =  self._sample_list_or_array(name, init, size, replace, sort_result, return_indices, axis)
        elif isinstance(init, pl.DataFrame):
            sample = self._sample_polars_dataframe(name, init, size, replace, sort_result, return_indices, axis)
        else:
            raise ValueError("Unsupported data type for sampling. Supported types: set, list, range, numpy.ndarray, polars.DataFrame")

        if type_is ==set:
            sample =  set(sample)
        elif type_is in [list, range]:
            sample = list(sample)
        elif return_indices:
            sample =  set(sample)
        else:
            sample =  sample

        return self.__keep(name, sample, neglect)

    def set_local_parameters(self):

        for key, value in self.data.items():
            locals()[key] = value      

    def set_global_parameters(self):

        for key, value in self.data.items():
            globals()[key] = value

    def load_from_excel(
        self,
        name: str,
        dim=None,
        labels: list = None,
        appearance: list = None,
        file_name: str = "data.xlsx",
        neglect: bool = False
    ):
        # save_to_excel lays a sheet out as `nc` header rows on top, `nr`
        # label columns on the left, then the data block. Read the sheet as
        # a faithful raw grid (None for every empty cell) so those regions
        # can be sliced off exactly as written.  polars' read_excel cannot
        # be trusted here: it silently drops fully blank leading rows and
        # columns, which shifts the whole block and mixes labels into data.
        try:
            from openpyxl import load_workbook

            cache_key = (os.path.abspath(file_name), os.path.getmtime(file_name))
            grids = _EXCEL_GRIDS.get(cache_key)
            if grids is None:
                wb = load_workbook(file_name, data_only=True)
                try:
                    grids = {ws.title: _sheet_grid(ws) for ws in wb.worksheets}
                finally:
                    wb.close()
                for stale in [
                    k for k in _EXCEL_GRIDS if k[0] == cache_key[0] and k != cache_key
                ]:
                    del _EXCEL_GRIDS[stale]
                _EXCEL_GRIDS[cache_key] = grids
            if name not in grids:
                raise ValueError(
                    f"Sheet '{name}' not found; available sheets: {sorted(grids)}"
                )
            grid = grids[name]
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Cannot read sheet '{name}':\n  {e}")

        if grid.size == 0:
            raise ValueError(f"Sheet '{name}' is empty")

        if dim is None:
            # Bare read: dt.lfe(name="...") alone must work, so derive
            # dim/labels/appearance from how the sheet is laid out.
            if labels is not None or appearance is not None:
                raise ValueError(
                    "dim is required when labels or appearance are given; "
                    "omit them to let load_from_excel infer the sheet layout"
                )
            dim, labels, appearance = _infer_sheet_layout(name, grid)

        scalar = (dim == 0)

        if scalar:
            labels = [""]
            appearance = [0, 0]
        elif isinstance(dim, int) and not isinstance(dim, bool):
            # Mirror save_to_excel: a non-zero int describes a single axis.
            dim = [dim]

        if labels is None and not isinstance(dim, int):
            labels = ["" for _ in dim]

        if isinstance(dim, list) and len(dim) >= 1 and isinstance(dim[0], set):
            dim = [len(d) for d in dim]
        dim = self.__fix_dims(dim, is_range=True)

        if isinstance(dim, set):
            raise TypeError(
                "load_from_excel(dim=...) expects sizes, not a set of "
                f"labels: {sorted(dim)!r}"
            )

        if appearance is None and not isinstance(dim, int):
            ndim = len(dim)
            if ndim <= 1:
                appearance = [0, 1]
            else:
                appearance = [ndim - 1, 1]

        if not (isinstance(appearance, (list, tuple)) and len(appearance) == 2):
            raise ValueError("appearance must be a list or tuple of length 2")

        nr, nc = appearance
        if not (isinstance(nr, int) and isinstance(nc, int)):
            raise TypeError("appearance values must be integers")
        if nr < 0 or nc < 0:
            raise ValueError(f"Invalid appearance values: nr={nr}, nc={nc}")

        block = grid[nc:, nr:]
        if block.size == 0:
            raise ValueError(
                f"Sheet '{name}' has no data cells for appearance=[{nr},{nc}]"
            )

        if scalar:
            # dim == 0 promises a single bare cell (a bare read only lands
            # here when _infer_sheet_layout saw exactly one value).
            # save_to_excel infers a shape for multi-cell arrays instead, so
            # a sheet with more than one value cannot be described by
            # dim == 0 -- say so rather than handing back whatever sits in
            # the top-left corner.
            # Filter out empty cells -- a scalar sheet may hold its value
            # anywhere in the used range, with surrounding blanks that
            # arrive as None (object columns), NaN (numeric columns) or "".
            non_null = np.array(
                [
                    v
                    for v in block.ravel()
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))
                    and v != ""
                ],
                dtype=object,
            )
            if non_null.size != 1:
                raise ValueError(
                    f"Sheet '{name}' holds {int(non_null.size)} values but "
                    "dim=0 describes a single bare cell; pass dim=<shape> "
                    "to read an array"
                )
            result = non_null[0]
            try:
                result = float(result)
            except (TypeError, ValueError):
                pass
            return self.__keep(name, result, neglect)

        sizes = tuple(
            len(d) if not isinstance(d, (int, np.integer)) else int(d)
            for d in dim
        )

        # dim[:nr] indexes the label columns, and every header row that
        # really labels the value region indexes the next dims.  Values are
        # then placed by those labels -- the way the previous pandas loader
        # did its .loc lookups -- so the row order in the sheet never has to
        # match C order (e.g. scp_ht groups its rows h-outer/t-inner while
        # its label columns are ordered t, h).
        header = grid[:nc, nr:]
        col_dim_rows = [
            i
            for i in range(nc)
            if any(header[i, j] is not None for j in range(header.shape[1]))
        ]
        nr_eff = nr
        if nr_eff == 0 and not col_dim_rows and len(sizes) > 0:
            # No label columns and no labelled header row: keep the old
            # pandas behaviour (index_col=0) and read the first column as
            # the index of the dimensions.
            nr_eff = 1
            header = grid[:nc, nr_eff:]
            col_dim_rows = [
                i
                for i in range(nc)
                if any(header[i, j] is not None for j in range(header.shape[1]))
            ]
        label_grid = grid[nc:, :nr_eff]
        block = grid[nc:, nr_eff:]

        if len(sizes) != nr_eff + len(col_dim_rows):
            raise ValueError(
                f"Sheet '{name}': appearance=[{nr},{nc}] splits the sheet "
                f"into {nr_eff} label column(s) and {len(col_dim_rows)} "
                f"labelled header row(s), i.e. {nr_eff + len(col_dim_rows)} "
                f"dimension(s), but dim has {len(sizes)}"
            )
        if not col_dim_rows and block.shape[1] > 1:
            raise ValueError(
                f"Sheet '{name}' has {block.shape[1]} value columns but no "
                "labelled header row to tell them apart; pass appearance "
                "with nc >= 1"
            )

        # label -> index map per dimension. Prefer exact matching against
        # the declared keys -- the key itself, its str() form and the
        # labels[i] + str(key) form -- which is exactly what the previous
        # pandas loader looked up with .loc.  Junk that matches nothing
        # (a header row read as data when nc undercounts the header rows,
        # rows for keys beyond the declared dim, blank rows) then falls
        # out as unmatched instead of shifting every index.  Sheets whose
        # labels the keys cannot reproduce ('t0', 'i3' with empty labels)
        # match nothing exactly, so they keep the order-of-first-appearance
        # map.
        def _build_map(cells, size, prefix=""):
            cells = [v for v in cells if v is not None]
            prefix = str(prefix) if prefix else ""
            exact = {k: k for k in range(size)}
            exact.update({str(k): k for k in range(size)})
            if prefix:
                exact.update({prefix + str(k): k for k in range(size)})
            if cells and sum(1 for v in cells if v in exact) * 2 >= len(cells):
                return exact
            m = {}
            for v in cells:
                if v not in m:
                    m[v] = len(m)
            return m

        row_maps = [
            _build_map(
                label_grid[:, j],
                sizes[j],
                labels[j] if j < len(labels) else "",
            )
            for j in range(nr_eff)
        ]
        col_maps = [
            _build_map(
                header[i_h, :],
                sizes[nr_eff + slot],
                labels[nr_eff + slot] if nr_eff + slot < len(labels) else "",
            )
            for slot, i_h in enumerate(col_dim_rows)
        ]

        out = np.full(sizes, np.nan, dtype=float)
        for i in range(block.shape[0]):
            ridx = []
            for j in range(nr_eff):
                idx = row_maps[j].get(label_grid[i, j], -1)
                if idx < 0 or idx >= sizes[j]:
                    # Blank label: rows the sheet keeps beyond the declared
                    # dim (left-over formatting) are ignored.
                    break
                ridx.append(idx)
            else:
                for k in range(block.shape[1]):
                    cidx = []
                    for slot, i_h in enumerate(col_dim_rows):
                        idx = col_maps[slot].get(header[i_h, k], -1)
                        if idx < 0 or idx >= sizes[nr_eff + slot]:
                            break
                        cidx.append(idx)
                    else:
                        try:
                            out[tuple(ridx) + tuple(cidx)] = float(block[i, k])
                        except (TypeError, ValueError):
                            # empty or non-numeric cell: reported below
                            pass

        missing = np.argwhere(np.isnan(out))
        if missing.size:
            spots = ", ".join(
                "(" + ", ".join(str(int(x)) for x in idx) + ")"
                for idx in missing[:5]
            )
            raise ValueError(
                f"Sheet '{name}' is missing {len(missing)} of {out.size} "
                f"value(s) for dim={list(sizes)}, appearance=[{nr},{nc}] "
                f"(e.g. at {spots}); check dim, labels and appearance "
                "against the sheet layout"
            )

        return self.__keep(name, out, neglect)
    
    def save_to_excel(
        self,
        name: str,
        array: np.ndarray,
        dim=0,
        labels: list = None,
        appearance: list = None,
        file_name: str = "data.xlsx"
    ) -> None:
        if not (isinstance(dim, (list, tuple)) or isinstance(dim, int)):
            raise TypeError("dim must be int or list/tuple")

        arr = np.asarray(array, dtype=float)

        scalar_sheet = False
        if isinstance(dim, int):
            if dim == 0:
                if arr.size == 1:
                    # ``dim == 0`` denotes a single value: ``load_from_excel``
                    # reads it back as ``result[0][0]``, so lay it out as one
                    # bare cell with no header rows or columns.
                    scalar_sheet = True
                    dim = []
                    appearance = [0, 0]
                else:
                    # No explicit shape was given, so lay the array out as-is.
                    dim = list(arr.shape)
            else:
                # A non-zero int describes a single axis of that length.
                dim = [dim]

        if isinstance(dim, (list, tuple)) and dim and isinstance(dim[0], set):
            dim = [len(s) for s in dim]

        if appearance is None:
            ndim = len(dim)
            if ndim <= 1:
                appearance = [0, 1]
            else:
                appearance = [ndim - 1, 1]

        if labels is None:
            labels = [""] * max(1, len(dim))

        # NOTE: this used to be guarded by `hasattr(self, '__fix_dims')`,
        # which is always False -- the string literal is not name-mangled, so
        # it never matched the real `_DataToolkit__fix_dims`. Call it directly,
        # exactly like load_from_excel does.
        dim = self.__fix_dims(dim, is_range=True)

        if isinstance(dim, set):
            # __fix_dims turns a list of strings into a set (that is a label
            # domain, not a shape). There is no meaningful sheet layout for it.
            raise TypeError(
                "save_to_excel(dim=...) expects sizes, not a set of "
                f"labels: {sorted(dim)!r}"
            )

        def _size(x):
            return x if isinstance(x, int) else len(x)

        if not (isinstance(appearance, (list, tuple)) and len(appearance) == 2):
            raise ValueError("appearance must be a list or tuple of length 2")

        nr, nc = appearance

        dim_len = len(dim)

        if not (isinstance(nr, int) and isinstance(nc, int)):
            raise TypeError("appearance values must be integers")

        if nr < 0 or nc < 0 or nr + nc > dim_len:
            raise ValueError(f"Invalid appearance values: nr={nr}, nc={nc}, dim length={dim_len}")

        if labels is not None and len(labels) < dim_len:
            raise ValueError(f"labels length {len(labels)} is less than dim length {dim_len}")

        if scalar_sheet:
            shape = ()
        elif dim:
            shape = tuple(_size(d) for d in dim)
        else:
            shape = arr.shape
        mat = arr.reshape(shape)

        # The sheet only faithfully represents `dim` when the row axes
        # (dim[:nr]) and the column axes (dim[nr:nr+nc]) account for every
        # axis. Otherwise `_save_2d_sheet` indexes `mat` with a partial tuple
        # and dies on `float(...)`, so reject it up front, readably.
        covered = 1
        for d in dim[:nr + nc]:
            covered *= _size(d)
        if covered != mat.size:
            raise ValueError(
                f"appearance nr={nr}, nc={nc} lays out {covered} cells but "
                f"dim carries {mat.size}; nr + nc must equal len(dim)="
                f"{dim_len} unless every uncovered axis has size 1"
            )

        col_dims = dim[nr:nr + nc]
        total_columns = 1
        for d in col_dims:
            total_columns *= _size(d)

        MAX_COLUMNS = 16384

        if total_columns <= MAX_COLUMNS:
            self._save_2d_sheet(name, mat, dim, nr, nc, labels, file_name, _size)
        else:
            self._save_long_format(name, arr, dim, labels, file_name, _size)

    def _save_2d_sheet(self, name, mat, dim, nr, nc, labels, file_name, _size):
        from openpyxl import Workbook, load_workbook
        import itertools as it
        import os

        wb = load_workbook(file_name) if os.path.exists(file_name) else Workbook()
        if name in wb.sheetnames:
            wb.remove(wb[name])
        ws = wb.create_sheet(title=name)

        col_dims = dim[nr:nr + nc]
        dim_sizes = [_size(d) for d in col_dims]
        all_col_keys = list(it.product(*(range(s) for s in dim_sizes)))

        for level in range(nc):
            for col_idx, keys in enumerate(all_col_keys):
                label_val = f"{labels[nr + level]}{keys[level]}"
                ws.cell(row=level + 1, column=nr + col_idx + 1, value=label_val)

        row_dims = dim[:nr]
        row_keys = list(it.product(*(range(_size(d)) for d in row_dims))) if nr else [()]

        for row_idx, rk in enumerate(row_keys):
            for i in range(nr):
                ws.cell(row=nc + 1 + row_idx, column=i + 1, value=f"{labels[i]}{rk[i]}")

        for row_idx, rk in enumerate(row_keys):
            for col_idx, ck in enumerate(all_col_keys):
                full_idx = rk + ck
                val = float(mat[full_idx])
                ws.cell(row=nc + 1 + row_idx, column=nr + col_idx + 1, value=val)

        if 'Sheet' in wb.sheetnames and len(wb.sheetnames) > 1:
            wb.remove(wb['Sheet'])

        wb.save(file_name)
        wb.close()
        _invalidate_grids(file_name)

    def _save_long_format(self, name, arr, dim, labels, file_name, _size):
        from openpyxl import Workbook, load_workbook
        import itertools as it
        import os

        dim_names = []
        for i, lab in enumerate(labels[:len(dim)] if labels else []):
            lab = str(lab).strip() if lab is not None else ""
            base = lab or f"dim_{i}"
            while base in dim_names:
                base = f"{base}_{len(dim_names)}"
            dim_names.append(base)
        value_col = "Value"
        while value_col in dim_names:
            value_col += "_"

        # Written straight to openpyxl, like _save_2d_sheet: the long format
        # only exists for sheets whose column count exceeds what a 2-D layout
        # can hold, and polars' ExcelWriter is not available in every version.
        wb = load_workbook(file_name) if os.path.exists(file_name) else Workbook()
        if name in wb.sheetnames:
            wb.remove(wb[name])
        ws = wb.create_sheet(title=name)

        for col, dim_name in enumerate(dim_names, start=1):
            ws.cell(row=1, column=col, value=dim_name)
        ws.cell(row=1, column=len(dim_names) + 1, value=value_col)

        row = 2
        for idx in it.product(*(range(_size(d)) for d in dim)):
            for i, v in enumerate(idx, start=1):
                ws.cell(row=row, column=i, value=int(v))
            ws.cell(row=row, column=len(dim_names) + 1, value=float(arr[idx]))
            row += 1

        if 'Sheet' in wb.sheetnames and len(wb.sheetnames) > 1:
            wb.remove(wb['Sheet'])
        wb.save(file_name)
        wb.close()
        _invalidate_grids(file_name)

    def save(self, name, format="json"):
        directory = os.path.join('results', 'data')
        if not os.path.exists(directory):
            os.makedirs(directory)

        extension = format.lower()
        file_path = os.path.abspath(os.path.join(directory, f"{name}.{extension}"))

        if format == "json":
            try:
                with open(file_path, 'w') as file:
                    json.dump(self.data, file, default=self._json_serializer, indent=4)
                print(f"Data successfully exported to JSON at: {file_path}")
            except TypeError as e:
                print(f"Serialization error: {e}")
            except Exception as e:
                print(f"An error occurred while saving JSON: {e}")

        elif format == "parquet":
            try:
                df = pl.DataFrame(self.data)
                df.write_parquet(file_path)
                print(f"Data successfully exported to Parquet at: {file_path}")
            except Exception as e:
                print(f"An error occurred while saving Parquet: {e}")

        else:
            print(f"Unsupported format: {format}. Supported formats: 'json', 'parquet'")

    def load(self, name, format="json", neglect=False):
        extension = format.lower()
        filename = f"{name}.{extension}"
        final_dir = os.path.join('.', 'data', 'final')
        results_dir = os.path.join('results', 'data')
        file_path_final = os.path.join(final_dir, filename)
        file_path_results = os.path.join(results_dir, filename)

        if os.path.exists(file_path_final):
            file_path = os.path.abspath(file_path_final)
        elif os.path.exists(file_path_results):
            file_path = os.path.abspath(file_path_results)
            print(f"Data file found in {results_dir}. For consistency, please move it to {final_dir}.")
        else:
            raise FileNotFoundError(f"File '{filename}' not found. Please place it in '{final_dir}' or '{results_dir}'.")

        data = None
        try:
            if extension == "json":
                with open(file_path, 'r') as file:
                    data = json.load(file, object_hook=self._json_decoder)
                print(f"Data successfully imported from JSON at: {file_path}")

            elif extension == "parquet":
                df = pl.read_parquet(file_path)
                data = df.to_dicts()
                print(f"Data successfully imported from Parquet at: {file_path}")

            else:
                raise ValueError(f"Unsupported format: {format}. Supported formats are 'json' and 'parquet'.")
        except Exception as e:
            print(f"An error occurred while loading {format.upper()}: {e}")
            data = None

        return self.__keep(name, data, neglect)

    def report(self, style=1, width=78, box=None):
        from ..helpers.reporter import report, format_string
        import numpy as _np

        _own_box = box is None

        if not self.data:
            if _own_box:
                box = report(width=width, style=style)
                box._buffered = True
            box.top(left="Data")
            box.row(left="  Empty dataset.")
            box.bottom()
            if _own_box:
                box.render()
            return
        if _own_box:
            box = report(width=width, style=style)
            box._buffered = True

        def _count_nonzero(value):
            if isinstance(value, NativeArray):
                value = value._arr
            if isinstance(value, _np.ndarray):
                return int(_np.count_nonzero(value))
            if isinstance(value, (int, float)):
                return 1 if value != 0 else 0
            if isinstance(value, (list, tuple)):
                return sum(1 for x in value if x != 0)
            if isinstance(value, dict):
                return sum(1 for v in value.values() if v != 0)
            if isinstance(value, set):
                return len(value)
            if isinstance(value, str):
                return len(value)
            return 1

        def _type_char(value):
            if isinstance(value, NativeArray):
                value = value._arr
            if isinstance(value, _np.ndarray):
                ndim = value.ndim
            elif isinstance(value, (list, tuple)):
                ndim = _np.asarray(value).ndim
            elif isinstance(value, (int, float, _np.generic)):
                ndim = 0
            else:
                ndim = 0
            if ndim == 0: return "S"
            if ndim == 1: return "V"
            if ndim == 2: return "M"
            return "T"

        _box_w = getattr(box, 'width', None) or width
        _content = max(int(_box_w) - 4, 0)

        # Column layout priority: value columns (Min/Max/Ave/Std) and
        # the Domain label always show their full content — only the
        # Name column (leftmost) may be shortened with a trailing "..."
        # when the row runs out of space.

        def _fit(text, col_width):
            text = str(text)
            if len(text) <= col_width:
                return text
            if col_width <= 3:
                return text[:col_width]
            return text[:col_width - 3] + "..."

        _rows = []
        for name, value in self.data.items():
            if value is None:
                continue

            _domain = str(self.type_params.get(name, '?')).strip()
            _size = self.size_params.get(name, '?')
            _min = self.minimum_params.get(name, '?')
            _max = self.maximum_params.get(name, '?')
            _ave = self.average_params.get(name, '?')
            _std = self.std_params.get(name, '?')

            try:
                _nz = _count_nonzero(value)
            except Exception:
                _nz = '?'

            try:
                _tchar = _type_char(value)
            except Exception:
                _tchar = '?'

            _cells = []
            if _min is not None and _max is not None:
                try:
                    _cells = [format_string(_min), format_string(_max),
                              format_string(_ave), format_string(_std)]
                except Exception:
                    _size, _cells = '?', []
            _rows.append((name, _domain, _tchar, _size, _nz, _cells))

        # Data-driven widths for the four value columns (a floor of 7
        # keeps the historical look); the Domain column fits its widest
        # label (never trimmed), and T/S/NZ fit their widest entries.
        _vw = [7, 7, 7, 7]
        for _r in _rows:
            if _r[5]:
                for _j in range(4):
                    if len(_r[5][_j]) > _vw[_j]:
                        _vw[_j] = len(_r[5][_j])
        _s_w = max(3, max((len(str(_r[3])) for _r in _rows), default=0))
        _nz_w = max(4, max((len(str(_r[4])) for _r in _rows), default=0))
        _domain_w = max(6, max((len(_r[1]) for _r in _rows), default=0))

        # Whatever room remains after values, domain and T/S/NZ belongs
        # to the Name column — it is the only column allowed to be
        # trimmed (the floor of 4 keeps the "Name" header plus at least
        # one name character before the "...").
        _name_w = max(
            _content - _domain_w - _s_w - _nz_w - sum(_vw) - 9, 4)

        def _emit_row(prefix, cells):
            """Emit one table row.

            Value cells wrap onto continuation rows aligned under the
            value columns instead of being clipped, so values are
            always shown in full.
            """
            indent = " " * min(len(prefix), max(_content - 8, 0))
            line = prefix
            for _j, cell in enumerate(cells):
                piece = str(cell).rjust(_vw[_j])
                while True:
                    sep = " " if line and line[-1] != " " else ""
                    if len(line) + len(sep) + len(piece) <= _content:
                        line += sep + piece
                        break
                    box.row(left=line)
                    line = indent
                    if len(line) + len(piece) <= _content:
                        continue
                    # A single cell wider than the row: split it across
                    # continuation rows — full content, never trimmed.
                    room = _content - len(line)
                    if room <= 0:
                        break
                    box.row(left=line + piece[:room])
                    piece = piece[room:]
                    line = indent
            box.row(left=line)

        box.top(left="Data")
        _emit_row(
            f"{'Name':<{_name_w}} {'Domain':<{_domain_w}} "
            f"{'T':>1} {'S':>{_s_w}} {'NZ':>{_nz_w}} ",
            ['Min', 'Max', 'Ave', 'Std'])

        for name, _domain, _tchar, _size, _nz, _cells in _rows:
            _emit_row(
                f"{_fit(name, _name_w):<{_name_w}} "
                f"{_domain:<{_domain_w}} "
                f"{_tchar:>1} {_size:>{_s_w}} {_nz:>{_nz_w}} ",
                _cells)

        box.bottom()

        if _own_box:
            box.render()

    
data_toolkit = DataToolkit
