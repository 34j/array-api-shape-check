from collections import defaultdict

from array_api.latest import Array
from more_itertools import partition
from typing import Sequence, Hashable
from types import EllipsisType
import re
import attrs
import numpy as np

@attrs.frozen(kw_only=True)
class SubscriptInfoFromSubcriptItem:
    name: str
    is_variable: bool = False

@attrs.frozen(kw_only=True)
class SubscriptInfoFromSubcript:
    all: tuple[tuple[SubscriptInfoFromSubcriptItem, ...], ...]
    unique: set[SubscriptInfoFromSubcriptItem]

@attrs.frozen(kw_only=True)
class SubscriptInfoFromShapeItemTemp(SubscriptInfoFromSubcriptItem):
    shape_current: tuple[int, ...]

@attrs.frozen(kw_only=True)
class SubscriptInfoFromShapeItem(SubscriptInfoFromShapeItemTemp):
    shape_broadcasted: tuple[int, ...]

@attrs.frozen(kw_only=True)
class SubscriptInfoFromShape:
    all: tuple[tuple[SubscriptInfoFromShapeItem, ...], ...]
    unique: set[SubscriptInfoFromShapeItemTemp]

def parse_subscripts(subscripts: str) -> SubscriptInfoFromSubcript:
    # If . other than ...
    if re.search(r"(?<!\.)\.(?!\.)", subscripts):
        raise ValueError("Invalid subscript: '.' is not allowed except for '...'")
    # Replace ... with *.
    subscripts = re.sub(r"\.\.\.", "*.", subscripts)
    subscripts = subscripts.rstrip()
    info_all: tuple[tuple[SubscriptInfoFromSubcriptItem, ...], ...] = ()
    info_array: tuple[SubscriptInfoFromSubcriptItem, ...] = ()
    for name in subscripts.split(","):
        is_variable = False
        if name == ",":
            info_all.append(info_array)
            info_array = []
        elif name == "*":
            is_variable = True
            continue
        else:
            info_array += (SubscriptInfoFromSubcriptItem(name=name, is_variable=is_variable),)
    else:
        info_all += (tuple(info_array),)

    info_unique: set[SubscriptInfoFromSubcriptItem] = set([x for info_array in info_all for x in info_array])
    # raise if there are subscripts with same name but different is_variable
    names = set()
    for info in info_unique:
        if info.name in names:
            raise ValueError(f"Subscript '{info.name}' is duplicated with different variable status")
        names.add(info.name)

    return SubscriptInfoFromSubcript(all=info_all, unique=info_unique)


def parse_variable_ndim(subscripts: str, ndims: Sequence[int]) -> dict[str, int]:
    info = parse_subscripts(subscripts)
    del subscripts

    # decide dimensions by solving linear equations
    if len(info.unique) > len(info.all):
        raise ValueError(f"Number of unique subscripts ({len(info.unique)}) is greater than number of operands ({len(info_all)}), making it impossible to assume the number of dimensions for each variable subscript")
    rhs = np.asarray(ndims, dtype=int)
    del ndims
    subscripts_variable_unique = [subscript for subscript in info.unique if subscript.is_variable]
    mat = np.zeros((len(info.unique), len(subscripts_variable_unique)), dtype=int)
    for i, info_array in enumerate(info.all):
        for subscript in info_array:
            if subscript.is_variable:
                j = subscripts_variable_unique.index(subscript)
                mat[i, j] += 1
            else:
                rhs[i] -= 1
    
    # solve overdetermined linear equations using least squares method
    variable_dims, residuals, rank, singular_values = np.linalg.lstsq(mat, rhs, rcond=None)
    if residuals.size > 0 and not np.isclose(residuals[0], 0):
        raise ValueError("Inconsistent shapes: no solution exists for the given subscripts and operand shapes")
    variable_dims = np.round(variable_dims).astype(int)
    if np.any(variable_dims < 0):
        raise ValueError("Invalid shapes: variable dimensions cannot be negative")
    return {subscript.name: variable_dims[j] for j, subscript in enumerate(subscripts_variable_unique)}
    

def check_shapes(subscripts: str, /, *operands: Array | tuple[int, ...]) -> None | SubscriptInfo:
    info = parse_subscripts(subscripts)
    if len(info.all) != len(operands):
        raise ValueError(f"Number of subscripts ({len(info.all)}) does not match number of operands ({len(operands)})")
    shapes: list[tuple[int, ...]] = []
    for operand in operands:
        shape: tuple[int, ...]
        if isinstance(operand, tuple):
            shape = operand
        elif hasattr(operand, "shape"):
            if None in operand.shape:
                raise ValueError("Operand shape cannot contain None")
            shape = operand.shape # type: ignore[assignment]
        else:
            raise TypeError(f"Invalid operand: expected an array or a shape tuple, but got {type(operand)}")
        shapes.append(shape)
    del operands
    ndims = [len(shape) for shape in shapes]
    name_to_ndim = defaultdict(lambda: 1, parse_variable_ndim(subscripts, ndims))
    del subscripts, ndims
    name_to_shapes: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
    info_all: tuple[tuple[SubscriptInfoFromShapeItemTemp, ...], ...] = ()
    for info_array, shape in zip(info.all, shapes):
        info_array_new = ()
        for item in info_array:
            info_array_new += (SubscriptInfoFromShapeItemTemp(name=item.name, is_variable=item.is_variable, shape_current=shape[name_to_ndim[item.name]:]),)
    


        
