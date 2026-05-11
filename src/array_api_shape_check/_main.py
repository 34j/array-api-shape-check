from array_api.latest import Array
from more_itertools import partition
from typing import Sequence, Hashable
from types import EllipsisType
import re
import attrs
import numpy as np

@attrs.frozen(kw_only=True)
class SubscriptInfoFromSubcript:
    name: str
    is_variable: bool = False

@attrs.frozen
class SubscripInfoFromNdim(SubscriptInfoFromSubcript):
    ndim: int

@attrs.frozen
class SubscriptInfoFromShape(SubscriptInfoFromSubcript):
    shape: tuple[int, ...] | None = None

def parse_subscripts(subscripts: str) -> tuple[tuple[SubscriptInfoFromSubcript, ...], ...]:
    # If . other than ...
    if re.search(r"(?<!\.)\.(?!\.)", subscripts):
        raise ValueError("Invalid subscript: '.' is not allowed except for '...'")
    # Replace ... with *.
    subscripts = re.sub(r"\.\.\.", "*.", subscripts)
    subscripts = subscripts.rstrip()
    info_all_array: Sequence[Sequence[SubscriptInfoFromSubcript]] = []
    info_array: Sequence[SubscriptInfoFromSubcript] = []
    for name in subscripts.split(","):
        is_variable = False
        if name == ",":
            info_all_array.append(info_array)
            info_array = []
        elif name == "*":
            is_variable = True
            continue
        else:
            info_array.append(SubscriptInfoFromSubcript(name=name, is_variable=is_variable))
    else:
        info_all_array.append(tuple(info_array))

    subscripts_unique: set[SubscriptInfoFromSubcript] = set([x for info_array in info_all_array for x in info_array])
    # raise if there are subscripts with same name but different is_variable
    names = set()
    for subscript in subscripts_unique:
        if subscript.name in names:
            raise ValueError(f"Subscript '{subscript.name}' is duplicated with different variable status")
        names.add(subscript.name)

    return tuple(tuple(info_array) for info_array in info_all_array)


def parse_variable_dims(subscripts: str, ndims: Sequence[int]) -> dict[str, int]:
    subscripts_all = parse_subscripts(subscripts)
    subscripts_unique = set([x for info_array in subscripts_all for x in info_array])

    # decide dimensions by solving linear equations
    if len(subscripts_unique) > len(subscripts_all):
        raise ValueError(f"Number of unique subscripts ({len(subscripts_unique)}) is greater than number of operands ({len(subscripts_all)}), making it impossible to assume the number of dimensions for each variable subscript")
    rhs = np.asarray(ndims, dtype=int)
    subscripts_variable_unique = [subscript for subscript in subscripts_unique if subscript.is_variable]
    mat = np.zeros((len(subscripts_unique), len(subscripts_variable_unique)), dtype=int)
    for i, info_array in enumerate(subscripts_all):
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
    subscripts_all = parse_subscripts(subscripts)
    if len(subscripts_all) != len(operands):
        raise ValueError(f"Number of subscripts ({len(subscripts_all)}) does not match number of operands ({len(operands)})")
    shapes = [operand if isinstance(operand, tuple) else operand.shape for operand in operands]
