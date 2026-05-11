import re
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby

import attrs
import numpy as np
from array_api.latest import Array


@attrs.frozen(kw_only=True)
class SubscriptInfoFromSubcriptItem:
    name: str
    """The name of the subscript, must be of length 1 and must not be '*' or '.'"""
    is_variable: bool
    """Whether the subscript is variable (* prefix or ...)"""

    def __repr__(self) -> str:
        return f"{'*' if self.is_variable else ''}{self.name}"


@attrs.frozen(kw_only=True)
class SubscriptInfoFromSubcript:
    all: tuple[tuple[SubscriptInfoFromSubcriptItem, ...], ...]
    unique: tuple[SubscriptInfoFromSubcriptItem, ...]


@attrs.frozen(kw_only=True)
class SubscriptInfoFromShapeItemUnchecked(SubscriptInfoFromSubcriptItem):
    shape_current: tuple[int, ...]
    """The shape of the operand corresponding to this subscript"""


@attrs.frozen(kw_only=True)
class SubscriptInfoFromShapeItem(SubscriptInfoFromShapeItemUnchecked):
    shape_broadcasted: tuple[int, ...]
    """The shape after broadcasting with other subscripts with the same name"""

    def __repr__(self) -> str:
        if self.is_variable:
            return f"*{self.name}:{self.shape_broadcasted}"
        else:
            return f"{self.name}:{self.shape_broadcasted[0]}"


@attrs.frozen(kw_only=True)
class SubscriptInfoFromShapeItemUnique(SubscriptInfoFromShapeItem):
    shape_broadcasted: tuple[int, ...]
    """The shape after broadcasting with other subscripts with the same name"""

    def __repr__(self) -> str:
        if self.is_variable:
            return f"*{self.name}:{self.shape_current}≦{self.shape_broadcasted}"
        else:
            return f"{self.name}:{self.shape_current[0]}"


@attrs.frozen(kw_only=True)
class SubscriptInfoFromShape:
    all: tuple[tuple[SubscriptInfoFromShapeItem, ...], ...]
    """The subscript info grouped by order, then by operand"""
    unique: tuple[SubscriptInfoFromShapeItemUnchecked, ...]
    """The unique subscripts,
    ignoring order and shapes not broadcasted,
    sorted lexicographically by name"""


def parse_subscripts(subscripts: str, /) -> SubscriptInfoFromSubcript:
    """
    Parse subscripts str.

    Parameters
    ----------
    subscripts : str
        Subscripts separated by "," per operand.

        1. Subscripts must be of length 1
        2. Subscripts must not be "*" or ".".
        3. If start with "*", the subscript is treated as variable.
        4. "..." is replaced with "*.".

    Returns
    -------
    SubscriptInfoFromSubcript
        The parsed subscript info.

    Raises
    ------
    ValueError
        If the subscript is invalid.

    Examples
    --------
    >>> parse_subscripts("ij,*k*l,*li")
    SubscriptInfoFromSubcript(all=((i, j), (*k, *l), (*l, i)), unique=(i, j, *k, *l))

    """
    # If . other than ...
    if re.search(r"(?<!\.)\.(?!\.)", subscripts):
        raise ValueError("Invalid subscript: '.' is not allowed except for '...'")
    # Replace ... with *.
    subscripts = re.sub(r"\.\.\.", "*.", subscripts)
    subscripts = subscripts.rstrip()
    info_all: tuple[tuple[SubscriptInfoFromSubcriptItem, ...], ...] = ()
    for name_ in subscripts.split(","):
        info_array: tuple[SubscriptInfoFromSubcriptItem, ...] = ()
        is_variable = False
        for name in name_:
            if name == "*":
                if is_variable:
                    raise ValueError("Invalid subscript: '*' cannot be repeated")
                is_variable = True
            else:
                info_array += (SubscriptInfoFromSubcriptItem(name=name, is_variable=is_variable),)
                is_variable = False
        info_all += (info_array,)

    info_unique: set[SubscriptInfoFromSubcriptItem] = {
        x for info_array in info_all for x in info_array
    }
    # raise if there are subscripts with same name but different is_variable
    names = set()
    for info in info_unique:
        if info.name in names:
            raise ValueError(
                f"Subscript '{info.name}' is duplicated with different variable status"
            )
        names.add(info.name)

    return SubscriptInfoFromSubcript(
        all=info_all, unique=tuple(sorted(info_unique, key=lambda x: x.name))
    )


def parse_variable_ndim(subscripts: str, ndims: Sequence[int], /) -> dict[str, int]:
    """
    Parse variable subscript ndims by solving linear equations.

    Parameters
    ----------
    subscripts : str
        Subscripts separated by "," per operand.

        1. Subscripts must be of length 1
        2. Subscripts must not be "*" or ".".
        3. If start with "*", the subscript is treated as variable.
        4. "..." is replaced with "*.".]
    ndims : Sequence[int]
        The number of dimensions for each operand.

    Returns
    -------
    SubscriptInfoFromSubcript
        The parsed subscript info.

    Raises
    ------
    ValueError
        If the subscript is invalid.

    Examples
    --------
    >>> parse_variable_ndim("ij,*k*l,*li", (2,4,4))
    {'k': 2, 'l': 2}

    """
    info = parse_subscripts(subscripts)
    del subscripts
    if len(info.all) != len(ndims):
        raise ValueError(
            f"Number of subscripts ({len(info.all)}) does not match number of ndims ({len(ndims)})"
        )

    # decide dimensions by solving linear equations
    info_variable_unique = [subscript for subscript in info.unique if subscript.is_variable]
    if len(info_variable_unique) > len(info.all):
        raise ValueError(
            f"The number of unique subscripts ({len(info_variable_unique)}) "
            "is greater than "
            f"the number of operands ({len(info.all)}), "
            "making it impossible to assume "
            "the number of dimensions "
            "for each variable subscript"
        )
    rhs = np.asarray(ndims, dtype=int)
    del ndims
    mat = np.zeros((len(info.unique), len(info_variable_unique)), dtype=int)
    for i, info_array in enumerate(info.all):
        for subscript in info_array:
            if subscript.is_variable:
                j = info_variable_unique.index(subscript)
                mat[i, j] += 1
            else:
                rhs[i] -= 1

    # solve overdetermined linear equations using least squares method
    variable_dims, residuals, _rank, _singular_values = np.linalg.lstsq(mat, rhs, rcond=None)
    if _rank != len(info_variable_unique):
        raise ValueError(
            "Inconsistent ndims: there are not enough informations to determine the number of dimensions for variable subscripts"
        )
    if residuals.size > 0 and not np.isclose(residuals[0], 0):
        raise ValueError(
            "Inconsistent ndims: there are no solution to determine the number of dimensions for variable subscripts"
        )
    variable_dims = np.round(variable_dims).astype(int)
    if np.any(variable_dims < 0):
        raise ValueError(
            "Inconsistent ndims: number of dimensions for variable subscripts cannot be negative"
        )
    return {subscript.name: variable_dims[j] for j, subscript in enumerate(info_variable_unique)}


def parse_shapes(
    subscripts: str, /, *operands: Array | tuple[int, ...]
) -> tuple[tuple[SubscriptInfoFromShapeItemUnchecked, ...], ...]:
    info = parse_subscripts(subscripts)
    if len(info.all) != len(operands):
        raise ValueError(
            f"Number of subscripts ({len(info.all)}) "
            f"does not match number of operands ({len(operands)})"
        )
    shapes: tuple[tuple[int, ...], ...] = ()
    for operand in operands:
        shape: tuple[int, ...]  # type: ignore
        if isinstance(operand, tuple):
            shape = operand
        elif hasattr(operand, "shape"):
            if None in operand.shape:
                raise ValueError("Operand shape cannot contain None")
            shape = operand.shape
        else:
            raise TypeError(
                f"Invalid operand: expected an array or a shape tuple, but got {type(operand)}"
            )
        shapes += (shape,)  # type: ignore
    del operands
    ndims = [len(shape) for shape in shapes]
    name_to_ndim = defaultdict(lambda: 1, parse_variable_ndim(subscripts, ndims))
    del subscripts, ndims
    info_all: tuple[tuple[SubscriptInfoFromShapeItemUnchecked, ...], ...] = ()
    for info_array, shape in zip(info.all, shapes, strict=False):
        info_array_new = ()
        for item in info_array:
            info_array_new += (
                SubscriptInfoFromShapeItemUnchecked(
                    name=item.name,
                    is_variable=item.is_variable,
                    shape_current=shape[name_to_ndim[item.name] :],
                ),
            )
        info_all += (info_array_new,)
    return info_all


def check_shapes(subscripts: str, /, *operands: Array | tuple[int, ...]) -> SubscriptInfoFromShape:
    info_all = parse_shapes(subscripts, *operands)
    info_flatten_keyed = [(i, item) for i, info_array in enumerate(info_all) for item in info_array]
    errors = []
    for key, group in groupby(info_flatten_keyed, key=lambda x: x[1].name):
        group_list = list(group)
        shapes = [item.shape_current for _, item in group_list]
        try:
            shape_broadcasted = np.broadcast_shapes(*shapes)
        except ValueError:
            inner_text = "".join([f"{shape} ({i})" for i, shape in group_list])
            errors.append(ValueError(f"Key {key}: shapes {inner_text} are not broadcastable"))
    if errors:
        raise ExceptionGroup("Shape check failed", errors)

    info_all_new = ()
    for info_array in info_all:
        info_array_new = ()
        for item in info_array:
            shape_broadcasted = np.broadcast_shapes(item.shape_current, ())
            info_array_new += (
                SubscriptInfoFromShapeItem(
                    name=item.name,
                    is_variable=item.is_variable,
                    shape_current=item.shape_current,
                    shape_broadcasted=shape_broadcasted,
                ),
            )
        info_all_new += (info_array_new,)
    info_unique = {
        SubscriptInfoFromShapeItemUnchecked(
            name=item.name,
            is_variable=item.is_variable,
            shape_current=item.shape_current,
        )
        for info_array in info_all_new
        for item in info_array
    }
    return SubscriptInfoFromShape(
        all=info_all_new, unique=tuple(sorted(info_unique, key=lambda x: x.name))
    )
