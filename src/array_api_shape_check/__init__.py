__version__ = "0.1.3"
from ._main import (
    InconsistentNdimError,
    InconsistentNdimErrorMultipleSolutions,
    InconsistentNdimErrorNoSolutions,
    InconsistentShapeError,
    SubscriptInfoFromShape,
    SubscriptInfoFromShapeItem,
    SubscriptInfoFromShapeItemUnchecked,
    SubscriptInfoFromShapeItemUnique,
    SubscriptInfoFromSubcript,
    SubscriptInfoFromSubcriptItem,
    _parse_shapes,
    check_shapes,
    parse_subscripts,
    parse_variable_ndim,
)

__all__ = [
    "InconsistentNdimError",
    "InconsistentNdimErrorMultipleSolutions",
    "InconsistentNdimErrorNoSolutions",
    "InconsistentShapeError",
    "SubscriptInfoFromShape",
    "SubscriptInfoFromShapeItem",
    "SubscriptInfoFromShapeItemUnchecked",
    "SubscriptInfoFromShapeItemUnique",
    "SubscriptInfoFromSubcript",
    "SubscriptInfoFromSubcriptItem",
    "_parse_shapes",
    "check_shapes",
    "parse_subscripts",
    "parse_variable_ndim",
]
