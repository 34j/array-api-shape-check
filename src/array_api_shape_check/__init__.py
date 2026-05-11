__version__ = "0.0.0"
from ._main import (
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
