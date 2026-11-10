import pytest

from array_api_shape_check import InconsistentShapeError, check_shapes
from array_api_shape_check._main import _get_operand_names  # noqa: PLC2701


def test_get_operand_names():
    def inner(a, b) -> list[str | None]:
        return _get_operand_names(a, b)

    c = 1
    assert inner(c, 4 * 4)[0] == "c"
    # assert inner(c, 4 * 4)[1] is not None
    # assert "@py_assert" in repr(inner(c, 4 * 4)[1])


@pytest.mark.parametrize(
    ("subscripts", "shapes", "key"),
    [
        ("i,i", ((3,), (4,)), "i"),
        ("ij,ji", ((3, 4), (4, 9)), "i"),
        ("*BM,*BN", ((2, 3), (4, 3)), "B"),
    ],
)
def test_incompatible_shapes(
    subscripts: str, shapes: tuple[tuple[int, ...], ...], key: str
) -> None:
    with pytest.raises(InconsistentShapeError, match=f"Key {key}") as exc:
        check_shapes(subscripts, *shapes, names="first,second")
    assert exc.value.subscript_info.unique[key].shape_broadcasted is None


def test_broadcasts_nonconsecutive_subscripts() -> None:
    info = check_shapes("ij,ji", (1, 4), (4, 9), names="first,second")
    assert info.unique["i"].shape_broadcasted == (9,)
    assert info.unique["j"].shape_broadcasted == (4,)


def test_reports_all_inconsistent_subscripts() -> None:
    with pytest.raises(InconsistentShapeError) as exc:
        check_shapes("ij,ij", (2, 3), (4, 5), names="first,second")
    assert "Key i" in str(exc.value)
    assert "Key j" in str(exc.value)
