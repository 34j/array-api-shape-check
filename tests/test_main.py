from array_api_shape_check._main import _get_operand_names


def test_get_operand_names():
    def inner(a, b) -> list[str | None]:
        return _get_operand_names(a, b)

    c = 1
    assert inner(c, 4 * 4)[0] == "c"
    # assert inner(c, 4 * 4)[1] is not None
    # assert "@py_assert" in repr(inner(c, 4 * 4)[1])
