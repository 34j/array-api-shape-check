from array_api_shape_check.main import add


def test_add():
    """Adding two number works as expected."""
    assert add(1, 1) == 2
