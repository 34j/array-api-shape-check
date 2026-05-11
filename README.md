# array-api-shape-check

<p align="center">
  <a href="https://github.com/34j/array-api-shape-check/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/array-api-shape-check/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://array-api-shape-check.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/array-api-shape-check.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/array-api-shape-check">
    <img src="https://img.shields.io/codecov/c/github/34j/array-api-shape-check.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/array-api-shape-check/">
    <img src="https://img.shields.io/pypi/v/array-api-shape-check.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/array-api-shape-check.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/array-api-shape-check.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://array-api-shape-check.readthedocs.io" target="_blank">https://array-api-shape-check.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/array-api-shape-check" target="_blank">https://github.com/34j/array-api-shape-check </a>

---

Check shapes of input arrays easily

## Installation

Install this via pip (or your favourite package manager):

`pip install array-api-shape-check`

## Usage

```python
>>> from array_api_shape_check import check_shapes
>>> info = check_shapes("ij,*k*l,*li", (1, 4), (5, 6, 7), (1, 7, 3))
>>> info.all
((i:1->3, j:4), (*k:(5,), *l:(6, 7)), (*l:(1, 7)->(6, 7), i:3))
>>> info.unique
(i:3, j:4, *k:(5,), *l:(6, 7))
```

Diving into the details of the first item:

```python
>>> item = info.all[0][0]
>>> item.name # the name of the subscript
'i'
>>> item.is_variable # whether the subscript is variable (starts with "*")
False
>>> item.shape_current # the current shape of the subscript
(1,)
>>> item.shape_broadcasted # the broadcasted shape of the subscript
(3,)
```

Not enough information to determine variable subscript ndims:

```python
>>> import pytest
>>> from array_api_shape_check import InconsistentNdimErrorMultipleSolutions, InconsistentNdimErrorNoSolutions, InconsistentShapeError
>>> with pytest.raises(InconsistentNdimErrorMultipleSolutions, match="number of variables"):
...     check_shapes(
...         "*i*j",
...         (
...             1,
...             1,
...         ),
...     )
>>> with pytest.raises(InconsistentNdimErrorMultipleSolutions, match="rank"):
...     check_shapes("*i*j,*i*j", (1, 1), (1, 1))
```

No solution to determine variable subscript ndims:

```python
>>> with pytest.raises(InconsistentNdimErrorNoSolutions, match="residuals"):
...     check_shapes("*i,*i", (1, 1), (1, 1, 1))
>>> with pytest.raises(InconsistentNdimErrorNoSolutions, match="negative"):
...     check_shapes("*ij", ())
```

Does not match:

```python
>>> with pytest.raises(InconsistentShapeError):
...     check_shapes("ij,*k*l,*li", (3, 4), (5, 6), (1, 7, 3))
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
