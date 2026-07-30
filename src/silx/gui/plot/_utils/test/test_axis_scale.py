import math
import pytest

from pytest import approx

from silx.gui.plot._utils import axis_scale


def test_is_valid_linear():
    assert axis_scale.isValid("linear", 0.0)
    assert axis_scale.isValid("linear", 1e34)
    assert axis_scale.isValid("linear", -1e34)
    assert not axis_scale.isValid("linear", float("nan"))
    assert not axis_scale.isValid("linear", float("inf"))
    assert not axis_scale.isValid("linear", float("-inf"))


def test_is_valid_log():
    assert axis_scale.isValid("log", 1.0)
    assert axis_scale.isValid("log", 1e34)
    assert not axis_scale.isValid("log", -1)
    assert not axis_scale.isValid("log", float("nan"))
    assert not axis_scale.isValid("log", float("inf"))


def test_is_valid_asinh():
    assert axis_scale.isValid("asinh", 0.0)
    assert axis_scale.isValid("asinh", 1e34)
    assert axis_scale.isValid("asinh", -1e34)
    assert not axis_scale.isValid("asinh", float("nan"))
    assert not axis_scale.isValid("asinh", float("inf"))


def test_apply_linear():
    assert axis_scale.apply("linear", 0.0) == approx(0.0)
    assert axis_scale.apply("linear", -5.0) == approx(-5.0)


def test_apply_log():
    assert axis_scale.apply("log", 1.0) == approx(0.0)
    assert axis_scale.apply("log", 100.0) == approx(2.0)
    assert math.isnan(axis_scale.apply("log", 0.0))
    assert math.isnan(axis_scale.apply("log", -1.0))


def test_apply_asinh():
    assert axis_scale.apply("asinh", 0.0) == approx(0.0)
    assert axis_scale.apply("asinh", 1.0) == approx(math.asinh(1.0))
    assert axis_scale.apply("asinh", -1.0) == approx(math.asinh(-1.0))


def test_revert_linear():
    assert axis_scale.revert("linear", 0.0) == approx(0.0)
    assert axis_scale.revert("linear", -5.0) == approx(-5.0)


def test_revert_log():
    assert axis_scale.revert("log", 0.0) == approx(1.0)
    assert axis_scale.revert("log", 2.0) == approx(100.0)
    with pytest.raises(OverflowError):
        axis_scale.revert("log", 310)


def test_revert_asinh():
    assert axis_scale.revert("asinh", 0.0) == approx(0.0)
    assert axis_scale.revert("asinh", 1.0) == approx(math.sinh(1.0))
    assert axis_scale.revert("asinh", -1.0) == approx(math.sinh(-1.0))
    with pytest.raises(OverflowError):
        axis_scale.revert("asinh", 750)


def test_revert_nan():
    assert math.isnan(axis_scale.revert("linear", float("nan")))
    assert math.isnan(axis_scale.revert("log", float("nan")))
    assert math.isnan(axis_scale.revert("asinh", float("nan")))
