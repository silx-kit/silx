import math
import numpy

from numpy.testing import assert_allclose, assert_array_equal
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


def test_apply_linear_array():
    array = numpy.array([0.0, -5.0, 1e34])
    result = axis_scale.apply("linear", array)
    assert_array_equal(result, array)


def test_apply_log_array():
    array = numpy.array([1.0, 100.0, 0.0, -1.0])
    result = axis_scale.apply("log", array)
    assert isinstance(result, numpy.ndarray)
    assert_allclose(result, [0.0, 2.0, numpy.nan, numpy.nan])


def test_apply_asinh_array():
    array = numpy.array([0.0, 1.0, -1.0])
    result = axis_scale.apply("asinh", array)
    assert isinstance(result, numpy.ndarray)
    assert_allclose(result, [0.0, math.asinh(1.0), math.asinh(-1.0)])


def test_revert_linear():
    assert axis_scale.revert("linear", 0.0) == approx(0.0)
    assert axis_scale.revert("linear", -5.0) == approx(-5.0)


def test_revert_log():
    assert axis_scale.revert("log", 0.0) == approx(1.0)
    assert axis_scale.revert("log", 2.0) == approx(100.0)
    assert axis_scale.revert("log", 310) == float("inf")


def test_revert_asinh():
    assert axis_scale.revert("asinh", 0.0) == approx(0.0)
    assert axis_scale.revert("asinh", 1.0) == approx(math.sinh(1.0))
    assert axis_scale.revert("asinh", -1.0) == approx(math.sinh(-1.0))
    assert axis_scale.revert("asinh", 750) == float("inf")


def test_revert_linear_array():
    array = numpy.array([0.0, -5.0])
    result = axis_scale.revert("linear", array)
    assert_array_equal(result, array)


def test_revert_log_array():
    array = numpy.array([0.0, 2.0, 310.0])
    result = axis_scale.revert("log", array)
    assert isinstance(result, numpy.ndarray)
    assert_allclose(result, [1.0, 100.0, numpy.inf])


def test_revert_asinh_array():
    array = numpy.array([0.0, 1.0, -1.0, 750.0])
    result = axis_scale.revert("asinh", array)
    assert isinstance(result, numpy.ndarray)
    assert_allclose(result, [0.0, math.sinh(1.0), math.sinh(-1.0), numpy.inf])


def test_revert_nan():
    assert math.isnan(axis_scale.revert("linear", float("nan")))
    assert math.isnan(axis_scale.revert("log", float("nan")))
    assert math.isnan(axis_scale.revert("asinh", float("nan")))


def test_revert_nan_array():
    array = numpy.array([float("nan")])
    assert math.isnan(axis_scale.revert("linear", array)[0])
    assert math.isnan(axis_scale.revert("log", array)[0])
    assert math.isnan(axis_scale.revert("asinh", array)[0])
