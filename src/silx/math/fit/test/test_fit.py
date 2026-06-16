# /*##########################################################################
# Copyright (C) 2016-2026 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
Nominal tests of the leastsq function.
"""

import logging
import numpy
import pytest

from silx.math.fit import leastsq, CDELTA, CFACTOR, CFIXED, CFREE, CPOSITIVE, CSUM


def _my_exp(x):
    # put a (bad) filter to avoid over/underflows
    # with no python looping
    with numpy.errstate(invalid="ignore"):
        return numpy.exp(x * numpy.less(abs(x), 250)) - 1.0 * numpy.greater_equal(
            abs(x), 250
        )


def _gauss(x, *params):
    params = numpy.asarray(params, dtype=numpy.float64)
    result = params[0] + params[1] * x
    for i in range(2, len(params), 3):
        p = params[i : (i + 3)]
        dummy = 2.3548200450309493 * (x - p[1]) / p[2]
        result += p[0] * _my_exp(-0.5 * dummy * dummy)
    return result


def _gauss_derivative(x, params, idx):
    if idx == 0:
        return numpy.ones(len(x), numpy.float64)
    if idx == 1:
        return x
    gaussian_peak = (idx - 2) // 3
    gaussian_parameter = (idx - 2) % 3
    actual_idx = 2 + 3 * gaussian_peak
    p = params[actual_idx : (actual_idx + 3)]
    if gaussian_parameter == 0:
        return _gauss(x, *[0, 0, 1.0, p[1], p[2]])
    if gaussian_parameter == 1:
        tmp = _gauss(x, *[0, 0, p[0], p[1], p[2]])
        tmp *= 2.3548200450309493 * (x - p[1]) / p[2]
        return tmp * 2.3548200450309493 / p[2]
    if gaussian_parameter == 2:
        tmp = _gauss(x, *[0, 0, p[0], p[1], p[2]])
        tmp *= 2.3548200450309493 * (x - p[1]) / p[2]
        return tmp * 2.3548200450309493 * (x - p[1]) / (p[2] * p[2])


def assert_fit_success(expected, actual):
    assert numpy.allclose(expected, actual), (
        f"Fit failed:\nExpected: {expected}\nActual:   {actual}"
    )


@pytest.mark.parametrize("with_sigma", [False, True])
def test_unconstrained_fit(with_sigma):
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y) if with_sigma else None
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]

    fittedpar, cov = leastsq(_gauss, x, y, parameters_estimate, sigma=sigma)
    assert_fit_success(parameters_actual, fittedpar)


def test_derivative_function():
    parameters_actual = [10.5, 2, 10000.0, 20.0, 150, 5000, 900.0, 300]
    x = numpy.arange(10000.0)

    delta = numpy.sqrt(numpy.finfo(numpy.float64).eps)
    for i in range(len(parameters_actual)):
        p = parameters_actual * 1
        if p[i] == 0:
            delta_par = delta
        else:
            delta_par = p[i] * delta
        if i > 2:
            p[0] = 0.0
            p[1] = 0.0
        p[i] += delta_par
        yPlus = _gauss(x, *p)
        p[i] = parameters_actual[i] - delta_par
        yMinus = _gauss(x, *p)
        numerical_derivative = (yPlus - yMinus) / (2 * delta_par)
        p[i] = parameters_actual[i]
        derivative = _gauss_derivative(x, p, i)
        diff = numerical_derivative - derivative
        assert numpy.allclose(numerical_derivative, derivative, atol=5.0e-6), (
            f"Error calculating derivative of parameter {i}. Diff min={diff.min():g}, max={diff.max():g}"
        )


CONSTRAINTS = {
    "none": None,
    "all_free": [[CFREE, 0, 0]] * 8,
    "all_positive": [[CPOSITIVE, 0, 0]] * 8,
    "delta_position": [[CFREE, 0, 0]] * 6 + [[CDELTA, 3, 880], [CFREE, 0, 0]],
    "sum_position": [[CPOSITIVE, 0, 0]] * 6 + [[CSUM, 3, 920], [CPOSITIVE, 0, 0]],
    "factor": [
        [CFREE, 0, 0],
        [CFREE, 0, 0],
        [CFACTOR, 5, 2],
        [CFREE, 0, 0],
        [CFREE, 0, 0],
        [CFREE, 0, 0],
        [CDELTA, 3, 880],
        [CFREE, 0, 0],
    ],
}


@pytest.mark.parametrize("constraints_name", CONSTRAINTS.keys())
@pytest.mark.parametrize("model_deriv", [None, _gauss_derivative])
@pytest.mark.parametrize("with_sigma", [False, True])
@pytest.mark.parametrize("full_output", [None, 0, True])
def test_constrained_fit(constraints_name, model_deriv, with_sigma, full_output):
    parameters_actual = [10.5, 2, 10000.0, 20.0, 150, 5000, 900.0, 300]
    x = numpy.arange(10000.0)
    y = _gauss(x, *parameters_actual)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10, 400, 850, 200]
    constraints = CONSTRAINTS[constraints_name]

    sigma = numpy.sqrt(y) if with_sigma else None
    fittedpar, cov = leastsq(
        _gauss,
        x,
        y,
        parameters_estimate,
        sigma=sigma,
        constraints=constraints,
        model_deriv=model_deriv,
        full_output=full_output,
    )[:2]
    assert_fit_success(parameters_actual, fittedpar)


def test_unconstrained_fit_analytical_derivative():
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]

    fittedpar, cov = leastsq(
        _gauss,
        x,
        y,
        parameters_estimate,
        sigma=sigma,
        model_deriv=_gauss_derivative,
    )
    assert_fit_success(parameters_actual, fittedpar)


@pytest.mark.parametrize("check_finite", [True, False])
def test_dadly_shaped_data(caplog, check_finite):
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0).reshape(1000, 10)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]

    with caplog.at_level(logging.WARNING, logger="silx.math.fit.leastsq"):
        fittedpar, cov = leastsq(
            _gauss,
            x,
            y,
            parameters_estimate,
            sigma=sigma,
            check_finite=check_finite,
        )
    assert caplog.record_tuples == [
        (
            "silx.math.fit.leastsq",
            logging.WARNING,
            "Supplied function does not return a 1D array of floats.\nFunction should be rewritten.\nTrying to reshape output.",
        )
    ]
    assert_fit_success(parameters_actual, fittedpar)


def test_xdata_non_finite_checked():
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0).reshape(1000, 10)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]
    x[500] = numpy.inf

    # check default behavior
    with pytest.raises(ValueError, match="array must not contain inf"):
        leastsq(_gauss, x, y, parameters_estimate, sigma=sigma)

    # check requested behavior
    with pytest.raises(ValueError, match="array must not contain inf"):
        leastsq(
            _gauss,
            x,
            y,
            parameters_estimate,
            sigma=sigma,
            check_finite=True,
        )


def test_xdata_non_finite_unchecked(caplog):
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0).reshape(1000, 10)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]
    x[500] = numpy.inf

    with caplog.at_level(logging.WARNING, logger="silx.math.fit.leastsq"):
        fittedpar, cov = leastsq(
            _gauss, x, y, parameters_estimate, sigma=sigma, check_finite=False
        )
    assert caplog.record_tuples == [
        (
            "silx.math.fit.leastsq",
            logging.WARNING,
            "Supplied function does not return a proper array of floats.\nFunction should be rewritten to return a 1D array of floats.\nTrying to reshape output.",
        ),
        (
            "silx.math.fit.leastsq",
            logging.WARNING,
            "Supplied function unable to handle non-finite x data\nAttempting to filter out those x data values.",
        ),
    ]
    assert_fit_success(parameters_actual, fittedpar)


def test_y_sigma_data_non_finite_unchecked(caplog):
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15]
    x = numpy.arange(10000.0).reshape(1000, 10)
    y = _gauss(x, *parameters_actual)
    sigma = numpy.sqrt(y)
    y[500] = numpy.nan

    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10]

    with caplog.at_level(logging.WARNING, logger="silx.math.fit.leastsq"):
        fittedpar, cov = leastsq(
            _gauss, x, y, parameters_estimate, sigma=sigma, check_finite=False
        )
    assert caplog.record_tuples == [
        (
            "silx.math.fit.leastsq",
            logging.WARNING,
            "Need to reshape input xdata.",
        )
    ]
    assert_fit_success(parameters_actual, fittedpar)

    # testing now with sigma containing NaN
    sigma[300] = numpy.nan
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="silx.math.fit.leastsq"):
        fittedpar, cov = leastsq(
            _gauss, x, y, parameters_estimate, sigma=sigma, check_finite=False
        )
    assert caplog.record_tuples == [
        (
            "silx.math.fit.leastsq",
            logging.WARNING,
            "Need to reshape input xdata.",
        )
    ]
    assert_fit_success(parameters_actual, fittedpar)


def test_uncertainties():
    """Test for validity of uncertainties in returned full-output
    dictionary. This is a non-regression test for pull request #197"""
    parameters_actual = [10.5, 2, 1000.0, 20.0, 15, 2001.0, 30.1, 16]
    x = numpy.arange(10000.0)
    y = _gauss(x, *parameters_actual)
    parameters_estimate = [0.0, 1.0, 900.0, 25.0, 10.0, 1500.0, 20.0, 2.0]

    # test that uncertainties are not 0.
    fittedpar, cov, infodict = leastsq(
        _gauss, x, y, parameters_estimate, full_output=True
    )
    uncertainties = infodict["uncertainties"]
    assert len(uncertainties) == len(parameters_actual)
    assert len(uncertainties) == len(fittedpar)
    for uncertainty in uncertainties:
        assert abs(uncertainty) > 1e-7

    # set constraint FIXED for half the parameters.
    # This should cause leastsq to return 100% uncertainty.
    parameters_estimate = [10.6, 2.1, 1000.1, 20.1, 15.1, 2001.1, 30.2, 16.1]
    constraints = []
    for i in range(len(parameters_estimate)):
        if i % 2:
            constraints.append([CFIXED, 0, 0])
        else:
            constraints.append([CFREE, 0, 0])
    fittedpar, cov, infodict = leastsq(
        _gauss,
        x,
        y,
        parameters_estimate,
        constraints=constraints,
        full_output=True,
    )
    uncertainties = infodict["uncertainties"]
    for i in range(len(parameters_estimate)):
        if i % 2:
            # test that all FIXED parameters have 100% uncertainty
            assert abs(uncertainties[i] - parameters_estimate[i]) <= 1e-7
