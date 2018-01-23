# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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

import unittest

import numpy
import sys

from silx.utils import testutils
from silx.math.fit.leastsq import _logger as fitlogger


class Test_leastsq(unittest.TestCase):
    """
    Unit tests of the leastsq function.
    """

    ndims = None

    def setUp(self):
        try:
            from silx.math.fit import leastsq
            self.instance = leastsq
        except ImportError:
            self.instance = None

        def myexp(x):
            # put a (bad) filter to avoid over/underflows
            # with no python looping
            return numpy.exp(x*numpy.less(abs(x), 250)) - \
                   1.0 * numpy.greater_equal(abs(x), 250)

        self.my_exp = myexp

        def gauss(x, *params):
            params = numpy.array(params, copy=False, dtype=numpy.float)
            result = params[0] + params[1] * x
            for i in range(2, len(params), 3):
                p = params[i:(i+3)]
                dummy = 2.3548200450309493*(x - p[1])/p[2]
                result += p[0] * self.my_exp(-0.5 * dummy * dummy)
            return result

        self.gauss = gauss

        def gauss_derivative(x, params, idx):
            if idx == 0:
                return numpy.ones(len(x), numpy.float)
            if idx == 1:
                return x
            gaussian_peak = (idx - 2) // 3
            gaussian_parameter = (idx - 2) % 3
            actual_idx = 2 + 3 * gaussian_peak
            p = params[actual_idx:(actual_idx+3)]
            if gaussian_parameter == 0:
                return self.gauss(x, *[0, 0, 1.0, p[1], p[2]])
            if gaussian_parameter == 1:
                tmp = self.gauss(x, *[0, 0, p[0], p[1], p[2]])
                tmp *= 2.3548200450309493*(x - p[1])/p[2]
                return tmp * 2.3548200450309493/p[2]
            if gaussian_parameter == 2:
                tmp = self.gauss(x, *[0, 0, p[0], p[1], p[2]])
                tmp *= 2.3548200450309493*(x - p[1])/p[2]
                return tmp * 2.3548200450309493*(x - p[1])/(p[2]*p[2])

        self.gauss_derivative = gauss_derivative

    def tearDown(self):
        self.instance = None
        self.gauss = None
        self.gauss_derivative = None
        self.my_exp = None
        self.model_function = None
        self.model_derivative = None

    def testImport(self):
        self.assertTrue(self.instance is not None,
                        "Cannot import leastsq from silx.math.fit")

    def testUnconstrainedFitNoWeight(self):
        parameters_actual = [10.5, 2, 1000.0, 20., 15]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10]
        model_function = self.gauss

        fittedpar, cov = self.instance(model_function, x, y, parameters_estimate)
        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

    def testUnconstrainedFitWeight(self):
        parameters_actual = [10.5,2,1000.0,20.,15]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        sigma = numpy.sqrt(y)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10]
        model_function = self.gauss

        fittedpar, cov = self.instance(model_function, x, y,
                                       parameters_estimate,
                                       sigma=sigma)
        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

    def testDerivativeFunction(self):
        parameters_actual = [10.5, 2, 10000.0, 20., 150, 5000, 900., 300]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        delta = numpy.sqrt(numpy.finfo(numpy.float).eps)
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
            yPlus = self.gauss(x, *p)
            p[i] = parameters_actual[i] - delta_par
            yMinus = self.gauss(x, *p)
            numerical_derivative = (yPlus - yMinus) / (2 * delta_par)
            #numerical_derivative = (self.gauss(x, *p) - y) / delta_par
            p[i] = parameters_actual[i]
            derivative = self.gauss_derivative(x, p, i)
            diff = numerical_derivative - derivative
            test_condition = numpy.allclose(numerical_derivative,
                                            derivative, atol=5.0e-6)
            if not test_condition:
                msg = "Error calculating derivative of parameter %d." % i
                msg += "\n diff min = %g diff max = %g" % (diff.min(), diff.max())
                self.assertTrue(test_condition, msg)

    def testConstrainedFit(self):
        CFREE       = 0
        CPOSITIVE   = 1
        CQUOTED     = 2
        CFIXED      = 3
        CFACTOR     = 4
        CDELTA      = 5
        CSUM        = 6
        parameters_actual = [10.5, 2, 10000.0, 20., 150, 5000, 900., 300]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10, 400, 850, 200]
        model_function = self.gauss
        model_deriv = self.gauss_derivative
        constraints_all_free = [[0, 0, 0]] * len(parameters_actual)
        constraints_all_positive = [[1, 0, 0]] * len(parameters_actual)
        constraints_delta_position = [[0, 0, 0]] * len(parameters_actual)
        constraints_delta_position[6] = [CDELTA, 3, 880]
        constraints_sum_position = constraints_all_positive * 1 
        constraints_sum_position[6] = [CSUM, 3, 920]
        constraints_factor = constraints_delta_position * 1
        constraints_factor[2] = [CFACTOR, 5, 2]
        constraints_list = [None,
                            constraints_all_free,
                            constraints_all_positive,
                            constraints_delta_position,
                            constraints_sum_position]

        # for better code coverage, the warning recommending to set full_output
        # to True when using constraints should be shown at least once
        full_output = True
        for index, constraints in enumerate(constraints_list):
            if index == 2:
                full_output = None
            elif index == 3:
                full_output = 0
            for model_deriv in [None, self.gauss_derivative]:
                for sigma in [None, numpy.sqrt(y)]:                    
                    fittedpar, cov = self.instance(model_function, x, y,
                                                   parameters_estimate,
                                                   sigma=sigma,
                                                   constraints=constraints,
                                                   model_deriv=model_deriv,
                                                   full_output=full_output)[:2]
                    full_output = True

        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

    def testUnconstrainedFitAnalyticalDerivative(self):
        parameters_actual = [10.5, 2, 1000.0, 20., 15]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        sigma = numpy.sqrt(y)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10]
        model_function = self.gauss
        model_deriv = self.gauss_derivative

        fittedpar, cov = self.instance(model_function, x, y,
                                       parameters_estimate,
                                       sigma=sigma,
                                       model_deriv=model_deriv)
        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

    @testutils.test_logging(fitlogger.name, warning=2)
    def testBadlyShapedData(self):
        parameters_actual = [10.5, 2, 1000.0, 20., 15]
        x = numpy.arange(10000.).reshape(1000, 10)
        y = self.gauss(x, *parameters_actual)
        sigma = numpy.sqrt(y)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10]
        model_function = self.gauss

        for check_finite in [True, False]:
            fittedpar, cov = self.instance(model_function, x, y,
                                           parameters_estimate,
                                           sigma=sigma,
                                           check_finite=check_finite)
            test_condition = numpy.allclose(parameters_actual, fittedpar)
            if not test_condition:
                msg = "Unsuccessfull fit\n"
                for i in range(len(fittedpar)):
                    msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                          fittedpar[i])
                self.assertTrue(test_condition, msg)

    @testutils.test_logging(fitlogger.name, warning=3)
    def testDataWithNaN(self):
        parameters_actual = [10.5, 2, 1000.0, 20., 15]
        x = numpy.arange(10000.).reshape(1000, 10)
        y = self.gauss(x, *parameters_actual)
        sigma = numpy.sqrt(y)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10]
        model_function = self.gauss
        x[500] = numpy.inf
        # check default behavior
        try:
            self.instance(model_function, x, y,
                          parameters_estimate,
                          sigma=sigma)
        except ValueError:
            info = "%s" % sys.exc_info()[1]
            self.assertTrue("array must not contain inf" in info)

        # check requested behavior
        try:
            self.instance(model_function, x, y,
                          parameters_estimate,
                          sigma=sigma,
                          check_finite=True)
        except ValueError:
            info = "%s" % sys.exc_info()[1]
            self.assertTrue("array must not contain inf" in info)

        fittedpar, cov = self.instance(model_function, x, y,
                                       parameters_estimate,
                                       sigma=sigma,
                                       check_finite=False)
        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

        # testing now with ydata containing NaN
        x = numpy.arange(10000.).reshape(1000, 10)
        y[500] = numpy.nan
        fittedpar, cov = self.instance(model_function, x, y,
                                       parameters_estimate,
                                       sigma=sigma,
                                       check_finite=False)

        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

        # testing now with sigma containing NaN
        sigma[300] = numpy.nan
        fittedpar, cov = self.instance(model_function, x, y,
                                       parameters_estimate,
                                       sigma=sigma,
                                       check_finite=False)
        test_condition = numpy.allclose(parameters_actual, fittedpar)
        if not test_condition:
            msg = "Unsuccessfull fit\n"
            for i in range(len(fittedpar)):
                msg += "Expected %g obtained %g\n" % (parameters_actual[i],
                                                      fittedpar[i])
            self.assertTrue(test_condition, msg)

    def testUncertainties(self):
        """Test for validity of uncertainties in returned full-output
        dictionary. This is a non-regression test for pull request #197"""
        parameters_actual = [10.5, 2, 1000.0, 20., 15, 2001.0, 30.1, 16]
        x = numpy.arange(10000.)
        y = self.gauss(x, *parameters_actual)
        parameters_estimate = [0.0, 1.0, 900.0, 25., 10., 1500., 20., 2.0]

        # test that uncertainties are not 0.
        fittedpar, cov, infodict = self.instance(self.gauss, x, y, parameters_estimate,
                                                 full_output=True)
        uncertainties = infodict["uncertainties"]
        self.assertEqual(len(uncertainties), len(parameters_actual))
        self.assertEqual(len(uncertainties), len(fittedpar))
        for uncertainty in uncertainties:
            self.assertNotAlmostEqual(uncertainty, 0.)

        # set constraint FIXED for half the parameters.
        # This should cause leastsq to return 100% uncertainty.
        parameters_estimate = [10.6, 2.1, 1000.1, 20.1, 15.1, 2001.1, 30.2, 16.1]
        CFIXED = 3
        CFREE = 0
        constraints = []
        for i in range(len(parameters_estimate)):
            if i % 2:
                constraints.append([CFIXED, 0, 0])
            else:
                constraints.append([CFREE, 0, 0])
        fittedpar, cov, infodict = self.instance(self.gauss, x, y, parameters_estimate,
                                                 constraints=constraints,
                                                 full_output=True)
        uncertainties = infodict["uncertainties"]
        for i in range(len(parameters_estimate)):
            if i % 2:
                # test that all FIXED parameters have 100% uncertainty
                self.assertAlmostEqual(uncertainties[i],
                                       parameters_estimate[i])


test_cases = (Test_leastsq,)

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
