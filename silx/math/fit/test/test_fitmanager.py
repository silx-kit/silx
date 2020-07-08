# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2020 European Synchrotron Radiation Facility
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
Tests for fitmanager module
"""

import unittest
import numpy
import os.path

from silx.math.fit import fitmanager
from silx.math.fit import fittheories
from silx.math.fit import bgtheories
from silx.math.fit.fittheory import FitTheory
from silx.math.fit.functions import sum_gauss, sum_stepdown, sum_stepup

from silx.utils.testutils import ParametricTestCase
from silx.test.utils import temp_dir

custom_function_definition = """
import copy
from silx.math.fit.fittheory import FitTheory

CONFIG = {'d': 1.}

def myfun(x, a, b, c):
    "Model function"
    return (a * x**2 + b * x + c) / CONFIG['d']

def myesti(x, y):
    "Initial parameters for iterative fit (a, b, c) = (1, 1, 1)"
    return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

def myconfig(d=1., **kw):
    "This function can modify CONFIG"
    CONFIG["d"] = d
    return CONFIG

def myderiv(x, parameters, index):
    "Custom derivative (does not work, causes singular matrix)"
    pars_plus = copy.copy(parameters)
    pars_plus[index] *= 1.0001

    pars_minus = parameters
    pars_minus[index] *= copy.copy(0.9999)

    delta_fun = myfun(x, *pars_plus) - myfun(x, *pars_minus)
    delta_par = parameters[index] * 0.0001 * 2

    return delta_fun / delta_par

THEORY = {
    'my fit theory':
        FitTheory(function=myfun,
                  parameters=('A', 'B', 'C'),
                  estimate=myesti,
                  configure=myconfig,
                  derivative=myderiv)
}

"""

old_custom_function_definition = """
CONFIG = {'d': 1.0}

def myfun(x, a, b, c):
    "Model function"
    return (a * x**2 + b * x + c) / CONFIG['d']

def myesti(x, y, bg, xscalinq, yscaling):
    "Initial parameters for iterative fit (a, b, c) = (1, 1, 1)"
    return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

def myconfig(**kw):
    "Update or complete CONFIG dictionary"
    for key in kw:
        CONFIG[key] = kw[key]
    return CONFIG

THEORY = ['my fit theory']
PARAMETERS = [('A', 'B', 'C')]
FUNCTION = [myfun]
ESTIMATE = [myesti]
CONFIGURE = [myconfig]

"""


def _order_of_magnitude(x):
    return numpy.log10(x).round()


class TestFitmanager(ParametricTestCase):
    """
    Unit tests of multi-peak functions.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFitManager(self):
        """Test fit manager on synthetic data using a gaussian function
        and a linear background"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(1000).astype(numpy.float)

        p = [1000, 100., 250,
             255, 650., 45,
             1500, 800.5, 95]
        linear_bg = 2.65 * x + 13
        y = linear_bg + sum_gauss(x, *p)

        y_with_nans = numpy.array(y)
        y_with_nans[::10] = numpy.nan

        x_with_nans = numpy.array(x)
        x_with_nans[5::15] = numpy.nan

        tests = {
            'all finite': (x, y),
            'y with NaNs': (x, y_with_nans),
            'x with NaNs': (x_with_nans, y),
            }

        for name, (xdata, ydata) in tests.items():
            with self.subTest(name=name):
                # Fitting
                fit = fitmanager.FitManager()
                fit.setdata(x=xdata, y=ydata)
                fit.loadtheories(fittheories)
                # Use one of the default fit functions
                fit.settheory('Gaussians')
                fit.setbackground('Linear')
                fit.estimate()
                fit.runfit()

                # fit.fit_results[]

                # first 2 parameters are related to the linear background
                self.assertEqual(fit.fit_results[0]["name"], "Constant")
                self.assertAlmostEqual(fit.fit_results[0]["fitresult"], 13)
                self.assertEqual(fit.fit_results[1]["name"], "Slope")
                self.assertAlmostEqual(fit.fit_results[1]["fitresult"], 2.65)

                for i, param in enumerate(fit.fit_results[2:]):
                    param_number = i // 3 + 1
                    if i % 3 == 0:
                        self.assertEqual(param["name"],
                                         "Height%d" % param_number)
                    elif i % 3 == 1:
                        self.assertEqual(param["name"],
                                         "Position%d" % param_number)
                    elif i % 3 == 2:
                        self.assertEqual(param["name"],
                                         "FWHM%d" % param_number)

                    self.assertAlmostEqual(param["fitresult"],
                                           p[i])
                    self.assertAlmostEqual(_order_of_magnitude(param["estimation"]),
                                           _order_of_magnitude(p[i]))

    def testLoadCustomFitFunction(self):
        """Test FitManager using a custom fit function defined in an external
        file and imported with FitManager.loadtheories"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = 1.5, 2.5, 3.5, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Create a temporary function definition file, and import it
        with temp_dir() as tmpDir:
            tmpfile = os.path.join(tmpDir, 'customfun.py')
            # custom_function_definition
            fd = open(tmpfile, "w")
            fd.write(custom_function_definition)
            fd.close()
            fit.loadtheories(tmpfile)
            tmpfile_pyc = os.path.join(tmpDir, 'customfun.pyc')
            if os.path.exists(tmpfile_pyc):
                os.unlink(tmpfile_pyc)
            os.unlink(tmpfile)

        fit.settheory('my fit theory')
        # Test configure
        fit.configure(d=4.5)
        fit.estimate()
        fit.runfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               1.5)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        self.assertAlmostEqual(fit.fit_results[1]["fitresult"],
                               2.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(fit.fit_results[2]["fitresult"],
                               3.5)

    def testLoadOldCustomFitFunction(self):
        """Test FitManager using a custom fit function defined in an external
        file and imported with FitManager.loadtheories (legacy PyMca format)"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = 1.5, 2.5, 3.5, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Create a temporary function definition file, and import it
        with temp_dir() as tmpDir:
            tmpfile = os.path.join(tmpDir, 'oldcustomfun.py')
            # custom_function_definition
            fd = open(tmpfile, "w")
            fd.write(old_custom_function_definition)
            fd.close()
            fit.loadtheories(tmpfile)
            tmpfile_pyc = os.path.join(tmpDir, 'oldcustomfun.pyc')
            if os.path.exists(tmpfile_pyc):
                os.unlink(tmpfile_pyc)
            os.unlink(tmpfile)

        fit.settheory('my fit theory')
        fit.configure(d=4.5)
        fit.estimate()
        fit.runfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               1.5)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        self.assertAlmostEqual(fit.fit_results[1]["fitresult"],
                               2.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(fit.fit_results[2]["fitresult"],
                               3.5)

    def testAddTheory(self, estimate=True):
        """Test FitManager using a custom fit function imported with
        FitManager.addtheory"""
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = -3.14, 1234.5, 10000, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        # Define and add the fit theory
        CONFIG = {'d': 1.}

        def myfun(x_, a_, b_, c_):
            """"Model function"""
            return (a_ * x_**2 + b_ * x_ + c_) / CONFIG['d']

        def myesti(x_, y_):
            """"Initial parameters for iterative fit:
                 (a, b, c) = (1, 1, 1)
            Constraints all set to 0 (FREE)"""
            return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

        def myconfig(d_=1., **kw):
            """This function can modify CONFIG"""
            CONFIG["d"] = d_
            return CONFIG

        def myderiv(x_, parameters, index):
            """Custom derivative"""
            pars_plus = numpy.array(parameters, copy=True)
            pars_plus[index] *= 1.001

            pars_minus = numpy.array(parameters, copy=True)
            pars_minus[index] *= 0.999

            delta_fun = myfun(x_, *pars_plus) - myfun(x_, *pars_minus)
            delta_par = parameters[index] * 0.001 * 2

            return delta_fun / delta_par

        fit.addtheory("polynomial",
                      FitTheory(function=myfun,
                                parameters=["A", "B", "C"],
                                estimate=myesti if estimate else None,
                                configure=myconfig,
                                derivative=myderiv))

        fit.settheory('polynomial')
        fit.configure(d_=4.5)
        fit.estimate()
        params1, sigmas, infodict = fit.runfit()

        self.assertEqual(fit.fit_results[0]["name"],
                         "A1")
        self.assertAlmostEqual(fit.fit_results[0]["fitresult"],
                               -3.14)
        self.assertEqual(fit.fit_results[1]["name"],
                         "B1")
        # params1[1] is the same as fit.fit_results[1]["fitresult"]
        self.assertAlmostEqual(params1[1],
                               1234.5)
        self.assertEqual(fit.fit_results[2]["name"],
                         "C1")
        self.assertAlmostEqual(params1[2],
                               10000)

        # change configuration scaling factor and check that the fit returns
        # different values
        fit.configure(d_=5.)
        fit.estimate()
        params2, sigmas, infodict = fit.runfit()
        for p1, p2 in zip(params1, params2):
            self.assertFalse(numpy.array_equal(p1, p2),
                             "Fit parameters are equal even though the " +
                             "configuration has been changed")

    def testNoEstimate(self):
        """Ensure that the in the absence of the estimation function,
        the default estimation function :meth:`FitTheory.default_estimate`
        is used."""
        self.testAddTheory(estimate=False)

    def testStep(self):
        """Test fit manager on a step function with a more complex estimate
        function than the gaussian (convolution filter)"""
        for theory_name, theory_fun in (('Step Down', sum_stepdown),
                                        ('Step Up', sum_stepup)):
            # Create synthetic data with a sum of gaussian functions
            x = numpy.arange(1000).astype(numpy.float)

            # ('Height', 'Position', 'FWHM')
            p = [1000, 439, 250]

            constantbg = 13
            y = theory_fun(x, *p) + constantbg

            # Fitting
            fit = fitmanager.FitManager()
            fit.setdata(x=x, y=y)
            fit.loadtheories(fittheories)
            fit.settheory(theory_name)
            fit.setbackground('Constant')

            fit.estimate()

            params, sigmas, infodict = fit.runfit()

            # first parameter is the constant background
            self.assertAlmostEqual(params[0], 13, places=5)
            for i, param in enumerate(params[1:]):
                self.assertAlmostEqual(param, p[i], places=5)
                self.assertAlmostEqual(_order_of_magnitude(fit.fit_results[i+1]["estimation"]),
                                       _order_of_magnitude(p[i]))


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


class TestPolynomials(unittest.TestCase):
    """Test polynomial fit theories and fit background"""
    def setUp(self):
        self.x = numpy.arange(100).astype(numpy.float)

    def testQuadraticBg(self):
        gaussian_params = [100, 45, 8]
        poly_params = [0.05, -2, 3]
        p = numpy.poly1d(poly_params)

        y = p(self.x) + sum_gauss(self.x, *gaussian_params)

        fm = fitmanager.FitManager(self.x, y)
        fm.loadbgtheories(bgtheories)
        fm.loadtheories(fittheories)
        fm.settheory("Gaussians")
        fm.setbackground("Degree 2 Polynomial")
        esti_params = fm.estimate()
        fit_params = fm.runfit()[0]

        for p, pfit in zip(poly_params + gaussian_params, fit_params):
            self.assertAlmostEqual(p,
                                   pfit)

    def testCubicBg(self):
        gaussian_params = [1000, 45, 8]
        poly_params = [0.0005, -0.05, 3, -4]
        p = numpy.poly1d(poly_params)

        y = p(self.x) + sum_gauss(self.x, *gaussian_params)

        fm = fitmanager.FitManager(self.x, y)
        fm.loadtheories(fittheories)
        fm.settheory("Gaussians")
        fm.setbackground("Degree 3 Polynomial")
        esti_params = fm.estimate()
        fit_params = fm.runfit()[0]

        for p, pfit in zip(poly_params + gaussian_params, fit_params):
            self.assertAlmostEqual(p,
                                   pfit)

    def testQuarticcBg(self):
        gaussian_params = [10000, 69, 25]
        poly_params = [5e-10, 0.0005, 0.005, 2, 4]
        p = numpy.poly1d(poly_params)

        y = p(self.x) + sum_gauss(self.x, *gaussian_params)

        fm = fitmanager.FitManager(self.x, y)
        fm.loadtheories(fittheories)
        fm.settheory("Gaussians")
        fm.setbackground("Degree 4 Polynomial")
        esti_params = fm.estimate()
        fit_params = fm.runfit()[0]

        for p, pfit in zip(poly_params + gaussian_params, fit_params):
            self.assertAlmostEqual(p,
                                   pfit,
                                   places=5)

    def _testPoly(self, poly_params, theory, places=5):
        p = numpy.poly1d(poly_params)

        y = p(self.x)

        fm = fitmanager.FitManager(self.x, y)
        fm.loadbgtheories(bgtheories)
        fm.loadtheories(fittheories)
        fm.settheory(theory)
        esti_params = fm.estimate()
        fit_params = fm.runfit()[0]

        for p, pfit in zip(poly_params, fit_params):
            self.assertAlmostEqual(p, pfit, places=places)

    def testQuadratic(self):
        self._testPoly([0.05, -2, 3],
                       "Degree 2 Polynomial")

    def testCubic(self):
        self._testPoly([0.0005, -0.05, 3, -4],
                       "Degree 3 Polynomial")

    def testQuartic(self):
        self._testPoly([1, -2, 3, -4, -5],
                       "Degree 4 Polynomial")

    def testQuintic(self):
        self._testPoly([1, -2, 3, -4, -5, 6],
                       "Degree 5 Polynomial",
                       places=4)


test_cases = (TestFitmanager, TestPolynomials)


def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
