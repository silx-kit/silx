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
Tests for fitmanager module
"""

import unittest
import numpy
import sys
import os
import os.path

from silx.math.fit import fitmanager
from silx.math.fit import fitestimatefunctions
from silx.math.fit.functions import sum_gauss

from silx.testutils import temp_dir

custom_function_definition = """
import numpy

CONFIG = {'d': 1.}

def myfun(x, a, b, c):
    "Model function"
    return (a * x**2 + b * x + c) / CONFIG['d']

def myesti(x, y, bg, yscaling):
    "Initial parameters for iterative fit (a, b, c) = (1, 1, 1)"
    return (1., 1., 1.), ((0, 0, 0), (0, 0, 0), (0, 0, 0))

def myconfig(d=1.):
    "This function cam modify CONFIG"
    CONFIG["d"] = d
    return CONFIG

def myderiv(x, parameters, index):
    "Custom derivative (does not work, causes singular matrix)"
    pars_plus = parameters
    pars_plus[index] *= 1.00001

    pars_minus = parameters
    pars_minus[index] *= 0.99999

    delta_fun = myfun(x, *pars_plus) - myfun(x, *pars_minus)
    delta_par = parameters[index] * 0.0001 * 2

    return delta_fun / delta_par

THEORY = {
    'my fit theory': {
        'function': myfun,
        'parameters': ('A', 'B', 'C'),
        'estimate': myesti,
        'configure': myconfig,
        # FIXME: using myderiv causes LinAlgError: Singular matrix
        'derivative': None
    }
}


"""


class TestFitmanager(unittest.TestCase):
    """
    Unit tests of multi-peak functions.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFitManager(self):
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(1000).astype(numpy.float)

        p = [1000, 100., 250,
             255, 700., 45,
             1500, 800.5, 95]
        linear_bg = 2.65 * x + 13
        y = linear_bg + sum_gauss(x, *p)

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)
        fit.importfun(fitestimatefunctions.__file__)
        fit.settheory('gauss')
        fit.setbackground('Linear')
        fit.estimate()
        fit.startfit()

        # first 2 parameters are related to the linear background
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
    def testCustomFitFunction(self):
        # Create synthetic data with a sum of gaussian functions
        x = numpy.arange(100).astype(numpy.float)

        # a, b, c are the fit parameters
        # d is a known scaling parameter that is set using configure()
        a, b, c, d = 1.5, 2.5, 3.5, 4.5
        y = (a * x**2 + b * x + c) / d

        # Fitting
        fit = fitmanager.FitManager()
        fit.setdata(x=x, y=y)

        with temp_dir() as tmpDir:
            tmpfile = os.path.join(tmpDir, 'customfun.py')
            fd = open(tmpfile, "w")
            if sys.version < '3.0':
                fd.write(custom_function_definition)
            else:
                fd.write(custom_function_definition, 'ascii')
            fd.close()
            fit.importfun(tmpfile)
            os.unlink(tmpfile)

        fit.settheory('my fit theory')
        fit.configure(d=4.5)
        fit.estimate()
        fit.startfit()

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

test_cases = (TestFitmanager,)

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
