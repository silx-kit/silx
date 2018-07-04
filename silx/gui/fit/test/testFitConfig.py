# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Basic tests for :class:`FitConfig`"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"

import unittest

from ...test.utils import TestCaseQt
from .. import FitConfig


class TestFitConfig(TestCaseQt):
    """Basic test for FitWidget"""

    def setUp(self):
        super(TestFitConfig, self).setUp()
        self.fit_config = FitConfig.getFitConfigDialog(modal=False)
        self.qWaitForWindowExposed(self.fit_config)

    def tearDown(self):
        del self.fit_config
        super(TestFitConfig, self).tearDown()

    def testShow(self):
        self.fit_config.show()
        self.fit_config.hide()

    def testAccept(self):
        self.fit_config.accept()
        self.assertTrue(self.fit_config.result())

    def testReject(self):
        self.fit_config.reject()
        self.assertFalse(self.fit_config.result())

    def testDefaultOutput(self):
        self.fit_config.accept()
        output = self.fit_config.output

        for key in ["AutoFwhm",
                    "PositiveHeightAreaFlag",
                    "QuotedPositionFlag",
                    "PositiveFwhmFlag",
                    "SameFwhmFlag",
                    "QuotedEtaFlag",
                    "NoConstraintsFlag",
                    "FwhmPoints",
                    "Sensitivity",
                    "Yscaling",
                    "ForcePeakPresence",
                    "StripBackgroundFlag",
                    "StripWidth",
                    "StripIterations",
                    "StripThreshold",
                    "SmoothingFlag"]:
            self.assertIn(key, output)

        self.assertTrue(output["AutoFwhm"])
        self.assertEqual(output["StripWidth"], 2)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestFitConfig))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
