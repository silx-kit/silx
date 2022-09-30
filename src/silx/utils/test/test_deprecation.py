# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Tests for html module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest
from .. import deprecation
from silx.utils import testutils


class TestDeprecation(unittest.TestCase):
    """Tests for deprecation module."""

    @deprecation.deprecated
    def deprecatedWithoutParam(self):
        pass

    @deprecation.deprecated(reason="r", replacement="r", since_version="v")
    def deprecatedWithParams(self):
        pass

    @deprecation.deprecated(reason="r", replacement="r", since_version="v", only_once=True)
    def deprecatedOnlyOnce(self):
        pass

    @deprecation.deprecated(reason="r", replacement="r", since_version="v", only_once=False)
    def deprecatedEveryTime(self):
        pass

    @testutils.validate_logging(deprecation.depreclog.name, warning=1)
    def testAnnotationWithoutParam(self):
        self.deprecatedWithoutParam()

    @testutils.validate_logging(deprecation.depreclog.name, warning=1)
    def testAnnotationWithParams(self):
        self.deprecatedWithParams()

    @testutils.validate_logging(deprecation.depreclog.name, warning=3)
    def testLoggedEveryTime(self):
        """Logged everytime cause it is 3 different locations"""
        self.deprecatedOnlyOnce()
        self.deprecatedOnlyOnce()
        self.deprecatedOnlyOnce()

    @testutils.validate_logging(deprecation.depreclog.name, warning=1)
    def testLoggedSingleTime(self):
        def log():
            self.deprecatedOnlyOnce()
        log()
        log()
        log()

    @testutils.validate_logging(deprecation.depreclog.name, warning=3)
    def testLoggedEveryTime2(self):
        self.deprecatedEveryTime()
        self.deprecatedEveryTime()
        self.deprecatedEveryTime()

    @testutils.validate_logging(deprecation.depreclog.name, warning=1)
    def testWarning(self):
        deprecation.deprecated_warning(type_="t", name="n")

    def testBacktrace(self):
        loggingValidator = testutils.LoggingValidator(deprecation.depreclog.name)
        with loggingValidator:
            self.deprecatedEveryTime()
        message = loggingValidator.records[0].getMessage()
        filename = __file__.replace(".pyc", ".py")
        self.assertTrue(filename in message)
        self.assertTrue("testBacktrace" in message)
