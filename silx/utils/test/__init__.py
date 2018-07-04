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
__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/05/2018"


import unittest
from . import test_weakref
from . import test_html
from . import test_array_like
from . import test_launcher
from . import test_deprecation
from . import test_proxy
from . import test_debug
from . import test_number


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_weakref.suite())
    test_suite.addTest(test_html.suite())
    test_suite.addTest(test_array_like.suite())
    test_suite.addTest(test_launcher.suite())
    test_suite.addTest(test_deprecation.suite())
    test_suite.addTest(test_proxy.suite())
    test_suite.addTest(test_debug.suite())
    test_suite.addTest(test_number.suite())
    return test_suite
