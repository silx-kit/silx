# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Test of functions available in silx.gui.utils module."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/08/2019"


import unittest
from silx.gui import qt
from silx.gui import utils
from silx.gui.utils.testutils import TestCaseQt


class TestQEventName(TestCaseQt):
    """Test QEvent names"""

    def testNoneType(self):
        result = utils.getQEventName(0)
        self.assertEqual(result, "None")

    def testNoneEvent(self):
        event = qt.QEvent(qt.QEvent.Type(0))
        result = utils.getQEventName(event)
        self.assertEqual(result, "None")

    def testUserType(self):
        result = utils.getQEventName(1050)
        self.assertIn("User", result)
        self.assertIn("1050", result)

    def testQtUndefinedType(self):
        result = utils.getQEventName(900)
        self.assertIn("Unknown", result)
        self.assertIn("900", result)

    def testUndefinedType(self):
        result = utils.getQEventName(70000)
        self.assertIn("Unknown", result)
        self.assertIn("70000", result)
