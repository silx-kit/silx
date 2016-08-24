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
"""Basic test of Qt icons module."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/08/2016"


import unittest
from silx.gui import qt
from silx.gui import testutils
from silx.gui import icons


class TestIcons(testutils.TestCaseQt):
    """Test to check that icons module."""

    def testExistingIcon(self):
        icon = icons.getQIcon("crop")
        self.assertIsNotNone(icon)

    def testUnexistingIcon(self):
        self.assertRaises(ValueError, icons.getQIcon, "not-exists")

    def testExistingQPixmap(self):
        icon = icons.getQPixmap("crop")
        self.assertIsNotNone(icon)

    def testUnexistingQPixmap(self):
        self.assertRaises(ValueError, icons.getQPixmap, "not-exists")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestIcons))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
