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
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "13/10/2016"

import numpy
import unittest
from .. import ArrayTableWidget
from ...testutils import TestCaseQt


class TestArrayWidget(TestCaseQt):
    """Basic test for ArrayTableWidget"""
    def setUp(self):
        super(TestArrayWidget, self).setUp()
        self.aw = ArrayTableWidget.ArrayTableWidget()

    def tearDown(self):
        del self.aw
        super(TestArrayWidget, self).tearDown()

    def testShow(self):
        self.aw.show()
        self.qWaitForWindowExposed(self.aw)

    def testSetData(self):
        a = numpy.reshape(numpy.linspace(0.213, 1.234, 10000),
                          (10, 10, 10, 10))
        self.aw.setArrayData(a)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestArrayWidget))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
