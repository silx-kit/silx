# coding: utf-8
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

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/08/2019"


import unittest
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt, SignalListener

from silx.gui.utils import blockSignals


class TestBlockSignals(TestCaseQt):
    """Test blockSignals context manager"""

    def _test(self, *objs):
        """Test for provided objects"""
        listener = SignalListener()
        for obj in objs:
            obj.objectNameChanged.connect(listener)
            obj.setObjectName("received")

        with blockSignals(*objs):
            for obj in objs:
                obj.setObjectName("silent")

        self.assertEqual(listener.arguments(), [("received",)] * len(objs))

    @unittest.skipUnless(qt.BINDING in ('PyQt5', 'PySide2'), 'Qt5 only test')
    def testManyObjects(self):
        """Test blockSignals with 2 QObjects"""
        self._test(qt.QObject(), qt.QObject())

    @unittest.skipUnless(qt.BINDING in ('PyQt5', 'PySide2'), 'Qt5 only test')
    def testOneObject(self):
        """Test blockSignals context manager with a single QObject"""
        self._test(qt.QObject())


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestBlockSignals))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
