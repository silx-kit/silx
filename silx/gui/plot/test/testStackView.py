# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Basic tests for StackView"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "19/12/2016"


import unittest
import numpy

from silx.gui.test.utils import TestCaseQt

from silx.gui import qt
from silx.gui.plot import StackView
from silx.gui.plot.StackView import StackViewMainWindow

from silx.utils.array_like import ListOfImages


# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


class TestStackView(TestCaseQt):
    """Base class for tests of StackView."""

    def setUp(self):
        super(TestStackView, self).setUp()
        self.stackview = StackView()
        self.stackview.show()
        self.qWaitForWindowExposed(self.stackview)
        self.mystack = numpy.fromfunction(
            lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
            (100, 200, 300)
        )

    def tearDown(self):
        self.stackview.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.stackview.close()
        del self.stackview
        super(TestStackView, self).tearDown()

    def testSetStack(self):
        self.stackview.setStack(self.mystack)
        self.stackview.setColormap("viridis", autoscale=True)
        my_trans_stack, params = self.stackview.getStack()
        self.assertEqual(my_trans_stack.shape, self.mystack.shape)
        self.assertTrue(numpy.array_equal(self.mystack,
                                          my_trans_stack))
        self.assertEqual(params["colormap"]["name"],
                         "viridis")

    def testSetStackPerspective(self):
        self.stackview.setStack(self.mystack, perspective=1)
        # my_orig_stack, params = self.stackview.getStack()
        my_trans_stack, params = self.stackview.getCurrentView()

        # get stack returns the transposed data, depending on the perspective
        self.assertEqual(my_trans_stack.shape,
                         (self.mystack.shape[1], self.mystack.shape[0], self.mystack.shape[2]))
        self.assertTrue(numpy.array_equal(numpy.transpose(self.mystack, axes=(1, 0, 2)),
                                          my_trans_stack))

    def testSetStackListOfImages(self):
        loi = [self.mystack[i] for i in range(self.mystack.shape[0])]

        self.stackview.setStack(loi)
        my_orig_stack, params = self.stackview.getStack(returnNumpyArray=True)
        my_trans_stack, params = self.stackview.getStack(returnNumpyArray=True)
        self.assertEqual(my_trans_stack.shape, self.mystack.shape)
        self.assertTrue(numpy.array_equal(self.mystack,
                                          my_trans_stack))
        self.assertTrue(numpy.array_equal(self.mystack,
                                          my_orig_stack))
        self.assertIsInstance(my_trans_stack, numpy.ndarray)

        self.stackview.setStack(loi, perspective=2)
        my_orig_stack, params = self.stackview.getStack(copy=False)
        my_trans_stack, params = self.stackview.getCurrentView(copy=False)
        # getStack(copy=False) must return the object set in setStack
        self.assertIs(my_orig_stack, loi)
        # getCurrentView(copy=False) returns a ListOfImages whose .images
        # attr is the original data
        self.assertEqual(my_trans_stack.shape,
                         (self.mystack.shape[2], self.mystack.shape[0], self.mystack.shape[1]))
        self.assertTrue(numpy.array_equal(numpy.array(my_trans_stack),
                                          numpy.transpose(self.mystack, axes=(2, 0, 1))))
        self.assertIsInstance(my_trans_stack,
                              ListOfImages)  # returnNumpyArray=False by default in getStack
        self.assertIs(my_trans_stack.images, loi)


class TestStackViewMainWindow(TestCaseQt):
    """Base class for tests of StackView."""

    def setUp(self):
        super(TestStackViewMainWindow, self).setUp()
        self.stackview = StackViewMainWindow()
        self.stackview.show()
        self.qWaitForWindowExposed(self.stackview)
        self.mystack = numpy.fromfunction(
            lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
            (100, 200, 300)
        )

    def tearDown(self):
        self.stackview.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.stackview.close()
        del self.stackview
        super(TestStackViewMainWindow, self).tearDown()

    def testSetStack(self):
        self.stackview.setStack(self.mystack)
        self.stackview.setColormap("viridis", autoscale=True)
        my_trans_stack, params = self.stackview.getStack()
        self.assertEqual(my_trans_stack.shape, self.mystack.shape)
        self.assertTrue(numpy.array_equal(self.mystack,
                                          my_trans_stack))
        self.assertEqual(params["colormap"]["name"],
                         "viridis")

    def testSetStackPerspective(self):
        self.stackview.setStack(self.mystack, perspective=1)
        my_trans_stack, params = self.stackview.getCurrentView()
        # get stack returns the transposed data, depending on the perspective
        self.assertEqual(my_trans_stack.shape,
                         (self.mystack.shape[1], self.mystack.shape[0], self.mystack.shape[2]))
        self.assertTrue(numpy.array_equal(numpy.transpose(self.mystack, axes=(1, 0, 2)),
                                          my_trans_stack))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestStackView))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestStackViewMainWindow))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
