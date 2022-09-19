# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
__date__ = "20/03/2017"


import unittest
import numpy

from silx.gui.utils.testutils import TestCaseQt, SignalListener

from silx.gui import qt
from silx.gui.plot import StackView
from silx.gui.plot.StackView import StackViewMainWindow

from silx.utils.array_like import ListOfImages


class TestStackView(TestCaseQt):
    """Base class for tests of StackView."""

    def setUp(self):
        super(TestStackView, self).setUp()
        self.stackview = StackView()
        self.stackview.show()
        self.qWaitForWindowExposed(self.stackview)
        self.mystack = numpy.fromfunction(
            lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
            (10, 20, 30)
        )

    def tearDown(self):
        self.stackview.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.stackview.close()
        del self.stackview
        super(TestStackView, self).tearDown()

    def testScaleColormapRangeToStack(self):
        """Test scaleColormapRangeToStack"""
        self.stackview.setStack(self.mystack)
        self.stackview.setColormap("viridis")
        colormap = self.stackview.getColormap()

        # Colormap autoscale to image
        self.assertEqual(colormap.getVRange(), (None, None))
        self.stackview.scaleColormapRangeToStack()

        # Colormap range set according to stack range
        self.assertEqual(colormap.getVRange(), (self.mystack.min(), self.mystack.max()))

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

    def testPerspective(self):
        self.stackview.setStack(numpy.arange(24).reshape((2, 3, 4)))
        self.assertEqual(self.stackview._perspective, 0,
                         "Default perspective is not 0 (dim1-dim2).")

        self.stackview._StackView__planeSelection.setPerspective(1)
        self.assertEqual(self.stackview._perspective, 1,
                         "Plane selection combobox not updating perspective")

        self.stackview.setStack(numpy.arange(6).reshape((1, 2, 3)))
        self.assertEqual(self.stackview._perspective, 1,
                         "Perspective not preserved when calling setStack "
                         "without specifying the perspective parameter.")

        self.stackview.setStack(numpy.arange(24).reshape((2, 3, 4)), perspective=2)
        self.assertEqual(self.stackview._perspective, 2,
                         "Perspective not set in setStack(..., perspective=2).")

    def testDefaultTitle(self):
        """Test that the plot title contains the proper Z information"""
        self.stackview.setStack(numpy.arange(24).reshape((4, 3, 2)),
                                calibrations=[(0, 1), (-10, 10), (3.14, 3.14)])
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=0")
        self.stackview.setFrameNumber(2)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=2")

        self.stackview._StackView__planeSelection.setPerspective(1)
        self.stackview.setFrameNumber(0)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=-10")
        self.stackview.setFrameNumber(2)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=10")

        self.stackview._StackView__planeSelection.setPerspective(2)
        self.stackview.setFrameNumber(0)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=3.14")
        self.stackview.setFrameNumber(1)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Image z=6.28")

    def testCustomTitle(self):
        """Test setting the plot title with a user defined callback"""
        self.stackview.setStack(numpy.arange(24).reshape((4, 3, 2)),
                                calibrations=[(0, 1), (-10, 10), (3.14, 3.14)])

        def title_callback(frame_idx):
            return "Cubed index title %d" % (frame_idx**3)

        self.stackview.setTitleCallback(title_callback)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Cubed index title 0")
        self.stackview.setFrameNumber(2)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Cubed index title 8")

        # perspective should not matter, only frame index
        self.stackview._StackView__planeSelection.setPerspective(1)
        self.stackview.setFrameNumber(0)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Cubed index title 0")
        self.stackview.setFrameNumber(2)
        self.assertEqual(self.stackview._plot.getGraphTitle(),
                         "Cubed index title 8")

        with self.assertRaises(TypeError):
            # setTitleCallback should not accept non-callable objects like strings
            self.stackview.setTitleCallback(
                    "Là, vous faites sirop de vingt-et-un et vous dites : "
                    "beau sirop, mi-sirop, siroté, gagne-sirop, sirop-grelot,"
                    " passe-montagne, sirop au bon goût.")

    def testStackFrameNumber(self):
        self.stackview.setStack(self.mystack)
        self.assertEqual(self.stackview.getFrameNumber(), 0)

        listener = SignalListener()
        self.stackview.sigFrameChanged.connect(listener)

        self.stackview.setFrameNumber(1)
        self.assertEqual(self.stackview.getFrameNumber(), 1)
        self.assertEqual(listener.arguments(), [(1,)])


class TestStackViewMainWindow(TestCaseQt):
    """Base class for tests of StackView."""

    def setUp(self):
        super(TestStackViewMainWindow, self).setUp()
        self.stackview = StackViewMainWindow()
        self.stackview.show()
        self.qWaitForWindowExposed(self.stackview)
        self.mystack = numpy.fromfunction(
            lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
            (10, 20, 30)
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
