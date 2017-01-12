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
"""Basic tests for Profile"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/12/2016"


import numpy
import unittest

from silx.test.utils import ParametricTestCase
from silx.gui.test.utils import (
    TestCaseQt, getQToolButtonFromAction)
from silx.gui import qt
from silx.gui.plot import PlotWindow, Profile


# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


class TestProfileToolBar(TestCaseQt, ParametricTestCase):
    """Tests for ProfileToolBar widget."""

    def setUp(self):
        super(TestProfileToolBar, self).setUp()
        profileWindow = PlotWindow()
        self.plot = PlotWindow()
        self.toolBar = Profile.ProfileToolBar(
            plot=self.plot, profileWindow=profileWindow)
        self.plot.addToolBar(self.toolBar)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)
        profileWindow.show()
        self.qWaitForWindowExposed(profileWindow)

        self.mouseMove(self.plot)  # Move to center
        self.qapp.processEvents()

    def tearDown(self):
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        del self.toolBar

        super(TestProfileToolBar, self).tearDown()

    def testAlignedProfile(self):
        """Test horizontal and vertical profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        widget = self.plot.getWidgetHandle()

        # 2 positions to use for mouse events
        pos1 = widget.width() * 0.4, widget.height() * 0.4
        pos2 = widget.width() * 0.6, widget.height() * 0.6

        for action in (self.toolBar.hLineAction, self.toolBar.vLineAction):
            with self.subTest(mode=action.text()):
                # Trigger tool button for mode
                toolButton = getQToolButtonFromAction(action)
                self.assertIsNot(toolButton, None)
                self.mouseMove(toolButton)
                self.mouseClick(toolButton, qt.Qt.LeftButton)

                # Without image
                self.mouseMove(widget, pos=pos1)
                self.mouseClick(widget, qt.Qt.LeftButton, pos=pos1)

                # with image
                self.plot.addImage(numpy.arange(100 * 100).reshape(100, -1))
                self.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                self.mouseMove(widget, pos=pos2)
                self.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)

                self.mouseMove(widget)
                self.mouseClick(widget, qt.Qt.LeftButton)

    def testDiagonalProfile(self):
        """Test diagonal profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        widget = self.plot.getWidgetHandle()

        # 2 positions to use for mouse events
        pos1 = widget.width() * 0.4, widget.height() * 0.4
        pos2 = widget.width() * 0.6, widget.height() * 0.6

        # Trigger tool button for diagonal profile mode
        toolButton = getQToolButtonFromAction(self.toolBar.lineAction)
        self.assertIsNot(toolButton, None)
        self.mouseMove(toolButton)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

        for image in (False, True):
            with self.subTest(image=image):
                if image:
                    self.plot.addImage(numpy.arange(100 * 100).reshape(100, -1))

                self.mouseMove(widget, pos=pos1)
                self.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                self.mouseMove(widget, pos=pos2)
                self.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)

                self.plot.clear()


def suite():
    test_suite = unittest.TestSuite()
    # test_suite.addTest(positionInfoTestSuite)
    for testClass in (TestProfileToolBar,):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
            testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
