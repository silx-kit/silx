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
"""Basic tests for PlotWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/01/2018"


import logging
import pytest
import unittest

from silx.gui.utils.testutils import TestCaseQt

from silx.gui import qt
from silx.gui.plot import PlotWidget


logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("test_options_class_attr")
class PlotWidgetTestCase(TestCaseQt):
    """Base class for tests of PlotWidget, not a TestCase in itself.

    plot attribute is the PlotWidget created for the test.
    """
    __screenshot_already_taken = False
    backend = None

    def _createPlot(self):
        return PlotWidget(backend=self.backend)

    def setUp(self):
        super(PlotWidgetTestCase, self).setUp()
        self.plot = self._createPlot()
        self.plot.show()
        self.plotAlive = True
        self.qWaitForWindowExposed(self.plot)
        TestCaseQt.mouseClick(self, self.plot, button=qt.Qt.LeftButton, pos=(0, 0))

    def __onPlotDestroyed(self):
        self.plotAlive = False

    def _waitForPlotClosed(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.destroyed.connect(self.__onPlotDestroyed)
        self.plot.close()
        del self.plot
        for _ in range(100):
            if not self.plotAlive:
                break
            self.qWait(10)
        else:
            logger.error("Plot is still alive")

    def tearDown(self):
        if not self._currentTestSucceeded():
            # MPL is the only widget which uses the real system mouse.
            # In case of a the windows is outside of the screen, minimzed,
            # overlapped by a system popup, the MPL widget will not receive the
            # mouse event.
            # Taking a screenshot help debuging this cases in the continuous
            # integration environement.
            if not PlotWidgetTestCase.__screenshot_already_taken:
                PlotWidgetTestCase.__screenshot_already_taken = True
                self.logScreenShot()
        self.qapp.processEvents()
        self._waitForPlotClosed()
        super(PlotWidgetTestCase, self).tearDown()
