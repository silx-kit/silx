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
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "18/02/2016"


import unittest

from .testColormapDialog import suite as testColormapDialogSuite
from .testColors import suite as testColorsSuite
from .testCurvesROIWidget import suite as testCurvesROIWidgetSuite
from .testInteraction import suite as testInteractionSuite
from .testLegendSelector import suite as testLegendSelectorSuite
from .testMaskToolsWidget import suite as testMaskToolsWidgetSuite
from .testPlotInteraction import suite as testPlotInteractionSuite
from .testPlotTools import suite as testPlotToolsSuite
from .testPlotWidget import suite as testPlotWidgetSuite
from .testPlotWindow import suite as testPlotWindowSuite
from .testPlot import suite as testPlotSuite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        [testColorsSuite(),
         testColormapDialogSuite(),
         testCurvesROIWidgetSuite(),
         testInteractionSuite(),
         testLegendSelectorSuite(),
         testMaskToolsWidgetSuite(),
         testPlotInteractionSuite(),
         testPlotSuite(),
         testPlotToolsSuite(),
         testPlotWidgetSuite(),
         testPlotWindowSuite()])
    return test_suite
