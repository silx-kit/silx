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
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/11/2017"


import unittest

from .._utils import test
from . import testColorBar
from . import testColormap
from . import testColormapDialog
from . import testColors
from . import testCurvesROIWidget
from . import testAlphaSlider
from . import testInteraction
from . import testLegendSelector
from . import testMaskToolsWidget
from . import testScatterMaskToolsWidget
from . import testPlotInteraction
from . import testPlotTools
from . import testPlotWidgetNoBackend
from . import testPlotWidget
from . import testPlotWindow
from . import testProfile
from . import testStackView
from . import testItem
from . import testUtilsAxis
from . import testLimitConstraints
from . import testComplexImageView
from . import testImageView
from . import testSaveAction


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        [test.suite(),
         testColorBar.suite(),
         testColors.suite(),
         testColormapDialog.suite(),
         testCurvesROIWidget.suite(),
         testAlphaSlider.suite(),
         testInteraction.suite(),
         testLegendSelector.suite(),
         testMaskToolsWidget.suite(),
         testScatterMaskToolsWidget.suite(),
         testPlotInteraction.suite(),
         testPlotWidgetNoBackend.suite(),
         testPlotTools.suite(),
         testPlotWidget.suite(),
         testPlotWindow.suite(),
         testProfile.suite(),
         testStackView.suite(),
         testColormap.suite(),
         testItem.suite(),
         testUtilsAxis.suite(),
         testLimitConstraints.suite(),
         testComplexImageView.suite(),
         testImageView.suite(),
         testSaveAction.suite()])
    return test_suite
