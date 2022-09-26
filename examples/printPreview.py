#!/usr/bin/env python
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
"""This script illustrates how to add a print preview tool button to any plot
widget inheriting :class:`~silx.gui.plot.PlotWidget`.

Three plot widgets are instantiated. One of them uses a standalone
:class:`~silx.gui.plot.PrintPreviewToolButton.PrintPreviewToolButton`,
while the other two use a
:class:`~silx.gui.plot.PrintPreviewToolButton.SingletonPrintPreviewToolButton`
which allows them to send their content to the same print preview page.
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "25/07/2017"

import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot import PrintPreviewToolButton


class MyPrintPreviewButton(PrintPreviewToolButton.PrintPreviewToolButton):
    """This class illustrates how to subclass PrintPreviewToolButton
    to add a title and a comment."""
    def getTitle(self):
        return "Widget 1's plot"

    def getCommentAndPosition(self):
        legends = self.getPlot().getAllCurves(just_legend=True)
        comment = "Curves displayed in widget 1:\n\t"
        if legends:
            comment += ", ".join(legends)
        else:
            comment += "none"
        return comment, "CENTER"


app = qt.QApplication([])

x = numpy.arange(1000)

# first widget has a standalone preview action with custom title and comment
pw1 = PlotWidget()
pw1.setWindowTitle("Widget 1 with standalone print preview")
toolbar1 = qt.QToolBar(pw1)
toolbutton1 = MyPrintPreviewButton(parent=toolbar1, plot=pw1)
pw1.addToolBar(toolbar1)
toolbar1.addWidget(toolbutton1)
pw1.show()
pw1.addCurve(x, numpy.tan(x * 2 * numpy.pi / 1000))

# next two plots share a common standard print preview
pw2 = PlotWidget()
pw2.setWindowTitle("Widget 2 with shared print preview")
toolbar2 = qt.QToolBar(pw2)
toolbutton2 = PrintPreviewToolButton.SingletonPrintPreviewToolButton(
        parent=toolbar2, plot=pw2)
pw2.addToolBar(toolbar2)
toolbar2.addWidget(toolbutton2)
pw2.show()
pw2.addCurve(x, numpy.sin(x * 2 * numpy.pi / 1000))


pw3 = PlotWidget()
pw3.setWindowTitle("Widget 3 with shared print preview")
toolbar3 = qt.QToolBar(pw3)
toolbutton3 = PrintPreviewToolButton.SingletonPrintPreviewToolButton(
        parent=toolbar3, plot=pw3)
pw3.addToolBar(toolbar3)
toolbar3.addWidget(toolbutton3)
pw3.show()
pw3.addCurve(x, numpy.cos(x * 2 * numpy.pi / 1000))


app.exec()
