#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This example illustrates how to use a :class:`ItemsSelectionDialog` widget
associated with a :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "28/06/2017"

from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget
from silx.gui.plot.ItemsSelectionDialog import ItemsSelectionDialog

app = qt.QApplication([])

pw = PlotWidget()
pw.addCurve([0, 1, 2], [3, 2, 1], "A curve")
pw.addScatter([0, 1, 2.5], [3, 2.5, 0.9], [8, 9, 72], "A scatter")
pw.addHistogram([0, 1, 2.5], [0, 1, 2, 3], "A histogram")
pw.addImage([[0, 1, 2], [3, 2, 1]], "An image")
pw.show()

isd = ItemsSelectionDialog(plot=pw)
isd.setItemsSelectionMode(qt.QTableWidget.ExtendedSelection)
result = isd.exec_()
if result:
    for item in isd.getSelectedItems():
        print(item.getLegend(), type(item))
else:
    print("Selection cancelled")

app.exec_()
