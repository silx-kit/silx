# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "07/03/2018"


from silx.gui import qt
from silx.gui.plot.items.curve import Curve as CurveItem
from collections import OrderedDict
from silx.math.combo import min_max
import functools
import numpy
import silx


class CurvesStatsTable(qt.QTableWidget):
    """
    TableWidget displying for each curves contained by the Plot some
    information:
    
    * legend
    * minimal value
    * maximal value
    * standard deviation (std)
    
    :param parent: The widget's parent.
    :param plot: :class:`.PlotWidget` instance on which to operate
    """
    COLUMNS_INDEX = OrderedDict([
        ('legend', 0),
        ('min', 1),
        ('max', 2),
        ('std', 3)
    ])

    COLUMNS = COLUMNS_INDEX.keys()


    def __init__(self, parent=None, plot=None):
        qt.QTableWidget.__init__(self, parent)
        """Next freeID for the curve"""
        self.plot = None
        self._curveToItems = {}
        """Associate the curve legend to his first item"""
        self.setColumnCount(len(self.COLUMNS))
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setPlot(plot)

    def setPlot(self, plot):
        """
        Define the plot to interact with

        :param plot: :class:`.PlotWidget` instance on which to operate 
        """
        if self.plot:
            self.plot.sigContentChanged.disconnect(self._plotContentChanged)
        self.plot = plot
        self.clear()
        if self.plot:
            [self._addCurve(curve) for curve in self.plot.getAllCurves()]
            self.plot.sigContentChanged.connect(self._plotContentChanged)

    def _plotContentChanged(self, action, kind, legend):
        if kind != 'curve':
            return
        if action == 'add':
            self._addCurve(self.plot.getCurve(legend))
        elif action == 'remove':
            self._removeCurve(legend)

    def clear(self):
        self._curveToItemsIndex = {}
        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    def _addCurve(self, curve):
        assert isinstance(curve, CurveItem)
        if curve.getLegend() in self._curveToItems:
            self._updateCurveStats(curve)
            return

        self.setRowCount(self.rowCount() + 1)

        itemLegend = qt.QTableWidgetItem(curve.getLegend(),
                                         type=qt.QTableWidgetItem.Type)
        itemMin = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemMax = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemStd = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)

        indexTable = self.rowCount() - 1
        self.setItem(indexTable, self.COLUMNS_INDEX['legend'], itemLegend)
        self.setItem(indexTable, self.COLUMNS_INDEX['min'], itemMin)
        self.setItem(indexTable, self.COLUMNS_INDEX['max'], itemMax)
        self.setItem(indexTable, self.COLUMNS_INDEX['std'], itemStd)
        self._curveToItems[curve.getLegend()] = itemLegend
        self._updateCurveStats(legend=curve.getLegend())

        callback = functools.partial(silx.utils.weakref.WeakMethodProxy(self._updateCurveStats), curve.getLegend())
        curve.sigItemChanged.connect(callback)

    def _removeCurve(self, legend):
        if legend not in self._curveToItems or not self.plot:
            return

        callback = functools.partial(
            silx.utils.weakref.WeakMethodProxy(self._updateCurveStats),
            legend)

        curve = self.plot.getCurve(legend)
        if curve:
            curve.sigItemChanged.discconnect(callback)
        self.firstItem = self._curveToItems[legend]
        del self._curveToItems[legend]
        self.removeRow(self.firstItem.row())

    def _updateCurveStats(self, legend, *args, **kwargs):
        curve = self.plot.getCurve(legend)
        if not curve:
            return
        assert isinstance(curve, CurveItem)
        assert curve.getLegend() in self._curveToItems

        itemLegend = self._curveToItems[curve.getLegend()]
        itemMin = self.item(itemLegend.row(), self.COLUMNS_INDEX['min'])
        itemMax = self.item(itemLegend.row(), self.COLUMNS_INDEX['max'])
        itemStd = self.item(itemLegend.row(), self.COLUMNS_INDEX['std'])

        yData = curve.getData(copy=False)[1]
        min, max = min_max(yData)
        itemMin.setText(str(min))
        itemMax.setText(str(max))

        std = numpy.std(yData)
        itemStd.setText(str(std))

    def currentChanged(self, current, previous):
        legendItem = self.item(current.row(), self.COLUMNS_INDEX['legend'])
        self.plot.setActiveCurve(legendItem.text())
        qt.QTableWidget.currentChanged(self, current, previous)


class CurvesStatsDockWidget(qt.QDockWidget):
    """
    """
    def __init__(self, parent=None, plot=None):
        qt.QDockWidget.__init__(self, parent)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._curveStatsTable = CurvesStatsTable(parent=self, plot=plot)
        self.setWidget(self._curveStatsTable)


if __name__ == "__main__":
    app = qt.QApplication([])

    tablewidget = CurvesStatsWidget()
    tablewidget.setWindowTitle("TableWidget")
    tablewidget.show()

    app.exec_()
