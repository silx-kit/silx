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
    TableWidget displaying for each curves contained by the Plot some
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
        ('coords min', 2),
        ('max', 3),
        ('coords max', 4),
        ('delta', 5),
        ('std', 6),
        ('mean', 7),
        ('COM', 8),
        ('COM coords', 9),
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

        itemLegend = None
        indexTable = self.rowCount() - 1
        print('set on %s' % indexTable)
        for itemName in self.COLUMNS:
            print(itemName)
            item = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
            if itemName == 'legend':
                item.setText(curve.getLegend())
                itemLegend = item
            self.setItem(indexTable, self.COLUMNS_INDEX[itemName], item)

        assert itemLegend
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
        def retrieveItems(curve):
            items = {}
            itemLegend = self._curveToItems[curve.getLegend()]
            items['legend'] = itemLegend
            assert itemLegend
            for itemName in self.COLUMNS:
                if itemName == 'legend':
                    continue
                items[itemName] = self.item(itemLegend.row(),
                                            self.COLUMNS_INDEX[itemName])
            return items

        curve = self.plot.getCurve(legend)
        if not curve:
            return
        assert isinstance(curve, CurveItem)
        assert curve.getLegend() in self._curveToItems

        items = retrieveItems(curve)

        # TODO: reduce data according to the plot zoom
        xData, yData = curve.getData(copy=False)[0:2]
        min, max = min_max(yData)
        items['min'].setText(str(min))
        items['coords min'].setText(str(xData[numpy.where(yData==min)]))
        items['max'].setText(str(max))
        items['coords max'].setText(str(xData[numpy.where(yData==max)]))
        items['delta'].setText(str(max - min))
        com = numpy.sum(yData) / len(yData)
        # TODO : add the coords of the COM
        comCoords = xData[(numpy.abs(yData - com)).argmin()]
        items['COM'].setText(str(com))
        items['COM coords'].setText(str(comCoords))

        std = numpy.std(yData)
        items['std'].setText(str(std))
        mean = numpy.mean(yData)
        items['mean'].setText(str(mean))

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
