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
from silx.gui.plot.items.image import ImageBase as ImageItem
from silx.gui.plot.items.scatter import Scatter as ScatterItem
from collections import OrderedDict
from silx.math.combo import min_max
import functools
import numpy
import silx


class StatsTable(qt.QTableWidget):
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
        ('kind', 9),
    ])

    COLUMNS = COLUMNS_INDEX.keys()

    COMPATAIBLE_ITEMS = (CurveItem, ImageItem, ScatterItem)


    def __init__(self, parent=None, plot=None):
        qt.QTableWidget.__init__(self, parent)
        """Next freeID for the curve"""
        self.plot = None
        self._legendToItems = {}
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
            [self._addItem(curve) for curve in self.plot.getAllCurves()]
            [self._addItem(image) for image in self.plot.getAllImages()]
            [self._addItem(scatter) for scatter in self.plot.getAllScatters()]
            self.plot.sigContentChanged.connect(self._plotContentChanged)

    def _plotContentChanged(self, action, kind, legend):
        if kind != 'curve':
            return
        if action == 'add':
            self._addItem(self.plot.getCurve(legend))
        elif action == 'remove':
            self._removeItem(legend, kind)

    def clear(self):
        self._plotItemToItemsIndex = {}
        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.setColumnHidden(self.COLUMNS_INDEX['kind'], True)

    def _addItem(self, item):
        def getKind(myItem):
            if isinstance(myItem, CurveItem):
                return 'curve'
            elif isinstance(myItem, ImageItem):
                return 'image'
            elif isinstance(myItem, ScatterItem):
                return 'scatter'
            else:
                return None
        assert isinstance(item, self.COMPATAIBLE_ITEMS)
        if item.getLegend() in self._legendToItems:
            self._updateStats(item, getKind(item))
            return

        self.setRowCount(self.rowCount() + 1)

        itemLegend = None
        indexTable = self.rowCount() - 1
        kind = getKind(item)
        for itemName in self.COLUMNS:
            _item = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
            if itemName == 'legend':
                _item.setText(item.getLegend())
                itemLegend = _item
            if itemName == 'kind':
                _item.setText(kind)
            self.setItem(indexTable, self.COLUMNS_INDEX[itemName], _item)

        assert itemLegend
        self._legendToItems[item.getLegend()] = itemLegend
        self._updateStats(legend=item.getLegend(), kind=kind)

        callback = functools.partial(
            silx.utils.weakref.WeakMethodProxy(self._updateStats),
            item.getLegend(), kind)
        item.sigItemChanged.connect(callback)

    def _removeItem(self, legend, kind):
        if legend not in self._legendToItems or not self.plot:
            return

        callback = functools.partial(
            silx.utils.weakref.WeakMethodProxy(self._updateStats),
            legend,
            kind)
        if kind == 'curve':
            item = self.plot.getCurve(legend)
        elif kind == 'image':
            item = self.plot.getImage(legend)
        elif kind == 'scatter':
            item = self.plot.getScatter(legend)
        else:
            raise ValueError('Kind not managed')
        if item:
            item.sigItemChanged.discconnect(callback)
        self.firstItem = self._legendToItems[legend]
        del self._legendToItems[legend]
        self.removeRow(self.firstItem.row())

    def _updateStats(self, legend, kind, *args, **kwargs):
        def computeCurveStats(item):
            res = {}
            if item is None:
                return res
            # TODO: reduce data according to the plot zoom
            xData, yData = item.getData(copy=False)[0:2]
            min, max = min_max(yData)
            res['min'], res['max'], res['delta'] = min, max, (max - min)
            res['coords min'] = xData[numpy.where(yData == min)]
            res['coords max'] = xData[numpy.where(yData == max)]
            com = numpy.sum(xData * yData) / numpy.sum(yData)
            res['COM'] = com
            res['std'] = numpy.std(yData)
            res['mean'] = numpy.mean(yData)
            return res

        def computeImageStats(item):
            def getCoordsFor(data, value):
                coordsY, coordsX = numpy.where(data==value)
                if len(coordsX) is 0:
                    return []
                if len(coordsX) is 1:
                    return (coordsX[0], coordsY[0])
                coords = []
                for xCoord, yCoord in zip(coordsX, coordsY):
                    coord = (xCoord, yCoord)
                    coords.append(coord)
                return coords

            res = {}
            if item is None:
                return res
            # TODO: reduce data according to the plot zoom
            data = item.getData(copy=False)
            min, max = min_max(data)
            res['min'], res['max'], res['delta'] = min, max, (max - min)
            res['coords min'] = getCoordsFor(min, data)
            res['coords max'] = getCoordsFor(max, data)

            com = numpy.sum(data) / data.size
            res['COM'] = com
            res['std'] = numpy.std(data)
            res['mean'] = numpy.mean(data)
            return res

        def computeScatterStats(item):
            def getCoordsFor(xData, yData, valueData, value):
                indexes = numpy.where(valueData == value)
                if len(indexes) is 0:
                    return []
                if len(indexes) is 1:
                    return (xData[indexes[0][0]], yData[indexes[0][0]])
                res = []
                for index in indexes:
                    assert(len(index) is 1)
                    res.append((xData[index[0]], yData[index[0]]))
                return res
            res = {}
            if item is None:
                return res
            # TODO: reduce data according to the plot zoom
            xData, yData, valueData, xerror, yerror = item.getData(copy=False)
            min, max = min_max(valueData)
            res['min'], res['max'], res['delta'] = min, max, (max - min)
            res['coords min'] = getCoordsFor(xData, yData, valueData, min)
            res['coords max'] = getCoordsFor(xData, yData, valueData, max)
            com = numpy.sum(xData * yData) / numpy.sum(yData)
            res['COM'] = com
            res['std'] = numpy.std(valueData)
            res['mean'] = numpy.mean(valueData)
            return res

        def retrieveItems(item):
            items = {}
            itemLegend = self._legendToItems[item.getLegend()]
            items['legend'] = itemLegend
            assert itemLegend
            for itemName in self.COLUMNS:
                if itemName == 'legend':
                    continue
                items[itemName] = self.item(itemLegend.row(),
                                            self.COLUMNS_INDEX[itemName])
            return items


        assert kind in ('curve', 'image', 'scatter')
        if kind == 'curve':
            item = self.plot.getCurve(legend)
            stats = computeCurveStats(item)
        elif kind == 'image':
            item = self.plot.getImage(legend)
            stats = computeImageStats(item)
        elif kind == 'scatter':
            item = self.plot.getScatter(legend)
            stats = computeScatterStats(item)
        else:
            raise ValueError('kind not managed')

        if not item:
            return
        assert item.getLegend() in self._legendToItems
        assert isinstance(item, self.COMPATAIBLE_ITEMS)

        items = retrieveItems(item)

        for itemLabel in self.COLUMNS:
            if itemLabel in ('legend', 'kind'):
                continue
            assert itemLabel in stats
            items[itemLabel].setText(str(stats[itemLabel]))

    def currentChanged(self, current, previous):
        legendItem = self.item(current.row(), self.COLUMNS_INDEX['legend'])
        kindItem = self.item(current.row(), self.COLUMNS_INDEX['kind'])
        kind = kindItem.text()
        if kind == 'curve':
            self.plot.setActiveCurve(legendItem.text())
        elif kind =='image':
            self.plot.setActiveImage(legendItem.text())
        elif kind =='scatter':
            self.plot._setActiveItem('scatter', legendItem.text())
        else:
            raise ValueError('kind not managed')
        qt.QTableWidget.currentChanged(self, current, previous)


class StatsDockWidget(qt.QDockWidget):
    """
    """
    def __init__(self, parent=None, plot=None):
        qt.QDockWidget.__init__(self, parent)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._statsTable = StatsTable(parent=self, plot=plot)
        self.setWidget(self._statsTable)
