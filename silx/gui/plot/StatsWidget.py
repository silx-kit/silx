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
from silx.gui.widgets.TableWidget import TableWidget
from collections import OrderedDict
from silx.math.combo import min_max
import functools
import numpy
import silx


class StatsWidget(qt.QWidget):
    class OptionsWidget(qt.QWidget):

        ITEM_SEL_OPTS = ('All items', 'Active item only')

        ITEM_DATA_RANGE_OPTS = ('full data range', 'visible data range')

        def __init__(self, parent=None):
            qt.QWidget.__init__(self, parent)
            self.setLayout(qt.QHBoxLayout())
            spacer = qt.QWidget(parent=self)
            spacer.setSizePolicy(qt.QSizePolicy.Expanding,
                                 qt.QSizePolicy.Minimum)
            self.layout().setContentsMargins(0, 0, 0, 0)
            self.layout().addWidget(qt.QLabel('Stats opts:', parent=self))
            self.layout().addWidget(spacer)
            self.layout().addWidget(qt.QLabel('display', parent=self))
            self.itemSelection = qt.QComboBox(parent=self)
            for opt in self.ITEM_SEL_OPTS:
                self.itemSelection.addItem(opt)
            self.layout().addWidget(self.itemSelection)
            self.layout().addWidget(qt.QLabel('compute stats on', parent=self))
            self.dataRangeSelection = qt.QComboBox(parent=self)
            for opt in self.ITEM_DATA_RANGE_OPTS:
                self.dataRangeSelection.addItem(opt)
            self.layout().addWidget(self.dataRangeSelection)

    def __init__(self, parent=None, plot=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._options = self.OptionsWidget(parent=self)
        self.layout().addWidget(self._options)
        self._statsTable = StatsTable(parent=self, plot=plot)
        self.layout().addWidget(self._statsTable)
        self.setPlot = self._statsTable.setPlot

        self._options.itemSelection.currentTextChanged.connect(
            self._optSelectionChanged)
        self._options.dataRangeSelection.currentTextChanged.connect(
            self._optDataRangeChanged)

    def _optSelectionChanged(self, opt):
        assert opt in self.OptionsWidget.ITEM_SEL_OPTS
        self._statsTable.setDisplayOnlyActiveItem(opt == 'Active item only')

    def _optDataRangeChanged(self, opt):
        assert opt in self.OptionsWidget.ITEM_DATA_RANGE_OPTS
        self._statsTable.setStatsOnVisibleData(opt == 'visible data range')


class _FloatItem(qt.QTableWidgetItem):
    def __init__(self):
        qt.QTableWidgetItem.__init__(self, type=qt.QTableWidgetItem.Type)

    def __lt__(self, other):
        return float(self.text()) < float(other.text())


class StatsTable(TableWidget):
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
        ('kind', 1),
        ('min', 2),
        ('coords min', 3),
        ('max', 4),
        ('coords max', 5),
        ('delta', 6),
        ('std', 7),
        ('mean', 8),
        ('COM', 9),
    ])

    COLUMNS = COLUMNS_INDEX.keys()

    COMPATIBLE_KINDS = {
        'curve': CurveItem,
        'image':ImageItem,
        'scatter': ScatterItem
    }

    COMPATIBLE_ITEMS = tuple(COMPATIBLE_KINDS.values())

    FORMATED_COLUMNS = ('mean', 'com', 'std', 'delta', 'min', 'max', 'delta')
    """The Columns for which we want to apply a specific format"""

    NUMBER_FORMAT = '{0:.3f}'
    """The format to apply to the `FORMATED_COLUMNS`"""

    @staticmethod
    def getKind(myItem):
        if isinstance(myItem, CurveItem):
            return 'curve'
        elif isinstance(myItem, ImageItem):
            return 'image'
        elif isinstance(myItem, ScatterItem):
            return 'scatter'
        else:
            return None

    def __init__(self, parent=None, plot=None):
        qt.QTableWidget.__init__(self, parent)
        """Next freeID for the curve"""
        self.plot = None
        self._displayOnlyActItem = False
        self._statsOnVisibleData = False
        self._lgdAndKindToItems = {}
        """Associate to a tuple(legend, kind) the items legend"""
        self.callbackImage = None
        self.callbackScatter = None
        self.callbackCurve = None
        """Associate the curve legend to his first item"""
        self.setColumnCount(len(self.COLUMNS))
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setPlot(plot)
        self.setSortingEnabled(True)

    def setPlot(self, plot):
        """
        Define the plot to interact with

        :param plot: :class:`.PlotWidget` instance on which to operate 
        """
        if self.plot:
            self.plot.sigContentChanged.disconnect(self._plotContentChanged)
            self.plot.sigPlotSignal.disconnect(self._zoomPlotChanged)
        self.plot = plot
        self.clear()
        if self.plot:
            self.plot.sigPlotSignal.connect(self._zoomPlotChanged)
            self._updateItemObserve(init=True)

    def _updateItemObserve(self, switchItemsDisplayedType=False, init=False):
        if self.plot:
            if init is False:
                self._dealWithPlotConnection(create=False)
            self.clear()
            if switchItemsDisplayedType:
                self._displayOnlyActItem = not self._displayOnlyActItem
            if self._displayOnlyActItem is True:
                activeCurve = self.plot.getActiveCurve(just_legend=False)
                activeScatter = self.plot._getActiveItem(kind='scatter',
                                                         just_legend=False)
                activeImage = self.plot.getActiveImage(just_legend=False)
                if activeCurve:
                    self._addItem(activeCurve)
                if activeImage:
                    self._addItem(activeImage)
                if activeScatter:
                    self._addItem(activeScatter)
            else:
                [self._addItem(curve) for curve in self.plot.getAllCurves()]
                [self._addItem(image) for image in self.plot.getAllImages()]
                [self._addItem(scatter) for scatter in self.plot.getAllScatters()]
                self.plot.sigContentChanged.connect(self._plotContentChanged)
            self._dealWithPlotConnection(create=True)

    def _dealWithPlotConnection(self, create=True):
        if self._displayOnlyActItem:
            if self.callbackImage is None:
                self.callbackImage = functools.partial(self._activeItemChanged, 'image')
                self.callbackScatter = functools.partial(self._activeItemChanged,
                                                    'scatter')
                self.callbackCurve = functools.partial(self._activeItemChanged, 'curve')
            if create is True:
                self.plot.sigActiveImageChanged.connect(self.callbackImage)
                self.plot.sigActiveScatterChanged.connect(self.callbackScatter)
                self.plot.sigActiveCurveChanged.connect(self.callbackCurve)
            else:
                self.plot.sigActiveImageChanged.disconnect(self.callbackImage)
                self.plot.sigActiveScatterChanged.disconnect(self.callbackScatter)
                self.plot.sigActiveCurveChanged.disconnect(self.callbackCurve)
        else:
            if create is True:
                self.plot.sigContentChanged.connect(self._plotContentChanged)
            else:
                self.plot.sigContentChanged.disconnect(self._plotContentChanged)
                # Note: connection on Item arre managed by the _removeItem function

    def clear(self, rmConnections=False):
        lgdsAndKinds = list(self._lgdAndKindToItems.keys())
        for lgdAndKind in lgdsAndKinds:
            self._removeItem(legend=lgdAndKind[0], kind=lgdAndKind[1])
        self._lgdAndKindToItems = {}
        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        if hasattr(self.horizontalHeader(), 'setSectionResizeMode'):  # Qt5
            self.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        else:  # Qt4
            self.horizontalHeader().setResizeMode(qt.QHeaderView.ResizeToContents)
        self.setColumnHidden(self.COLUMNS_INDEX['kind'], True)

    def _addItem(self, item):
        assert isinstance(item, self.COMPATIBLE_ITEMS)
        if (item.getLegend(), self.getKind(item)) in self._lgdAndKindToItems:
            self._updateStats(item, self.getKind(item))
            return

        self.setRowCount(self.rowCount() + 1)

        itemLegend = None
        indexTable = self.rowCount() - 1
        kind = self.getKind(item)
        for itemName in self.COLUMNS:
            if itemName in ('min', 'max', 'COM', 'delta', 'std', 'mean'):
                _item = _FloatItem()
            else:
                _item = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
            if itemName == 'legend':
                _item.setText(item.getLegend())
                itemLegend = _item
            if itemName == 'kind':
                _item.setText(kind)
            self.setItem(indexTable, self.COLUMNS_INDEX[itemName], _item)

        assert itemLegend
        self._lgdAndKindToItems[(item.getLegend(), kind)] = itemLegend
        self._updateStats(legend=item.getLegend(), kind=kind)

        callback = functools.partial(
            silx.utils.weakref.WeakMethodProxy(self._updateStats),
            item.getLegend(), kind)
        item.sigItemChanged.connect(callback)

    def _removeItem(self, legend, kind):
        if (legend, kind) not in self._lgdAndKindToItems or not self.plot:
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
        self.firstItem = self._lgdAndKindToItems[(legend, kind)]
        del self._lgdAndKindToItems[(legend, kind)]
        self.removeRow(self.firstItem.row())

    def _updateCurrentStats(self):
        for lgdAndKind in self._lgdAndKindToItems:
            self._updateStats(lgdAndKind[0], lgdAndKind[1])

    def _updateStats(self, legend, kind, *args, **kwargs):
        def noDataSelected():
            res = {}
            res['min'] = res['max'] = res['delta'] = None
            res['coords min'] = res['coords max'] = None
            res['COM'] = res['std'] = res['mean'] = None
            return res

        def computeCurveStats(item):
            res = {}
            if item is None:
                return res
            xData, yData = item.getData(copy=True)[0:2]

            assert self.plot
            if self._statsOnVisibleData:
                minX, maxX = self.plot.getXAxis().getLimits()
                yData = yData[(minX<=xData) & (xData<=maxX)]
                xData = xData[(minX<=xData) & (xData<=maxX)]

            if yData.size is 0:
                return noDataSelected()
            min, max = min_max(yData)
            res['min'], res['max'], res['delta'] = min, max, (max - min)
            res['coords min'] = xData[numpy.where(yData == min)]
            res['coords max'] = xData[numpy.where(yData == max)]
            com = numpy.sum(xData * yData).astype(numpy.float32) / numpy.sum(yData).astype(numpy.float32)
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

            data = item.getData(copy=True)
            assert self.plot
            if self._statsOnVisibleData:
                minX, maxX = self.plot.getXAxis().getLimits()
                minY, maxY = self.plot.getYAxis().getLimits()
                originX, originY = item.getOrigin()

                XMinBoundary = int(minX-originX)
                XMaxBoundary = int(maxX-originX)
                YMinBoundary = int(minY-originY)
                YMaxBoundary = int(maxY-originY)

                if XMaxBoundary < 0 or YMaxBoundary < 0:
                    return noDataSelected()
                XMinBoundary = max(XMinBoundary, 0)
                YMinBoundary = max(YMinBoundary, 0)
                data = data[XMinBoundary:XMaxBoundary+1,
                            YMinBoundary:YMaxBoundary+1]

            if data.size is 0:
                return noDataSelected()

            _min, _max = min_max(data)
            res['min'], res['max'], res['delta'] = _min, _max, (_max - _min)
            res['coords min'] = getCoordsFor(_min, data)
            res['coords max'] = getCoordsFor(_max, data)

            com = numpy.sum(data).astype(numpy.float32) / data.size.astype(numpy.float32)
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

            xData, yData, valueData, xerror, yerror = item.getData(copy=True)
            assert self.plot
            if self._statsOnVisibleData:
                minX, maxX = self.plot.getXAxis().getLimits()
                minY, maxY = self.plot.getYAxis().getLimits()
                # filter on X axis
                valueData = valueData[(minX<=xData) & (xData<=maxX)]
                yData = yData[(minX<=xData) & (xData<=maxX)]
                xData = xData[(minX<=xData) & (xData<=maxX)]
                # filter on Y axis
                valueData = valueData[(minY<=yData) & (yData<=maxY)]
                xData = xData[(minY<=yData) & (yData<=maxY)]
                yData = yData[(minY<=yData) & (yData<=maxY)]

            if valueData.size is 0:
                return noDataSelected()
            min, max = min_max(valueData)
            res['min'], res['max'], res['delta'] = min, max, (max - min)
            res['coords min'] = getCoordsFor(xData, yData, valueData, min)
            res['coords max'] = getCoordsFor(xData, yData, valueData, max)
            com = numpy.sum(xData * yData).astype(numpy.float32) / numpy.sum(yData).astype(numpy.float32)
            res['COM'] = com
            res['std'] = numpy.std(valueData)
            res['mean'] = numpy.mean(valueData)
            return res

        def retrieveItems(item, kind):
            items = {}
            itemLegend = self._lgdAndKindToItems[item.getLegend(), kind]
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

        if not item or (item.getLegend(), kind) not in self._lgdAndKindToItems:
            return

        assert (item.getLegend(), kind) in self._lgdAndKindToItems
        assert isinstance(item, self.COMPATIBLE_ITEMS)

        items = retrieveItems(item, kind)

        for itemLabel in self.COLUMNS:
            if itemLabel in ('legend', 'kind'):
                continue
            assert itemLabel in stats
            val = stats[itemLabel]
            if itemLabel in self.FORMATED_COLUMNS:
                val = self.NUMBER_FORMAT.format(val)
            assert items[itemLabel] is not None
            items[itemLabel].setText(str(val))

    def currentChanged(self, current, previous):
        if current.row() > 0:
            legendItem = self.item(current.row(), self.COLUMNS_INDEX['legend'])
            assert legendItem
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

    def setDisplayOnlyActiveItem(self, b):
        if self._displayOnlyActItem != b:
            self._updateItemObserve(switchItemsDisplayedType=True)

    def setStatsOnVisibleData(self, b):
        if self._statsOnVisibleData != b:
            self._statsOnVisibleData = b
            self._updateCurrentStats()

    def _activeItemChanged(self, kind):
        """Callback used when plotting only the active item"""
        assert kind in ('curve', 'image', 'scatter')

        if kind == 'curve':
            item = self.plot.getActiveCurve(just_legend=False)
        elif kind == 'image':
            item = self.plot.getActiveImage(just_legend=False)
        elif kind == 'scatter':
            item = self.plot.getActiveScatter(just_legend=False)
        else:
            raise ValueError('kind not managed')
        if (item.getLegend(), kind) not in self._lgdAndKindToItems:
            self.clear()
            self._addItem(item)
        else:
            self._updateCurrentStats()

    def _plotContentChanged(self, action, kind, legend):
        """Callback used when plotting all the plot items"""
        if kind not in ('curve', 'image', 'scatter'):
            return
        if action == 'add':
            self._addItem(self.plot.getCurve(legend))
        elif action == 'remove':
            self._removeItem(legend, kind)

    def _zoomPlotChanged(self, event):
        if self._statsOnVisibleData is True:
            if 'event' in event and event['event'] == 'limitsChanged':
                self._updateCurrentStats()


class StatsDockWidget(qt.QDockWidget):
    """
    """
    def __init__(self, parent=None, plot=None):
        qt.QDockWidget.__init__(self, parent)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._statsTable = StatsWidget(parent=self, plot=plot)
        self.setWidget(self._statsTable)
