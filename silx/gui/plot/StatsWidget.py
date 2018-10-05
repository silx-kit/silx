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
"""
Module containing widgets displaying stats from items of a plot.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "24/07/2018"


import functools
import logging
import numpy
from collections import OrderedDict

import silx.utils.weakref
from silx.gui import qt
from silx.gui import icons
from silx.gui.plot.items.curve import Curve as CurveItem
from silx.gui.plot.items.histogram import Histogram as HistogramItem
from silx.gui.plot.items.image import ImageBase as ImageItem
from silx.gui.plot.items.scatter import Scatter as ScatterItem
from silx.gui.plot import stats as statsmdl
from silx.gui.widgets.TableWidget import TableWidget
from silx.gui.plot.stats.statshandler import StatsHandler, StatFormatter

logger = logging.getLogger(__name__)


class StatsWidget(qt.QWidget):
    """
    Widget displaying a set of :class:`Stat` to be displayed on a
    :class:`StatsTable` and to be apply on items contained in the :class:`Plot`
    Also contains options to:

    * compute statistics on all the data or on visible data only
    * show statistics of all items or only the active one

    :param parent: Qt parent
    :param plot: the plot containing items on which we want statistics.
    """

    sigVisibilityChanged = qt.Signal(bool)

    NUMBER_FORMAT = '{0:.3f}'

    class OptionsWidget(qt.QToolBar):

        def __init__(self, parent=None):
            qt.QToolBar.__init__(self, parent)
            self.setIconSize(qt.QSize(16, 16))

            action = qt.QAction(self)
            action.setIcon(icons.getQIcon("stats-active-items"))
            action.setText("Active items only")
            action.setToolTip("Display stats for active items only.")
            action.setCheckable(True)
            action.setChecked(True)
            self.__displayActiveItems = action

            action = qt.QAction(self)
            action.setIcon(icons.getQIcon("stats-whole-items"))
            action.setText("All items")
            action.setToolTip("Display stats for all available items.")
            action.setCheckable(True)
            self.__displayWholeItems = action

            action = qt.QAction(self)
            action.setIcon(icons.getQIcon("stats-visible-data"))
            action.setText("Use the visible data range")
            action.setToolTip("Use the visible data range.<br/>"
                              "If activated the data is filtered to only use"
                              "visible data of the plot."
                              "The filtering is a data sub-sampling."
                              "No interpolation is made to fit data to"
                              "boundaries.")
            action.setCheckable(True)
            self.__useVisibleData = action

            action = qt.QAction(self)
            action.setIcon(icons.getQIcon("stats-whole-data"))
            action.setText("Use the full data range")
            action.setToolTip("Use the full data range.")
            action.setCheckable(True)
            action.setChecked(True)
            self.__useWholeData = action

            self.addAction(self.__displayWholeItems)
            self.addAction(self.__displayActiveItems)
            self.addSeparator()
            self.addAction(self.__useVisibleData)
            self.addAction(self.__useWholeData)

            self.itemSelection = qt.QActionGroup(self)
            self.itemSelection.setExclusive(True)
            self.itemSelection.addAction(self.__displayActiveItems)
            self.itemSelection.addAction(self.__displayWholeItems)

            self.dataRangeSelection = qt.QActionGroup(self)
            self.dataRangeSelection.setExclusive(True)
            self.dataRangeSelection.addAction(self.__useWholeData)
            self.dataRangeSelection.addAction(self.__useVisibleData)

        def isActiveItemMode(self):
            return self.itemSelection.checkedAction() is self.__displayActiveItems

        def isVisibleDataRangeMode(self):
            return self.dataRangeSelection.checkedAction() is self.__useVisibleData

    def __init__(self, parent=None, plot=None, stats=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._options = self.OptionsWidget(parent=self)
        self.layout().addWidget(self._options)
        self._statsTable = StatsTable(parent=self, plot=plot)
        self.setStats = self._statsTable.setStats
        self.setStats(stats)

        self.layout().addWidget(self._statsTable)
        self.setPlot = self._statsTable.setPlot

        self._options.itemSelection.triggered.connect(
            self._optSelectionChanged)
        self._options.dataRangeSelection.triggered.connect(
            self._optDataRangeChanged)
        self._optSelectionChanged()
        self._optDataRangeChanged()

        self.setDisplayOnlyActiveItem = self._statsTable.setDisplayOnlyActiveItem
        self.setStatsOnVisibleData = self._statsTable.setStatsOnVisibleData

    def showEvent(self, event):
        self.sigVisibilityChanged.emit(True)
        qt.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self.sigVisibilityChanged.emit(False)
        qt.QWidget.hideEvent(self, event)

    def _optSelectionChanged(self, action=None):
        self._statsTable.setDisplayOnlyActiveItem(self._options.isActiveItemMode())

    def _optDataRangeChanged(self, action=None):
        self._statsTable.setStatsOnVisibleData(self._options.isVisibleDataRangeMode())


class BasicStatsWidget(StatsWidget):
    """
    Widget defining a simple set of :class:`Stat` to be displayed on a
    :class:`StatsWidget`.

    :param parent: Qt parent
    :param plot: the plot containing items on which we want statistics.
    """

    STATS = StatsHandler((
        (statsmdl.StatMin(), StatFormatter()),
        statsmdl.StatCoordMin(),
        (statsmdl.StatMax(), StatFormatter()),
        statsmdl.StatCoordMax(),
        (('std', numpy.std), StatFormatter()),
        (('mean', numpy.mean), StatFormatter()),
        statsmdl.StatCOM()
    ))

    def __init__(self, parent=None, plot=None):
        StatsWidget.__init__(self, parent=parent, plot=plot, stats=self.STATS)


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

    COMPATIBLE_KINDS = {
        'curve': CurveItem,
        'image': ImageItem,
        'scatter': ScatterItem,
        'histogram': HistogramItem
    }

    COMPATIBLE_ITEMS = tuple(COMPATIBLE_KINDS.values())

    def __init__(self, parent=None, plot=None):
        TableWidget.__init__(self, parent)
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
        self._statsHandler = None
        self._legendsSet = []
        """list of legends actually displayed"""
        self._resetColumns()

        self.setColumnCount(len(self._columns))
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setPlot(plot)
        self.setSortingEnabled(True)

    def _resetColumns(self):
        self._columns_index = OrderedDict([('legend', 0), ('kind', 1)])
        self._columns = self._columns_index.keys()
        self.setColumnCount(len(self._columns))

    def setStats(self, statsHandler):
        """

        :param statsHandler: Set the statistics to be displayed and how to
                             format them using
        :rtype: :class:`StatsHandler`
        """
        _statsHandler = statsHandler
        if statsHandler is None:
            _statsHandler = StatsHandler(statFormatters=())
        if isinstance(_statsHandler, (list, tuple)):
            _statsHandler = StatsHandler(_statsHandler)
        assert isinstance(_statsHandler, StatsHandler)
        self._resetColumns()
        self.clear()

        for statName, stat in list(_statsHandler.stats.items()):
            assert isinstance(stat, statsmdl.StatBase)
            self._columns_index[statName] = len(self._columns_index)
        self._statsHandler = _statsHandler
        self._columns = self._columns_index.keys()
        self.setColumnCount(len(self._columns))

        self._updateItemObserve()
        self._updateAllStats()

    def getStatsHandler(self):
        return self._statsHandler

    def _updateAllStats(self):
        for (legend, kind) in self._lgdAndKindToItems:
            self._updateStats(legend, kind)

    @staticmethod
    def _getKind(myItem):
        if isinstance(myItem, CurveItem):
            return 'curve'
        elif isinstance(myItem, ImageItem):
            return 'image'
        elif isinstance(myItem, ScatterItem):
            return 'scatter'
        elif isinstance(myItem, HistogramItem):
            return 'histogram'
        else:
            return None

    def setPlot(self, plot):
        """
        Define the plot to interact with

        :param plot: the plot containing the items on which statistics are
                     applied
        :rtype: :class:`.PlotWidget`
        """
        if self.plot:
            self._dealWithPlotConnection(create=False)
        self.plot = plot
        self.clear()
        if self.plot:
            self._dealWithPlotConnection(create=True)
            self._updateItemObserve()

    def _updateItemObserve(self):
        if self.plot:
            self.clear()
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
                scatters = self.plot._getItems(kind='scatter',
                                               just_legend=False,
                                               withhidden=True)
                [self._addItem(scatter) for scatter in scatters]
                histograms = self.plot._getItems(kind='histogram',
                                                 just_legend=False,
                                                 withhidden=True)
                [self._addItem(histogram) for histogram in histograms]

    def _dealWithPlotConnection(self, create=True):
        """
        Manage connection to plot signals

        Note: connection on Item are managed by the _removeItem function
        """
        if self.plot is None:
            return
        if self._displayOnlyActItem:
            if create is True:
                if self.callbackImage is None:
                    self.callbackImage = functools.partial(self._activeItemChanged, 'image')
                    self.callbackScatter = functools.partial(self._activeItemChanged, 'scatter')
                    self.callbackCurve = functools.partial(self._activeItemChanged, 'curve')
                self.plot.sigActiveImageChanged.connect(self.callbackImage)
                self.plot.sigActiveScatterChanged.connect(self.callbackScatter)
                self.plot.sigActiveCurveChanged.connect(self.callbackCurve)
            else:
                if self.callbackImage is not None:
                    self.plot.sigActiveImageChanged.disconnect(self.callbackImage)
                    self.plot.sigActiveScatterChanged.disconnect(self.callbackScatter)
                    self.plot.sigActiveCurveChanged.disconnect(self.callbackCurve)
                self.callbackImage = None
                self.callbackScatter = None
                self.callbackCurve = None
        else:
            if create is True:
                self.plot.sigContentChanged.connect(self._plotContentChanged)
            else:
                self.plot.sigContentChanged.disconnect(self._plotContentChanged)
        if create is True:
            self.plot.sigPlotSignal.connect(self._zoomPlotChanged)
        else:
            self.plot.sigPlotSignal.disconnect(self._zoomPlotChanged)

    def clear(self):
        """
        Clear all existing items
        """
        lgdsAndKinds = list(self._lgdAndKindToItems.keys())
        for lgdAndKind in lgdsAndKinds:
            self._removeItem(legend=lgdAndKind[0], kind=lgdAndKind[1])
        self._lgdAndKindToItems = {}
        qt.QTableWidget.clear(self)
        self.setRowCount(0)

        # It have to called befor3e accessing to the header items
        self.setHorizontalHeaderLabels(list(self._columns))

        if self._statsHandler is not None:
            for columnId, name in enumerate(self._columns):
                item = self.horizontalHeaderItem(columnId)
                if name in self._statsHandler.stats:
                    stat = self._statsHandler.stats[name]
                    text = stat.name[0].upper() + stat.name[1:]
                    if stat.description is not None:
                        tooltip = stat.description
                    else:
                        tooltip = ""
                else:
                    text = name[0].upper() + name[1:]
                    tooltip = ""
                item.setToolTip(tooltip)
                item.setText(text)

        if hasattr(self.horizontalHeader(), 'setSectionResizeMode'):  # Qt5
            self.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        else:  # Qt4
            self.horizontalHeader().setResizeMode(qt.QHeaderView.ResizeToContents)
        self.setColumnHidden(self._columns_index['kind'], True)

    def _addItem(self, item):
        assert isinstance(item, self.COMPATIBLE_ITEMS)
        if (item.getLegend(), self._getKind(item)) in self._lgdAndKindToItems:
            self._updateStats(item.getLegend(), self._getKind(item))
            return

        self.setRowCount(self.rowCount() + 1)
        indexTable = self.rowCount() - 1
        kind = self._getKind(item)

        self._lgdAndKindToItems[(item.getLegend(), kind)] = {}

        # the get item will manage the item creation of not existing
        _createItem = self._getItem
        for itemName in self._columns:
            _createItem(name=itemName, legend=item.getLegend(), kind=kind,
                        indexTable=indexTable)

        self._updateStats(legend=item.getLegend(), kind=kind)

        callback = functools.partial(
            silx.utils.weakref.WeakMethodProxy(self._updateStats),
            item.getLegend(), kind)
        item.sigItemChanged.connect(callback)
        self.setColumnHidden(self._columns_index['kind'],
                             item.getLegend() not in self._legendsSet)
        self._legendsSet.append(item.getLegend())

    def _getItem(self, name, legend, kind, indexTable):
        if (legend, kind) not in self._lgdAndKindToItems:
            self._lgdAndKindToItems[(legend, kind)] = {}
        if not (name in self._lgdAndKindToItems[(legend, kind)] and
                self._lgdAndKindToItems[(legend, kind)]):
            if name in ('legend', 'kind'):
                _item = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
                if name == 'legend':
                    _item.setText(legend)
                else:
                    assert name == 'kind'
                    _item.setText(kind)
            else:
                if self._statsHandler.formatters[name]:
                    _item = self._statsHandler.formatters[name].tabWidgetItemClass()
                else:
                    _item = qt.QTableWidgetItem()
                tooltip = self._statsHandler.stats[name].getToolTip(kind=kind)
                if tooltip is not None:
                    _item.setToolTip(tooltip)

            _item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
            self.setItem(indexTable, self._columns_index[name], _item)
            self._lgdAndKindToItems[(legend, kind)][name] = _item

        return self._lgdAndKindToItems[(legend, kind)][name]

    def _removeItem(self, legend, kind):
        if (legend, kind) not in self._lgdAndKindToItems or not self.plot:
            return

        self.firstItem = self._lgdAndKindToItems[(legend, kind)]['legend']
        del self._lgdAndKindToItems[(legend, kind)]
        self.removeRow(self.firstItem.row())
        self._legendsSet.remove(legend)
        self.setColumnHidden(self._columns_index['kind'],
                             legend not in self._legendsSet)

    def _updateCurrentStats(self):
        for lgdAndKind in self._lgdAndKindToItems:
            self._updateStats(lgdAndKind[0], lgdAndKind[1])

    def _updateStats(self, legend, kind, event=None):
        if self._statsHandler is None:
            return

        assert kind in ('curve', 'image', 'scatter', 'histogram')
        if kind == 'curve':
            item = self.plot.getCurve(legend)
        elif kind == 'image':
            item = self.plot.getImage(legend)
        elif kind == 'scatter':
            item = self.plot.getScatter(legend)
        elif kind == 'histogram':
            item = self.plot.getHistogram(legend)
        else:
            raise ValueError('kind not managed')

        if not item or (item.getLegend(), kind) not in self._lgdAndKindToItems:
            return

        assert isinstance(item, self.COMPATIBLE_ITEMS)

        statsValDict = self._statsHandler.calculate(item, self.plot,
                                                    self._statsOnVisibleData)

        lgdItem = self._lgdAndKindToItems[(item.getLegend(), kind)]['legend']
        assert lgdItem
        rowStat = lgdItem.row()

        for statName, statVal in list(statsValDict.items()):
            assert statName in self._lgdAndKindToItems[(item.getLegend(), kind)]
            tableItem = self._getItem(name=statName, legend=item.getLegend(),
                                      kind=kind, indexTable=rowStat)
            tableItem.setText(str(statVal))

    def currentChanged(self, current, previous):
        if current.row() >= 0:
            legendItem = self.item(current.row(), self._columns_index['legend'])
            assert legendItem
            kindItem = self.item(current.row(), self._columns_index['kind'])
            kind = kindItem.text()
            if kind == 'curve':
                self.plot.setActiveCurve(legendItem.text())
            elif kind == 'image':
                self.plot.setActiveImage(legendItem.text())
            elif kind == 'scatter':
                self.plot._setActiveItem('scatter', legendItem.text())
            elif kind == 'histogram':
                # active histogram not managed by the plot actually
                pass
            else:
                raise ValueError('kind not managed')
        qt.QTableWidget.currentChanged(self, current, previous)

    def setDisplayOnlyActiveItem(self, displayOnlyActItem):
        """

        :param bool displayOnlyActItem: True if we want to only show active
                                        item
        """
        if self._displayOnlyActItem == displayOnlyActItem:
            return
        self._displayOnlyActItem = displayOnlyActItem
        self._dealWithPlotConnection(create=False)
        self._updateItemObserve()
        self._dealWithPlotConnection(create=True)

    def setStatsOnVisibleData(self, b):
        """
        .. warning:: When visible data is activated we will process to a simple
                     filtering of visible data by the user. The filtering is a
                     simple data sub-sampling. No interpolation is made to fit
                     data to boundaries.

        :param bool b: True if we want to apply statistics only on visible data
        """
        if self._statsOnVisibleData != b:
            self._statsOnVisibleData = b
            self._updateCurrentStats()

    def _activeItemChanged(self, kind, previous, current):
        """Callback used when plotting only the active item"""
        assert kind in ('curve', 'image', 'scatter', 'histogram')
        self._updateItemObserve()

    def _plotContentChanged(self, action, kind, legend):
        """Callback used when plotting all the plot items"""
        if kind not in ('curve', 'image', 'scatter', 'histogram'):
            return
        if kind == 'curve':
            item = self.plot.getCurve(legend)
        elif kind == 'image':
            item = self.plot.getImage(legend)
        elif kind == 'scatter':
            item = self.plot.getScatter(legend)
        elif kind == 'histogram':
            item = self.plot.getHistogram(legend)
        else:
            raise ValueError('kind not managed')

        if action == 'add':
            if item is None:
                raise ValueError('Item from legend "%s" do not exists' % legend)
            self._addItem(item)
        elif action == 'remove':
            self._removeItem(legend, kind)

    def _zoomPlotChanged(self, event):
        if self._statsOnVisibleData is True:
            if 'event' in event and event['event'] == 'limitsChanged':
                self._updateCurrentStats()
