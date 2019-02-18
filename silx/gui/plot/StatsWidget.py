# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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


from collections import OrderedDict
from contextlib import contextmanager
import logging
import weakref

import numpy

from silx.gui import qt
from silx.gui import icons
from silx.gui.plot import stats as statsmdl
from silx.gui.widgets.TableWidget import TableWidget
from silx.gui.plot.stats.statshandler import StatsHandler, StatFormatter
from silx.gui.plot.items.core import ItemChangedType
from silx.gui.widgets.FlowLayout import FlowLayout
from . import PlotWidget
from . import items as plotitems


_logger = logging.getLogger(__name__)


# Helper class to handle specific calls to PlotWidget and SceneWidget

class _Wrapper(qt.QObject):
    """Base class for connection with PlotWidget and SceneWidget.

    This class is used when no PlotWidget or SceneWidget is connected.

    :param plot: The plot to be used
    """

    sigItemAdded = qt.Signal(object)
    """Signal emitted when a new item is added.

    It provides the added item.
    """

    sigItemRemoved = qt.Signal(object)
    """Signal emitted when an item is (about to be) removed.

    It provides the removed item.
    """

    sigCurrentChanged = qt.Signal(object)
    """Signal emitted when the current item has changed.

    It provides the current item.
    """

    sigVisibleDataChanged = qt.Signal()
    """Signal emitted when the visible data area has changed"""

    def __init__(self, plot=None):
        super(_Wrapper, self).__init__(parent=None)
        self._plotRef = None if plot is None else weakref.ref(plot)

    def getPlot(self):
        """Returns the plot attached to this widget"""
        return None if self._plotRef is None else self._plotRef()

    def getItems(self):
        """Returns the list of items in the plot

        :rtype: List[object]
        """
        return ()

    def getSelectedItems(self):
        """Returns the list of selected items in the plot

        :rtype: List[object]
        """
        return ()

    def setCurrentItem(self, item):
        """Set the current/active item in the plot

        :param item: The plot item to set as active/current
        """
        pass

    def getLabel(self, item):
        """Returns the label of the given item.

        :param item:
        :rtype: str
        """
        return ''

    def getKind(self, item):
        """Returns the kind of an item or None if not supported

        :param item:
        :rtype: Union[str,None]
        """
        return None


class _PlotWidgetWrapper(_Wrapper):
    """Class handling PlotWidget specific calls and signal connections

    See :class:`._Wrapper` for documentation

    :param PlotWidget plot:
    """

    def __init__(self, plot):
        assert isinstance(plot, PlotWidget)
        super(_PlotWidgetWrapper, self).__init__(plot)
        plot.sigItemAdded.connect(self.sigItemAdded.emit)
        plot.sigItemAboutToBeRemoved.connect(self.sigItemRemoved.emit)
        plot.sigActiveCurveChanged.connect(self._activeCurveChanged)
        plot.sigActiveImageChanged.connect(self._activeImageChanged)
        plot.sigActiveScatterChanged.connect(self._activeScatterChanged)
        plot.sigPlotSignal.connect(self._limitsChanged)

    def _activeChanged(self, kind):
        """Handle change of active curve/image/scatter"""
        plot = self.getPlot()
        if plot is not None:
            item = plot._getActiveItem(kind=kind)
            if item is None or self.getKind(item) is not None:
                self.sigCurrentChanged.emit(item)

    def _activeCurveChanged(self, previous, current):
        self._activeChanged(kind='curve')

    def _activeImageChanged(self, previous, current):
        self._activeChanged(kind='image')

    def _activeScatterChanged(self, previous, current):
        self._activeChanged(kind='scatter')

    def _limitsChanged(self, event):
        """Handle change of plot area limits."""
        if event['event'] == 'limitsChanged':
                self.sigVisibleDataChanged.emit()

    def getItems(self):
        plot = self.getPlot()
        return () if plot is None else plot._getItems()

    def getSelectedItems(self):
        plot = self.getPlot()
        items = []
        if plot is not None:
            for kind in plot._ACTIVE_ITEM_KINDS:
                item = plot._getActiveItem(kind=kind)
                if item is not None:
                    items.append(item)
        return tuple(items)

    def setCurrentItem(self, item):
        plot = self.getPlot()
        if plot is not None:
            kind = self.getKind(item)
            if kind in plot._ACTIVE_ITEM_KINDS:
                if plot._getActiveItem(kind) != item:
                    plot._setActiveItem(kind, item.getLegend())

    def getLabel(self, item):
        return item.getLegend()

    def getKind(self, item):
        if isinstance(item, plotitems.Curve):
            return 'curve'
        elif isinstance(item, plotitems.ImageData):
            return 'image'
        elif isinstance(item, plotitems.Scatter):
            return 'scatter'
        elif isinstance(item, plotitems.Histogram):
            return 'histogram'
        else:
            return None


class _SceneWidgetWrapper(_Wrapper):
    """Class handling SceneWidget specific calls and signal connections

    See :class:`._Wrapper` for documentation

    :param SceneWidget plot:
    """

    def __init__(self, plot):
        # Lazy-import to avoid circular imports
        from ..plot3d.SceneWidget import SceneWidget

        assert isinstance(plot, SceneWidget)
        super(_SceneWidgetWrapper, self).__init__(plot)
        plot.getSceneGroup().sigItemAdded.connect(self.sigItemAdded)
        plot.getSceneGroup().sigItemRemoved.connect(self.sigItemRemoved)
        plot.selection().sigCurrentChanged.connect(self._currentChanged)
        # sigVisibleDataChanged is never emitted

    def _currentChanged(self, current, previous):
        self.sigCurrentChanged.emit(current)

    def getItems(self):
        plot = self.getPlot()
        return () if plot is None else tuple(plot.getSceneGroup().visit())

    def getSelectedItems(self):
        plot = self.getPlot()
        return () if plot is None else (plot.selection().getCurrentItem(),)

    def setCurrentItem(self, item):
        plot = self.getPlot()
        if plot is not None:
            plot.selection().setCurrentItem(item)

    def getLabel(self, item):
        return item.getLabel()

    def getKind(self, item):
        from ..plot3d import items as plot3ditems

        if isinstance(item, (plot3ditems.ImageData,
                             plot3ditems.ScalarField3D)):
            return 'image'
        elif isinstance(item, (plot3ditems.Scatter2D,
                               plot3ditems.Scatter3D)):
            return 'scatter'
        else:
            return None


class _ScalarFieldViewWrapper(_Wrapper):
    """Class handling ScalarFieldView specific calls and signal connections

    See :class:`._Wrapper` for documentation

    :param SceneWidget plot:
    """

    def __init__(self, plot):
        # Lazy-import to avoid circular imports
        from ..plot3d.ScalarFieldView import ScalarFieldView
        from ..plot3d.items import ScalarField3D

        assert isinstance(plot, ScalarFieldView)
        super(_ScalarFieldViewWrapper, self).__init__(plot)
        self._item = ScalarField3D()
        self._dataChanged()
        plot.sigDataChanged.connect(self._dataChanged)
        # sigItemAdded, sigItemRemoved, sigVisibleDataChanged are never emitted

    def _dataChanged(self):
        plot = self.getPlot()
        if plot is not None:
            self._item.setData(plot.getData(copy=False), copy=False)
            self.sigCurrentChanged.emit(self._item)

    def getItems(self):
        plot = self.getPlot()
        return () if plot is None else (self._item,)

    def getSelectedItems(self):
        return self.getItems()

    def setCurrentItem(self, item):
        pass

    def getLabel(self, item):
        return 'Data'

    def getKind(self, item):
        return 'image'


class _Container(object):
    """Class to contain a plot item.

    This is apparently needed for compatibility with PySide2,

    :param QObject obj:
    """
    def __init__(self, obj):
        self._obj = obj

    def __call__(self):
        return self._obj


class _StatsWidgetBase(object):
    """
    Base class for all widgets which want to display statistics
    """
    def __init__(self, statsOnVisibleData, displayOnlyActItem):
        self._displayOnlyActItem = displayOnlyActItem
        self._statsOnVisibleData = statsOnVisibleData
        self._statsHandler = None

        self.__default_skipped_events = (
            ItemChangedType.ALPHA,
            ItemChangedType.COLOR,
            ItemChangedType.COLORMAP,
            ItemChangedType.SYMBOL,
            ItemChangedType.SYMBOL_SIZE,
            ItemChangedType.LINE_WIDTH,
            ItemChangedType.LINE_STYLE,
            ItemChangedType.LINE_BG_COLOR,
            ItemChangedType.FILL,
            ItemChangedType.HIGHLIGHTED_COLOR,
            ItemChangedType.HIGHLIGHTED_STYLE,
            ItemChangedType.TEXT,
            ItemChangedType.OVERLAY,
            ItemChangedType.VISUALIZATION_MODE,
        )

        self._plotWrapper = _Wrapper()
        self._dealWithPlotConnection(create=True)

    def setPlot(self, plot):
        """Define the plot to interact with

        :param Union[PlotWidget,SceneWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        try:
            import OpenGL
        except ImportError:
            has_opengl = False
        else:
            has_opengl = True
            from ..plot3d.SceneWidget import SceneWidget  # Lazy import
        self._dealWithPlotConnection(create=False)
        self.clear()
        if plot is None:
            self._plotWrapper = _Wrapper()
        elif isinstance(plot, PlotWidget):
            self._plotWrapper = _PlotWidgetWrapper(plot)
        else:
            if has_opengl is True:
                if isinstance(plot, SceneWidget):
                    self._plotWrapper = _SceneWidgetWrapper(plot)
                else:  # Expect a ScalarFieldView
                    self._plotWrapper = _ScalarFieldViewWrapper(plot)
            else:
                _logger.warning('OpenGL not installed, %s not managed' % ('SceneWidget qnd ScalarFieldView'))
        self._dealWithPlotConnection(create=True)

    def setStats(self, statsHandler):
        """Set which stats to display and the associated formatting.

        :param StatsHandler statsHandler:
            Set the statistics to be displayed and how to format them using
        """
        if statsHandler is None:
            statsHandler = StatsHandler(statFormatters=())
        elif isinstance(statsHandler, (list, tuple)):
            statsHandler = StatsHandler(statsHandler)
        assert isinstance(statsHandler, StatsHandler)

        self._statsHandler = statsHandler

    def getStatsHandler(self):
        """Returns the :class:`StatsHandler` in use.

        :rtype: StatsHandler
        """
        return self._statsHandler

    def getPlot(self):
        """Returns the plot attached to this widget

        :rtype: Union[PlotWidget,SceneWidget,None]
        """
        return self._plotWrapper.getPlot()

    def _dealWithPlotConnection(self, create=True):
        """Manage connection to plot signals

        Note: connection on Item are managed by _addItem and _removeItem methods
        """
        connections = []  # List of (signal, slot) to connect/disconnect
        if self._statsOnVisibleData:
            connections.append(
                (self._plotWrapper.sigVisibleDataChanged, self._updateAllStats))

        if self._displayOnlyActItem:
            connections.append(
                (self._plotWrapper.sigCurrentChanged, self._updateItemObserve))
        else:
            connections += [
                (self._plotWrapper.sigItemAdded, self._addItem),
                (self._plotWrapper.sigItemRemoved, self._removeItem),
                (self._plotWrapper.sigCurrentChanged, self._plotCurrentChanged)]

        for signal, slot in connections:
            if create:
                signal.connect(slot)
            else:
                signal.disconnect(slot)

    def _updateItemObserve(self, *args):
        """Reload table depending on mode"""
        raise NotImplementedError('Base class')

    def _updateStats(self, item):
        """Update displayed information for given plot item

        :param item: The plot item
        """
        raise NotImplementedError('Base class')

    def _updateAllStats(self):
        """Update stats for all rows in the table"""
        raise NotImplementedError('Base class')

    def setDisplayOnlyActiveItem(self, displayOnlyActItem):
        """Toggle display off all items or only the active/selected one

        :param bool displayOnlyActItem:
            True if we want to only show active item
        """
        self._displayOnlyActItem = displayOnlyActItem

    def setStatsOnVisibleData(self, b):
        """Toggle computation of statistics on whole data or only visible ones.

        .. warning:: When visible data is activated we will process to a simple
                     filtering of visible data by the user. The filtering is a
                     simple data sub-sampling. No interpolation is made to fit
                     data to boundaries.

        :param bool b: True if we want to apply statistics only on visible data
        """
        if self._statsOnVisibleData != b:
            self._dealWithPlotConnection(create=False)
            self._statsOnVisibleData = b
            self._dealWithPlotConnection(create=True)
            self._updateAllStats()

    def _addItem(self, item):
        """Add a plot item to the table

        If item is not supported, it is ignored.

        :param item: The plot item
        :returns: True if the item is added to the widget.
        :rtype: bool
        """
        raise NotImplementedError('Base class')

    def _removeItem(self, item):
        """Remove table items corresponding to given plot item from the table.

        :param item: The plot item
        """
        raise NotImplementedError('Base class')

    def _plotCurrentChanged(self, current):
        """Handle change of current item and update selection in table

        :param current:
        """
        raise NotImplementedError('Base class')

    def clear(self):
        """clear GUI"""
        pass

    def _skipPlotItemChangedEvent(self, event):
        """

        :param ItemChangedtype event: event to filter or not
        :return: True if we want to ignore this ItemChangedtype
        :rtype: bool
        """
        return event in self.__default_skipped_events


class StatsTable(_StatsWidgetBase, TableWidget):
    """
    TableWidget displaying for each curves contained by the Plot some
    information:

    * legend
    * minimal value
    * maximal value
    * standard deviation (std)

    :param QWidget parent: The widget's parent.
    :param Union[PlotWidget,SceneWidget] plot:
        :class:`PlotWidget` or :class:`SceneWidget` instance on which to operate
    """

    _LEGEND_HEADER_DATA = 'legend'
    _KIND_HEADER_DATA = 'kind'

    def __init__(self, parent=None, plot=None):
        TableWidget.__init__(self, parent)
        _StatsWidgetBase.__init__(self, statsOnVisibleData=False,
                                  displayOnlyActItem=False)

        # Init for _displayOnlyActItem == False
        assert self._displayOnlyActItem is False
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.currentItemChanged.connect(self._currentItemChanged)

        self.setRowCount(0)
        self.setColumnCount(2)

        # Init headers
        headerItem = qt.QTableWidgetItem('Legend')
        headerItem.setData(qt.Qt.UserRole, self._LEGEND_HEADER_DATA)
        self.setHorizontalHeaderItem(0, headerItem)
        headerItem = qt.QTableWidgetItem('Kind')
        headerItem.setData(qt.Qt.UserRole, self._KIND_HEADER_DATA)
        self.setHorizontalHeaderItem(1, headerItem)

        self.setSortingEnabled(True)
        self.setPlot(plot)

    @contextmanager
    def _disableSorting(self):
        """Context manager that disables table sorting

        Previous state is restored when leaving
        """
        sorting = self.isSortingEnabled()
        if sorting:
            self.setSortingEnabled(False)
        yield
        if sorting:
            self.setSortingEnabled(sorting)

    def setStats(self, statsHandler):
        """Set which stats to display and the associated formatting.

        :param StatsHandler statsHandler:
            Set the statistics to be displayed and how to format them using
        """
        self._removeAllItems()
        _StatsWidgetBase.setStats(self, statsHandler)

        self.setRowCount(0)
        self.setColumnCount(len(self._statsHandler.stats) + 2)  # + legend and kind

        for index, stat in enumerate(self._statsHandler.stats.values()):
            headerItem = qt.QTableWidgetItem(stat.name.capitalize())
            headerItem.setData(qt.Qt.UserRole, stat.name)
            if stat.description is not None:
                headerItem.setToolTip(stat.description)
            self.setHorizontalHeaderItem(2 + index, headerItem)

        horizontalHeader = self.horizontalHeader()
        if hasattr(horizontalHeader, 'setSectionResizeMode'):  # Qt5
            horizontalHeader.setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        else:  # Qt4
            horizontalHeader.setResizeMode(qt.QHeaderView.ResizeToContents)

        self._updateItemObserve()

    def setPlot(self, plot):
        """Define the plot to interact with

        :param Union[PlotWidget,SceneWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        _StatsWidgetBase.setPlot(self, plot)
        self._updateItemObserve()

    def clear(self):
        """Define the plot to interact with

        :param Union[PlotWidget,SceneWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        self._removeAllItems()

    def _updateItemObserve(self, *args):
        """Reload table depending on mode"""
        self._removeAllItems()

        # Get selected or all items from the plot
        if self._displayOnlyActItem:  # Only selected
            items = self._plotWrapper.getSelectedItems()
        else:  # All items
            items = self._plotWrapper.getItems()

        # Add items to the plot
        for item in items:
            self._addItem(item)

    def _plotCurrentChanged(self, current):
        """Handle change of current item and update selection in table

        :param current:
        """
        row = self._itemToRow(current)
        if row is None:
            if self.currentRow() >= 0:
                self.setCurrentCell(-1, -1)
        elif row != self.currentRow():
            self.setCurrentCell(row, 0)

    def _tableItemToItem(self, tableItem):
        """Find the plot item corresponding to a table item

        :param QTableWidgetItem tableItem:
        :rtype: QObject
        """
        container = tableItem.data(qt.Qt.UserRole)
        return container()

    def _itemToRow(self, item):
        """Find the row corresponding to a plot item

        :param item: The plot item
        :return: The corresponding row index
        :rtype: Union[int,None]
        """
        for row in range(self.rowCount()):
            tableItem = self.item(row, 0)
            if self._tableItemToItem(tableItem) == item:
                return row
        return None

    def _itemToTableItems(self, item):
        """Find all table items corresponding to a plot item

        :param item: The plot item
        :return: An ordered dict of column name to QTableWidgetItem mapping
            for the given plot item.
        :rtype: OrderedDict
        """
        result = OrderedDict()
        row = self._itemToRow(item)
        if row is not None:
            for column in range(self.columnCount()):
                tableItem = self.item(row, column)
                if self._tableItemToItem(tableItem) != item:
                    _logger.error("Table item/plot item mismatch")
                else:
                    header = self.horizontalHeaderItem(column)
                    name = header.data(qt.Qt.UserRole)
                    result[name] = tableItem
        return result

    def _plotItemChanged(self, event):
        """Handle modifications of the items.

        :param event:
        """
        if self._skipPlotItemChangedEvent(event) is True:
            return
        else:
            item = self.sender()
            self._updateStats(item)

    def _addItem(self, item):
        """Add a plot item to the table

        If item is not supported, it is ignored.

        :param item: The plot item
        :returns: True if the item is added to the widget.
        :rtype: bool
        """
        if self._itemToRow(item) is not None:
            _logger.info("Item already present in the table")
            self._updateStats(item)
            return True

        kind = self._plotWrapper.getKind(item)
        if kind not in statsmdl.BASIC_COMPATIBLE_KINDS:
            _logger.info("Item has not a supported type: %s", item)
            return False

        # Prepare table items
        tableItems = [
            qt.QTableWidgetItem(),  # Legend
            qt.QTableWidgetItem()]  # Kind

        for column in range(2, self.columnCount()):
            header = self.horizontalHeaderItem(column)
            name = header.data(qt.Qt.UserRole)

            formatter = self._statsHandler.formatters[name]
            if formatter:
                tableItem = formatter.tabWidgetItemClass()
            else:
                tableItem = qt.QTableWidgetItem()

            tooltip = self._statsHandler.stats[name].getToolTip(kind=kind)
            if tooltip is not None:
                tableItem.setToolTip(tooltip)

            tableItems.append(tableItem)

        # Disable sorting while adding table items
        with self._disableSorting():
            # Add a row to the table
            self.setRowCount(self.rowCount() + 1)

            # Add table items to the last row
            row = self.rowCount() - 1
            for column, tableItem in enumerate(tableItems):
                tableItem.setData(qt.Qt.UserRole, _Container(item))
                tableItem.setFlags(
                    qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                self.setItem(row, column, tableItem)

            # Update table items content
            self._updateStats(item)

        # Listen for item changes
        # Using queued connection to avoid issue with sender
        # being that of the signal calling the signal
        item.sigItemChanged.connect(self._plotItemChanged,
                                    qt.Qt.QueuedConnection)

        return True

    def _removeItem(self, item):
        """Remove table items corresponding to given plot item from the table.

        :param item: The plot item
        """
        row = self._itemToRow(item)
        if row is None:
            kind = self._plotWrapper.getKind(item)
            if kind in statsmdl.BASIC_COMPATIBLE_KINDS:
                _logger.error("Removing item that is not in table: %s", str(item))
            return
        item.sigItemChanged.disconnect(self._plotItemChanged)
        self.removeRow(row)

    def _removeAllItems(self):
        """Remove content of the table"""
        for row in range(self.rowCount()):
            tableItem = self.item(row, 0)
            item = self._tableItemToItem(tableItem)
            item.sigItemChanged.disconnect(self._plotItemChanged)
        self.clearContents()
        self.setRowCount(0)

    def _updateStats(self, item):
        """Update displayed information for given plot item

        :param item: The plot item
        """
        if item is None:
            return
        plot = self.getPlot()
        if plot is None:
            _logger.info("Plot not available")
            return

        row = self._itemToRow(item)
        if row is None:
            _logger.error("This item is not in the table: %s", str(item))
            return

        statsHandler = self.getStatsHandler()
        if statsHandler is not None:
            stats = statsHandler.calculate(
                item, plot, self._statsOnVisibleData)
        else:
            stats = {}

        with self._disableSorting():
            for name, tableItem in self._itemToTableItems(item).items():
                if name == self._LEGEND_HEADER_DATA:
                    text = self._plotWrapper.getLabel(item)
                    tableItem.setText(text)
                elif name == self._KIND_HEADER_DATA:
                    tableItem.setText(self._plotWrapper.getKind(item))
                else:
                    value = stats.get(name)
                    if value is None:
                        _logger.error("Value not found for: %s", name)
                        tableItem.setText('-')
                    else:
                        tableItem.setText(str(value))

    def _updateAllStats(self):
        """Update stats for all rows in the table"""
        with self._disableSorting():
            for row in range(self.rowCount()):
                tableItem = self.item(row, 0)
                item = self._tableItemToItem(tableItem)
                self._updateStats(item)

    def _currentItemChanged(self, current, previous):
        """Handle change of selection in table and sync plot selection

        :param QTableWidgetItem current:
        :param QTableWidgetItem previous:
        """
        if current and current.row() >= 0:
            item = self._tableItemToItem(current)
            self._plotWrapper.setCurrentItem(item)

    def setDisplayOnlyActiveItem(self, displayOnlyActItem):
        """Toggle display off all items or only the active/selected one

        :param bool displayOnlyActItem:
            True if we want to only show active item
        """
        if self._displayOnlyActItem == displayOnlyActItem:
            return
        self._dealWithPlotConnection(create=False)
        if not self._displayOnlyActItem:
            self.currentItemChanged.disconnect(self._currentItemChanged)

        _StatsWidgetBase.setDisplayOnlyActiveItem(self, displayOnlyActItem)

        self._updateItemObserve()
        self._dealWithPlotConnection(create=True)

        if not self._displayOnlyActItem:
            self.currentItemChanged.connect(self._currentItemChanged)
            self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        else:
            self.setSelectionMode(qt.QAbstractItemView.NoSelection)


class _OptionsWidget(qt.QToolBar):

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

    def setVisibleDataRangeModeEnabled(self, enabled):
        """Enable/Disable the visible data range mode

        :param bool enabled: True to allow user to choose
            stats on visible data
        """
        self.__useVisibleData.setEnabled(enabled)
        if not enabled:
            self.__useWholeData.setChecked(True)


class StatsWidget(qt.QWidget):
    """
    Widget displaying a set of :class:`Stat` to be displayed on a
    :class:`StatsTable` and to be apply on items contained in the :class:`Plot`
    Also contains options to:

    * compute statistics on all the data or on visible data only
    * show statistics of all items or only the active one

    :param QWidget parent: Qt parent
    :param Union[PlotWidget,SceneWidget] plot:
        The plot containing items on which we want statistics.
    :param StatsHandler stats:
        Set the statistics to be displayed and how to format them using
    """

    sigVisibilityChanged = qt.Signal(bool)
    """Signal emitted when the visibility of this widget changes.

    It Provides the visibility of the widget.
    """

    NUMBER_FORMAT = '{0:.3f}'

    def __init__(self, parent=None, plot=None, stats=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._options = _OptionsWidget(parent=self)
        self.layout().addWidget(self._options)
        self._statsTable = StatsTable(parent=self, plot=plot)
        self.setStats(stats)

        self.layout().addWidget(self._statsTable)

        self._options.itemSelection.triggered.connect(
            self._optSelectionChanged)
        self._options.dataRangeSelection.triggered.connect(
            self._optDataRangeChanged)
        self._optSelectionChanged()
        self._optDataRangeChanged()

    def _getStatsTable(self):
        """Returns the :class:`StatsTable` used by this widget.

        :rtype: StatsTable
        """
        return self._statsTable

    def showEvent(self, event):
        self.sigVisibilityChanged.emit(True)
        qt.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self.sigVisibilityChanged.emit(False)
        qt.QWidget.hideEvent(self, event)

    def _optSelectionChanged(self, action=None):
        self._getStatsTable().setDisplayOnlyActiveItem(
            self._options.isActiveItemMode())

    def _optDataRangeChanged(self, action=None):
        self._getStatsTable().setStatsOnVisibleData(
            self._options.isVisibleDataRangeMode())

    # Proxy methods

    def setStats(self, statsHandler):
        return self._getStatsTable().setStats(statsHandler=statsHandler)

    setStats.__doc__ = StatsTable.setStats.__doc__

    def setPlot(self, plot):
        self._options.setVisibleDataRangeModeEnabled(
            plot is None or isinstance(plot, PlotWidget))
        return self._getStatsTable().setPlot(plot=plot)

    setPlot.__doc__ = StatsTable.setPlot.__doc__

    def getPlot(self):
        return self._getStatsTable().getPlot()

    getPlot.__doc__ = StatsTable.getPlot.__doc__

    def setDisplayOnlyActiveItem(self, displayOnlyActItem):
        return self._getStatsTable().setDisplayOnlyActiveItem(
            displayOnlyActItem=displayOnlyActItem)

    setDisplayOnlyActiveItem.__doc__ = StatsTable.setDisplayOnlyActiveItem.__doc__

    def setStatsOnVisibleData(self, b):
        return self._getStatsTable().setStatsOnVisibleData(b=b)

    setStatsOnVisibleData.__doc__ = StatsTable.setStatsOnVisibleData.__doc__


DEFAULT_STATS = StatsHandler((
    (statsmdl.StatMin(), StatFormatter()),
    statsmdl.StatCoordMin(),
    (statsmdl.StatMax(), StatFormatter()),
    statsmdl.StatCoordMax(),
    statsmdl.StatCOM(),
    (('mean', numpy.mean), StatFormatter()),
    (('std', numpy.std), StatFormatter()),
))


class BasicStatsWidget(StatsWidget):
    """
    Widget defining a simple set of :class:`Stat` to be displayed on a
    :class:`StatsWidget`.

    :param QWidget parent: Qt parent
    :param PlotWidget plot:
        The plot containing items on which we want statistics.
    :param StatsHandler stats:
        Set the statistics to be displayed and how to format them using

    .. snapshotqt:: img/BasicStatsWidget.png
     :width: 300px
     :align: center

     from silx.gui.plot import Plot1D
     from silx.gui.plot.StatsWidget import BasicStatsWidget
     
     plot = Plot1D()
     x = range(100)
     y = x
     plot.addCurve(x, y, legend='curve_0')
     plot.setActiveCurve('curve_0')
     
     widget = BasicStatsWidget(plot=plot)
     widget.show()
    """
    def __init__(self, parent=None, plot=None):
        StatsWidget.__init__(self, parent=parent, plot=plot,
                             stats=DEFAULT_STATS)


class _BaseLineStatsWidget(_StatsWidgetBase, qt.QWidget):
    """
    Widget made to display stats into a QLayout with for all stat a couple
     (QLabel, QLineEdit) created.
     The the layout can be defined prior of adding any statistic.

    :param QWidget parent: Qt parent
    :param Union[PlotWidget,SceneWidget] plot:
        The plot containing items on which we want statistics.
    :param str kind: the kind of plotitems we want to display
    :param StatsHandler stats:
        Set the statistics to be displayed and how to format them using
    :param bool statsOnVisibleData: compute statistics for the whole data or
                                    only visible ones.
    """

    def __init__(self, parent=None, plot=None, kind='curve', stats=None,
                 statsOnVisibleData=False):
        self._item_kind = kind
        """The item displayed"""
        self._statQlineEdit = {}
        """list of legends actually displayed"""
        self._n_statistics_per_line = 4
        """number of statistics displayed per line in the grid layout"""
        qt.QWidget.__init__(self, parent)
        _StatsWidgetBase.__init__(self,
                                  statsOnVisibleData=statsOnVisibleData,
                                  displayOnlyActItem=True)
        self.setLayout(self._createLayout())
        self.setPlot(plot)
        if stats is not None:
            self.setStats(stats)

    def _addItemForStatistic(self, statistic):
        assert isinstance(statistic, statsmdl.StatBase)
        assert statistic.name in self._statsHandler.stats

        self.layout().setSpacing(2)
        self.layout().setContentsMargins(2, 2, 2, 2)

        if isinstance(self.layout(), qt.QGridLayout):
            parent = self
        else:
            widget = qt.QWidget(parent=self)
            parent = widget

        qLabel = qt.QLabel(statistic.name + ':', parent=parent)
        qLineEdit = qt.QLineEdit('', parent=parent)
        qLineEdit.setReadOnly(True)

        self._addStatsWidgetsToLayout(qLabel=qLabel, qLineEdit=qLineEdit)
        self._statQlineEdit[statistic.name] = qLineEdit

    def setPlot(self, plot):
        """Define the plot to interact with

        :param Union[PlotWidget,SceneWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        _StatsWidgetBase.setPlot(self, plot)
        self._updateAllStats()

    def _addStatsWidgetsToLayout(self, qLabel, qLineEdit):
        raise NotImplementedError('Base class')

    def setStats(self, statsHandler):
        """Set which stats to display and the associated formatting.
        :param StatsHandler statsHandler:
            Set the statistics to be displayed and how to format them using
        """
        _StatsWidgetBase.setStats(self, statsHandler)
        for statName, stat in list(self._statsHandler.stats.items()):
            self._addItemForStatistic(stat)
        self._updateAllStats()

    def _activeItemChanged(self, kind, previous, current):
        if kind == self._item_kind:
            self._updateAllStats()

    def _updateAllStats(self):
        plot = self.getPlot()
        if plot is not None:
            _items = self._plotWrapper.getSelectedItems()
            def kind_filter(_item):
                return self._plotWrapper.getKind(_item) == self.getKind()

            items = list(filter(kind_filter, _items))
            assert len(items) in (0, 1)
            if len(items) is 1:
                self._setItem(items[0])

    def setKind(self, kind):
        """Change the kind of active item to display
        :param str kind: kind of item to display information for ('curve' ...)
        """
        if self._item_kind != kind:
            self._item_kind = kind
            self._updateItemObserve()

    def getKind(self):
        """
        :return: kind of item we want to compute statistic for
         :rtype: str
        """
        return self._item_kind

    def _setItem(self, item):
        if item is None:
            for stat_name, stat_widget in self._statQlineEdit.items():
                stat_widget.setText('')
        elif (self._statsHandler is not None and len(
                self._statsHandler.stats) > 0):
            plot = self.getPlot()
            if plot is not None:
                statsValDict = self._statsHandler.calculate(item,
                                                            plot,
                                                            self._statsOnVisibleData)
                for statName, statVal in list(statsValDict.items()):
                    self._statQlineEdit[statName].setText(statVal)

    def _updateItemObserve(self, *argv):
        assert self._displayOnlyActItem
        _items = self._plotWrapper.getSelectedItems()
        def kind_filter(_item):
            return self._plotWrapper.getKind(_item) == self.getKind()
        items = list(filter(kind_filter, _items))
        assert len(items) in (0, 1)
        _item = items[0] if len(items) is 1 else None
        self._setItem(_item)

    def _createLayout(self):
        """create an instance of the main QLayout"""
        raise NotImplementedError('Base class')

    def _addItem(self, item):
        raise NotImplementedError('Display only the active item')

    def _removeItem(self, item):
        raise NotImplementedError('Display only the active item')

    def _plotCurrentChanged(selfself, current):
        raise NotImplementedError('Display only the active item')


class BasicLineStatsWidget(_BaseLineStatsWidget):
    """
    Widget defining a simple set of :class:`Stat` to be displayed on a
    :class:`LineStatsWidget`.

    :param QWidget parent: Qt parent
    :param Union[PlotWidget,SceneWidget] plot:
        The plot containing items on which we want statistics.
    :param str kind: the kind of plotitems we want to display
    :param StatsHandler stats:
        Set the statistics to be displayed and how to format them using
    :param bool statsOnVisibleData: compute statistics for the whole data or
                                    only visible ones.
    """

    def __init__(self, parent=None, plot=None, kind='curve',
                 stats=DEFAULT_STATS, statsOnVisibleData=False):
        _BaseLineStatsWidget.__init__(self, parent=parent, kind=kind,
                                      plot=plot, stats=stats,
                                      statsOnVisibleData=statsOnVisibleData)

    def _createLayout(self):
        return FlowLayout()

    def _addStatsWidgetsToLayout(self, qLabel, qLineEdit):
        # create a mother widget to make sure both qLabel & qLineEdit will
        # always be displayed side by side
        widget = qt.QWidget(parent=self)
        widget.setLayout(qt.QHBoxLayout())
        widget.layout().setSpacing(0)
        widget.layout().setContentsMargins(0, 0, 0, 0)

        widget.layout().addWidget(qLabel)
        widget.layout().addWidget(qLineEdit)

        self.layout().addWidget(widget)


class BasicGridStatsWidget(_BaseLineStatsWidget):
    """
    pymca design like widget
    
    :param QWidget parent: Qt parent
    :param Union[PlotWidget,SceneWidget] plot:
        The plot containing items on which we want statistics.
    :param StatsHandler stats:
        Set the statistics to be displayed and how to format them using
    :param str kind: the kind of plotitems we want to display
    :param bool statsOnVisibleData: compute statistics for the whole data or
                                    only visible ones.
    :param int statsPerLine: number of statistic to be displayed per line

    .. snapshotqt:: img/BasicGridStatsWidget.png
     :width: 600px
     :align: center

     from silx.gui.plot import Plot1D
     from silx.gui.plot.StatsWidget import BasicGridStatsWidget
     
     plot = Plot1D()
     x = range(100)
     y = x
     plot.addCurve(x, y, legend='curve_0')
     plot.setActiveCurve('curve_0')
     
     widget = BasicGridStatsWidget(plot=plot, kind='curve')
     widget.show()
    """

    def __init__(self, parent=None, plot=None, kind='curve',
                 stats=DEFAULT_STATS, statsOnVisibleData=False,
                 statsPerLine=4):
        _BaseLineStatsWidget.__init__(self, parent=parent, kind=kind,
                                      plot=plot, stats=stats,
                                      statsOnVisibleData=statsOnVisibleData)
        self._n_statistics_per_line = statsPerLine

    def _addStatsWidgetsToLayout(self, qLabel, qLineEdit):
        column = len(self._statQlineEdit) % self._n_statistics_per_line
        row = len(self._statQlineEdit) // self._n_statistics_per_line
        self.layout().addWidget(qLabel, row, column * 2)
        self.layout().addWidget(qLineEdit, row, column * 2 + 1)

    def _createLayout(self):
        return qt.QGridLayout()
