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
"""This module provides widget for displaying statistics relative to a
Region of interest and an item
"""


__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "22/07/2019"


from contextlib import contextmanager
from silx.gui import qt
from silx.gui import icons
from silx.gui.plot.StatsWidget import _StatsWidgetBase, StatsTable, _Container
from silx.gui.plot.StatsWidget import UpdateModeWidget, UpdateMode
from silx.gui.widgets.TableWidget import TableWidget
from silx.gui.plot.items.roi import RegionOfInterest
from silx.gui.plot import items as plotitems
from silx.gui.plot.items.core import ItemChangedType
from silx.gui.plot3d import items as plot3ditems
from silx.gui.plot.CurvesROIWidget import ROI
from silx.gui.plot import stats as statsmdl
from collections import OrderedDict
from silx.utils.proxy import docstring
import silx.gui.plot.items.marker
import silx.gui.plot.items.shape
import functools
import logging

_logger = logging.getLogger(__name__)


class _GetROIItemCoupleDialog(qt.QDialog):
    """
    Dialog used to know which plot item and which roi he wants
    """
    _COMPATIBLE_KINDS = ('curve', 'image', 'scatter', 'histogram')

    def __init__(self, parent=None, plot=None, rois=None):
        qt.QDialog.__init__(self, parent=parent)
        assert plot is not None
        assert rois is not None
        self._plot = plot
        self._rois = rois

        self.setLayout(qt.QVBoxLayout())

        # define the selection widget
        self._selection_widget = qt.QWidget()
        self._selection_widget.setLayout(qt.QHBoxLayout())
        self._kindCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._kindCB)
        self._itemCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._itemCB)
        self._roiCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._roiCB)
        self.layout().addWidget(self._selection_widget)

        # define modal buttons
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self._buttonsModal = qt.QDialogButtonBox(parent=self)
        self._buttonsModal.setStandardButtons(types)
        self.layout().addWidget(self._buttonsModal)
        self._buttonsModal.accepted.connect(self.accept)
        self._buttonsModal.rejected.connect(self.reject)

        # connect signal / slot
        self._kindCB.currentIndexChanged.connect(self._updateValidItemAndRoi)

    def _getCompatibleRois(self, kind):
        """Return compatible rois for the given item kind"""
        def is_compatible(roi, kind):
            if isinstance(roi, RegionOfInterest):
                return kind in ('image', 'scatter')
            elif isinstance(roi, ROI):
                return kind in ('curve', 'histogram')
            else:
                raise ValueError('kind not managed')
        return list(filter(lambda x: is_compatible(x, kind), self._rois))

    def exec(self):
        self._kindCB.clear()
        self._itemCB.clear()
        # filter kind without any items
        self._valid_kinds = {}
        # key is item type, value kinds
        self._valid_rois = {}
        # key is item type, value rois
        self._kind_name_to_roi = {}
        # key is (kind, roi name) value is roi
        self._kind_name_to_item = {}
        # key is (kind, legend name) value is item
        for kind in _GetROIItemCoupleDialog._COMPATIBLE_KINDS:
            def getItems(kind):
                output = []
                for item in self._plot.getItems():
                    type_ = self._plot._itemKind(item)
                    if type_ in kind and item.isVisible():
                        output.append(item)
                return output

            items = getItems(kind=kind)
            rois = self._getCompatibleRois(kind=kind)
            if len(items) > 0 and len(rois) > 0:
                self._valid_kinds[kind] = items
                self._valid_rois[kind] = rois
                for roi in rois:
                    name = roi.getName()
                    self._kind_name_to_roi[(kind, name)] = roi
                for item in items:
                    self._kind_name_to_item[(kind, item.getLegend())] = item

        # filter roi according to kinds
        if len(self._valid_kinds) == 0:
            _logger.warning('no couple item/roi detected for displaying stats')
            return self.reject()

        for kind in self._valid_kinds:
            self._kindCB.addItem(kind)
        self._updateValidItemAndRoi()

        return qt.QDialog.exec(self)

    def exec_(self):  # Qt5 compatibility
        return self.exec()

    def _updateValidItemAndRoi(self, *args, **kwargs):
        self._itemCB.clear()
        self._roiCB.clear()
        kind = self._kindCB.currentText()
        for roi in self._valid_rois[kind]:
            self._roiCB.addItem(roi.getName())
        for item in self._valid_kinds[kind]:
            self._itemCB.addItem(item.getLegend())

    def getROI(self):
        kind = self._kindCB.currentText()
        roi_name = self._roiCB.currentText()
        return self._kind_name_to_roi[(kind, roi_name)]

    def getItem(self):
        kind = self._kindCB.currentText()
        item_name = self._itemCB.currentText()
        return self._kind_name_to_item[(kind, item_name)]


class ROIStatsItemHelper(object):
    """Item utils to associate a plot item and a roi

    Display on one row statistics regarding the couple
    (Item (plot item) / roi).

    :param Item plot_item: item for which we want statistics 
    :param Union[ROI,RegionOfInterest]: region of interest to use for
                                        statistics.
    """
    def __init__(self, plot_item, roi):
        self._plot_item = plot_item
        self._roi = roi

    @property
    def roi(self):
        """roi"""
        return self._roi

    def roi_name(self):
        if isinstance(self._roi, ROI):
            return self._roi.getName()
        elif isinstance(self._roi, RegionOfInterest):
            return self._roi.getName()
        else:
            raise TypeError('Unmanaged roi type')

    @property
    def roi_kind(self):
        """roi class"""
        return self._roi.__class__

    # TODO: should call a util function from the wrapper ?
    def item_kind(self):
        """item kind"""
        if isinstance(self._plot_item, plotitems.Curve):
            return 'curve'
        elif isinstance(self._plot_item, plotitems.ImageData):
            return 'image'
        elif isinstance(self._plot_item, plotitems.Scatter):
            return 'scatter'
        elif isinstance(self._plot_item, plotitems.Histogram):
            return 'histogram'
        elif isinstance(self._plot_item, (plot3ditems.ImageData,
                                          plot3ditems.ScalarField3D)):
            return 'image'
        elif isinstance(self._plot_item, (plot3ditems.Scatter2D,
                                          plot3ditems.Scatter3D)):
            return 'scatter'

    @property
    def item_legend(self):
        """legend of the plot Item"""
        return self._plot_item.getLegend()

    def id_key(self):
        """unique key to represent the couple (item, roi)"""
        return (self.item_kind(), self.item_legend, self.roi_kind,
                self.roi_name())


class _StatsROITable(_StatsWidgetBase, TableWidget):
    """
    Table sued to display some statistics regarding a couple (item/roi)
    """
    _LEGEND_HEADER_DATA = 'legend'

    _KIND_HEADER_DATA = 'kind'

    _ROI_HEADER_DATA = 'roi'

    sigUpdateModeChanged = qt.Signal(object)
    """Signal emitted when the update mode changed"""

    def __init__(self, parent, plot):
        TableWidget.__init__(self, parent)
        _StatsWidgetBase.__init__(self, statsOnVisibleData=False,
                                  displayOnlyActItem=False)
        self.__region_edition_callback = {}
        """We need to keep trace of the roi signals connection because
        the roi emits the sigChanged during roi edition"""
        self._items = {}
        self.setRowCount(0)
        self.setColumnCount(3)

        # Init headers
        headerItem = qt.QTableWidgetItem(self._LEGEND_HEADER_DATA.title())
        headerItem.setData(qt.Qt.UserRole, self._LEGEND_HEADER_DATA)
        self.setHorizontalHeaderItem(0, headerItem)
        headerItem = qt.QTableWidgetItem(self._KIND_HEADER_DATA.title())
        headerItem.setData(qt.Qt.UserRole, self._KIND_HEADER_DATA)
        self.setHorizontalHeaderItem(1, headerItem)
        headerItem = qt.QTableWidgetItem(self._ROI_HEADER_DATA.title())
        headerItem.setData(qt.Qt.UserRole, self._ROI_HEADER_DATA)
        self.setHorizontalHeaderItem(2, headerItem)

        self.setSortingEnabled(True)
        self.setPlot(plot)

        self.__plotItemToItems = {}
        """Key is plotItem, values is list of __RoiStatsItemWidget"""
        self.__roiToItems = {}
        """Key is roi, values is list of __RoiStatsItemWidget"""
        self.__roisKeyToRoi = {}

    def add(self, item):
        assert isinstance(item, ROIStatsItemHelper)
        if item.id_key() in self._items:
            _logger.warning("Item %s is already present", item.id_key())
            return None
        self._items[item.id_key()] = item
        self._addItem(item)
        return item

    def _addItem(self, item):
        """
        Add a _RoiStatsItemWidget item to the table.
        
        :param item: 
        :return: True if successfully added.
        """
        if not isinstance(item, ROIStatsItemHelper):
            # skipped because also receive all new plot item (Marker...) that
            # we don't want to manage in this case.
            return
        # plotItem = item.getItem()
        # roi = item.getROI()
        kind = item.item_kind()
        if kind not in statsmdl.BASIC_COMPATIBLE_KINDS:
            _logger.info("Item has not a supported type: %s", item)
            return False

        # register the roi and the kind
        self._registerPlotItem(item)
        self._registerROI(item)

        # Prepare table items
        tableItems = [
            qt.QTableWidgetItem(),  # Legend
            qt.QTableWidgetItem(),  # Kind
            qt.QTableWidgetItem()]  # roi

        for column in range(3, self.columnCount()):
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
            self._updateStats(item, data_changed=True)

        # Listen for item changes
        # Using queued connection to avoid issue with sender
        # being that of the signal calling the signal
        item._plot_item.sigItemChanged.connect(self._plotItemChanged,
                                               qt.Qt.QueuedConnection)
        return True

    def _removeAllItems(self):
        for row in range(self.rowCount()):
            tableItem = self.item(row, 0)
            # item = self._tableItemToItem(tableItem)
            # item.sigItemChanged.disconnect(self._plotItemChanged)
        self.clearContents()
        self.setRowCount(0)

    def clear(self):
        self._removeAllItems()

    def setStats(self, statsHandler):
        """Set which stats to display and the associated formatting.

        :param StatsHandler statsHandler:
            Set the statistics to be displayed and how to format them using
        """
        self._removeAllItems()
        _StatsWidgetBase.setStats(self, statsHandler)

        self.setRowCount(0)
        self.setColumnCount(len(self._statsHandler.stats) + 3)  # + legend, kind and roi # noqa

        for index, stat in enumerate(self._statsHandler.stats.values()):
            headerItem = qt.QTableWidgetItem(stat.name.capitalize())
            headerItem.setData(qt.Qt.UserRole, stat.name)
            if stat.description is not None:
                headerItem.setToolTip(stat.description)
            self.setHorizontalHeaderItem(3 + index, headerItem)

        horizontalHeader = self.horizontalHeader()
        horizontalHeader.setSectionResizeMode(qt.QHeaderView.ResizeToContents)

        self._updateItemObserve()

    def _updateItemObserve(self, *args):
        pass

    def _dataChanged(self, item):
        pass

    def _updateStats(self, item, data_changed=False, roi_changed=False):
        assert isinstance(item, ROIStatsItemHelper)
        plotItem = item._plot_item
        roi = item._roi
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
            stats = statsHandler.calculate(plotItem, plot,
                                           onlimits=self._statsOnVisibleData,
                                           roi=roi, data_changed=data_changed,
                                           roi_changed=roi_changed)
        else:
            stats = {}

        with self._disableSorting():
            for name, tableItem in self._itemToTableItems(item).items():
                if name == self._LEGEND_HEADER_DATA:
                    text = self._plotWrapper.getLabel(plotItem)
                    tableItem.setText(text)
                elif name == self._KIND_HEADER_DATA:
                    tableItem.setText(self._plotWrapper.getKind(plotItem))
                elif name == self._ROI_HEADER_DATA:
                    name = roi.getName()
                    tableItem.setText(name)
                else:
                    value = stats.get(name)
                    if value is None:
                        _logger.error("Value not found for: %s", name)
                        tableItem.setText('-')
                    else:
                        tableItem.setText(str(value))

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

    def _tableItemToItem(self, tableItem):
        """Find the plot item corresponding to a table item

        :param QTableWidgetItem tableItem:
        :rtype: QObject
        """
        container = tableItem.data(qt.Qt.UserRole)
        return container()

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

    def _plotItemToItems(self, plotItem):
        """Return all _RoiStatsItemWidget associated to the plotItem
        Needed for updating on itemChanged signal
        """
        if plotItem in self.__plotItemToItems:
            return []
        else:
            return self.__plotItemToItems[plotItem]

    def _registerPlotItem(self, item):
        if item._plot_item not in self.__plotItemToItems:
            self.__plotItemToItems[item._plot_item] = set()
        self.__plotItemToItems[item._plot_item].add(item)

    def _roiToItems(self, roi):
        """Return all _RoiStatsItemWidget associated to the roi
        Needed for updating on roiChanged signal
        """
        if roi in self.__roiToItems:
            return []
        else:
            return self.__roiToItems[roi]

    def _registerROI(self, item):
        if item._roi not in self.__roiToItems:
            self.__roiToItems[item._roi] = set()
            # TODO: normalize also sig name
            if isinstance(item._roi, RegionOfInterest):
                # item connection within sigRegionChanged should only be
                # stopped during the region edition
                self.__region_edition_callback[item._roi] = functools.partial(
                    self._updateAllStats, False, True)
                item._roi.sigRegionChanged.connect(self.__region_edition_callback[item._roi])
                item._roi.sigEditingStarted.connect(functools.partial(
                    self._startFiltering, item._roi))
                item._roi.sigEditingFinished.connect(functools.partial(
                    self._endFiltering, item._roi))
            else:
                item._roi.sigChanged.connect(functools.partial(
                    self._updateAllStats, False, True))
        self.__roiToItems[item._roi].add(item)

    def _startFiltering(self, roi):
        roi.sigRegionChanged.disconnect(self.__region_edition_callback[roi])

    def _endFiltering(self, roi):
        roi.sigRegionChanged.connect(self.__region_edition_callback[roi])
        self._updateAllStats(roi_changed=True)

    def unregisterROI(self, roi):
        if roi in self.__roiToItems:
            del self.__roiToItems[roi]
            if isinstance(roi, RegionOfInterest):
                roi.sigRegionEditionStarted.disconnect(functools.partial(
                    self._startFiltering, roi))
                roi.sigRegionEditionFinished.disconnect(functools.partial(
                    self._startFiltering, roi))
                try:
                    roi.sigRegionChanged.disconnect(self._updateAllStats)
                except:
                    pass
            else:
                roi.sigChanged.disconnect(self._updateAllStats)

    def _plotItemChanged(self, event):
        """Handle modifications of the items.

        :param event:
        """
        if event is ItemChangedType.DATA:
            if self.getUpdateMode() is UpdateMode.MANUAL:
                return
            if self._skipPlotItemChangedEvent(event) is True:
                return
            else:
                sender = self.sender()
                for item in self.__plotItemToItems[sender]:
                    # TODO: get all concerned items
                    self._updateStats(item, data_changed=True)
                    # deal with stat items visibility
                    if event is ItemChangedType.VISIBLE:
                        if len(self._itemToTableItems(item).items()) > 0:
                            item_0 = list(self._itemToTableItems(item).values())[0]
                            row_index = item_0.row()
                            self.setRowHidden(row_index, not item.isVisible())

    def _removeItem(self, itemKey):
        if isinstance(itemKey, (silx.gui.plot.items.marker.Marker,
                                silx.gui.plot.items.shape.Shape)):
            return
        if itemKey not in self._items:
            _logger.warning('key not recognized. Won\'t remove any item')
            return
        item = self._items[itemKey]
        row = self._itemToRow(item)
        if row is None:
            kind = self._plotWrapper.getKind(item)
            if kind in statsmdl.BASIC_COMPATIBLE_KINDS:
                _logger.error("Removing item that is not in table: %s", str(item))
            return
        item._plot_item.sigItemChanged.disconnect(self._plotItemChanged)
        self.removeRow(row)
        del self._items[itemKey]

    def _updateAllStats(self, is_request=False, roi_changed=False):
        """Update stats for all rows in the table

        :param bool is_request: True if come from a manual request
        """
        if (self.getUpdateMode() is UpdateMode.MANUAL and
                not is_request and not roi_changed):
            return

        with self._disableSorting():
            for row in range(self.rowCount()):
                tableItem = self.item(row, 0)
                item = self._tableItemToItem(tableItem)
                self._updateStats(item, roi_changed=roi_changed,
                                  data_changed=is_request)

    def _plotCurrentChanged(self, *args):
        pass

    def _getRoi(self, kind, name):
        """return the roi fitting the requirement kind, name. This information
        is enough to be sure it is unique (in the widget)"""
        for roi in self.__roiToItems:
            roiName = roi.getName()
            if isinstance(roi, kind) and name == roiName:
                return roi
        return None

    def _getPlotItem(self, kind, legend):
        """return the plotItem fitting the requirement kind, legend.
        This information is enough to be sure it is unique (in the widget)"""
        for plotItem in self.__plotItemToItems:
            if legend == plotItem.getLegend() and self._plotWrapper.getKind(plotItem) == kind:
                return plotItem
        return None


class ROIStatsWidget(qt.QMainWindow):
    """
    Widget used to define stats item for a couple(roi, plotItem).
    Stats will be computing on a given item (curve, image...) in the given
    region of interest.

    It also provide an interface for adding and removing items.

    .. snapshotqt:: img/ROIStatsWidget.png
        :width: 300px
        :align: center

        from silx.gui import qt
        from silx.gui.plot import Plot2D
        from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
        from silx.gui.plot.items.roi import RectangleROI
        import numpy
        plot = Plot2D()
        plot.addImage(numpy.arange(10000).reshape(100, 100), legend='img')
        plot.show()
        rectangleROI = RectangleROI()
        rectangleROI.setGeometry(origin=(0, 100), size=(20, 20))
        rectangleROI.setName('Initial ROI')
        widget = ROIStatsWidget(plot=plot)
        widget.setStats([('sum', numpy.sum), ('mean', numpy.mean)])
        widget.registerROI(rectangleROI)
        widget.addItem(roi=rectangleROI, plotItem=plot.getImage('img'))
        widget.show()

    :param Union[qt.QWidget,None] parent: parent qWidget
    :param PlotWindow plot: plot widget containing the items
    :param stats: stats to display
    :param tuple rois: tuple of rois to manage
    """

    def __init__(self, parent=None, plot=None, stats=None, rois=None):
        qt.QMainWindow.__init__(self, parent)

        toolbar = qt.QToolBar(self)
        icon = icons.getQIcon('add')
        self._rois = list(rois) if rois is not None else []
        self._addAction = qt.QAction(icon, 'add item/roi', toolbar)
        self._addAction.triggered.connect(self._addRoiStatsItem)
        icon = icons.getQIcon('rm')
        self._removeAction = qt.QAction(icon, 'remove item/roi', toolbar)
        self._removeAction.triggered.connect(self._removeCurrentRow)

        toolbar.addAction(self._addAction)
        toolbar.addAction(self._removeAction)
        self.addToolBar(toolbar)

        self._plot = plot
        self._statsROITable = _StatsROITable(parent=self, plot=self._plot)
        self.setStats(stats=stats)
        self.setCentralWidget(self._statsROITable)
        self.setWindowFlags(qt.Qt.Widget)

        # expose API
        self._setUpdateMode = self._statsROITable.setUpdateMode
        self._updateAllStats = self._statsROITable._updateAllStats

        # setup
        self._statsROITable.setSelectionBehavior(qt.QTableWidget.SelectRows)

    def registerROI(self, roi):
        """For now there is no direct link between roi and plot. That is why
        we need to add/register them to be able to associate them"""
        self._rois.append(roi)

    def setPlot(self, plot):
        """Define the plot to interact with

        :param Union[PlotWidget,SceneWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        self._plot = plot

    def getPlot(self):
        return self._plot

    @docstring(_StatsROITable)
    def setStats(self, stats):
        if stats is not None:
            self._statsROITable.setStats(statsHandler=stats)

    @docstring(_StatsROITable)
    def getStatsHandler(self):
        """
        
        :return: 
        """
        return self._statsROITable.getStatsHandler()

    def _addRoiStatsItem(self):
        """Ask the user what couple ROI / item he want to display"""
        dialog = _GetROIItemCoupleDialog(parent=self, plot=self._plot,
                                         rois=self._rois)
        if dialog.exec():
            self.addItem(roi=dialog.getROI(), plotItem=dialog.getItem())

    def addItem(self, plotItem, roi):
        """
        Add a row of statitstic regarding the couple (plotItem, roi)

        :param Item plotItem: item to use for statistics
        :param roi: region of interest to limit the statistic.
        :type: Union[ROI, RegionOfInterest]
        :return: None of failed to add the item
        :rtype: Union[None,ROIStatsItemHelper]
        """
        statsItem = ROIStatsItemHelper(roi=roi, plot_item=plotItem)
        return self._statsROITable.add(item=statsItem)

    def removeItem(self, plotItem, roi):
        """
        Remove the row associated to the couple (plotItem, roi)

        :param Item plotItem: item to use for statistics
        :param roi: region of interest to limit the statistic.
        :type: Union[ROI,RegionOfInterest]
        """
        statsItem = ROIStatsItemHelper(roi=roi, plot_item=plotItem)
        self._statsROITable._removeItem(itemKey=statsItem.id_key())

    def _removeCurrentRow(self):
        def is1DKind(kind):
            if kind in ('curve', 'histogram', 'scatter'):
                return True
            else:
                return False

        currentRow = self._statsROITable.currentRow()
        item_kind = self._statsROITable.item(currentRow, 1).text()
        item_legend = self._statsROITable.item(currentRow, 0).text()

        roi_name = self._statsROITable.item(currentRow, 2).text()
        roi_kind = ROI if is1DKind(item_kind) else RegionOfInterest
        roi = self._statsROITable._getRoi(kind=roi_kind, name=roi_name)
        if roi is None:
            _logger.warning('failed to retrieve the roi you want to remove')
            return False
        plot_item = self._statsROITable._getPlotItem(kind=item_kind,
                                                     legend=item_legend)
        if plot_item is None:
            _logger.warning('failed to retrieve the plot item you want to'
                            'remove')
            return False
        return self.removeItem(plotItem=plot_item, roi=roi)
