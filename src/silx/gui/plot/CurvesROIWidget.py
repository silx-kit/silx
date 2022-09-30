# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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
Widget to handle regions of interest (:class:`ROI`) on curves displayed in a
:class:`PlotWindow`.

This widget is meant to work with :class:`PlotWindow`.
"""

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "13/03/2018"

from collections import OrderedDict
import logging
import os
import sys
import functools
import numpy
from silx.io import dictdump
from silx.utils import deprecation
from silx.utils.weakref import WeakMethodProxy
from silx.utils.proxy import docstring
from .. import icons, qt
from silx.math.combo import min_max
import weakref
from silx.gui.widgets.TableWidget import TableWidget
from . import items
from .items.roi import _RegionOfInterestBase


_logger = logging.getLogger(__name__)


class CurvesROIWidget(qt.QWidget):
    """
    Widget displaying a table of ROI information.

    Implements also the following behavior:

    * if the roiTable has no ROI when showing create the default ICR one

    :param parent: See :class:`QWidget`
    :param str name: The title of this widget
    """

    sigROIWidgetSignal = qt.Signal(object)
    """Signal of ROIs modifications.

    Modification information if given as a dict with an 'event' key
    providing the type of events.

    Type of events:

    - AddROI, DelROI, LoadROI and ResetROI with keys: 'roilist', 'roidict'
    - selectionChanged with keys: 'row', 'col' 'roi', 'key', 'colheader',
      'rowheader'
    """

    sigROISignal = qt.Signal(object)

    def __init__(self, parent=None, name=None, plot=None):
        super(CurvesROIWidget, self).__init__(parent)
        if name is not None:
            self.setWindowTitle(name)
        self.__lastSigROISignal = None
        """Store the last value emitted for the sigRoiSignal. In the case the
        active curve change we need to add this extra step in order to make
        sure we won't send twice the sigROISignal.
        This come from the fact sigROISignal is connected to the 
        activeROIChanged signal which is emitted when raw and net counts
        values are changing but are not embed in the sigROISignal.
        """
        assert plot is not None
        self._plotRef = weakref.ref(plot)
        self._showAllMarkers = False
        self.currentROI = None

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.headerLabel = qt.QLabel(self)
        self.headerLabel.setAlignment(qt.Qt.AlignHCenter)
        self.setHeader()
        layout.addWidget(self.headerLabel)

        widgetAllCheckbox = qt.QWidget(parent=self)
        self._showAllCheckBox = qt.QCheckBox("show all ROI",
                                             parent=widgetAllCheckbox)
        widgetAllCheckbox.setLayout(qt.QHBoxLayout())
        spacer = qt.QWidget(parent=widgetAllCheckbox)
        spacer.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        widgetAllCheckbox.layout().addWidget(spacer)
        widgetAllCheckbox.layout().addWidget(self._showAllCheckBox)
        layout.addWidget(widgetAllCheckbox)

        self.roiTable = ROITable(self, plot=plot)
        rheight = self.roiTable.horizontalHeader().sizeHint().height()
        self.roiTable.setMinimumHeight(4 * rheight)
        layout.addWidget(self.roiTable)
        self._roiFileDir = qt.QDir.home().absolutePath()
        self._showAllCheckBox.toggled.connect(self.roiTable.showAllMarkers)

        hbox = qt.QWidget(self)
        hboxlayout = qt.QHBoxLayout(hbox)
        hboxlayout.setContentsMargins(0, 0, 0, 0)
        hboxlayout.setSpacing(0)

        hboxlayout.addStretch(0)

        self.addButton = qt.QPushButton(hbox)
        self.addButton.setText("Add ROI")
        self.addButton.setToolTip('Create a new ROI')
        self.delButton = qt.QPushButton(hbox)
        self.delButton.setText("Delete ROI")
        self.addButton.setToolTip('Remove the selected ROI')
        self.resetButton = qt.QPushButton(hbox)
        self.resetButton.setText("Reset")
        self.addButton.setToolTip('Clear all created ROIs. We only let the '
                                  'default ROI')

        hboxlayout.addWidget(self.addButton)
        hboxlayout.addWidget(self.delButton)
        hboxlayout.addWidget(self.resetButton)

        hboxlayout.addStretch(0)

        self.loadButton = qt.QPushButton(hbox)
        self.loadButton.setText("Load")
        self.loadButton.setToolTip('Load ROIs from a .ini file')
        self.saveButton = qt.QPushButton(hbox)
        self.saveButton.setText("Save")
        self.loadButton.setToolTip('Save ROIs to a .ini file')
        hboxlayout.addWidget(self.loadButton)
        hboxlayout.addWidget(self.saveButton)
        layout.setStretchFactor(self.headerLabel, 0)
        layout.setStretchFactor(self.roiTable, 1)
        layout.setStretchFactor(hbox, 0)

        layout.addWidget(hbox)

        # Signal / Slot connections
        self.addButton.clicked.connect(self._add)
        self.delButton.clicked.connect(self._del)
        self.resetButton.clicked.connect(self._reset)

        self.loadButton.clicked.connect(self._load)
        self.saveButton.clicked.connect(self._save)

        self.roiTable.activeROIChanged.connect(self._emitCurrentROISignal)

        self._isConnected = False  # True if connected to plot signals
        self._isInit = False

    def getROIListAndDict(self):
        return self.roiTable.getROIListAndDict()

    def getPlotWidget(self):
        """Returns the associated PlotWidget or None

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def showEvent(self, event):
        self._visibilityChangedHandler(visible=True)
        qt.QWidget.showEvent(self, event)

    @property
    def roiFileDir(self):
        """The directory from which to load/save ROI from/to files."""
        if not os.path.isdir(self._roiFileDir):
            self._roiFileDir = qt.QDir.home().absolutePath()
        return self._roiFileDir

    @roiFileDir.setter
    def roiFileDir(self, roiFileDir):
        self._roiFileDir = str(roiFileDir)

    def setRois(self, rois, order=None):
        return self.roiTable.setRois(rois, order)

    def getRois(self, order=None):
        return self.roiTable.getRois(order)

    def setMiddleROIMarkerFlag(self, flag=True):
        return self.roiTable.setMiddleROIMarkerFlag(flag)

    def _add(self):
        """Add button clicked handler"""
        def getNextRoiName():
            rois = self.roiTable.getRois(order=None)
            roisNames = []
            [roisNames.append(roiName) for roiName in rois]
            nrois = len(rois)
            if nrois == 0:
                return "ICR"
            else:
                i = 1
                newroi = "newroi %d" % i
                while newroi in roisNames:
                    i += 1
                    newroi = "newroi %d" % i
                return newroi
        roi = ROI(name=getNextRoiName())

        if roi.getName() == "ICR":
            roi.setType("Default")
        else:
            roi.setType(self.getPlotWidget().getXAxis().getLabel())

        xmin, xmax = self.getPlotWidget().getXAxis().getLimits()
        fromdata = xmin + 0.25 * (xmax - xmin)
        todata = xmin + 0.75 * (xmax - xmin)
        if roi.isICR():
            fromdata, dummy0, todata, dummy1 = self._getAllLimits()
        roi.setFrom(fromdata)
        roi.setTo(todata)
        self.roiTable.addRoi(roi)

        # back compatibility pymca roi signals
        ddict = {}
        ddict['event'] = "AddROI"
        ddict['roilist'] = self.roiTable.roidict.values()
        ddict['roidict'] = self.roiTable.roidict
        self.sigROIWidgetSignal.emit(ddict)
        # end back compatibility pymca roi signals

    def _del(self):
        """Delete button clicked handler"""
        self.roiTable.deleteActiveRoi()

        # back compatibility pymca roi signals
        ddict = {}
        ddict['event'] = "DelROI"
        ddict['roilist'] = self.roiTable.roidict.values()
        ddict['roidict'] = self.roiTable.roidict
        self.sigROIWidgetSignal.emit(ddict)
        # end back compatibility pymca roi signals

    def _reset(self):
        """Reset button clicked handler"""
        self.roiTable.clear()
        old = self.blockSignals(True)  # avoid several sigROISignal emission
        self._add()
        self.blockSignals(old)

        # back compatibility pymca roi signals
        ddict = {}
        ddict['event'] = "ResetROI"
        ddict['roilist'] = self.roiTable.roidict.values()
        ddict['roidict'] = self.roiTable.roidict
        self.sigROIWidgetSignal.emit(ddict)
        # end back compatibility pymca roi signals

    def _load(self):
        """Load button clicked handler"""
        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(
            ['INI File  *.ini', 'JSON File *.json', 'All *.*'])
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        dialog.setDirectory(self.roiFileDir)
        if not dialog.exec():
            dialog.close()
            return

        # pyflakes bug http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=666494
        outputFile = dialog.selectedFiles()[0]
        dialog.close()

        self.roiFileDir = os.path.dirname(outputFile)
        self.roiTable.load(outputFile)

        # back compatibility pymca roi signals
        ddict = {}
        ddict['event'] = "LoadROI"
        ddict['roilist'] = self.roiTable.roidict.values()
        ddict['roidict'] = self.roiTable.roidict
        self.sigROIWidgetSignal.emit(ddict)
        # end back compatibility pymca roi signals

    def load(self, filename):
        """Load ROI widget information from a file storing a dict of ROI.

        :param str filename: The file from which to load ROI
        """
        self.roiTable.load(filename)

    def _save(self):
        """Save button clicked handler"""
        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(['INI File  *.ini', 'JSON File *.json'])
        dialog.setFileMode(qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setDirectory(self.roiFileDir)
        if not dialog.exec():
            dialog.close()
            return

        outputFile = dialog.selectedFiles()[0]
        extension = '.' + dialog.selectedNameFilter().split('.')[-1]
        dialog.close()

        if not outputFile.endswith(extension):
            outputFile += extension

        if os.path.exists(outputFile):
            try:
                os.remove(outputFile)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec()
                return
        self.roiFileDir = os.path.dirname(outputFile)
        self.save(outputFile)

    def save(self, filename):
        """Save current ROIs of the widget as a dict of ROI to a file.

        :param str filename: The file to which to save the ROIs
        """
        self.roiTable.save(filename)

    def setHeader(self, text='ROIs'):
        """Set the header text of this widget"""
        self.headerLabel.setText("<b>%s<\b>" % text)

    @deprecation.deprecated(replacement="calculateRois",
                            reason="CamelCase convention",
                            since_version="0.7")
    def calculateROIs(self, *args, **kw):
        self.calculateRois(*args, **kw)

    def calculateRois(self, roiList=None, roiDict=None):
        """Compute ROI information"""
        return self.roiTable.calculateRois()

    def showAllMarkers(self, _show=True):
        self.roiTable.showAllMarkers(_show)

    def _getAllLimits(self):
        """Retrieve the limits based on the curves."""
        plot = self.getPlotWidget()
        curves = () if plot is None else plot.getAllCurves()
        if not curves:
            return 1.0, 1.0, 100., 100.

        xmin, ymin = None, None
        xmax, ymax = None, None

        for curve in curves:
            x = curve.getXData(copy=False)
            y = curve.getYData(copy=False)
            if xmin is None:
                xmin = x.min()
            else:
                xmin = min(xmin, x.min())
            if xmax is None:
                xmax = x.max()
            else:
                xmax = max(xmax, x.max())
            if ymin is None:
                ymin = y.min()
            else:
                ymin = min(ymin, y.min())
            if ymax is None:
                ymax = y.max()
            else:
                ymax = max(ymax, y.max())

        return xmin, ymin, xmax, ymax

    def showEvent(self, event):
        self._visibilityChangedHandler(visible=True)
        qt.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self._visibilityChangedHandler(visible=False)
        qt.QWidget.hideEvent(self, event)

    def _visibilityChangedHandler(self, visible):
        """Handle widget's visibility updates.

        It is connected to plot signals only when visible.
        """
        if visible:
            # if no ROI existing yet, add the default one
            if self.roiTable.rowCount() == 0:
                old = self.blockSignals(True)  # avoid several sigROISignal emission
                self._add()
                self.blockSignals(old)
                self.calculateRois()

    def fillFromROIDict(self, *args, **kwargs):
        self.roiTable.fillFromROIDict(*args, **kwargs)

    def _emitCurrentROISignal(self):
        ddict = {}
        ddict['event'] = "currentROISignal"
        if self.roiTable.activeRoi is not None:
            ddict['ROI'] = self.roiTable.activeRoi.toDict()
            ddict['current'] = self.roiTable.activeRoi.getName()
        else:
            ddict['current'] = None

        if self.__lastSigROISignal != ddict:
            self.__lastSigROISignal = ddict
            self.sigROISignal.emit(ddict)

    @property
    def currentRoi(self):
        return self.roiTable.activeRoi


class _FloatItem(qt.QTableWidgetItem):
    """
    Simple QTableWidgetItem overloading the < operator to deal with ordering
    """
    def __init__(self):
        qt.QTableWidgetItem.__init__(self, type=qt.QTableWidgetItem.Type)

    def __lt__(self, other):
        if self.text() in ('', ROITable.INFO_NOT_FOUND):
            return False
        if other.text() in ('', ROITable.INFO_NOT_FOUND):
            return True
        return float(self.text()) < float(other.text())


class ROITable(TableWidget):
    """Table widget displaying ROI information.

    See :class:`QTableWidget` for constructor arguments.

    Behavior: listen at the active curve changed only when the widget is
    visible. Otherwise won't compute the row and net counts...
    """

    activeROIChanged = qt.Signal()
    """Signal emitted when the active roi changed or when the value of the
    active roi are changing"""

    COLUMNS_INDEX = OrderedDict([
        ('ID', 0),
        ('ROI', 1),
        ('Type', 2),
        ('From', 3),
        ('To', 4),
        ('Raw Counts', 5),
        ('Net Counts', 6),
        ('Raw Area', 7),
        ('Net Area', 8),
    ])

    COLUMNS = list(COLUMNS_INDEX.keys())

    INFO_NOT_FOUND = '????????'

    def __init__(self, parent=None, plot=None, rois=None):
        super(ROITable, self).__init__(parent)
        self._showAllMarkers = False
        self._userIsEditingRoi = False
        """bool used to avoid conflict when editing the ROI object"""
        self._isConnected = False
        self._roiToItems = {}
        self._roiDict = {}
        """dict of ROI object. Key is ROi id, value is the ROI object"""
        self._markersHandler = _RoiMarkerManager()

        """
        Associate for each marker legend used when the `_showAllMarkers` option
        is active a roi.
        """
        self.setColumnCount(len(self.COLUMNS))
        self.setPlot(plot)
        self.__setTooltip()
        self.setSortingEnabled(True)
        self.itemChanged.connect(self._itemChanged)

    @property
    def roidict(self):
        return self._getRoiDict()

    @property
    def activeRoi(self):
        return self._markersHandler._activeRoi

    def _getRoiDict(self):
        ddict = {}
        for id in self._roiDict:
            ddict[self._roiDict[id].getName()] = self._roiDict[id]
        return ddict

    def clear(self):
        """
        .. note:: clear the interface only. keep the roidict...
        """
        self._markersHandler.clear()
        self._roiToItems = {}
        self._roiDict = {}

        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.horizontalHeader()
        header.setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.sortByColumn(0, qt.Qt.AscendingOrder)
        self.hideColumn(self.COLUMNS_INDEX['ID'])

    def setPlot(self, plot):
        self.clear()
        self.plot = plot

    def __setTooltip(self):
        self.horizontalHeaderItem(self.COLUMNS_INDEX['ROI']).setToolTip(
            'Region of interest identifier')
        self.horizontalHeaderItem(self.COLUMNS_INDEX['Type']).setToolTip(
            'Type of the ROI')
        self.horizontalHeaderItem(self.COLUMNS_INDEX['From']).setToolTip(
            'X-value of the min point')
        self.horizontalHeaderItem(self.COLUMNS_INDEX['To']).setToolTip(
            'X-value of the max point')
        self.horizontalHeaderItem(self.COLUMNS_INDEX['Raw Counts']).setToolTip(
            'Estimation of the integral between y=0 and the selected curve')
        self.horizontalHeaderItem(self.COLUMNS_INDEX['Net Counts']).setToolTip(
            'Estimation of the integral between the segment [maxPt, minPt] '
            'and the selected curve')

    def setRois(self, rois, order=None):
        """Set the ROIs by providing a dictionary of ROI information.

        The dictionary keys are the ROI names.
        Each value is a sub-dictionary of ROI info with the following fields:

        - ``"from"``: x coordinate of the left limit, as a float
        - ``"to"``: x coordinate of the right limit, as a float
        - ``"type"``: type of ROI, as a string (e.g "channels", "energy")


        :param roidict: Dictionary of ROIs
        :param str order: Field used for ordering the ROIs.
             One of "from", "to", "type".
             None (default) for no ordering, or same order as specified
             in parameter ``roidict`` if provided as an OrderedDict.
        """
        assert order in [None, "from", "to", "type"]
        self.clear()

        # backward compatibility since 0.10.0
        if isinstance(rois, dict):
            for roiName, roi in rois.items():
                if isinstance(roi, ROI):
                    _roi = roi
                else:
                    roi['name'] = roiName
                    _roi = ROI._fromDict(roi)
                self.addRoi(_roi)
        else:
            for roi in rois:
                assert isinstance(roi, ROI)
                self.addRoi(roi)
        self._updateMarkers()

    def addRoi(self, roi):
        """

        :param :class:`ROI` roi: roi to add to the table
        """
        assert isinstance(roi, ROI)
        self._getItem(name='ID', row=None, roi=roi)
        self._roiDict[roi.getID()] = roi
        self._markersHandler.add(roi, _RoiMarkerHandler(roi, self.plot))
        self._updateRoiInfo(roi.getID())
        callback = functools.partial(WeakMethodProxy(self._updateRoiInfo),
                                     roi.getID())
        roi.sigChanged.connect(callback)
        # set it as the active one
        self.setActiveRoi(roi)

    def _getItem(self, name, row, roi):
        if row:
            item = self.item(row, self.COLUMNS_INDEX[name])
        else:
            item = None
        if item:
            return item
        else:
            if name == 'ID':
                assert roi
                if roi.getID() in self._roiToItems:
                    return self._roiToItems[roi.getID()]
                else:
                    # create a new row
                    row = self.rowCount()
                    self.setRowCount(self.rowCount() + 1)
                    item = qt.QTableWidgetItem(str(roi.getID()),
                                               type=qt.QTableWidgetItem.Type)
                    self._roiToItems[roi.getID()] = item
            elif name == 'ROI':
                item = qt.QTableWidgetItem(roi.getName() if roi else '',
                                           type=qt.QTableWidgetItem.Type)
                if roi.getName().upper() in ('ICR', 'DEFAULT'):
                    item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
                else:
                    item.setFlags(qt.Qt.ItemIsSelectable |
                                  qt.Qt.ItemIsEnabled |
                                  qt.Qt.ItemIsEditable)
            elif name == 'Type':
                item = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
                item.setFlags((qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled))
            elif name in ('To', 'From'):
                item = _FloatItem()
                if roi.getName().upper() in ('ICR', 'DEFAULT'):
                    item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
                else:
                    item.setFlags(qt.Qt.ItemIsSelectable |
                                  qt.Qt.ItemIsEnabled |
                                  qt.Qt.ItemIsEditable)
            elif name in ('Raw Counts', 'Net Counts', 'Raw Area', 'Net Area'):
                item = _FloatItem()
                item.setFlags((qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled))
            else:
                raise ValueError('item type not recognized')

            self.setItem(row, self.COLUMNS_INDEX[name], item)
            return item

    def _itemChanged(self, item):
        def getRoi():
            IDItem = self.item(item.row(), self.COLUMNS_INDEX['ID'])
            assert IDItem
            id = int(IDItem.text())
            assert id in self._roiDict
            roi = self._roiDict[id]
            return roi

        def signalChanged(roi):
            if self.activeRoi and roi.getID() == self.activeRoi.getID():
                self.activeROIChanged.emit()

        self._userIsEditingRoi = True
        if item.column() in (self.COLUMNS_INDEX['To'], self.COLUMNS_INDEX['From']):
            roi = getRoi()

            if item.text() not in ('', self.INFO_NOT_FOUND):
                try:
                    value = float(item.text())
                except ValueError:
                    value = 0
                changed = False
                if item.column() == self.COLUMNS_INDEX['To']:
                    if value != roi.getTo():
                        roi.setTo(value)
                        changed = True
                else:
                    assert(item.column() == self.COLUMNS_INDEX['From'])
                    if value != roi.getFrom():
                        roi.setFrom(value)
                        changed = True
                if changed:
                    self._updateMarker(roi.getName())
                    signalChanged(roi)

        if item.column() is self.COLUMNS_INDEX['ROI']:
            roi = getRoi()
            if roi.getName() != item.text():
                roi.setName(item.text())
                self._markersHandler.getMarkerHandler(roi.getID()).updateTexts()
                signalChanged(roi)

        self._userIsEditingRoi = False

    def deleteActiveRoi(self):
        """
        remove the current active roi
        """
        activeItems = self.selectedItems()
        if len(activeItems) == 0:
            return
        old = self.blockSignals(True)  # avoid several emission of sigROISignal
        roiToRm = set()
        for item in activeItems:
            row = item.row()
            itemID = self.item(row, self.COLUMNS_INDEX['ID'])
            roiToRm.add(self._roiDict[int(itemID.text())])
        [self.removeROI(roi) for roi in roiToRm]
        self.blockSignals(old)
        self.setActiveRoi(None)

    def removeROI(self, roi):
        """
        remove the requested roi

        :param str name: the name of the roi to remove from the table
        """
        if roi and roi.getID() in self._roiToItems:
            item = self._roiToItems[roi.getID()]
            self.removeRow(item.row())
            del self._roiToItems[roi.getID()]

            assert roi.getID() in self._roiDict
            del self._roiDict[roi.getID()]
            self._markersHandler.remove(roi)

            callback = functools.partial(WeakMethodProxy(self._updateRoiInfo),
                                         roi.getID())
            roi.sigChanged.connect(callback)

    def setActiveRoi(self, roi):
        """
        Define the given roi as the active one.

        .. warning:: this roi should already be registred / added to the table

        :param :class:`ROI` roi: the roi to defined as active
        """
        if roi is None:
            self.clearSelection()
            self._markersHandler.setActiveRoi(None)
            self.activeROIChanged.emit()
        else:
            assert isinstance(roi, ROI)
            if roi and roi.getID() in self._roiToItems.keys():
                # avoid several call back to setActiveROI
                old = self.blockSignals(True)
                self.selectRow(self._roiToItems[roi.getID()].row())
                self.blockSignals(old)
                self._markersHandler.setActiveRoi(roi)
                self.activeROIChanged.emit()

    def _updateRoiInfo(self, roiID):
        if self._userIsEditingRoi is True:
            return
        if roiID not in self._roiDict:
            return
        roi = self._roiDict[roiID]
        if roi.isICR():
            activeCurve = self.plot.getActiveCurve()
            if activeCurve:
                xData = activeCurve.getXData()
                if len(xData) > 0:
                    min, max = min_max(xData)
                    roi.blockSignals(True)
                    roi.setFrom(min)
                    roi.setTo(max)
                    roi.blockSignals(False)

        itemID = self._getItem(name='ID', roi=roi, row=None)
        itemName = self._getItem(name='ROI', row=itemID.row(), roi=roi)
        itemName.setText(roi.getName())

        itemType = self._getItem(name='Type', row=itemID.row(), roi=roi)
        itemType.setText(roi.getType() or self.INFO_NOT_FOUND)

        itemFrom = self._getItem(name='From', row=itemID.row(), roi=roi)
        fromdata = str(roi.getFrom()) if roi.getFrom() is not None else self.INFO_NOT_FOUND
        itemFrom.setText(fromdata)

        itemTo = self._getItem(name='To', row=itemID.row(), roi=roi)
        todata = str(roi.getTo()) if roi.getTo() is not None else self.INFO_NOT_FOUND
        itemTo.setText(todata)

        rawCounts, netCounts = roi.computeRawAndNetCounts(
            curve=self.plot.getActiveCurve(just_legend=False))
        itemRawCounts = self._getItem(name='Raw Counts', row=itemID.row(),
                                      roi=roi)
        rawCounts = str(rawCounts) if rawCounts is not None else self.INFO_NOT_FOUND
        itemRawCounts.setText(rawCounts)

        itemNetCounts = self._getItem(name='Net Counts', row=itemID.row(),
                                      roi=roi)
        netCounts = str(netCounts) if netCounts is not None else self.INFO_NOT_FOUND
        itemNetCounts.setText(netCounts)

        rawArea, netArea = roi.computeRawAndNetArea(
            curve=self.plot.getActiveCurve(just_legend=False))
        itemRawArea = self._getItem(name='Raw Area', row=itemID.row(),
                                      roi=roi)
        rawArea = str(rawArea) if rawArea is not None else self.INFO_NOT_FOUND
        itemRawArea.setText(rawArea)

        itemNetArea = self._getItem(name='Net Area', row=itemID.row(),
                                    roi=roi)
        netArea = str(netArea) if netArea is not None else self.INFO_NOT_FOUND
        itemNetArea.setText(netArea)

        if self.activeRoi and roi.getID() == self.activeRoi.getID():
            self.activeROIChanged.emit()

    def currentChanged(self, current, previous):
        if previous and current.row() != previous.row() and current.row() >= 0:
            roiItem = self.item(current.row(),
                                self.COLUMNS_INDEX['ID'])

            assert roiItem
            self.setActiveRoi(self._roiDict[int(roiItem.text())])
            self._markersHandler.updateAllMarkers()
        qt.QTableWidget.currentChanged(self, current, previous)

    @deprecation.deprecated(reason="Removed",
                            replacement="roidict and roidict.values()",
                            since_version="0.10.0")
    def getROIListAndDict(self):
        """

        :return: the list of roi objects and the dictionary of roi name to roi
                 object.
        """
        roidict = self._roiDict
        return list(roidict.values()), roidict

    def calculateRois(self, roiList=None, roiDict=None):
        """
        Update values of all registred rois (raw and net counts in particular)

        :param roiList: deprecated parameter
        :param roiDict: deprecated parameter
        """
        if roiDict:
            deprecation.deprecated_warning(name='roiDict', type_='Parameter',
                                           reason='Unused parameter',
                                           since_version="0.10.0")
        if roiList:
            deprecation.deprecated_warning(name='roiList', type_='Parameter',
                                           reason='Unused parameter',
                                           since_version="0.10.0")

        for roiID in self._roiDict:
            self._updateRoiInfo(roiID)

    def _updateMarker(self, roiID):
        """Make sure the marker of the given roi name is updated"""
        if self._showAllMarkers or (self.activeRoi
                                    and self.activeRoi.getName() == roiID):
            self._updateMarkers()

    def _updateMarkers(self):
        if self._showAllMarkers is True:
            self._markersHandler.updateMarkers()
        else:
            if not self.activeRoi or not self.plot:
                return
            assert isinstance(self.activeRoi, ROI)
            markerHandler = self._markersHandler.getMarkerHandler(self.activeRoi.getID())
            if markerHandler is not None:
                markerHandler.updateMarkers()

    def getRois(self, order):
        """
        Return the currently defined ROIs, as an ordered dict.

        The dictionary keys are the ROI names.
        Each value is a :class:`ROI` object..

        :param order: Field used for ordering the ROIs.
             One of "from", "to", "type", "netcounts", "rawcounts".
             None (default) to get the same order as displayed in the widget.
        :return: Ordered dictionary of ROI information
        """

        if order is None or order.lower() == "none":
            ordered_roilist = list(self._roiDict.values())
            res = OrderedDict([(roi.getName(), self._roiDict[roi.getID()]) for roi in ordered_roilist])
        else:
            assert order in ["from", "to", "type", "netcounts", "rawcounts"]
            ordered_roilist = sorted(self._roiDict.keys(),
                                     key=lambda roi_id: self._roiDict[roi_id].get(order))
            res = OrderedDict([(roi.getName(), self._roiDict[id]) for id in ordered_roilist])

        return res

    def save(self, filename):
        """
        Save current ROIs of the widget as a dict of ROI to a file.

        :param str filename: The file to which to save the ROIs
        """
        roilist = []
        roidict = {}
        for roiID, roi in self._roiDict.items():
            roilist.append(roi.toDict())
            roidict[roi.getName()] = roi.toDict()
        datadict = {'ROI': {'roilist': roilist, 'roidict': roidict}}
        dictdump.dump(datadict, filename)

    def load(self, filename):
        """
        Load ROI widget information from a file storing a dict of ROI.

        :param str filename: The file from which to load ROI
        """
        roisDict = dictdump.load(filename)
        rois = []

        # Remove rawcounts and netcounts from ROIs
        for roiDict in roisDict['ROI']['roidict'].values():
            roiDict.pop('rawcounts', None)
            roiDict.pop('netcounts', None)
            rois.append(ROI._fromDict(roiDict))

        self.setRois(rois)

    def showAllMarkers(self, _show=True):
        """

        :param bool _show: if true show all the markers of all the ROIs
                           boundaries otherwise will only show the one of
                           the active ROI.
        """
        self._markersHandler.setShowAllMarkers(_show)

    def setMiddleROIMarkerFlag(self, flag=True):
        """
        Activate or deactivate middle marker.

        This allows shifting both min and max limits at once, by dragging
        a marker located in the middle.

        :param bool flag: True to activate middle ROI marker
        """
        self._markersHandler._middleROIMarkerFlag = flag

    def _handleROIMarkerEvent(self, ddict):
        """Handle plot signals related to marker events."""
        if ddict['event'] == 'markerMoved':
            label = ddict['label']
            roiID = self._markersHandler.getRoiID(markerID=label)
            if roiID is not None:
                # avoid several emission of sigROISignal
                old = self.blockSignals(True)
                self._markersHandler.changePosition(markerID=label,
                                                    x=ddict['x'])
                self.blockSignals(old)
                self._updateRoiInfo(roiID)

    def showEvent(self, event):
        self._visibilityChangedHandler(visible=True)
        qt.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self._visibilityChangedHandler(visible=False)
        qt.QWidget.hideEvent(self, event)

    def _visibilityChangedHandler(self, visible):
        """Handle widget's visibility updates.

        It is connected to plot signals only when visible.
        """
        if visible:
            assert self.plot
            if self._isConnected is False:
                self.plot.sigPlotSignal.connect(self._handleROIMarkerEvent)
                self.plot.sigActiveCurveChanged.connect(self._activeCurveChanged)
                self._isConnected = True
            self.calculateRois()
        else:
            if self._isConnected:
                self.plot.sigPlotSignal.disconnect(self._handleROIMarkerEvent)
                self.plot.sigActiveCurveChanged.disconnect(self._activeCurveChanged)
                self._isConnected = False

    def _activeCurveChanged(self, curve):
        self.calculateRois()

    def setCountsVisible(self, visible):
        """
        Display the columns relative to areas or not

        :param bool visible: True if the columns 'Raw Area' and 'Net Area'
                             should be visible.
        """
        if visible is True:
            self.showColumn(self.COLUMNS_INDEX['Raw Counts'])
            self.showColumn(self.COLUMNS_INDEX['Net Counts'])
        else:
            self.hideColumn(self.COLUMNS_INDEX['Raw Counts'])
            self.hideColumn(self.COLUMNS_INDEX['Net Counts'])

    def setAreaVisible(self, visible):
        """
        Display the columns relative to areas or not

        :param bool visible: True if the columns 'Raw Area' and 'Net Area'
                             should be visible.
        """
        if visible is True:
            self.showColumn(self.COLUMNS_INDEX['Raw Area'])
            self.showColumn(self.COLUMNS_INDEX['Net Area'])
        else:
            self.hideColumn(self.COLUMNS_INDEX['Raw Area'])
            self.hideColumn(self.COLUMNS_INDEX['Net Area'])

    def fillFromROIDict(self, roilist=(), roidict=None, currentroi=None):
        """
        This function API is kept for compatibility.
        But `setRois` should be preferred.

        Set the ROIs by providing a list of ROI names and a dictionary
        of ROI information for each ROI.
        The ROI names must match an existing dictionary key.
        The name list is used to provide an order for the ROIs.
        The dictionary's values are sub-dictionaries containing 3
        mandatory fields:

        - ``"from"``: x coordinate of the left limit, as a float
        - ``"to"``: x coordinate of the right limit, as a float
        - ``"type"``: type of ROI, as a string (e.g "channels", "energy")

        :param roilist: List of ROI names (keys of roidict)
        :type roilist: List
        :param dict roidict: Dict of ROI information
        :param currentroi: Name of the selected ROI or None (no selection)
        """
        if roidict is not None:
            self.setRois(roidict)
        else:
            self.setRois(roilist)
        if currentroi:
            self.setActiveRoi(currentroi)


_indexNextROI = 0


class ROI(_RegionOfInterestBase):
    """The Region Of Interest is defined by:

    - A name
    - A type. The type is the label of the x axis. This can be used to apply or
      not some ROI to a curve and do some post processing.
    - The x coordinate of the left limit (fromdata)
    - The x coordinate of the right limit (todata)

    :param str: name of the ROI
    :param fromdata: left limit of the roi
    :param todata: right limit of the roi
    :param type: type of the ROI
    """

    sigChanged = qt.Signal()
    """Signal emitted when the ROI is edited"""

    def __init__(self, name, fromdata=None, todata=None, type_=None):
        _RegionOfInterestBase.__init__(self)
        self.setName(name)
        global _indexNextROI
        self._id = _indexNextROI
        _indexNextROI += 1

        self._fromdata = fromdata
        self._todata = todata
        self._type = type_ or 'Default'

        self.sigItemChanged.connect(self.__itemChanged)

    def __itemChanged(self, event):
        """Handle name change"""
        if event == items.ItemChangedType.NAME:
            self.sigChanged.emit()

    def getID(self):
        """

        :return int: the unique ID of the ROI
        """
        return self._id

    def setType(self, type_):
        """

        :param str type_:
        """
        if self._type != type_:
            self._type = type_
            self.sigChanged.emit()

    def getType(self):
        """

        :return str: the type of the ROI.
        """
        return self._type

    def setFrom(self, frm):
        """

        :param frm: set x coordinate of the left limit
        """
        if self._fromdata != frm:
            self._fromdata = frm
            self.sigChanged.emit()

    def getFrom(self):
        """

        :return: x coordinate of the left limit
        """
        return self._fromdata

    def setTo(self, to):
        """

        :param to: x coordinate of the right limit
        """
        if self._todata != to:
            self._todata = to
            self.sigChanged.emit()

    def getTo(self):
        """

        :return: x coordinate of the right limit
        """
        return self._todata

    def getMiddle(self):
        """

        :return: middle position between 'from' and 'to' values
        """
        return 0.5 * (self.getFrom() + self.getTo())

    def toDict(self):
        """

        :return: dict containing the roi parameters
        """
        ddict = {
            'type': self._type,
            'name': self.getName(),
            'from': self._fromdata,
            'to': self._todata,
        }
        if hasattr(self, '_extraInfo'):
            ddict.update(self._extraInfo)
        return ddict

    @staticmethod
    def _fromDict(dic):
        assert 'name' in dic
        roi = ROI(name=dic['name'])
        roi._extraInfo = {}
        for key in dic:
            if key == 'from':
                roi.setFrom(dic['from'])
            elif key == 'to':
                roi.setTo(dic['to'])
            elif key == 'type':
                roi.setType(dic['type'])
            else:
                roi._extraInfo[key] = dic[key]

        return roi

    def isICR(self):
        """

        :return: True if the ROI is the `ICR`
        """
        return self.getName() == 'ICR'

    def computeRawAndNetCounts(self, curve):
        """Compute the Raw and net counts in the ROI for the given curve.

        - Raw count: Points values sum of the curve in the defined Region Of
           Interest.

          .. image:: img/rawCounts.png

        - Net count: Raw counts minus background

          .. image:: img/netCounts.png

        :param CurveItem curve:
        :return tuple: rawCount, netCount
        """
        assert isinstance(curve, items.Curve) or curve is None

        if curve is None:
            return None, None

        x = curve.getXData(copy=False)
        y = curve.getYData(copy=False)

        idx = numpy.nonzero((self._fromdata <= x) &
                            (x <= self._todata))[0]
        if len(idx):
            xw = x[idx]
            yw = y[idx]
            rawCounts = yw.sum(dtype=numpy.float64)
            deltaX = xw[-1] - xw[0]
            deltaY = yw[-1] - yw[0]
            if deltaX > 0.0:
                slope = (deltaY / deltaX)
                background = yw[0] + slope * (xw - xw[0])
                netCounts = (rawCounts -
                             background.sum(dtype=numpy.float64))
            else:
                netCounts = 0.0
        else:
            rawCounts = 0.0
            netCounts = 0.0
        return rawCounts, netCounts

    def computeRawAndNetArea(self, curve):
        """Compute the Raw and net counts in the ROI for the given curve.

        - Raw area: integral of the curve between the min ROI point and the
           max ROI point to the y = 0 line.

          .. image:: img/rawArea.png

        - Net area: Raw counts minus background

          .. image:: img/netArea.png

        :param CurveItem curve:
        :return tuple: rawArea, netArea
        """
        assert isinstance(curve, items.Curve) or curve is None

        if curve is None:
            return None, None

        x = curve.getXData(copy=False)
        y = curve.getYData(copy=False)

        y = y[(x >= self._fromdata) & (x <= self._todata)]
        x = x[(x >= self._fromdata) & (x <= self._todata)]

        if x.size == 0:
            return 0.0, 0.0

        rawArea = numpy.trapz(y, x=x)
        # to speed up and avoid an intersection calculation we are taking the
        # closest index to the ROI
        closestXLeftIndex = (numpy.abs(x - self.getFrom())).argmin()
        closestXRightIndex = (numpy.abs(x - self.getTo())).argmin()
        yBackground = y[closestXLeftIndex], y[closestXRightIndex]
        background = numpy.trapz(yBackground, x=x)
        netArea = rawArea - background
        return rawArea, netArea

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        return self._fromdata <= position[0] <= self._todata


class _RoiMarkerManager(object):
    """
    Deal with all the ROI markers
    """
    def __init__(self):
        self._roiMarkerHandlers = {}
        self._middleROIMarkerFlag = False
        self._showAllMarkers = False
        self._activeRoi = None

    def setActiveRoi(self, roi):
        self._activeRoi = roi
        self.updateAllMarkers()

    def setShowAllMarkers(self, show):
        if show != self._showAllMarkers:
            self._showAllMarkers = show
            self.updateAllMarkers()

    def add(self, roi, markersHandler):
        assert isinstance(roi, ROI)
        assert isinstance(markersHandler, _RoiMarkerHandler)
        if roi.getID() in self._roiMarkerHandlers:
            raise ValueError('roi with the same ID already existing')
        else:
            self._roiMarkerHandlers[roi.getID()] = markersHandler

    def getMarkerHandler(self, roiID):
        if roiID in self._roiMarkerHandlers:
            return self._roiMarkerHandlers[roiID]
        else:
            return None

    def clear(self):
        roisHandler = list(self._roiMarkerHandlers.values())
        for roiHandler in roisHandler:
            self.remove(roiHandler.roi)

    def remove(self, roi):
        if roi is None:
            return
        assert isinstance(roi, ROI)
        if roi.getID() in self._roiMarkerHandlers:
            self._roiMarkerHandlers[roi.getID()].clear()
            del self._roiMarkerHandlers[roi.getID()]

    def hasMarker(self, markerID):
        assert type(markerID) is str
        return self.getMarker(markerID) is not None

    def changePosition(self, markerID, x):
        markerHandler = self.getMarker(markerID)
        if markerHandler is None:
            raise ValueError('Marker %s not register' % markerID)
        markerHandler.changePosition(markerID=markerID, x=x)

    def updateMarker(self, markerID):
        markerHandler = self.getMarker(markerID)
        if markerHandler is None:
            raise ValueError('Marker %s not register' % markerID)
        roiID = self.getRoiID(markerID)
        visible = (self._activeRoi and self._activeRoi.getID() == roiID) or self._showAllMarkers is True
        markerHandler.setVisible(visible)
        markerHandler.updateAllMarkers()

    def updateRoiMarkers(self, roiID):
        if roiID in self._roiMarkerHandlers:
            visible = ((self._activeRoi and self._activeRoi.getID() == roiID)
                       or self._showAllMarkers is True)
            _roi = self._roiMarkerHandlers[roiID]._roi()
            if _roi and not _roi.isICR():
                self._roiMarkerHandlers[roiID].showMiddleMarker(self._middleROIMarkerFlag)
            self._roiMarkerHandlers[roiID].setVisible(visible)
            self._roiMarkerHandlers[roiID].updateMarkers()

    def getMarker(self, markerID):
        assert type(markerID) is str
        for marker in list(self._roiMarkerHandlers.values()):
            if marker.hasMarker(markerID):
                return marker

    def updateMarkers(self):
        for markerHandler in list(self._roiMarkerHandlers.values()):
            markerHandler.updateMarkers()

    def getRoiID(self, markerID):
        for roiID, markerHandler in self._roiMarkerHandlers.items():
            if markerHandler.hasMarker(markerID):
                return roiID
        return None

    def setShowMiddleMarkers(self, show):
        self._middleROIMarkerFlag = show
        self._roiMarkerHandlers.updateAllMarkers()

    def updateAllMarkers(self):
        for roiID in self._roiMarkerHandlers:
            self.updateRoiMarkers(roiID)

    def getVisibleRois(self):
        res = {}
        for roiID, roiHandler in self._roiMarkerHandlers.items():
            markers = (roiHandler.getMarker('min'), roiHandler.getMarker('max'),
                       roiHandler.getMarker('middle'))
            for marker in markers:
                if marker.isVisible():
                    if roiID not in res:
                        res[roiID] = []
                    res[roiID].append(marker)
        return res


class _RoiMarkerHandler(object):
    """Used to deal with ROI markers used in ROITable"""
    def __init__(self, roi, plot):
        assert roi and isinstance(roi, ROI)
        assert plot

        self._roi = weakref.ref(roi)
        self._plot = weakref.ref(plot)
        self._draggable = False if roi.isICR() else True
        self._color = 'black' if roi.isICR() else 'blue'
        self._displayMidMarker = False
        self._visible = True

    @property
    def draggable(self):
        return self._draggable

    @property
    def plot(self):
        return self._plot()

    def clear(self):
        if self.plot and self.roi:
            self.plot.removeMarker(self._markerID('min'))
            self.plot.removeMarker(self._markerID('max'))
            self.plot.removeMarker(self._markerID('middle'))

    @property
    def roi(self):
        return self._roi()

    def setVisible(self, visible):
        if visible != self._visible:
            self._visible = visible
            self.updateMarkers()

    def showMiddleMarker(self, visible):
        if self.draggable is False and visible is True:
            _logger.warning("ROI is not draggable. Won't display middle marker")
            return
        self._displayMidMarker = visible
        self.getMarker('middle').setVisible(self._displayMidMarker)

    def updateMarkers(self):
        if self.roi is None:
            return
        self._updateMinMarkerPos()
        self._updateMaxMarkerPos()
        self._updateMiddleMarkerPos()

    def _updateMinMarkerPos(self):
        self.getMarker('min').setPosition(x=self.roi.getFrom(), y=None)
        self.getMarker('min').setVisible(self._visible)

    def _updateMaxMarkerPos(self):
        self.getMarker('max').setPosition(x=self.roi.getTo(), y=None)
        self.getMarker('max').setVisible(self._visible)

    def _updateMiddleMarkerPos(self):
        self.getMarker('middle').setPosition(x=self.roi.getMiddle(), y=None)
        self.getMarker('middle').setVisible(self._displayMidMarker and self._visible)

    def getMarker(self, markerType):
        if self.plot is None:
            return None
        assert markerType in ('min', 'max', 'middle')
        if self.plot._getMarker(self._markerID(markerType)) is None:
            assert self.roi
            if markerType == 'min':
                val = self.roi.getFrom()
            elif markerType == 'max':
                val = self.roi.getTo()
            else:
                val = self.roi.getMiddle()

            _color = self._color
            if markerType == 'middle':
                _color = 'yellow'
            self.plot.addXMarker(val,
                                 legend=self._markerID(markerType),
                                 text=self.getMarkerName(markerType),
                                 color=_color,
                                 draggable=self.draggable)
        return self.plot._getMarker(self._markerID(markerType))

    def _markerID(self, markerType):
        assert markerType in ('min', 'max', 'middle')
        assert self.roi
        return '_'.join((str(self.roi.getID()), markerType))

    def getMarkerName(self, markerType):
        assert markerType in ('min', 'max', 'middle')
        assert self.roi
        return ' '.join((self.roi.getName(), markerType))

    def updateTexts(self):
        self.getMarker('min').setText(self.getMarkerName('min'))
        self.getMarker('max').setText(self.getMarkerName('max'))
        self.getMarker('middle').setText(self.getMarkerName('middle'))

    def changePosition(self, markerID, x):
        assert self.hasMarker(markerID)
        markerType = self._getMarkerType(markerID)
        assert markerType is not None
        if self.roi is None:
            return
        if markerType == 'min':
            self.roi.setFrom(x)
            self._updateMiddleMarkerPos()
        elif markerType == 'max':
            self.roi.setTo(x)
            self._updateMiddleMarkerPos()
        else:
            delta = x - 0.5 * (self.roi.getFrom() + self.roi.getTo())
            self.roi.setFrom(self.roi.getFrom() + delta)
            self.roi.setTo(self.roi.getTo() + delta)
            self._updateMinMarkerPos()
            self._updateMaxMarkerPos()

    def hasMarker(self, marker):
        return marker in (self._markerID('min'),
                          self._markerID('max'),
                          self._markerID('middle'))

    def _getMarkerType(self, markerID):
        if markerID.endswith('_min'):
            return 'min'
        elif markerID.endswith('_max'):
            return 'max'
        elif markerID.endswith('_middle'):
            return 'middle'
        else:
            return None


class CurvesROIDockWidget(qt.QDockWidget):
    """QDockWidget with a :class:`CurvesROIWidget` connected to a PlotWindow.

    It makes the link between the :class:`CurvesROIWidget` and the PlotWindow.

    :param parent: See :class:`QDockWidget`
    :param plot: :class:`.PlotWindow` instance on which to operate
    :param name: See :class:`QDockWidget`
    """
    sigROISignal = qt.Signal(object)
    """Deprecated signal for backward compatibility with silx < 0.7.
    Prefer connecting directly to :attr:`CurvesRoiWidget.sigRoiSignal`
    """

    def __init__(self, parent=None, plot=None, name=None):
        super(CurvesROIDockWidget, self).__init__(name, parent)

        assert plot is not None
        self.plot = plot
        self.roiWidget = CurvesROIWidget(self, name, plot=plot)
        """Main widget of type :class:`CurvesROIWidget`"""

        # convenience methods to offer a simpler API allowing to ignore
        # the details of the underlying implementation
        # (ALL DEPRECATED)
        self.calculateROIs = self.calculateRois = self.roiWidget.calculateRois
        self.setRois = self.roiWidget.setRois
        self.getRois = self.roiWidget.getRois

        self.roiWidget.sigROISignal.connect(self._forwardSigROISignal)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(self.roiWidget)

        self.setAreaVisible = self.roiWidget.roiTable.setAreaVisible
        self.setCountsVisible = self.roiWidget.roiTable.setCountsVisible

    def _forwardSigROISignal(self, ddict):
        # emit deprecated signal for backward compatibility (silx < 0.7)
        self.sigROISignal.emit(ddict)

    def toggleViewAction(self):
        """Returns a checkable action that shows or closes this widget.

        See :class:`QMainWindow`.
        """
        action = super(CurvesROIDockWidget, self).toggleViewAction()
        action.setIcon(icons.getQIcon('plot-roi'))
        return action

    @property
    def currentROI(self):
        return self.roiWidget.currentRoi
