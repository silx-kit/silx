# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Widget to handle regions of interest on curves displayed in a PlotWindow.

This widget is meant to work with :class:`PlotWindow`.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "15/09/2016"


import logging
import os
import sys

import numpy

from silx.io import dictdump
from .. import icons, qt


_logger = logging.getLogger(__name__)


class CurvesROIWidget(qt.QWidget):
    """Widget displaying a table of ROI information.

    :param parent: See :class:`QWidget`
    :paran str name: The title of this widget
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

    def __init__(self, parent=None, name=None):
        super(CurvesROIWidget, self).__init__(parent)
        if name is not None:
            self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        ##############
        self.headerLabel = qt.QLabel(self)
        self.headerLabel.setAlignment(qt.Qt.AlignHCenter)
        self.setHeader()
        layout.addWidget(self.headerLabel)
        ##############
        self.roiTable = ROITable(self)
        rheight = self.roiTable.horizontalHeader().sizeHint().height()
        self.roiTable.setMinimumHeight(4 * rheight)
        self.fillFromROIDict = self.roiTable.fillFromROIDict
        self.getROIListAndDict = self.roiTable.getROIListAndDict
        layout.addWidget(self.roiTable)
        self._roiFileDir = qt.QDir.home().absolutePath()
        #################

        hbox = qt.QWidget(self)
        hboxlayout = qt.QHBoxLayout(hbox)
        hboxlayout.setContentsMargins(0, 0, 0, 0)
        hboxlayout.setSpacing(0)

        hboxlayout.addStretch(0)

        self.addButton = qt.QPushButton(hbox)
        self.addButton.setText("Add ROI")
        self.delButton = qt.QPushButton(hbox)
        self.delButton.setText("Delete ROI")
        self.resetButton = qt.QPushButton(hbox)
        self.resetButton.setText("Reset")

        hboxlayout.addWidget(self.addButton)
        hboxlayout.addWidget(self.delButton)
        hboxlayout.addWidget(self.resetButton)

        hboxlayout.addStretch(0)

        self.loadButton = qt.QPushButton(hbox)
        self.loadButton.setText("Load")
        self.saveButton = qt.QPushButton(hbox)
        self.saveButton.setText("Save")
        hboxlayout.addWidget(self.loadButton)
        hboxlayout.addWidget(self.saveButton)
        layout.setStretchFactor(self.headerLabel, 0)
        layout.setStretchFactor(self.roiTable, 1)
        layout.setStretchFactor(hbox, 0)

        layout.addWidget(hbox)

        self.addButton.clicked.connect(self._add)
        self.delButton.clicked.connect(self._del)
        self.resetButton.clicked.connect(self._reset)

        self.loadButton.clicked.connect(self._load)
        self.saveButton.clicked.connect(self._save)
        self.roiTable.sigROITableSignal.connect(self._forward)

    @property
    def roiFileDir(self):
        """The directory from which to load/save ROI from/to files."""
        if not os.path.isdir(self._roiFileDir):
            self._roiFileDir = qt.QDir.home().absolutePath()
        return self._roiFileDir

    @roiFileDir.setter
    def roiFileDir(self, roiFileDir):
        self._roiFileDir = str(roiFileDir)

    def _add(self):
        """Add button clicked handler"""
        ddict = {}
        ddict['event'] = "AddROI"
        roilist, roidict = self.roiTable.getROIListAndDict()
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        self.sigROIWidgetSignal.emit(ddict)

    def _del(self):
        """Delete button clicked handler"""
        row = self.roiTable.currentRow()
        if row >= 0:
            index = self.roiTable.labels.index('Type')
            text = str(self.roiTable.item(row, index).text())
            if text.upper() != 'DEFAULT':
                index = self.roiTable.labels.index('ROI')
                key = str(self.roiTable.item(row, index).text())
            else:
                # This is to prevent deleting ICR ROI, that is
                # usually initialized as "Default" type.
                return
            roilist, roidict = self.roiTable.getROIListAndDict()
            row = roilist.index(key)
            del roilist[row]
            del roidict[key]
            if len(roilist) > 0:
                currentroi = roilist[0]
            else:
                currentroi = None

            self.roiTable.fillFromROIDict(roilist=roilist,
                                          roidict=roidict,
                                          currentroi=currentroi)
            ddict = {}
            ddict['event'] = "DelROI"
            ddict['roilist'] = roilist
            ddict['roidict'] = roidict
            self.sigROIWidgetSignal.emit(ddict)

    def _forward(self, ddict):
        """Broadcast events from ROITable signal"""
        self.sigROIWidgetSignal.emit(ddict)

    def _reset(self):
        """Reset button clicked handler"""
        ddict = {}
        ddict['event'] = "ResetROI"
        roilist0, roidict0 = self.roiTable.getROIListAndDict()
        index = 0
        for key in roilist0:
            if roidict0[key]['type'].upper() == 'DEFAULT':
                index = roilist0.index(key)
                break
        roilist = []
        roidict = {}
        if len(roilist0):
            roilist.append(roilist0[index])
            roidict[roilist[0]] = {}
            roidict[roilist[0]].update(roidict0[roilist[0]])
            self.roiTable.fillFromROIDict(roilist=roilist, roidict=roidict)
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        self.sigROIWidgetSignal.emit(ddict)

    def _load(self):
        """Load button clicked handler"""
        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(
            ['INI File  *.ini', 'JSON File *.json', 'All *.*'])
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        dialog.setDirectory(self.roiFileDir)
        if not dialog.exec_():
            dialog.close()
            return

        # pyflakes bug http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=666494
        outputFile = dialog.selectedFiles()[0]
        dialog.close()

        self.roiFileDir = os.path.dirname(outputFile)
        self.load(outputFile)

    def load(self, filename):
        """Load ROI widget information from a file storing a dict of ROI.

        :param str filename: The file from which to load ROI
        """
        rois = dictdump.load(filename)
        currentROI = None
        if self.roiTable.rowCount():
            item = self.roiTable.item(self.roiTable.currentRow(), 0)
            if item is not None:
                currentROI = str(item.text())

        # Remove rawcounts and netcounts from ROIs
        for roi in rois['ROI']['roidict'].values():
            roi.pop('rawcounts', None)
            roi.pop('netcounts', None)

        self.roiTable.fillFromROIDict(roilist=rois['ROI']['roilist'],
                                      roidict=rois['ROI']['roidict'],
                                      currentroi=currentROI)

        roilist, roidict = self.roiTable.getROIListAndDict()
        event = {'event': 'LoadROI', 'roilist': roilist, 'roidict': roidict}
        self.sigROIWidgetSignal.emit(event)

    def _save(self):
        """Save button clicked handler"""
        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(['INI File  *.ini', 'JSON File *.json'])
        dialog.setFileMode(qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setDirectory(self.roiFileDir)
        if not dialog.exec_():
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
                msg.exec_()
                return
        self.roiFileDir = os.path.dirname(outputFile)
        self.save(outputFile)

    def save(self, filename):
        """Save current ROIs of the widget as a dict of ROI to a file.

        :param str filename: The file to which to save the ROIs
        """
        roilist, roidict = self.roiTable.getROIListAndDict()
        datadict = {'ROI': {'roilist': roilist, 'roidict': roidict}}
        dictdump.dump(datadict, filename)

    def setHeader(self, text='ROIs'):
        """Set the header text of this widget"""
        self.headerLabel.setText("<b>%s<\b>" % text)


class ROITable(qt.QTableWidget):
    """Table widget displaying ROI information.

    See :class:`QTableWidget` for constructor arguments.
    """

    sigROITableSignal = qt.Signal(object)
    """Signal of ROI table modifications.
    """

    def __init__(self, *args, **kwargs):
        super(ROITable, self).__init__(*args, **kwargs)
        self.setRowCount(1)
        self.labels = 'ROI', 'Type', 'From', 'To', 'Raw Counts', 'Net Counts'
        self.setColumnCount(len(self.labels))
        self.setSortingEnabled(False)

        for index, label in enumerate(self.labels):
            item = self.horizontalHeaderItem(index)
            if item is None:
                item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
            item.setText(label)
            self.setHorizontalHeaderItem(index, item)

        self.roidict = {}
        self.roilist = []

        self.building = False
        self.fillFromROIDict(roilist=self.roilist, roidict=self.roidict)

        self.cellClicked[(int, int)].connect(self._cellClickedSlot)
        self.cellChanged[(int, int)].connect(self._cellChangedSlot)
        verticalHeader = self.verticalHeader()
        verticalHeader.sectionClicked[int].connect(self._rowChangedSlot)

    def fillFromROIDict(self, roilist=(), roidict=None, currentroi=None):
        """Set the ROIs

        :param roilist: List of ROI names (keys of roidict)
        :type roilist: List
        :param dict roidict: Dict of ROI information
        :param currentroi: Name of the selected ROI or None (no selection)
        """
        if roidict is None:
            roidict = {}

        self.building = True
        line0 = 0
        self.roilist = []
        self.roidict = {}
        for key in roilist:
            if key in roidict.keys():
                roi = roidict[key]
                self.roilist.append(key)
                self.roidict[key] = {}
                self.roidict[key].update(roi)
                line0 = line0 + 1
                nlines = self.rowCount()
                if (line0 > nlines):
                    self.setRowCount(line0)
                line = line0 - 1
                self.roidict[key]['line'] = line
                ROI = key
                roitype = "%s" % roi['type']
                fromdata = "%6g" % (roi['from'])
                todata = "%6g" % (roi['to'])
                if 'rawcounts' in roi:
                    rawcounts = "%6g" % (roi['rawcounts'])
                else:
                    rawcounts = " ?????? "
                if 'netcounts' in roi:
                    netcounts = "%6g" % (roi['netcounts'])
                else:
                    netcounts = " ?????? "
                fields = [ROI, roitype, fromdata, todata, rawcounts, netcounts]
                col = 0
                for field in fields:
                    key2 = self.item(line, col)
                    if key2 is None:
                        key2 = qt.QTableWidgetItem(field,
                                                   qt.QTableWidgetItem.Type)
                        self.setItem(line, col, key2)
                    else:
                        key2.setText(field)
                    if (ROI.upper() == 'ICR') or (ROI.upper() == 'DEFAULT'):
                            key2.setFlags(qt.Qt.ItemIsSelectable |
                                          qt.Qt.ItemIsEnabled)
                    else:
                        if col in [0, 2, 3]:
                            key2.setFlags(qt.Qt.ItemIsSelectable |
                                          qt.Qt.ItemIsEnabled |
                                          qt.Qt.ItemIsEditable)
                        else:
                            key2.setFlags(qt.Qt.ItemIsSelectable |
                                          qt.Qt.ItemIsEnabled)
                    col = col + 1
        self.setRowCount(line0)
        i = 0
        for _label in self.labels:
            self.resizeColumnToContents(i)
            i = i + 1
        self.sortByColumn(2, qt.Qt.AscendingOrder)
        for i in range(len(self.roilist)):
            key = str(self.item(i, 0).text())
            self.roilist[i] = key
            self.roidict[key]['line'] = i
        if len(self.roilist) == 1:
            self.selectRow(0)
        else:
            if currentroi in self.roidict.keys():
                self.selectRow(self.roidict[currentroi]['line'])
                _logger.debug("Qt4 ensureCellVisible to be implemented")
        self.building = False

    def getROIListAndDict(self):
        """Return the currently defined ROIs

        :return: ROIs information
        :rtype: ordered dict as a tuple of (list of ROI names, dict of info)
        """
        return self.roilist, self.roidict

    def _cellClickedSlot(self, *var, **kw):
        # selection changed event, get the current selection
        row = self.currentRow()
        col = self.currentColumn()
        if row >= 0 and row < len(self.roilist):
            item = self.item(row, 0)
            text = '' if item is None else str(item.text())
            self.roilist[row] = text
            self._emitSelectionChangedSignal(row, col)

    def _rowChangedSlot(self, row):
        self._emitSelectionChangedSignal(row, 0)

    def _cellChangedSlot(self, row, col):
        _logger.debug("_cellChangedSlot(%d, %d)", row, col)
        if self.building:
            return
        if col == 0:
            self.nameSlot(row, col)
        else:
            self._valueChanged(row, col)

    def _valueChanged(self, row, col):
        if col not in [2, 3]:
            return
        item = self.item(row, col)
        if item is None:
            return
        text = str(item.text())
        try:
            value = float(text)
        except:
            return
        if row >= len(self.roilist):
            _logger.debug("deleting???")
            return
        item = self.item(row, 0)
        if item is None:
            text = ""
        else:
            text = str(item.text())
        if not len(text):
            return
        if col == 2:
            self.roidict[text]['from'] = value
        elif col == 3:
            self.roidict[text]['to'] = value
        self._emitSelectionChangedSignal(row, col)

    def nameSlot(self, row, col):
        if col != 0:
            return
        if row >= len(self.roilist):
            _logger.debug("deleting???")
            return
        item = self.item(row, col)
        if item is None:
            text = ""
        else:
            text = str(item.text())
        if len(text) and (text not in self.roilist):
            old = self.roilist[row]
            self.roilist[row] = text
            self.roidict[text] = {}
            self.roidict[text].update(self.roidict[old])
            del self.roidict[old]
            self._emitSelectionChangedSignal(row, col)

    def _emitSelectionChangedSignal(self, row, col):
        ddict = {}
        ddict['event'] = "selectionChanged"
        ddict['row'] = row
        ddict['col'] = col
        ddict['roi'] = self.roidict[self.roilist[row]]
        ddict['key'] = self.roilist[row]
        ddict['colheader'] = self.labels[col]
        ddict['rowheader'] = "%d" % row
        self.sigROITableSignal.emit(ddict)


class CurvesROIDockWidget(qt.QDockWidget):
    """QDockWidget with a :class:`CurvesROIWidget` connected to a PlotWindow.

    It makes the link between the CurvesROIWidget and the PlotWindow.

    :param parent: See :class:`QDockWidget`
    :param plot: :class:`.PlotWindow` instance on which to operate
    :param name: See :class:`QDockWidget`
    """
    sigROISignal = qt.Signal(object)

    def __init__(self, parent=None, plot=None, name=None):
        super(CurvesROIDockWidget, self).__init__(name, parent)

        assert plot is not None
        self.plot = plot

        self.currentROI = None
        self._middleROIMarkerFlag = False

        self._isConnected = False  # True if connected to plot signals
        self._isInit = False

        self.roiWidget = CurvesROIWidget(self, name)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(self.roiWidget)

        self.visibilityChanged.connect(self._visibilityChangedHandler)

    def toggleViewAction(self):
        """Returns a checkable action that shows or closes this widget.

        See :class:`QMainWindow`.
        """
        action = super(CurvesROIDockWidget, self).toggleViewAction()
        action.setIcon(icons.getQIcon('plot-roi'))
        return action

    def _visibilityChangedHandler(self, visible):
        """Handle widget's visibilty updates.

        It is connected to plot signals only when visible.
        """
        if visible:
            if not self._isInit:
                # Deferred ROI widget init finalization
                self._isInit = True
                self.roiWidget.sigROIWidgetSignal.connect(self._roiSignal)
                # initialize with the ICR
                self._roiSignal({'event': "AddROI"})

            if not self._isConnected:
                self.plot.sigPlotSignal.connect(self._handleROIMarkerEvent)
                self.plot.sigActiveCurveChanged.connect(
                    self._activeCurveChanged)
                self._isConnected = True

                self.calculateROIs()
        else:
            if self._isConnected:
                self.plot.sigPlotSignal.disconnect(self._handleROIMarkerEvent)
                self.plot.sigActiveCurveChanged.disconnect(
                    self._activeCurveChanged)
                self._isConnected = False

    def _handleROIMarkerEvent(self, ddict):
        """Handle plot signals related to marker events."""
        if ddict['event'] == 'markerMoved':

            label = ddict['label']
            if label not in ['ROI min', 'ROI max', 'ROI middle']:
                return

            roiList, roiDict = self.roiWidget.getROIListAndDict()
            if self.currentROI is None:
                return
            if self.currentROI not in roiDict:
                return
            x = ddict['x']

            if label == 'ROI min':
                roiDict[self.currentROI]['from'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +
                                 roiDict[self.currentROI]['from'])
                    self.plot.addXMarker(pos,
                                         legend='ROI middle',
                                         text='',
                                         color='yellow',
                                         draggable=True)
            elif label == 'ROI max':
                roiDict[self.currentROI]['to'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +
                                 roiDict[self.currentROI]['from'])
                    self.plot.addXMarker(pos,
                                         legend='ROI middle',
                                         text='',
                                         color='yellow',
                                         draggable=True)
            elif label == 'ROI middle':
                delta = x - 0.5 * (roiDict[self.currentROI]['from'] +
                                   roiDict[self.currentROI]['to'])
                roiDict[self.currentROI]['from'] += delta
                roiDict[self.currentROI]['to'] += delta
                self.plot.addXMarker(roiDict[self.currentROI]['from'],
                                     legend='ROI min',
                                     text='ROI min',
                                     color='blue',
                                     draggable=True)
                self.plot.addXMarker(roiDict[self.currentROI]['to'],
                                     legend='ROI max',
                                     text='ROI max',
                                     color='blue',
                                     draggable=True)
            else:
                return
            self.calculateROIs(roiList, roiDict)
            self._emitCurrentROISignal()

    def _roiSignal(self, ddict):
        """Handle ROI widget signal"""
        _logger.debug("PlotWindow._roiSignal %s", str(ddict))
        if ddict['event'] == "AddROI":
            xmin, xmax = self.plot.getGraphXLimits()
            fromdata = xmin + 0.25 * (xmax - xmin)
            todata = xmin + 0.75 * (xmax - xmin)
            self.plot.remove('ROI min', kind='marker')
            self.plot.remove('ROI max', kind='marker')
            if self._middleROIMarkerFlag:
                self.remove('ROI middle', kind='marker')
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            nrois = len(roiList)
            if nrois == 0:
                newroi = "ICR"
                fromdata, dummy0, todata, dummy1 = self._getAllLimits()
                draggable = False
                color = 'black'
            else:
                for i in range(nrois):
                    i += 1
                    newroi = "newroi %d" % i
                    if newroi not in roiList:
                        break
                color = 'blue'
                draggable = True
            self.plot.addXMarker(fromdata,
                                 legend='ROI min',
                                 text='ROI min',
                                 color=color,
                                 draggable=draggable)
            self.plot.addXMarker(todata,
                                 legend='ROI max',
                                 text='ROI max',
                                 color=color,
                                 draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.plot.addXMarker(pos,
                                     legend='ROI middle',
                                     text="",
                                     color='yellow',
                                     draggable=draggable)
            roiList.append(newroi)
            roiDict[newroi] = {}
            if newroi == "ICR":
                roiDict[newroi]['type'] = "Default"
            else:
                roiDict[newroi]['type'] = self.plot.getGraphXLabel()
            roiDict[newroi]['from'] = fromdata
            roiDict[newroi]['to'] = todata
            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=newroi)
            self.currentROI = newroi
            self.calculateROIs()
        elif ddict['event'] in ['DelROI', "ResetROI"]:
            self.plot.remove('ROI min', kind='marker')
            self.plot.remove('ROI max', kind='marker')
            if self._middleROIMarkerFlag:
                self.plot.remove('ROI middle', kind='marker')
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            roiDictKeys = list(roiDict.keys())
            if len(roiDictKeys):
                currentroi = roiDictKeys[0]
            else:
                # create again the ICR
                ddict = {"event": "AddROI"}
                return self._roiSignal(ddict)

            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=currentroi)
            self.currentROI = currentroi

        elif ddict['event'] == 'LoadROI':
            self.calculateROIs()

        elif ddict['event'] == 'selectionChanged':
            _logger.debug("Selection changed")
            self.roilist, self.roidict = self.roiWidget.getROIListAndDict()
            fromdata = ddict['roi']['from']
            todata = ddict['roi']['to']
            self.plot.remove('ROI min', kind='marker')
            self.plot.remove('ROI max', kind='marker')
            if ddict['key'] == 'ICR':
                draggable = False
                color = 'black'
            else:
                draggable = True
                color = 'blue'
            self.plot.addXMarker(fromdata,
                                 legend='ROI min',
                                 text='ROI min',
                                 color=color,
                                 draggable=draggable)
            self.plot.addXMarker(todata,
                                 legend='ROI max',
                                 text='ROI max',
                                 color=color,
                                 draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.plot.addXMarker(pos,
                                     legend='ROI middle',
                                     text="",
                                     color='yellow',
                                     draggable=True)
            self.currentROI = ddict['key']
            if ddict['colheader'] in ['From', 'To']:
                dict0 = {}
                dict0['event'] = "SetActiveCurveEvent"
                dict0['legend'] = self.plot.getActiveCurve(just_legend=1)
                self.plot.setActiveCurve(dict0['legend'])
            elif ddict['colheader'] == 'Raw Counts':
                pass
            elif ddict['colheader'] == 'Net Counts':
                pass
            else:
                self._emitCurrentROISignal()

        else:
            _logger.debug("Unknown or ignored event %s", ddict['event'])

    def _activeCurveChanged(self, *args):
        """Recompute ROIs when active curve changed."""
        self.calculateROIs()

    def calculateROIs(self, roiList=None, roiDict=None):
        """Compute ROI information"""
        if roiList is None or roiDict is None:
            roiList, roiDict = self.roiWidget.getROIListAndDict()

        activeCurve = self.plot.getActiveCurve(just_legend=False)
        if activeCurve is None:
            xproc = None
            self.roiWidget.setHeader()
        else:
            x, y, legend = activeCurve[0:3]
            idx = numpy.argsort(x, kind='mergesort')
            xproc = numpy.take(x, idx)
            self.roiWidget.setHeader('ROIs of %s' % legend)

        for key in roiList:
            if key == 'ICR':
                if xproc is not None:
                    roiDict[key]['from'] = xproc.min()
                    roiDict[key]['to'] = xproc.max()
                else:
                    roiDict[key]['from'] = 0
                    roiDict[key]['to'] = -1
            fromData = roiDict[key]['from']
            toData = roiDict[key]['to']
            if xproc is not None:
                idx = numpy.nonzero((fromData <= xproc) &
                                    (xproc <= toData))[0]
                if len(idx):
                    xw = x[idx]
                    yw = y[idx]
                    rawCounts = yw.sum(dtype=numpy.float)
                    deltaX = xw[-1] - xw[0]
                    deltaY = yw[-1] - yw[0]
                    if deltaX > 0.0:
                        slope = (deltaY / deltaX)
                        background = yw[0] + slope * (xw - xw[0])
                        netCounts = (rawCounts -
                                     background.sum(dtype=numpy.float))
                    else:
                        netCounts = 0.0
                else:
                    rawCounts = 0.0
                    netCounts = 0.0
                roiDict[key]['rawcounts'] = rawCounts
                roiDict[key]['netcounts'] = netCounts
            else:
                roiDict[key].pop('rawcounts', None)
                roiDict[key].pop('netcounts', None)

        self.roiWidget.fillFromROIDict(
            roilist=roiList,
            roidict=roiDict,
            currentroi=self.currentROI if self.currentROI in roiList else None)

    def _emitCurrentROISignal(self):
        ddict = {}
        ddict['event'] = "currentROISignal"
        _roiList, roiDict = self.roiWidget.getROIListAndDict()
        if self.currentROI in roiDict:
            ddict['ROI'] = roiDict[self.currentROI]
        else:
            self.currentROI = None
        ddict['current'] = self.currentROI
        self.sigROISignal.emit(ddict)

    def _getAllLimits(self):
        """Retrieve the limits based on the curves."""
        curves = self.plot.getAllCurves()
        if not curves:
            return 1.0, 1.0, 100., 100.

        xmin, ymin = None, None
        xmax, ymax = None, None

        for curve in curves:
            x, y = curve[0:2]
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
