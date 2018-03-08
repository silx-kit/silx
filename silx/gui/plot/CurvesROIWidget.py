# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""Widget to handle regions of interest (ROI) on curves displayed in a PlotWindow.

This widget is meant to work with :class:`PlotWindow`.

ROI are defined by :

- A name (`ROI` column)
- A type. The type is the label of the x axis.
  This can be used to apply or not some ROI to a curve and do some post processing.
- The x coordinate of the left limit (`from` column)
- The x coordinate of the right limit (`to` column)
- Raw counts: Sum of the curve's values in the defined Region Of Intereset.

  .. image:: img/rawCounts.png

- Net counts: Raw counts minus background

  .. image:: img/netCounts.png
"""

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "13/11/2017"

from collections import OrderedDict
import logging
import os
import sys
import weakref
import functools
import numpy
from silx.io import dictdump
from silx.utils import deprecation
from silx.utils.weakref import WeakMethodProxy
from .. import icons, qt


_logger = logging.getLogger(__name__)


class CurvesROIWidget(qt.QWidget):
    """Widget displaying a table of ROI information.

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
        assert plot is not None
        self._plotRef = weakref.ref(plot)
        self._showAllMarkers = False
        self.currentROI = None

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
        self.addButton.setToolTip('Create a new ROI')
        self.delButton = qt.QPushButton(hbox)
        self.delButton.setText("Delete ROI")
        self.addButton.setToolTip('Remove the selected ROI')
        self.resetButton = qt.QPushButton(hbox)
        self.resetButton.setText("Reset")
        self.addButton.setToolTip('Clear all created ROIs. We only let the default ROI')

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

        self.addButton.clicked.connect(self._add)
        self.delButton.clicked.connect(self._del)
        self.resetButton.clicked.connect(self._reset)

        self.loadButton.clicked.connect(self._load)
        self.saveButton.clicked.connect(self._save)
        self.roiTable.sigROITableSignal.connect(self._forward)

        self._middleROIMarkerFlag = False
        self._isConnected = False  # True if connected to plot signals
        self._isInit = False

    def getPlotWidget(self):
        """Returns the associated PlotWidget or None

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def showEvent(self, event):
        self._visibilityChangedHandler(visible=True)
        qt.QWidget.showEvent(self, event)

    def hideEvent(self, event):
        self._visibilityChangedHandler(visible=False)
        qt.QWidget.hideEvent(self, event)

    @property
    def roiFileDir(self):
        """The directory from which to load/save ROI from/to files."""
        if not os.path.isdir(self._roiFileDir):
            self._roiFileDir = qt.QDir.home().absolutePath()
        return self._roiFileDir

    @roiFileDir.setter
    def roiFileDir(self, roiFileDir):
        self._roiFileDir = str(roiFileDir)

    def setRois(self, roidict, order=None):
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
        if order is None or order.lower() == "none":
            roilist = list(roidict.keys())
        else:
            assert order in ["from", "to", "type"]
            roilist = sorted(roidict.keys(),
                             key=lambda roi_name: roidict[roi_name].get(order))

        return self.roiTable.fillFromROIDict(roilist, roidict)

    def getRois(self, order=None):
        """Return the currently defined ROIs, as an ordered dict.

        The dictionary keys are the ROI names.
        Each value is a sub-dictionary of ROI info with the following fields:

        - ``"from"``: x coordinate of the left limit, as a float
        - ``"to"``: x coordinate of the right limit, as a float
        - ``"type"``: type of ROI, as a string (e.g "channels", "energy")


        :param order: Field used for ordering the ROIs.
             One of "from", "to", "type", "netcounts", "rawcounts".
             None (default) to get the same order as displayed in the widget.
        :return: Ordered dictionary of ROI information
        """
        roilist, roidict = self.roiTable.getROIListAndDict()
        if order is None or order.lower() == "none":
            ordered_roilist = roilist
        else:
            assert order in ["from", "to", "type", "netcounts", "rawcounts"]
            ordered_roilist = sorted(roidict.keys(),
                                     key=lambda roi_name: roidict[roi_name].get(order))

        return OrderedDict([(name, roidict[name]) for name in ordered_roilist])

    def setMiddleROIMarkerFlag(self, flag=True):
        """Activate or deactivate middle marker.

        This allows shifting both min and max limits at once, by dragging
        a marker located in the middle.

        :param bool flag: True to activate middle ROI marker
        """
        if flag:
            self._middleROIMarkerFlag = True
        else:
            self._middleROIMarkerFlag = False

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
            self.currentROI = None
            if len(roilist) > 0:
                currentroi = roilist[0]

            self.roiTable.fillFromROIDict(roilist=roilist,
                                          roidict=roidict,
                                          currentroi=self.currentROI)
            ddict = {}
            ddict['event'] = "DelROI"
            ddict['roilist'] = roilist
            ddict['roidict'] = roidict
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

    def _roiSignal(self, ddict):
        """Handle ROI widget signal"""

        def getNextROIName():
            roiList, roiDict = self.roiTable.getROIListAndDict()
            nrois = len(roiList)
            if nrois == 0:
                return "ICR"
            else:
                for i in range(nrois):
                    i += 1
                    newroi = "newroi %d" % i
                    if newroi not in roiList:
                        return newroi

        def addROI():
            print('add ROI')
            roi = _ROI(name=getNextROIName())
            roi.color = 'black' if roi.name == 'ICR' else 'blue'
            roi.draggable = False if roi.name == 'ICR' else True

            if roi.name == "ICR":
                roi.type = "Default"
            else:
                roi.type = self.plot.getXAxis().getLabel()

            xmin, xmax = self.plot.getXAxis().getLimits()
            fromdata = xmin + 0.25 * (xmax - xmin)
            todata = xmin + 0.75 * (xmax - xmin)
            if roi.name == 'ICR':
                fromdata, dummy0, todata, dummy1 = self._getAllLimits()
            roi.fromdata = fromdata
            roi.todata = todata

            self.currentROI = roi

            roiList, roiDict = self.roiTable.getROIListAndDict()
            roiList.append(roi)
            roiDict[roi.name] = roi.toDict()

            self._updateMarkers()

            self.roiTable.fillFromROIDict(roilist=roiList,
                                          roidict=roiDict,
                                          currentroi=roi.name)
            self.calculateRois()


        _logger.debug("CurvesROIWidget._roiSignal %s", str(ddict))
        if ddict['event'] == "AddROI":
            _logger.debug("Add Roi")
            addROI()
        elif ddict['event'] == "ResetROI":
            _logger.debug("Reset Roi")
            self._updateMarkers()
        elif ddict['event'] == 'DelROI':
            _logger.debug("Del ROI %s" % self.currentROI)
            if self.currentROI:
                del self.roilist
                self.currentROI = None
                self._updateMarkers()
            self.roiTable.fillFromROIDict(roilist=roiList,
                                          roidict=roiDict,
                                          currentroi=currentroi)
        elif ddict['event'] == 'LoadROI':
            _logger.debug("Load ROI")
            self.calculateRois()
        elif ddict['event'] == 'selectionChanged':
            _logger.debug("Selection changed")
            self.currentROI = ddict['key']
            self._updateMarkers()
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

    def _updateMarkers(self):
        self._clearMarkers()
        if self._showAllMarkers is True:

            if self._middleROIMarkerFlag:
                self.plot.remove('ROI middle', kind='marker')
            roiList, roiDict = self.roiTable.getROIListAndDict()

            for roi in roiDict:
                fromdata = roiDict[roi]['from']
                todata = roiDict[roi]['to']
                _name = roi
                _nameRoiMin = _name + ' ROI min'
                _nameRoiMax = _name + ' ROI max'
                self.plot.remove(_nameRoiMin, kind='marker')
                self.plot.remove(_nameRoiMax, kind='marker')
                if _show:
                    draggable = False if _name == 'ICR' else True
                    color = 'blue'
                    self.plot.addXMarker(fromdata,
                                         legend=_nameRoiMin,
                                         text=_nameRoiMin,
                                         color=color,
                                         draggable=draggable)
                    self.plot.addXMarker(todata,
                                         legend=_nameRoiMax,
                                         text=_nameRoiMax,
                                         color=color,
                                         draggable=draggable)

        else:
            if not self.currentROI or not self.plot:
                return
            assert isinstance(self.currentROI, _ROI)

            self.plot.addXMarker(self.currentROI.fromdata,
                                 legend='ROI min',
                                 text='ROI min',
                                 color=self.currentROI.color,
                                 draggable=self.currentROI.draggable)
            self.plot.addXMarker(self.currentROI.todata,
                                 legend='ROI max',
                                 text='ROI max',
                                 color=self.currentROI.color,
                                 draggable=self.currentROI.draggable)
            if self.currentROI.draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.plot.addXMarker(pos,
                                     legend='ROI middle',
                                     text="",
                                     color='yellow',
                                     draggable=self.currentROI.draggable)

    def _clearMarkers(self):
        if not self.plot:
            return

        self.plot.remove('ROI min', kind='marker')
        self.plot.remove('ROI max', kind='marker')
        self.plot.remove('ROI middle', kind='marker')

        roilist, roidict = self.roiTable.getROIListAndDict()
        for roi in roidict:
            fromdata = roidict[roi]['from']
            todata = roidict[roi]['to']
            _name = roi
            _nameRoiMin = _name + ' ROI min'
            _nameRoiMax = _name + ' ROI max'
            self.plot.remove(_nameRoiMin, kind='marker')
            self.plot.remove(_nameRoiMax, kind='marker')

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

    @deprecation.deprecated(replacement="calculateRois",
                            reason="CamelCase convention")
    def calculateROIs(self, *args, **kw):
        self.calculateRois(*args, **kw)

    def calculateRois(self, roiList=None, roiDict=None):
        """Compute ROI information"""
        if roiList is None or roiDict is None:
            roiList, roiDict = self.roiTable.getROIListAndDict()

        for roi in roiList:
            assert isinstance(roi, _ROI)
        for roiName, roi in roiDict:
            assert type(roiName) is str
            assert isinstance(roi, _ROI)

        plot = self.getPlotWidget()
        if plot is None:
            activeCurve = None
        else:
            activeCurve = plot.getActiveCurve(just_legend=False)

        if activeCurve is None:
            xproc = None
            yproc = None
            self.setHeader()
        else:
            x = activeCurve.getXData(copy=False)
            y = activeCurve.getYData(copy=False)
            legend = activeCurve.getLegend()
            idx = numpy.argsort(x, kind='mergesort')
            xproc = numpy.take(x, idx)
            yproc = numpy.take(y, idx)
            self.setHeader('ROIs of %s' % legend)

        for roi in roiList:
            key = roi.name
            if key == 'ICR':
                if xproc is not None:
                    roiDict[key].fromdata = xproc.min()
                    roiDict[key].todata = xproc.max()
                else:
                    roiDict[key].fromdata = 0
                    roiDict[key].todata = -1
            fromData = roiDict[key].fromdata
            toData = roiDict[key].todata
            if xproc is not None:
                idx = numpy.nonzero((fromData <= xproc) &
                                    (xproc <= toData))[0]
                if len(idx):
                    xw = xproc[idx]
                    yw = yproc[idx]
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

        self.roiTable.fillFromROIDict(
                roilist=roiList,
                roidict=roiDict,
                currentroi=self.currentROI if self.currentROI in roiList else None)

    def _emitCurrentROISignal(self):
        ddict = {}
        ddict['event'] = "currentROISignal"
        _roiList, roiDict = self.roiTable.getROIListAndDict()
        if self.activeROI in roiDict:
            ddict['ROI'] = roiDict[self.activeROI]
        else:
            self.activeROI = None
        ddict['current'] = self.activeROI
        self.sigROISignal.emit(ddict)

    def _handleROIMarkerEvent(self, ddict):
        """Handle plot signals related to marker events."""
        if ddict['event'] == 'markerMoved':

            label = ddict['label']
            if label not in ['ROI min', 'ROI max', 'ROI middle']:
                return

            roiList, roiDict = self.roiTable.getROIListAndDict()
            if self.currentROI is None:
                return
            if self.currentROI not in roiDict:
                return

            plot = self.getPlotWidget()
            if plot is None:
                return

            x = ddict['x']

            if label == 'ROI min':
                roiDict[self.currentROI]['from'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +
                                 roiDict[self.currentROI]['from'])
                    plot.addXMarker(pos,
                                    legend='ROI middle',
                                    text='',
                                    color='yellow',
                                    draggable=True)
            elif label == 'ROI max':
                roiDict[self.currentROI]['to'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +
                                 roiDict[self.currentROI]['from'])
                    plot.addXMarker(pos,
                                    legend='ROI middle',
                                    text='',
                                    color='yellow',
                                    draggable=True)
            elif label == 'ROI middle':
                delta = x - 0.5 * (roiDict[self.currentROI]['from'] +
                                   roiDict[self.currentROI]['to'])
                roiDict[self.currentROI]['from'] += delta
                roiDict[self.currentROI]['to'] += delta
                plot.addXMarker(roiDict[self.currentROI]['from'],
                                legend='ROI min',
                                text='ROI min',
                                color='blue',
                                draggable=True)
                plot.addXMarker(roiDict[self.currentROI]['to'],
                                legend='ROI max',
                                text='ROI max',
                                color='blue',
                                draggable=True)
            else:
                return
            self.calculateRois(roiList, roiDict)
            self._emitCurrentROISignal()

    def _visibilityChangedHandler(self, visible):
        """Handle widget's visibility updates.

        It is connected to plot signals only when visible.
        """
        plot = self.getPlotWidget()

        if visible:
            if not self._isInit:
                # Deferred ROI widget init finalization
                self._finalizeInit()

            if not self._isConnected and plot is not None:
                plot.sigPlotSignal.connect(self._handleROIMarkerEvent)
                plot.sigActiveCurveChanged.connect(
                    self._activeCurveChanged)
                self._isConnected = True

                self.calculateRois()
        else:
            if self._isConnected:
                if plot is not None:
                    plot.sigPlotSignal.disconnect(self._handleROIMarkerEvent)
                    plot.sigActiveCurveChanged.disconnect(
                        self._activeCurveChanged)
                self._isConnected = False

    def _activeCurveChanged(self, *args):
        """Recompute ROIs when active curve changed."""
        self.calculateRois()

    def _finalizeInit(self):
        self._isInit = True
        self.sigROIWidgetSignal.connect(self._roiSignal)
        # initialize with the ICR if no ROi existing yet
        if len(self.getRois()) is 0:
            self._roiSignal({'event': "AddROI"})

    def showAllMarkers(self, _show=True):
        if self._showAllMarkers == _show:
            return

        self._showAllMarkers = _show
        self._updateMarkers()


class ROITable(qt.QTableWidget):
    """Table widget displaying ROI information.

    See :class:`QTableWidget` for constructor arguments.
    """

    sigROITableSignal = qt.Signal(object)
    """Signal of ROI table modifications.
    """

    COLUMNS_INDEX = OrderedDict([
        ('ROI', 0),
        ('Type', 1),
        ('From', 2),
        ('To', 3),
        ('Raw Counts', 4),
        ('Net Counts', 5),
        ('ROI Object', 6)
    ])

    COLUMNS = list(COLUMNS_INDEX.keys())

    INFO_NOT_FOUND = '????????'


    def __init__(self, parent=None, plot=None, rois=None):
        super(ROITable, self).__init__(parent)
        self._RoiToItems = {}
        self.roidict = {}
        self.setColumnCount(len(self.COLUMNS))
        self.setPlot(plot)
        self.__setTooltip()

    def clear(self):
        self._RoiToItems = {}
        qt.QTableWidget.clear(self)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(
            qt.QHeaderView.ResizeToContents)
        self.setColumnHidden(self.COLUMNS_INDEX['ROI Object'], True)
        self.sortByColumn(2, qt.Qt.AscendingOrder)

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

    def addRoi(self, roi):
        assert isinstance(roi, _ROI)
        self.setRowCount(self.rowCount() + 1)
        itemName = qt.QTableWidgetItem(roi.name, type=qt.QTableWidgetItem.Type)
        if roi.name.upper() in ('ICR', 'DEFAULT'):
            itemName.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
        else:
            itemName.setFlags(qt.Qt.ItemIsSelectable |
                              qt.Qt.ItemIsEnabled |
                              qt.Qt.ItemIsEditable)
        itemType = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemType.setFlags((qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled))
        itemFrom = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemFrom.setFlags(qt.Qt.ItemIsSelectable |
                          qt.Qt.ItemIsEnabled |
                          qt.Qt.ItemIsEditable)
        itemTo = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemTo.setFlags(qt.Qt.ItemIsSelectable |
                        qt.Qt.ItemIsEnabled |
                        qt.Qt.ItemIsEditable)
        itemRawCounts = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemNetCounts = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemRoi = qt.QTableWidgetItem(type=qt.QTableWidgetItem.Type)
        itemRoi.setData(qt.QTableWidgetItem.Type, weakref.ref(roi))

        indexTable = self.rowCount() - 1
        self.setItem(indexTable, self.COLUMNS_INDEX['ROI'], itemName)
        self.setItem(indexTable, self.COLUMNS_INDEX['Type'], itemType)
        self.setItem(indexTable, self.COLUMNS_INDEX['From'], itemFrom)
        self.setItem(indexTable, self.COLUMNS_INDEX['To'], itemTo)
        self.setItem(indexTable, self.COLUMNS_INDEX['Raw Counts'], itemRawCounts)
        self.setItem(indexTable, self.COLUMNS_INDEX['Net Counts'], itemNetCounts)
        self.setItem(indexTable, self.COLUMNS_INDEX['ROI Object'], itemRoi)

        self._RoiToItems[roi.name] = itemName
        self.roidict[roi.name] = weakref.ref(roi)
        self.activeRoi = weakref.ref(roi)
        self._updateRoiInfo(roi.name)
        callback = functools.partial(WeakMethodProxy(self._updateRoiInfo),
                                     roi.name)
        roi.sigChanged.connect(callback)

    def _computeRawAndNetCounts(self, roi):
        if not self.plot:
            return None, None

        activeCurve = self.plot.getActiveCurve(just_legend=False)
        if activeCurve is None:
            if roiName == 'ICR':
                assert roi.editable is False
                roi.fromdata = 0
                roi.todata = -1
            return None, None

        x = activeCurve.getXData(copy=False)
        y = activeCurve.getYData(copy=False)
        idx = numpy.argsort(x, kind='mergesort')
        xproc = numpy.take(x, idx)
        yproc = numpy.take(y, idx)

        # update from and to only in the case of the non editable 'ICR' ROI
        if roiName == 'ICR':
            roi.fromdata = xproc.min()
            roi.todata = xproc.max()

        idx = numpy.nonzero((roi.fromdata <= xproc) & (xproc <= roi.todata))[0]
        if len(idx):
            xw = xproc[idx]
            yw = yproc[idx]
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
            return rawCounts, netCounts
        else:
            return 0.0, 0.0

    def _updateRoiInfo(self, roiName):
        if roiName not in self.roidict:
            return
        if not self.roidict[roiName]():
            del self.roidict[roiName]
            self._RoiToItems[roiName]
            return
        roi = self.roidict[roiName]()
        assert roi.name in self._RoiToItems

        itemName = self._RoiToItems[roi.name]
        itemType = self.item(itemName.row(), self.COLUMNS_INDEX['Type'])
        itemType.setText(roi.type or self.INFO_NOT_FOUND)

        itemFrom = self.item(itemName.row(), self.COLUMNS_INDEX['From'])
        fromdata = str(roi.fromdata) if roi.fromdata is not None else self.INFO_NOT_FOUND
        itemFrom.setText(fromdata)

        itemTo = self.item(itemName.row(), self.COLUMNS_INDEX['To'])
        todata = str(roi.todata) if roi.todata is not None else self.INFO_NOT_FOUND
        itemTo.setText(todata)

        self._computeRawAndNetCounts(roi)
        itemRawCounts = self.item(itemName.row(), self.COLUMNS_INDEX['Raw Counts'])
        rawCounts = str(roi.rawCounts) if roi.rawCounts is not None else self.INFO_NOT_FOUND
        itemRawCounts.setText(rawCounts)

        itemNetCounts = self.item(itemName.row(), self.COLUMNS_INDEX['Net Counts'])
        netCounts = str(roi.netCounts) if roi.netCounts is not None else self.INFO_NOT_FOUND
        itemNetCounts.setText(netCounts)

    def currentChanged(self, current, previous):
        if previous and current.row() != previous.row():
            # note: roi is registred as a weak ref
            self.activeRoi = self.item(current.row(), self.COLUMNS_INDEX['ROI Object']).data(
                qt.QTableWidgetItem.Type)()
            assert self.activeRoi
            # self._updateMarkers()
        qt.QTableWidget.currentChanged(self, current, previous)
    #
    # def _cellClickedSlot(self, *var, **kw):
    #     # selection changed event, get the current selection
    #     row = self.currentRow()
    #     col = self.currentColumn()
    #     if row >= 0 and row < len(self.roilist):
    #         item = self.item(row, 0)
    #         text = '' if item is None else str(item.text())
    #         self.roilist[row] = text
    #         self._emitSelectionChangedSignal(row, col)
    #
    # def _rowChangedSlot(self, row):
    #     self._emitSelectionChangedSignal(row, 0)
    #
    # def _cellChangedSlot(self, row, col):
    #     _logger.debug("_cellChangedSlot(%d, %d)", row, col)
    #     if self.building:
    #         return
    #     if col == 0:
    #         self.nameSlot(row, col)
    #     else:
    #         self._valueChanged(row, col)
    #
    # def _valueChanged(self, row, col):
    #     if col not in [2, 3]:
    #         return
    #     item = self.item(row, col)
    #     if item is None:
    #         return
    #     text = str(item.text())
    #     try:
    #         value = float(text)
    #     except:
    #         return
    #     if row >= len(self.roilist):
    #         _logger.debug("deleting???")
    #         return
    #     item = self.item(row, 0)
    #     if item is None:
    #         text = ""
    #     else:
    #         text = str(item.text())
    #     if not len(text):
    #         return
    #     if col == 2:
    #         self.roidict[text]['from'] = value
    #     elif col == 3:
    #         self.roidict[text]['to'] = value
    #     self._emitSelectionChangedSignal(row, col)
    #
    # def nameSlot(self, row, col):
    #     if col != 0:
    #         return
    #     if row >= len(self.roilist):
    #         _logger.debug("deleting???")
    #         return
    #     item = self.item(row, col)
    #     if item is None:
    #         text = ""
    #     else:
    #         text = str(item.text())
    #     if len(text) and (text not in self.roilist):
    #         old = self.roilist[row]
    #         self.roilist[row] = text
    #         self.roidict[text] = {}
    #         self.roidict[text].update(self.roidict[old])
    #         del self.roidict[old]
    #         self._emitSelectionChangedSignal(row, col)
    #
    # def _emitSelectionChangedSignal(self, row, col):
    #     ddict = {}
    #     ddict['event'] = "selectionChanged"
    #     ddict['row'] = row
    #     ddict['col'] = col
    #     ddict['roi'] = self.roidict[self.roilist[row]]
    #     ddict['key'] = self.roilist[row]
    #     ddict['colheader'] = self.labels[col]
    #     ddict['rowheader'] = "%d" % row
    #     self.sigROITableSignal.emit(ddict)


class _ROI(qt.QObject):

    sigChanged = qt.Signal()

    def __init__(self, name, fromdata=None, todata=None):
        qt.QObject.__init__(self)
        assert type(name) is str
        self.name = name
        self.fromdata = fromdata
        self.todata = todata
        self.marker = None
        self.draggable = False
        self.color = 'blue'
        self.type = 'Default'
        self.editable=True
        self.rawCounts = None
        self.netCounts = None

    def fromDict(self, ddict):
        pass

    def toDict(self):
        return {
            'type': self.type,
            'name': self.name,
            'from': self.fromdata,
            'to': self.todata,
        }


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

        self.roiWidget = CurvesROIWidget(self, name, plot=plot)
        """Main widget of type :class:`CurvesROIWidget`"""

        # convenience methods to offer a simpler API allowing to ignore
        # the details of the underlying implementation
        # (ALL DEPRECATED)
        self.calculateROIs = self.calculateRois = self.roiWidget.calculateRois
        self.setRois = self.roiWidget.setRois
        self.getRois = self.roiWidget.getRois
        self.roiWidget.sigROISignal.connect(self._forwardSigROISignal)
        self.currentROI = self.roiWidget.currentROI

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(self.roiWidget)

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

    def showEvent(self, event):
        """Make sure this widget is raised when it is shown
        (when it is first created as a tab in PlotWindow or when it is shown
        again after hiding).
        """
        self.raise_()
        qt.QDockWidget.showEvent(self, event)
