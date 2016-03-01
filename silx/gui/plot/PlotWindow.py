#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This window handles plugins and adds a toolbar to the PlotWidget.

Currently the only dependency on PyMca is through the Icons.

"""
import copy
import sys
import os
import time
import traceback
import numpy
from numpy import argsort, nonzero, take
from . import LegendSelector
from .ObjectPrintConfigurationDialog import ObjectPrintConfigurationDialog
from . import McaROIWidget
from . import PlotWidget
from . import MaskImageTools
from . import RenameCurveDialog

try:
    from . import ColormapDialog
    COLORMAP_DIALOG = True
except:
    COLORMAP_DIALOG = False

from .PyMca_Icons import IconDict
from PyMca5.PyMcaGui import PyMcaQt as qt

if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str

QTVERSION = qt.qVersion()

DEBUG = 0

class PlotWindow(PlotWidget.PlotWidget):
    sigROISignal = qt.pyqtSignal(object)
    sigIconSignal = qt.pyqtSignal(object)
    sigColormapChangedSignal = qt.pyqtSignal(object)
    DEFAULT_COLORMAP_INDEX = MaskImageTools.DEFAULT_COLORMAP_INDEX
    DEFAULT_COLORMAP_LOG_FLAG = MaskImageTools.DEFAULT_COLORMAP_LOG_FLAG

    def __init__(self, parent=None, backend=None, plugins=True, newplot=False,
                 control=False, position=False, **kw):
        super(PlotWindow, self).__init__(parent=parent, backend=backend)
        self.pluginsIconFlag = plugins
        self.newplotIconsFlag = newplot
        self.setWindowType(None)      # None, "SCAN", "MCA"
        self._initIcons()
        self._buildToolBar(kw)
        self.setIconSize(qt.QSize(16, 16))
        self._toggleCounter = 0
        self._keepDataAspectRatioFlag = False
        self.gridLevel = 0
        self.legendWidget = None
        self.usePlotBackendColormap = False  # Toggle usage of backend colormap
        self.setCallback(self.graphCallback)
        if control or position:
            self._buildGraphBottomWidget(control, position)
            self._controlMenu = None

        # default print configuration (uses full page)
        self._printMenu = None
        self._printConfigurationDialog = None
        self._printConfiguration = {"xOffset": 0.1,
                                    "yOffset": 0.1,
                                    "width": 0.9,
                                    "height": 0.9,
                                    "units": "page",
                                    "keepAspectRatio": True}
        # own save action
        self.enableOwnSave(True)

        # activeCurve handling
        self.enableActiveCurveHandling(True)
        self.setActiveCurveColor('black')

        # default ROI handling
        self.roiWidget = None
        self._middleROIMarkerFlag = False

        #colormap handling
        self.colormapDialog = None
        self.colormap = None

    def enableOwnSave(self, flag=True):
        if flag:
            self._ownSave = True
        else:
            self._ownSave = False

    def _buildGraphBottomWidget(self, control, position):
        widget = self.centralWidget()
        self.graphBottom = qt.QWidget(widget)
        self.graphBottomLayout = qt.QHBoxLayout(self.graphBottom)
        self.graphBottomLayout.setContentsMargins(0, 0, 0, 0)
        self.graphBottomLayout.setSpacing(0)

        if control:
            self.graphControlButton = qt.QPushButton(self.graphBottom)
            self.graphControlButton.setText("Options")
            self.graphControlButton.setAutoDefault(False)
            self.graphBottomLayout.addWidget(self.graphControlButton)
            self.graphControlButton.clicked.connect(self._graphControlClicked)

        if position:
            label=qt.QLabel(self.graphBottom)
            label.setText('<b>X:</b>')
            self.graphBottomLayout.addWidget(label)

            self._xPos = qt.QLineEdit(self.graphBottom)
            self._xPos.setText('------')
            self._xPos.setReadOnly(1)
            self._xPos.setFixedWidth(self._xPos.fontMetrics().width('##############'))
            self.graphBottomLayout.addWidget(self._xPos)

            label=qt.QLabel(self.graphBottom)
            label.setText('<b>Y:</b>')
            self.graphBottomLayout.addWidget(label)

            self._yPos = qt.QLineEdit(self.graphBottom)
            self._yPos.setText('------')
            self._yPos.setReadOnly(1)
            self._yPos.setFixedWidth(self._yPos.fontMetrics().width('##############'))
            self.graphBottomLayout.addWidget(self._yPos)
            self.graphBottomLayout.addWidget(qt.HorizontalSpacer(self.graphBottom))
        widget.layout().addWidget(self.graphBottom)

    def setPrintMenu(self, menu):
        self._printMenu = menu

    def setWindowType(self, wtype=None):
        if wtype not in [None, "SCAN", "MCA"]:
            print("Unsupported window type. Default to None")
        self._plotType = wtype

    def _graphControlClicked(self):
        if self._controlMenu is None:
            #create a default menu
            controlMenu = qt.QMenu()
            controlMenu.addAction(QString("Show/Hide Legends"),
                                       self.toggleLegendWidget)
            controlMenu.addAction(QString("Toggle Crosshair"),
                                       self.toggleCrosshairCursor)
            controlMenu.addAction(QString("Toggle Arrow Keys Panning"),
                                       self.toggleArrowKeysPanning)
            controlMenu.exec_(self.cursor().pos())
        else:
            self._controlMenu.exec_(self.cursor().pos())

    def setControlMenu(self, menu=None):
        self._controlMenu = menu

    def _initIcons(self):
        self.normalIcon	= qt.QIcon(qt.QPixmap(IconDict["normal"]))
        self.zoomIcon	= qt.QIcon(qt.QPixmap(IconDict["zoom"]))
        self.roiIcon	= qt.QIcon(qt.QPixmap(IconDict["roi"]))
        self.peakIcon	= qt.QIcon(qt.QPixmap(IconDict["peak"]))
        self.energyIcon = qt.QIcon(qt.QPixmap(IconDict["energy"]))

        self.zoomResetIcon	= qt.QIcon(qt.QPixmap(IconDict["zoomreset"]))
        self.roiResetIcon	= qt.QIcon(qt.QPixmap(IconDict["roireset"]))
        self.peakResetIcon	= qt.QIcon(qt.QPixmap(IconDict["peakreset"]))
        self.refreshIcon	= qt.QIcon(qt.QPixmap(IconDict["reload"]))

        self.logxIcon	= qt.QIcon(qt.QPixmap(IconDict["logx"]))
        self.logyIcon	= qt.QIcon(qt.QPixmap(IconDict["logy"]))
        self.xAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["xauto"]))
        self.yAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["yauto"]))
        self.gridIcon	= qt.QIcon(qt.QPixmap(IconDict["grid16"]))
        self.hFlipIcon	= qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"]))
        self.togglePointsIcon = qt.QIcon(qt.QPixmap(IconDict["togglepoints"]))

        self.solidCircleIcon = qt.QIcon(qt.QPixmap(IconDict["solidcircle"]))
        self.solidEllipseIcon = qt.QIcon(qt.QPixmap(IconDict["solidellipse"]))

        self.fitIcon	= qt.QIcon(qt.QPixmap(IconDict["fit"]))
        self.searchIcon	= qt.QIcon(qt.QPixmap(IconDict["peaksearch"]))

        self.averageIcon	= qt.QIcon(qt.QPixmap(IconDict["average16"]))
        self.deriveIcon	= qt.QIcon(qt.QPixmap(IconDict["derive"]))
        self.smoothIcon     = qt.QIcon(qt.QPixmap(IconDict["smooth"]))
        self.swapSignIcon	= qt.QIcon(qt.QPixmap(IconDict["swapsign"]))
        self.yMinToZeroIcon	= qt.QIcon(qt.QPixmap(IconDict["ymintozero"]))
        self.subtractIcon	= qt.QIcon(qt.QPixmap(IconDict["subtract"]))

        self.colormapIcon   = qt.QIcon(qt.QPixmap(IconDict["colormap"]))
        self.imageIcon     = qt.QIcon(qt.QPixmap(IconDict["image"]))
        self.eraseSelectionIcon = qt.QIcon(qt.QPixmap(IconDict["eraseselect"]))
        self.rectSelectionIcon  = qt.QIcon(qt.QPixmap(IconDict["boxselect"]))
        self.brushSelectionIcon = qt.QIcon(qt.QPixmap(IconDict["brushselect"]))
        self.brushIcon          = qt.QIcon(qt.QPixmap(IconDict["brush"]))
        self.additionalIcon     = qt.QIcon(qt.QPixmap(IconDict["additionalselect"]))
        self.polygonIcon = qt.QIcon(qt.QPixmap(IconDict["polygon"]))

        self.printIcon	= qt.QIcon(qt.QPixmap(IconDict["fileprint"]))
        self.saveIcon	= qt.QIcon(qt.QPixmap(IconDict["filesave"]))

        self.pluginIcon     = qt.QIcon(qt.QPixmap(IconDict["plugin"]))

    def _buildToolBar(self, kw=None):
        if kw is None:
            kw = {}
        self.toolBar = qt.QToolBar(self)
        self.toolBarActionsDict = {}
        #Autoscale
        self._addToolButton(self.zoomResetIcon,
                            self._zoomReset,
                            'Auto-Scale the Graph',
                            key=None)

        #y Autoscale
        self.yAutoScaleButton = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True,
                            key=None)
        self.yAutoScaleButton.setChecked(True)
        self.yAutoScaleButton.setDown(True)


        #x Autoscale
        self.xAutoScaleButton = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True,
                            key=None)
        self.xAutoScaleButton.setChecked(True)
        self.xAutoScaleButton.setDown(True)

        #y Logarithmic
        if kw.get('logy', True):
            self.yLogButton = self._addToolButton(self.logyIcon,
                                self._toggleLogY,
                                'Toggle Logarithmic Y Axis (On/Off)',
                                toggle = True,
                                key='logy')
            self.yLogButton.setChecked(False)
            self.yLogButton.setDown(False)

        #x Logarithmic
        if kw.get('logx', True):
            self.xLogButton = self._addToolButton(self.logxIcon,
                                self._toggleLogX,
                                'Toggle Logarithmic X Axis (On/Off)',
                                toggle = True,
                                key='logx')
            self.xLogButton.setChecked(False)
            self.xLogButton.setDown(False)

        #Aspect ratio
        if kw.get('aspect', False):
            self.aspectButton = self._addToolButton(self.solidCircleIcon,
                                self._aspectButtonSignal,
                                'Keep data aspect ratio',
                                toggle = False,
                                key='aspect')
            self.aspectButton.setChecked(False)
            #self.aspectButton.setDown(False)

        #colormap
        if kw.get('colormap', False):
            tb = self._addToolButton(self.colormapIcon,
                                     self._colormapIconSignal,
                                    'Change Colormap',
                                    key='colormap')
            self.colormapToolButton = tb

        if kw.get('normal', False):
            tb = self._addToolButton(self.normalIcon,
                                     self._normalIconSignal,
                                    'Set normal (default) mode',
                                    key='normal')
            self.normalToolButton = tb

        # image and selection related icons
        if kw.get('imageIcons', False) or kw.get('imageicons', False):
            tb = self._addToolButton(self.imageIcon,
                                     self._imageIconSignal,
                                     'Reset',
                                      key='image')
            self.imageToolButton = tb

            tb = self._addToolButton(self.eraseSelectionIcon,
                                     self._eraseSelectionIconSignal,
                                     'Erase Selection',
                                     key="erase")
            self.eraseSelectionToolButton = tb

            tb = self._addToolButton(self.rectSelectionIcon,
                                     self._rectSelectionIconSignal,
                                     'Rectangular Selection',
                                     key="rectangle")
            self.rectSelectionToolButton = tb

            tb = self._addToolButton(self.brushSelectionIcon,
                                     self._brushSelectionIconSignal,
                                     'Brush Selection',
                                     key="brushSelection")
            self.brushSelectionToolButton = tb


            tb = self._addToolButton(self.brushIcon,
                                     self._brushIconSignal,
                                     'Select Brush',
                                     key="brush")
            self.brushToolButton = tb

            if kw.get("polygon", False):
                tb = self._addToolButton(self.polygonIcon,
                                self._polygonIconSignal,
                                'Polygon selection',
                                 key="polygon")
                self.polygonSelectionToolButton = tb
            tb = self._addToolButton(self.additionalIcon,
                                     self._additionalIconSignal,
                                     'Additional Selections Menu',
                                     key="additional")
            self.additionalSelectionToolButton = tb
        else:
            if kw.get("polygon", False):
                tb = self._addToolButton(self.polygonIcon,
                                self._polygonIconSignal,
                                'Polygon selection',
                                key="polygon")
                self.polygonSelectionToolButton = tb
            self.imageToolButton = None

        #flip
        if kw.get('flip', False) or kw.get('hflip', False):
            tb = self._addToolButton(self.hFlipIcon,
                                 self._hFlipIconSignal,
                                 'Flip Horizontal',
                                 key="hflip")
            self.hFlipToolButton = tb

        #grid
        if kw.get('grid', True):
            tb = self._addToolButton(self.gridIcon,
                                self.changeGridLevel,
                                'Change Grid',
                                toggle = False,
                                key="grid")
            self.gridTb = tb


        #toggle Points/Lines
        if kw.get('togglePoints', True):
            tb = self._addToolButton(self.togglePointsIcon,
                                     self._togglePointsSignal,
                                     'Toggle Points/Lines',
                                     key="togglePoints")

        #energy icon
        if kw.get('energy', False):
            self.energyButton = self._addToolButton(self.energyIcon,
                            self._energyIconSignal,
                            'Toggle Energy Axis (On/Off)',
                            toggle=True,
                            key="energy")

        #roi icon
        if kw.get('roi', False):
            self.roiButton = self._addToolButton(self.roiIcon,
                                         self.__toggleROI,
                                         'Show/Hide ROI widget',
                                         toggle=False,
                                         key="roi")
            self.currentROI = None
            self.middleROIMarkerFlag = False

        #fit icon
        if kw.get('fit', False):
            self.fitButton = self._addToolButton(self.fitIcon,
                                         self._fitIconSignal,
                                         'Fit of Active Curve',
                                         key="fit")

        if self.newplotIconsFlag:
            tb = self._addToolButton(self.averageIcon,
                                self._averageIconSignal,
                                 'Average Plotted Curves')

            tb = self._addToolButton(self.deriveIcon,
                                self._deriveIconSignal,
                                 'Take Derivative of Active Curve')

            tb = self._addToolButton(self.smoothIcon,
                                 self._smoothIconSignal,
                                 'Smooth Active Curve')

            tb = self._addToolButton(self.swapSignIcon,
                                self._swapSignIconSignal,
                                'Multiply Active Curve by -1')

            tb = self._addToolButton(self.yMinToZeroIcon,
                                self._yMinToZeroIconSignal,
                                'Force Y Minimum to be Zero')

            tb = self._addToolButton(self.subtractIcon,
                                self._subtractIconSignal,
                                'Subtract Active Curve')

        #save
        infotext = 'Save Active Curve or Widget'
        tb = self._addToolButton(self.saveIcon,
                                 self._saveIconSignal,
                                 infotext)

        if self.pluginsIconFlag:
            infotext = "Call/Load 1D Plugins"
            tb = self._addToolButton(self.pluginIcon,
                                 self._pluginClicked,
                                 infotext)

        self.toolBar.addWidget(qt.HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self._printGraph,
                                 'Prints the Graph')

        self.addToolBar(self.toolBar)

    def _printGraph(self):
        if self._printMenu is None:
            printMenu = qt.QMenu()
            #printMenu.addAction(QString("Select printer"),
            #                        self._printerSelect)
            printMenu.addAction(QString("Customize printing"),
                            self._getPrintConfigurationFromDialog)
            printMenu.addAction(QString("Print"),
                                       self.printGraph)
            printMenu.exec_(self.cursor().pos())
        else:
            self._printMenu.exec_(self.cursor().pos())

    def printGraph(self, *var, **kw):
        config = self.getPrintConfiguration()
        PlotWidget.PlotWidget.printGraph(self,
                            width=config['width'],
                            height=config['height'],
                            xOffset=config['xOffset'],
                            yOffset=config['yOffset'],
                            units=config['units'],
                            keepAspectRatio=config['keepAspectRatio'],
                            printer=self._printer)

    def setPrintConfiguration(self, configuration, printer=None):
        for key in self._printConfiguration:
            if key in configuration:
                self._printConfiguration[key] = configuration[key]
        if printer is not None:
            # printer should be a global thing ...
            self._printer = printer

    def getPrintConfiguration(self, dialog=False):
        if dialog:
            self._getPrintConfigurationFromDialog()
        return copy.deepcopy(self._printConfiguration)


    def _getPrintConfigurationFromDialog(self):
        if self._printConfigurationDialog is None:
            self._printConfigurationDialog = \
                                ObjectPrintConfigurationDialog(self)
        oldConfig = self.getPrintConfiguration()
        self._printConfigurationDialog.setPrintConfiguration(oldConfig,
                                                    printer=self._printer)
        if self._printConfigurationDialog.exec_():
            self.setPrintConfiguration(\
                self._printConfigurationDialog.getPrintConfiguration())

    def _addToolButton(self, icon, action, tip, toggle=None, key=None):
        tb      = qt.QToolButton(self.toolBar)
        tb.setIcon(icon)
        tb.setToolTip(tip)
        if toggle is not None:
            if toggle:
                tb.setCheckable(1)
        qtAction = self.toolBar.addWidget(tb)
        if key is not None:
            if not hasattr(self, "toolBarActionsDict"):
                self.toolBarActionsDict = {}
            self.toolBarActionsDict[key] = qtAction
        tb.clicked.connect(action)
        return tb

    def setToolBarActionVisible(self, action, visible=True):
        if hasattr(self, "toolBarActionsDict"):
            for key in self.toolBarActionsDict:
                if hasattr(key, "lower") and hasattr(action, "lower"):
                    if key.lower() == action.lower():
                        self.toolBarActionsDict[key].setVisible(visible)
                        return
                elif key == action:
                    self.toolBarActionsDict[key].setVisible(visible)
                    return
        if DEBUG:
            print("Unhandled action %s" % action)

    def _aspectButtonSignal(self):
        if DEBUG:
            print("_aspectButtonSignal")
        if self._keepDataAspectRatioFlag:
            self.keepDataAspectRatio(False)
        else:
            self.keepDataAspectRatio(True)

    def keepDataAspectRatio(self, flag=True):
        if flag:
            self._keepDataAspectRatioFlag = True
            self.aspectButton.setIcon(self.solidEllipseIcon)
            self.aspectButton.setToolTip("Set free data aspect ratio")
        else:
            self._keepDataAspectRatioFlag = False
            self.aspectButton.setIcon(self.solidCircleIcon)
            self.aspectButton.setToolTip("Keep data aspect ratio")
        super(PlotWindow, self).keepDataAspectRatio(self._keepDataAspectRatioFlag)

    def _zoomReset(self):
        if DEBUG:
            print("_zoomReset")
        self.resetZoom()

    def _yAutoScaleToggle(self):
        if DEBUG:
            print("toggle Y auto scaling")
        if self.isYAxisAutoScale():
            self.setYAxisAutoScale(False)
            self.yAutoScaleButton.setDown(False)
            self.yAutoScaleButton.setChecked(False)
            ymin, ymax = self.getGraphYLimits()
            self.setGraphYLimits(ymin, ymax)
        else:
            self.setYAxisAutoScale(True)
            self.yAutoScaleButton.setDown(True)
            self.resetZoom()

    def _xAutoScaleToggle(self):
        if DEBUG:
            print("toggle X auto scaling")
        if self.isXAxisAutoScale():
            self.setXAxisAutoScale(False)
            self.xAutoScaleButton.setDown(False)
            self.xAutoScaleButton.setChecked(False)
            xmin, xmax = self.getGraphXLimits()
            self.setGraphXLimits(xmin, xmax)
        else:
            self.setXAxisAutoScale(True)
            self.xAutoScaleButton.setDown(True)
            self.resetZoom()

    def _toggleLogX(self):
        if DEBUG:
            print("toggle logarithmic X scale")
        if self.isXAxisLogarithmic():
            self.setXAxisLogarithmic(False)
        else:
            self.setXAxisLogarithmic(True)

    def setXAxisLogarithmic(self, flag=True):
        super(PlotWindow, self).setXAxisLogarithmic(flag)
        self.xLogButton.setChecked(flag)
        self.xLogButton.setDown(flag)
        self.replot()
        self.resetZoom()

    def _toggleLogY(self):
        if DEBUG:
            print("_toggleLogY")
        if self.isYAxisLogarithmic():
            self.setYAxisLogarithmic(False)
        else:
            self.setYAxisLogarithmic(True)

    def setYAxisLogarithmic(self, flag=True):
        super(PlotWindow, self).setYAxisLogarithmic(flag)
        self.yLogButton.setChecked(flag)
        self.yLogButton.setDown(flag)
        # TODO: setYAxisLogarithmic already calls replot
        # in addition resetZoom also does it
        self.replot()
        self.resetZoom()

    def _togglePointsSignal(self):
        if DEBUG:
            print("toggle points signal")
        self._toggleCounter = (self._toggleCounter + 1) % 3
        if self._toggleCounter == 1:
            self.setDefaultPlotLines(True)
            self.setDefaultPlotPoints(True)
        elif self._toggleCounter == 2:
            self.setDefaultPlotLines(False)
            self.setDefaultPlotPoints(True)
        else:
            self.setDefaultPlotLines(True)
            self.setDefaultPlotPoints(False)
        self.replot()

    def _hFlipIconSignal(self):
        if DEBUG:
            print("_hFlipIconSignal called")
        if self.isYAxisInverted():
            self.invertYAxis(False)
        else:
            self.invertYAxis(True)

    def _colormapIconSignal(self):
        image = self.getActiveImage()
        if image is None:
            return
        image, legend, info, pixmap = image[:4]

        if pixmap is not None:
            # image contains the data and pixmap contains its representation
            if self.colormapDialog is None:
                self._initColormapDialog(image)
            self.colormapDialog.show()
        elif image is not None and info["plot_colormap"] is not None:
            if self.colormapDialog is None:
                self._initColormapDialog(image, info['plot_colormap'])
            self.colormapDialog.show()
        else:
            print("No colormap to be handled")
            return

    def _initColormapDialog(self, imageData, colormap=None):
        """Set-up the colormap dialog default values.

        :param numpy.ndarray imageData: data used to init dialog.
        :param dict colormap: Description of the colormap as a dict.
                              See :class:`PlotBackend` for details.
                              If None, use default values.
        """
        if not COLORMAP_DIALOG:
            raise ImportError("ColormapDialog could not be imported")
        goodData = imageData[numpy.isfinite(imageData)]
        if goodData.size > 0:
            maxData = goodData.max()
            minData = goodData.min()
        else:
            qt.QMessageBox.critical(self, "No Data",
                "Image data does not contain any real value")
            return

        self.colormapDialog = ColormapDialog.ColormapDialog(self)

        if colormap is None:
            colormapIndex = self.DEFAULT_COLORMAP_INDEX
            if colormapIndex == 6:
                colormapIndex = 1
            self.colormapDialog.setColormap(colormapIndex)
            self.colormapDialog.setDataMinMax(minData, maxData)
            self.colormapDialog.setAutoscale(1)
            self.colormapDialog.setColormap(self.colormapDialog.colormapIndex)
            # linear or logarithmic
            self.colormapDialog.setColormapType(self.DEFAULT_COLORMAP_LOG_FLAG,
                                                update=False)
        else:
            # Set-up colormap dialog from provided colormap dict
            cmapList = ColormapDialog.colormapDictToList(colormap)
            index, autoscale, vMin, vMax, dataMin, dataMax, cmapType = cmapList
            self.colormapDialog.setColormap(index)
            self.colormapDialog.setAutoscale(autoscale)
            self.colormapDialog.setMinValue(vMin)
            self.colormapDialog.setMaxValue(vMax)
            self.colormapDialog.setDataMinMax(minData, maxData)
            self.colormapDialog.setColormapType(cmapType, update=False)

        self.colormap = self.colormapDialog.getColormap()  # Is it used?
        self.colormapDialog.setWindowTitle("Colormap Dialog")
        self.colormapDialog.sigColormapChanged.connect(\
                    self.updateActiveImageColormap)
        self.colormapDialog._update()

    def updateActiveImageColormap(self, colormap, replot=True):
        if len(colormap) == 1:
            colormap = colormap[0]
        # TODO: Once everything is ready to work with dict instead of
        # list, we can remove this translation
        plotBackendColormap = ColormapDialog.colormapListToDict(colormap)
        self.setDefaultColormap(plotBackendColormap)
        self.sigColormapChangedSignal.emit(plotBackendColormap)

        image = self.getActiveImage()
        if image is None:
            if self.colormapDialog is not None:
                self.colormapDialog.hide()
            return
        image, legend, info, pixmap = image[:4]
        if self.usePlotBackendColormap:
            self.addImage(image, legend=legend, info=info,
                          colormap=plotBackendColormap, replot=replot)
        else:
            if pixmap is None:
                if self.colormapDialog is not None:
                     self.colormapDialog.hide()
                return
            pixmap = MaskImageTools.getPixmapFromData(image, colormap)
            self.addImage(image, legend=legend, info=info,
                          pixmap=pixmap, replot=replot)

    def _normalIconSignal(self):
        if DEBUG:
            print("_normalIconSignal")
        # default implementation is setting zoom mode
        self.setZoomModeEnabled(True)

    def showRoiWidget(self, position=None):
        self._toggleROI(position)

    def __toggleROI(self):
        self._toggleROI()

    def _toggleROI(self, position=None):
        if DEBUG:
            print("_toggleROI called")
        if self.roiWidget is None:
            self.roiWidget = McaROIWidget.McaROIWidget()
            self.roiDockWidget = qt.QDockWidget(self)
            self.roiDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.roiDockWidget.setWidget(self.roiWidget)
            if position in [None, False]:
                w = self.centralWidget().width()
                h = self.centralWidget().height()
                if w > (1.25 * h):
                    self.addDockWidget(qt.Qt.RightDockWidgetArea,
                                       self.roiDockWidget)
                else:
                    self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                                       self.roiDockWidget)
            else:
                self.addDockWidget(position, self.roiDockWidget)
            if hasattr(self, "legendDockWidget"):
                self.tabifyDockWidget(self.legendDockWidget,
                                      self.roiDockWidget)
            self.roiWidget.sigMcaROIWidgetSignal.connect(self._roiSignal)
            self.roiDockWidget.setWindowTitle(self.windowTitle()+(" ROI"))
            # initialize with the ICR
            self._roiSignal({'event': "AddROI"})

        if self.roiDockWidget.isHidden():
            self.roiDockWidget.show()
        else:
            self.roiDockWidget.hide()

    def changeGridLevel(self):
        self.gridLevel += 1
        #self.gridLevel = self.gridLevel % 3
        self.gridLevel = self.gridLevel % 2
        if self.gridLevel == 0:
            self.showGrid(False)
        elif self.gridLevel == 1:
            self.showGrid(1)
        elif self.gridLevel == 2:
            self.showGrid(2)
        self.replot()

    def emitIconSignal(self, key, event="iconClicked"):
        ddict = {}
        ddict["key"] = key
        ddict["event"] = event
        self.sigIconSignal.emit(ddict)

    def _energyIconSignal(self):
        if DEBUG:
            print("energy icon signal default implementation")
        self.emitIconSignal("energy")
        
    def _fitIconSignal(self):
        if DEBUG:
            print("fit icon signal default implementation")
        self.emitIconSignal("fit")

    def _averageIconSignal(self):
        if DEBUG:
            print("average icon signal default implementation")
        self.emitIconSignal("average")

    def _deriveIconSignal(self):
        if DEBUG:
            print("deriveIconSignal default implementation")
        self.emitIconSignal("derive")

    def _smoothIconSignal(self):
        if DEBUG:
            print("smoothIconSignal default implementation")
        self.emitIconSignal("smooth")

    def _swapSignIconSignal(self):
        if DEBUG:
            print("_swapSignIconSignal default implementation")
        self.emitIconSignal("swap")

    def _yMinToZeroIconSignal(self):
        if DEBUG:
            print("_yMinToZeroIconSignal default implementation")
        self.emitIconSignal("ymintozero")

    def _subtractIconSignal(self):
        if DEBUG:
            print("_subtractIconSignal default implementation")
        self.emitIconSignal("subtract")

    def _saveIconSignal(self):
        if DEBUG:
            print("_saveIconSignal default implementation")
        if self._ownSave:
            self.defaultSaveAction()
        else:
            self.emitIconSignal("save")

    def _imageIconSignal(self):
        if DEBUG:
            print("_imageIconSignal default implementation")
        self.emitIconSignal("image")

    def _eraseSelectionIconSignal(self):
        if DEBUG:
            print("_eraseSelectionIconSignal default implementation")
        self.emitIconSignal("erase")

    def _rectSelectionIconSignal(self):
        if DEBUG:
            print("_rectSelectionIconSignal")
        #default implementation set drawing mode with a mask
        self.setDrawModeEnabled(True, shape="rectangle", label="mask")

    def _brushSelectionIconSignal(self):
        if DEBUG:
            print("_brushSelectionIconSignal default implementation")
        self.emitIconSignal("brushSelection")

    def _brushIconSignal(self):
        if DEBUG:
            print("_brushIconSignal default implementation")
        self.emitIconSignal("brush")

    def _additionalIconSignal(self):
        if DEBUG:
            print("_additionalIconSignal default implementation")
        self.emitIconSignal("additional")

    def _polygonIconSignal(self):
        if DEBUG:
            print("_polygonIconSignal")
        #default implementation set drawing mode with a mask
        self.setDrawModeEnabled(True, shape="polygon", label="mask")

    def _pluginClicked(self):
        actionList = []
        menu = qt.QMenu(self)
        text = QString("Reload Plugins")
        menu.addAction(text)
        actionList.append(text)
        text = QString("Set User Plugin Directory")
        menu.addAction(text)
        actionList.append(text)
        global DEBUG
        if DEBUG:
            text = QString("Toggle DEBUG mode OFF")
        else:
            text = QString("Toggle DEBUG mode ON")
        menu.addAction(text)
        actionList.append(text)
        menu.addSeparator()
        callableKeys = ["Dummy0", "Dummy1", "Dummy2"]
        for m in self.pluginList:
            if m in ["PyMcaPlugins.Plugin1DBase", "Plugin1DBase"]:
                continue
            module = sys.modules[m]
            if hasattr(module, 'MENU_TEXT'):
                text = QString(module.MENU_TEXT)
            else:
                text = os.path.basename(module.__file__)
                if text.endswith('.pyc'):
                    text = text[:-4]
                elif text.endswith('.py'):
                    text = text[:-3]
                text = QString(text)
            methods = self.pluginInstanceDict[m].getMethods(plottype=self._plotType)
            if not len(methods):
                continue
            elif len(methods) == 1:
                pixmap = self.pluginInstanceDict[m].getMethodPixmap(methods[0])
                tip = QString(self.pluginInstanceDict[m].getMethodToolTip(methods[0]))
                if pixmap is not None:
                    action = qt.QAction(qt.QIcon(qt.QPixmap(pixmap)), text, self)
                else:
                    action = qt.QAction(text, self)
                if tip is not None:
                    action.setToolTip(tip)
                menu.addAction(action)
            else:
                menu.addAction(text)
            actionList.append(text)
            callableKeys.append(m)
        menu.hovered.connect(self._actionHovered)
        a = menu.exec_(qt.QCursor.pos())
        if a is None:
            return None
        idx = actionList.index(a.text())
        if idx == 0:
            n, message = self.getPlugins(exceptions=True)
            if n < 1:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                msg.setWindowTitle("No plugins")
                msg.setInformativeText(" Problem loading plugins ")
                msg.setDetailedText(message)
                msg.exec_()
            return
        if idx == 1:
            dirName = qt.safe_str(qt.QFileDialog.getExistingDirectory(self,
                                "Enter user plugins directory",
                                os.getcwd()))
            if len(dirName):
                pluginsDir = self.getPluginDirectoryList()
                pluginsDirList = [pluginsDir[0], dirName]
                self.setPluginDirectoryList(pluginsDirList)
            return
        if idx == 2:
            if DEBUG:
                DEBUG = 0
            else:
                DEBUG = 1
            return
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods(plottype=self._plotType)
        if len(methods) == 1:
            idx = 0
        else:
            actionList = []
            # allow the plugin designer to specify the order
            #methods.sort()
            menu = qt.QMenu(self)
            for method in methods:
                text = QString(method)
                pixmap = self.pluginInstanceDict[key].getMethodPixmap(method)
                tip = QString(self.pluginInstanceDict[key].getMethodToolTip(method))
                if pixmap is not None:
                    action = qt.QAction(qt.QIcon(qt.QPixmap(pixmap)), text, self)
                else:
                    action = qt.QAction(text, self)
                if tip is not None:
                    action.setToolTip(tip)
                menu.addAction(action)
                actionList.append((text, pixmap, tip, action))
            #qt.QObject.connect(menu, qt.SIGNAL("hovered(QAction *)"), self._actionHovered)
            menu.hovered.connect(self._actionHovered)
            a = menu.exec_(qt.QCursor.pos())
            if a is None:
                return None
            idx = -1
            for action in actionList:
                if a.text() == action[0]:
                    idx = actionList.index(action)
        try:
            self.pluginInstanceDict[key].applyMethod(methods[idx])
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Plugin error")
            msg.setText("An error has occured while executing the plugin:")
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()

    def _actionHovered(self, action):
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)
        else:
            # hideText was introduced in Qt 4.2
            if hasattr(qt.QToolTip, "hideText"):
                qt.QToolTip.hideText()
            else:
                qt.QToolTip.showText(qt.QCursor.pos(), "")

    ########### ROI HANDLING ###############
    def graphCallback(self, ddict=None):
        if DEBUG:
            print("_graphSignalReceived", ddict)
        if ddict is None:
            ddict = {}
        if ddict['event'] in ['markerMoved', 'markerSelected']:
            label = ddict['label']
            if label in ['ROI min', 'ROI max', 'ROI middle']:
                self._handleROIMarkerEvent(ddict)
        if ddict['event'] in ["curveClicked", "legendClicked"] and \
           self.isActiveCurveHandlingEnabled():
            legend = ddict["label"]
            self.setActiveCurve(legend)
        if ddict['event'] in ['mouseMoved']:
            self._handleMouseMovedEvent(ddict)
        #make sure the signal is forwarded
        #super(PlotWindow, self).graphCallback(ddict)
        self.sigPlotSignal.emit(ddict)

    def _handleMouseMovedEvent(self, ddict):
        if hasattr(self, "_xPos"):
            self._xPos.setText('%.7g' % ddict['x'])
            self._yPos.setText('%.7g' % ddict['y'])

    def setActiveCurve(self, legend, replot=True):
        PlotWidget.PlotWidget.setActiveCurve(self, legend, replot=replot)
        self.calculateROIs()
        self.updateLegends()

    def _handleROIMarkerEvent(self, ddict):
        if ddict['event'] == 'markerMoved':
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            if self.currentROI is None:
                return
            if self.currentROI not in roiDict:
                return
            x = ddict['x']
            label = ddict['label']
            if label == 'ROI min':
                roiDict[self.currentROI]['from'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +\
                                 roiDict[self.currentROI]['from'])
                    self.insertXMarker(pos,
                                       legend='ROI middle',
                                       text='',
                                       color='yellow',
                                       draggable=True)
            elif label == 'ROI max':
                roiDict[self.currentROI]['to'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +\
                                 roiDict[self.currentROI]['from'])
                    self.insertXMarker(pos,
                                       legend='ROI middle',
                                       text='',
                                       color='yellow',
                                       draggable=True)
            elif label == 'ROI middle':
                delta = x - 0.5 * (roiDict[self.currentROI]['from'] + \
                                   roiDict[self.currentROI]['to'])
                roiDict[self.currentROI]['from'] += delta
                roiDict[self.currentROI]['to'] += delta
                self.insertXMarker(roiDict[self.currentROI]['from'],
                                   legend='ROI min',
                                   text='ROI min',
                                   color='blue',
                                   draggable=True)
                self.insertXMarker(roiDict[self.currentROI]['to'],
                                   legend='ROI max',
                                   text='ROI max',
                                   color='blue',
                                   draggable=True)
            else:
                return
            self.calculateROIs(roiList, roiDict)
            self.emitCurrentROISignal()

    def _roiSignal(self, ddict):
        if DEBUG:
            print("PlotWindow._roiSignal ", ddict)
        if ddict['event'] == "AddROI":
            xmin,xmax = self.getGraphXLimits()
            fromdata = xmin + 0.25 * (xmax - xmin)
            todata   = xmin + 0.75 * (xmax - xmin)
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if self._middleROIMarkerFlag:
                self.removeMarker('ROI middle')
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
            self.insertXMarker(fromdata,
                               legend='ROI min',
                               text='ROI min',
                               color=color,
                               draggable=draggable)
            self.insertXMarker(todata,
                               legend='ROI max',
                               text='ROI max',
                               color=color,
                               draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.insertXMarker(pos,
                                   legend='ROI middle',
                                   text="",
                                   color='yellow',
                                   draggable=draggable)
            roiList.append(newroi)
            roiDict[newroi] = {}
            if newroi == "ICR":
                roiDict[newroi]['type'] = "Default"
            else:
                roiDict[newroi]['type'] = self.getGraphXLabel()
            roiDict[newroi]['from'] = fromdata
            roiDict[newroi]['to'] = todata
            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=newroi)
            self.currentROI = newroi
            self.calculateROIs()
        elif ddict['event'] in ['DelROI', "ResetROI"]:
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if self._middleROIMarkerFlag:
                self.removeMarker('ROI middle')
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            roiDictKeys = list(roiDict.keys())
            if len(roiDictKeys):
                currentroi = roiDictKeys[0]
            else:
                # create again the ICR
                ddict = {"event":"AddROI"}
                return self._roiSignal(ddict)
                currentroi = None
            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=currentroi)
            self.currentROI = currentroi
        elif ddict['event'] == 'ActiveROI':
            print("ActiveROI event")
            pass
        elif ddict['event'] == 'selectionChanged':
            if DEBUG:
                print("Selection changed")
            self.roilist, self.roidict = self.roiWidget.getROIListAndDict()
            fromdata = ddict['roi']['from']
            todata   = ddict['roi']['to']
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if ddict['key'] == 'ICR':
                draggable = False
                color = 'black'
            else:
                draggable = True
                color = 'blue'
            self.insertXMarker(fromdata,
                               legend= 'ROI min',
                               text= 'ROI min',
                               color=color,
                               draggable=draggable)
            self.insertXMarker(todata,
                               legend= 'ROI max',
                               text= 'ROI max',
                               color=color,
                               draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.insertXMarker(pos,
                                   legend='ROI middle',
                                   text="",
                                   color='yellow',
                                   draggable=True)
            self.currentROI = ddict['key']
            if ddict['colheader'] in ['From', 'To']:
                dict0 ={}
                dict0['event']  = "SetActiveCurveEvent"
                dict0['legend'] = self.getActiveCurve(just_legend=1)
                self.setActiveCurve(dict0['legend'])
            elif ddict['colheader'] == 'Raw Counts':
                pass
            elif ddict['colheader'] == 'Net Counts':
                pass
            else:
                self.emitCurrentROISignal()
        else:
            if DEBUG:
                print("Unknown or ignored event", ddict['event'])

    def emitCurrentROISignal(self):
        ddict = {}
        ddict['event'] = "currentROISignal"
        roiList, roiDict = self.roiWidget.getROIListAndDict()
        if self.currentROI in roiDict:
            ddict['ROI'] = roiDict[self.currentROI]
        else:
            self.currentROI = None
        ddict['current'] = self.currentROI
        self.sigROISignal.emit(ddict)

    def calculateROIs(self, *var, **kw):
        if not hasattr(self, "roiWidget"):
            return
        if self.roiWidget is None:
            return
        if len(var) == 0:
            roiList, roiDict = self.roiWidget.getROIListAndDict()
        elif len(var) == 2:
            roiList = var[0]
            roiDict = var[1]
        else:
            raise ValueError("Expected roiList and roiDict or nothing")
        update = kw.get("update", True)
        activeCurve = self.getActiveCurve(just_legend=False)
        if activeCurve is None:
            xproc = None
            yproc = None
            self.roiWidget.setHeader('<b>ROIs of XXXXXXXXXX<\b>')
        elif len(activeCurve):
            x, y, legend = activeCurve[0:3]
            idx = argsort(x, kind='mergesort')
            xproc = take(x, idx)
            yproc = take(y, idx)
            self.roiWidget.setHeader('<b>ROIs of %s<\b>' % legend)
        else:
            xproc = None
            yproc = None
            self.roiWidget.setHeader('<b>ROIs of XXXXXXXXXX<\b>')
        for key in roiList:
            #roiDict[key]['rawcounts'] = " ?????? "
            #roiDict[key]['netcounts'] = " ?????? "
            if key == 'ICR':
                if xproc is not None:
                    roiDict[key]['from'] = xproc.min()
                    roiDict[key]['to'] = xproc.max()
                else:
                    roiDict[key]['from'] = 0
                    roiDict[key]['to'] = -1
            fromData  = roiDict[key]['from']
            toData = roiDict[key]['to']
            if xproc is not None:
                idx = nonzero((fromData <= xproc) &\
                                   (xproc <= toData))[0]
                if len(idx):
                    xw = x[idx]
                    yw = y[idx]
                    rawCounts = yw.sum(dtype=numpy.float)
                    deltaX = xw[-1] - xw[0]
                    deltaY = yw[-1] - yw[0]
                    if deltaX > 0.0:
                        slope = (deltaY/deltaX)
                        background = yw[0] + slope * (xw - xw[0])
                        netCounts = rawCounts -\
                                    background.sum(dtype=numpy.float)
                    else:
                        netCounts = 0.0
                else:
                    rawCounts = 0.0
                    netCounts = 0.0
                roiDict[key]['rawcounts'] = rawCounts
                roiDict[key]['netcounts'] = netCounts
        if update:
            if self.currentROI in roiList:
                self.roiWidget.fillFromROIDict(roilist=roiList,
                                               roidict=roiDict,
                                               currentroi=self.currentROI)
            else:
                self.roiWidget.fillFromROIDict(roilist=roiList,
                                               roidict=roiDict)
        else:
            return roiList, roiDict

    def _buildLegendWidget(self):
        if self.legendWidget is None:
            self.legendWidget = LegendSelector.LegendListView()
            self.legendDockWidget = qt.QDockWidget(self)
            self.legendDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.legendDockWidget.setWidget(self.legendWidget)
            w = self.centralWidget().width()
            h = self.centralWidget().height()
            if w > (1.25 * h):
                self.addDockWidget(qt.Qt.RightDockWidgetArea,
                                   self.legendDockWidget)
            else:
                self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                                   self.legendDockWidget)
            if hasattr(self, "roiDockWidget"):
                if self.roiDockWidget is not None:
                    self.tabifyDockWidget(self.roiDockWidget,
                                      self.legendDockWidget)
            self.legendWidget.sigLegendSignal.connect(self._legendSignal)
            self.legendDockWidget.setWindowTitle(self.windowTitle()+(" Legend"))

    def _legendSignal(self, ddict):
        if DEBUG:
            print("Legend signal ddict = ", ddict)
        if ddict['event'] == "legendClicked":
            if ddict['button'] == "left":
                ddict['label'] = ddict['legend']
                self.graphCallback(ddict)
        elif ddict['event'] == "removeCurve":
            ddict['label'] = ddict['legend']
            self.removeCurve(ddict['legend'], replot=True)
        elif ddict['event'] == "renameCurve":
            ddict['label'] = ddict['legend']
            curveList = self.getAllCurves(just_legend=True)
            oldLegend = ddict['legend']
            dialog = RenameCurveDialog.RenameCurveDialog(self,
                                                         oldLegend,
                                                         curveList)
            ret = dialog.exec_()
            if ret:
                newLegend = dialog.getText()
                self.renameCurve(oldLegend, newLegend, replot=True)
        elif ddict['event'] == "setActiveCurve":
            ddict['event'] = 'legendClicked'
            ddict['label'] = ddict['legend']
            self.graphCallback(ddict)
        elif ddict['event'] == "checkBoxClicked":
            if ddict['selected']:
                self.hideCurve(ddict['legend'], False)
            else:
                self.hideCurve(ddict['legend'], True)
        elif ddict['event'] in ["mapToRight", "mapToLeft"]:
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            if ddict['event'] == "mapToRight":
                yaxis = "right"
            else:
                yaxis = "left"
            self.addCurve(x, y, legend=legend, info=info, yaxis=yaxis)
        elif ddict['event'] == "togglePoints":
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            if ddict['points']:
                symbol = 'o'
            else:
                symbol = ''
            # TODO: Limits should be kept
            self.addCurve(x, y, legend=legend, info=info, symbol=symbol)
            self.updateLegends()
        elif ddict['event'] == "toggleLine":
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            # TODO: Limits should be kept
            if ddict['line']:
                self.addCurve(x, y, legend=legend, info=info, linestyle="-")
            else:
                self.addCurve(x, y, legend, info=info, linestyle="")
            self.updateLegends()
        elif DEBUG:
            print("unhandled event", ddict['event'])

    def renameCurve(self, oldLegend, newLegend, replot=True):
        x, y,legend, info = self._curveDict[oldLegend][0:4]
        self.removeCurve(oldLegend, replot=False)
        self.addCurve(x, y, legend=newLegend, info=info, replot=True)
        self.updateLegends()

    def toggleLegendWidget(self):
        if self.legendWidget is None:
            self.showLegends(True)
        elif self.legendDockWidget.isHidden():
            self.showLegends(True)
        else:
            self.showLegends(False)

    def toggleCrosshairCursor(self):
        if self.getGraphCursor():
            self.setGraphCursor(False)
        else:
            self.setGraphCursor(True, color="red", linewidth=1, linestyle="-")

    def toggleArrowKeysPanning(self):
        if self.isPanWithArrowKeys():
            self.setPanWithArrowKeys(False)
        else:
            self.setPanWithArrowKeys(True)

    def showLegends(self, flag=True):
        if self.legendWidget is None:
            self._buildLegendWidget()
            self.updateLegends()
        if flag:
            self.legendDockWidget.show()
            self.updateLegends()
        else:
            self.legendDockWidget.hide()

    def updateLegends(self):
        if self.legendWidget is None:
            return
        if self.legendDockWidget.isHidden():
            return
        legendList = [] * len(self._curveList)
        for i in range(len(self._curveList)):
            legend = self._curveList[i]
            color = self._curveDict[legend][3].get('plot_color',
                                                         '#000000')
            color = qt.QColor(color)
            linewidth = self._curveDict[legend][3].get('plot_line_width',
                                                             2)
            symbol = self._curveDict[legend][3].get('plot_symbol',
                                                    None)
            if self.isCurveHidden(legend):
                selected = False
            else:
                selected = True
            ddict={'color':color,
                   'linewidth':linewidth,
                   'symbol':symbol,
                   'selected':selected}
            legendList.append((legend, ddict))
        self.legendWidget.setLegendList(legendList)


    def setMiddleROIMarkerFlag(self, flag=True):
        if flag:
            self._middleROIMarkerFlag = True
        else:
            self._middleROIMarkerFlag= False

    def setMouseText(self, text=""):
        try:
            if len(text):
                qt.QToolTip.showText(self.cursor().pos(),
                                     text, self, qt.QRect())
            else:
                qt.QToolTip.hideText()
        except:
            print("Error trying to show mouse text <%s>" % text)

    def defaultSaveAction(self):
        """
        Default save implementation.

        It handles saving of curves or the complete widget.
        """
        filename = self._getOutputFileName()
        if filename is None:
            return
        filterused = filename[2]
        filetype = filename[1]
        filename = filename[0]

        if os.path.exists(filename):
            os.remove(filename)
        if filterused[0].upper() == "WIDGET":
            fformat = filename[-3:].upper()
            pixmap = qt.QPixmap.grabWidget(self)
            if not pixmap.save(filename, fformat):
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setInformativeText(str(sys.exc_info()[1]))
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()
            return
        try:
            if sys.version_info.major >= 3:
                ffile = open(filename, 'w', newline='\n')
            else:
                ffile = open(filename,'wb')
        except IOError:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setInformativeText("Input Output Error: %s" % (sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            return
        try:
            if not len(self._curveList):
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setInformativeText("No curve to be saved")
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()
                return
            activeCurve = self.getActiveCurve()
            if activeCurve is None:
                activeCurve = self._curveDict[self._curveList[0]]
            x, y, legend, info = activeCurve
            xlabel = self.getGraphXLabel()
            ylabel = self.getGraphYLabel()
            if filetype.lower() in ["scan", "multiscan"]:
                # write header
                ffile.write("#F %s\n" % filename)
                savingDate = "#D %s\n"%(time.ctime(time.time()))
                ffile.write(savingDate)
                ffile.write("\n")
                ffile.write("#S 1 %s\n" % legend)
                ffile.write(savingDate)
                ffile.write("#N 2\n")
                ffile.write("#L %s  %s\n" % (info.get("xlabel", xlabel),
                                             info.get("ylabel", ylabel)))
                for i in range(len(y)):
                    ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
                ffile.write("\n")
                if filetype.lower() == "multiscan":
                    scan_n  = 1
                    for key in self._curveList:
                        if key not in self._curveDict:
                            continue
                        if key == legend:
                            # active curve already saved
                            continue
                        x, y, newLegend, info = self._curveDict[key]
                        scan_n += 1
                        ffile.write("#S %d %s\n" % (scan_n, key))
                        ffile.write(savingDate)
                        ffile.write("#N 2\n")
                        ffile.write("#L %s  %s\n" % (info.get("xlabel", xlabel),
                                                     info.get("ylabel", ylabel)))
                        for i in range(len(y)):
                            ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
                        ffile.write("\n")
            elif filetype == 'ASCII':
                for i in range(len(y)):
                    ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
            elif filetype == 'CSV':
                if "," in filterused[0]:
                    csvseparator = ","
                elif ";" in filterused[0]:
                    csvseparator = ";"
                elif "OMNIC" in filterused[0]:
                    csvseparator = ","
                else:
                    csvseparator = "\t"
                if "OMNIC" not in filterused[0]:
                    ffile.write('"%s"%s"%s"\n' % (xlabel, csvseparator, ylabel))
                for i in range(len(y)):
                    ffile.write("%.7E%s%.7E\n" % (x[i], csvseparator,y[i]))
            else:
                ffile.write("#F %s\n" % filename)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("\n")
                ffile.write("#S 1 %s\n" % legend)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("#@MCA %16C\n")
                ffile.write("#@CHANN %d %d %d 1\n" %  (len(y), x[0], x[-1]))
                ffile.write("#@CALIB %.7g %.7g %.7g\n" % (0, 1, 0))
                ffile.write(self.array2SpecMca(y))
                ffile.write("\n")
            ffile.close()
        except:
            ffile.close()
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setInformativeText("Error while saving: %s" % (sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()

    def _getOutputFileName(self):
        outfile = qt.QFileDialog(self)
        outfile.setWindowTitle("Output File Selection")
        outfile.setModal(1)
        filterlist = ['Specfile MultiScan *.dat',
                      'Specfile Scan *.dat',
                      'Specfile MCA  *.mca',
                      'Raw ASCII *.txt',
                      '","-separated CSV *.csv',
                      '";"-separated CSV *.csv',
                      '"tab"-separated CSV *.csv',
                      'OMNIC CSV *.csv',
                      'Widget PNG *.png',
                      'Widget JPG *.jpg']
        if hasattr(outfile, "setFilters"):
            outfile.setFilters(filterlist)
        else:
            outfile.setNameFilters(filterlist)
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(outfile.AcceptSave)
        ret = outfile.exec_()
        if not ret:
            return None
        if hasattr(outfile, "selectredFilter"):
            outputFilter = qt.safe_str(outfile.selectedFilter())
        else:
            outputFilter = qt.safe_str(outfile.selectedNameFilter())
        filterused = outputFilter.split()
        filetype  = filterused[1]
        extension = filterused[2]
        outputFile = qt.safe_str(outfile.selectedFiles()[0])
        outfile.close()
        del outfile
        if len(outputFile) < 5:
            outputFile = outputFile + extension[-4:]
        elif outputFile[-4:] != extension[-4:]:
            outputFile = outputFile + extension[-4:]
        return outputFile, filetype, filterused
        
    def array2SpecMca(self, data):
        """ Write a python array into a Spec array.
            Return the string containing the Spec array
        """
        tmpstr = "@A "
        length = len(data)
        for idx in range(0, length, 16):
            if idx+15 < length:
                for i in range(0, 16):
                    tmpstr += "%.7g " % data[idx+i]
                if idx+16 != length:
                    tmpstr += "\\"
            else:
                for i in range(idx, length):
                    tmpstr += "%.7g " % data[i]
            tmpstr += "\n"
        return tmpstr

if __name__ == "__main__":
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    backend = None
    if ("opengl" in sys.argv) or ("gl" in sys.argv) or ("OpenGL" in sys.argv):
        backend = "opengl"
    elif "pyqtgraph" in sys.argv:
        backend = "pyqtgraph"
    plot = PlotWindow(backend=backend, roi=True, control=True,
                          position=True, colormap=True)#uselegendmenu=True)
    plot.setPanWithArrowKeys(True)
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, x*x)
    plot.addCurve(x, -y, "- dummy")
    print("Active curve = ", plot.getActiveCurve(just_legend=True))
    print("X Limits = ",     plot.getGraphXLimits())
    print("Y Limits = ",     plot.getGraphYLimits())
    print("All curves = ",   plot.getAllCurves(just_legend=True))
    image = numpy.arange(10000).reshape(100, 100)
    plot.addImage(image, xScale=(0, 1), yScale=(0, 10), pixmap=MaskImageTools.getPixmapFromData(image))
    def iconSlot(ddict):
        print(ddict)
    plot.sigIconSignal.connect(iconSlot)
    #plot.removeCurve("dummy")
    #plot.addCurve(x, 2 * y, "dummy 2")
    #print("All curves = ",   plot.getAllCurves())
    app.exec_()
