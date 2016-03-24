#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
import sys

from PyMca5.PyMcaGui import PyMcaQt as qt
from . import PlotWidget

QTVERSION = qt.qVersion()
DEBUG = 0

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=""):
        qt.QLineEdit.__init__(self,parent)

    def focusInEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('yellow'))

    def focusOutEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('white'))
        self.returnPressed[()].emit()

    def setPaletteBackgroundColor(self, color):
        if DEBUG:
            print("setPalettebackgroundColor not implemented yet")
        pass

"""
Manage colormap Widget class
"""
class ColormapDialog(qt.QDialog):
    sigColormapChanged = qt.pyqtSignal(object)
    def __init__(self, parent=None, name="Colormap Dialog"):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle(name)
        self.title = name


        self.colormapList = ["Greyscale", "Reverse Grey", "Temperature",
                             "Red", "Green", "Blue", "Many"]

        # histogramData is tupel(bins, counts)
        self.histogramData = None

        # default values
        self.dataMin   = -10
        self.dataMax   = 10
        self.minValue  = 0
        self.maxValue  = 1

        self.colormapIndex  = 2
        self.colormapType   = 0

        self.autoscale   = False
        self.autoscale90 = False
        # main layout
        vlayout = qt.QVBoxLayout(self)
        vlayout.setContentsMargins(10, 10, 10, 10)
        vlayout.setSpacing(0)

        # layout 1 : -combo to choose colormap
        #            -autoscale button
        #            -autoscale 90% button
        hbox1    = qt.QWidget(self)
        hlayout1 = qt.QHBoxLayout(hbox1)
        vlayout.addWidget(hbox1)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(10)

        # combo
        self.combo = qt.QComboBox(hbox1)
        for colormap in self.colormapList:
            self.combo.addItem(colormap)
        self.combo.activated[int].connect(self.colormapChange)
        hlayout1.addWidget(self.combo)

        # autoscale
        self.autoScaleButton = qt.QPushButton("Autoscale", hbox1)
        self.autoScaleButton.setCheckable(True)
        self.autoScaleButton.setAutoDefault(False)
        self.autoScaleButton.toggled[bool].connect(self.autoscaleChange)
        hlayout1.addWidget(self.autoScaleButton)

        # autoscale 90%
        self.autoScale90Button = qt.QPushButton("Autoscale 90%", hbox1)
        self.autoScale90Button.setCheckable(True)
        self.autoScale90Button.setAutoDefault(False)

        self.autoScale90Button.toggled[bool].connect(self.autoscale90Change)
        hlayout1.addWidget(self.autoScale90Button)

        # hlayout
        hbox0    = qt.QWidget(self)
        self.__hbox0 = hbox0
        hlayout0 = qt.QHBoxLayout(hbox0)
        hlayout0.setContentsMargins(0, 0, 0, 0)
        hlayout0.setSpacing(0)
        vlayout.addWidget(hbox0)
        #hlayout0.addStretch(10)

        self.buttonGroup = qt.QButtonGroup()
        g1 = qt.QCheckBox(hbox0)
        g1.setText("Linear")
        g2 = qt.QCheckBox(hbox0)
        g2.setText("Logarithmic")
        g3 = qt.QCheckBox(hbox0)
        g3.setText("Gamma")
        self.buttonGroup.addButton(g1, 0)
        self.buttonGroup.addButton(g2, 1)
        self.buttonGroup.addButton(g3, 2)
        self.buttonGroup.setExclusive(True)
        if self.colormapType == 1:
            self.buttonGroup.button(1).setChecked(True)
        elif self.colormapType == 2:
            self.buttonGroup.button(2).setChecked(True)
        else:
            self.buttonGroup.button(0).setChecked(True)
        hlayout0.addWidget(g1)
        hlayout0.addWidget(g2)
        hlayout0.addWidget(g3)
        vlayout.addWidget(hbox0)
        self.buttonGroup.buttonClicked[int].connect(self.buttonGroupChange)
        vlayout.addSpacing(20)

        hboxlimits = qt.QWidget(self)
        hboxlimitslayout = qt.QHBoxLayout(hboxlimits)
        hboxlimitslayout.setContentsMargins(0, 0, 0, 0)
        hboxlimitslayout.setSpacing(0)

        self.slider = None

        vlayout.addWidget(hboxlimits)

        vboxlimits = qt.QWidget(hboxlimits)
        vboxlimitslayout = qt.QVBoxLayout(vboxlimits)
        vboxlimitslayout.setContentsMargins(0, 0, 0, 0)
        vboxlimitslayout.setSpacing(0)
        hboxlimitslayout.addWidget(vboxlimits)

        # hlayout 2 : - min label
        #             - min texte
        hbox2    = qt.QWidget(vboxlimits)
        self.__hbox2 = hbox2
        hlayout2 = qt.QHBoxLayout(hbox2)
        hlayout2.setContentsMargins(0, 0, 0, 0)
        hlayout2.setSpacing(0)
        #vlayout.addWidget(hbox2)
        vboxlimitslayout.addWidget(hbox2)
        hlayout2.addStretch(10)

        self.minLabel  = qt.QLabel(hbox2)
        self.minLabel.setText("Minimum")
        hlayout2.addWidget(self.minLabel)

        hlayout2.addSpacing(5)
        hlayout2.addStretch(1)
        self.minText  = MyQLineEdit(hbox2)
        self.minText.setFixedWidth(150)
        self.minText.setAlignment(qt.Qt.AlignRight)
        self.minText.returnPressed[()].connect(self.minTextChanged)
        hlayout2.addWidget(self.minText)

        # hlayout 3 : - min label
        #             - min text
        hbox3    = qt.QWidget(vboxlimits)
        self.__hbox3 = hbox3
        hlayout3 = qt.QHBoxLayout(hbox3)
        hlayout3.setContentsMargins(0, 0, 0, 0)
        hlayout3.setSpacing(0)
        #vlayout.addWidget(hbox3)
        vboxlimitslayout.addWidget(hbox3)

        hlayout3.addStretch(10)
        self.maxLabel = qt.QLabel(hbox3)
        self.maxLabel.setText("Maximum")
        hlayout3.addWidget(self.maxLabel)

        hlayout3.addSpacing(5)
        hlayout3.addStretch(1)

        self.maxText = MyQLineEdit(hbox3)
        self.maxText.setFixedWidth(150)
        self.maxText.setAlignment(qt.Qt.AlignRight)

        self.maxText.returnPressed[()].connect(self.maxTextChanged)
        hlayout3.addWidget(self.maxText)


        # Graph widget for color curve...
        self.c = PlotWidget.PlotWidget(self, backend=None)
        self.c.setGraphXLabel("Data Values")
        self.c.setZoomModeEnabled(False)

        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge

        self.c.setGraphXLimits(self.minmd, self.maxpd)
        self.c.setGraphYLimits(-11.5, 11.5)

        x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        y = [-10, -10, 10, 10 ]
        self.c.addCurve(x, y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.markers = []
        self.__x = x
        self.__y = y
        labelList = ["","Min", "Max", ""]
        for i in range(4):
            if i in [1, 2]:
                draggable = True
                color = "blue"
            else:
                draggable = False
                color = "black"
            #TODO symbol
            legend = "%d" % i
            self.c.insertXMarker(x[i],
                                 legend=legend,
                                 text=labelList[i],
                                 draggable=draggable,
                                 color=color)
            self.markers.append((legend, ""))

        self.c.setMinimumSize(qt.QSize(250,200))
        vlayout.addWidget(self.c)

        self.c.sigPlotSignal.connect(self.chval)

        # colormap window can not be resized
        self.setFixedSize(vlayout.minimumSize())

    def plotHistogram(self, data=None):
        if data is not None:
            self.histogramData = data
        if self.histogramData is None:
            return False
        bins, counts = self.histogramData
        self.c.addCurve(bins, counts,
                        "Histogram",
                        color='pink', # TODO: Change fill color
                        symbol='s',
                        linestyle='-', # Line style
                        #fill=True,
                        yaxis='right')
                        # TODO: Do not use info!

    def _update(self):
        if DEBUG:
            print("colormap _update called")
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge
        self.c.setGraphXLimits(self.minmd, self.maxpd)
        self.c.setGraphYLimits( -11.5, 11.5)

        self.__x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        self.__y = [-10, -10, 10, 10]
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.c.clearMarkers()
        for i in range(4):
            if i in [1, 2]:
                draggable = True
                color = "blue"
            else:
                draggable = False
                color = "black"
            key = self.markers[i][0]
            label = self.markers[i][1]
            self.c.insertXMarker(self.__x[i],
                                 legend=key,
                                 text=label,
                                 draggable=draggable,
                                 color=color)
        self.c.replot()
        self.sendColormap()

    def buttonGroupChange(self, val):
        if DEBUG:
            print("buttonGroup asking to update colormap")
        self.setColormapType(val, update=True)
        self._update()

    def setColormapType(self, val, update=False):
        self.colormapType = val
        if self.colormapType == 1:
            self.buttonGroup.button(1).setChecked(True)
        elif self.colormapType == 2:
            self.buttonGroup.button(2).setChecked(True)
        else:
            self.colormapType = 0
            self.buttonGroup.button(0).setChecked(True)
        if update:
            self._update()

    def chval(self, ddict):
        if DEBUG:
            print("Received ", ddict)
        if ddict['event'] == 'markerMoving':
            diam = int(ddict['label'])
            x = ddict['x']
            if diam == 1:
                self.setDisplayedMinValue(x)
            elif diam == 2:
                self.setDisplayedMaxValue(x)
        elif ddict['event'] == 'markerMoved':
            diam = int(ddict['label'])
            x = ddict['x']
            if diam == 1:
                self.setMinValue(x)
            if diam == 2:
                self.setMaxValue(x)

    """
    Colormap
    """
    def setColormap(self, colormap):
        self.colormapIndex = colormap
        if QTVERSION < '4.0.0':
            self.combo.setCurrentItem(colormap)
        else:
            self.combo.setCurrentIndex(colormap)

    def colormapChange(self, colormap):
        self.colormapIndex = colormap
        self.sendColormap()

    # AUTOSCALE
    """
    Autoscale
    """
    def autoscaleChange(self, val):
        self.autoscale = val
        self.setAutoscale(val)
        self.sendColormap()

    def setAutoscale(self, val):
        if DEBUG:
            print("setAutoscale called", val)
        if val:
            self.autoScaleButton.setChecked(True)
            self.autoScale90Button.setChecked(False)
            #self.autoScale90Button.setDown(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax)
            self.maxText.setEnabled(0)
            self.minText.setEnabled(0)
            self.c.setEnabled(0)
            #self.c.disablemarkermode()
        else:
            self.autoScaleButton.setChecked(False)
            self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(1)
            #self.c.enablemarkermode()

    """
    set rangeValues to dataMin ; dataMax-10%
    """
    def autoscale90Change(self, val):
        self.autoscale90 = val
        self.setAutoscale90(val)
        self.sendColormap()

    def setAutoscale90(self, val):
        if val:
            self.autoScaleButton.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax - abs(self.dataMax/10))
            self.minText.setEnabled(0)
            self.maxText.setEnabled(0)
            self.c.setEnabled(0)
        else:
            self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(1)
            self.c.setFocus()



    # MINIMUM
    """
    change min value and update colormap
    """
    def setMinValue(self, val):
        v = float(str(val))
        self.minValue = v
        self.minText.setText("%g" % v)
        self.__x[1] = v
        key = self.markers[1][0]
        label = self.markers[1][1]
        self.c.insertXMarker(v, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x,
                        self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.sendColormap()

    """
    min value changed by text
    """
    def minTextChanged(self):
        text = str(self.minText.text())
        if not len(text):
            return
        val = float(text)
        self.setMinValue(val)
        if self.minText.hasFocus():
            self.c.setFocus()

    """
    change only the displayed min value
    """
    def setDisplayedMinValue(self, val):
        val = float(val)
        self.minValue = val
        self.minText.setText("%g"%val)
        self.__x[1] = val
        key = self.markers[1][0]
        label = self.markers[1][1]
        self.c.insertXMarker(val, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
    # MAXIMUM
    """
    change max value and update colormap
    """
    def setMaxValue(self, val):
        v = float(str(val))
        self.maxValue = v
        self.maxText.setText("%g"%v)
        self.__x[2] = v
        key = self.markers[2][0]
        label = self.markers[2][1]
        self.c.insertXMarker(v, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')
        self.sendColormap()

    """
    max value changed by text
    """
    def maxTextChanged(self):
        text = str(self.maxText.text())
        if not len(text):return
        val = float(text)
        self.setMaxValue(val)
        if self.maxText.hasFocus():
            self.c.setFocus()

    """
    change only the displayed max value
    """
    def setDisplayedMaxValue(self, val):
        val = float(val)
        self.maxValue = val
        self.maxText.setText("%g"%val)
        self.__x[2] = val
        key = self.markers[2][0]
        label = self.markers[2][1]
        self.c.insertXMarker(val, legend=key, text=label, color="blue", draggable=True)
        self.c.addCurve(self.__x, self.__y,
                        legend="ConstrainedCurve",
                        color='black',
                        symbol='o',
                        linestyle='-')


    # DATA values
    """
    set min/max value of data source
    """
    def setDataMinMax(self, minVal, maxVal, update=True):
        if minVal is not None:
            vmin = float(str(minVal))
            self.dataMin = vmin
        if maxVal is not None:
            vmax = float(str(maxVal))
            self.dataMax = vmax

        if update:
            # are current values in the good range ?
            self._update()

    def getColormap(self):
        if self.minValue > self.maxValue:
            vmax = self.minValue
            vmin = self.maxValue
        else:
            vmax = self.maxValue
            vmin = self.minValue
        cmap = [self.colormapIndex, self.autoscale,
                vmin, vmax,
                self.dataMin, self.dataMax,
                self.colormapType]
        return cmap

    """
    send 'ColormapChanged' signal
    """
    def sendColormap(self):
        if DEBUG:
            print("sending colormap")
        try:
            cmap = self.getColormap()
            self.sigColormapChanged.emit(cmap)
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])


# Conversion of colormap descriptions #########################################

# Convert colormap between PlotBackend names and ColormapDialog index
# Tuple of PlotBackend names in the order of colormapDialog
_COLORMAP_NAMES = ('gray', 'reversed gray', 'temperature',
                   'red', 'green', 'blue', 'temperature')

def colormapListToDict(colormapList):
    """Convert colormap from this dialog to :class:`PlotBackend`.

    :param colormapList: Colormap as returned by :meth:`getColormap`.
    :type colormapList: list or tuple
    :return: Colormap as used in :class:`PlotBackend`.
    :rtype: dict
    """
    index, autoscale, vMin, vMax, dataMin, dataMax, cmapType = colormapList
    # Warning, gamma cmapType is not supported in plot backend
    # Here it is silently replaced by linear as the default colormap
    return {
        'name': _COLORMAP_NAMES[index],
        'autoscale': autoscale,
        'vmin': vMin,
        'vmax': vMax,
        'normalization': 'log' if cmapType == 1 else 'linear',
        'colors': 256
    }

def colormapDictToList(colormapDict):
    """Convert colormap from :class:`PlotBackend` to this dialog.

    :param dict colormapDict: Colormap as used in :class:`PlotBackend`.
    :return: Colormap as returned by :meth:`getColormap`.
    :rtype: list
    """
    cmapIndex = _COLORMAP_NAMES.index(colormapDict['name'])
    cmapType = 1 if colormapDict['normalization'].startswith('log') else 0
    return [cmapIndex,
            colormapDict['autoscale'],
            colormapDict['vmin'],
            colormapDict['vmax'],
            0,  # dataMin is not defined in PlotBackend colormap
            0,  # dataMax is not defined in PlotBackend colormap
            cmapType]


###############################################################################

def test():
    app = qt.QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    demo = ColormapDialog()

    # Histogram demo
    import numpy as np
    x = np.linspace(-10, 10, 50)
    y = abs(9. * np.exp(-x**2) + np.random.randn(len(x)) + 1.)
    demo.plotHistogram((x,y))

    def call(*var):
        print("Received", var)

    demo.sigColormapChanged.connect(call)

    demo.setAutoscale(1)
    demo.show()
    app.exec_()


if __name__ == "__main__":
    test()
