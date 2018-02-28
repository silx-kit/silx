#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
Example to show the use of `ColorBarWidget` widget.
It can be associated to a plot.

In this exqmple the `ColorBarWidget` widget will display the colormap of the
active image.

To change the active image slick on the image you want to set active.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "03/05/2017"


from silx.gui import qt
import numpy
from silx.gui.plot import Colormap
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.plot.PlotWidget import PlotWidget

IMG_WIDTH = 100


class ColorBarShower(qt.QWidget):
    """
    Simple widget displaying n image with different colormap,
    an active colorbar
    """
    def __init__(self):
        qt.QWidget.__init__(self)

        self.setLayout(qt.QHBoxLayout())
        self._buildActiveImageSelector()
        self.layout().addWidget(self.image_selector)
        self._buildPlot()
        self.layout().addWidget(self.plot)
        self._buildColorbar()
        self.layout().addWidget(self.colorbar)
        self._buildColormapEditors()
        self.layout().addWidget(self._cmpEditor)

        # connect radio button with the plot
        self.image_selector._qr1.toggled.connect(self.activateImageChanged)
        self.image_selector._qr2.toggled.connect(self.activateImageChanged)
        self.image_selector._qr3.toggled.connect(self.activateImageChanged)
        self.image_selector._qr4.toggled.connect(self.activateImageChanged)
        self.image_selector._qr5.toggled.connect(self.activateImageChanged)
        self.image_selector._qr6.toggled.connect(self.activateImageChanged)

    def _buildActiveImageSelector(self):
        """Build the image selector widget"""
        self.image_selector = qt.QGroupBox(parent=self)
        self.image_selector.setLayout(qt.QVBoxLayout())
        self.image_selector._qr1 = qt.QRadioButton('image1')
        self.image_selector._qr1.setChecked(True)
        self.image_selector._qr2 = qt.QRadioButton('image2')
        self.image_selector._qr3 = qt.QRadioButton('image3')
        self.image_selector._qr4 = qt.QRadioButton('image4')
        self.image_selector._qr5 = qt.QRadioButton('image5')
        self.image_selector._qr6 = qt.QRadioButton('image6')
        self.image_selector.layout().addWidget(self.image_selector._qr6)
        self.image_selector.layout().addWidget(self.image_selector._qr5)
        self.image_selector.layout().addWidget(self.image_selector._qr4)
        self.image_selector.layout().addWidget(self.image_selector._qr3)
        self.image_selector.layout().addWidget(self.image_selector._qr2)
        self.image_selector.layout().addWidget(self.image_selector._qr1)

    def activateImageChanged(self):
        if self.image_selector._qr1.isChecked():
            self.plot.setActiveImage('image1')
        if self.image_selector._qr2.isChecked():
            self.plot.setActiveImage('image2')
        if self.image_selector._qr3.isChecked():
            self.plot.setActiveImage('image3')
        if self.image_selector._qr4.isChecked():
            self.plot.setActiveImage('image4')
        if self.image_selector._qr5.isChecked():
            self.plot.setActiveImage('image5')
        if self.image_selector._qr6.isChecked():
            self.plot.setActiveImage('image6')

    def _buildColorbar(self):
        self.colorbar = ColorBarWidget(parent=self)
        self.colorbar.setPlot(self.plot)

    def _buildPlot(self):
        image1 = numpy.exp(numpy.random.rand(IMG_WIDTH, IMG_WIDTH) * 10)
        image2 = numpy.linspace(-100, 1000, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                           IMG_WIDTH)
        image3 = numpy.linspace(-1, 1, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                      IMG_WIDTH)
        image4 = numpy.linspace(-20, 50, IMG_WIDTH * IMG_WIDTH).reshape(IMG_WIDTH,
                                                                        IMG_WIDTH)
        image5 = image3
        image6 = image4

        # viridis colormap
        self.colormap1 = Colormap.Colormap(name='green',
                                           normalization='log',
                                           vmin=None,
                                           vmax=None)
        self.plot = PlotWidget(parent=self)
        self.plot.addImage(data=image1,
                           origin=(0, 0),
                           legend='image1',
                           colormap=self.colormap1)
        self.plot.addImage(data=image2,
                           origin=(100, 0),
                           legend='image2',
                           colormap=self.colormap1)

        # red colormap
        self.colormap2 = Colormap.Colormap(name='red',
                                           normalization='linear',
                                           vmin=None,
                                           vmax=None)
        self.plot.addImage(data=image3,
                           origin=(0, 100),
                           legend='image3',
                           colormap=self.colormap2)
        self.plot.addImage(data=image4,
                           origin=(100, 100),
                           legend='image4',
                           colormap=self.colormap2)
        # gray colormap
        self.colormap3 = Colormap.Colormap(name='gray',
                                           normalization='linear',
                                           vmin=1.0,
                                           vmax=20.0)
        self.plot.addImage(data=image5,
                           origin=(0, 200),
                           legend='image5',
                           colormap=self.colormap3)
        self.plot.addImage(data=image6,
                           origin=(100, 200),
                           legend='image6',
                           colormap=self.colormap3)

    def _buildColormapEditors(self):
        self._cmpEditor = qt.QWidget(parent=self)
        self._cmpEditor.setLayout(qt.QVBoxLayout())
        self._cmpEditor.layout().addWidget(_ColormapEditor(self.colormap3))
        self._cmpEditor.layout().addWidget(_ColormapEditor(self.colormap2))
        self._cmpEditor.layout().addWidget(_ColormapEditor(self.colormap1))


class _ColormapEditor(qt.QWidget):
    """Simple colormap editor"""

    def __init__(self, colormap):
        qt.QWidget.__init__(self)
        self.setLayout(qt.QVBoxLayout())
        self._colormap = colormap
        self._buildGUI()

        self.setColormap(colormap)

        # connect GUI and colormap
        self._qcbName.currentIndexChanged.connect(self._nameChanged)
        self._qgbNorm._qrbLinear.toggled.connect(self._normalizationHaschanged)
        self._qgbNorm._qrbLog.toggled.connect(self._normalizationHaschanged)
        self._vminWidget.sigValueChanged.connect(self._vminHasChanged)
        self._vmaxWidget.sigValueChanged.connect(self._vmaxHasChanged)

    def _buildGUI(self):
        # build name
        self._buildName()

        self.layout().addWidget(self._qcbName)

        # build vmin
        vmin = self._colormap.getVMin()
        if vmin is None:
            vmin= self._colormap._getDefaultMin()

        self._vminWidget = _BoundSetter(text='vmin',
                                        parent=self,
                                        value=vmin,
                                        checked=self._colormap.getVMin() is not None)
        self.layout().addWidget(self._vminWidget)

        # build vmax
        vmax = self._colormap.getVMax()
        if vmax is None:
            vmax= self._colormap._getDefaultMax()
        self._vmaxWidget = _BoundSetter(text='vmax',
                                        parent=self,
                                        value=vmax,
                                        checked=self._colormap.getVMax() is not None)
        self.layout().addWidget(self._vmaxWidget)

        # build norm
        self._buildNorm()
        self.layout().addWidget(self._qgbNorm)

        spacer = qt.QWidget()
        spacer.setSizePolicy(qt.QSizePolicy.Minimum,
                             qt.QSizePolicy.Expanding)
        self.layout().addWidget(spacer)

    def setColormap(self, colormap):
        self._colormap = colormap
        self._setName(colormap.getName())
        self._setVMin(colormap.getVMin())
        self._setVMax(colormap.getVMax())
        self._setNorm(colormap.getNormalization())

    def _setName(self, name):
        assert name in Colormap.Colormap.getSupportedColormaps()
        self._qcbName.setCurrentIndex(self._qcbName.findText(name))

    def _nameChanged(self, name):
        self._colormap.setName(self._qcbName.currentText())

    def _setVMin(self, vmin):
        vmin = self._colormap.getVMin()
        if vmin is None:
            vmin= self._colormap._getDefaultMin()
        self._vminWidget.set(vmin, checked=self._colormap.getVMin() is not None)

    def _setVMax(self, max):
        vmax = self._colormap.getVMax()
        if vmax is None:
            vmax= self._colormap._getDefaultMax()
        self._vmaxWidget.set(vmax, checked=self._colormap.getVMax() is not None)

    def _setNorm(self, norm):
        if norm == 'linear':
            self._qgbNorm._qrbLinear.setChecked(True)
        if norm == 'log':
            self._qgbNorm._qrbLog.setChecked(True)

    def _buildName(self):
        wName = qt.QWidget(self)
        wName.setLayout(qt.QHBoxLayout())

        wName.layout().addWidget(qt.QLabel('name'))
        self._qcbName = qt.QComboBox(wName)
        for val in Colormap.Colormap.getSupportedColormaps():
            self._qcbName.addItem(val)

    def _buildNorm(self):
        self._qgbNorm = qt.QGroupBox(parent=self)
        self._qgbNorm.setLayout(qt.QHBoxLayout())
        self._qgbNorm._qrbLinear = qt.QRadioButton('linear')
        self._qgbNorm._qrbLog = qt.QRadioButton('log')
        self._qgbNorm.layout().addWidget(self._qgbNorm._qrbLinear)
        self._qgbNorm.layout().addWidget(self._qgbNorm._qrbLog)

    def _normalizationHaschanged(self):
        if self._qgbNorm._qrbLinear.isChecked():
            self._colormap.setNormalization('linear')

        if self._qgbNorm._qrbLog.isChecked():
            self._colormap.setNormalization('log')

    def _vminHasChanged(self):
        try:
            value = self._vminWidget.getBound()
        except ValueError:
            pass
        else:
            self._colormap.setVMin(value)

    def _vmaxHasChanged(self):
        try:
            value = self._vmaxWidget.getBound()
        except ValueError:
            pass
        else:
            self._colormap.setVMax(value)


class _BoundSetter(qt.QWidget):

    sigValueChanged = qt.Signal()

    def __init__(self, text, parent, value, checked=False):
        qt.QWidget.__init__(self, parent=parent)

        self.setLayout(qt.QHBoxLayout())

        self._qcb = qt.QCheckBox(parent=self, text=text)
        self.layout().addWidget(self._qcb)

        self._qLineEdit = qt.QLineEdit()

        self.layout().addWidget(self._qLineEdit)
        self._qcb.toggled.connect(self.updateVis)

        self.set(value=value, checked=checked)

        self._qLineEdit.textChanged.connect(self._valueChanged)

    def getBound(self):
        if self._qcb.isChecked():
            return float(self._qLineEdit.text())
        else:
            return None

    def updateVis(self):
        self._qLineEdit.setEnabled(self._qcb.isChecked())
        self.sigValueChanged.emit()

    def _valueChanged(self, val):
        self.sigValueChanged.emit()

    def set(self, value, checked):
        self._qLineEdit.setText(str(value))
        self._qcb.setChecked(checked)
        self._qLineEdit.setEnabled(checked)


if __name__ == '__main__':
    app = qt.QApplication([])
    widget = ColorBarShower()
    widget.show()
    app.exec_()
