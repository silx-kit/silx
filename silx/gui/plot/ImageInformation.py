# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""Define interface for defining `Image Information` Widget.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "16/05/2019"


from silx.gui import qt


class BaseImageInformation(object):
    """Base class for widget displaying information concerning the active image
    """
    def __init__(self):
        self._plot = None

    def setPlot(self, plot):
        """

        :param `.PlotWidget` plot:
        """
        if self._plot is not None:
            self._disconnectPlot()
        self._plot = plot
        if self._plot is not None:
            self._connectPlot()

    def _disconnectPlot(self):
        raise NotImplementedError("Base class")

    def _connectPlot(self):
        raise NotImplementedError("Base class")


class ImageInformationWidget(qt.QWidget, BaseImageInformation):
    """
    Simple widget used to display the active image dimension

    :param QWidget parent: parent QWidget
    :param `.PlotWidget` plot: plot containing the image we want to display the
                           information
    """
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent=parent)
        BaseImageInformation.__init__(self)
        self._qLabel = qt.QLabel('', self)
        self._qLabel.setEnabled(False)
        self.setLayout(qt.QHBoxLayout())
        self.layout().addWidget(self._qLabel)

    def setPlot(self, plot):
        BaseImageInformation.setPlot(self, plot)
        self._imageChanged()

    def _disconnectPlot(self):
        self._plot.sigActiveImageChanged.disconnect(self._imageChanged)

    def _connectPlot(self):
        self._plot.sigActiveImageChanged.connect(self._imageChanged)

    def _imageChanged(self):
        text = ''
        if self._plot is not None:
            activeImage = self._plot.getActiveImage()
            if activeImage is not None:
                text = "dims: %s x %s" % (activeImage.getData(copy=False).shape[::-1])
        self._qLabel.setText(text)
