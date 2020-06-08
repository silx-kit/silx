# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
:mod:`silx.gui.plot.actions.histogram` provides actions relative to histograms
for :class:`.PlotWidget`.

The following QAction are available:

- :class:`PixelIntensitiesHistoAction`
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__date__ = "10/10/2018"
__license__ = "MIT"

import numpy
import logging
import weakref

from .PlotToolAction import PlotToolAction
from silx.math.histogram import Histogramnd
from silx.math.combo import min_max
from silx.gui import qt
from silx.gui.plot import items

_logger = logging.getLogger(__name__)


class _LastActiveItem(qt.QObject):

    sigActiveItemChanged = qt.Signal(object, object)
    """Emitted when the active plot item have changed"""

    def __init__(self, parent, plot):
        assert plot is not None
        super(_LastActiveItem, self).__init__(parent=parent)
        self.__plot = weakref.ref(plot)
        self.__item = None
        item = self.__findActiveItem()
        self.setActiveItem(item)
        plot.sigActiveImageChanged.connect(self._activeImageChanged)
        plot.sigActiveScatterChanged.connect(self._activeScatterChanged)

    def getPlotWidget(self):
        return self.__plot()

    def __findActiveItem(self):
        plot = self.getPlotWidget()
        image = plot.getActiveImage()
        if image is not None:
            return image
        scatter = plot.getActiveScatter()
        if scatter is not None:
            return scatter

    def getActiveItem(self):
        if self.__item is None:
            return None
        item = self.__item()
        if item is None:
            self.__item = None
        return item

    def setActiveItem(self, item):
        previous = self.getActiveItem()
        if previous is item:
            return
        if item is None:
            self.__item = None
        else:
            self.__item = weakref.ref(item)
        self.sigActiveItemChanged.emit(previous, item)

    def _activeImageChanged(self, previous, current):
        """Handle active image change"""
        plot = self.getPlotWidget()
        item = plot.getImage(current)
        if item is None:
            self.setActiveItem(None)
        elif isinstance(item, items.ImageBase):
            self.setActiveItem(item)
        else:
            # Do not touch anything, which is consistent with silx v0.12 behavior
            pass

    def _activeScatterChanged(self, previous, current):
        """Handle active scatter change"""
        plot = self.getPlotWidget()
        item = plot.getScatter(current)
        self.setActiveItem(item)


class PixelIntensitiesHistoAction(PlotToolAction):
    """QAction to plot the pixels intensities diagram

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        PlotToolAction.__init__(self,
                                plot,
                                icon='pixel-intensities',
                                text='pixels intensity',
                                tooltip='Compute image intensity distribution',
                                parent=parent)
        self._lastItemFilter = _LastActiveItem(self, plot)
        self._histo = None
        self._item = None

    def _connectPlot(self, window):
        self._lastItemFilter.sigActiveItemChanged.connect(self._activeItemChanged)
        item = self._lastItemFilter.getActiveItem()
        self._setSelectedItem(item)
        PlotToolAction._connectPlot(self, window)

    def _disconnectPlot(self, window):
        self._lastItemFilter.sigActiveItemChanged.disconnect(self._activeItemChanged)
        PlotToolAction._disconnectPlot(self, window)
        self._setSelectedItem(None)

    def _getSelectedItem(self):
        item = self._item
        if item is None:
            return None
        else:
            return item()

    def _activeItemChanged(self, previous, current):
        if self._isWindowInUse():
            self._setSelectedItem(current)

    def _setSelectedItem(self, item):
        if item is not None:
            if not isinstance(item, (items.ImageBase, items.Scatter)):
                # Filter out other things
                return

        old = self._getSelectedItem()
        if item is old:
            return
        if old is not None:
            old.sigItemChanged.disconnect(self._itemUpdated)
        if item is None:
            self._item = None
        else:
            self._item = weakref.ref(item)
            item.sigItemChanged.connect(self._itemUpdated)
        self.computeIntensityDistribution()

    def _itemUpdated(self, event):
        if event == items.ItemChangedType.DATA:
            self.computeIntensityDistribution()

    def _cleanUp(self):
        plot = self.getHistogramPlotWidget()
        try:
            plot.remove('pixel intensity', kind='item')
        except Exception:
            pass

    def computeIntensityDistribution(self):
        """Get the active image and compute the image intensity distribution
        """
        item = self._getSelectedItem()

        if item is None:
            self._cleanUp()
            return

        if isinstance(item, items.ImageBase):
            array = item.getData(copy=False)
            if array.ndim == 3:  # RGB(A) images
                _logger.info('Converting current image from RGB(A) to grayscale\
                    in order to compute the intensity distribution')
                array = (array[:, :, 0] * 0.299 +
                         array[:, :, 1] * 0.587 +
                         array[:, :, 2] * 0.114)
        elif isinstance(item, items.Scatter):
            array = item.getValueData(copy=False)
        else:
            assert(False)

        if array.size == 0:
            self._cleanUp()
            return

        xmin, xmax = min_max(array, min_positive=False, finite=True)
        nbins = min(1024, int(numpy.sqrt(array.size)))
        data_range = xmin, xmax

        # bad hack: get 256 bins in the case we have a B&W
        if numpy.issubdtype(array.dtype, numpy.integer):
            if nbins > xmax - xmin:
                nbins = xmax - xmin

        nbins = max(2, nbins)

        data = array.ravel().astype(numpy.float32)
        histogram = Histogramnd(data, n_bins=nbins, histo_range=data_range)
        assert len(histogram.edges) == 1
        self._histo = histogram.histo
        edges = histogram.edges[0]
        plot = self.getHistogramPlotWidget()
        plot.addHistogram(histogram=self._histo,
                          edges=edges,
                          legend='pixel intensity',
                          fill=True,
                          color='#66aad7')
        plot.resetZoom()

    def getHistogramPlotWidget(self):
        """Create the plot histogram if needed, otherwise create it

        :return: the PlotWidget showing the histogram of the pixel intensities
        """
        return self._getToolWindow()

    def _createToolWindow(self):
        from silx.gui.plot.PlotWindow import Plot1D
        window = Plot1D(parent=self.plot)
        window.setWindowFlags(qt.Qt.Window)
        window.setWindowTitle('Image Intensity Histogram')
        window.setDataMargins(0.1, 0.1, 0.1, 0.1)
        window.getXAxis().setLabel("Value")
        window.getYAxis().setLabel("Count")
        return window

    def getHistogram(self):
        """Return the last computed histogram

        :return: the histogram displayed in the HistogramPlotWiget
        """
        return self._histo
