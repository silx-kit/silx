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

from .PlotToolAction import PlotToolAction
from silx.math.histogram import Histogramnd
from silx.math.combo import min_max
import numpy
import logging
from silx.gui import qt

_logger = logging.getLogger(__name__)


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
        self._connectedToActiveImage = False
        self._histo = None

    def _connectPlot(self, window):
        if not self._connectedToActiveImage:
            self.plot.sigActiveImageChanged.connect(
                self._activeImageChanged)
            self._connectedToActiveImage = True
            self.computeIntensityDistribution()
        PlotToolAction._connectPlot(self, window)

    def _disconnectPlot(self, window):
        if self._connectedToActiveImage:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChanged)
            self._connectedToActiveImage = False
        PlotToolAction._disconnectPlot(self, window)

    def _activeImageChanged(self, previous, legend):
        """Handle active image change: toggle enabled toolbar, update curve"""
        if self._isWindowInUse():
            self.computeIntensityDistribution()

    def computeIntensityDistribution(self):
        """Get the active image and compute the image intensity distribution
        """
        activeImage = self.plot.getActiveImage()

        if activeImage is not None:
            image = activeImage.getData(copy=False)
            if image.ndim == 3:  # RGB(A) images
                _logger.info('Converting current image from RGB(A) to grayscale\
                    in order to compute the intensity distribution')
                image = (image[:, :, 0] * 0.299 +
                         image[:, :, 1] * 0.587 +
                         image[:, :, 2] * 0.114)

            xmin, xmax = min_max(image, min_positive=False, finite=True)
            nbins = min(1024, int(numpy.sqrt(image.size)))
            data_range = xmin, xmax

            # bad hack: get 256 bins in the case we have a B&W
            if numpy.issubdtype(image.dtype, numpy.integer):
                if nbins > xmax - xmin:
                    nbins = xmax - xmin

            nbins = max(2, nbins)

            data = image.ravel().astype(numpy.float32)
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
        window.getXAxis().setLabel("Value")
        window.getYAxis().setLabel("Count")
        return window

    def getHistogram(self):
        """Return the last computed histogram

        :return: the histogram displayed in the HistogramPlotWiget
        """
        return self._histo
