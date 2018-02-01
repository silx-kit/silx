# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Widget displaying a 3D volume or a list of 2D images as a grid of
plotted images. Each plot has a slider to allow selecting any one of the
images. All plots have their axes synchronized.

:meth:`GridImageWidget.setImages`

"""
import contextlib

from .. import qt
from . import Plot2D
# from .Colormap import Colormap

from silx.third_party import six
from silx.gui.plot.utils.axis import SyncAxes
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser


class PlotWithSlider(qt.QWidget):
    #                                value id
    sigSliderValueChanged = qt.Signal(int, object)
    sigKeepAspectRatioChanged = qt.Signal(bool, object)

    def __init__(self, parent=None, backend=None):
        qt.QWidget.__init__(self, parent)

        self.plot = Plot2D(parent=self, backend=backend)
        self.plot.sigSetKeepDataAspectRatio.connect(
                self._emitPlotKeepAspectRatio)
        self.slider = HorizontalSliderWithBrowser(self)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(
                self._emitSliderValueChanged)

        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        layout.addWidget(self.plot)
        layout.addWidget(self.slider)

    def _emitSliderValueChanged(self, value):
        self.sigSliderValueChanged.emit(value, id(self))

    def _emitPlotKeepAspectRatio(self, isKeepAspectRatio):
        self.sigKeepAspectRatioChanged.emit(isKeepAspectRatio,
                                            id(self))


class GridImageWidget(qt.QWidget):
    def __init__(self, parent=None, backend=None):
        qt.QWidget.__init__(self, parent)

        self._backend = backend
        self._nrows = 2
        self._ncols = 2
        self._maxNPlots = 100
        self._plots = {}
        """:class:`Plot2D` indexed by 2-tuples (row, col)"""
        self._plotCoords = {}
        """Plot coordinates (row, col) indexed by the widget's id"""
        self._xAxesSynchro = None
        self._yAxesSynchro = None
        self._data = []
        """List of 2D arrays or 3D array"""

        self.gridLayout = qt.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.gridLayout)

        self._updateGrid()

    def setNRows(self, nrows):
        assert isinstance(nrows, six.integer_types)
        if nrows * self._ncols > self._maxNPlots:
            max_nrows = self._maxNPlots // self._ncols
            raise ValueError(
                "Cannot increase nrows to more than %d." % max_nrows +
                "You need to decrease the number of cols first, or explicitly"
                " increase the limit (probably a bad idea).")
        if nrows == self._nrows:
            return
        self._nrows = nrows
        self._updateGrid()

    def setNCols(self, ncols):
        assert isinstance(ncols, six.integer_types)
        if ncols * self._nrows > self._maxNPlots:
            max_ncols = self._maxNPlots // self._nrows
            raise ValueError(
                "Cannot increase ncols to more than %d." % max_ncols +
                "You need to decrease the number of rows first, or explicitly"
                " increase the limit (probably a bad idea).")
        if ncols == self._ncols:
            return
        self._ncols = ncols
        self._updateGrid()

    def setMaxNPlots(self, nplots):
        """Update the max number of plots allowed.
        This is probably a bad idea.

        The maximum number of plots constrains the max number of rows
        and columns.

        :param int nplots: Maximum number of plots allowed.
        """
        assert isinstance(nplots, six.integer_types)
        self._maxNPlots = nplots

    def _updateGrid(self):
        """Show or hide the plots according to the current grid shape.
        Instantiate new plots if necessary."""
        # instantiate new plots as needed
        areNewPlotsAdded = False
        for r in range(self._nrows):
            for c in range(self._ncols):
                if (r, c) not in self._plots:
                    areNewPlotsAdded = True
                    self._plots[(r, c)] = self._instantiateNewPlot()
                    self._plotCoords[id(self._plots[(r, c)])] = (r, c)
                    self.gridLayout.addWidget(self._plots[(r, c)],
                                              r, c)
                    self._plots[(r, c)].sigSliderValueChanged.connect(
                            self._onSliderValueChanged)
                    self._plots[(r, c)].sigKeepAspectRatioChanged.connect(
                            self._onKeepAspectRatioChanged)

        # show or hide plots as needed
        for idx in self._plots:
            r, c = idx
            if r < self._nrows and c < self._ncols:
                self._plots[idx].show()
            else:
                self._plots[idx].hide()

        # synchronize all axes
        if areNewPlotsAdded:
            self._yAxesSynchro = SyncAxes([plt.plot.getYAxis() for plt in self._plots.values()])
            self._xAxesSynchro = SyncAxes([plt.plot.getXAxis() for plt in self._plots.values()])

    def _instantiateNewPlot(self):
        plot = PlotWithSlider(parent=self,
                              backend=self._backend)
        return plot

    @property
    def _nframes(self):
        """Number of frames (images) in the current data"""
        # 3D array
        if hasattr(self._data, "shape"):
            return self._data.shape[0]
        # list of 2D arrays
        return len(self._data)

    @property
    def _nplots(self):
        """Number of visible plots in the grid"""
        return self._nrows * self._ncols

    def _onSliderValueChanged(self, value, plotId):
        """Plot the requested image, if any data is loaded."""
        assert value < self._nframes
        row, col = self._plotCoords[plotId]
        self._plots[(row, col)].plot.addImage(self._data[value])

    def _onKeepAspectRatioChanged(self, isKeepAspectRatio, plotId):
        """If any plot changes its keepAspectRatio policy,
        apply it to all other plots."""
        with self._disconnectAllAspectRatioSignals():
            for r, c in self._plots:
                self._plots[(r, c)].plot.setKeepDataAspectRatio(isKeepAspectRatio)

    @contextlib.contextmanager
    def _disconnectAllAspectRatioSignals(self):
        for r, c in self._plots:
            self._plots[(r, c)].sigKeepAspectRatioChanged.disconnect(
                self._onKeepAspectRatioChanged)
        yield
        for r, c in self._plots:
            self._plots[(r, c)].sigKeepAspectRatioChanged.connect(
                self._onKeepAspectRatioChanged)

    def setFrames(self, data):
        """

        :param data: List of 2D arrays of identical size, or 3D array
            (first dimension is the frame index)
        """
        self.clear()
        if hasattr(data, "shape"):
            assert len(data.shape) == 3, "Array must be 3D"
        else:
            assert hasattr(data, "__len__"), "Not a list of 2D arrays"
            for array in data:
                assert hasattr(array, "shape"), "Not an array"
                assert len(array.shape) == 2, "Array is not 2D"
        self._data = data
        self._updateGrid()
        self._initPlots()

    def _initPlots(self):
        """Update plots after setting the data:
        plot an image on all visible plots (if there are enough images).
        """
        for r, c in self._plots:
            plotIdx = r * self._ncols + c
            if r < self._nrows and c < self._ncols:
                if plotIdx < self._nframes:
                    self._plots[(r, c)].slider.setMaximum(self._nframes - 1)
                    oldValue = self._plots[(r, c)].slider.value()
                    # this should emit a slider.valueChanged signal and
                    # trigger plotting (in self._onSliderValueChanged)
                    self._plots[(r, c)].slider.setValue(plotIdx)
                    if oldValue == plotIdx:
                        # value not changed, we must plot
                        self._plots[(r, c)].plot.addImage(self._data[plotIdx])
                else:
                    self._plots[(r, c)].slider.setValue(0)
                    self._plots[(r, c)].plot.clear()

    def clear(self):
        """Clear all plots and set all slider values to 0"""
        for (r, c) in self._plots:
            self._plots[(r, c)].slider.setValue(0)
            self._plots[(r, c)].plot.clear()
