# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module defines widgets used by _NXdataView.
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "10/10/2018"

import numpy

from silx.gui import qt
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.plot import Plot1D, Plot2D, StackView, ScatterView
from silx.gui.colors import Colormap
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser

from silx.math.calibration import ArrayCalibration, NoCalibration, LinearCalibration


class ArrayCurvePlot(qt.QWidget):
    """
    Widget for plotting a curve from a multi-dimensional signal array
    and a 1D axis array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last dimension must have the same length as
    the axis array.

    The widget provides sliders to select indices on the first (n - 1)
    dimensions of the signal array, and buttons to add/replace selected
    curves to the plot.

    This widget also handles simple 2D or 3D scatter plots (third dimension
    displayed as colour of points).
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayCurvePlot, self).__init__(parent)

        self.__signals = None
        self.__signals_names = None
        self.__signal_errors = None
        self.__axis = None
        self.__axis_name = None
        self.__x_axis_errors = None
        self.__values = None

        self._plot = Plot1D(self)

        self.selectorDock = qt.QDockWidget("Data selector", self._plot)
        # not closable
        self.selectorDock.setFeatures(qt.QDockWidget.DockWidgetMovable |
                                      qt.QDockWidget.DockWidgetFloatable)
        self._selector = NumpyAxesSelector(self.selectorDock)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False
        self.selectorDock.setWidget(self._selector)
        self._plot.addTabbedDockWidget(self.selectorDock)

        self._plot.sigActiveCurveChanged.connect(self._setYLabelFromActiveLegend)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0)

        self.setLayout(layout)

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: Plot1D
        """
        return self._plot

    def setCurvesData(self, ys, x=None,
                      yerror=None, xerror=None,
                      ylabels=None, xlabel=None, title=None):
        """

        :param List[ndarray] ys: List of arrays to be represented by the y (vertical) axis.
            It can be multiple n-D array whose last dimension must
            have the same length as x (and values must be None)
        :param ndarray x: 1-D dataset used as the curve's x values. If provided,
            its lengths must be equal to the length of the last dimension of
            ``y`` (and equal to the length of ``value``, for a scatter plot).
        :param ndarray yerror: Single array of errors for y (same shape), or None.
            There can only be one array, and it applies to the first/main y
            (no y errors for auxiliary_signals curves).
        :param ndarray xerror: 1-D dataset of errors for x, or None
        :param str ylabels: Labels for each curve's Y axis
        :param str xlabel: Label for X axis
        :param str title: Graph title
        """
        self.__signals = ys
        self.__signals_names = ylabels or (["Y"] * len(ys))
        self.__signal_errors = yerror
        self.__axis = x
        self.__axis_name = xlabel
        self.__x_axis_errors = xerror

        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._updateCurve)
            self.__selector_is_connected = False
        self._selector.setData(ys[0])
        self._selector.setAxisNames(["Y"])

        if len(ys[0].shape) < 2:
            self.selectorDock.hide()
        else:
            self.selectorDock.show()

        self._plot.setGraphTitle(title or "")
        self._updateCurve()

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._updateCurve)
            self.__selector_is_connected = True

    def _updateCurve(self):
        selection = self._selector.selection()
        ys = [sig[selection] for sig in self.__signals]
        y0 = ys[0]
        len_y = len(y0)
        x = self.__axis
        if x is None:
            x = numpy.arange(len_y)
        elif numpy.isscalar(x) or len(x) == 1:
            # constant axis
            x = x * numpy.ones_like(y0)
        elif len(x) == 2 and len_y != 2:
            # linear calibration a + b * x
            x = x[0] + x[1] * numpy.arange(len_y)

        self._plot.remove(kind=("curve",))

        for i in range(len(self.__signals)):
            legend = self.__signals_names[i]

            # errors only supported for primary signal in NXdata
            y_errors = None
            if i == 0 and self.__signal_errors is not None:
                y_errors = self.__signal_errors[self._selector.selection()]
            self._plot.addCurve(x, ys[i], legend=legend,
                                xerror=self.__x_axis_errors,
                                yerror=y_errors)
            if i == 0:
                self._plot.setActiveCurve(legend)

        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__axis_name)
        self._plot.getYAxis().setLabel(self.__signals_names[0])

    def _setYLabelFromActiveLegend(self, previous_legend, new_legend):
        for ylabel in self.__signals_names:
            if new_legend is not None and new_legend == ylabel:
                self._plot.getYAxis().setLabel(ylabel)
                break

    def clear(self):
        self._plot.clear()


class XYVScatterPlot(qt.QWidget):
    """
    Widget for plotting one or more scatters
    (with identical x, y coordinates).
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(XYVScatterPlot, self).__init__(parent)

        self.__y_axis = None
        """1D array"""
        self.__y_axis_name = None
        self.__values = None
        """List of 1D arrays (for multiple scatters with identical
        x, y coordinates)"""

        self.__x_axis = None
        self.__x_axis_name = None
        self.__x_axis_errors = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__y_axis_errors = None

        self._plot = ScatterView(self)
        self._plot.setColormap(Colormap(name="viridis",
                                        vmin=None, vmax=None,
                                        normalization=Colormap.LINEAR))

        self._slider = HorizontalSliderWithBrowser(parent=self)
        self._slider.setMinimum(0)
        self._slider.setValue(0)
        self._slider.valueChanged[int].connect(self._sliderIdxChanged)
        self._slider.setToolTip("Select auxiliary signals")

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0)
        layout.addWidget(self._slider, 1, 0)

        self.setLayout(layout)

    def _sliderIdxChanged(self, value):
        self._updateScatter()

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: PlotWidget
        """
        return self._plot.getPlotWidget()

    def setScattersData(self, y, x, values,
                        yerror=None, xerror=None,
                        ylabel=None, xlabel=None,
                        title="", scatter_titles=None):
        """

        :param ndarray y: 1D array  for y (vertical) coordinates.
        :param ndarray x: 1D array for x coordinates.
        :param List[ndarray] values: List of 1D arrays of values.
            This will be used to compute the color map and assign colors
            to the points. There should be as many arrays in the list as
            scatters to be represented.
        :param ndarray yerror: 1D array of errors for y (same shape), or None.
        :param ndarray xerror: 1D array of errors for x, or None
        :param str ylabel: Label for Y axis
        :param str xlabel: Label for X axis
        :param str title: Main graph title
        :param List[str] scatter_titles:  Subtitles (one per scatter)
        """
        self.__y_axis = y
        self.__x_axis = x
        self.__x_axis_name = xlabel or "X"
        self.__y_axis_name = ylabel or "Y"
        self.__x_axis_errors = xerror
        self.__y_axis_errors = yerror
        self.__values = values

        self.__graph_title = title or ""
        self.__scatter_titles = scatter_titles

        self._slider.valueChanged[int].disconnect(self._sliderIdxChanged)
        self._slider.setMaximum(len(values) - 1)
        if len(values) > 1:
            self._slider.show()
        else:
            self._slider.hide()
        self._slider.setValue(0)
        self._slider.valueChanged[int].connect(self._sliderIdxChanged)

        self._updateScatter()

    def _updateScatter(self):
        x = self.__x_axis
        y = self.__y_axis

        idx = self._slider.value()

        title = ""
        if self.__graph_title:
            title += self.__graph_title + "\n"  # main NXdata @title
        title += self.__scatter_titles[idx]     # scatter dataset name

        self._plot.setGraphTitle(title)
        self._plot.setData(x, y, self.__values[idx],
                           xerror=self.__x_axis_errors,
                           yerror=self.__y_axis_errors)
        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__x_axis_name)
        self._plot.getYAxis().setLabel(self.__y_axis_name)

    def clear(self):
        self._plot.getPlotWidget().clear()


class ArrayImagePlot(qt.QWidget):
    """
    Widget for plotting an image from a multi-dimensional signal array
    and two 1D axes array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last two dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 2) dimensions of
    the signal array, and the plot is updated to show the image corresponding
    to the selection.

    If one or both of the axes does not have regularly spaced values, the
    the image is plotted as a coloured scatter plot.
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayImagePlot, self).__init__(parent)

        self.__signals = None
        self.__signals_names = None
        self.__x_axis = None
        self.__x_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None

        self._plot = Plot2D(self)
        self._plot.setDefaultColormap(Colormap(name="viridis",
                                               vmin=None, vmax=None,
                                               normalization=Colormap.LINEAR))
        self._plot.getIntensityHistogramAction().setVisible(True)

        self.selectorDock = qt.QDockWidget("Data selector", self._plot)
        # not closable
        self.selectorDock.setFeatures(qt.QDockWidget.DockWidgetMovable |
                                      qt.QDockWidget.DockWidgetFloatable)
        self._selector = NumpyAxesSelector(self.selectorDock)
        self._selector.setNamedAxesSelectorVisibility(False)
        self._selector.selectionChanged.connect(self._updateImage)

        self._auxSigSlider = HorizontalSliderWithBrowser(parent=self)
        self._auxSigSlider.setMinimum(0)
        self._auxSigSlider.setValue(0)
        self._auxSigSlider.valueChanged[int].connect(self._sliderIdxChanged)
        self._auxSigSlider.setToolTip("Select auxiliary signals")

        layout = qt.QVBoxLayout()
        layout.addWidget(self._plot)
        layout.addWidget(self._auxSigSlider)
        self.selectorDock.setWidget(self._selector)
        self._plot.addTabbedDockWidget(self.selectorDock)

        self.setLayout(layout)

    def _sliderIdxChanged(self, value):
        self._updateImage()

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: Plot2D
        """
        return self._plot

    def setImageData(self, signals,
                     x_axis=None, y_axis=None,
                     signals_names=None,
                     xlabel=None, ylabel=None,
                     title=None, isRgba=False):
        """

        :param signals: list of n-D datasets, whose last 2 dimensions are used as the
            image's values, or list of 3D datasets interpreted as RGBA image.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param signals_names: Names for each image, used as subtitle and legend.
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param title: Graph title
        :param isRgba: True if data is a 3D RGBA image
        """
        self._selector.selectionChanged.disconnect(self._updateImage)
        self._auxSigSlider.valueChanged.disconnect(self._sliderIdxChanged)

        self.__signals = signals
        self.__signals_names = signals_names
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__title = title

        self._selector.clear()
        if not isRgba:
            self._selector.setAxisNames(["Y", "X"])
            img_ndim = 2
        else:
            self._selector.setAxisNames(["Y", "X", "RGB(A) channel"])
            img_ndim = 3
        self._selector.setData(signals[0])

        if len(signals[0].shape) <= img_ndim:
            self.selectorDock.hide()
        else:
            self.selectorDock.show()

        self._auxSigSlider.setMaximum(len(signals) - 1)
        if len(signals) > 1:
            self._auxSigSlider.show()
        else:
            self._auxSigSlider.hide()
        self._auxSigSlider.setValue(0)

        self._updateImage()

        self._selector.selectionChanged.connect(self._updateImage)
        self._auxSigSlider.valueChanged.connect(self._sliderIdxChanged)

    def _updateImage(self):
        selection = self._selector.selection()
        auxSigIdx = self._auxSigSlider.value()

        legend = self.__signals_names[auxSigIdx]

        images = [img[selection] for img in self.__signals]
        image = images[auxSigIdx]

        x_axis = self.__x_axis
        y_axis = self.__y_axis

        if x_axis is None and y_axis is None:
            xcalib = NoCalibration()
            ycalib = NoCalibration()
        else:
            if x_axis is None:
                # no calibration
                x_axis = numpy.arange(image.shape[1])
            elif numpy.isscalar(x_axis) or len(x_axis) == 1:
                # constant axis
                x_axis = x_axis * numpy.ones((image.shape[1], ))
            elif len(x_axis) == 2:
                # linear calibration
                x_axis = x_axis[0] * numpy.arange(image.shape[1]) + x_axis[1]

            if y_axis is None:
                y_axis = numpy.arange(image.shape[0])
            elif numpy.isscalar(y_axis) or len(y_axis) == 1:
                y_axis = y_axis * numpy.ones((image.shape[0], ))
            elif len(y_axis) == 2:
                y_axis = y_axis[0] * numpy.arange(image.shape[0]) + y_axis[1]

            xcalib = ArrayCalibration(x_axis)
            ycalib = ArrayCalibration(y_axis)

        self._plot.remove(kind=("scatter", "image",))
        if xcalib.is_affine() and ycalib.is_affine():
            # regular image
            xorigin, xscale = xcalib(0), xcalib.get_slope()
            yorigin, yscale = ycalib(0), ycalib.get_slope()
            origin = (xorigin, yorigin)
            scale = (xscale, yscale)

            self._plot.addImage(image, legend=legend,
                                origin=origin, scale=scale,
                                replace=True)
        else:
            scatterx, scattery = numpy.meshgrid(x_axis, y_axis)
            # fixme: i don't think this can handle "irregular" RGBA images
            self._plot.addScatter(numpy.ravel(scatterx),
                                  numpy.ravel(scattery),
                                  numpy.ravel(image),
                                  legend=legend)

        title = ""
        if self.__title:
            title += self.__title
        if not title.strip().endswith(self.__signals_names[auxSigIdx]):
            title += "\n" + self.__signals_names[auxSigIdx]
        self._plot.setGraphTitle(title)
        self._plot.getXAxis().setLabel(self.__x_axis_name)
        self._plot.getYAxis().setLabel(self.__y_axis_name)
        self._plot.resetZoom()

    def clear(self):
        self._plot.clear()


class ArrayStackPlot(qt.QWidget):
    """
    Widget for plotting a n-D array (n >= 3) as a stack of images.
    Three axis arrays can be provided to calibrate the axes.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last 3 dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 3) dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """
    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super(ArrayStackPlot, self).__init__(parent)

        self.__signal = None
        self.__signal_name = None
        # the Z, Y, X axes apply to the last three dimensions of the signal
        # (in that order)
        self.__z_axis = None
        self.__z_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__x_axis = None
        self.__x_axis_name = None

        self._stack_view = StackView(self)
        self._hline = qt.QFrame(self)
        self._hline.setFrameStyle(qt.QFrame.HLine)
        self._hline.setFrameShadow(qt.QFrame.Sunken)
        self._legend = qt.QLabel(self)
        self._selector = NumpyAxesSelector(self)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._stack_view)
        layout.addWidget(self._hline)
        layout.addWidget(self._legend)
        layout.addWidget(self._selector)

        self.setLayout(layout)

    def getStackView(self):
        """Returns the plot used for the display

        :rtype: StackView
        """
        return self._stack_view

    def setStackData(self, signal,
                     x_axis=None, y_axis=None, z_axis=None,
                     signal_name=None,
                     xlabel=None, ylabel=None, zlabel=None,
                     title=None):
        """

        :param signal: n-D dataset, whose last 3 dimensions are used as the
            3D stack values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param z_axis: 1-D dataset used as the image's z. If provided,
            its lengths must be equal to the length of the 3rd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param zlabel: Label for Z axis
        :param title: Graph title
        """
        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._updateStack)
            self.__selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__z_axis = z_axis
        self.__z_axis_name = zlabel

        self._selector.setData(signal)
        self._selector.setAxisNames(["Y", "X", "Z"])

        self._stack_view.setGraphTitle(title or "")
        # by default, the z axis is the image position (dimension not plotted)
        self._stack_view.getPlot().getXAxis().setLabel(self.__x_axis_name or "X")
        self._stack_view.getPlot().getYAxis().setLabel(self.__y_axis_name or "Y")

        self._updateStack()

        ndims = len(signal.shape)
        self._stack_view.setFirstStackDimension(ndims - 3)

        # the legend label shows the selection slice producing the volume
        # (only interesting for ndim > 3)
        if ndims > 3:
            self._selector.setVisible(True)
            self._legend.setVisible(True)
            self._hline.setVisible(True)
        else:
            self._selector.setVisible(False)
            self._legend.setVisible(False)
            self._hline.setVisible(False)

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._updateStack)
            self.__selector_is_connected = True

    @staticmethod
    def _get_origin_scale(axis):
        """Assuming axis is a regularly spaced 1D array,
        return a tuple (origin, scale) where:
            - origin = axis[0]
            - scale = (axis[n-1] - axis[0]) / (n -1)
        :param axis: 1D numpy array
        :return: Tuple (axis[0], (axis[-1] - axis[0]) / (len(axis) - 1))
        """
        return axis[0], (axis[-1] - axis[0]) / (len(axis) - 1)

    def _updateStack(self):
        """Update displayed stack according to the current axes selector
        data."""
        stk = self._selector.selectedData()
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        z_axis = self.__z_axis

        calibrations = []
        for axis in [z_axis, y_axis, x_axis]:

            if axis is None:
                calibrations.append(NoCalibration())
            elif len(axis) == 2:
                calibrations.append(
                        LinearCalibration(y_intercept=axis[0],
                                          slope=axis[1]))
            else:
                calibrations.append(ArrayCalibration(axis))

        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        self._stack_view.setStack(stk, calibrations=calibrations)
        self._stack_view.setLabels(
                labels=[self.__z_axis_name,
                        self.__y_axis_name,
                        self.__x_axis_name])

    def clear(self):
        self._stack_view.clear()
