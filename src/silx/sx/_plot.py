# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""This module adds convenient functions to use plot widgets from the console.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/11/2018"


import collections
try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import logging
import weakref

import numpy

from ..utils.weakref import WeakList
from ..gui import qt
from ..gui.plot import Plot1D, Plot2D, ScatterView
from ..gui.plot import items
from ..gui import colors
from ..gui.plot.tools import roi
from ..gui.plot.items import roi as roi_items
from ..gui.plot.tools.toolbars import InteractiveModeToolBar

_logger = logging.getLogger(__name__)

_plots = WeakList()
"""List of widgets created through plot and imshow"""


def plot(*args, **kwargs):
    """
    Plot curves in a :class:`~silx.gui.plot.PlotWindow.Plot1D` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    Plot a single curve given some values:

    >>> values = numpy.random.random(100)
    >>> plot_1curve = sx.plot(values, title='Random data')

    Plot a single curve given the x and y values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> sin_a = numpy.sin(angles)
    >>> plot_sinus = sx.plot(angles, sin_a, xlabel='angle (radian)', ylabel='sin(a)')

    Plot many curves by giving a 2D array, provided xn, yn arrays:

    >>> plot_curves = sx.plot(x0, y0, x1, y1, x2, y2, ...)

    Plot curve with style giving a style string:

    >>> plot_styled = sx.plot(x0, y0, 'ro-', x1, y1, 'b.')

    Supported symbols:

        - 'o' circle
        - '.' point
        - ',' pixel
        - '+' cross
        - 'x' x-cross
        - 'd' diamond
        - 's' square

    Supported types of line:

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

    If provided, the names arguments color, linestyle, linewidth and marker
    override any style provided to a curve.

    This function supports a subset of `matplotlib.pyplot.plot
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_
    arguments.

    :param str color: Color to use for all curves (default: None)
    :param str linestyle: Type of line to use for all curves (default: None)
    :param float linewidth: With of all the curves (default: 1)
    :param str marker: Symbol to use for all the curves (default: None)
    :param str title: The title of the Plot widget (default: None)
    :param str xlabel: The label of the X axis (default: None)
    :param str ylabel: The label of the Y axis (default: None)
    :return: The widget plotting the curve(s)
    :rtype: silx.gui.plot.Plot1D
    """
    plt = Plot1D()
    if 'title' in kwargs:
        plt.setGraphTitle(kwargs['title'])
    if 'xlabel' in kwargs:
        plt.getXAxis().setLabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.getYAxis().setLabel(kwargs['ylabel'])

    color = kwargs.get('color')
    linestyle = kwargs.get('linestyle')
    linewidth = kwargs.get('linewidth')
    marker = kwargs.get('marker')

    # Parse args and store curves as (x, y, style string)
    args = list(args)
    curves = []
    while args:
        first_arg = args.pop(0)  # Process an arg

        if len(args) == 0:
            # Last curve defined as (y,)
            curves.append((numpy.arange(len(first_arg)), first_arg, None))
        else:
            second_arg = args.pop(0)
            if isinstance(second_arg, str):
                # curve defined as (y, style)
                y = first_arg
                style = second_arg
                curves.append((numpy.arange(len(y)), y, style))
            else:  # second_arg must be an array-like
                x = first_arg
                y = second_arg
                if len(args) >= 1 and isinstance(args[0], str):
                    # Curve defined as (x, y, style)
                    style = args.pop(0)
                    curves.append((x, y, style))
                else:
                    # Curve defined as (x, y)
                    curves.append((x, y, None))

    for index, curve in enumerate(curves):
        x, y, style = curve

        # Default style
        curve_symbol, curve_linestyle, curve_color = None, None, None

        # Parse style
        if style:
            # Handle color first
            possible_colors = [c for c in colors.COLORDICT if style.startswith(c)]
            if possible_colors:  # Take the longest string matching a color name
                curve_color = possible_colors[0]
                for c in possible_colors[1:]:
                    if len(c) > len(curve_color):
                        curve_color = c
                style = style[len(curve_color):]

            if style:
                # Run twice to handle inversion symbol/linestyle
                for _i in range(2):
                    # Handle linestyle
                    for line in (' ', '--', '-', '-.', ':'):
                        if style.endswith(line):
                            curve_linestyle = line
                            style = style[:-len(line)]
                            break

                    # Handle symbol
                    for curve_marker in ('o', '.', ',', '+', 'x', 'd', 's'):
                        if style.endswith(curve_marker):
                            curve_symbol = style[-1]
                            style = style[:-1]
                            break

        # As in matplotlib, marker, linestyle and color override other style
        plt.addCurve(x, y,
                     legend=('curve_%d' % index),
                     symbol=marker or curve_symbol,
                     linestyle=linestyle or curve_linestyle,
                     linewidth=linewidth,
                     color=color or curve_color)

    plt.show()
    _plots.insert(0, plt)
    return plt


def imshow(data=None, cmap=None, norm=colors.Colormap.LINEAR,
           vmin=None, vmax=None,
           aspect=False,
           origin='upper', scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """
    Plot an image in a :class:`~silx.gui.plot.PlotWindow.Plot2D` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plt = sx.imshow(data, title='Random data')

    By default, the image origin is displayed in the upper left
    corner of the plot. To invert the Y axis, and place the image origin
    in the lower left corner of the plot, use the *origin* parameter:

     >>> plt = sx.imshow(data, origin='lower')

    This function supports a subset of `matplotlib.pyplot.imshow
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_
    arguments.

    :param data: data to plot as an image
    :type data: numpy.ndarray-like with 2 dimensions
    :param str cmap: The name of the colormap to use for the plot. It also
        supports a numpy array containing a RGB LUT, or a `colors.Colormap`
        instance.
    :param str norm: The normalization of the colormap:
                     'linear' (default) or 'log'
    :param float vmin: The value to use for the min of the colormap
    :param float vmax: The value to use for the max of the colormap
    :param bool aspect: True to keep aspect ratio (Default: False)
    :param origin: Either image origin as the Y axis orientation:
        'upper' (default) or 'lower'
        or the coordinates (ox, oy) of the image origin in the plot.
    :type origin: str or 2-tuple of floats
    :param scale: (sx, sy) The scale of the image in the plot
                  (i.e., the size of the image's pixel in plot coordinates)
    :type scale: 2-tuple of floats
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    :return: The widget plotting the image
    :rtype: silx.gui.plot.Plot2D
    """
    plt = Plot2D()
    plt.setGraphTitle(title)
    plt.getXAxis().setLabel(xlabel)
    plt.getYAxis().setLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plt.getDefaultColormap()
    if isinstance(cmap, colors.Colormap):
        colormap = cmap
        plt.setDefaultColormap(colormap)
    elif isinstance(cmap, numpy.ndarray):
        colormap.setColors(cmap)
    elif cmap is not None:
        colormap.setName(cmap)
    assert norm in colors.Colormap.NORMALIZATIONS
    colormap.setNormalization(norm)
    colormap.setVMin(vmin)
    colormap.setVMax(vmax)

    # Handle aspect
    if aspect in (None, False, 'auto', 'normal'):
        plt.setKeepDataAspectRatio(False)
    elif aspect in (True, 'equal') or aspect == 1:
        plt.setKeepDataAspectRatio(True)
    else:
        _logger.warning(
            'imshow: Unhandled aspect argument: %s', str(aspect))

    # Handle matplotlib-like origin
    if origin in ('upper', 'lower'):
        plt.setYAxisInverted(origin == 'upper')
        origin = 0., 0.  # Set origin to the definition of silx

    if data is not None:
        data = numpy.array(data, copy=True)

        assert data.ndim in (2, 3)  # data or RGB(A)
        if data.ndim == 3:
            assert data.shape[-1] in (3, 4)  # RGB(A) image

        plt.addImage(data, origin=origin, scale=scale)

    plt.show()
    _plots.insert(0, plt)
    return plt


def scatter(x=None, y=None, value=None, size=None,
            marker=None,
            cmap=None, norm=colors.Colormap.LINEAR,
            vmin=None, vmax=None):
    """
    Plot scattered data in a :class:`~silx.gui.plot.ScatterView` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    >>> x = numpy.random.random(100)
    >>> y = numpy.random.random(100)
    >>> values = numpy.random.random(100)
    >>> plt = sx.scatter(x, y, values, cmap='viridis')

        Supported symbols:

        - 'o' circle
        - '.' point
        - ',' pixel
        - '+' cross
        - 'x' x-cross
        - 'd' diamond
        - 's' square

    This function supports a subset of `matplotlib.pyplot.scatter
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_
    arguments.

    :param numpy.ndarray x: 1D array-like of x coordinates
    :param numpy.ndarray y: 1D array-like of y coordinates
    :param numpy.ndarray value: 1D array-like of data values
    :param float size: Size^2 of the markers
    :param str marker: Symbol used to represent the points
    :param str cmap: The name of the colormap to use for the plot
    :param str norm: The normalization of the colormap:
                     'linear' (default) or 'log'
    :param float vmin: The value to use for the min of the colormap
    :param float vmax: The value to use for the max of the colormap
    :return: The widget plotting the scatter plot
    :rtype: silx.gui.plot.ScatterView.ScatterView
    """
    plt = ScatterView()

    # Update default colormap with input parameters
    colormap = plt.getPlotWidget().getDefaultColormap()
    if cmap is not None:
        colormap.setName(cmap)
    assert norm in colors.Colormap.NORMALIZATIONS
    colormap.setNormalization(norm)
    colormap.setVMin(vmin)
    colormap.setVMax(vmax)
    plt.getPlotWidget().setDefaultColormap(colormap)

    if x is not None and y is not None:  # Add a scatter plot
        x = numpy.array(x, copy=True).reshape(-1)
        y = numpy.array(y, copy=True).reshape(-1)
        assert len(x) == len(y)

        if value is None:
            value = numpy.ones(len(x), dtype=numpy.float32)

        elif isinstance(value, abc.Iterable):
            value = numpy.array(value, copy=True).reshape(-1)
            assert len(x) == len(value)

        else:
            value = numpy.ones(len(x), dtype=numpy.float64) * value

        plt.setData(x, y, value)
        item = plt.getScatterItem()
        if marker is not None:
            item.setSymbol(marker)
        if size is not None:
            item.setSymbolSize(numpy.sqrt(size))

        plt.resetZoom()

    plt.show()
    _plots.insert(0, plt.getPlotWidget())
    return plt


class _GInputResult(tuple):
    """Object storing :func:`ginput` result

    :param position: Selected point coordinates in the plot (x, y)
    :param Item item: Plot item under the selected position
    :param indices: Selected indices in the data of the item.
       For a curve it is a list of indices, for an image it is (row, column)
    :param data: Value of data at selected indices.
       For a curve it is an array of values, for an image it is a single value
    """

    def __new__(cls, position, item, indices, data):
        return super(_GInputResult, cls).__new__(cls, position)

    def __init__(self, position, item, indices, data):
        self._itemRef = weakref.ref(item) if item is not None else None
        self._indices = numpy.array(indices, copy=True)
        if isinstance(data, abc.Iterable):
            self._data = numpy.array(data, copy=True)
        else:
            self._data = data

    def getItem(self):
        """Returns the item at the selected position if any.

        :return: plot item under the selected postion.
           It is None if there was no item at that position or if
           it is no more in the plot.
        :rtype: silx.gui.plot.items.Item"""
        return None if self._itemRef is None else self._itemRef()

    def getIndices(self):
        """Returns indices in data array at the select position

        :return: 1D array of indices for curve and (row, column) for images
        :rtype: numpy.ndarray
        """
        return numpy.array(self._indices, copy=True)

    def getData(self):
        """Returns data value at the selected position.

        For curves, an array of (x, y) values close to the point is returned.
        For images, either a single value or a RGB(A) array is returned.

        :return: 2D array of (x, y) data values for curves (Nx2),
            a single value for data images and RGB(A) array for images.
        """
        if isinstance(self._data, numpy.ndarray):
            return numpy.array(self._data, copy=True)
        else:
            return self._data


class _GInputHandler(roi.InteractiveRegionOfInterestManager):
    """Implements :func:`ginput`

    :param PlotWidget plot:
    :param int n:  Max number of points to request
    :param float timeout: Timeout in seconds
    """

    def __init__(self, plot, n, timeout):
        super(_GInputHandler, self).__init__(plot)

        self._timeout = timeout
        self.__selections = collections.OrderedDict()

        window = plot.window()  # Retrieve window containing PlotWidget
        statusBar = window.statusBar()
        self.sigMessageChanged.connect(statusBar.showMessage)
        self.setMaxRois(n)
        self.setValidationMode(self.ValidationMode.AUTO_ENTER)
        self.sigRoiAdded.connect(self.__added)
        self.sigRoiAboutToBeRemoved.connect(self.__removed)

    def exec(self):
        """Request user inputs

        :return: List of selection points information
        """
        plot = self.parent()
        if plot is None:
            return

        window = plot.window()  # Retrieve window containing PlotWidget

        # Add ROI point interactive mode action
        for toolbar in window.findChildren(qt.QToolBar):
            if isinstance(toolbar, InteractiveModeToolBar):
                break
        else:  # Add a toolbar
            toolbar = qt.QToolBar()
            window.addToolBar(toolbar)
        toolbar.addAction(self.getInteractionModeAction(roi_items.PointROI))

        super(_GInputHandler, self).exec(roiClass=roi_items.PointROI, timeout=self._timeout)

        if isinstance(toolbar, InteractiveModeToolBar):
            toolbar.removeAction(self.getInteractionModeAction(roi_items.PointROI))
        else:
            toolbar.setParent(None)

        return tuple(self.__selections.values())

    def exec_(self):  # Qt5-like compatibility
        return self.exec()

    def __updateSelection(self, roi):
        """Perform picking and update selection list

        :param RegionOfInterest roi:
        """
        plot = self.parent()
        if plot is None:
            return  # No plot, abort

        if not isinstance(roi, roi_items.PointROI):
            # Only handle points
            raise RuntimeError("Unexpected item")

        x, y = roi.getPosition()
        xPixel, yPixel = plot.dataToPixel(x, y, axis='left', check=False)

        # Pick item at selected position
        pickingResult = plot._pickTopMost(
            xPixel, yPixel,
            lambda item: isinstance(item, (items.ImageBase, items.Curve)))

        if pickingResult is None:
            result = _GInputResult((x, y),
                                   item=None,
                                   indices=numpy.array((), dtype=int),
                                   data=None)
        else:
            item = pickingResult.getItem()
            indices = pickingResult.getIndices(copy=True)

            if isinstance(item, items.Curve):
                xData = item.getXData(copy=False)[indices]
                yData = item.getYData(copy=False)[indices]
                result = _GInputResult((x, y),
                                       item=item,
                                       indices=indices,
                                       data=numpy.array((xData, yData)).T)

            elif isinstance(item, items.ImageBase):
                row, column = indices[0][0], indices[1][0]
                data = item.getData(copy=False)[row, column]
                result = _GInputResult((x, y),
                                       item=item,
                                       indices=(row, column),
                                       data=data)

        self.__selections[roi] = result

    def __added(self, roi):
        """Handle new ROI added

        :param RegionOfInterest roi:
        """
        if isinstance(roi, roi_items.PointROI):
            # Only handle points
            roi.setName('%d' % len(self.__selections))
            self.__updateSelection(roi)
            roi.sigRegionChanged.connect(self.__regionChanged)

    def __removed(self, roi):
        """Handle ROI removed"""
        if self.__selections.pop(roi, None) is not None:
            roi.sigRegionChanged.disconnect(self.__regionChanged)

    def __regionChanged(self):
        """Handle update of a ROI"""
        roi = self.sender()
        self.__updateSelection(roi)


def ginput(n=1, timeout=30, plot=None):
    """Get input points on a plot.

    If no plot is provided, it uses a plot widget created with
    either :func:`silx.sx.plot` or :func:`silx.sx.imshow`.

    How to use:

    >>> from silx import sx

    >>> sx.imshow(image)  # Plot the image
    >>> sx.ginput(1)  # Request selection on the image plot
    ((0.598, 1.234))

    How to get more information about the selected positions:

    >>> positions = sx.ginput(1)

    >>> positions[0].getData()  # Returns value(s) at selected position

    >>> positions[0].getIndices()  # Returns data indices at selected position

    >>> positions[0].getItem()  # Returns plot item at selected position

    :param int n: Number of points the user need to select
    :param float timeout: Timeout in seconds before ginput returns
        event if selection is not completed
    :param silx.gui.plot.PlotWidget.PlotWidget plot: An optional PlotWidget
        from which to get input
    :return: List of clicked points coordinates (x, y) in plot
    :raise ValueError: If provided plot is not a PlotWidget
    """
    if plot is None:
        # Select most recent visible plot widget
        for widget in _plots:
            if widget.isVisible():
                plot = widget
                break
        else:  # If no plot widget is visible, take the most recent one
            try:
                plot = _plots[0]
            except IndexError:
                pass
            else:
                plot.show()

        if plot is None:
            _logger.warning('No plot available to perform ginput, create one')
            plot = Plot1D()
            plot.show()
            _plots.insert(0, plot)

    plot.raise_()  # So window becomes the top level one

    _logger.info('Performing ginput with plot widget %s', str(plot))
    handler = _GInputHandler(plot, n, timeout)
    points = handler.exec()

    return points
