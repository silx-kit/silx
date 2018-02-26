# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
__date__ = "26/02/2018"


import collections
import logging
import time
import weakref

import numpy

from ..utils.weakref import WeakList
from ..gui import qt
from ..gui.plot import Plot1D, Plot2D, PlotWidget
from ..gui.plot.Colors import COLORDICT
from ..gui.plot.Colormap import Colormap
from silx.third_party import six


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
            if isinstance(second_arg, six.string_types):
                # curve defined as (y, style)
                y = first_arg
                style = second_arg
                curves.append((numpy.arange(len(y)), y, style))
            else:  # second_arg must be an array-like
                x = first_arg
                y = second_arg
                if len(args) >= 1 and isinstance(args[0], six.string_types):
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
            possible_colors = [c for c in COLORDICT if style.startswith(c)]
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


def imshow(data=None, cmap=None, norm=Colormap.LINEAR,
           vmin=None, vmax=None,
           aspect=False,
           origin=(0., 0.), scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """
    Plot an image in a :class:`~silx.gui.plot.PlotWindow.Plot2D` widget.

    How to use:

    >>> from silx import sx
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plt = sx.imshow(data, title='Random data')

    This function supports a subset of `matplotlib.pyplot.imshow
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_
    arguments.

    :param data: data to plot as an image
    :type data: numpy.ndarray-like with 2 dimensions
    :param str cmap: The name of the colormap to use for the plot.
    :param str norm: The normalization of the colormap:
                     'linear' (default) or 'log'
    :param float vmin: The value to use for the min of the colormap
    :param float vmax: The value to use for the max of the colormap
    :param bool aspect: True to keep aspect ratio (Default: False)
    :param origin: (ox, oy) The coordinates of the image origin in the plot
    :type origin: 2-tuple of floats
    :param scale: (sx, sy) The scale of the image in the plot
                  (i.e., the size of the image's pixel in plot coordinates)
    :type scale: 2-tuple of floats
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plt = Plot2D()
    plt.setGraphTitle(title)
    plt.getXAxis().setLabel(xlabel)
    plt.getYAxis().setLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plt.getDefaultColormap()
    if cmap is not None:
        colormap.setName(cmap)
    assert norm in Colormap.NORMALIZATIONS
    colormap.setNormalization(norm)
    colormap.setVMin(vmin)
    colormap.setVMax(vmax)
    plt.setDefaultColormap(colormap)

    # Handle aspect
    if aspect in (None, False, 'auto', 'normal'):
        plt.setKeepDataAspectRatio(False)
    elif aspect in (True, 'equal') or aspect == 1:
        plt.setKeepDataAspectRatio(True)
    else:
        _logger.warning(
            'imshow: Unhandled aspect argument: %s', str(aspect))

    if data is not None:
        data = numpy.array(data, copy=True)

        assert data.ndim in (2, 3)  # data or RGB(A)
        if data.ndim == 3:
            assert data.shape[-1] in (3, 4)  # RGB(A) image

        plt.addImage(data, origin=origin, scale=scale)

    plt.show()
    _plots.insert(0, plt)
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
        if isinstance(data, collections.Iterable):
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


class _GInputHandler(qt.QEventLoop):
    """Implements :func:`ginput`

    :param PlotWidget plot:
    :param int n:
    :param float timeout:
    """

    def __init__(self, plot, n, timeout):
        super(_GInputHandler, self).__init__()

        if not isinstance(plot, PlotWidget):
            raise ValueError('plot is not a PlotWidget: %s', plot)

        self._plot = plot
        self._timeout = timeout
        self._markersAndResult = []
        self._totalPoints = n
        self._endTime = 0.

    def eventFilter(self, obj, event):
        """Event filter for plot hide event"""
        if event.type() == qt.QEvent.Hide:
            self.quit()

        elif event.type() == qt.QEvent.KeyPress:
            if event.key() in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) or (
                    event.key() == qt.Qt.Key_Z and event.modifiers() & qt.Qt.ControlModifier):
                if len(self._markersAndResult) > 0:
                    legend, _ = self._markersAndResult.pop()
                    self._plot.remove(legend=legend, kind='marker')

                    self._updateStatusBar()
                    return True  # Stop further handling of those keys

            elif event.key() == qt.Qt.Key_Return:
                self.quit()
                return True  # Stop further handling of those keys

        return super(_GInputHandler, self).eventFilter(obj, event)

    def exec_(self):
        """Run blocking ginput handler

        :returns: List of selected points
        """
        # Bootstrap
        self._plot.setInteractiveMode(mode='zoom')
        self._handleInteractiveModeChanged(None)
        self._plot.sigInteractiveModeChanged.connect(
            self._handleInteractiveModeChanged)

        self._plot.installEventFilter(self)

        # Run
        if self._timeout:
            timeoutTimer = qt.QTimer()
            timeoutTimer.timeout.connect(self._updateStatusBar)
            timeoutTimer.start(1000)

            self._endTime = time.time() + self._timeout
            self._updateStatusBar()

            returnCode = super(_GInputHandler, self).exec_()

            timeoutTimer.stop()
        else:
            returnCode = super(_GInputHandler, self).exec_()

        # Clean-up
        self._plot.removeEventFilter(self)

        self._plot.sigInteractiveModeChanged.disconnect(
            self._handleInteractiveModeChanged)

        currentMode = self._plot.getInteractiveMode()
        if currentMode['mode'] == 'zoom':  # Stop handling mouse click
            self._plot.sigPlotSignal.disconnect(self._handleSelect)

        self._plot.statusBar().clearMessage()

        points = tuple(result for _, result in self._markersAndResult)

        for legend, _ in self._markersAndResult:
            self._plot.remove(legend=legend, kind='marker')
        self._markersAndResult = []

        return points if returnCode == 0 else ()

    def _updateStatusBar(self):
        """Update status bar message"""
        msg = 'ginput: %d/%d input points' % (len(self._markersAndResult),
                                              self._totalPoints)
        if self._timeout:
            remaining = self._endTime - time.time()
            if remaining < 0:
                self.quit()
                return
            msg += ', %d seconds remaining' % max(1, int(remaining))

        currentMode = self._plot.getInteractiveMode()
        if currentMode['mode'] != 'zoom':
            msg += ' (Use zoom mode to add/remove points)'

        self._plot.statusBar().showMessage(msg)

    def _handleSelect(self, event):
        """Handle mouse events"""
        if event['event'] == 'mouseClicked' and event['button'] == 'left':
            x, y = event['x'], event['y']
            xPixel, yPixel = event['xpixel'], event['ypixel']

            # Add marker
            legend = "sx.ginput %d" % len(self._markersAndResult)
            self._plot.addMarker(
                x, y,
                legend=legend,
                text='%d' % len(self._markersAndResult),
                color='red',
                draggable=False)

            # Pick item at selected position
            picked = self._plot._pickImageOrCurve(xPixel, yPixel)

            if picked is None:
                result = _GInputResult((x, y),
                                       item=None,
                                       indices=numpy.array((), dtype=int),
                                       data=None)

            elif picked[0] == 'curve':
                curve = picked[1]
                indices = picked[2]
                xData = curve.getXData(copy=False)[indices]
                yData = curve.getYData(copy=False)[indices]
                result = _GInputResult((x, y),
                                       item=curve,
                                       indices=indices,
                                       data=numpy.array((xData, yData)).T)

            elif picked[0] == 'image':
                image = picked[1]
                # Get corresponding coordinate in image
                origin = image.getOrigin()
                scale = image.getScale()
                column = int((x - origin[0]) / float(scale[0]))
                row = int((y - origin[1]) / float(scale[1]))
                data = image.getData(copy=False)[row, column]
                result = _GInputResult((x, y),
                                       item=image,
                                       indices=(row, column),
                                       data=data)

            self._markersAndResult.append((legend, result))
            self._updateStatusBar()
            if len(self._markersAndResult) == self._totalPoints:
                self.quit()

    def _handleInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot

        :param source: Objects that triggered the mode change
        """
        mode = self._plot.getInteractiveMode()
        if mode['mode'] == 'zoom':  # Handle click events
            self._plot.sigPlotSignal.connect(self._handleSelect)
        else:  # Do not handle click event
            self._plot.sigPlotSignal.disconnect(self._handleSelect)
        self._updateStatusBar()


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
        else:  # If no plot widgets are visible, take most recent one
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

    plot.raise_()  # So window becomes the top level one

    _logger.info('Performing ginput with plot widget %s', str(plot))
    handler = _GInputHandler(plot, n, timeout)
    points = handler.exec_()

    return points
