# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "27/09/2016"


import logging
import numpy

from ..gui.plot import Plot1D, Plot2D
from ..gui.plot.Colors import COLORDICT

try:
    from ..third_party import six
except ImportError:
    import six


_logger = logging.getLogger(__name__)


def plot(*args, **kwargs):
    """
    Plot curves in a dedicated widget.

    This function supports a subset of matplotlib.pyplot.plot arguments.
    See: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

    It opens a silx PlotWindow with its associated tools.

    Examples:

    First import :mod:`sx` function:

    >>> from silx import sx
    >>> import numpy

    Plot a single curve given some values:

    >>> values = numpy.random.random(100)
    >>> plot_1curve = sx.plot(values, title='Random data')

    Plot a single curve given the x and y values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> sin_a = numpy.sin(angles)
    >>> plot_sinus = sx.plot(angles, sin_a,
    ...                      xlabel='angle (radian)', ylabel='sin(a)')

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

    Remark: The first curve will always be displayed in black no matter the
    given color. This is because it is selected by default and this is shown
    by using the black color.

    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plt = Plot1D()
    if 'title' in kwargs:
        plt.setGraphTitle(kwargs['title'])
    if 'xlabel' in kwargs:
        plt.setGraphXLabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        plt.setGraphYLabel(kwargs['ylabel'])

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
        symbol, linestyle, color = None, None, None

        # Parse style
        if style:
            # Handle color first
            for c in COLORDICT:
                if style.startswith(c):
                    color = c
                    style = style[len(c):]

            if style:
                # Run twice to handle inversion symbol/linestyle
                for i in range(2):
                    # Handle linestyle
                    for line in (' ', '-', '--', '-.', ':'):
                        if style.endswith(line):
                            linestyle = line
                            style = style[:-len(line)]
                            break

                    # Handle symbol
                    for marker in ('o', '.', ',', '+', 'x', 'd', 's'):
                        if style.endswith(marker):
                            symbol = style[-1]
                            style = style[:-1]
                            break

        plt.addCurve(x, y,
                     legend=('curve_%d' % index),
                     symbol=symbol,
                     linestyle=linestyle,
                     color=color)

    plt.show()
    return plt


def imshow(data=None, cmap=None, norm='linear',
           vmin=None, vmax=None,
           aspect=False,
           origin=(0., 0.), scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """Plot an image in a dedicated widget.

    This function supports a subset of matplotlib.pyplot.imshow arguments.
    See: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

    It opens a silx PlotWindow with its associated tools.

    Example to plot an image:

    >>> from silx import sx
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plt = sx.imshow(data, title='Random data')

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
    plt.setGraphXLabel(xlabel)
    plt.setGraphYLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plt.getDefaultColormap()
    if cmap is not None:
        colormap['name'] = cmap
    assert norm in ('linear', 'log')
    colormap['normalization'] = norm
    if vmin is not None:
        colormap['vmin'] = vmin
    if vmax is not None:
        colormap['vmax'] = vmax
    if vmin is not None and vmax is not None:
        colormap['autoscale'] = False
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
    return plt
