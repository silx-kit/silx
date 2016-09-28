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

from silx.gui.plot import Plot1D, Plot2D

import matplotlib
import matplotlib.colors


_logger = logging.getLogger(__name__)


def plot1d(x_or_y=None, y=None, title='', xlabel='X', ylabel='Y'):
    """Plot curves in a dedicated widget.

    Examples:

    The following examples must run with a Qt QApplication initialized.

    First import :mod:`sx` function:

    >>> from silx import sx
    >>> import numpy

    Plot a single curve given some values:

    >>> values = numpy.random.random(100)
    >>> plot_1curve = sx.plot1d(values, title='Random data')

    Plot a single curve given the x and y values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> sin_a = numpy.sin(angles)
    >>> plot_sinus = sx.plot1d(angles, sin_a,
    ...                        xlabel='angle (radian)', ylabel='sin(a)')

    Plot many curves by giving a 2D array:

    >>> curves = numpy.random.random(10 * 100).reshape(10, 100)
    >>> plot_curves = sx.plot1d(curves)

    Plot many curves sharing the same x values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> values = (numpy.sin(angles), numpy.cos(angles))
    >>> plt = sx.plot1d(angles, values)

    :param x_or_y: x values or y values if y is not provided
    :param y: y values (x_or_y) must be provided
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plot = Plot1D()
    plot.setGraphTitle(title)
    plot.setGraphXLabel(xlabel)
    plot.setGraphYLabel(ylabel)

    # Handle x_or_y and y arguments
    if x_or_y is None and y is not None:
        # Only y is provided, reorder arguments
        x_or_y, y = y, None

    if x_or_y is not None:
        x_or_y = numpy.array(x_or_y, copy=False)

        if y is None:  # x_or_y is y and no x provided, create x values
            y = x_or_y
            x_or_y = numpy.arange(x_or_y.shape[-1], dtype=numpy.float32)

        y = numpy.array(y, copy=False)
        y = y.reshape(-1, y.shape[-1])  # Make it 2D array

        if x_or_y.ndim == 1:
            for index, ycurve in enumerate(y):
                plot.addCurve(x_or_y, ycurve, legend=('curve_%d' % index))

        else:
            # Make x a 2D array as well
            x_or_y = x_or_y.reshape(-1, x_or_y.shape[-1])
            if x_or_y.shape[0] != y.shape[0]:
                raise ValueError(
                    'Not the same dimensions for x and y (%d != %d)' %
                    (x_or_y.shape[0], y.shape[0]))
            for index, (xcurve, ycurve) in enumerate(zip(x_or_y, y)):
                plot.addCurve(xcurve, ycurve, legend=('curve_%d' % index))

    plot.show()
    return plot


def plot2d(data=None, cmap=None, norm='linear',
           vmin=None, vmax=None,
           aspect=False,
           origin=(0., 0.), scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """Plot an image in a dedicated widget.

    Example to plot an image.
    This example must run with a Qt QApplication initialized.

    >>> from silx import sx
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plt = sx.plot2d(data, title='Random data')

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
    plot = Plot2D()
    plot.setGraphTitle(title)
    plot.setGraphXLabel(xlabel)
    plot.setGraphYLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plot.getDefaultColormap()
    if cmap is not None:
        colormap['name'] = cmap
    colormap['normalization'] = norm
    if vmin is not None:
        colormap['vmin'] = vmin
    if vmax is not None:
        colormap['vmax'] = vmax
    if vmin is not None and vmax is not None:
        colormap['autoscale'] = False
    plot.setDefaultColormap(colormap)

    plot.setKeepDataAspectRatio(aspect)

    if data is not None:
        plot.addImage(data, origin=origin, scale=scale)

    plot.show()
    return plot


# matplotlib compatible functions

def plot(*args, **kwargs):
    """matplotlib.pyplot.plot compatible function.

    Some arguments are ignored.

    For documentation, see:
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    """
    raise NotImplementedError()


def imshow(X,
           cmap=None,
           norm=None,
           aspect=None,
           interpolation=None,  # ignored
           alpha=None,  # ignored
           vmin=None,
           vmax=None,
           origin=None,
           extent=None,
           shape=None,
           # ignored:
           # filternorm=None,
           # filterrad=None,
           # imlim=None,
           # resample=None,
           # url=None,
           # hold=None,
           # data=None,
           **kwargs):
    """matplotlib.pyplot.imshow compatible function.

    For documentation, see:
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

    The following arguments are ignored:
    interpolation, alpha, filternorm, filterrad, imlim,
    resample, url, hold, data.
    """

    # Warning for ignored arguments
    if interpolation is not None:
        kwargs['interpolation'] = interpolation
    if alpha is not None:
        kwargs['alpha'] = alpha
    if kwargs:
        _logger.warning(
            'imshow ignores argument%s: %s',
            's' if len(kwargs) > 1 else '',
            ', '.join(arg for arg, val in kwargs.items() if val is not None))

    # Handle input data
    X = numpy.array(X, copy=True)
    if shape is not None:  # Override data shape
        X.shape = shape

    assert X.ndim in (2, 3)  # data or RGB(A)
    if X.ndim == 3:
        assert X.shape[-1] in (3, 4)  # RGB(A) image

    # Create widget
    plot = Plot2D()
    
    # Handle aspect
    if aspect is None:
        aspect = matplotlib.rcParams.get('image.aspect', 'auto')
    if aspect in ('auto', 'normal'):
        plot.setKeepDataAspectRatio(False)
    elif aspect == 'equal' or aspect == 1:
        plot.setKeepDataAspectRatio(True)
    else:
        _logger.warning('imshow: Unhandled aspect argument: %s', str(aspect))

    # convert origin to invert Y axis
    if origin is None:
        origin = matplotlib.rcParams.get('image.origin', 'lower')
    if origin in ('upper', 'lower'):
        plot.setYAxisInverted(origin == 'upper')
    else:
        _logger.warning('imshow: Unhandled origin argument: %s', str(origin))

    # Get colormap
    colormap = plot.getDefaultColormap()
    if cmap is None:
        cmap = matplotlib.rcParams.get('image.cmap', colormap['name'])
    if hasattr(cmap, 'name'):  # Convert colormap object to str
        cmap = cmap.name
    colormap['name'] = cmap
    if vmin is not None:
        colormap['vmin'] = float(vmin)
    if vmax is not None:
        colormap['vmax'] = float(vmax)
    if vmin is not None and vmax is not None:
        colormap['autoscale'] = False
    if norm is not None:  # This can override vmin, vmax and autoscale
        colormap['autoscale'] = norm.scaled()
        if norm.scaled():
            colormap['vmin'] = norm.vmin
            colormap['vmax'] = norm.vmax
        if isinstance(norm, (matplotlib.colors.PowerNorm,
                             matplotlib.colors.SymLogNorm)):
            _logger.warning('Only supports linear and log scale')
        is_log = (norm.startswith('log') or
                  isinstance(norm, matplotlib.colors.LogNorm))
        colorma['norm'] = 'log' if is_log else 'linear'

    # convert extent to origin and scale
    if extent is None:
        img_origin = 0, 0
        img_scale = 1., 1.
    else:
        left, right, bottom, top = extent
        img_origin = left, bottom
        img_scale = (right - left) / X.shape[1], (top - bottom) / X.shape[0]

    # Add image to plot
    plot.addImage(X, colormap=colormap, origin=img_origin, scale=img_scale)

    plot.show()
    return plot
