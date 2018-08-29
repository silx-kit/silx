# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides the :class:`Histogram` item of the :class:`Plot`.
"""

__authors__ = ["H. Payno", "T. Vincent"]
__license__ = "MIT"
__date__ = "28/08/2018"

import logging

import numpy

from .core import (Item, AlphaMixIn, ColorMixIn, FillMixIn,
                   LineMixIn, YAxisMixIn, ItemChangedType)

_logger = logging.getLogger(__name__)


def _computeEdges(x, histogramType):
    """Compute the edges from a set of xs and a rule to generate the edges

    :param x: the x value of the curve to transform into an histogram
    :param histogramType: the type of histogram we wan't to generate.
         This define the way to center the histogram values compared to the
         curve value. Possible values can be::

         - 'left'
         - 'right'
         - 'center'

    :return: the edges for the given x and the histogramType
    """
    # for now we consider that the spaces between xs are constant
    edges = x.copy()
    if histogramType is 'left':
        width = 1
        if len(x) > 1:
            width = x[1] - x[0]
        edges = numpy.append(x[0] - width, edges)
    if histogramType is 'center':
        edges = _computeEdges(edges, 'right')
        widths = (edges[1:] - edges[0:-1]) / 2.0
        widths = numpy.append(widths, widths[-1])
        edges = edges - widths
    if histogramType is 'right':
        width = 1
        if len(x) > 1:
            width = x[-1] - x[-2]
        edges = numpy.append(edges, x[-1] + width)

    return edges


def _getHistogramCurve(histogram, edges):
    """Returns the x and y value of a curve corresponding to the histogram

    :param numpy.ndarray histogram: The values of the histogram
    :param numpy.ndarray edges: The bin edges of the histogram
    :return: a tuple(x, y) which contains the value of the curve to use
             to display the histogram
    """
    assert len(histogram) + 1 == len(edges)
    x = numpy.empty(len(histogram) * 2, dtype=edges.dtype)
    y = numpy.empty(len(histogram) * 2, dtype=histogram.dtype)
    # Make a curve with stairs
    x[:-1:2] = edges[:-1]
    x[1::2] = edges[1:]
    y[:-1:2] = histogram
    y[1::2] = histogram

    return x, y


# TODO: Yerror, test log scale
class Histogram(Item, AlphaMixIn, ColorMixIn, FillMixIn,
                LineMixIn, YAxisMixIn):
    """Description of an histogram"""

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for histograms"""

    _DEFAULT_SELECTABLE = False
    """Default selectable state for histograms"""

    _DEFAULT_LINEWIDTH = 1.
    """Default line width of the histogram"""

    _DEFAULT_LINESTYLE = '-'
    """Default line style of the histogram"""

    def __init__(self):
        Item.__init__(self)
        AlphaMixIn.__init__(self)
        ColorMixIn.__init__(self)
        FillMixIn.__init__(self)
        LineMixIn.__init__(self)
        YAxisMixIn.__init__(self)

        self._histogram = ()
        self._edges = ()

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        values, edges = self.getData(copy=False)

        if values.size == 0:
            return None  # No data to display, do not add renderer

        if values.size == 0:
            return None  # No data to display, do not add renderer to backend

        x, y = _getHistogramCurve(values, edges)

        # Filter-out values <= 0
        plot = self.getPlot()
        if plot is not None:
            xPositive = plot.getXAxis()._isLogarithmic()
            yPositive = plot.getYAxis()._isLogarithmic()
        else:
            xPositive = False
            yPositive = False

        if xPositive or yPositive:
            clipped = numpy.logical_or(
                (x <= 0) if xPositive else False,
                (y <= 0) if yPositive else False)
            # Make a copy and replace negative points by NaN
            x = numpy.array(x, dtype=numpy.float)
            y = numpy.array(y, dtype=numpy.float)
            x[clipped] = numpy.nan
            y[clipped] = numpy.nan

        return backend.addCurve(x, y, self.getLegend(),
                                color=self.getColor(),
                                symbol='',
                                linestyle=self.getLineStyle(),
                                linewidth=self.getLineWidth(),
                                yaxis=self.getYAxis(),
                                xerror=None,
                                yerror=None,
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                fill=self.isFill(),
                                alpha=self.getAlpha(),
                                symbolsize=1)

    def _getBounds(self):
        values, edges = self.getData(copy=False)

        plot = self.getPlot()
        if plot is not None:
            xPositive = plot.getXAxis()._isLogarithmic()
            yPositive = plot.getYAxis()._isLogarithmic()
        else:
            xPositive = False
            yPositive = False

        if xPositive or yPositive:
            values = numpy.array(values, copy=True, dtype=numpy.float)

            if xPositive:
                # Replace edges <= 0 by NaN and corresponding values by NaN
                clipped_edges = (edges <= 0)
                edges = numpy.array(edges, copy=True, dtype=numpy.float)
                edges[clipped_edges] = numpy.nan
                clipped_values = numpy.logical_or(clipped_edges[:-1],
                                                  clipped_edges[1:])
            else:
                clipped_values = numpy.zeros_like(values, dtype=numpy.bool)

            if yPositive:
                # Replace values <= 0 by NaN, do not modify edges
                clipped_values = numpy.logical_or(clipped_values, values <= 0)

            values[clipped_values] = numpy.nan

        if xPositive or yPositive:
            return (numpy.nanmin(edges),
                    numpy.nanmax(edges),
                    numpy.nanmin(values),
                    numpy.nanmax(values))

        else:  # No log scale, include 0 in bounds
            return (numpy.nanmin(edges),
                    numpy.nanmax(edges),
                    min(0, numpy.nanmin(values)),
                    max(0, numpy.nanmax(values)))

    def setVisible(self, visible):
        """Set visibility of item.

        :param bool visible: True to display it, False otherwise
        """
        visible = bool(visible)
        # TODO hackish data range implementation
        if self.isVisible() != visible:
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()
        super(Histogram, self).setVisible(visible)

    def getValueData(self, copy=True):
        """The values of the histogram

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: The bin edges of the histogram
        :rtype: numpy.ndarray
        """
        return numpy.array(self._histogram, copy=copy)

    def getBinEdgesData(self, copy=True):
        """The bin edges of the histogram (number of histogram values + 1)

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: The bin edges of the histogram
        :rtype: numpy.ndarray
        """
        return numpy.array(self._edges, copy=copy)

    def getData(self, copy=True):
        """Return the histogram values and the bin edges

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: (N histogram value, N+1 bin edges)
        :rtype: 2-tuple of numpy.nadarray
        """
        return self.getValueData(copy), self.getBinEdgesData(copy)

    def setData(self, histogram, edges, align='center', copy=True):
        """Set the histogram values and bin edges.

        :param numpy.ndarray histogram: The values of the histogram.
        :param numpy.ndarray edges:
            The bin edges of the histogram.
            If histogram and edges have the same length, the bin edges
            are computed according to the align parameter.
        :param str align:
            In case histogram values and edges have the same length N,
            the N+1 bin edges are computed according to the alignment in:
            'center' (default), 'left', 'right'.
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        histogram = numpy.array(histogram, copy=copy)
        edges = numpy.array(edges, copy=copy)

        assert histogram.ndim == 1
        assert edges.ndim == 1
        assert edges.size in (histogram.size, histogram.size + 1)
        assert align in ('center', 'left', 'right')

        if histogram.size == 0:  # No data
            self._histogram = ()
            self._edges = ()
        else:
            if edges.size == histogram.size:  # Compute true bin edges
                edges = _computeEdges(edges, align)

            # Check that bin edges are monotonic
            edgesDiff = numpy.diff(edges)
            assert numpy.all(edgesDiff >= 0) or numpy.all(edgesDiff <= 0)

            self._histogram = histogram
            self._edges = edges
            self._alignement = align

        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        self._updated(ItemChangedType.DATA)

    def getAlignment(self):
        """

        :return: histogram alignement. Value in ('center', 'left', 'right').
        """
        return self._alignement

    def _revertComputeEdges(self, x, histogramType):
        """Compute the edges from a set of xs and a rule to generate the edges

        :param x: the x value of the curve to transform into an histogram
        :param histogramType: the type of histogram we wan't to generate.
             This define the way to center the histogram values compared to the
             curve value. Possible values can be::

             - 'left'
             - 'right'
             - 'center'

        :return: the edges for the given x and the histogramType
        """
        # for now we consider that the spaces between xs are constant
        edges = x.copy()
        if histogramType is 'left':
            return edges[1:]
        if histogramType is 'center':
            edges = (edges[1:] + edges[:-1]) / 2.0
        if histogramType is 'right':
            width = 1
            if len(x) > 1:
                width = x[-1] + x[-2]
            edges = edges[:-1]
        return edges
