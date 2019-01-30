# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""This module provides the :class:`Scatter` item of the :class:`Plot`.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "06/06/2018"


from collections import OrderedDict
import logging

import numpy

from .. import items
from ....math.combo import min_max


logger = logging.getLogger(__name__)


class Stats(OrderedDict):
    """Class to define a set of statistic relative to a dataset
    (image, curve...).

    The goal of this class is to avoid multiple recalculation of some
    basic operations such as filtering data area where the statistics has to
    be apply.
    Min and max are also stored because they can be used several time.

    :param List statslist: List of the :class:`Stat` object to be computed.
    """
    def __init__(self, statslist=None):
        OrderedDict.__init__(self)
        _statslist = statslist if not None else []
        if statslist is not None:
            for stat in _statslist:
                self.add(stat)

    def calculate(self, item, plot, onlimits):
        """
        Call all :class:`Stat` object registered and return the result of the
        computation.

        :param item: the item for which we want statistics
        :param plot: plot containing the item
        :param bool onlimits: True if we want to apply statistic only on
                              visible data.
        :return dict: dictionary with :class:`Stat` name as ket and result
                      of the calculation as value
        """
        context = None
        # Check for PlotWidget items
        if isinstance(item, items.Curve):
            context = _CurveContext(item, plot, onlimits)
        elif isinstance(item, items.ImageData):
            context = _ImageContext(item, plot, onlimits)
        elif isinstance(item, items.Scatter):
            context = _ScatterContext(item, plot, onlimits)
        elif isinstance(item, items.Histogram):
            context = _HistogramContext(item, plot, onlimits)
        else:
            # Check for SceneWidget items
            from ...plot3d import items as items3d  # Lazy import

            if isinstance(item, (items3d.Scatter2D, items3d.Scatter3D)):
                context = _plot3DScatterContext(item, plot, onlimits)
            elif isinstance(item, (items3d.ImageData, items3d.ScalarField3D)):
                context = _plot3DArrayContext(item, plot, onlimits)

        if context is None:
                raise ValueError('Item type not managed')

        res = {}
        for statName, stat in list(self.items()):
            if context.kind not in stat.compatibleKinds:
                logger.debug('kind %s not managed by statistic %s'
                             % (context.kind, stat.name))
                res[statName] = None
            else:
                res[statName] = stat.calculate(context)
        return res

    def __setitem__(self, key, value):
        assert isinstance(value, StatBase)
        OrderedDict.__setitem__(self, key, value)

    def add(self, stat):
        self.__setitem__(key=stat.name, value=stat)


class _StatsContext(object):
    """
    The context is designed to be a simple buffer and avoid repetition of
    calculations that can appear during stats evaluation.

    .. warning:: this class gives access to the data to be used for computation
                 . It deal with filtering data visible by the user on plot.
                 The filtering is a simple data sub-sampling. No interpolation
                 is made to fit data to boundaries.

    :param item: the item for which we want to compute the context
    :param str kind: the kind of the item
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, kind, plot, onlimits):
        assert item
        assert plot
        assert type(onlimits) is bool
        self.kind = kind
        self.min = None
        self.max = None
        self.data = None

        self.values = None
        """The array of data"""

        self.axes = None
        """A list of array of position on each axis.

        If the signal is an array,
        then each axis has the length of that dimension,
        and the order is (z, y, x) (i.e., as the array shape).
        If the signal is not an array,
        then each axis has the same length as the signal,
        and the order is (x, y, z).
        """

        self.createContext(item, plot, onlimits)

    def createContext(self, item, plot, onlimits):
        raise NotImplementedError("Base class")

    def isStructuredData(self):
        """Returns True if data as an array-like structure.

        :rtype: bool
        """
        if self.values is None or self.axes is None:
            return False

        if numpy.prod([len(axis) for axis in self.axes]) == self.values.size:
            return True
        else:
            # Make sure there is the right number of value in axes
            for axis in self.axes:
                assert len(axis) == self.values.size
            return False

    def isScalarData(self):
        """Returns True if data is a scalar.

        :rtype: bool
        """
        if self.values is None or self.axes is None:
            return False
        if self.isStructuredData():
            return len(self.axes) == self.values.ndim
        else:
            return self.values.ndim == 1


class _CurveContext(_StatsContext):
    """
    StatsContext for :class:`Curve`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='curve', item=item,
                               plot=plot, onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        xData, yData = item.getData(copy=True)[0:2]

        if onlimits:
            minX, maxX = plot.getXAxis().getLimits()
            mask = (minX <= xData) & (xData <= maxX)
            yData = yData[mask]
            xData = xData[mask]

        self.xData = xData
        self.yData = yData
        if len(yData) > 0:
            self.min, self.max = min_max(yData)
        else:
            self.min, self.max = None, None
        self.data = (xData, yData)
        self.values = yData
        self.axes = (xData,)


class _HistogramContext(_StatsContext):
    """
    StatsContext for :class:`Histogram`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='histogram', item=item,
                               plot=plot, onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        yData, edges = item.getData(copy=True)[0:2]
        xData = item._revertComputeEdges(x=edges, histogramType=item.getAlignment())
        if onlimits:
            minX, maxX = plot.getXAxis().getLimits()
            mask = (minX <= xData) & (xData <= maxX)
            yData = yData[mask]
            xData = xData[mask]

        self.xData = xData
        self.yData = yData
        if len(yData) > 0:
            self.min, self.max = min_max(yData)
        else:
            self.min, self.max = None, None
        self.data = (xData, yData)
        self.values = yData
        self.axes = (xData,)


class _ScatterContext(_StatsContext):
    """StatsContext scatter plots.

    It supports :class:`~silx.gui.plot.items.Scatter`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='scatter', item=item, plot=plot,
                               onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        valueData = item.getValueData(copy=True)
        xData = item.getXData(copy=True)
        yData = item.getYData(copy=True)

        if onlimits:
            minX, maxX = plot.getXAxis().getLimits()
            minY, maxY = plot.getYAxis().getLimits()

            # filter on X axis
            valueData = valueData[(minX <= xData) & (xData <= maxX)]
            yData = yData[(minX <= xData) & (xData <= maxX)]
            xData = xData[(minX <= xData) & (xData <= maxX)]
            # filter on Y axis
            valueData = valueData[(minY <= yData) & (yData <= maxY)]
            xData = xData[(minY <= yData) & (yData <= maxY)]
            yData = yData[(minY <= yData) & (yData <= maxY)]

        if len(valueData) > 0:
            self.min, self.max = min_max(valueData)
        else:
            self.min, self.max = None, None
        self.data = (xData, yData, valueData)
        self.values = valueData
        self.axes = (xData, yData)


class _ImageContext(_StatsContext):
    """StatsContext for images.

    It supports :class:`~silx.gui.plot.items.ImageData`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='image', item=item,
                               plot=plot, onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        self.origin = item.getOrigin()
        self.scale = item.getScale()

        self.data = item.getData(copy=True)

        if onlimits:
            minX, maxX = plot.getXAxis().getLimits()
            minY, maxY = plot.getYAxis().getLimits()

            XMinBound = int((minX - self.origin[0]) / self.scale[0])
            YMinBound = int((minY - self.origin[1]) / self.scale[1])
            XMaxBound = int((maxX - self.origin[0]) / self.scale[0])
            YMaxBound = int((maxY - self.origin[1]) / self.scale[1])

            XMinBound = max(XMinBound, 0)
            YMinBound = max(YMinBound, 0)

            if XMaxBound <= XMinBound or YMaxBound <= YMinBound:
                self.data = None
            else:
                self.data = self.data[YMinBound:YMaxBound + 1,
                                      XMinBound:XMaxBound + 1]
        if self.data.size > 0:
            self.min, self.max = min_max(self.data)
        else:
            self.min, self.max = None, None
        self.values = self.data

        if self.values is not None:
            self.axes = (self.origin[1] + self.scale[1] * numpy.arange(self.data.shape[0]),
                         self.origin[0] + self.scale[0] * numpy.arange(self.data.shape[1]))


class _plot3DScatterContext(_StatsContext):
    """StatsContext for 3D scatter plots.

    It supports :class:`~silx.gui.plot3d.items.Scatter2D` and
    :class:`~silx.gui.plot3d.items.Scatter3D`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='scatter', item=item, plot=plot,
                               onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        if onlimits:
            raise RuntimeError("Unsupported plot %s" % str(plot))

        values = item.getValueData(copy=False)

        if values is not None and len(values) > 0:
            self.values = values
            axes = [item.getXData(copy=False), item.getYData(copy=False)]
            if self.values.ndim == 3:
                axes.append(item.getZData(copy=False))
            self.axes = tuple(axes)

            self.min, self.max = min_max(self.values)
        else:
            self.values = None
            self.axes = None
            self.min, self.max = None, None


class _plot3DArrayContext(_StatsContext):
    """StatsContext for 3D scalar field and data image.

    It supports :class:`~silx.gui.plot3d.items.ScalarField3D` and
    :class:`~silx.gui.plot3d.items.ImageData`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='image', item=item, plot=plot,
                               onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        if onlimits:
            raise RuntimeError("Unsupported plot %s" % str(plot))

        values = item.getData(copy=False)

        if values is not None and len(values) > 0:
            self.values = values
            self.axes = tuple([numpy.arange(size) for size in self.values.shape])
            self.min, self.max = min_max(self.values)
        else:
            self.values = None
            self.axes = None
            self.min, self.max = None, None


BASIC_COMPATIBLE_KINDS = 'curve', 'image', 'scatter', 'histogram'


class StatBase(object):
    """
    Base class for defining a statistic.

    :param str name: the name of the statistic. Must be unique.
    :param List[str] compatibleKinds:
        The kind of items (curve, scatter...) for which the statistic apply.
    """
    def __init__(self, name, compatibleKinds=BASIC_COMPATIBLE_KINDS, description=None):
        self.name = name
        self.compatibleKinds = compatibleKinds
        self.description = description

    def calculate(self, context):
        """
        compute the statistic for the given :class:`StatsContext`

        :param _StatsContext context:
        :return dict: key is stat name, statistic computed is the dict value
        """
        raise NotImplementedError('Base class')

    def getToolTip(self, kind):
        """
        If necessary add a tooltip for a stat kind

        :param str kind: the kind of item the statistic is compute for.
        :return: tooltip or None if no tooltip
        """
        return None


class Stat(StatBase):
    """
    Create a StatBase class based on a function pointer.

    :param str name: name of the statistic. Used as id
    :param fct: function which should have as unique mandatory parameter the
                data. Should be able to adapt to all `kinds` defined as
                compatible
    :param tuple kinds: the compatible item kinds of the function (curve,
                        image...)
    """
    def __init__(self, name, fct, kinds=BASIC_COMPATIBLE_KINDS):
        StatBase.__init__(self, name, kinds)
        self._fct = fct

    def calculate(self, context):
        if context.values is not None:
            if context.kind in self.compatibleKinds:
                return self._fct(context.values)
            else:
                raise ValueError('Kind %s not managed by %s'
                                 '' % (context.kind, self.name))
        else:
            return None


class StatMin(StatBase):
    """Compute the minimal value on data"""
    def __init__(self):
        StatBase.__init__(self, name='min')

    def calculate(self, context):
        return context.min


class StatMax(StatBase):
    """Compute the maximal value on data"""
    def __init__(self):
        StatBase.__init__(self, name='max')

    def calculate(self, context):
        return context.max


class StatDelta(StatBase):
    """Compute the delta between minimal and maximal on data"""
    def __init__(self):
        StatBase.__init__(self, name='delta')

    def calculate(self, context):
        return context.max - context.min


class _StatCoord(StatBase):
    """Base class for argmin and argmax stats"""

    def _indexToCoordinates(self, context, index):
        """Returns the coordinates of data point at given index

        If data is an array, coordinates are in reverse order from data shape.

        :param _StatsContext context:
        :param int index: Index in the flattened data array
        :rtype: List[int]
        """
        if context.isStructuredData():
            coordinates = []
            for axis in reversed(context.axes):
                coordinates.append(axis[index % len(axis)])
                index = index // len(axis)
            return tuple(coordinates)
        else:
            return tuple(axis[index] for axis in context.axes)


class StatCoordMin(_StatCoord):
    """Compute the coordinates of the first minimum value of the data"""
    def __init__(self):
        _StatCoord.__init__(self, name='coords min')

    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        index = numpy.argmin(context.values)
        return self._indexToCoordinates(context, index)

    def getToolTip(self, kind):
        return "Coordinates of the first minimum value of the data"


class StatCoordMax(_StatCoord):
    """Compute the coordinates of the first maximum value of the data"""
    def __init__(self):
        _StatCoord.__init__(self, name='coords max')

    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        index = numpy.argmax(context.values)
        return self._indexToCoordinates(context, index)

    def getToolTip(self, kind):
        return "Coordinates of the first maximum value of the data"


class StatCOM(StatBase):
    """Compute data center of mass"""
    def __init__(self):
        StatBase.__init__(self, name='COM', description='Center of mass')

    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        values = numpy.array(context.values, dtype=numpy.float64)
        sum_ = numpy.sum(values)
        if sum_ == 0.:
            return (numpy.nan,) * len(context.axes)

        if context.isStructuredData():
            centerofmass = []
            for index, axis in enumerate(context.axes):
                axes = tuple([i for i in range(len(context.axes)) if i != index])
                centerofmass.append(
                    numpy.sum(axis * numpy.sum(values, axis=axes)) / sum_)
            return tuple(reversed(centerofmass))
        else:
            return tuple(
                numpy.sum(axis * values) / sum_ for axis in context.axes)

    def getToolTip(self, kind):
        return "Compute the center of mass of the dataset"
