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
"""This module provides the :class:`Scatter` item of the :class:`Plot`.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "09/04/2018"


import numpy
from silx.gui.plot.items.curve import Curve as CurveItem
from silx.gui.plot.items.image import ImageBase as ImageItem
from silx.gui.plot.items.scatter import Scatter as ScatterItem
from silx.gui.plot.items.histogram import Histogram as HistogramItem
from silx.math.combo import min_max
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class Stats(OrderedDict):
    """Class to define a set of statistic relative to a dataset
    (image, curve...).

    The goal of this class is to avoid multiple recalculation of some
    basic operations such as filtering data area where the statistics has to
    be apply.
    Min and max are also stored because they can be used several time.

    :param list: statlist list of the :class:`Stat` object to be computed.
    """
    def __init__(self, statslist=None):
        OrderedDict.__init__(self)
        _statslist = statslist if not None else []
        if statslist is not None:
            for stat in _statslist:
                self.add(stat)

    def calculate(self, item, plot, onlimits):
        """
        Call all :class:`Stat` object registred and return the result of the
        computation.

        :param item: the item for which we want statistics
        :param plot: plot containing the item
        :param bool onlimits: True if we want to apply statistic only on
                              visible data.
        :return dict: dictionary with :class:`Stat` name as ket and result
                      of the calculation as value
        """
        res = {}
        if isinstance(item, CurveItem):
            context = _CurveContext(item, plot, onlimits)
        elif isinstance(item, ImageItem):
            context = _ImageContext(item, plot, onlimits)
        elif isinstance(item, ScatterItem):
            context = _ScatterContext(item, plot, onlimits)
        elif isinstance(item, HistogramItem):
            context = _HistogramContext(item, plot, onlimits)
        else:
            raise ValueError('Item type not managed')
        for statName, stat in list(self.items()):
            if context.kind not in stat.compatibleKinds:
                logger.debug('kind %s not managed by statistic %s'
                               '' % (context.kind, stat.name))
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
        self.createContext(item, plot, onlimits)

    def createContext(self, item, plot, onlimits):
        raise NotImplementedError("Base class")


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
            yData = yData[(minX <= xData) & (xData <= maxX)]
            xData = xData[(minX <= xData) & (xData <= maxX)]

        self.xData = xData
        self.yData = yData
        if len(yData) > 0:
            self.min, self.max = min_max(yData)
        else:
            self.min, self.max = None, None
        self.data = (xData, yData)
        self.values = yData


class _HistogramContext(_StatsContext):
    """
    StatsContext for :class:`Curve`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='histogram', item=item,
                               plot=plot, onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        xData, edges = item.getData(copy=True)[0:2]
        yData = item._revertComputeEdges(x=edges, histogramType=item.getAlignment())
        if onlimits:
            minX, maxX = plot.getXAxis().getLimits()
            yData = yData[(minX <= xData) & (xData <= maxX)]
            xData = xData[(minX <= xData) & (xData <= maxX)]

        self.xData = xData
        self.yData = yData
        if len(yData) > 0:
            self.min, self.max = min_max(yData)
        else:
            self.min, self.max = None, None
        self.data = (xData, yData)
        self.values = yData


class _ScatterContext(_StatsContext):
    """
    StatsContext for :class:`Scatter`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='scatter', item=item, plot=plot,
                               onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        xData, yData, valueData, xerror, yerror = item.getData(copy=True)
        assert plot
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


class _ImageContext(_StatsContext):
    """
    StatsContext for :class:`ImageBase`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    """
    def __init__(self, item, plot, onlimits):
        _StatsContext.__init__(self, kind='image', item=item,
                               plot=plot, onlimits=onlimits)

    def createContext(self, item, plot, onlimits):
        minX, maxX = plot.getXAxis().getLimits()
        minY, maxY = plot.getYAxis().getLimits()
        originX, originY = item.getOrigin()

        XMinBound = int(minX - originX)
        XMaxBound = int(maxX - originX)
        YMinBound = int(minY - originY)
        YMaxBound = int(maxY - originY)

        if XMaxBound < 0 or YMaxBound < 0:
            return self.noDataSelected()
        XMinBound = max(XMinBound, 0)
        YMinBound = max(YMinBound, 0)
        data = item.getData()
        self.data = data[YMinBound:YMaxBound + 1, XMinBound:XMaxBound + 1]
        if len(data) > 0:
            self.min, self.max = min_max(self.data)
        else:
            self.min, self.max = None, None
        self.values = self.data


BASIC_COMPATIBLE_KINDS = {
    'curve': CurveItem,
    'image': ImageItem,
    'scatter': ScatterItem,
    'histogram': HistogramItem,
}


class StatBase(object):
    """
    Base class for defining a statistic.

    :param str name: the name of the statistic. Must be unique.
    :param compatibleKinds: the kind of items (curve, scatter...) for which
                            the statistic apply.
    :rtype: tuple or list
    """
    def __init__(self, name, compatibleKinds=BASIC_COMPATIBLE_KINDS):
        self.name = name
        self.compatibleKinds = compatibleKinds

    def calculate(self, context):
        """
        compute the statistic for the given :class:`StatsContext`

        :param context:
        :return dict: key is stat name, statistic computed is the dict value
        """
        raise NotImplementedError('Base class')


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
        if context.kind in self.compatibleKinds:
            return self._fct(context.values)
        else:
            raise ValueError('Kind %s not managed by %s'
                             '' % (context.kind, self.name))


class StatMin(StatBase):
    """
    Compute the minimal value on data
    """
    def __init__(self):
        StatBase.__init__(self, name='min')

    def calculate(self, context):
        return context.min


class StatMax(StatBase):
    """
    Compute the maximal value on data
    """
    def __init__(self):
        StatBase.__init__(self, name='max')

    def calculate(self, context):
        return context.max


class StatDelta(StatBase):
    """
    Compute the delta between minimal and maximal on data
    """
    def __init__(self):
        StatBase.__init__(self, name='delta')

    def calculate(self, context):
        return context.max - context.min


def _getImgCoordsFor(data, searchValue):
    coordsY, coordsX = numpy.where(data == searchValue)
    if len(coordsX) is 0:
        return []
    if len(coordsX) is 1:
        return (coordsX[0], coordsY[0])
    coords = []
    for xCoord, yCoord in zip(coordsX, coordsY):
        coord = (xCoord, yCoord)
        coords.append(coord)
    return coords


def _getScatterCoordsFor(scatterData, searchValue):
    xData, yData, values = scatterData
    indexes = numpy.where(values == searchValue)[0]
    if len(indexes) is 0:
        return []
    if len(indexes) is 1:
        return (yData[indexes[0]], xData[indexes[0]])
    coords = []
    for index in indexes:
        coords.append((yData[index], xData[index]))
    return coords


class StatCoordMin(StatBase):
    """
    Compute the coordinates of the data minimal value
    """
    def __init__(self):
        StatBase.__init__(self, name='coords min')

    def calculate(self, context):
        if context.kind in ('curve', 'histogram'):
            xData, yData = context.data
            return xData[numpy.where(yData == context.min)]
        elif context.kind == 'scatter':
            return _getScatterCoordsFor(context.data, context.min)
        elif context.kind == 'image':
            return _getImgCoordsFor(context.data, context.min)
        else:
            raise ValueError('kind not managed')


class StatCoordMax(StatBase):
    """
    Compute the coordinates of the data minimal value
    """
    def __init__(self):
        StatBase.__init__(self, name='coords max')

    def calculate(self, context):
        if context.kind in ('curve', 'histogram'):
            xData, yData = context.data
            return xData[numpy.where(yData == context.max)]
        elif context.kind == 'scatter':
            return _getScatterCoordsFor(context.data, context.max)
        elif context.kind == 'image':
            return _getImgCoordsFor(context.data, context.max)
        else:
            raise ValueError('kind not managed')


class StatCOM(StatBase):
    """
    Compute data center of mass
    """
    def __init__(self):
        StatBase.__init__(self, name='COM')

    def calculate(self, context):
        if context.kind in ('curve', 'histogram'):
            xData, yData = context.data
            com = numpy.sum(xData * yData).astype(numpy.float32) / numpy.sum(
                yData).astype(numpy.float32)
            return com
        elif context.kind == 'scatter':
            xData = context.data[0]
            values = context.values
            com = numpy.sum(xData * values).astype(numpy.float32) / numpy.sum(
                values).astype(numpy.float32)
            return com
        elif context.kind == 'image':
            yData = numpy.sum(context.data, axis=1)
            xData = numpy.sum(context.data, axis=0)
            dataXRange = range(context.data.shape[1])
            dataYRange = range(context.data.shape[0])

            ycom = numpy.sum(yData * dataYRange) / numpy.sum(yData)
            xcom = numpy.sum(xData * dataXRange) / numpy.sum(xData)
            return (xcom, ycom)
        else:
            raise ValueError('kind not managed')
