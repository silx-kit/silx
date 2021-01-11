# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides ROI interaction for :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "08/01/2020"


import numpy
from ..items.roi import _RegionOfInterestBase
from ..items.core import DataItem


class DataFiltererBase:
    """Base class for filtering data from an instance of a `DataItem` with
    an instance of a `RegionOfInterest`
    """
    def __init__(self, data_item, roi_item):
        if not isinstance(roi_item, _RegionOfInterestBase):
            raise TypeError("roi_item should be an instance of `_RegionOfInterestBase`")
        if not isinstance(data_item, DataItem):
            raise TypeError("data_item should be an instance of `DataItem`")
        self._data_item = data_item
        self._roi_item = roi_item
        self.__mask = None

    @property
    def data_item(self):
        return self._data_item

    @property
    def roi_item(self):
        return self._roi_item

    @property
    def mask(self):
        if self.__mask is None:
            self.__mask = self._build_mask()
        return self.__mask

    def _build_mask(self):
        raise NotImplemented("Base class")


class CurveFilter(DataFiltererBase):
    """
    Filter `Curve` item
    """

    def _build_mask(self):
        minX, maxX = self.roi_item.getFrom(), self.roi_item.getTo()
        xData, yData = self.data_item.getData(copy=True)[0:2]

        mask = (minX <= xData) & (xData <= maxX)
        return mask


class HistogramFilter(DataFiltererBase):
    """
    Filter `Histogram` item
    """
    def _build_mask(self):
        yData, edges = self.data_item.getData(copy=True)[0:2]
        xData = self.data_item._revertComputeEdges(x=edges,
                                                   histogramType=self.data_item.getAlignment())
        mask = (self.roi_item._fromdata <= xData) & (xData <= self.roi_item._todata)
        return mask


class ScatterFilter(DataFiltererBase):
    """
    Filter `Scatter` item
    """

    def _build_mask(self):
        xData = self.data_item.getXData(copy=True)
        from ..CurvesROIWidget import ROI
        if self.roi_item:
            if isinstance(self.roi_item, ROI):
                return (xData < self.roi_item.getFrom()) | (xData > self.roi_item.getTo())
            else:
                mask = numpy.zeros_like(self.data_item.getValueData())
                yData = self.data_item.getYData(copy=False)
                for i_value, (x, y) in enumerate(zip(xData, yData)):
                    mask[i_value] = self.roi_item.contains((x, y))
                return mask
        else:
            return numpy.zeros_like(xData)


class ImageFilter(DataFiltererBase):
    """
    Filter `Image` item
    """

    def _build_mask(self):
        data = self.data_item.getData()
        minX, maxX = 0, data.shape[1]
        minY, maxY = 0, data.shape[0]

        XMinBound = max(minX, 0)
        YMinBound = max(minY, 0)
        XMaxBound = min(maxX, data.shape[1])
        YMaxBound = min(maxY, data.shape[0])

        mask = numpy.zeros_like(data)
        for x in range(XMinBound, XMaxBound):
            for y in range(YMinBound, YMaxBound):
                _x = (x * self.data_item.getScale()[0]) + self.data_item.getOrigin()[0]
                _y = (y * self.data_item.getScale()[1]) + self.data_item.getOrigin()[1]
                mask[y, x] = not self.roi_item.contains((_x, _y))
        return mask
