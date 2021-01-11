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


class DataFiltererBase:
    def __init__(self, data_item, roi_item):
        self._data_item = data_item
        self._roi_item = roi_item
        self.__data = None
        self.__mask = self._build_mask()

    @property
    def data_item(self):
        return self._data_item

    @property
    def roi_item(self):
        return self._roi_item

    @property
    def mask(self):
        return self.__mask

    @property
    def data(self):
        return self.__data

    def _build_mask(self):
        raise NotImplemented("Base class")


class CurveFilter(DataFiltererBase):

    def _build_mask(self):
        minX, maxX = self.roi_item.getFrom(), self.roi_item.getTo()
        xData, yData = self.data_item.getData(copy=True)[0:2]

        mask = (minX <= xData) & (xData <= maxX)
        mask = mask == 0
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
        mask = mask == 0
        return mask


class ScatterFilter(DataFiltererBase):
    def _build_mask(self):
        xData = self.data_item.getXData(copy=True)
        if self.roi_item:
            return (xData < self.roi_item.getFrom()) | (xData > self.roi_item.getTo())
        else:
            return numpy.zeros_like(xData)


class ImageFilter(DataFiltererBase):
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
