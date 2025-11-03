# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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
"""This module provides mechanism relative to stats calculation within a
:class:`PlotWidget`.
It also include the implementation of the statistics themselves.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "06/06/2018"


from functools import lru_cache
import logging

import numpy
import numpy.ma

from .. import items
from ..CurvesROIWidget import ROI
from ..items.roi import RegionOfInterest
from ....math.combo import min_max
from ....utils.proxy import docstring

logger = logging.getLogger(__name__)


class Stats(dict):
    """Class to define a set of statistic relative to a dataset
    (image, curve...).

    The goal of this class is to avoid multiple recalculation of some
    basic operations such as filtering data area where the statistics has to
    be apply.
    Min and max are also stored because they can be used several time.

    :param List statslist: List of the :class:`Stat` object to be computed.
    """

    def __init__(self, statslist=None):
        super().__init__()
        _statslist = statslist if not None else []
        if statslist is not None:
            for stat in _statslist:
                self.add(stat)

    def calculate(
        self, item, plot, onlimits, roi, data_changed=False, roi_changed=False
    ):
        """
        Call all :class:`Stat` object registered and return the result of the
        computation.

        :param item: the item for which we want statistics
        :param plot: plot containing the item
        :param bool onlimits: True if we want to apply statistic only on
                              visible data.
        :param roi: region of interest for statistic calculation. Incompatible
                    with the `onlimits` option.
        :type roi: Union[None, :class:`~_RegionOfInterestBase`]
        :param bool data_changed: did the data changed since last calculation.
        :param bool roi_changed: did the associated roi (if any) has changed
                                 since last calculation.
        :return dict: dictionary with :class:`Stat` name as ket and result
                      of the calculation as value
        """
        res = {}
        context = self._getContext(item=item, plot=plot, onlimits=onlimits, roi=roi)
        for statName, stat in list(self.items()):
            if context.kind not in stat.compatibleKinds:
                logger.debug(
                    f"kind {context.kind} not managed by statistic {stat.name}"
                )
                res[statName] = None
            else:
                if roi_changed is True:
                    context.clear_mask()
                if data_changed is True or roi_changed is True:
                    # if data changed or mask changed
                    context.clipData(item=item, plot=plot, onlimits=onlimits, roi=roi)
                # init roi and data
                res[statName] = stat.calculate(context)
        return res

    def __setitem__(self, key, value):
        assert isinstance(value, StatBase)
        super().__setitem__(key, value)

    def add(self, stat):
        """Add a :class:`Stat` to the set

        :param Stat stat: stat to add to the set
        """
        self.__setitem__(key=stat.name, value=stat)

    @staticmethod
    @lru_cache(maxsize=50)
    def _getContext(item, plot, onlimits, roi):
        context = None
        # Check for PlotWidget items
        if isinstance(item, items.Curve):
            context = _CurveContext(item, plot, onlimits, roi=roi)
        elif isinstance(item, items.ImageData):
            context = _ImageContext(item, plot, onlimits, roi=roi)
        elif isinstance(item, items.Scatter):
            context = _ScatterContext(item, plot, onlimits, roi=roi)
        elif isinstance(item, items.Histogram):
            context = _HistogramContext(item, plot, onlimits, roi=roi)
        else:
            # Check for SceneWidget items
            from ...plot3d import items as items3d  # Lazy import

            if isinstance(item, (items3d.Scatter2D, items3d.Scatter3D)):
                context = _plot3DScatterContext(item, plot, onlimits, roi=roi)
            elif isinstance(item, (items3d.ImageData, items3d.ScalarField3D)):
                context = _plot3DArrayContext(item, plot, onlimits, roi=roi)
        if context is None:
            raise ValueError("Item type not managed")
        return context


class _StatsContext:
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
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlimits` calculation
    :type roi: Union[None,:class:`_RegionOfInterestBase`]
    """

    def __init__(self, item, kind, plot, onlimits, roi):
        assert item
        assert plot
        assert type(onlimits) is bool
        self.kind = kind
        self.min = None
        self.max = None
        self.data = None
        self.roi = None
        self.onlimits = onlimits

        self.values = None
        """The array of data with limit filtering if any. Is a numpy.ma.array,
        meaning that it embed the mask applied by the roi if any"""

        self.axes = None
        """A list of array of position on each axis.

        If the signal is an array,
        then each axis has the length of that dimension,
        and the order is (z, y, x) (i.e., as the array shape).
        If the signal is not an array,
        then each axis has the same length as the signal,
        and the order is (x, y, z).
        """

        self.clipData(item, plot, onlimits, roi=roi)

    def clear_mask(self):
        """
        Remove the mask to force recomputation of it on next iteration
        :return:
        """
        raise NotImplementedError()

    @property
    def mask(self):
        if self.values is not None:
            assert isinstance(self.values, numpy.ma.MaskedArray)
            return self.values.mask
        else:
            return None

    @property
    def is_mask_valid(self, **kwargs):
        """Return if the mask is valid for the data or need to be recomputed"""
        raise NotImplementedError("Base class")

    def _set_mask_validity(self, **kwargs):
        """User to set some values that allows to define the mask properties
        and boundaries"""
        raise NotImplementedError("Base class")

    def clipData(self, item, plot, onlimits, roi):
        """Clip the data to the current mask to have accurate statistics

        Function called before computing each statistics associated to this
        context. It will insure the context for the (item, plot, onlimits, roi)
        is created.

        :param item: item for which we want statistics
        :param plot: plot containing the statistics
        :param bool onlimits: True if we want to apply statistic only on
                         visible data.
        :param roi: Region of interest for computing the statistics.
                    For now, incompatible with `onlimits` calculation
        :type roi: Union[None,:class:`_RegionOfInterestBase`]
        """
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

    def _checkContextInputs(self, item, plot, onlimits, roi):
        if roi is not None and onlimits is True:
            raise ValueError(
                "Stats context is unable to manage both a ROI"
                "and the `onlimits` option"
            )


class _ScatterCurveHistoMixInContext(_StatsContext):
    def __init__(self, kind, item, plot, onlimits, roi):
        self.clear_mask()
        _StatsContext.__init__(
            self, item=item, kind=kind, plot=plot, onlimits=onlimits, roi=roi
        )

    def _set_mask_validity(self, onlimits, from_, to_):
        self._onlimits = onlimits
        self._from_ = from_
        self._to_ = to_

    def clear_mask(self):
        self._onlimits = None
        self._from_ = None
        self._to_ = None

    def is_mask_valid(self, onlimits, from_, to_):
        return onlimits == self.onlimits and from_ == self._from_ and to_ == self._to_


class _CurveContext(_ScatterCurveHistoMixInContext):
    """
    StatsContext for :class:`Curve`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        _ScatterCurveHistoMixInContext.__init__(
            self, kind="curve", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    @docstring(_StatsContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        self.roi = roi
        self.onlimits = onlimits
        xData, yData = item.getData(copy=True)[0:2]

        # Determine mask
        if onlimits:
            # Mask defined by viewport
            minX, maxX = plot.getXAxis().getLimits()
        elif roi:
            # Mask defined by ROI
            minX, maxX = roi.getFrom(), roi.getTo()
        else:
            # No mask
            minX, maxX = None, None

        if minX is not None and maxX is not None:
            if self.is_mask_valid(onlimits=onlimits, from_=minX, to_=maxX):
                mask = self.mask
            else:
                mask = ~((minX <= xData) & (xData <= maxX))
                self._set_mask_validity(onlimits=onlimits, from_=minX, to_=maxX)
        else:
            mask = numpy.zeros_like(yData, dtype=bool)

        # Apply mask
        self.xData = xData
        self.yData = yData
        self.values = numpy.ma.array(yData, mask=mask)
        self.min, self.max = _minmax_masked_array(self.values)
        self.data = (xData, yData)
        self.axes = (xData,)

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )
        if roi is not None and not isinstance(roi, ROI):
            raise TypeError("curve `context` can ony manage 1D roi")


class _HistogramContext(_ScatterCurveHistoMixInContext):
    """
    StatsContext for :class:`Histogram`

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        _ScatterCurveHistoMixInContext.__init__(
            self, kind="histogram", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    @docstring(_StatsContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        yData, edges = item.getData(copy=True)[0:2]
        xData = item._revertComputeEdges(x=edges, histogramType=item.getAlignment())

        # Determine mask
        if onlimits:
            # Mask defined by viewport
            minX, maxX = plot.getXAxis().getLimits()
        elif roi:
            # Mask defined by ROI
            minX, maxX = roi._fromdata, roi._todata
        else:
            # No mask
            minX, maxX = None, None

        if minX is not None and maxX is not None:
            if self.is_mask_valid(onlimits=onlimits, from_=minX, to_=maxX):
                mask = self.mask
            else:
                mask = ~((minX <= xData) & (xData <= maxX))
                self._set_mask_validity(onlimits=onlimits, from_=minX, to_=maxX)
        else:
            mask = numpy.zeros_like(yData, dtype=bool)

        # Apply mask
        self.xData = xData
        self.yData = yData
        self.values = numpy.ma.array(yData, mask=mask)
        self.min, self.max = _minmax_masked_array(self.values)
        self.data = (xData, yData)
        self.axes = (xData,)

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )

        if roi is not None and not isinstance(roi, ROI):
            raise TypeError("curve `context` can ony manage 1D roi")


class _ScatterContext(_ScatterCurveHistoMixInContext):
    """StatsContext scatter plots.

    It supports :class:`~silx.gui.plot.items.Scatter`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        _ScatterCurveHistoMixInContext.__init__(
            self, kind="scatter", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    @docstring(_ScatterCurveHistoMixInContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        valueData = item.getValueData(copy=True)
        xData = item.getXData(copy=True)
        yData = item.getYData(copy=True)

        # Determine mask
        if onlimits:
            # Mask defined by viewport
            minX, maxX = plot.getXAxis().getLimits()
            minY, maxY = plot.getYAxis().getLimits()

            mask = (xData >= minX) & (xData <= maxX) & (yData >= minY) & (yData <= maxY)

            # Apply and reset mask
            valueData = valueData[mask]
            xData = xData[mask]
            yData = yData[mask]
            mask = numpy.zeros_like(xData, dtype=bool)
        elif roi:
            # Mask defined by ROI
            if self.is_mask_valid(
                onlimits=onlimits, from_=roi.getFrom(), to_=roi.getTo()
            ):
                mask = self.mask
            else:
                mask = (xData < roi.getFrom()) | (xData > roi.getTo())
        else:
            # No mask
            mask = numpy.zeros_like(xData, dtype=bool)

        # Apply mask
        self.data = (xData, yData, valueData)
        self.values = numpy.ma.array(valueData, mask=mask)
        self.min, self.max = _minmax_masked_array(self.values)
        self.axes = (xData, yData)

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )

        if roi is not None and not isinstance(roi, ROI):
            raise TypeError("curve `context` can ony manage 1D roi")


class _ImageContext(_StatsContext):
    """StatsContext for images.

    It supports :class:`~silx.gui.plot.items.ImageData`.

    :warning: behaviour of scale images: now the statistics are computed on
              the entire data array (there is no sampling in the array or
              interpolation regarding the scale).
              This also mean that the result can differ from what is displayed.
              But I guess there is no perfect behaviour.

    :warning: `isIn` functions for image context: for now have basically a
              binary approach, the pixel is in a roi or not. To have a fully
              'correct behaviour' we should add a weight on stats calculation
              to moderate the pixel value.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        self.clear_mask()
        _StatsContext.__init__(
            self, kind="image", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    def _set_mask_validity(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self._mask_x_min = xmin
        self._mask_x_max = xmax
        self._mask_y_min = ymin
        self._mask_y_max = ymax

    def clear_mask(self):
        self._mask_x_min = None
        self._mask_x_max = None
        self._mask_y_min = None
        self._mask_y_max = None

    def is_mask_valid(self, xmin, xmax, ymin, ymax):
        return (
            xmin == self._mask_x_min
            and xmax == self._mask_x_max
            and ymin == self._mask_y_min
            and ymax == self._mask_y_max
        )

    @docstring(_StatsContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        self.origin = item.getOrigin()
        self.scale = item.getScale()
        self.data = item.getData(copy=True)

        xsize = self.data.shape[1]
        ysize = self.data.shape[0]
        xaxis = self.origin[0] + self.scale[0] * numpy.arange(xsize)
        yaxis = self.origin[1] + self.scale[1] * numpy.arange(ysize)

        # Determine mask
        if onlimits:
            # Mask defined by viewport

            # Coordinates
            minX, maxX = plot.getXAxis().getLimits()
            minY, maxY = plot.getYAxis().getLimits()

            # Indices
            xmin = int((minX - self.origin[0]) / self.scale[0])
            ymin = int((minY - self.origin[1]) / self.scale[1])
            xmax = int((maxX - self.origin[0]) / self.scale[0])
            ymax = int((maxY - self.origin[1]) / self.scale[1])

            xmin, xmax = numpy.clip([xmin, xmax], 0, xsize - 1)
            ymin, ymax = numpy.clip([ymin, ymax], 0, ysize - 1)

            if xmax <= xmin or ymax <= ymin:
                self.data = None
                self.values = None
                self.min, self.max, self.axes = None, None, None
                return

            mask = numpy.ones_like(self.data, dtype=bool)
            mask[ymin : ymax + 1, xmin : xmax + 1] = False
        elif roi:
            # Mask defined by ROI
            xmin, xmax = 0, xsize
            ymin, ymax = 0, ysize

            if self.is_mask_valid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
                mask = self.mask
            else:
                xpix, ypix = numpy.meshgrid(xaxis, yaxis)

                points = numpy.column_stack([xpix.ravel(), ypix.ravel()])
                mask = ~roi.contains(points).reshape(ypix.shape)

                self._set_mask_validity(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        else:
            # No mask
            mask = numpy.zeros_like(self.data, dtype=bool)

        # Apply mask
        self.values = numpy.ma.array(self.data, mask=mask)
        self.min, self.max = _minmax_masked_array(self.values)
        self.axes = yaxis, xaxis

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )

        if roi is not None and not isinstance(roi, RegionOfInterest):
            raise TypeError("curve `context` can ony manage 2D roi")


class _plot3DScatterContext(_StatsContext):
    """StatsContext for 3D scatter plots.

    It supports :class:`~silx.gui.plot3d.items.Scatter2D` and
    :class:`~silx.gui.plot3d.items.Scatter3D`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        _StatsContext.__init__(
            self, kind="scatter", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    @docstring(_StatsContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        values = item.getValueData(copy=False)
        if values is None or len(values) == 0:
            self.values = None
            self.min, self.max = None, None
            self.axes = None
            return

        if onlimits:
            # Mask defined by viewport
            raise RuntimeError("Unsupported plot %s" % str(plot))
        elif roi:
            # Mask defined by ROI
            logger.warning("ROI are unsupported on volume for now")
            mask = numpy.zeros_like(values, dtype=bool)
        else:
            # No mask
            mask = numpy.zeros_like(values, dtype=bool)

        # Apply mask on values
        self.min, self.max = min_max(values)
        self.values = numpy.ma.array(values, mask=mask)

        # Apply mask on axes
        axes = [item.getXData(copy=False), item.getYData(copy=False)]
        if self.values.ndim == 3:
            axes.append(item.getZData(copy=False))
        self.axes = tuple(axes)

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )

        if roi is not None and not isinstance(roi, RegionOfInterest):
            raise TypeError("curve `context` can ony manage 2D roi")


class _plot3DArrayContext(_StatsContext):
    """StatsContext for 3D scalar field and data image.

    It supports :class:`~silx.gui.plot3d.items.ScalarField3D` and
    :class:`~silx.gui.plot3d.items.ImageData`.

    :param item: the item for which we want to compute the context
    :param plot: the plot containing the item
    :param bool onlimits: True if we want to apply statistic only on
                          visible data.
    :param roi: Region of interest for computing the statistics.
                For now, incompatible with `onlinits` calculation
    :type roi: Union[None, :class:`ROI`]
    """

    def __init__(self, item, plot, onlimits, roi):
        _StatsContext.__init__(
            self, kind="image", item=item, plot=plot, onlimits=onlimits, roi=roi
        )

    @docstring(_StatsContext)
    def clipData(self, item, plot, onlimits, roi):
        self._checkContextInputs(item=item, plot=plot, onlimits=onlimits, roi=roi)

        values = item.getData(copy=False)
        if values is None or len(values) == 0:
            self.values = None
            self.min, self.max = None, None
            self.axes = None
            return

        if onlimits:
            # Mask defined by viewport
            raise RuntimeError("Unsupported plot %s" % str(plot))
        elif roi:
            # Mask defined by ROI
            logger.warning("ROI are unsupported on volume for now")
            mask = numpy.zeros_like(values, dtype=bool)
        else:
            # No mask
            mask = numpy.zeros_like(values, dtype=bool)

        self.min, self.max = min_max(values)
        self.values = numpy.ma.array(values, mask=mask)
        self.axes = tuple([numpy.arange(size) for size in self.values.shape])

    def _checkContextInputs(self, item, plot, onlimits, roi):
        _StatsContext._checkContextInputs(
            self, item=item, plot=plot, onlimits=onlimits, roi=roi
        )

        if roi is not None and not isinstance(roi, RegionOfInterest):
            raise TypeError("curve `context` can ony manage 2D roi")


BASIC_COMPATIBLE_KINDS = "curve", "image", "scatter", "histogram"


class StatBase:
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
        raise NotImplementedError("Base class")

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

    @docstring(StatBase)
    def calculate(self, context):
        if context.values is not None:
            if context.kind in self.compatibleKinds:
                return self._fct(context.values)
            else:
                raise ValueError(
                    "Kind %s not managed by %s" "" % (context.kind, self.name)
                )
        else:
            return None


class StatMin(StatBase):
    """Compute the minimal value on data"""

    def __init__(self):
        StatBase.__init__(self, name="min")

    @docstring(StatBase)
    def calculate(self, context):
        return context.min


class StatMax(StatBase):
    """Compute the maximal value on data"""

    def __init__(self):
        StatBase.__init__(self, name="max")

    @docstring(StatBase)
    def calculate(self, context):
        return context.max


class StatDelta(StatBase):
    """Compute the delta between minimal and maximal on data"""

    def __init__(self):
        StatBase.__init__(self, name="delta")

    @docstring(StatBase)
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

        axes = context.axes

        if context.isStructuredData() or context.roi:
            coordinates = []
            for axis in reversed(axes):
                coordinates.append(axis[index % len(axis)])
                index = index // len(axis)
            return tuple(coordinates)
        else:
            return tuple(axis[index] for axis in axes)


class StatCoordMin(_StatCoord):
    """Compute the coordinates of the first minimum value of the data"""

    def __init__(self):
        _StatCoord.__init__(self, name="coords min")

    @docstring(StatBase)
    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        index = context.values.argmin()
        return self._indexToCoordinates(context, index)

    @docstring(StatBase)
    def getToolTip(self, kind):
        return "Coordinates of the first minimum value of the data"


class StatCoordMax(_StatCoord):
    """Compute the coordinates of the first maximum value of the data"""

    def __init__(self):
        _StatCoord.__init__(self, name="coords max")

    @docstring(StatBase)
    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        # TODO: the values should be a mask array by default, will be simpler
        # if possible
        index = context.values.argmax()
        return self._indexToCoordinates(context, index)

    @docstring(StatBase)
    def getToolTip(self, kind):
        return "Coordinates of the first maximum value of the data"


class StatCOM(StatBase):
    """Compute data center of mass"""

    def __init__(self):
        StatBase.__init__(self, name="COM", description="Center of mass")

    @docstring(StatBase)
    def calculate(self, context):
        if context.values is None or not context.isScalarData():
            return None

        values = numpy.ma.array(context.values, mask=context.mask, dtype=numpy.float64)
        sum_ = numpy.sum(values)
        if sum_ == 0.0 or numpy.ma.is_masked(sum_):
            return (numpy.nan,) * len(context.axes)

        if context.isStructuredData():
            centerofmass = []
            for index, axis in enumerate(context.axes):
                axes = tuple([i for i in range(len(context.axes)) if i != index])
                centerofmass.append(
                    numpy.sum(axis * numpy.sum(values, axis=axes)) / sum_
                )
            return tuple(reversed(centerofmass))
        else:
            return tuple(numpy.sum(axis * values) / sum_ for axis in context.axes)

    @docstring(StatBase)
    def getToolTip(self, kind):
        return "Compute the center of mass of the dataset"


def _minmax_masked_array(
    masked: numpy.ma.MaskedArray,
) -> tuple[float, float] | tuple[None, None]:
    non_masked_values = masked.compressed()
    if non_masked_values.size > 0:
        return min_max(non_masked_values)
    return None, None
