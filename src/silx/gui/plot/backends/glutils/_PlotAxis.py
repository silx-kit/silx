import math
from typing import Sequence

import datetime as dt
import numpy
import logging
import weakref

from .... import qt
from ....utils.matplotlib import DefaultTickFormatter
from .GLText import Text2D, CENTER
from ..._utils import axis_scale
from ..._utils.ticklayout import niceNumbersAdaptative, niceNumbersForLog10
from ..._utils.dtime_ticklayout import (
    DtUnit,
    bestUnit,
    calcTicksAdaptive,
    formatDatetimes,
)
from ...items.types import AxisScaleType

_logger = logging.getLogger(__name__)


class PlotAxis:
    """Represents a 1D axis of the plot.
    This class is intended to be used with :class:`GLPlotFrame`.
    """

    def __init__(
        self,
        plotFrame,
        tickLength=(0.0, 0.0),
        foregroundColor=(0.0, 0.0, 0.0, 1.0),
        labelAlign=CENTER,
        labelVAlign=CENTER,
        titleAlign=CENTER,
        titleVAlign=CENTER,
        orderOffsetAlign=CENTER,
        orderOffsetVAlign=CENTER,
        titleRotate=0,
        titleOffset=(0.0, 0.0),
        font: qt.QFont | None = None,
    ):
        self._tickFormatter = DefaultTickFormatter()
        self._ticks = None
        self._orderAndOffsetText = ""

        self._plotFrameRef = weakref.ref(plotFrame)

        self._isDateTime = False
        self._timeZone = None
        self._scale: AxisScaleType = "linear"
        self._dataRange = 1.0, 100.0
        self._displayCoords = (0.0, 0.0), (1.0, 0.0)
        self._title = ""

        self._tickLength = tickLength
        self._foregroundColor = foregroundColor
        self._labelAlign = labelAlign
        self._labelVAlign = labelVAlign
        self._orderOffsetAnchor = (1.0, 0.0)
        self._orderOffsetAlign = orderOffsetAlign
        self._orderOffsetVAlign = orderOffsetVAlign
        self._titleAlign = titleAlign
        self._titleVAlign = titleVAlign
        self._titleRotate = titleRotate
        self._titleOffset = titleOffset
        self._font = font

    @property
    def dataRange(self):
        """The range of the data represented on the axis as a tuple
        of 2 floats: (min, max)."""
        return self._dataRange

    @property
    def font(self) -> qt.QFont:
        if self._font is None:
            return qt.QApplication.instance().font()
        return self._font

    @dataRange.setter
    def dataRange(self, dataRange):
        assert len(dataRange) == 2
        assert dataRange[0] <= dataRange[1]
        dataRange = float(dataRange[0]), float(dataRange[1])

        if dataRange != self._dataRange:
            self._dataRange = dataRange
            self._dirtyTicks()

    @property
    def scale(self) -> AxisScaleType:
        return self._scale

    @scale.setter
    def scale(self, scale: AxisScaleType):
        if scale != self._scale:
            self._scale = scale
            self._dirtyTicks()

    @property
    def timeZone(self):
        """Returns datetime.tzinfo that is used if this axis plots date times."""
        return self._timeZone

    @timeZone.setter
    def timeZone(self, tz):
        """Sets datetime.tzinfo that is used if this axis plots date times."""
        self._timeZone = tz
        self._dirtyTicks()

    @property
    def isTimeSeries(self):
        """Whether the axis is showing floats as datetime objects"""
        return self._isDateTime

    @isTimeSeries.setter
    def isTimeSeries(self, isTimeSeries):
        isTimeSeries = bool(isTimeSeries)
        if isTimeSeries != self._isDateTime:
            self._isDateTime = isTimeSeries
            self._dirtyTicks()

    @property
    def displayCoords(self):
        """The coordinates of the start and end points of the axis
        in display space (i.e., in pixels) as a tuple of 2 tuples of
        2 floats: ((x0, y0), (x1, y1)).
        """
        return self._displayCoords

    @displayCoords.setter
    def displayCoords(self, displayCoords):
        assert len(displayCoords) == 2
        assert len(displayCoords[0]) == 2
        assert len(displayCoords[1]) == 2
        displayCoords = tuple(displayCoords[0]), tuple(displayCoords[1])
        if displayCoords != self._displayCoords:
            self._displayCoords = displayCoords
            self._dirtyTicks()

    @property
    def devicePixelRatio(self):
        """Returns the ratio between qt pixels and device pixels."""
        plotFrame = self._plotFrameRef()
        return plotFrame.devicePixelRatio if plotFrame is not None else 1.0

    @property
    def dotsPerInch(self):
        """Returns the screen DPI"""
        plotFrame = self._plotFrameRef()
        return plotFrame.dotsPerInch if plotFrame is not None else 92

    @property
    def title(self):
        """The text label associated with this axis as a str in latin-1."""
        return self._title

    @title.setter
    def title(self, title):
        if title != self._title:
            self._title = title
            self._dirtyPlotFrame()

    @property
    def orderOffsetAnchor(self) -> tuple[float, float]:
        """Anchor position for the tick order&offset text"""
        return self._orderOffsetAnchor

    @orderOffsetAnchor.setter
    def orderOffsetAnchor(self, position: tuple[float, float]):
        if position != self._orderOffsetAnchor:
            self._orderOffsetAnchor = position
            self._dirtyTicks()

    @property
    def titleOffset(self):
        """Title offset in pixels (x: int, y: int)"""
        return self._titleOffset

    @titleOffset.setter
    def titleOffset(self, offset):
        if offset != self._titleOffset:
            self._titleOffset = offset
            self._dirtyTicks()

    @property
    def foregroundColor(self):
        """Color used for frame and labels"""
        return self._foregroundColor

    @foregroundColor.setter
    def foregroundColor(self, color):
        """Color used for frame and labels"""
        assert len(color) == 4, (
            f"foregroundColor must have length 4, got {len(self._foregroundColor)}"
        )
        if self._foregroundColor != color:
            self._foregroundColor = color
            self._dirtyTicks()

    @property
    def ticks(self):
        """Ticks as tuples: ((x, y) in display, dataPos, textLabel)."""
        if self._ticks is None:
            self._ticks = tuple(self._ticksGenerator())
        return self._ticks

    def applyScale(self, data: float | numpy.ndarray) -> float | numpy.ndarray:
        return axis_scale.apply(self.scale, data)

    def revertScale(self, data: float | numpy.ndarray) -> float | numpy.ndarray:
        return axis_scale.revert(self.scale, data)

    def getVerticesAndLabels(self):
        """Create the list of vertices for axis and associated text labels.

        :returns: A tuple: List of 2D line vertices, List of Text2D labels.
        """
        vertices = list(self.displayCoords)  # Add start and end points
        labels = []

        xTickLength, yTickLength = self._tickLength
        xTickLength *= self.devicePixelRatio
        yTickLength *= self.devicePixelRatio
        for (xPixel, yPixel), dataPos, text in self.ticks:
            if text is None:
                tickScale = 0.5
            else:
                tickScale = 1.0

                label = Text2D(
                    text=text,
                    font=self.font,
                    color=self._foregroundColor,
                    x=xPixel - xTickLength,
                    y=yPixel - yTickLength,
                    align=self._labelAlign,
                    valign=self._labelVAlign,
                    devicePixelRatio=self.devicePixelRatio,
                )
                labels.append(label)

            vertices.append((xPixel, yPixel))
            vertices.append(
                (xPixel + tickScale * xTickLength, yPixel + tickScale * yTickLength)
            )

        (x0, y0), (x1, y1) = self.displayCoords
        xAxisCenter = 0.5 * (x0 + x1)
        yAxisCenter = 0.5 * (y0 + y1)

        xOffset, yOffset = self.titleOffset

        # Adaptative title positioning:
        # tickNorm = math.sqrt(xTickLength ** 2 + yTickLength ** 2)
        # xOffset = -tickLabelsSize[0] * xTickLength / tickNorm
        # xOffset -= 3 * xTickLength
        # yOffset = -tickLabelsSize[1] * yTickLength / tickNorm
        # yOffset -= 3 * yTickLength

        axisTitle = Text2D(
            text=self.title,
            font=self.font,
            color=self._foregroundColor,
            x=xAxisCenter + xOffset,
            y=yAxisCenter + yOffset,
            align=self._titleAlign,
            valign=self._titleVAlign,
            rotate=self._titleRotate,
            devicePixelRatio=self.devicePixelRatio,
        )
        labels.append(axisTitle)

        if self._orderAndOffsetText:
            orderAndOffsetFont = self._orderAndOffsetFont(self.font)

            xOrderOffset, yOrderOffset = self.orderOffsetAnchor
            labels.append(
                Text2D(
                    text=self._orderAndOffsetText,
                    font=orderAndOffsetFont,
                    color=self._foregroundColor,
                    x=xOrderOffset,
                    y=yOrderOffset,
                    align=self._orderOffsetAlign,
                    valign=self._orderOffsetVAlign,
                    devicePixelRatio=self.devicePixelRatio,
                )
            )
        return vertices, labels

    @staticmethod
    def _orderAndOffsetFont(font: qt.QFont) -> qt.QFont:
        """Returns a larger bold font"""
        boldBiggerFont = qt.QFont(font)
        boldBiggerFont.setWeight(qt.QFont.ExtraBold)
        # Increase font size which is either in pixel or in points
        pointSize = boldBiggerFont.pointSizeF()
        if pointSize > 0:
            boldBiggerFont.setPointSizeF(1.1 * pointSize)
        pixelSize = boldBiggerFont.pixelSize()
        if pixelSize > 0:
            boldBiggerFont.setPixelSize(int(1.1 * pixelSize))
        return boldBiggerFont

    def _dirtyPlotFrame(self):
        """Dirty parent GLPlotFrame"""
        plotFrame = self._plotFrameRef()
        if plotFrame is not None:
            plotFrame._dirty()

    def _dirtyTicks(self):
        """Mark ticks as dirty and notify listener (i.e., background)."""
        self._ticks = None
        self._dirtyPlotFrame()

    @staticmethod
    def _frange(start, stop, step):
        """range for float (including stop)."""
        while start <= stop:
            yield start
            start += step

    def _defaultFormatTicks(
        self, min_: float, max_: float, tickValues: Sequence[float]
    ) -> tuple[str, list[str]]:
        """Returns offset/order text and list of tick labels"""
        self._tickFormatter.axis.set_view_interval(min_, max_)
        self._tickFormatter.axis.set_data_interval(min_, max_)
        tickLabels = self._tickFormatter.format_ticks(tickValues)
        return self._tickFormatter.get_offset(), tickLabels

    def _ticksGenerator(self):
        """Generator of ticks as tuples:
        ((x, y) in display, dataPos, textLabel).
        """
        self._orderAndOffsetText = ""

        dataMin, dataMax = self.dataRange
        if not axis_scale.isValid(self.scale, dataMin):
            _logger.warning(
                "Getting ticks with dataRange[0] outside axis scale valid range"
            )
            dataMin = 1.0
            if dataMax < dataMin:
                dataMax = 1.0

        if dataMin != dataMax:  # data range is not null
            (x0, y0), (x1, y1) = self.displayCoords

            if self.scale == "asinh":
                if self.isTimeSeries:
                    _logger.warning("Time series not implemented for asinh axes")

                axisLengthInches = (
                    numpy.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / self.dotsPerInch
                )
                # ~1.3 tick per inch
                nTicks = max(3, 2 * int(round(1.3 * axisLengthInches)) // 2 + 1)

                scaledMin, scaledMax = self.applyScale((dataMin, dataMax))
                scaledTentativeTicks = numpy.linspace(scaledMin, scaledMax, nTicks)
                tentativeTicks = self.revertScale(scaledTentativeTicks)
                with numpy.errstate(divide="ignore"):
                    log10Ticks = numpy.sign(tentativeTicks) * 10 ** numpy.floor(
                        numpy.log10(abs(tentativeTicks))
                    )
                uniqueLog10Ticks = set(
                    pos
                    for pos in log10Ticks
                    if numpy.isfinite(pos) and dataMin <= pos <= dataMax
                )
                if dataMin * dataMax < 0:  # crossing zero: ensure 0
                    uniqueLog10Ticks.add(0.0)

                tickPositions = numpy.array(sorted(uniqueLog10Ticks))

                if dataMin * dataMax < 0:
                    # Remove ticks too close to 0
                    scaledTicks = self.applyScale(tickPositions)
                    distanceToZero = axisLengthInches * abs(
                        scaledTicks / (scaledMax - scaledMin)
                    )
                    tickPositions = tickPositions[
                        numpy.logical_or(distanceToZero == 0, distanceToZero >= 0.5)
                    ]

                xScale = (x1 - x0) / (scaledMax - scaledMin)
                yScale = (y1 - y0) / (scaledMax - scaledMin)

                if len(tickPositions) >= 2:
                    for dataPos in tickPositions:
                        scaledPos = self.applyScale(dataPos)
                        xPixel = x0 + (scaledPos - scaledMin) * xScale
                        yPixel = y0 + (scaledPos - scaledMin) * yScale
                        if dataPos != 0:
                            sign = "" if dataPos >= 0 else "-"
                            exp = int(numpy.floor(numpy.log10(abs(dataPos))))
                            text = f"{sign}1e{exp}"
                        else:
                            text = "0"
                        yield ((xPixel, yPixel), dataPos, text)
                else:  # Not enough ticks: Fallback to linear ticks in scaled space
                    tickPositions = self.revertScale(
                        numpy.linspace(scaledMin, scaledMax, nTicks)
                    )
                    offsetText, tickLabels = self._defaultFormatTicks(
                        dataMin, dataMax, tickPositions
                    )
                    self._orderAndOffsetText = offsetText

                    for dataPos, text in zip(tickPositions, tickLabels):
                        xPixel = x0 + (self.applyScale(dataPos) - scaledMin) * xScale
                        yPixel = y0 + (self.applyScale(dataPos) - scaledMin) * yScale
                        yield ((xPixel, yPixel), dataPos, text)

            elif self.scale == "log":
                if self.isTimeSeries:
                    _logger.warning("Time series not implemented for log-scale")

                scaledMin, scaledMax = self.applyScale((dataMin, dataMax))
                tickMin, tickMax, step, _ = niceNumbersForLog10(scaledMin, scaledMax)

                xScale = (x1 - x0) / (scaledMax - scaledMin)
                yScale = (y1 - y0) / (scaledMax - scaledMin)

                for scaledPos in self._frange(tickMin, tickMax, step):
                    if scaledMin <= scaledPos <= scaledMax:
                        dataPos = self.revertScale(scaledPos)
                        xPixel = x0 + (scaledPos - scaledMin) * xScale
                        yPixel = y0 + (scaledPos - scaledMin) * yScale
                        text = "1e%+03d" % scaledPos
                        yield ((xPixel, yPixel), dataPos, text)

                if step == 1:
                    ticks = list(self._frange(tickMin, tickMax, step))[:-1]
                    for scaledPos in ticks:
                        dataOrigPos = self.revertScale(scaledPos)
                        for index in range(2, 10):
                            dataPos = dataOrigPos * index
                            if dataMin <= dataPos <= dataMax:
                                scaledSubPos = self.applyScale(dataPos)
                                xPixel = x0 + (scaledSubPos - scaledMin) * xScale
                                yPixel = y0 + (scaledSubPos - scaledMin) * yScale
                                yield ((xPixel, yPixel), dataPos, None)

            elif self.scale == "linear":
                xScale = (x1 - x0) / (dataMax - dataMin)
                yScale = (y1 - y0) / (dataMax - dataMin)

                nbPixels = (
                    math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2)) / self.devicePixelRatio
                )

                # Density of 1.3 label per 92 pixels
                # i.e., 1.3 label per inch on a 92 dpi screen
                tickDensity = 1.3 * self.devicePixelRatio / self.dotsPerInch

                if not self.isTimeSeries:
                    tickMin, tickMax, step, _ = niceNumbersAdaptative(
                        dataMin, dataMax, nbPixels, tickDensity
                    )

                    visibleTickPositions = [
                        pos
                        for pos in self._frange(tickMin, tickMax, step)
                        if dataMin <= pos <= dataMax
                    ]
                    offsetText, tickLabels = self._defaultFormatTicks(
                        dataMin, dataMax, visibleTickPositions
                    )
                    self._orderAndOffsetText = offsetText

                    for dataPos, text in zip(visibleTickPositions, tickLabels):
                        xPixel = x0 + (dataPos - dataMin) * xScale
                        yPixel = y0 + (dataPos - dataMin) * yScale
                        yield ((xPixel, yPixel), dataPos, text)

                else:
                    # Time series
                    try:
                        dtMin = dt.datetime.fromtimestamp(dataMin, tz=self.timeZone)
                        dtMax = dt.datetime.fromtimestamp(dataMax, tz=self.timeZone)
                    except ValueError:
                        _logger.warning("Data range cannot be displayed with time axis")
                        return  # Range is out of bound of the datetime

                    if bestUnit(
                        (dtMax - dtMin).total_seconds() == DtUnit.MICRO_SECONDS
                    ):
                        # Special case for micro seconds: Reduce tick density
                        tickDensity = 1.0 * self.devicePixelRatio / self.dotsPerInch

                    tickDateTimes, spacing, unit = calcTicksAdaptive(
                        dtMin, dtMax, nbPixels, tickDensity
                    )
                    visibleDatetimes = tuple(
                        dt for dt in tickDateTimes if dtMin <= dt <= dtMax
                    )
                    ticks = formatDatetimes(visibleDatetimes, spacing, unit)

                    for tickDateTime, text in ticks.items():
                        dataPos = tickDateTime.timestamp()
                        xPixel = x0 + (dataPos - dataMin) * xScale
                        yPixel = y0 + (dataPos - dataMin) * yScale
                        yield ((xPixel, yPixel), dataPos, text)
            else:
                raise RuntimeError(f"Unsupported axis scale: {self.scale}")
