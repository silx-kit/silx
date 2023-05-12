# /*##########################################################################
#
# Copyright (c) 2014-2022 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This modules provides the rendering of plot titles, axes and grid.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


# TODO
# keep aspect ratio managed here?
# smarter dirty flag handling?

import datetime as dt
import math
import weakref
import logging
import numbers
from typing import Optional, Union
from collections import namedtuple

import numpy

from ...._glutils import gl, Program
from ..._utils import checkAxisLimits, FLOAT32_MINPOS
from .GLSupport import mat4Ortho
from .GLText import Text2D, CENTER, BOTTOM, TOP, LEFT, RIGHT, ROTATE_270
from ..._utils.ticklayout import niceNumbersAdaptative, niceNumbersForLog10
from ..._utils.dtime_ticklayout import calcTicksAdaptive, bestFormatString
from ..._utils.dtime_ticklayout import timestamp

_logger = logging.getLogger(__name__)


# PlotAxis ####################################################################

class PlotAxis(object):
    """Represents a 1D axis of the plot.
    This class is intended to be used with :class:`GLPlotFrame`.
    """

    def __init__(self, plotFrame,
                 tickLength=(0., 0.),
                 foregroundColor=(0., 0., 0., 1.0),
                 labelAlign=CENTER, labelVAlign=CENTER,
                 titleAlign=CENTER, titleVAlign=CENTER,
                 titleRotate=0, titleOffset=(0., 0.)):
        self._ticks = None

        self._plotFrameRef = weakref.ref(plotFrame)

        self._isDateTime = False
        self._timeZone = None
        self._isLog = False
        self._dataRange = 1., 100.
        self._displayCoords = (0., 0.), (1., 0.)
        self._title = ''

        self._tickLength = tickLength
        self._foregroundColor = foregroundColor
        self._labelAlign = labelAlign
        self._labelVAlign = labelVAlign
        self._titleAlign = titleAlign
        self._titleVAlign = titleVAlign
        self._titleRotate = titleRotate
        self._titleOffset = titleOffset

    @property
    def dataRange(self):
        """The range of the data represented on the axis as a tuple
        of 2 floats: (min, max)."""
        return self._dataRange

    @dataRange.setter
    def dataRange(self, dataRange):
        assert len(dataRange) == 2
        assert dataRange[0] <= dataRange[1]
        dataRange = float(dataRange[0]), float(dataRange[1])

        if dataRange != self._dataRange:
            self._dataRange = dataRange
            self._dirtyTicks()

    @property
    def isLog(self):
        """Whether the axis is using a log10 scale or not as a bool."""
        return self._isLog

    @isLog.setter
    def isLog(self, isLog):
        isLog = bool(isLog)
        if isLog != self._isLog:
            self._isLog = isLog
            self._dirtyTicks()

    @property
    def timeZone(self):
        """Returnss datetime.tzinfo that is used if this axis plots date times."""
        return self._timeZone

    @timeZone.setter
    def timeZone(self, tz):
        """Sets dateetime.tzinfo that is used if this axis plots date times."""
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
        return plotFrame.devicePixelRatio if plotFrame is not None else 1.

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
        assert len(color) == 4, \
            "foregroundColor must have length 4, got {}".format(len(self._foregroundColor))
        if self._foregroundColor != color:
            self._foregroundColor = color
            self._dirtyTicks()

    @property
    def ticks(self):
        """Ticks as tuples: ((x, y) in display, dataPos, textLabel)."""
        if self._ticks is None:
            self._ticks = tuple(self._ticksGenerator())
        return self._ticks

    def getVerticesAndLabels(self):
        """Create the list of vertices for axis and associated text labels.

        :returns: A tuple: List of 2D line vertices, List of Text2D labels.
        """
        vertices = list(self.displayCoords)  # Add start and end points
        labels = []
        tickLabelsSize = [0., 0.]

        xTickLength, yTickLength = self._tickLength
        xTickLength *= self.devicePixelRatio
        yTickLength *= self.devicePixelRatio
        for (xPixel, yPixel), dataPos, text in self.ticks:
            if text is None:
                tickScale = 0.5
            else:
                tickScale = 1.

                label = Text2D(text=text,
                               color=self._foregroundColor,
                               x=xPixel - xTickLength,
                               y=yPixel - yTickLength,
                               align=self._labelAlign,
                               valign=self._labelVAlign,
                               devicePixelRatio=self.devicePixelRatio)

                width, height = label.size
                if width > tickLabelsSize[0]:
                    tickLabelsSize[0] = width
                if height > tickLabelsSize[1]:
                    tickLabelsSize[1] = height

                labels.append(label)

            vertices.append((xPixel, yPixel))
            vertices.append((xPixel + tickScale * xTickLength,
                             yPixel + tickScale * yTickLength))

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

        axisTitle = Text2D(text=self.title,
                           color=self._foregroundColor,
                           x=xAxisCenter + xOffset,
                           y=yAxisCenter + yOffset,
                           align=self._titleAlign,
                           valign=self._titleVAlign,
                           rotate=self._titleRotate,
                           devicePixelRatio=self.devicePixelRatio)
        labels.append(axisTitle)

        return vertices, labels

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

    def _ticksGenerator(self):
        """Generator of ticks as tuples:
        ((x, y) in display, dataPos, textLabel).
        """
        dataMin, dataMax = self.dataRange
        if self.isLog and dataMin <= 0.:
            _logger.warning(
                'Getting ticks while isLog=True and dataRange[0]<=0.')
            dataMin = 1.
            if dataMax < dataMin:
                dataMax = 1.

        if dataMin != dataMax:  # data range is not null
            (x0, y0), (x1, y1) = self.displayCoords

            if self.isLog:

                if self.isTimeSeries:
                    _logger.warning("Time series not implemented for log-scale")

                logMin, logMax = math.log10(dataMin), math.log10(dataMax)
                tickMin, tickMax, step, _ = niceNumbersForLog10(logMin, logMax)

                xScale = (x1 - x0) / (logMax - logMin)
                yScale = (y1 - y0) / (logMax - logMin)

                for logPos in self._frange(tickMin, tickMax, step):
                    if logMin <= logPos <= logMax:
                        dataPos = 10 ** logPos
                        xPixel = x0 + (logPos - logMin) * xScale
                        yPixel = y0 + (logPos - logMin) * yScale
                        text = '1e%+03d' % logPos
                        yield ((xPixel, yPixel), dataPos, text)

                if step == 1:
                    ticks = list(self._frange(tickMin, tickMax, step))[:-1]
                    for logPos in ticks:
                        dataOrigPos = 10 ** logPos
                        for index in range(2, 10):
                            dataPos = dataOrigPos * index
                            if dataMin <= dataPos <= dataMax:
                                logSubPos = math.log10(dataPos)
                                xPixel = x0 + (logSubPos - logMin) * xScale
                                yPixel = y0 + (logSubPos - logMin) * yScale
                                yield ((xPixel, yPixel), dataPos, None)

            else:
                xScale = (x1 - x0) / (dataMax - dataMin)
                yScale = (y1 - y0) / (dataMax - dataMin)

                nbPixels = math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2)) / self.devicePixelRatio

                # Density of 1.3 label per 92 pixels
                # i.e., 1.3 label per inch on a 92 dpi screen
                tickDensity = 1.3 / 92

                if not self.isTimeSeries:
                    tickMin, tickMax, step, nbFrac = niceNumbersAdaptative(
                        dataMin, dataMax, nbPixels, tickDensity)

                    for dataPos in self._frange(tickMin, tickMax, step):
                        if dataMin <= dataPos <= dataMax:
                            xPixel = x0 + (dataPos - dataMin) * xScale
                            yPixel = y0 + (dataPos - dataMin) * yScale

                            if nbFrac == 0:
                                text = '%g' % dataPos
                            else:
                                text = ('%.' + str(nbFrac) + 'f') % dataPos
                            yield ((xPixel, yPixel), dataPos, text)
                else:
                    # Time series
                    try:
                        dtMin = dt.datetime.fromtimestamp(dataMin, tz=self.timeZone)
                        dtMax = dt.datetime.fromtimestamp(dataMax, tz=self.timeZone)
                    except ValueError:
                        _logger.warning("Data range cannot be displayed with time axis")
                        return  # Range is out of bound of the datetime

                    tickDateTimes, spacing, unit = calcTicksAdaptive(
                        dtMin, dtMax, nbPixels, tickDensity)

                    for tickDateTime in tickDateTimes:
                        if dtMin <= tickDateTime <= dtMax:

                            dataPos = timestamp(tickDateTime)
                            xPixel = x0 + (dataPos - dataMin) * xScale
                            yPixel = y0 + (dataPos - dataMin) * yScale

                            fmtStr = bestFormatString(spacing, unit)
                            text = tickDateTime.strftime(fmtStr)

                            yield ((xPixel, yPixel), dataPos, text)


# GLPlotFrame #################################################################

class GLPlotFrame(object):
    """Base class for rendering a 2D frame surrounded by axes."""

    _TICK_LENGTH_IN_PIXELS = 5
    _LINE_WIDTH = 1

    _SHADERS = {
        'vertex': """
    attribute vec2 position;
    uniform mat4 matrix;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """,
        'fragment': """
    uniform vec4 color;
    uniform float tickFactor; /* = 1./tickLength or 0. for solid line */

    void main(void) {
        if (mod(tickFactor * (gl_FragCoord.x + gl_FragCoord.y), 2.) < 1.) {
            gl_FragColor = color;
        } else {
            discard;
        }
    }
    """
    }

    _Margins = namedtuple('Margins', ('left', 'right', 'top', 'bottom'))

    # Margins used when plot frame is not displayed
    _NoDisplayMargins = _Margins(0, 0, 0, 0)

    def __init__(self, marginRatios, foregroundColor, gridColor):
        """
        :param List[float] marginRatios:
            The ratios of margins around plot area for axis and labels.
            (left, top, right, bottom) as float in [0., 1.]
        :param foregroundColor: color used for the frame and labels.
        :type foregroundColor: tuple with RGBA values ranging from 0.0 to 1.0
        :param gridColor: color used for grid lines.
        :type gridColor: tuple RGBA with RGBA values ranging from 0.0 to 1.0
        """
        self._renderResources = None

        self.__marginRatios = marginRatios
        self.__marginsCache = None

        self._foregroundColor = foregroundColor
        self._gridColor = gridColor

        self.axes = []  # List of PlotAxis to be updated by subclasses

        self._grid = False
        self._size = 0., 0.
        self._title = ''

        self._devicePixelRatio = 1.

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return self._renderResources is None

    GRID_NONE = 0
    GRID_MAIN_TICKS = 1
    GRID_SUB_TICKS = 2
    GRID_ALL_TICKS = (GRID_MAIN_TICKS + GRID_SUB_TICKS)

    @property
    def foregroundColor(self):
        """Color used for frame and labels"""
        return self._foregroundColor
        
    @foregroundColor.setter
    def foregroundColor(self, color):
        """Color used for frame and labels"""
        assert len(color) == 4, \
            "foregroundColor must have length 4, got {}".format(len(self._foregroundColor))
        if self._foregroundColor != color:
            self._foregroundColor = color
            for axis in self.axes:
                axis.foregroundColor = color
            self._dirty()

    @property
    def gridColor(self):
        """Color used for frame and labels"""
        return self._gridColor
        
    @gridColor.setter
    def gridColor(self, color):
        """Color used for frame and labels"""
        assert len(color) == 4, \
            "gridColor must have length 4, got {}".format(len(self._gridColor))
        if self._gridColor != color:
            self._gridColor = color
            self._dirty()

    @property
    def marginRatios(self):
        """Plot margin ratios: (left, top, right, bottom) as 4 float in [0, 1].
        """
        return self.__marginRatios

    @marginRatios.setter
    def marginRatios(self, ratios):
        ratios = tuple(float(v) for v in ratios)
        assert len(ratios) == 4
        for value in ratios:
            assert 0. <= value <= 1.
        assert ratios[0] + ratios[2] < 1.
        assert ratios[1] + ratios[3] < 1.

        if self.__marginRatios != ratios:
            self.__marginRatios = ratios
            self.__marginsCache = None  # Clear cached margins
            self._dirty()

    @property
    def margins(self):
        """Margins in pixels around the plot."""
        if self.__marginsCache is None:
            width, height = self.size
            left, top, right, bottom = self.marginRatios
            self.__marginsCache = self._Margins(
                left=int(left*width),
                right=int(right*width),
                top=int(top*height),
                bottom=int(bottom*height))
        return self.__marginsCache

    @property
    def devicePixelRatio(self):
        return self._devicePixelRatio

    @devicePixelRatio.setter
    def devicePixelRatio(self, ratio):
        if ratio != self._devicePixelRatio:
            self._devicePixelRatio = ratio
            self._dirty()

    @property
    def grid(self):
        """Grid display mode:
        - 0: No grid.
        - 1: Grid on main ticks.
        - 2: Grid on sub-ticks for log scale axes.
        - 3: Grid on main and sub ticks."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert grid in (self.GRID_NONE, self.GRID_MAIN_TICKS,
                        self.GRID_SUB_TICKS, self.GRID_ALL_TICKS)
        if grid != self._grid:
            self._grid = grid
            self._dirty()

    @property
    def size(self):
        """Size in device pixels of the plot area including margins."""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 2
        size = tuple(size)
        if size != self._size:
            self._size = size
            self.__marginsCache = None  # Clear cached margins
            self._dirty()

    @property
    def plotOrigin(self):
        """Plot area origin (left, top) in widget coordinates in pixels."""
        return self.margins.left, self.margins.top

    @property
    def plotSize(self):
        """Plot area size (width, height) in pixels."""
        w, h = self.size
        w -= self.margins.left + self.margins.right
        h -= self.margins.top + self.margins.bottom
        return w, h

    @property
    def title(self):
        """Main title as a str in latin-1."""
        return self._title

    @title.setter
    def title(self, title):
        if title != self._title:
            self._title = title
            self._dirty()

        # In-place update
        # if self._renderResources is not None:
        #    self._renderResources[-1][-1].text = title

    def _dirty(self):
        # When Text2D require discard we need to handle it
        self._renderResources = None

    def _buildGridVertices(self):
        if self._grid == self.GRID_NONE:
            return []

        elif self._grid == self.GRID_MAIN_TICKS:
            def test(text):
                return text is not None
        elif self._grid == self.GRID_SUB_TICKS:
            def test(text):
                return text is None
        elif self._grid == self.GRID_ALL_TICKS:
            def test(_):
                return True
        else:
            logging.warning('Wrong grid mode: %d' % self._grid)
            return []

        return self._buildGridVerticesWithTest(test)

    def _buildGridVerticesWithTest(self, test):
        """Override in subclass to generate grid vertices"""
        return []

    def _buildVerticesAndLabels(self):
        # To fill with copy of axes lists
        vertices = []
        labels = []

        for axis in self.axes:
            axisVertices, axisLabels = axis.getVerticesAndLabels()
            vertices += axisVertices
            labels += axisLabels

        vertices = numpy.array(vertices, dtype=numpy.float32)

        # Add main title
        xTitle = (self.size[0] + self.margins.left -
                  self.margins.right) // 2
        yTitle = self.margins.top - self._TICK_LENGTH_IN_PIXELS
        labels.append(Text2D(text=self.title,
                             color=self._foregroundColor,
                             x=xTitle,
                             y=yTitle,
                             align=CENTER,
                             valign=BOTTOM,
                             devicePixelRatio=self.devicePixelRatio))

        # grid
        gridVertices = numpy.array(self._buildGridVertices(),
                                   dtype=numpy.float32)

        self._renderResources = (vertices, gridVertices, labels)

    _program = Program(
        _SHADERS['vertex'], _SHADERS['fragment'], attrib0='position')

    def render(self):
        if self.margins == self._NoDisplayMargins:
            return

        if self._renderResources is None:
            self._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        width, height = self.size
        matProj = mat4Ortho(0, width, height, 0, 1, -1)

        gl.glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        gl.glLineWidth(self._LINE_WIDTH)

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE,
                              matProj.astype(numpy.float32))
        gl.glUniform4f(prog.uniforms['color'], *self._foregroundColor)
        gl.glUniform1f(prog.uniforms['tickFactor'], 0.)

        gl.glEnableVertexAttribArray(prog.attributes['position'])
        gl.glVertexAttribPointer(prog.attributes['position'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0, vertices)

        gl.glDrawArrays(gl.GL_LINES, 0, len(vertices))

        for label in labels:
            label.render(matProj)

    def renderGrid(self):
        if self._grid == self.GRID_NONE:
            return

        if self._renderResources is None:
            self._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        width, height = self.size
        matProj = mat4Ortho(0, width, height, 0, 1, -1)

        gl.glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        gl.glLineWidth(self._LINE_WIDTH)
        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE,
                              matProj.astype(numpy.float32))
        gl.glUniform4f(prog.uniforms['color'], *self._gridColor)
        gl.glUniform1f(prog.uniforms['tickFactor'], 0.)  # 1/2.)  # 1/tickLen

        gl.glEnableVertexAttribArray(prog.attributes['position'])
        gl.glVertexAttribPointer(prog.attributes['position'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0, gridVertices)

        gl.glDrawArrays(gl.GL_LINES, 0, len(gridVertices))


# GLPlotFrame2D ###############################################################

class GLPlotFrame2D(GLPlotFrame):
    def __init__(self, marginRatios, foregroundColor, gridColor):
        """
        :param List[float] marginRatios:
            The ratios of margins around plot area for axis and labels.
            (left, top, right, bottom) as float in [0., 1.]
        :param foregroundColor: color used for the frame and labels.
        :type foregroundColor: tuple with RGBA values ranging from 0.0 to 1.0
        :param gridColor: color used for grid lines.
        :type gridColor: tuple RGBA with RGBA values ranging from 0.0 to 1.0

        """
        super(GLPlotFrame2D, self).__init__(marginRatios, foregroundColor, gridColor)
        self.axes.append(PlotAxis(self,
                                  tickLength=(0., -5.),
                                  foregroundColor=self._foregroundColor,
                                  labelAlign=CENTER, labelVAlign=TOP,
                                  titleAlign=CENTER, titleVAlign=TOP,
                                  titleRotate=0))

        self._x2AxisCoords = ()

        self.axes.append(PlotAxis(self,
                                  tickLength=(5., 0.),
                                  foregroundColor=self._foregroundColor,
                                  labelAlign=RIGHT, labelVAlign=CENTER,
                                  titleAlign=CENTER, titleVAlign=BOTTOM,
                                  titleRotate=ROTATE_270))

        self._y2Axis = PlotAxis(self,
                                tickLength=(-5., 0.),
                                foregroundColor=self._foregroundColor,
                                labelAlign=LEFT, labelVAlign=CENTER,
                                titleAlign=CENTER, titleVAlign=TOP,
                                titleRotate=ROTATE_270)

        self._isYAxisInverted = False

        self._dataRanges = {
            'x': (1., 100.), 'y': (1., 100.), 'y2': (1., 100.)}

        self._baseVectors = (1., 0.), (0., 1.)

        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    def _dirty(self):
        super(GLPlotFrame2D, self)._dirty()
        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return (super(GLPlotFrame2D, self).isDirty or
                self._transformedDataRanges is None or
                self._transformedDataProjMat is None or
                self._transformedDataY2ProjMat is None)

    @property
    def xAxis(self):
        return self.axes[0]

    @property
    def yAxis(self):
        return self.axes[1]

    @property
    def y2Axis(self):
        return self._y2Axis

    @property
    def isY2Axis(self):
        """Whether to display the left Y axis or not."""
        return len(self.axes) == 3

    @isY2Axis.setter
    def isY2Axis(self, isY2Axis):
        if isY2Axis != self.isY2Axis:
            if isY2Axis:
                self.axes.append(self._y2Axis)
            else:
                self.axes = self.axes[:2]

            self._dirty()

    @property
    def isYAxisInverted(self):
        """Whether Y axes are inverted or not as a bool."""
        return self._isYAxisInverted

    @isYAxisInverted.setter
    def isYAxisInverted(self, value):
        value = bool(value)
        if value != self._isYAxisInverted:
            self._isYAxisInverted = value
            self._dirty()

    DEFAULT_BASE_VECTORS = (1., 0.), (0., 1.)
    """Values of baseVectors for orthogonal axes."""

    @property
    def baseVectors(self):
        """Coordinates of the X and Y axes in the orthogonal plot coords.

        Raises ValueError if corresponding matrix is singular.

        2 tuples of 2 floats: (xx, xy), (yx, yy)
        """
        return self._baseVectors

    @baseVectors.setter
    def baseVectors(self, baseVectors):
        self._dirty()

        (xx, xy), (yx, yy) = baseVectors
        vectors = (float(xx), float(xy)), (float(yx), float(yy))

        det = (vectors[0][0] * vectors[1][1] - vectors[1][0] * vectors[0][1])
        if det == 0.:
            raise ValueError("Singular matrix for base vectors: " +
                             str(vectors))

        if vectors != self._baseVectors:
            self._baseVectors = vectors
            self._dirty()

    def _updateTitleOffset(self):
        """Update axes title offset according to margins"""
        margins = self.margins
        self.xAxis.titleOffset = 0, margins.bottom // 2
        self.yAxis.titleOffset = -3 * margins.left // 4, 0
        self.y2Axis.titleOffset = 3 * margins.right // 4, 0

    # Override size and marginRatios setters to update titleOffsets
    @GLPlotFrame.size.setter
    def size(self, size):
        GLPlotFrame.size.fset(self, size)
        self._updateTitleOffset()

    @GLPlotFrame.marginRatios.setter
    def marginRatios(self, ratios):
        GLPlotFrame.marginRatios.fset(self, ratios)
        self._updateTitleOffset()

    @property
    def dataRanges(self):
        """Ranges of data visible in the plot on x, y and y2 axes.

        This is different to the axes range when axes are not orthogonal.

        Type: ((xMin, xMax), (yMin, yMax), (y2Min, y2Max))
        """
        return self._DataRanges(self._dataRanges['x'],
                                self._dataRanges['y'],
                                self._dataRanges['y2'])

    def setDataRanges(self, x=None, y=None, y2=None):
        """Set data range over each axes.

        The provided ranges are clipped to possible values
        (i.e., 32 float range + positive range for log scale).

        :param x: (min, max) data range over X axis
        :param y: (min, max) data range over Y axis
        :param y2: (min, max) data range over Y2 axis
        """
        if x is not None:
            self._dataRanges['x'] = checkAxisLimits(
                x[0], x[1], self.xAxis.isLog, name='x')

        if y is not None:
            self._dataRanges['y'] = checkAxisLimits(
                y[0], y[1], self.yAxis.isLog, name='y')

        if y2 is not None:
            self._dataRanges['y2'] = checkAxisLimits(
                y2[0], y2[1], self.y2Axis.isLog, name='y2')

        self.xAxis.dataRange = self._dataRanges['x']
        self.yAxis.dataRange = self._dataRanges['y']
        self.y2Axis.dataRange = self._dataRanges['y2']

    _DataRanges = namedtuple('dataRanges', ('x', 'y', 'y2'))

    @property
    def transformedDataRanges(self):
        """Bounds of the displayed area in transformed data coordinates
        (i.e., log scale applied if any as well as skew)

        3-tuple of 2-tuple (min, max) for each axis: x, y, y2.
        """
        if self._transformedDataRanges is None:
            (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = self.dataRanges

            if self.xAxis.isLog:
                try:
                    xMin = math.log10(xMin)
                except ValueError:
                    _logger.info('xMin: warning log10(%f)', xMin)
                    xMin = 0.
                try:
                    xMax = math.log10(xMax)
                except ValueError:
                    _logger.info('xMax: warning log10(%f)', xMax)
                    xMax = 0.

            if self.yAxis.isLog:
                try:
                    yMin = math.log10(yMin)
                except ValueError:
                    _logger.info('yMin: warning log10(%f)', yMin)
                    yMin = 0.
                try:
                    yMax = math.log10(yMax)
                except ValueError:
                    _logger.info('yMax: warning log10(%f)', yMax)
                    yMax = 0.

                try:
                    y2Min = math.log10(y2Min)
                except ValueError:
                    _logger.info('yMin: warning log10(%f)', y2Min)
                    y2Min = 0.
                try:
                    y2Max = math.log10(y2Max)
                except ValueError:
                    _logger.info('yMax: warning log10(%f)', y2Max)
                    y2Max = 0.

            self._transformedDataRanges = self._DataRanges(
                (xMin, xMax), (yMin, yMax), (y2Min, y2Max))

        return self._transformedDataRanges

    @property
    def transformedDataProjMat(self):
        """Orthographic projection matrix for rendering transformed data

        :type: numpy.matrix
        """
        if self._transformedDataProjMat is None:
            xMin, xMax = self.transformedDataRanges.x
            yMin, yMax = self.transformedDataRanges.y

            if self.isYAxisInverted:
                mat = mat4Ortho(xMin, xMax, yMax, yMin, 1, -1)
            else:
                mat = mat4Ortho(xMin, xMax, yMin, yMax, 1, -1)
            self._transformedDataProjMat = mat

        return self._transformedDataProjMat

    @property
    def transformedDataY2ProjMat(self):
        """Orthographic projection matrix for rendering transformed data
        for the 2nd Y axis

        :type: numpy.matrix
        """
        if self._transformedDataY2ProjMat is None:
            xMin, xMax = self.transformedDataRanges.x
            y2Min, y2Max = self.transformedDataRanges.y2

            if self.isYAxisInverted:
                mat = mat4Ortho(xMin, xMax, y2Max, y2Min, 1, -1)
            else:
                mat = mat4Ortho(xMin, xMax, y2Min, y2Max, 1, -1)
            self._transformedDataY2ProjMat = mat

        return self._transformedDataY2ProjMat

    @staticmethod
    def __applyLog(
        data: Union[float, numpy.ndarray],
        isLog: bool
    ) -> Optional[Union[float, numpy.ndarray]]:
        """Apply log to data filtering out """
        if not isLog:
            return data

        if isinstance(data, numbers.Real):
            return None if data < FLOAT32_MINPOS else math.log10(data)

        isBelowMin = data < FLOAT32_MINPOS
        if numpy.any(isBelowMin):
            data = numpy.array(data, copy=True, dtype=numpy.float64)
            data[isBelowMin] = numpy.nan

        with numpy.errstate(divide='ignore'):
            return numpy.log10(data)

    def dataToPixel(self, x, y, axis='left'):
        """Convert data coordinate to widget pixel coordinate.
        """
        assert axis in ('left', 'right')

        trBounds = self.transformedDataRanges

        xDataTr = self.__applyLog(x, self.xAxis.isLog)
        if xDataTr is None:
            return None

        yDataTr = self.__applyLog(y, self.yAxis.isLog)
        if yDataTr is None:
            return None

        # Non-orthogonal axes
        if self.baseVectors != self.DEFAULT_BASE_VECTORS:
            (xx, xy), (yx, yy) = self.baseVectors
            skew_mat = numpy.array(((xx, yx), (xy, yy)))

            coords = numpy.dot(skew_mat, numpy.array((xDataTr, yDataTr)))
            xDataTr, yDataTr = coords

        plotWidth, plotHeight = self.plotSize

        xPixel = (self.margins.left +
            plotWidth * (xDataTr - trBounds.x[0]) /
            (trBounds.x[1] - trBounds.x[0]))

        usedAxis = trBounds.y if axis == "left" else trBounds.y2
        yOffset = (plotHeight * (yDataTr - usedAxis[0]) /
                   (usedAxis[1] - usedAxis[0]))

        if self.isYAxisInverted:
            yPixel = self.margins.top + yOffset
        else:
            yPixel = self.size[1] - self.margins.bottom - yOffset

        return (
            int(xPixel) if isinstance(xPixel, numbers.Real) else xPixel.astype(numpy.int64),
            int(yPixel) if isinstance(yPixel, numbers.Real) else yPixel.astype(numpy.int64),
        )

    def pixelToData(self, x, y, axis="left"):
        """Convert pixel position to data coordinates.

        :param float x: X coord
        :param float y: Y coord
        :param str axis: Y axis to use in ('left', 'right')
        :return: (x, y) position in data coords
        """
        assert axis in ("left", "right")

        plotWidth, plotHeight = self.plotSize

        trBounds = self.transformedDataRanges

        xData = (x - self.margins.left + 0.5) / float(plotWidth)
        xData = trBounds.x[0] + xData * (trBounds.x[1] - trBounds.x[0])

        usedAxis = trBounds.y if axis == "left" else trBounds.y2
        if self.isYAxisInverted:
            yData = (y - self.margins.top + 0.5) / float(plotHeight)
            yData = usedAxis[0] + yData * (usedAxis[1] - usedAxis[0])
        else:
            yData = self.size[1] - self.margins.bottom - y - 0.5
            yData /= float(plotHeight)
            yData = usedAxis[0] + yData * (usedAxis[1] - usedAxis[0])

        # non-orthogonal axis
        if self.baseVectors != self.DEFAULT_BASE_VECTORS:
            (xx, xy), (yx, yy) = self.baseVectors
            skew_mat = numpy.array(((xx, yx), (xy, yy)))
            skew_mat = numpy.linalg.inv(skew_mat)

            coords = numpy.dot(skew_mat, numpy.array((xData, yData)))
            xData, yData = coords

        if self.xAxis.isLog:
            xData = pow(10, xData)
        if self.yAxis.isLog:
            yData = pow(10, yData)

        return xData, yData

    def _buildGridVerticesWithTest(self, test):
        vertices = []

        if self.baseVectors == self.DEFAULT_BASE_VECTORS:
            for axis in self.axes:
                for (xPixel, yPixel), data, text in axis.ticks:
                    if test(text):
                        vertices.append((xPixel, yPixel))
                        if axis == self.xAxis:
                            vertices.append((xPixel, self.margins.top))
                        elif axis == self.yAxis:
                            vertices.append((self.size[0] - self.margins.right,
                                             yPixel))
                        else:  # axis == self.y2Axis
                            vertices.append((self.margins.left, yPixel))

        else:
            # Get plot corners in data coords
            plotLeft, plotTop = self.plotOrigin
            plotWidth, plotHeight = self.plotSize

            corners = [(plotLeft, plotTop),
                       (plotLeft, plotTop + plotHeight),
                       (plotLeft + plotWidth, plotTop + plotHeight),
                       (plotLeft + plotWidth, plotTop)]

            for axis in self.axes:
                if axis == self.xAxis:
                    cornersInData = numpy.array([
                        self.pixelToData(x, y) for (x, y) in corners])
                    borders = ((cornersInData[0], cornersInData[3]),  # top
                               (cornersInData[1], cornersInData[0]),  # left
                               (cornersInData[3], cornersInData[2]))  # right

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if min(x0, x1) <= data < max(x0, x1):
                                    yIntersect = (data - x0) * \
                                        (y1 - y0) / (x1 - x0) + y0

                                    pixelPos = self.dataToPixel(
                                        data, yIntersect)
                                    if pixelPos is not None:
                                        vertices.append((xPixel, yPixel))
                                        vertices.append(pixelPos)
                                    break  # Stop at first intersection

                else:  # y or y2 axes
                    if axis == self.yAxis:
                        axis_name = 'left'
                        cornersInData = numpy.array([
                            self.pixelToData(x, y) for (x, y) in corners])
                        borders = (
                            (cornersInData[3], cornersInData[2]),  # right
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]))  # bottom

                    else:  # axis == self.y2Axis
                        axis_name = 'right'
                        corners = numpy.array([self.pixelToData(
                            x, y, axis='right') for (x, y) in corners])
                        borders = (
                            (cornersInData[1], cornersInData[0]),  # left
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]))  # bottom

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if min(y0, y1) <= data < max(y0, y1):
                                    xIntersect = (data - y0) * \
                                        (x1 - x0) / (y1 - y0) + x0

                                    pixelPos = self.dataToPixel(
                                        xIntersect, data, axis=axis_name)
                                    if pixelPos is not None:
                                        vertices.append((xPixel, yPixel))
                                        vertices.append(pixelPos)
                                    break  # Stop at first intersection

        return vertices

    def _buildVerticesAndLabels(self):
        width, height = self.size

        xCoords = (self.margins.left - 0.5,
                   width - self.margins.right + 0.5)
        yCoords = (height - self.margins.bottom + 0.5,
                   self.margins.top - 0.5)

        self.axes[0].displayCoords = ((xCoords[0], yCoords[0]),
                                      (xCoords[1], yCoords[0]))

        self._x2AxisCoords = ((xCoords[0], yCoords[1]),
                              (xCoords[1], yCoords[1]))

        if self.isYAxisInverted:
            # Y axes are inverted, axes coordinates are inverted
            yCoords = yCoords[1], yCoords[0]

        self.axes[1].displayCoords = ((xCoords[0], yCoords[0]),
                                      (xCoords[0], yCoords[1]))

        self._y2Axis.displayCoords = ((xCoords[1], yCoords[0]),
                                      (xCoords[1], yCoords[1]))

        super(GLPlotFrame2D, self)._buildVerticesAndLabels()

        vertices, gridVertices, labels = self._renderResources

        # Adds vertices for borders without axis
        extraVertices = []
        extraVertices += self._x2AxisCoords
        if not self.isY2Axis:
            extraVertices += self._y2Axis.displayCoords

        extraVertices = numpy.array(
            extraVertices, copy=False, dtype=numpy.float32)
        vertices = numpy.append(vertices, extraVertices, axis=0)

        self._renderResources = (vertices, gridVertices, labels)

    @property
    def foregroundColor(self):
        """Color used for frame and labels"""
        return self._foregroundColor

    @foregroundColor.setter
    def foregroundColor(self, color):
        """Color used for frame and labels"""
        assert len(color) == 4, \
            "foregroundColor must have length 4, got {}".format(len(self._foregroundColor))
        if self._foregroundColor != color:
            self._y2Axis.foregroundColor = color
            GLPlotFrame.foregroundColor.fset(self, color) # call parent property
