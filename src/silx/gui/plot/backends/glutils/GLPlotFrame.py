# /*##########################################################################
#
# Copyright (c) 2014-2023 European Synchrotron Radiation Facility
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

import logging
import numbers
from collections import namedtuple

import numpy


from .....utils.deprecation import deprecated_warning
from .... import qt
from ...._glutils import gl, Program
from ..._utils import checkAxisLimits
from .GLSupport import mat4Ortho
from .GLText import Text2D, CENTER, BOTTOM, TOP, LEFT, RIGHT, ROTATE_270
from ._PlotAxis import PlotAxis as _PlotAxis


class PlotAxis(_PlotAxis):
    def __init__(self, *args, **kwargs):
        deprecated_warning(
            type_="Class",
            name="PlotAxis",
            reason="PlotAxis will be removed from the public API.",
            since_version="3.1.0",
        )
        super().__init__(*args, **kwargs)


# GLPlotFrame #################################################################


class GLPlotFrame:
    """Base class for rendering a 2D frame surrounded by axes."""

    _TICK_LENGTH_IN_PIXELS = 5
    _LINE_WIDTH = 1

    _SHADERS = {
        "vertex": """
    attribute vec2 position;
    uniform mat4 matrix;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """,
        "fragment": """
    uniform vec4 color;
    uniform float tickFactor; /* = 1./tickLength or 0. for solid line */

    void main(void) {
        if (mod(tickFactor * (gl_FragCoord.x + gl_FragCoord.y), 2.) < 1.) {
            gl_FragColor = color;
        } else {
            discard;
        }
    }
    """,
    }

    _Margins = namedtuple("Margins", ("left", "right", "top", "bottom"))

    # Margins used when plot frame is not displayed
    _NoDisplayMargins = _Margins(0, 0, 0, 0)

    def __init__(self, marginRatios, foregroundColor, gridColor, font: qt.QFont):
        """
        :param List[float] marginRatios:
            The ratios of margins around plot area for axis and labels.
            (left, top, right, bottom) as float in [0., 1.]
        :param foregroundColor: color used for the frame and labels.
        :type foregroundColor: tuple with RGBA values ranging from 0.0 to 1.0
        :param gridColor: color used for grid lines.
        :type gridColor: tuple RGBA with RGBA values ranging from 0.0 to 1.0
        :param font: Font used by the axes label
        """
        self._renderResources = None

        self.__marginRatios = marginRatios
        self.__marginsCache = None

        self._foregroundColor = foregroundColor
        self._gridColor = gridColor

        self.axes = []  # List of PlotAxis to be updated by subclasses

        self._grid = False
        self._size = 0.0, 0.0
        self._title = ""
        self._font: qt.QFont = font

        self._devicePixelRatio = 1.0
        self._dpi = 92

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return self._renderResources is None

    GRID_NONE = 0
    GRID_MAIN_TICKS = 1
    GRID_SUB_TICKS = 2
    GRID_ALL_TICKS = GRID_MAIN_TICKS + GRID_SUB_TICKS

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
        assert len(color) == 4, (
            f"gridColor must have length 4, got {len(self._gridColor)}"
        )
        if self._gridColor != color:
            self._gridColor = color
            self._dirty()

    @property
    def marginRatios(self):
        """Plot margin ratios: (left, top, right, bottom) as 4 float in [0, 1]."""
        return self.__marginRatios

    @marginRatios.setter
    def marginRatios(self, ratios):
        ratios = tuple(float(v) for v in ratios)
        assert len(ratios) == 4
        for value in ratios:
            assert 0.0 <= value <= 1.0
        assert ratios[0] + ratios[2] < 1.0
        assert ratios[1] + ratios[3] < 1.0

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
                left=int(left * width),
                right=int(right * width),
                top=int(top * height),
                bottom=int(bottom * height),
            )
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
    def dotsPerInch(self):
        return self._dpi

    @dotsPerInch.setter
    def dotsPerInch(self, dpi):
        if dpi != self._dpi:
            self._dpi = dpi
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
        assert grid in (
            self.GRID_NONE,
            self.GRID_MAIN_TICKS,
            self.GRID_SUB_TICKS,
            self.GRID_ALL_TICKS,
        )
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
            logging.warning("Wrong grid mode: %d" % self._grid)
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
        xTitle = (self.size[0] + self.margins.left - self.margins.right) // 2
        yTitle = self.margins.top - self._TICK_LENGTH_IN_PIXELS
        labels.append(
            Text2D(
                text=self.title,
                font=self._font,
                color=self._foregroundColor,
                x=xTitle,
                y=yTitle,
                align=CENTER,
                valign=BOTTOM,
                devicePixelRatio=self.devicePixelRatio,
            )
        )

        # grid
        gridVertices = numpy.array(self._buildGridVertices(), dtype=numpy.float32)

        self._renderResources = (vertices, gridVertices, labels)

    _program = Program(_SHADERS["vertex"], _SHADERS["fragment"], attrib0="position")

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

        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, matProj.astype(numpy.float32)
        )
        gl.glUniform4f(prog.uniforms["color"], *self._foregroundColor)
        gl.glUniform1f(prog.uniforms["tickFactor"], 0.0)

        gl.glEnableVertexAttribArray(prog.attributes["position"])
        gl.glVertexAttribPointer(
            prog.attributes["position"], 2, gl.GL_FLOAT, gl.GL_FALSE, 0, vertices
        )

        gl.glDrawArrays(gl.GL_LINES, 0, len(vertices))

        for label in labels:
            label.render(matProj, self.dotsPerInch)

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
        gl.glUniformMatrix4fv(
            prog.uniforms["matrix"], 1, gl.GL_TRUE, matProj.astype(numpy.float32)
        )
        gl.glUniform4f(prog.uniforms["color"], *self._gridColor)
        gl.glUniform1f(prog.uniforms["tickFactor"], 0.0)  # 1/2.)  # 1/tickLen

        gl.glEnableVertexAttribArray(prog.attributes["position"])
        gl.glVertexAttribPointer(
            prog.attributes["position"], 2, gl.GL_FLOAT, gl.GL_FALSE, 0, gridVertices
        )

        gl.glDrawArrays(gl.GL_LINES, 0, len(gridVertices))


# GLPlotFrame2D ###############################################################


class GLPlotFrame2D(GLPlotFrame):
    def __init__(self, marginRatios, foregroundColor, gridColor, font: qt.QFont):
        """
        :param List[float] marginRatios:
            The ratios of margins around plot area for axis and labels.
            (left, top, right, bottom) as float in [0., 1.]
        :param foregroundColor: color used for the frame and labels.
        :type foregroundColor: tuple with RGBA values ranging from 0.0 to 1.0
        :param gridColor: color used for grid lines.
        :type gridColor: tuple RGBA with RGBA values ranging from 0.0 to 1.0
        :param font: Font used by the axes label
        """
        super().__init__(marginRatios, foregroundColor, gridColor, font)
        self._font = font

        self.axes.append(
            PlotAxis(
                self,
                tickLength=(0.0, -5.0),
                foregroundColor=self._foregroundColor,
                labelAlign=CENTER,
                labelVAlign=TOP,
                orderOffsetAlign=RIGHT,
                orderOffsetVAlign=TOP,
                titleAlign=CENTER,
                titleVAlign=TOP,
                titleRotate=0,
                font=self._font,
            )
        )

        self._x2AxisCoords = ()

        self.axes.append(
            PlotAxis(
                self,
                tickLength=(5.0, 0.0),
                foregroundColor=self._foregroundColor,
                labelAlign=RIGHT,
                labelVAlign=CENTER,
                orderOffsetAlign=RIGHT,
                orderOffsetVAlign=BOTTOM,
                titleAlign=CENTER,
                titleVAlign=BOTTOM,
                titleRotate=ROTATE_270,
                font=self._font,
            )
        )

        self._y2Axis = PlotAxis(
            self,
            tickLength=(-5.0, 0.0),
            foregroundColor=self._foregroundColor,
            labelAlign=LEFT,
            labelVAlign=CENTER,
            orderOffsetAlign=LEFT,
            orderOffsetVAlign=BOTTOM,
            titleAlign=CENTER,
            titleVAlign=TOP,
            titleRotate=ROTATE_270,
            font=self._font,
        )

        self._isXAxisInverted = False
        self._isYAxisInverted = False

        self._dataRanges = {"x": (1.0, 100.0), "y": (1.0, 100.0), "y2": (1.0, 100.0)}

        self._baseVectors = (1.0, 0.0), (0.0, 1.0)

        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    def _dirty(self):
        super()._dirty()
        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return (
            super().isDirty
            or self._transformedDataRanges is None
            or self._transformedDataProjMat is None
            or self._transformedDataY2ProjMat is None
        )

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
    def isYAxisInverted(self) -> bool:
        """Whether Y axes are inverted or not as a bool."""
        return self._isYAxisInverted

    @isYAxisInverted.setter
    def isYAxisInverted(self, value: bool):
        value = bool(value)
        if value != self._isYAxisInverted:
            self._isYAxisInverted = value
            self._dirty()

    @property
    def isXAxisInverted(self) -> bool:
        return self._isXAxisInverted

    @isXAxisInverted.setter
    def isXAxisInverted(self, value: bool):
        value = bool(value)
        if value != self._isXAxisInverted:
            self._isXAxisInverted = value
            self._dirty()

    DEFAULT_BASE_VECTORS = (1.0, 0.0), (0.0, 1.0)
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

        det = vectors[0][0] * vectors[1][1] - vectors[1][0] * vectors[0][1]
        if det == 0.0:
            raise ValueError("Singular matrix for base vectors: " + str(vectors))

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
        return self._DataRanges(
            self._dataRanges["x"], self._dataRanges["y"], self._dataRanges["y2"]
        )

    def setDataRanges(self, x=None, y=None, y2=None):
        """Set data range over each axes.

        The provided ranges are clipped to possible values
        (i.e., 32 float range + positive range for log scale).

        :param x: (min, max) data range over X axis
        :param y: (min, max) data range over Y axis
        :param y2: (min, max) data range over Y2 axis
        """
        if x is not None:
            self._dataRanges["x"] = checkAxisLimits(
                self.xAxis.scale, x[0], x[1], name="x"
            )

        if y is not None:
            self._dataRanges["y"] = checkAxisLimits(
                self.xAxis.scale, y[0], y[1], name="y"
            )

        if y2 is not None:
            self._dataRanges["y2"] = checkAxisLimits(
                self.xAxis.scale, y2[0], y2[1], name="y2"
            )

        self.xAxis.dataRange = self._dataRanges["x"]
        self.yAxis.dataRange = self._dataRanges["y"]
        self.y2Axis.dataRange = self._dataRanges["y2"]

    _DataRanges = namedtuple("dataRanges", ("x", "y", "y2"))

    @property
    def transformedDataRanges(self):
        """Bounds of the displayed area in transformed data coordinates
        (i.e., log scale applied if any as well as skew)

        3-tuple of 2-tuple (min, max) for each axis: x, y, y2.
        """
        if self._transformedDataRanges is None:
            xRange, yRange, y2Range = self.dataRanges
            scaledRanges = []
            for range_, axis in [
                (xRange, self.xAxis),
                (yRange, self.yAxis),
                (y2Range, self.y2Axis),
            ]:
                scaledRange = axis.applyScale(range_)
                scaledRange[~numpy.isfinite(scaledRange)] = 0.0
                scaledRanges.append(tuple(scaledRange))

            self._transformedDataRanges = self._DataRanges(*scaledRanges)

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
                yMax, yMin = yMin, yMax

            if self.isXAxisInverted:
                xMax, xMin = xMin, xMax

            self._transformedDataProjMat = mat4Ortho(xMin, xMax, yMin, yMax, 1, -1)

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
                y2Max, y2Min = y2Min, y2Max

            if self.isXAxisInverted:
                xMax, xMin = xMin, xMax

            self._transformedDataY2ProjMat = mat4Ortho(xMin, xMax, y2Min, y2Max, 1, -1)

        return self._transformedDataY2ProjMat

    def dataToPixel(self, x, y, axis="left"):
        """Convert data coordinate to widget pixel coordinate."""
        assert axis in ("left", "right")

        trBounds = self.transformedDataRanges

        xDataTr = self.xAxis.applyScale(x)
        if isinstance(xDataTr, numbers.Real) and not numpy.isfinite(xDataTr):
            return None

        yDataTr = self.yAxis.applyScale(y)
        if isinstance(yDataTr, numbers.Real) and not numpy.isfinite(yDataTr):
            return None

        # Non-orthogonal axes
        if self.baseVectors != self.DEFAULT_BASE_VECTORS:
            (xx, xy), (yx, yy) = self.baseVectors
            skew_mat = numpy.array(((xx, yx), (xy, yy)))

            coords = numpy.dot(skew_mat, numpy.array((xDataTr, yDataTr)))
            xDataTr, yDataTr = coords

        plotWidth, plotHeight = self.plotSize

        xOffset = (
            plotWidth * (xDataTr - trBounds.x[0]) / (trBounds.x[1] - trBounds.x[0])
        )
        if self.isXAxisInverted:
            xPixel = self.size[0] - self.margins.right - xOffset
        else:
            xPixel = self.margins.left + xOffset

        usedAxis = trBounds.y if axis == "left" else trBounds.y2
        yOffset = plotHeight * (yDataTr - usedAxis[0]) / (usedAxis[1] - usedAxis[0])

        if self.isYAxisInverted:
            yPixel = self.margins.top + yOffset
        else:
            yPixel = self.size[1] - self.margins.bottom - yOffset

        return (
            (
                int(xPixel)
                if isinstance(xPixel, numbers.Real)
                else xPixel.astype(numpy.int64)
            ),
            (
                int(yPixel)
                if isinstance(yPixel, numbers.Real)
                else yPixel.astype(numpy.int64)
            ),
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

        if self.isXAxisInverted:
            xPlotPixel = self.size[0] - self.margins.right - x - 0.5
        else:
            xPlotPixel = x - self.margins.left + 0.5
        xScaledData = trBounds.x[0] + xPlotPixel / float(plotWidth) * (
            trBounds.x[1] - trBounds.x[0]
        )

        if self.isYAxisInverted:
            yPlotPixel = y - self.margins.top + 0.5
        else:
            yPlotPixel = self.size[1] - self.margins.bottom - y - 0.5
        usedAxis = trBounds.y if axis == "left" else trBounds.y2
        yScaledData = usedAxis[0] + yPlotPixel / float(plotHeight) * (
            usedAxis[1] - usedAxis[0]
        )

        # non-orthogonal axis
        if self.baseVectors != self.DEFAULT_BASE_VECTORS:
            (xx, xy), (yx, yy) = self.baseVectors
            skew_mat = numpy.array(((xx, yx), (xy, yy)))
            skew_mat = numpy.linalg.inv(skew_mat)

            coords = numpy.dot(skew_mat, numpy.array((xScaledData, yScaledData)))
            xScaledData, yScaledData = coords

        xData = self.xAxis.revertScale(xScaledData)
        yData = self.yAxis.revertScale(yScaledData)
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
                            vertices.append((self.size[0] - self.margins.right, yPixel))
                        else:  # axis == self.y2Axis
                            vertices.append((self.margins.left, yPixel))

        else:
            # Get plot corners in data coords
            plotLeft, plotTop = self.plotOrigin
            plotWidth, plotHeight = self.plotSize

            corners = [
                (plotLeft, plotTop),
                (plotLeft, plotTop + plotHeight),
                (plotLeft + plotWidth, plotTop + plotHeight),
                (plotLeft + plotWidth, plotTop),
            ]

            for axis in self.axes:
                if axis == self.xAxis:
                    cornersInData = numpy.array(
                        [self.pixelToData(x, y) for (x, y) in corners]
                    )
                    borders = (
                        (cornersInData[0], cornersInData[3]),  # top
                        (cornersInData[1], cornersInData[0]),  # left
                        (cornersInData[3], cornersInData[2]),
                    )  # right

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if min(x0, x1) <= data < max(x0, x1):
                                    yIntersect = (data - x0) * (y1 - y0) / (
                                        x1 - x0
                                    ) + y0

                                    pixelPos = self.dataToPixel(data, yIntersect)
                                    if pixelPos is not None:
                                        vertices.append((xPixel, yPixel))
                                        vertices.append(pixelPos)
                                    break  # Stop at first intersection

                else:  # y or y2 axes
                    if axis == self.yAxis:
                        axis_name = "left"
                        cornersInData = numpy.array(
                            [self.pixelToData(x, y) for (x, y) in corners]
                        )
                        borders = (
                            (cornersInData[3], cornersInData[2]),  # right
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]),
                        )  # bottom

                    else:  # axis == self.y2Axis
                        axis_name = "right"
                        corners = numpy.array(
                            [self.pixelToData(x, y, axis="right") for (x, y) in corners]
                        )
                        borders = (
                            (cornersInData[1], cornersInData[0]),  # left
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]),
                        )  # bottom

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if min(y0, y1) <= data < max(y0, y1):
                                    xIntersect = (data - y0) * (x1 - x0) / (
                                        y1 - y0
                                    ) + x0

                                    pixelPos = self.dataToPixel(
                                        xIntersect, data, axis=axis_name
                                    )
                                    if pixelPos is not None:
                                        vertices.append((xPixel, yPixel))
                                        vertices.append(pixelPos)
                                    break  # Stop at first intersection

        return vertices

    def _buildVerticesAndLabels(self):
        width, height = self.size

        xLeft = self.margins.left - 0.5
        xRight = width - self.margins.right + 0.5
        yBottom = height - self.margins.bottom + 0.5
        yTop = self.margins.top - 0.5

        self._x2AxisCoords = ((xLeft, yTop), (xRight, yTop))

        # Set order&offset anchor **before** handling axis inversion
        fontPixelSize = self._font.pixelSize()
        if fontPixelSize == -1:
            fontPixelSize = self._font.pointSizeF() / 72.0 * self.dotsPerInch

        self.axes[0].orderOffsetAnchor = (
            xRight,
            yBottom + fontPixelSize * 1.2,
        )
        self.axes[1].orderOffsetAnchor = (
            xLeft,
            yTop - 4 * self.devicePixelRatio - fontPixelSize / 2.0,
        )
        self._y2Axis.orderOffsetAnchor = (
            xRight,
            yTop - 4 * self.devicePixelRatio - fontPixelSize / 2.0,
        )

        if self.isYAxisInverted:
            # Y axis is inverted: goes top to bottom
            yCoords = yTop, yBottom
        else:
            yCoords = yBottom, yTop

        if self.isXAxisInverted:
            # X axis is inverted: goes right to left
            xCoords = xRight, xLeft
        else:
            xCoords = xLeft, xRight

        self.axes[0].displayCoords = (
            (xCoords[0], yBottom),
            (xCoords[1], yBottom),
        )

        self.axes[1].displayCoords = (
            (xLeft, yCoords[0]),
            (xLeft, yCoords[1]),
        )

        self._y2Axis.displayCoords = (
            (xRight, yCoords[0]),
            (xRight, yCoords[1]),
        )

        super()._buildVerticesAndLabels()

        vertices, gridVertices, labels = self._renderResources

        # Adds vertices for borders without axis
        extraVertices = []
        extraVertices += self._x2AxisCoords
        if not self.isY2Axis:
            extraVertices += self._y2Axis.displayCoords

        extraVertices = numpy.asarray(extraVertices, dtype=numpy.float32)
        vertices = numpy.append(vertices, extraVertices, axis=0)

        self._renderResources = (vertices, gridVertices, labels)

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
            self._y2Axis.foregroundColor = color
            GLPlotFrame.foregroundColor.fset(self, color)  # call parent property
