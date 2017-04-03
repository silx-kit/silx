# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
from __future__ import with_statement

__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This modules provides the rendering of plot titles, axes and grid.
"""

# TODO
# keep aspect ratio managed here?
# smarter dirty flag handling?


# import ######################################################################

import logging
import numpy as np
import math
import weakref
import warnings

from collections import namedtuple

from .gl import *  # noqa
from .GLSupport import mat4Ortho, clamp, \
    FLOAT32_SAFE_MIN, FLOAT32_MINPOS, FLOAT32_SAFE_MAX
from .GLProgram import GLProgram
from .GLText import Text2D, CENTER, BOTTOM, TOP, LEFT, RIGHT, ROTATE_270
from .LabelLayout import niceNumbersAdaptative, niceNumbersForLog10


# PlotAxis ####################################################################

class PlotAxis(object):
    """Represents a 1D axis of the plot.
    This class is intended to be used with :class:`GLPlotFrame`.
    """

    def __init__(self, plot,
                 tickLength=(0., 0.),
                 labelAlign=CENTER, labelVAlign=CENTER,
                 titleAlign=CENTER, titleVAlign=CENTER,
                 titleRotate=0, titleOffset=(0., 0.)):
        self._ticks = None

        self._plot = weakref.ref(plot)

        self._isLog = False
        self._dataRange = 1., 100.
        self._displayCoords = (0., 0.), (1., 0.)
        self._title = ''

        self._tickLength = tickLength
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
    def title(self):
        """The text label associated with this axis as a str in latin-1."""
        return self._title

    @title.setter
    def title(self, title):
        if title != self._title:
            self._title = title

            plot = self._plot()
            if plot is not None:
                plot._dirty()

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
        for (xPixel, yPixel), dataPos, text in self.ticks:
            if text is None:
                tickScale = 0.5
            else:
                tickScale = 1.

                label = Text2D(text=text,
                               x=xPixel - xTickLength,
                               y=yPixel - yTickLength,
                               align=self._labelAlign,
                               valign=self._labelVAlign)

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

        xOffset, yOffset = self._titleOffset

        # Adaptative title positioning:
        # tickNorm = math.sqrt(xTickLength ** 2 + yTickLength ** 2)
        # xOffset = -tickLabelsSize[0] * xTickLength / tickNorm
        # xOffset -= 3 * xTickLength
        # yOffset = -tickLabelsSize[1] * yTickLength / tickNorm
        # yOffset -= 3 * yTickLength

        axisTitle = Text2D(text=self.title,
                           x=xAxisCenter + xOffset,
                           y=yAxisCenter + yOffset,
                           align=self._titleAlign,
                           valign=self._titleVAlign,
                           rotate=self._titleRotate)
        labels.append(axisTitle)

        return vertices, labels

    def _dirtyTicks(self):
        """Mark ticks as dirty and notify listener (i.e., background)."""
        self._ticks = None
        plot = self._plot()
        if plot is not None:
            plot._dirty()

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
            warnings.warn(
                'Getting ticks while isLog=True and dataRange[0]<=0.',
                RuntimeWarning)
            dataMin = 1.
            if dataMax < dataMin:
                dataMax = 1.

        if dataMin != dataMax:  # data range is not null
            (x0, y0), (x1, y1) = self.displayCoords

            if self.isLog:
                logMin, logMax = math.log10(dataMin), math.log10(dataMax)
                tickMin, tickMax, step = niceNumbersForLog10(logMin, logMax)

                xScale = (x1 - x0) / (logMax - logMin)
                yScale = (y1 - y0) / (logMax - logMin)

                for logPos in self._frange(tickMin, tickMax, step):
                    if logPos >= logMin and logPos <= logMax:
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
                            if dataPos >= dataMin and dataPos <= dataMax:
                                logSubPos = math.log10(dataPos)
                                xPixel = x0 + (logSubPos - logMin) * xScale
                                yPixel = y0 + (logSubPos - logMin) * yScale
                                yield ((xPixel, yPixel), dataPos, None)

            else:
                xScale = (x1 - x0) / (dataMax - dataMin)
                yScale = (y1 - y0) / (dataMax - dataMin)

                nbPixels = math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))

                # Density of 1.3 label per 92 pixels
                # i.e., 1.3 label per inch on a 92 dpi screen
                tickMin, tickMax, step, nbFrac = niceNumbersAdaptative(
                    dataMin, dataMax, nbPixels, 1.3 / 92)

                for dataPos in self._frange(tickMin, tickMax, step):
                    if dataPos >= dataMin and dataPos <= dataMax:
                        xPixel = x0 + (dataPos - dataMin) * xScale
                        yPixel = y0 + (dataPos - dataMin) * yScale

                        if nbFrac == 0:
                            text = '%g' % dataPos
                        else:
                            text = ('%.' + str(nbFrac) + 'f') % dataPos
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

    def __init__(self, margins):
        """
        :param margins: The margins around plot area for axis and labels.
        :type margins: dict with 'left', 'right', 'top', 'bottom' keys and
                       values as ints.
        """
        self._renderResources = None

        self._margins = self._Margins(**margins)

        self.axes = []  # List of PlotAxis to be updated by subclasses

        self._grid = False
        self._size = 0., 0.
        self._title = ''
        self._baseVectors = (1., 0.), (0., 1.)
        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return (self._renderResources is None or
                self._transformedDataRanges is None or
                self._transformedDataProjMat is None or
                self._transformedDataY2ProjMat is None)

    GRID_NONE = 0
    GRID_MAIN_TICKS = 1
    GRID_SUB_TICKS = 2
    GRID_ALL_TICKS = (GRID_MAIN_TICKS + GRID_SUB_TICKS)

    @property
    def margins(self):
        """Margins in pixels around the plot."""
        return self._margins

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
        """Size in pixels of the plot area including margins."""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 2
        size = tuple(size)
        if size != self._size:
            self._size = size
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
                    print('xMin: warning log10({0})'.format(xMin))
                    xMin = 0.
                try:
                    xMax = math.log10(xMax)
                except ValueError:
                    print('xMax: warning log10({0})'.format(xMax))
                    xMax = 0.

            if self.yAxis.isLog:
                try:
                    yMin = math.log10(yMin)
                except ValueError:
                    print('yMin: warning log10({0})'.format(yMin))
                    yMin = 0.
                try:
                    yMax = math.log10(yMax)
                except ValueError:
                    print('yMax: warning log10({0})'.format(yMax))
                    yMax = 0.

                try:
                    y2Min = math.log10(y2Min)
                except ValueError:
                    print('yMin: warning log10({0})'.format(y2Min))
                    y2Min = 0.
                try:
                    y2Max = math.log10(y2Max)
                except ValueError:
                    print('yMax: warning log10({0})'.format(y2Max))
                    y2Max = 0.

            # Non-orthogonal axes
            if self.baseVectors != self.DEFAULT_BASE_VECTORS:
                (xx, xy), (yx, yy) = self.baseVectors
                skew_mat = np.array(((xx, yx), (xy, yy)))

                corners = [(xMin, yMin), (xMin, yMax),
                           (xMax, yMin), (xMax, yMax),
                           (xMin, y2Min), (xMin, y2Max),
                           (xMax, y2Min), (xMax, y2Max)]

                corners = np.array(
                    [np.dot(skew_mat, corner) for corner in corners],
                    dtype=np.float32)
                xMin, xMax = corners[:, 0].min(),  corners[:, 0].max()
                yMin, yMax = corners[0:4, 1].min(), corners[0:4, 1].max()
                y2Min, y2Max = corners[4:, 1].min(), corners[4:, 1].max()

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

            # Non-orthogonal axes
            if self.baseVectors != self.DEFAULT_BASE_VECTORS:
                (xx, xy), (yx, yy) = self.baseVectors
                mat = mat * np.matrix((
                    (xx, yx, 0., 0.),
                    (xy, yy, 0., 0.),
                    (0., 0., 1., 0.),
                    (0., 0., 0., 1.)), dtype=np.float32)

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

            # Non-orthogonal axes
            if self.baseVectors != self.DEFAULT_BASE_VECTORS:
                (xx, xy), (yx, yy) = self.baseVectors
                mat = mat * np.matrix((
                    (xx, yx, 0., 0.),
                    (xy, yy, 0., 0.),
                    (0., 0., 1., 0.),
                    (0., 0., 0., 1.)), dtype=np.float32)

            self._transformedDataY2ProjMat = mat

        return self._transformedDataY2ProjMat

    def dataToPixel(self, x, y, axis='left'):
        """Convert data coordinate to widget pixel coordinate.
        """
        assert axis in ('left', 'right')

        trBounds = self.transformedDataRanges

        if self.xAxis.isLog:
            if x < FLOAT32_MINPOS:
                return None
            xDataTr = math.log10(x)
        else:
            xDataTr = x

        if self.yAxis.isLog:
            if y < FLOAT32_MINPOS:
                return None
            yDataTr = math.log10(y)
        else:
            yDataTr = y

        # Non-orthogonal axes
        if self.baseVectors != self.DEFAULT_BASE_VECTORS:
            (xx, xy), (yx, yy) = self.baseVectors
            skew_mat = np.array(((xx, yx), (xy, yy)))

            coords = np.dot(skew_mat, np.array((xDataTr, yDataTr)))
            xDataTr, yDataTr = coords

        plotWidth, plotHeight = self.plotSize

        xPixel = int(self.margins.left +
                     plotWidth * (xDataTr - trBounds.x[0]) /
                     (trBounds.x[1] - trBounds.x[0]))

        usedAxis = trBounds.y if axis == "left" else trBounds.y2
        yOffset = (plotHeight * (yDataTr - usedAxis[0]) /
                (usedAxis[1] - usedAxis[0]))

        if self.isYAxisInverted:
            yPixel = int(self.margins.top + yOffset)
        else:
            yPixel = int(self.size[1] - self.margins.bottom - yOffset)

        return xPixel, yPixel

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
            skew_mat = np.array(((xx, yx), (xy, yy)))
            skew_mat = np.linalg.inv(skew_mat)

            coords = np.dot(skew_mat, np.array((xData, yData)))
            xData, yData = coords

        if self.xAxis.isLog:
            xData = pow(10, xData)
        if self.yAxis.isLog:
            yData = pow(10, yData)

        return xData, yData

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
        self._transformedDataRanges = None
        self._transformedDataProjMat = None
        self._transformedDataY2ProjMat = None

    def _updateAxes(self):
        """Override in subclass to update PlotAxis in axes."""
        pass

    def _buildGridVertices(self):
        if self._grid == self.GRID_NONE:
            return []

        elif self._grid == self.GRID_MAIN_TICKS:
            test = lambda text: text is not None
        elif self._grid == self.GRID_SUB_TICKS:
            test = lambda text: text is None
        elif self._grid == self.GRID_ALL_TICKS:
            test = lambda text: True
        else:
            logging.warning('Wrong grid mode: %d' % self._grid)
            return []

        return self._buildGridVerticesWithTest(test)

    def _buildGridVerticesWithTest(self, test):
        """Override in subclass to generate grid vertices"""
        return []

    def _buildVerticesAndLabels(self):
        self._updateAxes()

        # To fill with copy of axes lists
        vertices = []
        labels = []

        for axis in self.axes:
            axisVertices, axisLabels = axis.getVerticesAndLabels()
            vertices += axisVertices
            labels += axisLabels

        vertices = np.array(vertices, dtype=np.float32)

        # Add main title
        xTitle = (self.size[0] + self.margins.left -
                  self.margins.right) // 2
        yTitle = self.margins.top - self._TICK_LENGTH_IN_PIXELS
        labels.append(Text2D(text=self.title,
                             x=xTitle,
                             y=yTitle,
                             align=CENTER,
                             valign=BOTTOM))

        # grid
        gridVertices = np.array(self._buildGridVertices(), dtype=np.float32)

        self._renderResources = (vertices, gridVertices, labels)

    _program = GLProgram(_SHADERS['vertex'], _SHADERS['fragment'])

    def render(self):
        if self._renderResources is None:
            self._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        width, height = self.size
        matProj = mat4Ortho(0, width, height, 0, 1, -1)

        glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        glLineWidth(self._LINE_WIDTH)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matProj)
        glUniform4f(prog.uniforms['color'], 0., 0., 0., 1.)
        glUniform1f(prog.uniforms['tickFactor'], 0.)

        glEnableVertexAttribArray(prog.attributes['position'])
        glVertexAttribPointer(prog.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, vertices)

        glDrawArrays(GL_LINES, 0, len(vertices))

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

        glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        glLineWidth(self._LINE_WIDTH)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matProj)
        glUniform4f(prog.uniforms['color'], 0.7, 0.7, 0.7, 1.)
        glUniform1f(prog.uniforms['tickFactor'], 0.) #1/2.)  # 1/tickLen

        glEnableVertexAttribArray(prog.attributes['position'])
        glVertexAttribPointer(prog.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, gridVertices)

        glDrawArrays(GL_LINES, 0, len(gridVertices))


# GLPlotFrame2D ###############################################################

class GLPlotFrame2D(GLPlotFrame):
    def __init__(self, margins):
        """
        :param margins: The margins around plot area for axis and labels.
        :type margins: dict with 'left', 'right', 'top', 'bottom' keys and
                       values as ints.
        """
        super(GLPlotFrame2D, self).__init__(margins)
        self.axes.append(PlotAxis(self,
                                  tickLength=(0., -5.),
                                  labelAlign=CENTER, labelVAlign=TOP,
                                  titleAlign=CENTER, titleVAlign=TOP,
                                  titleRotate=0,
                                  titleOffset=(0, self.margins.bottom // 2)))

        self._x2AxisCoords = ()

        self.axes.append(PlotAxis(self,
                                  tickLength=(5., 0.),
                                  labelAlign=RIGHT, labelVAlign=CENTER,
                                  titleAlign=CENTER, titleVAlign=BOTTOM,
                                  titleRotate=ROTATE_270,
                                  titleOffset=(-3 * self.margins.left // 4,
                                               0)))

        self._y2Axis = PlotAxis(self,
                                tickLength=(-5., 0.),
                                labelAlign=LEFT, labelVAlign=CENTER,
                                titleAlign=CENTER, titleVAlign=TOP,
                                titleRotate=ROTATE_270,
                                titleOffset=(3 * self.margins.right // 4,
                                             0))

        self._isYAxisInverted = False

        self._dataRanges = {
            'x': (1., 100.), 'y': (1., 100.), 'y2': (1., 100.)}

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

    @property
    def dataRanges(self):
        """Ranges of data visible in the plot on x, y and y2 axes.

        This is different to the axes range when axes are not orthogonal.

        Type: ((xMin, xMax), (yMin, yMax), (y2Min, y2Max))
        """
        return self._DataRanges(self._dataRanges['x'],
                                self._dataRanges['y'],
                                self._dataRanges['y2'])

    @staticmethod
    def _clipToSafeRange(min_, max_, isLog):
        # Clip range if needed
        minLimit = FLOAT32_MINPOS if isLog else FLOAT32_SAFE_MIN
        min_ = clamp(min_, minLimit, FLOAT32_SAFE_MAX)
        max_ = clamp(max_, minLimit, FLOAT32_SAFE_MAX)
        assert min_ < max_
        return min_, max_

    def setDataRanges(self, x=None, y=None, y2=None):
        """Set data range over each axes.

        The provided ranges are clipped to possible values
        (i.e., 32 float range + positive range for log scale).

        :param x: (min, max) data range over X axis
        :param y: (min, max) data range over Y axis
        :param y2: (min, max) data range over Y2 axis
        """
        if x is not None:
            self._dataRanges['x'] = \
                self._clipToSafeRange(x[0], x[1], self.xAxis.isLog)

        if y is not None:
            self._dataRanges['y'] = \
                self._clipToSafeRange(y[0], y[1], self.yAxis.isLog)

        if y2 is not None:
            self._dataRanges['y2'] = \
                self._clipToSafeRange(y2[0], y2[1], self.y2Axis.isLog)

        self.xAxis.dataRange = self._dataRanges['x']
        self.yAxis.dataRange = self._dataRanges['y']
        self.y2Axis.dataRange = self._dataRanges['y2']

    def _updateAxes(self):
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
                    cornersInData = np.array([
                        self.pixelToData(x, y) for (x, y) in corners])
                    borders = ((cornersInData[0], cornersInData[3]),  # top
                               (cornersInData[1], cornersInData[0]),  # left
                               (cornersInData[3], cornersInData[2]))  # right

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if data >= min(x0, x1) and data < max(x0, x1):
                                    yIntersect = (data - x0) * \
                                        (y1 - y0) / (x1 - x0) + y0

                                    pixelPos = self.dataToPixel(
                                        data, yIntersect)
                                    if pixelPos is not None:
                                        vertices.append((xPixel, yPixel))
                                        vertices.append(pixelPos)
                                    break  # Stop at first intersection

                else: # y or y2 axes
                    if axis == self.yAxis:
                        axis_name = 'left'
                        cornersInData = np.array([
                            self.pixelToData(x, y) for (x, y) in corners])
                        borders = (
                            (cornersInData[3], cornersInData[2]),  # right
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]))  # bottom

                    else:  # axis == self.y2Axis
                        axis_name = 'right'
                        corners = np.array([self.pixelToData(
                            x, y, axis='right') for (x, y) in corners])
                        borders = (
                            (cornersInData[1], cornersInData[0]),  # left
                            (cornersInData[0], cornersInData[3]),  # top
                            (cornersInData[2], cornersInData[1]))  # bottom

                    for (xPixel, yPixel), data, text in axis.ticks:
                        if test(text):
                            for (x0, y0), (x1, y1) in borders:
                                if data >= min(y0, y1) and data < max(y0, y1):
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
        super(GLPlotFrame2D, self)._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        # Adds vertices for borders without axis
        extraVertices = []
        extraVertices += self._x2AxisCoords
        if not self.isY2Axis:
            extraVertices += self._y2Axis.displayCoords

        extraVertices = np.array(extraVertices, copy=False, dtype=np.float32)
        vertices = np.append(vertices, extraVertices, axis=0)

        self._renderResources = (vertices, gridVertices, labels)
