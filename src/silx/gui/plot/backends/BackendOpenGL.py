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
"""OpenGL Plot backend."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "21/12/2018"

import logging
import weakref

import numpy

from .. import items
from .._utils import FLOAT32_MINPOS
from . import BackendBase
from ... import colors
from ... import qt

from ..._glutils import gl
from ... import _glutils as glu
from . import glutils
from .glutils.PlotImageFile import saveImageToFile

_logger = logging.getLogger(__name__)


# TODO idea: BackendQtMixIn class to share code between mpl and gl
# TODO check if OpenGL is available
# TODO make an off-screen mesa backend

# Content #####################################################################

class _ShapeItem(dict):
    def __init__(self, x, y, shape, color, fill, overlay,
                 linestyle, linewidth, linebgcolor):
        super(_ShapeItem, self).__init__()

        if shape not in ('polygon', 'rectangle', 'line',
                         'vline', 'hline', 'polylines'):
            raise NotImplementedError("Unsupported shape {0}".format(shape))

        x = numpy.array(x, copy=False)
        y = numpy.array(y, copy=False)

        if shape == 'rectangle':
            xMin, xMax = x
            x = numpy.array((xMin, xMin, xMax, xMax))
            yMin, yMax = y
            y = numpy.array((yMin, yMax, yMax, yMin))

        # Ignore fill for polylines to mimic matplotlib
        fill = fill if shape != 'polylines' else False

        self.update({
            'shape': shape,
            'color': colors.rgba(color),
            'fill': 'hatch' if fill else None,
            'x': x,
            'y': y,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'linebgcolor': linebgcolor,
        })


class _MarkerItem(dict):
    def __init__(self, x, y, text, color,
                 symbol, linestyle, linewidth, constraint, yaxis):
        super(_MarkerItem, self).__init__()

        if symbol is None:
            symbol = '+'

        # Apply constraint to provided position
        isConstraint = (constraint is not None and
                        x is not None and y is not None)
        if isConstraint:
            x, y = constraint(x, y)

        self.update({
            'x': x,
            'y': y,
            'text': text,
            'color': colors.rgba(color),
            'constraint': constraint if isConstraint else None,
            'symbol': symbol,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'yaxis': yaxis,
        })


# shaders #####################################################################

_baseVertShd = """
    attribute vec2 position;
    uniform mat4 matrix;
    uniform bvec2 isLog;

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        vec2 posTransformed = position;
        if (isLog.x) {
            posTransformed.x = oneOverLog10 * log(position.x);
        }
        if (isLog.y) {
            posTransformed.y = oneOverLog10 * log(position.y);
        }
        gl_Position = matrix * vec4(posTransformed, 0.0, 1.0);
    }
    """

_baseFragShd = """
    uniform vec4 color;
    uniform int hatchStep;
    uniform float tickLen;

    void main(void) {
        if (tickLen != 0.) {
            if (mod((gl_FragCoord.x + gl_FragCoord.y) / tickLen, 2.) < 1.) {
                gl_FragColor = color;
            } else {
                discard;
            }
        } else if (hatchStep == 0 ||
            mod(gl_FragCoord.x - gl_FragCoord.y, float(hatchStep)) == 0.) {
            gl_FragColor = color;
        } else {
            discard;
        }
    }
    """

_texVertShd = """
   attribute vec2 position;
   attribute vec2 texCoords;
   uniform mat4 matrix;

   varying vec2 coords;

   void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        coords = texCoords;
   }
   """

_texFragShd = """
    uniform sampler2D tex;

    varying vec2 coords;

    void main(void) {
        gl_FragColor = texture2D(tex, coords);
        gl_FragColor.a = 1.0;
    }
    """

# BackendOpenGL ###############################################################


class BackendOpenGL(BackendBase.BackendBase, glu.OpenGLWidget):
    """OpenGL-based Plot backend.

    WARNINGS:
    Unless stated otherwise, this API is NOT thread-safe and MUST be
    called from the main thread.
    When numpy arrays are passed as arguments to the API (through
    :func:`addCurve` and :func:`addImage`), they are copied only if
    required.
    So, the caller should not modify these arrays afterwards.
    """

    def __init__(self, plot, parent=None, f=qt.Qt.Widget):
        glu.OpenGLWidget.__init__(self, parent,
                                  alphaBufferSize=8,
                                  depthBufferSize=0,
                                  stencilBufferSize=0,
                                  version=(2, 1),
                                  f=f)
        BackendBase.BackendBase.__init__(self, plot, parent)

        self.__isOpenGLValid = False

        self._backgroundColor = 1., 1., 1., 1.
        self._dataBackgroundColor = 1., 1., 1., 1.

        self.matScreenProj = glutils.mat4Identity()

        self._progBase = glu.Program(
            _baseVertShd, _baseFragShd, attrib0='position')
        self._progTex = glu.Program(
            _texVertShd, _texFragShd, attrib0='position')
        self._plotFBOs = weakref.WeakKeyDictionary()

        self._keepDataAspectRatio = False

        self._crosshairCursor = None
        self._mousePosInPixels = None

        self._glGarbageCollector = []

        self._plotFrame = glutils.GLPlotFrame2D(
            foregroundColor=(0., 0., 0., 1.),
            gridColor=(.7, .7, .7, 1.),
            marginRatios=(.15, .1, .1, .15))
        self._plotFrame.size = (  # Init size with size int
            int(self.getDevicePixelRatio() * 640),
            int(self.getDevicePixelRatio() * 480))

        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    # QWidget

    _MOUSE_BTNS = {
        qt.Qt.LeftButton: 'left',
        qt.Qt.RightButton: 'right',
        qt.Qt.MiddleButton: 'middle',
    }

    def sizeHint(self):
        return qt.QSize(8 * 80, 6 * 80)  # Mimic MatplotlibBackend

    def mousePressEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super(BackendOpenGL, self).mousePressEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMousePress(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def mouseMoveEvent(self, event):
        qtPos = qt.getMouseEventPosition(event)

        previousMousePosInPixels = self._mousePosInPixels
        if qtPos == self._mouseInPlotArea(*qtPos):
            devicePixelRatio = self.getDevicePixelRatio()
            devicePos = qtPos[0] * devicePixelRatio, qtPos[1] * devicePixelRatio
            self._mousePosInPixels = devicePos  # Mouse in plot area
        else:
            self._mousePosInPixels = None  # Mouse outside plot area

        if (self._crosshairCursor is not None and
                previousMousePosInPixels != self._mousePosInPixels):
            # Avoid replot when cursor remains outside plot area
            self._plot._setDirtyPlot(overlayOnly=True)

        self._plot.onMouseMove(*qtPos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() not in self._MOUSE_BTNS:
            return super(BackendOpenGL, self).mouseReleaseEvent(event)
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseRelease(x, y, self._MOUSE_BTNS[event.button()])
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        angleInDegrees = delta / 8.
        x, y = qt.getMouseEventPosition(event)
        self._plot.onMouseWheel(x, y, angleInDegrees)
        event.accept()

    def leaveEvent(self, _):
        self._plot.onMouseLeaveWidget()

    # OpenGLWidget API

    def initializeGL(self):
        self.__isOpenGLValid = gl.testGL()
        if not self.__isOpenGLValid:
            return

        gl.glClearStencil(0)

        gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA,
                               gl.GL_ONE_MINUS_SRC_ALPHA,
                               gl.GL_ONE,
                               gl.GL_ONE)

        # For lines
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        # For points
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    def _paintDirectGL(self):
        self._renderPlotAreaGL()
        self._plotFrame.render()
        self._renderOverlayGL()

    def _paintFBOGL(self):
        context = glu.Context.getCurrent()
        plotFBOTex = self._plotFBOs.get(context)
        if (self._plot._getDirtyPlot() or self._plotFrame.isDirty or
                plotFBOTex is None):
            self._plotVertices = (
                # Vertex coordinates
                numpy.array(((-1., -1.), (1., -1.), (-1., 1.), (1., 1.)),
                             dtype=numpy.float32),
                 # Texture coordinates
                 numpy.array(((0., 0.), (1., 0.), (0., 1.), (1., 1.)),
                             dtype=numpy.float32))
            if plotFBOTex is None or \
               plotFBOTex.shape[1] != self._plotFrame.size[0] or \
               plotFBOTex.shape[0] != self._plotFrame.size[1]:
                if plotFBOTex is not None:
                    plotFBOTex.discard()
                plotFBOTex = glu.FramebufferTexture(
                    gl.GL_RGBA,
                    shape=(self._plotFrame.size[1],
                           self._plotFrame.size[0]),
                    minFilter=gl.GL_NEAREST,
                    magFilter=gl.GL_NEAREST,
                    wrap=(gl.GL_CLAMP_TO_EDGE,
                          gl.GL_CLAMP_TO_EDGE))
                self._plotFBOs[context] = plotFBOTex

            with plotFBOTex:
                gl.glClearColor(*self._backgroundColor)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
                self._renderPlotAreaGL()
                self._plotFrame.render()

        # Render plot in screen coords
        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        self._progTex.use()
        texUnit = 0

        gl.glUniform1i(self._progTex.uniforms['tex'], texUnit)
        gl.glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, gl.GL_TRUE,
                              glutils.mat4Identity().astype(numpy.float32))

        gl.glEnableVertexAttribArray(self._progTex.attributes['position'])
        gl.glVertexAttribPointer(self._progTex.attributes['position'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0,
                                 self._plotVertices[0])

        gl.glEnableVertexAttribArray(self._progTex.attributes['texCoords'])
        gl.glVertexAttribPointer(self._progTex.attributes['texCoords'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0,
                                 self._plotVertices[1])

        with plotFBOTex.texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self._plotVertices[0]))

        self._renderOverlayGL()

    def paintGL(self):
        if not self.__isOpenGLValid:
            return

        plot = self._plotRef()
        if plot is None:
            return

        with plot._paintContext():
            with glu.Context.current(self.context()):
                # Release OpenGL resources
                for item in self._glGarbageCollector:
                    item.discard()
                self._glGarbageCollector = []

                gl.glClearColor(*self._backgroundColor)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)

                # Check if window is large enough
                if self._plotFrame.plotSize <= (2, 2):
                    return

                # Sync plot frame with window
                self._plotFrame.devicePixelRatio = self.getDevicePixelRatio()
                # self._paintDirectGL()
                self._paintFBOGL()

    def _renderItems(self, overlay=False):
        """Render items according to :class:`PlotWidget` order

        Note: Scissor test should already be set.

        :param bool overlay:
            False (the default) to render item that are not overlays.
            True to render items that are overlays.
        """
        # Values that are often used
        plotWidth, plotHeight = self._plotFrame.plotSize
        isXLog = self._plotFrame.xAxis.isLog
        isYLog = self._plotFrame.yAxis.isLog
        isYInverted = self._plotFrame.isYAxisInverted

        # Used by marker rendering
        labels = []
        pixelOffset = 3

        context = glutils.RenderContext(
            isXLog=isXLog,
            isYLog=isYLog,
            dpi=self.getDotsPerInch(),
            plotFrame=self._plotFrame,
        )

        for plotItem in self.getItemsFromBackToFront(
                condition=lambda i: i.isVisible() and i.isOverlay() == overlay):
            if plotItem._backendRenderer is None:
                continue

            item = plotItem._backendRenderer

            if isinstance(item, glutils.GLPlotItem):  # Render data items
                gl.glViewport(self._plotFrame.margins.left,
                              self._plotFrame.margins.bottom,
                              plotWidth, plotHeight)
                # Set matrix
                if item.yaxis == 'right':
                    context.matrix = self._plotFrame.transformedDataY2ProjMat
                else:
                    context.matrix = self._plotFrame.transformedDataProjMat
                item.render(context)

            elif isinstance(item, _ShapeItem):  # Render shape items
                gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

                if ((isXLog and numpy.min(item['x']) < FLOAT32_MINPOS) or
                        (isYLog and numpy.min(item['y']) < FLOAT32_MINPOS)):
                    # Ignore items <= 0. on log axes
                    continue

                if item['shape'] == 'hline':
                    width = self._plotFrame.size[0]
                    _, yPixel = self._plotFrame.dataToPixel(
                        0.5 * sum(self._plotFrame.dataRanges[0]),
                        item['y'],
                        axis='left')
                    subShapes = [numpy.array(((0., yPixel), (width, yPixel)),
                                             dtype=numpy.float32)]

                elif item['shape'] == 'vline':
                    xPixel, _ = self._plotFrame.dataToPixel(
                        item['x'],
                        0.5 * sum(self._plotFrame.dataRanges[1]),
                        axis='left')
                    height = self._plotFrame.size[1]
                    subShapes = [numpy.array(((xPixel, 0), (xPixel, height)),
                                             dtype=numpy.float32)]

                else:
                    # Split sub-shapes at not finite values
                    splits = numpy.nonzero(numpy.logical_not(numpy.logical_and(
                        numpy.isfinite(item['x']), numpy.isfinite(item['y']))))[0]
                    splits = numpy.concatenate(([-1], splits, [len(item['x'])]))
                    subShapes = []
                    for begin, end in zip(splits[:-1] + 1, splits[1:]):
                        if end > begin:
                            subShapes.append(numpy.array([
                                self._plotFrame.dataToPixel(x, y, axis='left')
                                for (x, y) in zip(item['x'][begin:end], item['y'][begin:end])]))

                for points in subShapes:  # Draw each sub-shape
                    # Draw the fill
                    if (item['fill'] is not None and
                            item['shape'] not in ('hline', 'vline')):
                        self._progBase.use()
                        gl.glUniformMatrix4fv(
                            self._progBase.uniforms['matrix'], 1, gl.GL_TRUE,
                            self.matScreenProj.astype(numpy.float32))
                        gl.glUniform2i(self._progBase.uniforms['isLog'], False, False)
                        gl.glUniform1f(self._progBase.uniforms['tickLen'], 0.)

                        shape2D = glutils.FilledShape2D(
                            points, style=item['fill'], color=item['color'])
                        shape2D.render(
                            posAttrib=self._progBase.attributes['position'],
                            colorUnif=self._progBase.uniforms['color'],
                            hatchStepUnif=self._progBase.uniforms['hatchStep'])

                    # Draw the stroke
                    if item['linestyle'] not in ('', ' ', None):
                        if item['shape'] != 'polylines':
                            # close the polyline
                            points = numpy.append(points,
                                                  numpy.atleast_2d(points[0]), axis=0)

                        lines = glutils.GLLines2D(
                            points[:, 0], points[:, 1],
                            style=item['linestyle'],
                            color=item['color'],
                            dash2ndColor=item['linebgcolor'],
                            width=item['linewidth'])
                        context.matrix = self.matScreenProj
                        lines.render(context)

            elif isinstance(item, _MarkerItem):
                gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

                xCoord, yCoord, yAxis = item['x'], item['y'], item['yaxis']

                if ((isXLog and xCoord is not None and xCoord <= 0) or
                        (isYLog and yCoord is not None and yCoord <= 0)):
                    # Do not render markers with negative coords on log axis
                    continue

                color = item['color']
                intensity = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                bgColor = (1., 1., 1., 0.75) if intensity <= 0.5 else (0., 0., 0., 0.75)
                if xCoord is None or yCoord is None:
                    if xCoord is None:  # Horizontal line in data space
                        pixelPos = self._plotFrame.dataToPixel(
                            0.5 * sum(self._plotFrame.dataRanges[0]),
                            yCoord,
                            axis=yAxis)

                        if item['text'] is not None:
                            x = self._plotFrame.size[0] - \
                                self._plotFrame.margins.right - pixelOffset
                            y = pixelPos[1] - pixelOffset
                            label = glutils.Text2D(
                                item['text'], x, y,
                                color=item['color'],
                                bgColor=bgColor,
                                align=glutils.RIGHT,
                                valign=glutils.BOTTOM,
                                devicePixelRatio=self.getDevicePixelRatio())
                            labels.append(label)

                        width = self._plotFrame.size[0]
                        lines = glutils.GLLines2D(
                            (0, width), (pixelPos[1], pixelPos[1]),
                            style=item['linestyle'],
                            color=item['color'],
                            width=item['linewidth'])
                        context.matrix = self.matScreenProj
                        lines.render(context)

                    else:  # yCoord is None: vertical line in data space
                        yRange = self._plotFrame.dataRanges[1 if yAxis == 'left' else 2]
                        pixelPos = self._plotFrame.dataToPixel(
                            xCoord, 0.5 * sum(yRange), axis=yAxis)

                        if item['text'] is not None:
                            x = pixelPos[0] + pixelOffset
                            y = self._plotFrame.margins.top + pixelOffset
                            label = glutils.Text2D(
                                item['text'], x, y,
                                color=item['color'],
                                bgColor=bgColor,
                                align=glutils.LEFT,
                                valign=glutils.TOP,
                                devicePixelRatio=self.getDevicePixelRatio())
                            labels.append(label)

                        height = self._plotFrame.size[1]
                        lines = glutils.GLLines2D(
                            (pixelPos[0], pixelPos[0]), (0, height),
                            style=item['linestyle'],
                            color=item['color'],
                            width=item['linewidth'])
                        context.matrix = self.matScreenProj
                        lines.render(context)

                else:
                    xmin, xmax = self._plot.getXAxis().getLimits()
                    ymin, ymax = self._plot.getYAxis(axis=yAxis).getLimits()
                    if not xmin < xCoord < xmax or not ymin < yCoord < ymax:
                        # Do not render markers outside visible plot area
                        continue
                    pixelPos = self._plotFrame.dataToPixel(
                        xCoord, yCoord, axis=yAxis)

                    if isYInverted:
                        valign = glutils.BOTTOM
                        vPixelOffset = -pixelOffset
                    else:
                        valign = glutils.TOP
                        vPixelOffset = pixelOffset

                    if item['text'] is not None:
                        x = pixelPos[0] + pixelOffset
                        y = pixelPos[1] + vPixelOffset
                        label = glutils.Text2D(
                            item['text'], x, y,
                            color=item['color'],
                            bgColor=bgColor,
                            align=glutils.LEFT,
                            valign=valign,
                            devicePixelRatio=self.getDevicePixelRatio())
                        labels.append(label)

                    # For now simple implementation: using a curve for each marker
                    # Should pack all markers to a single set of points
                    marker = glutils.Points2D(
                        (pixelPos[0],),
                        (pixelPos[1],),
                        marker=item['symbol'],
                        color=item['color'],
                        size=11,
                    )
                    context.matrix = self.matScreenProj
                    marker.render(context)

            else:
                _logger.error('Unsupported item: %s', str(item))
                continue

        # Render marker labels
        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])
        for label in labels:
            label.render(self.matScreenProj)

    def _renderOverlayGL(self):
        """Render overlay layer: overlay items and crosshair."""
        plotWidth, plotHeight = self._plotFrame.plotSize

        # Scissor to plot area
        gl.glScissor(self._plotFrame.margins.left,
                     self._plotFrame.margins.bottom,
                     plotWidth, plotHeight)
        gl.glEnable(gl.GL_SCISSOR_TEST)

        self._renderItems(overlay=True)

        # Render crosshair cursor
        if self._crosshairCursor is not None and self._mousePosInPixels is not None:
            self._progBase.use()
            gl.glUniform2i(self._progBase.uniforms['isLog'], False, False)
            gl.glUniform1f(self._progBase.uniforms['tickLen'], 0.)
            posAttrib = self._progBase.attributes['position']
            matrixUnif = self._progBase.uniforms['matrix']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']

            gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

            gl.glUniformMatrix4fv(matrixUnif, 1, gl.GL_TRUE,
                                  self.matScreenProj.astype(numpy.float32))

            color, lineWidth = self._crosshairCursor
            gl.glUniform4f(colorUnif, *color)
            gl.glUniform1i(hatchStepUnif, 0)

            xPixel, yPixel = self._mousePosInPixels
            xPixel, yPixel = xPixel + 0.5, yPixel + 0.5
            vertices = numpy.array(((0., yPixel),
                                    (self._plotFrame.size[0], yPixel),
                                    (xPixel, 0.),
                                    (xPixel, self._plotFrame.size[1])),
                                   dtype=numpy.float32)

            gl.glEnableVertexAttribArray(posAttrib)
            gl.glVertexAttribPointer(posAttrib,
                                     2,
                                     gl.GL_FLOAT,
                                     gl.GL_FALSE,
                                     0, vertices)
            gl.glLineWidth(lineWidth)
            gl.glDrawArrays(gl.GL_LINES, 0, len(vertices))

        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _renderPlotAreaGL(self):
        """Render base layer of plot area.

        It renders the background, grid and items except overlays
        """
        plotWidth, plotHeight = self._plotFrame.plotSize

        gl.glScissor(self._plotFrame.margins.left,
                     self._plotFrame.margins.bottom,
                     plotWidth, plotHeight)
        gl.glEnable(gl.GL_SCISSOR_TEST)

        if self._dataBackgroundColor != self._backgroundColor:
            gl.glClearColor(*self._dataBackgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self._plotFrame.renderGrid()

        # Matrix
        trBounds = self._plotFrame.transformedDataRanges
        if trBounds.x[0] != trBounds.x[1] and trBounds.y[0] != trBounds.y[1]:
            # Do rendering of items
            self._renderItems(overlay=False)

        gl.glDisable(gl.GL_SCISSOR_TEST)

    def resizeGL(self, width, height):
        if width == 0 or height == 0:  # Do not resize
            return

        self._plotFrame.size = (
            int(self.getDevicePixelRatio() * width),
            int(self.getDevicePixelRatio() * height))

        self.matScreenProj = glutils.mat4Ortho(
            0, self._plotFrame.size[0],
            self._plotFrame.size[1], 0,
            1, -1)

        # Store current ranges
        previousXRange = self.getGraphXLimits()
        previousYRange = self.getGraphYLimits(axis='left')
        previousYRightRange = self.getGraphYLimits(axis='right')

        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = \
            self._plotFrame.dataRanges
        self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        # If plot range has changed, then emit signal
        if previousXRange != self.getGraphXLimits():
            self._plot.getXAxis()._emitLimitsChanged()
        if previousYRange != self.getGraphYLimits(axis='left'):
            self._plot.getYAxis(axis='left')._emitLimitsChanged()
        if previousYRightRange != self.getGraphYLimits(axis='right'):
            self._plot.getYAxis(axis='right')._emitLimitsChanged()

    # Add methods

    @staticmethod
    def _castArrayTo(v):
        """Returns best floating type to cast the array to.

        :param numpy.ndarray v: Array to cast
        :rtype: numpy.dtype
        :raise ValueError: If dtype is not supported
        """
        if numpy.issubdtype(v.dtype, numpy.floating):
            return numpy.float32 if v.itemsize <= 4 else numpy.float64
        elif numpy.issubdtype(v.dtype, numpy.integer):
            return numpy.float32 if v.itemsize <= 2 else numpy.float64
        else:
            raise ValueError('Unsupported data type')

    def addCurve(self, x, y,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror,
                 fill, alpha, symbolsize, baseline):
        for parameter in (x, y, color, symbol, linewidth, linestyle,
                          yaxis, fill, symbolsize):
            assert parameter is not None
        assert yaxis in ('left', 'right')

        # Convert input data
        x = numpy.array(x, copy=False)
        y = numpy.array(y, copy=False)

        # Check if float32 is enough
        if (self._castArrayTo(x) is numpy.float32 and
                self._castArrayTo(y) is numpy.float32):
            dtype = numpy.float32
        else:
            dtype = numpy.float64

        x = numpy.array(x, dtype=dtype, copy=False, order='C')
        y = numpy.array(y, dtype=dtype, copy=False, order='C')

        # Convert errors to float32
        if xerror is not None:
            xerror = numpy.array(
                xerror, dtype=numpy.float32, copy=False, order='C')
        if yerror is not None:
            yerror = numpy.array(
                yerror, dtype=numpy.float32, copy=False, order='C')

        # Handle axes log scale: convert data

        if self._plotFrame.xAxis.isLog:
            logX = numpy.log10(x)

            if xerror is not None:
                # Transform xerror so that
                # log10(x) +/- xerror' = log10(x +/- xerror)
                if hasattr(xerror, 'shape') and len(xerror.shape) == 2:
                    xErrorMinus, xErrorPlus = xerror[0], xerror[1]
                else:
                    xErrorMinus, xErrorPlus = xerror, xerror
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    # Ignore divide by zero, invalid value encountered in log10
                    xErrorMinus = logX - numpy.log10(x - xErrorMinus)
                xErrorPlus = numpy.log10(x + xErrorPlus) - logX
                xerror = numpy.array((xErrorMinus, xErrorPlus),
                                     dtype=numpy.float32)

            x = logX

        isYLog = (yaxis == 'left' and self._plotFrame.yAxis.isLog) or (
            yaxis == 'right' and self._plotFrame.y2Axis.isLog)

        if isYLog:
            logY = numpy.log10(y)

            if yerror is not None:
                # Transform yerror so that
                # log10(y) +/- yerror' = log10(y +/- yerror)
                if hasattr(yerror, 'shape') and len(yerror.shape) == 2:
                    yErrorMinus, yErrorPlus = yerror[0], yerror[1]
                else:
                    yErrorMinus, yErrorPlus = yerror, yerror
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    # Ignore divide by zero, invalid value encountered in log10
                    yErrorMinus = logY - numpy.log10(y - yErrorMinus)
                yErrorPlus = numpy.log10(y + yErrorPlus) - logY
                yerror = numpy.array((yErrorMinus, yErrorPlus),
                                     dtype=numpy.float32)

            y = logY

        # TODO check if need more filtering of error (e.g., clip to positive)

        # TODO check and improve this
        if (len(color) == 4 and
                type(color[3]) in [type(1), numpy.uint8, numpy.int8]):
            color = numpy.array(color, dtype=numpy.float32) / 255.

        if isinstance(color, numpy.ndarray) and color.ndim == 2:
            colorArray = color
            color = None
        else:
            colorArray = None
            color = colors.rgba(color)

        if alpha < 1.:  # Apply image transparency
            if colorArray is not None and colorArray.shape[1] == 4:
                # multiply alpha channel
                colorArray[:, 3] = colorArray[:, 3] * alpha
            if color is not None:
                color = color[0], color[1], color[2], color[3] * alpha

        fillColor = None
        if fill is True:
            fillColor = color
        curve = glutils.GLPlotCurve2D(
            x, y, colorArray,
            xError=xerror,
            yError=yerror,
            lineStyle=linestyle,
            lineColor=color,
            lineWidth=linewidth,
            marker=symbol,
            markerColor=color,
            markerSize=symbolsize,
            fillColor=fillColor,
            baseline=baseline,
            isYLog=isYLog)
        curve.yaxis = 'left' if yaxis is None else yaxis

        if yaxis == "right":
            self._plotFrame.isY2Axis = True

        return curve

    def addImage(self, data,
                 origin, scale,
                 colormap, alpha):
        for parameter in (data, origin, scale):
            assert parameter is not None

        if data.ndim == 2:
            # Ensure array is contiguous and eventually convert its type
            dtypes = [dtype for dtype in (
                numpy.float32, numpy.float16, numpy.uint8, numpy.uint16)
                if glu.isSupportedGLType(dtype)]
            if data.dtype in dtypes:
                data = numpy.array(data, copy=False, order='C')
            else:
                _logger.info(
                    'addImage: Convert %s data to float32', str(data.dtype))
                data = numpy.array(data, dtype=numpy.float32, order='C')

            normalization = colormap.getNormalization()
            if normalization in glutils.GLPlotColormap.SUPPORTED_NORMALIZATIONS:
                # Fast path applying colormap on the GPU
                cmapRange = colormap.getColormapRange(data=data)
                colormapLut = colormap.getNColors(nbColors=256)
                gamma = colormap.getGammaNormalizationParameter()
                nanColor = colors.rgba(colormap.getNaNColor())

                image = glutils.GLPlotColormap(
                    data,
                    origin,
                    scale,
                    colormapLut,
                    normalization,
                    gamma,
                    cmapRange,
                    alpha,
                    nanColor)

            else:  # Fallback applying colormap on CPU
                rgba = colormap.applyToData(data)
                image = glutils.GLPlotRGBAImage(rgba, origin, scale, alpha)

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert data.shape[2] in (3, 4)

            if numpy.issubdtype(data.dtype, numpy.floating):
                data = numpy.array(data, dtype=numpy.float32, copy=False)
            elif data.dtype in [numpy.uint8, numpy.uint16]:
                pass
            elif numpy.issubdtype(data.dtype, numpy.integer):
                data = numpy.array(data, dtype=numpy.uint8, copy=False)
            else:
                raise ValueError('Unsupported data type')

            image = glutils.GLPlotRGBAImage(data, origin, scale, alpha)

        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        # TODO is this needed?
        if self._plotFrame.xAxis.isLog and image.xMin <= 0.:
            raise RuntimeError(
                'Cannot add image with X <= 0 with X axis log scale')
        if self._plotFrame.yAxis.isLog and image.yMin <= 0.:
            raise RuntimeError(
                'Cannot add image with Y <= 0 with Y axis log scale')

        return image

    def addTriangles(self, x, y, triangles,
                     color, alpha):
        # Handle axes log scale: convert data
        if self._plotFrame.xAxis.isLog:
            x = numpy.log10(x)
        if self._plotFrame.yAxis.isLog:
            y = numpy.log10(y)

        triangles = glutils.GLPlotTriangles(x, y, color, triangles, alpha)

        return triangles

    def addShape(self, x, y, shape, color, fill, overlay,
                 linestyle, linewidth, linebgcolor):
        x = numpy.array(x, copy=False)
        y = numpy.array(y, copy=False)

        # TODO is this needed?
        if self._plotFrame.xAxis.isLog and x.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with X <= 0 with X axis log scale')
        if self._plotFrame.yAxis.isLog and y.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with Y <= 0 with Y axis log scale')

        return _ShapeItem(x, y, shape, color, fill, overlay,
                          linestyle, linewidth, linebgcolor)

    def addMarker(self, x, y, text, color,
                  symbol, linestyle, linewidth, constraint, yaxis):
        return _MarkerItem(x, y, text, color,
                           symbol, linestyle, linewidth, constraint, yaxis)

    # Remove methods

    def remove(self, item):
        if isinstance(item, glutils.GLPlotItem):
            if item.yaxis == 'right':
                # Check if some curves remains on the right Y axis
                y2AxisItems = (item for item in self._plot.getItems()
                               if isinstance(item, items.YAxisMixIn) and
                               item.getYAxis() == 'right')
                self._plotFrame.isY2Axis = next(y2AxisItems, None) is not None

            if item.isInitialized():
                self._glGarbageCollector.append(item)

        elif isinstance(item, (_MarkerItem, _ShapeItem)):
            pass  # No-op

        else:
            _logger.error('Unsupported item: %s', str(item))

    # Interaction methods

    _QT_CURSORS = {
        BackendBase.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        BackendBase.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        BackendBase.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        BackendBase.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setGraphCursorShape(self, cursor):
        if cursor is None:
            super(BackendOpenGL, self).unsetCursor()
        else:
            cursor = self._QT_CURSORS[cursor]
            super(BackendOpenGL, self).setCursor(qt.QCursor(cursor))

    def setGraphCursor(self, flag, color, linewidth, linestyle):
        if linestyle != '-':
            _logger.warning(
                "BackendOpenGL.setGraphCursor linestyle parameter ignored")

        if flag:
            color = colors.rgba(color)
            crosshairCursor = color, linewidth
        else:
            crosshairCursor = None

        if crosshairCursor != self._crosshairCursor:
            self._crosshairCursor = crosshairCursor

    _PICK_OFFSET = 3  # Offset in pixel used for picking

    def _mouseInPlotArea(self, x, y):
        """Returns closest visible position in the plot.

        This is performed in Qt widget pixel, not device pixel.

        :param float x: X coordinate in Qt widget pixel
        :param float y: Y coordinate in Qt widget pixel
        :return: (x, y) closest point in the plot.
        :rtype: List[float]
        """
        left, top, width, height = self.getPlotBoundsInPixels()
        return (numpy.clip(x, left, left + width - 1),  # TODO -1?
                numpy.clip(y, top, top + height - 1))

    def __pickCurves(self, item, x, y):
        """Perform picking on a curve item.

        :param GLPlotCurve2D item:
        :param float x: X position of the mouse in widget coordinates
        :param float y: Y position of the mouse in widget coordinates
        :return: List of indices of picked points or None if not picked
        :rtype: Union[List[int],None]
        """
        offset = self._PICK_OFFSET
        if item.marker is not None:
            # Convert markerSize from points to qt pixels
            qtDpi = self.getDotsPerInch() / self.getDevicePixelRatio()
            size = item.markerSize / 72. * qtDpi
            offset = max(size / 2., offset)
        if item.lineStyle is not None:
            # Convert line width from points to qt pixels
            qtDpi = self.getDotsPerInch() / self.getDevicePixelRatio()
            lineWidth = item.lineWidth / 72. * qtDpi
            offset = max(lineWidth / 2., offset)

        inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
        dataPos = self._plot.pixelToData(inAreaPos[0], inAreaPos[1],
                                         axis=item.yaxis, check=True)
        if dataPos is None:
            return None
        xPick0, yPick0 = dataPos

        inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
        dataPos = self._plot.pixelToData(inAreaPos[0], inAreaPos[1],
                                         axis=item.yaxis, check=True)
        if dataPos is None:
            return None
        xPick1, yPick1 = dataPos

        if xPick0 < xPick1:
            xPickMin, xPickMax = xPick0, xPick1
        else:
            xPickMin, xPickMax = xPick1, xPick0

        if yPick0 < yPick1:
            yPickMin, yPickMax = yPick0, yPick1
        else:
            yPickMin, yPickMax = yPick1, yPick0

        # Apply log scale if axis is log
        if self._plotFrame.xAxis.isLog:
            xPickMin = numpy.log10(xPickMin)
            xPickMax = numpy.log10(xPickMax)

        if (item.yaxis == 'left' and self._plotFrame.yAxis.isLog) or (
                item.yaxis == 'right' and self._plotFrame.y2Axis.isLog):
            yPickMin = numpy.log10(yPickMin)
            yPickMax = numpy.log10(yPickMax)

        return item.pick(xPickMin, yPickMin,
                         xPickMax, yPickMax)

    def pickItem(self, x, y, item):
        # Picking is performed in Qt widget pixels not device pixels
        dataPos = self._plot.pixelToData(x, y, axis='left', check=True)
        if dataPos is None:
            return None  # Outside plot area

        if item is None:
            _logger.error("No item provided for picking")
            return None

        # Pick markers
        if isinstance(item, _MarkerItem):
            yaxis = item['yaxis']
            pixelPos = self._plot.dataToPixel(
                item['x'], item['y'], axis=yaxis, check=False)
            if pixelPos is None:
                return None  # negative coord on a log axis

            if item['x'] is None:  # Horizontal line
                pt1 = self._plot.pixelToData(
                    x, y - self._PICK_OFFSET, axis=yaxis, check=False)
                pt2 = self._plot.pixelToData(
                    x, y + self._PICK_OFFSET, axis=yaxis, check=False)
                isPicked = (min(pt1[1], pt2[1]) <= item['y'] <=
                            max(pt1[1], pt2[1]))

            elif item['y'] is None:  # Vertical line
                pt1 = self._plot.pixelToData(
                    x - self._PICK_OFFSET, y, axis=yaxis, check=False)
                pt2 = self._plot.pixelToData(
                    x + self._PICK_OFFSET, y, axis=yaxis, check=False)
                isPicked = (min(pt1[0], pt2[0]) <= item['x'] <=
                            max(pt1[0], pt2[0]))

            else:
                isPicked = (
                    numpy.fabs(x - pixelPos[0]) <= self._PICK_OFFSET and
                    numpy.fabs(y - pixelPos[1]) <= self._PICK_OFFSET)

            return (0,) if isPicked else None

        # Pick image, curve, triangles
        elif isinstance(item, glutils.GLPlotItem):
            if isinstance(item, glutils.GLPlotCurve2D):
                return self.__pickCurves(item, x, y)
            else:
                return item.pick(*dataPos)  # Might be None

    # Update curve

    def setCurveColor(self, curve, color):
        pass  # TODO

    # Misc.

    def getWidgetHandle(self):
        return self

    def postRedisplay(self):
        self.update()

    def replot(self):
        self.update()  # async redraw

    def saveGraph(self, fileName, fileFormat, dpi):
        if dpi is not None:
            _logger.warning("saveGraph ignores dpi parameter")

        if fileFormat not in ['png', 'ppm', 'svg', 'tiff']:
            raise NotImplementedError('Unsupported format: %s' % fileFormat)

        if not self.isValid():
            _logger.error('OpenGL 2.1 not available, cannot save OpenGL image')
            width, height = self._plotFrame.size
            data = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        else:
            self.makeCurrent()

            data = numpy.empty(
                (self._plotFrame.size[1], self._plotFrame.size[0], 3),
                dtype=numpy.uint8, order='C')

            context = self.context()
            framebufferTexture = self._plotFBOs.get(context)
            if framebufferTexture is None:
                # Fallback, supports direct rendering mode: _paintDirectGL
                # might have issues as it can read on-screen framebuffer
                fboName = self.defaultFramebufferObject()
                width, height = self._plotFrame.size
            else:
                fboName = framebufferTexture.name
                height, width = framebufferTexture.shape

            previousFramebuffer = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fboName)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            gl.glReadPixels(0, 0, width, height,
                            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, previousFramebuffer)

            # glReadPixels gives bottom to top,
            # while images are stored as top to bottom
            data = numpy.flipud(data)

        # fileName is either a file-like object or a str
        saveImageToFile(data, fileName, fileFormat)

    # Graph labels

    def setGraphTitle(self, title):
        self._plotFrame.title = title

    def setGraphXLabel(self, label):
        self._plotFrame.xAxis.title = label

    def setGraphYLabel(self, label, axis):
        if axis == 'left':
            self._plotFrame.yAxis.title = label
        else:  # right axis
            self._plotFrame.y2Axis.title = label

    # Graph limits

    def _setDataRanges(self, xlim=None, ylim=None, y2lim=None):
        """Set the visible range of data in the plot frame.

        This clips the ranges to possible values (takes care of float32
        range + positive range for log).
        This also takes care of non-orthogonal axes.

        This should be moved to PlotFrame.
        """
        # Update axes range with a clipped range if too wide
        self._plotFrame.setDataRanges(xlim, ylim, y2lim)

    def _ensureAspectRatio(self, keepDim=None):
        """Update plot bounds in order to keep aspect ratio.

        Warning: keepDim on right Y axis is not implemented !

        :param str keepDim: The dimension to maintain: 'x', 'y' or None.
            If None (the default), the dimension with the largest range.
        """
        plotWidth, plotHeight = self._plotFrame.plotSize
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if keepDim is None:
            ranges = self._plot.getDataRange()
            if (ranges.y is not None and
                ranges.x is not None and
                (ranges.y[1] - ranges.y[0]) != 0.):
                dataRatio = (ranges.x[1] - ranges.x[0]) / float(ranges.y[1] - ranges.y[0])
                plotRatio = plotWidth / float(plotHeight)  # Test != 0 before

                keepDim = 'x' if dataRatio > plotRatio else 'y'
            else:  # Limit case
                keepDim = 'x'

        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = \
            self._plotFrame.dataRanges
        if keepDim == 'y':
            dataW = (yMax - yMin) * plotWidth / float(plotHeight)
            xCenter = 0.5 * (xMin + xMax)
            xMin = xCenter - 0.5 * dataW
            xMax = xCenter + 0.5 * dataW
        elif keepDim == 'x':
            dataH = (xMax - xMin) * plotHeight / float(plotWidth)
            yCenter = 0.5 * (yMin + yMax)
            yMin = yCenter - 0.5 * dataH
            yMax = yCenter + 0.5 * dataH
            y2Center = 0.5 * (y2Min + y2Max)
            y2Min = y2Center - 0.5 * dataH
            y2Max = y2Center + 0.5 * dataH
        else:
            raise RuntimeError('Unsupported dimension to keep: %s' % keepDim)

        # Update plot frame bounds
        self._setDataRanges(xlim=(xMin, xMax),
                            ylim=(yMin, yMax),
                            y2lim=(y2Min, y2Max))

    def _setPlotBounds(self, xRange=None, yRange=None, y2Range=None,
                       keepDim=None):
        # Update axes range with a clipped range if too wide
        self._setDataRanges(xlim=xRange,
                            ylim=yRange,
                            y2lim=y2Range)

        # Keep data aspect ratio
        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio(keepDim)

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        assert xmin < xmax
        assert ymin < ymax

        if y2min is None or y2max is None:
            y2Range = None
        else:
            assert y2min < y2max
            y2Range = y2min, y2max
        self._setPlotBounds((xmin, xmax), (ymin, ymax), y2Range)

    def getGraphXLimits(self):
        return self._plotFrame.dataRanges.x

    def setGraphXLimits(self, xmin, xmax):
        assert xmin < xmax
        self._setPlotBounds(xRange=(xmin, xmax), keepDim='x')

    def getGraphYLimits(self, axis):
        assert axis in ("left", "right")
        if axis == "left":
            return self._plotFrame.dataRanges.y
        else:
            return self._plotFrame.dataRanges.y2

    def setGraphYLimits(self, ymin, ymax, axis):
        assert ymin < ymax
        assert axis in ("left", "right")

        if axis == "left":
            self._setPlotBounds(yRange=(ymin, ymax), keepDim='y')
        else:
            self._setPlotBounds(y2Range=(ymin, ymax), keepDim='y')

    # Graph axes

    def getXAxisTimeZone(self):
        return self._plotFrame.xAxis.timeZone

    def setXAxisTimeZone(self, tz):
        self._plotFrame.xAxis.timeZone = tz

    def isXAxisTimeSeries(self):
        return self._plotFrame.xAxis.isTimeSeries

    def setXAxisTimeSeries(self, isTimeSeries):
        self._plotFrame.xAxis.isTimeSeries = isTimeSeries

    def setXAxisLogarithmic(self, flag):
        if flag != self._plotFrame.xAxis.isLog:
            if flag and self._keepDataAspectRatio:
                _logger.warning(
                    "KeepDataAspectRatio is ignored with log axes")

            self._plotFrame.xAxis.isLog = flag

    def setYAxisLogarithmic(self, flag):
        if (flag != self._plotFrame.yAxis.isLog or
                flag != self._plotFrame.y2Axis.isLog):
            if flag and self._keepDataAspectRatio:
                _logger.warning(
                    "KeepDataAspectRatio is ignored with log axes")

            self._plotFrame.yAxis.isLog = flag
            self._plotFrame.y2Axis.isLog = flag

    def setYAxisInverted(self, flag):
        if flag != self._plotFrame.isYAxisInverted:
            self._plotFrame.isYAxisInverted = flag

    def isYAxisInverted(self):
        return self._plotFrame.isYAxisInverted

    def isKeepDataAspectRatio(self):
        if self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog:
            return False
        else:
            return self._keepDataAspectRatio

    def setKeepDataAspectRatio(self, flag):
        if flag and (self._plotFrame.xAxis.isLog or
                     self._plotFrame.yAxis.isLog):
            _logger.warning("KeepDataAspectRatio is ignored with log axes")

        self._keepDataAspectRatio = flag

    def setGraphGrid(self, which):
        assert which in (None, 'major', 'both')
        self._plotFrame.grid = which is not None  # TODO True grid support

    # Data <-> Pixel coordinates conversion

    def dataToPixel(self, x, y, axis):
        result = self._plotFrame.dataToPixel(x, y, axis)
        if result is None:
            return None
        else:
            devicePixelRatio = self.getDevicePixelRatio()
            return tuple(value/devicePixelRatio for value in result)

    def pixelToData(self, x, y, axis):
        devicePixelRatio = self.getDevicePixelRatio()
        return self._plotFrame.pixelToData(
            x * devicePixelRatio, y * devicePixelRatio, axis)

    def getPlotBoundsInPixels(self):
        devicePixelRatio = self.getDevicePixelRatio()
        return tuple(int(value / devicePixelRatio)
            for value in self._plotFrame.plotOrigin + self._plotFrame.plotSize)

    def setAxesMargins(self, left: float, top: float, right: float, bottom: float):
        self._plotFrame.marginRatios = left, top, right, bottom

    def setForegroundColors(self, foregroundColor, gridColor):
        self._plotFrame.foregroundColor = foregroundColor
        self._plotFrame.gridColor = gridColor

    def setBackgroundColors(self, backgroundColor, dataBackgroundColor):
        self._backgroundColor = backgroundColor
        self._dataBackgroundColor = dataBackgroundColor
