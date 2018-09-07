# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2018 European Synchrotron Radiation Facility
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

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/08/2018"

from collections import OrderedDict, namedtuple
from ctypes import c_void_p
import logging

import numpy

from .._utils import FLOAT32_MINPOS
from . import BackendBase
from ... import colors
from ... import qt

from ..._glutils import gl
from ... import _glutils as glu
from .glutils import (
    GLPlotCurve2D, GLPlotColormap, GLPlotRGBAImage, GLPlotFrame2D,
    mat4Ortho, mat4Identity,
    LEFT, RIGHT, BOTTOM, TOP,
    Text2D, Shape2D)
from .glutils.PlotImageFile import saveImageToFile

_logger = logging.getLogger(__name__)


# TODO idea: BackendQtMixIn class to share code between mpl and gl
# TODO check if OpenGL is available
# TODO make an off-screen mesa backend

# Bounds ######################################################################

class Range(namedtuple('Range', ('min_', 'max_'))):
    """Describes a 1D range"""

    @property
    def range_(self):
        return self.max_ - self.min_

    @property
    def center(self):
        return 0.5 * (self.min_ + self.max_)


class Bounds(object):
    """Describes plot bounds with 2 y axis"""

    def __init__(self, xMin, xMax, yMin, yMax, y2Min, y2Max):
        self._xAxis = Range(xMin, xMax)
        self._yAxis = Range(yMin, yMax)
        self._y2Axis = Range(y2Min, y2Max)

    def __repr__(self):
        return "x: %s, y: %s, y2: %s" % (repr(self._xAxis),
                                         repr(self._yAxis),
                                         repr(self._y2Axis))

    @property
    def xAxis(self):
        return self._xAxis

    @property
    def yAxis(self):
        return self._yAxis

    @property
    def y2Axis(self):
        return self._y2Axis


# Content #####################################################################

class PlotDataContent(object):
    """Manage plot data content: images and curves.

    This class is only meant to work with _OpenGLPlotCanvas.
    """

    _PRIMITIVE_TYPES = 'curve', 'image'

    def __init__(self):
        self._primitives = OrderedDict()  # For images and curves

    def add(self, primitive):
        """Add a curve or image to the content dictionary.

        This function generates the key in the dict from the primitive.

        :param primitive: The primitive to add.
        :type primitive: Instance of GLPlotCurve2D, GLPlotColormap,
                         GLPlotRGBAImage.
        """
        if isinstance(primitive, GLPlotCurve2D):
            primitiveType = 'curve'
        elif isinstance(primitive, (GLPlotColormap, GLPlotRGBAImage)):
            primitiveType = 'image'
        else:
            raise RuntimeError('Unsupported object type: %s', primitive)

        key = primitiveType, primitive.info['legend']
        self._primitives[key] = primitive

    def get(self, primitiveType, legend):
        """Get the corresponding primitive of given type with given legend.

        :param str primitiveType: Type of primitive ('curve' or 'image').
        :param str legend: The legend of the primitive to retrieve.
        :return: The corresponding curve or None if no such curve.
        """
        assert primitiveType in self._PRIMITIVE_TYPES
        return self._primitives.get((primitiveType, legend))

    def pop(self, primitiveType, key):
        """Pop the corresponding curve or return None if no such curve.

        :param str primitiveType:
        :param str key:
        :return:
        """
        assert primitiveType in self._PRIMITIVE_TYPES
        return self._primitives.pop((primitiveType, key), None)

    def zOrderedPrimitives(self, reverse=False):
        """List of primitives sorted according to their z order.

        It is a stable sort (as sorted):
        Original order is preserved when key is the same.

        :param bool reverse: Ascending (True, default) or descending (False).
        """
        return sorted(self._primitives.values(),
                      key=lambda primitive: primitive.info['zOrder'],
                      reverse=reverse)

    def primitives(self):
        """Iterator over all primitives."""
        return self._primitives.values()

    def primitiveKeys(self, primitiveType):
        """Iterator over primitives of a specific type."""
        assert primitiveType in self._PRIMITIVE_TYPES
        for type_, key in self._primitives.keys():
            if type_ == primitiveType:
                yield key

    def getBounds(self, xPositive=False, yPositive=False):
        """Bounds of the data.

        Can return strictly positive bounds (for log scale).
        In this case, curves are clipped to their smaller positive value
        and images with negative min are ignored.

        :param bool xPositive: True to get strictly positive range.
        :param bool yPositive: True to get strictly positive range.
        :return: The range of data for x, y and y2, or default (1., 100.)
                 if no range found for one dimension.
        :rtype: Bounds
        """
        xMin, yMin, y2Min = float('inf'), float('inf'), float('inf')
        xMax = 0. if xPositive else -float('inf')
        if yPositive:
            yMax, y2Max = 0., 0.
        else:
            yMax, y2Max = -float('inf'), -float('inf')

        for item in self._primitives.values():
            # To support curve <= 0. and log and bypass images:
            # If positive only, uses x|yMinPos if available
            # and bypass other data with negative min bounds
            if xPositive:
                itemXMin = getattr(item, 'xMinPos', item.xMin)
                if itemXMin is None or itemXMin < FLOAT32_MINPOS:
                    continue
            else:
                itemXMin = item.xMin

            if yPositive:
                itemYMin = getattr(item, 'yMinPos', item.yMin)
                if itemYMin is None or itemYMin < FLOAT32_MINPOS:
                    continue
            else:
                itemYMin = item.yMin

            if itemXMin < xMin:
                xMin = itemXMin
            if item.xMax > xMax:
                xMax = item.xMax

            if item.info.get('yAxis') == 'right':
                if itemYMin < y2Min:
                    y2Min = itemYMin
                if item.yMax > y2Max:
                    y2Max = item.yMax
            else:
                if itemYMin < yMin:
                    yMin = itemYMin
                if item.yMax > yMax:
                    yMax = item.yMax

        # One of the limit has not been updated, return default range
        if xMin >= xMax:
            xMin, xMax = 1., 100.
        if yMin >= yMax:
            yMin, yMax = 1., 100.
        if y2Min >= y2Max:
            y2Min, y2Max = 1., 100.

        return Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)


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

_current_context = None


def _getContext():
    assert _current_context is not None
    return _current_context


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

    _sigPostRedisplay = qt.Signal()
    """Signal handling automatic asynchronous replot"""

    def __init__(self, plot, parent=None, f=qt.Qt.WindowFlags()):
        glu.OpenGLWidget.__init__(self, parent,
                                  alphaBufferSize=8,
                                  depthBufferSize=0,
                                  stencilBufferSize=0,
                                  version=(2, 1),
                                  f=f)
        BackendBase.BackendBase.__init__(self, plot, parent)

        self.matScreenProj = mat4Identity()

        self._progBase = glu.Program(
            _baseVertShd, _baseFragShd, attrib0='position')
        self._progTex = glu.Program(
            _texVertShd, _texFragShd, attrib0='position')
        self._plotFBOs = {}

        self._keepDataAspectRatio = False

        self._crosshairCursor = None
        self._mousePosInPixels = None

        self._markers = OrderedDict()
        self._items = OrderedDict()
        self._plotContent = PlotDataContent()  # For images and curves
        self._glGarbageCollector = []

        self._plotFrame = GLPlotFrame2D(
            margins={'left': 100, 'right': 50, 'top': 50, 'bottom': 50})

        # Make postRedisplay asynchronous using Qt signal
        self._sigPostRedisplay.connect(
            super(BackendOpenGL, self).postRedisplay,
            qt.Qt.QueuedConnection)

        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

    # QWidget

    _MOUSE_BTNS = {1: 'left', 2: 'right', 4: 'middle'}

    def contextMenuEvent(self, event):
        """Override QWidget.contextMenuEvent to implement the context menu"""
        # Makes sure it is overridden (issue with PySide)
        BackendBase.BackendBase.contextMenuEvent(self, event)

    def sizeHint(self):
        return qt.QSize(8 * 80, 6 * 80)  # Mimic MatplotlibBackend

    def mousePressEvent(self, event):
        xPixel = event.x() * self.getDevicePixelRatio()
        yPixel = event.y() * self.getDevicePixelRatio()
        btn = self._MOUSE_BTNS[event.button()]
        self._plot.onMousePress(xPixel, yPixel, btn)
        event.accept()

    def mouseMoveEvent(self, event):
        xPixel = event.x() * self.getDevicePixelRatio()
        yPixel = event.y() * self.getDevicePixelRatio()

        # Handle crosshair
        inXPixel, inYPixel = self._mouseInPlotArea(xPixel, yPixel)
        isCursorInPlot = inXPixel == xPixel and inYPixel == yPixel

        previousMousePosInPixels = self._mousePosInPixels
        self._mousePosInPixels = (xPixel, yPixel) if isCursorInPlot else None
        if (self._crosshairCursor is not None and
                previousMousePosInPixels != self._mousePosInPixels):
            # Avoid replot when cursor remains outside plot area
            self._plot._setDirtyPlot(overlayOnly=True)

        self._plot.onMouseMove(xPixel, yPixel)
        event.accept()

    def mouseReleaseEvent(self, event):
        xPixel = event.x() * self.getDevicePixelRatio()
        yPixel = event.y() * self.getDevicePixelRatio()

        btn = self._MOUSE_BTNS[event.button()]
        self._plot.onMouseRelease(xPixel, yPixel, btn)
        event.accept()

    def wheelEvent(self, event):
        xPixel = event.x() * self.getDevicePixelRatio()
        yPixel = event.y() * self.getDevicePixelRatio()

        if hasattr(event, 'angleDelta'):  # Qt 5
            delta = event.angleDelta().y()
        else:  # Qt 4 support
            delta = event.delta()
        angleInDegrees = delta / 8.
        self._plot.onMouseWheel(xPixel, yPixel, angleInDegrees)
        event.accept()

    def leaveEvent(self, _):
        self._plot.onMouseLeaveWidget()

    # OpenGLWidget API

    def initializeGL(self):
        gl.testGL()

        gl.glClearColor(1., 1., 1., 1.)
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
        self._renderMarkersGL()
        self._renderOverlayGL()

    def _paintFBOGL(self):
        context = glu.getGLContext()
        plotFBOTex = self._plotFBOs.get(context)
        if (self._plot._getDirtyPlot() or self._plotFrame.isDirty or
                plotFBOTex is None):
            self._plotVertices = numpy.array(((-1., -1., 0., 0.),
                                             (1., -1., 1., 0.),
                                             (-1., 1., 0., 1.),
                                             (1., 1., 1., 1.)),
                                             dtype=numpy.float32)
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
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
                self._renderPlotAreaGL()
                self._plotFrame.render()

        # Render plot in screen coords
        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        self._progTex.use()
        texUnit = 0

        gl.glUniform1i(self._progTex.uniforms['tex'], texUnit)
        gl.glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, gl.GL_TRUE,
                              mat4Identity().astype(numpy.float32))

        stride = self._plotVertices.shape[-1] * self._plotVertices.itemsize
        gl.glEnableVertexAttribArray(self._progTex.attributes['position'])
        gl.glVertexAttribPointer(self._progTex.attributes['position'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 stride, self._plotVertices)

        texCoordsPtr = c_void_p(self._plotVertices.ctypes.data +
                                2 * self._plotVertices.itemsize)  # Better way?
        gl.glEnableVertexAttribArray(self._progTex.attributes['texCoords'])
        gl.glVertexAttribPointer(self._progTex.attributes['texCoords'],
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 stride, texCoordsPtr)

        with plotFBOTex.texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self._plotVertices))

        self._renderMarkersGL()
        self._renderOverlayGL()

    def paintGL(self):
        global _current_context
        _current_context = self.context()

        glu.setGLContextGetter(_getContext)

        # Release OpenGL resources
        for item in self._glGarbageCollector:
            item.discard()
        self._glGarbageCollector = []

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)

        # Check if window is large enough
        plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]
        if plotWidth <= 2 or plotHeight <= 2:
            return

        # self._paintDirectGL()
        self._paintFBOGL()

        glu.setGLContextGetter()
        _current_context = None

    def _nonOrthoAxesLineMarkerPrimitives(self, marker, pixelOffset):
        """Generates the vertices and label for a line marker.

        :param dict marker: Description of a line marker
        :param int pixelOffset: Offset of text from borders in pixels
        :return: Line vertices and Text label or None
        :rtype: 2-tuple (2x2 numpy.array of float, Text2D)
        """
        label, vertices = None, None

        xCoord, yCoord = marker['x'], marker['y']
        assert xCoord is None or yCoord is None  # Specific to line markers

        # Get plot corners in data coords
        plotLeft, plotTop, plotWidth, plotHeight = self.getPlotBoundsInPixels()

        corners = [(plotLeft, plotTop),
                   (plotLeft, plotTop + plotHeight),
                   (plotLeft + plotWidth, plotTop + plotHeight),
                   (plotLeft + plotWidth, plotTop)]
        corners = numpy.array([self.pixelToData(x, y, axis='left', check=False)
                               for (x, y) in corners])

        borders = {
            'right': (corners[3], corners[2]),
            'top': (corners[0], corners[3]),
            'bottom': (corners[2], corners[1]),
            'left': (corners[1], corners[0])
        }

        textLayouts = {  # align, valign, offsets
            'right': (RIGHT, BOTTOM, (-1., -1.)),
            'top': (LEFT, TOP, (1., 1.)),
            'bottom': (LEFT, BOTTOM, (1., -1.)),
            'left': (LEFT, BOTTOM, (1., -1.))
        }

        if xCoord is None:  # Horizontal line in data space
            if marker['text'] is not None:
                # Find intersection of hline with borders in data
                # Order is important as it stops at first intersection
                for border_name in ('right', 'top', 'bottom', 'left'):
                    (x0, y0), (x1, y1) = borders[border_name]

                    if min(y0, y1) <= yCoord < max(y0, y1):
                        xIntersect = (yCoord - y0) * (x1 - x0) / (y1 - y0) + x0

                        # Add text label
                        pixelPos = self.dataToPixel(
                            xIntersect, yCoord, axis='left', check=False)

                        align, valign, offsets = textLayouts[border_name]

                        x = pixelPos[0] + offsets[0] * pixelOffset
                        y = pixelPos[1] + offsets[1] * pixelOffset
                        label = Text2D(marker['text'], x, y,
                                       color=marker['color'],
                                       bgColor=(1., 1., 1., 0.5),
                                       align=align, valign=valign)
                        break  # Stop at first intersection

            xMin, xMax = corners[:, 0].min(), corners[:, 0].max()
            vertices = numpy.array(
                ((xMin, yCoord), (xMax, yCoord)), dtype=numpy.float32)

        else:  # yCoord is None: vertical line in data space
            if marker['text'] is not None:
                # Find intersection of hline with borders in data
                # Order is important as it stops at first intersection
                for border_name in ('top', 'bottom', 'right', 'left'):
                    (x0, y0), (x1, y1) = borders[border_name]
                    if min(x0, x1) <= xCoord < max(x0, x1):
                        yIntersect = (xCoord - x0) * (y1 - y0) / (x1 - x0) + y0

                        # Add text label
                        pixelPos = self.dataToPixel(
                            xCoord, yIntersect, axis='left', check=False)

                        align, valign, offsets = textLayouts[border_name]

                        x = pixelPos[0] + offsets[0] * pixelOffset
                        y = pixelPos[1] + offsets[1] * pixelOffset
                        label = Text2D(marker['text'], x, y,
                                       color=marker['color'],
                                       bgColor=(1., 1., 1., 0.5),
                                       align=align, valign=valign)
                        break  # Stop at first intersection

            yMin, yMax = corners[:, 1].min(), corners[:, 1].max()
            vertices = numpy.array(
                ((xCoord, yMin), (xCoord, yMax)), dtype=numpy.float32)

        return vertices, label

    def _renderMarkersGL(self):
        if len(self._markers) == 0:
            return

        plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]

        # Render in plot area
        gl.glScissor(self._plotFrame.margins.left,
                     self._plotFrame.margins.bottom,
                     plotWidth, plotHeight)
        gl.glEnable(gl.GL_SCISSOR_TEST)

        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        # Prepare vertical and horizontal markers rendering
        self._progBase.use()
        gl.glUniformMatrix4fv(
            self._progBase.uniforms['matrix'], 1, gl.GL_TRUE,
            self.matScreenProj.astype(numpy.float32))
        gl.glUniform2i(self._progBase.uniforms['isLog'], False, False)
        gl.glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        gl.glUniform1f(self._progBase.uniforms['tickLen'], 0.)
        posAttrib = self._progBase.attributes['position']

        labels = []
        pixelOffset = 3

        for marker in self._markers.values():
            xCoord, yCoord = marker['x'], marker['y']

            if ((self._plotFrame.xAxis.isLog and
                    xCoord is not None and
                    xCoord <= 0) or
                    (self._plotFrame.yAxis.isLog and
                    yCoord is not None and
                    yCoord <= 0)):
                # Do not render markers with negative coords on log axis
                continue

            if xCoord is None or yCoord is None:
                if not self.isDefaultBaseVectors():  # Non-orthogonal axes
                    vertices, label = self._nonOrthoAxesLineMarkerPrimitives(
                        marker, pixelOffset)
                    if label is not None:
                        labels.append(label)

                else:  # Orthogonal axes
                    pixelPos = self.dataToPixel(
                        xCoord, yCoord, axis='left', check=False)

                    if xCoord is None:  # Horizontal line in data space
                        if marker['text'] is not None:
                            x = self._plotFrame.size[0] - \
                                self._plotFrame.margins.right - pixelOffset
                            y = pixelPos[1] - pixelOffset
                            label = Text2D(marker['text'], x, y,
                                           color=marker['color'],
                                           bgColor=(1., 1., 1., 0.5),
                                           align=RIGHT, valign=BOTTOM)
                            labels.append(label)

                        width = self._plotFrame.size[0]
                        vertices = numpy.array(((0, pixelPos[1]),
                                                (width, pixelPos[1])),
                                               dtype=numpy.float32)

                    else:  # yCoord is None: vertical line in data space
                        if marker['text'] is not None:
                            x = pixelPos[0] + pixelOffset
                            y = self._plotFrame.margins.top + pixelOffset
                            label = Text2D(marker['text'], x, y,
                                           color=marker['color'],
                                           bgColor=(1., 1., 1., 0.5),
                                           align=LEFT, valign=TOP)
                            labels.append(label)

                        height = self._plotFrame.size[1]
                        vertices = numpy.array(((pixelPos[0], 0),
                                                (pixelPos[0], height)),
                                               dtype=numpy.float32)

                self._progBase.use()
                gl.glUniform4f(self._progBase.uniforms['color'],
                               *marker['color'])

                gl.glEnableVertexAttribArray(posAttrib)
                gl.glVertexAttribPointer(posAttrib,
                                         2,
                                         gl.GL_FLOAT,
                                         gl.GL_FALSE,
                                         0, vertices)
                gl.glLineWidth(1)
                gl.glDrawArrays(gl.GL_LINES, 0, len(vertices))

            else:
                pixelPos = self.dataToPixel(
                    xCoord, yCoord, axis='left', check=True)
                if pixelPos is None:
                    # Do not render markers outside visible plot area
                    continue

                if marker['text'] is not None:
                    x = pixelPos[0] + pixelOffset
                    y = pixelPos[1] + pixelOffset
                    label = Text2D(marker['text'], x, y,
                                   color=marker['color'],
                                   bgColor=(1., 1., 1., 0.5),
                                   align=LEFT, valign=TOP)
                    labels.append(label)

                # For now simple implementation: using a curve for each marker
                # Should pack all markers to a single set of points
                markerCurve = GLPlotCurve2D(
                    numpy.array((pixelPos[0],), dtype=numpy.float64),
                    numpy.array((pixelPos[1],), dtype=numpy.float64),
                    marker=marker['symbol'],
                    markerColor=marker['color'],
                    markerSize=11)
                markerCurve.render(self.matScreenProj, False, False)
                markerCurve.discard()

        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        # Render marker labels
        for label in labels:
            label.render(self.matScreenProj)

        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _renderOverlayGL(self):
        # Render crosshair cursor
        if self._crosshairCursor is not None:
            plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]

            # Scissor to plot area
            gl.glScissor(self._plotFrame.margins.left,
                         self._plotFrame.margins.bottom,
                         plotWidth, plotHeight)
            gl.glEnable(gl.GL_SCISSOR_TEST)

            self._progBase.use()
            gl.glUniform2i(self._progBase.uniforms['isLog'], False, False)
            gl.glUniform1f(self._progBase.uniforms['tickLen'], 0.)
            posAttrib = self._progBase.attributes['position']
            matrixUnif = self._progBase.uniforms['matrix']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']

            # Render crosshair cursor in screen frame but with scissor
            if (self._crosshairCursor is not None and
                    self._mousePosInPixels is not None):
                gl.glViewport(
                    0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

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
        plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]

        self._plotFrame.renderGrid()

        gl.glScissor(self._plotFrame.margins.left,
                     self._plotFrame.margins.bottom,
                     plotWidth, plotHeight)
        gl.glEnable(gl.GL_SCISSOR_TEST)

        # Matrix
        trBounds = self._plotFrame.transformedDataRanges
        if trBounds.x[0] == trBounds.x[1] or \
           trBounds.y[0] == trBounds.y[1]:
            return

        isXLog = self._plotFrame.xAxis.isLog
        isYLog = self._plotFrame.yAxis.isLog

        gl.glViewport(self._plotFrame.margins.left,
                      self._plotFrame.margins.bottom,
                      plotWidth, plotHeight)

        # Render images and curves
        # sorted is stable: original order is preserved when key is the same
        for item in self._plotContent.zOrderedPrimitives():
            if item.info.get('yAxis') == 'right':
                item.render(self._plotFrame.transformedDataY2ProjMat,
                            isXLog, isYLog)
            else:
                item.render(self._plotFrame.transformedDataProjMat,
                            isXLog, isYLog)

        # Render Items
        gl.glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        self._progBase.use()
        gl.glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, gl.GL_TRUE,
                              self.matScreenProj.astype(numpy.float32))
        gl.glUniform2i(self._progBase.uniforms['isLog'], False, False)
        gl.glUniform1f(self._progBase.uniforms['tickLen'], 0.)

        for item in self._items.values():
            if ((isXLog and numpy.min(item['x']) < FLOAT32_MINPOS) or
                    (isYLog and numpy.min(item['y']) < FLOAT32_MINPOS)):
                # Ignore items <= 0. on log axes
                continue

            closed = item['shape'] != 'polylines'
            points = [self.dataToPixel(x, y, axis='left', check=False)
                      for (x, y) in zip(item['x'], item['y'])]
            shape2D = Shape2D(points,
                              fill=item['fill'],
                              fillColor=item['color'],
                              stroke=True,
                              strokeColor=item['color'],
                              strokeClosed=closed)

            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            shape2D.render(posAttrib, colorUnif, hatchStepUnif)

        gl.glDisable(gl.GL_SCISSOR_TEST)

    def resizeGL(self, width, height):
        if width == 0 or height == 0:  # Do not resize
            return

        self._plotFrame.size = (
            int(self.getDevicePixelRatio() * width),
            int(self.getDevicePixelRatio() * height))

        self.matScreenProj = mat4Ortho(0, self._plotFrame.size[0],
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

    def addCurve(self, x, y, legend,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill, alpha, symbolsize):
        for parameter in (x, y, legend, color, symbol, linewidth, linestyle,
                          yaxis, z, selectable, fill, symbolsize):
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

        behaviors = set()
        if selectable:
            behaviors.add('selectable')

        curve = GLPlotCurve2D(x, y, colorArray,
                              xError=xerror,
                              yError=yerror,
                              lineStyle=linestyle,
                              lineColor=color,
                              lineWidth=linewidth,
                              marker=symbol,
                              markerColor=color,
                              markerSize=symbolsize,
                              fillColor=color if fill else None,
                              isYLog=isYLog)
        curve.info = {
            'legend': legend,
            'zOrder': z,
            'behaviors': behaviors,
            'yAxis': 'left' if yaxis is None else yaxis,
        }

        if yaxis == "right":
            self._plotFrame.isY2Axis = True

        self._plotContent.add(curve)

        return legend, 'curve'

    def addImage(self, data, legend,
                 origin, scale, z,
                 selectable, draggable,
                 colormap, alpha):
        for parameter in (data, legend, origin, scale, z,
                          selectable, draggable):
            assert parameter is not None

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        if data.ndim == 2:
            # Ensure array is contiguous and eventually convert its type
            if data.dtype in (numpy.float32, numpy.uint8, numpy.uint16):
                data = numpy.array(data, copy=False, order='C')
            else:
                _logger.info(
                    'addImage: Convert %s data to float32', str(data.dtype))
                data = numpy.array(data, dtype=numpy.float32, order='C')

            colormapIsLog = colormap.getNormalization() == 'log'
            cmapRange = colormap.getColormapRange(data=data)
            colormapLut = colormap.getNColors(nbColors=256)

            image = GLPlotColormap(data,
                                   origin,
                                   scale,
                                   colormapLut,
                                   colormapIsLog,
                                   cmapRange,
                                   alpha)
            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }
            self._plotContent.add(image)

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert data.shape[2] in (3, 4)

            if numpy.issubdtype(data.dtype, numpy.floating):
                data = numpy.array(data, dtype=numpy.float32, copy=False)
            elif numpy.issubdtype(data.dtype, numpy.integer):
                data = numpy.array(data, dtype=numpy.uint8, copy=False)
            else:
                raise ValueError('Unsupported data type')

            image = GLPlotRGBAImage(data, origin, scale, alpha)

            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }

            if self._plotFrame.xAxis.isLog and image.xMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with X <= 0 with X axis log scale')
            if self._plotFrame.yAxis.isLog and image.yMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with Y <= 0 with Y axis log scale')

            self._plotContent.add(image)

        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        return legend, 'image'

    def addItem(self, x, y, legend, shape, color, fill, overlay, z):
        # TODO handle overlay
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

        # TODO is this needed?
        if self._plotFrame.xAxis.isLog and x.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with X <= 0 with X axis log scale')
        if self._plotFrame.yAxis.isLog and y.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with Y <= 0 with Y axis log scale')

        # Ignore fill for polylines to mimic matplotlib
        fill = fill if shape != 'polylines' else False

        self._items[legend] = {
            'shape': shape,
            'color': colors.rgba(color),
            'fill': 'hatch' if fill else None,
            'x': x,
            'y': y
        }

        return legend, 'item'

    def addMarker(self, x, y, legend, text, color,
                  selectable, draggable,
                  symbol, linestyle, linewidth, constraint):

        if symbol is None:
            symbol = '+'

        if linestyle != '-' or linewidth != 1:
            _logger.warning(
                'OpenGL backend does not support marker line style and width.')

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        # Apply constraint to provided position
        isConstraint = (draggable and constraint is not None and
                        x is not None and y is not None)
        if isConstraint:
            x, y = constraint(x, y)

        self._markers[legend] = {
            'x': x,
            'y': y,
            'legend': legend,
            'text': text,
            'color': colors.rgba(color),
            'behaviors': behaviors,
            'constraint': constraint if isConstraint else None,
            'symbol': symbol,
        }

        return legend, 'marker'

    # Remove methods

    def remove(self, item):
        legend, kind = item

        if kind == 'curve':
            curve = self._plotContent.pop('curve', legend)
            if curve is not None:
                # Check if some curves remains on the right Y axis
                y2AxisItems = (item for item in self._plotContent.primitives()
                               if item.info.get('yAxis', 'left') == 'right')
                self._plotFrame.isY2Axis = next(y2AxisItems, None) is not None

                self._glGarbageCollector.append(curve)

        elif kind == 'image':
            image = self._plotContent.pop('image', legend)
            if image is not None:
                self._glGarbageCollector.append(image)

        elif kind == 'marker':
            self._markers.pop(legend, False)

        elif kind == 'item':
            self._items.pop(legend, False)

        else:
            _logger.error('Unsupported kind: %s', str(kind))

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
        if linestyle is not '-':
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
        xPlot = numpy.clip(
            x, self._plotFrame.margins.left,
            self._plotFrame.size[0] - self._plotFrame.margins.right - 1)
        yPlot = numpy.clip(
            y, self._plotFrame.margins.top,
            self._plotFrame.size[1] - self._plotFrame.margins.bottom - 1)
        return xPlot, yPlot

    def pickItems(self, x, y, kinds):
        picked = []

        dataPos = self.pixelToData(x, y, axis='left', check=True)
        if dataPos is not None:
            # Pick markers
            if 'marker' in kinds:
                for marker in reversed(list(self._markers.values())):
                    pixelPos = self.dataToPixel(
                        marker['x'], marker['y'], axis='left', check=False)
                    if pixelPos is None:  # negative coord on a log axis
                        continue

                    if marker['x'] is None:  # Horizontal line
                        pt1 = self.pixelToData(
                            x, y - self._PICK_OFFSET, axis='left', check=False)
                        pt2 = self.pixelToData(
                            x, y + self._PICK_OFFSET, axis='left', check=False)
                        isPicked = (min(pt1[1], pt2[1]) <= marker['y'] <=
                                    max(pt1[1], pt2[1]))

                    elif marker['y'] is None:  # Vertical line
                        pt1 = self.pixelToData(
                            x - self._PICK_OFFSET, y, axis='left', check=False)
                        pt2 = self.pixelToData(
                            x + self._PICK_OFFSET, y, axis='left', check=False)
                        isPicked = (min(pt1[0], pt2[0]) <= marker['x'] <=
                                    max(pt1[0], pt2[0]))

                    else:
                        isPicked = (
                            numpy.fabs(x - pixelPos[0]) <= self._PICK_OFFSET and
                            numpy.fabs(y - pixelPos[1]) <= self._PICK_OFFSET)

                    if isPicked:
                        picked.append(dict(kind='marker',
                                           legend=marker['legend']))

            # Pick image and curves
            if 'image' in kinds or 'curve' in kinds:
                for item in self._plotContent.zOrderedPrimitives(reverse=True):
                    if ('image' in kinds and
                            isinstance(item, (GLPlotColormap, GLPlotRGBAImage))):
                        pickedPos = item.pick(*dataPos)
                        if pickedPos is not None:
                            picked.append(dict(kind='image',
                                               legend=item.info['legend']))

                    elif 'curve' in kinds and isinstance(item, GLPlotCurve2D):
                        offset = self._PICK_OFFSET
                        if item.marker is not None:
                            offset = max(item.markerSize / 2., offset)
                        if item.lineStyle is not None:
                            offset = max(item.lineWidth / 2., offset)

                        yAxis = item.info['yAxis']

                        inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
                        dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                                   axis=yAxis, check=True)
                        if dataPos is None:
                            continue
                        xPick0, yPick0 = dataPos

                        inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
                        dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                                   axis=yAxis, check=True)
                        if dataPos is None:
                            continue
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

                        if (yAxis == 'left' and self._plotFrame.yAxis.isLog) or (
                                yAxis == 'right' and self._plotFrame.y2Axis.isLog):
                            yPickMin = numpy.log10(yPickMin)
                            yPickMax = numpy.log10(yPickMax)

                        pickedIndices = item.pick(xPickMin, yPickMin,
                                                  xPickMax, yPickMax)
                        if pickedIndices:
                            picked.append(dict(kind='curve',
                                               legend=item.info['legend'],
                                               indices=pickedIndices))

        return picked

    # Update curve

    def setCurveColor(self, curve, color):
        pass  # TODO

    # Misc.

    def getWidgetHandle(self):
        return self

    def postRedisplay(self):
        self._sigPostRedisplay.emit()

    def replot(self):
        self.update()  # async redraw
        # self.repaint()  # immediate redraw

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
            if label:
                _logger.warning('Right axis label not implemented')

    # Non orthogonal axes

    def setBaseVectors(self, x=(1., 0.), y=(0., 1.)):
        """Set base vectors.

        Useful for non-orthogonal axes.
        If an axis is in log scale, skew is applied to log transformed values.

        Base vector does not work well with log axes, to investi
        """
        if x != (1., 0.) and y != (0., 1.):
            if self._plotFrame.xAxis.isLog:
                _logger.warning("setBaseVectors disables X axis logarithmic.")
                self.setXAxisLogarithmic(False)
            if self._plotFrame.yAxis.isLog:
                _logger.warning("setBaseVectors disables Y axis logarithmic.")
                self.setYAxisLogarithmic(False)

            if self.isKeepDataAspectRatio():
                _logger.warning("setBaseVectors disables keepDataAspectRatio.")
                self.keepDataAspectRatio(False)

        self._plotFrame.baseVectors = x, y

    def getBaseVectors(self):
        return self._plotFrame.baseVectors

    def isDefaultBaseVectors(self):
        return self._plotFrame.baseVectors == \
            self._plotFrame.DEFAULT_BASE_VECTORS

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

        if not self.isDefaultBaseVectors():
            # Update axes range with axes bounds in data coords
            plotLeft, plotTop, plotWidth, plotHeight = \
                self.getPlotBoundsInPixels()

            self._plotFrame.xAxis.dataRange = sorted([
                self.pixelToData(x, y, axis='left', check=False)[0]
                for (x, y) in ((plotLeft, plotTop + plotHeight),
                               (plotLeft + plotWidth, plotTop + plotHeight))])

            self._plotFrame.yAxis.dataRange = sorted([
                self.pixelToData(x, y, axis='left', check=False)[1]
                for (x, y) in ((plotLeft, plotTop + plotHeight),
                               (plotLeft, plotTop))])

            self._plotFrame.y2Axis.dataRange = sorted([
                self.pixelToData(x, y, axis='right', check=False)[1]
                for (x, y) in ((plotLeft + plotWidth, plotTop + plotHeight),
                               (plotLeft + plotWidth, plotTop))])

    def _ensureAspectRatio(self, keepDim=None):
        """Update plot bounds in order to keep aspect ratio.

        Warning: keepDim on right Y axis is not implemented !

        :param str keepDim: The dimension to maintain: 'x', 'y' or None.
            If None (the default), the dimension with the largest range.
        """
        plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if keepDim is None:
            dataBounds = self._plotContent.getBounds(
                self._plotFrame.xAxis.isLog, self._plotFrame.yAxis.isLog)
            if dataBounds.yAxis.range_ != 0.:
                dataRatio = dataBounds.xAxis.range_
                dataRatio /= float(dataBounds.yAxis.range_)

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

            if flag and not self.isDefaultBaseVectors():
                _logger.warning(
                    "setXAxisLogarithmic ignored because baseVectors are set")
                return

            self._plotFrame.xAxis.isLog = flag

    def setYAxisLogarithmic(self, flag):
        if (flag != self._plotFrame.yAxis.isLog or
                flag != self._plotFrame.y2Axis.isLog):
            if flag and self._keepDataAspectRatio:
                _logger.warning(
                    "KeepDataAspectRatio is ignored with log axes")

            if flag and not self.isDefaultBaseVectors():
                _logger.warning(
                    "setYAxisLogarithmic ignored because baseVectors are set")
                return

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
        if flag and not self.isDefaultBaseVectors():
            _logger.warning(
                "keepDataAspectRatio ignored because baseVectors are set")

        self._keepDataAspectRatio = flag

    def setGraphGrid(self, which):
        assert which in (None, 'major', 'both')
        self._plotFrame.grid = which is not None  # TODO True grid support

    # Data <-> Pixel coordinates conversion

    def dataToPixel(self, x, y, axis, check=False):
        assert axis in ('left', 'right')

        if x is None or y is None:
            dataBounds = self._plotContent.getBounds(
                self._plotFrame.xAxis.isLog, self._plotFrame.yAxis.isLog)

            if x is None:
                x = dataBounds.xAxis.center

            if y is None:
                if axis == 'left':
                    y = dataBounds.yAxis.center
                else:
                    y = dataBounds.y2Axis.center

        result = self._plotFrame.dataToPixel(x, y, axis)

        if check and result is not None:
            xPixel, yPixel = result
            width, height = self._plotFrame.size
            if (xPixel < self._plotFrame.margins.left or
                    xPixel > (width - self._plotFrame.margins.right) or
                    yPixel < self._plotFrame.margins.top or
                    yPixel > height - self._plotFrame.margins.bottom):
                return None  # (x, y) is out of plot area

        return result

    def pixelToData(self, x, y, axis, check):
        assert axis in ("left", "right")

        if x is None:
            x = self._plotFrame.size[0] / 2.
        if y is None:
            y = self._plotFrame.size[1] / 2.

        if check and (x < self._plotFrame.margins.left or
                      x > (self._plotFrame.size[0] -
                           self._plotFrame.margins.right) or
                      y < self._plotFrame.margins.top or
                      y > (self._plotFrame.size[1] -
                           self._plotFrame.margins.bottom)):
            return None  # (x, y) is out of plot area

        return self._plotFrame.pixelToData(x, y, axis)

    def getPlotBoundsInPixels(self):
        return self._plotFrame.plotOrigin + self._plotFrame.plotSize

    def setAxesDisplayed(self, displayed):
        BackendBase.BackendBase.setAxesDisplayed(self, displayed)
        self._plotFrame.displayed = displayed
