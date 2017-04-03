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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
OpenGL plot backend with no dependencies on the control of the OpenGL context.
"""


# import ######################################################################

from collections import namedtuple
import math
import numpy as np
import warnings

try:
    from ..PlotBackend import PlotBackend
except ImportError:
    from PyMca5.PyMcaGraph.PlotBackend import PlotBackend

from .GLSupport import *  # noqa
from .GLSupport.gl import *  # noqa
from .GLSupport.PlotEvents import prepareMouseSignal,\
    prepareLimitsChangedSignal
from .GLSupport.PlotImageFile import saveImageToFile
from .GLSupport.PlotInteraction import PlotInteraction
from . import _utils


# OrderedDict #################################################################

class MiniOrderedDict(object):
    """Simple subset of OrderedDict for python 2.6 support"""

    _DEFAULT_ARG = object()

    def __init__(self):
        self._dict = {}
        self._orderedKeys = []

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if key not in self._orderedKeys:
            self._orderedKeys.append(key)
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]
        self._orderedKeys.remove(key)

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._orderedKeys[:]

    def values(self):
        return [self._dict[key] for key in self._orderedKeys]

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def pop(self, key, default=_DEFAULT_ARG):
        value = self._dict.pop(key, self._DEFAULT_ARG)
        if value is not self._DEFAULT_ARG:
            self._orderedKeys.remove(key)
            return value
        elif default is self._DEFAULT_ARG:
            raise KeyError
        else:
            return default


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
        self._primitives = MiniOrderedDict()  # For images and curves

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
    }
    """


# OpenGLPlotCanvas ############################################################

CURSOR_DEFAULT = 'default'
CURSOR_POINTING = 'pointing'
CURSOR_SIZE_HOR = 'size horizontal'
CURSOR_SIZE_VER = 'size vertical'
CURSOR_SIZE_ALL = 'size all'


class OpenGLPlotCanvas(PlotBackend):
    """Implements PlotBackend API using OpenGL.

    WARNINGS:
    Unless stated otherwise, this API is NOT thread-safe and MUST be
    called from the main thread.
    When numpy arrays are passed as arguments to the API (through
    :func:`addCurve` and :func:`addImage`), they are copied only if
    required.
    So, the caller should not modify these arrays afterwards.
    """
    _UNNAMED_ITEM = '__unnamed_item__'

    _PICK_OFFSET = 3

    _DEFAULT_COLORMAP = {'name': 'gray', 'normalization': 'linear',
                         'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                         'colors': 256}

    def __init__(self, parent=None, glContextGetter=None, **kw):
        self._eventCallback = self._noopCallback
        self._defaultColormap = self._DEFAULT_COLORMAP

        self._progBase = GLProgram(_baseVertShd, _baseFragShd)
        self._progTex = GLProgram(_texVertShd, _texFragShd)
        self._plotFBOs = {}

        self._keepDataAspectRatio = False

        self._activeCurveLegend = None

        self._crosshairCursor = None
        self._mousePosInPixels = None

        self._markers = MiniOrderedDict()
        self._items = MiniOrderedDict()
        self._plotContent = PlotDataContent()  # For images and curves
        self._selectionAreas = MiniOrderedDict()
        self._glGarbageCollector = []

        self._lineWidth = 1
        self._tickLen = 5

        self._plotDirtyFlag = True

        self.eventHandler = PlotInteraction(self)
        self.eventHandler.setInteractiveMode('zoom', color=(0., 0., 0., 1.))

        self._pressedButtons = []  # Currently pressed mouse buttons

        self._plotFrame = GLPlotFrame2D(
            margins={'left': 100, 'right': 50, 'top': 50, 'bottom': 50})

        PlotBackend.__init__(self, parent, **kw)

    # Callback #

    @staticmethod
    def _noopCallback(eventDict):
        """Default no-op callback."""
        pass

    def setCallback(self, func):
        if func is None:
            self._eventCallback = self._noopCallback
        else:
            assert callable(func)
            self._eventCallback = func

    def sendEvent(self, event):
        """Send the event to the registered callback.

        :param dict event: The event information (See PlotBackend for details).
        """
        self._eventCallback(event)

    # Link with embedding toolkit #

    def makeCurrent(self):
        """Override this method to allow to set the current OpenGL context."""
        pass

    def postRedisplay(self):
        raise NotImplementedError("This method must be provided by \
                                  subclass to trigger redraw")

    def setCursor(self, cursor=CURSOR_DEFAULT):
        """Override this method in subclass to enable cursor shape changes
        """
        print('setCursor:', cursor)

    # User event handling #

    def _mouseInPlotArea(self, x, y):
        xPlot = clamp(
            x, self._plotFrame.margins.left,
            self._plotFrame.size[0] - self._plotFrame.margins.right - 1)
        yPlot = clamp(
            y, self._plotFrame.margins.top,
            self._plotFrame.size[1] - self._plotFrame.margins.bottom - 1)
        return xPlot, yPlot

    def onMousePress(self, xPixel, yPixel, btn):
        if self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self._pressedButtons.append(btn)
            self.eventHandler.handleEvent('press', xPixel, yPixel, btn)

    def onMouseMove(self, xPixel, yPixel):
        inXPixel, inYPixel = self._mouseInPlotArea(xPixel, yPixel)
        isCursorInPlot = inXPixel == xPixel and inYPixel == yPixel

        previousMousePosInPixels = self._mousePosInPixels
        self._mousePosInPixels = (xPixel, yPixel) if isCursorInPlot else None
        if (self._crosshairCursor is not None and
                previousMousePosInPixels != self._crosshairCursor):
            # Avoid replot when cursor remains outside plot area
            self.replot()

        if isCursorInPlot:
            # Signal mouse move event
            dataPos = self.pixelToData(inXPixel, inYPixel)
            assert dataPos is not None

            btn = self._pressedButtons[-1] if self._pressedButtons else None
            eventDict = prepareMouseSignal(
                'mouseMoved', btn, dataPos[0], dataPos[1], xPixel, yPixel)
            self.sendEvent(eventDict)

        # Either button was pressed in the plot or cursor is in the plot
        if isCursorInPlot or self._pressedButtons:
            self.eventHandler.handleEvent('move', inXPixel, inYPixel)

    def onMouseRelease(self, xPixel, yPixel, btn):
        try:
            self._pressedButtons.remove(btn)
        except ValueError:
            pass
        else:
            xPixel, yPixel = self._mouseInPlotArea(xPixel, yPixel)
            self.eventHandler.handleEvent('release', xPixel, yPixel, btn)

    def onMouseWheel(self, xPixel, yPixel, angleInDegrees):
        if self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self.eventHandler.handleEvent('wheel', xPixel, yPixel,
                                          angleInDegrees)

    # Picking #

    def pickMarker(self, x, y, test=None):
        if test is None:
            test = lambda marker: True

        for marker in reversed(self._markers.values()):
            pixelPos = self.dataToPixel(marker['x'], marker['y'], check=False)
            if pixelPos is None:  # negative coord on a log axis
                continue

            if marker['x'] is None:  # Horizontal line
                pt1 = self.pixelToData(x, y - self._PICK_OFFSET, check=False)
                pt2 = self.pixelToData(x, y + self._PICK_OFFSET, check=False)
                isPicked = (marker['y'] >= min(pt1[1], pt2[1]) and
                    marker['y'] <= max(pt1[1], pt2[1]))

            elif marker['y'] is None:  # Vertical line
                pt1 = self.pixelToData(x - self._PICK_OFFSET, y, check=False)
                pt2 = self.pixelToData(x + self._PICK_OFFSET, y, check=False)
                isPicked = (marker['x'] >= min(pt1[0], pt2[0]) and
                    marker['x'] <= max(pt1[0], pt2[0]))

            else:
                isPicked = (
                    math.fabs(x - pixelPos[0]) <= self._PICK_OFFSET and \
                    math.fabs(y - pixelPos[1]) <= self._PICK_OFFSET
                )

            if isPicked:
                if test(marker):
                    return marker

        return None

    def pickImageOrCurve(self, x, y, test=None):
        if test is None:
            test = lambda item: True

        dataPos = self.pixelToData(x, y)
        assert dataPos is not None

        for item in self._plotContent.zOrderedPrimitives(reverse=True):
            if test(item):
                if isinstance(item, (GLPlotColormap, GLPlotRGBAImage)):
                    pickedPos = item.pick(*dataPos)
                    if pickedPos is not None:
                        return 'image', item, pickedPos

                elif isinstance(item, GLPlotCurve2D):
                    offset = self._PICK_OFFSET
                    if item.marker is not None:
                        offset = max(item.markerSize / 2., offset)
                    if item.lineStyle is not None:
                        offset = max(item.lineWidth / 2., offset)

                    yAxis = item.info['yAxis']

                    inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
                    dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                               axis=yAxis)
                    assert dataPos is not None
                    xPick0, yPick0 = dataPos

                    inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
                    dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                               axis=yAxis)
                    assert dataPos is not None
                    xPick1, yPick1 = dataPos

                    if xPick0 < xPick1:
                        xPickMin, xPickMax = xPick0, xPick1
                    else:
                        xPickMin, xPickMax = xPick1, xPick0

                    if yPick0 < yPick1:
                        yPickMin, yPickMax = yPick0, yPick1
                    else:
                        yPickMin, yPickMax = yPick1, yPick0

                    pickedIndices = item.pick(xPickMin, yPickMin,
                                              xPickMax, yPickMax)
                    if pickedIndices:
                        return 'curve', item, pickedIndices
        return None

    # Default colormap #

    def getSupportedColormaps(self):
        return GLPlotColormap.COLORMAPS

    def getDefaultColormap(self):
        return self._defaultColormap.copy()

    def setDefaultColormap(self, colormap=None):
        if colormap is None:
            self._defaultColormap = self._DEFAULT_COLORMAP
        else:
            assert colormap['name'] in self.getSupportedColormaps()
            if colormap['colors'] != 256:
                warnings.warn("Colormap 'colors' field is ignored",
                              RuntimeWarning)
            self._defaultColormap = colormap.copy()

    # Manage Plot #

    def setSelectionArea(self, points, fill=None, color=None, name=None):
        """Set a polygon selection area overlaid on the plot.
        Multiple simultaneous areas are supported through the name parameter.

        :param points: The 2D coordinates of the points of the polygon
        :type points: An iterable of (x, y) coordinates
        :param str fill: The fill mode: 'hatch', 'solid' or None (default)
        :param color: RGBA color to use (default: black) or 'video inverted'
                      to use video inverted mode.
        :type color: list or tuple of 4 float in the range [0, 1]
        :param name: The key associated with this selection area
        """
        if color is None:
            color = 0., 0., 0., 1.

        isVideoInverted = (color == 'video inverted')
        if isVideoInverted:
            color = 1., 1., 1., 1.

        shape = Shape2D(points, fill=fill, fillColor=color,
                        stroke=True, strokeColor=color)
        shape.isVideoInverted = isVideoInverted
        self._selectionAreas[name] = shape

    def resetSelectionArea(self, name=None):
        """Remove the name selection area set by setSelectionArea.
        If name is None (the default), it removes all selection areas.

        :param name: The name key provided to setSelectionArea or None
        """
        if name is None:
            self._selectionAreas = MiniOrderedDict()
        elif name in self._selectionAreas:
            del self._selectionAreas[name]

    # Coordinate systems #

    def dataToPixel(self, x=None, y=None, axis='left', check=True):
        """Convert data coordinate to widget pixel coordinate.

        :param bool check: Toggle checking if data position is in displayed
                           area.
                           If False, this method never returns None.
        :return: pixel position or None if coord <= 0 on a log axis or
                 check failed.
        :rtype: tuple of 2 ints or None.
        """
        assert axis in ('left', 'right')

        if x is None or y is None:
            dataBounds = self._plotContent.getBounds(
                self.isXAxisLogarithmic(), self.isYAxisLogarithmic())

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

    def pixelToData(self, x=None, y=None, axis="left", check=True):
        """
        :param bool check: Toggle checking if pixel is in plot area.
                           If False, this method never returns None.
        """
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

    def plotOriginInPixels(self):
        """Plot area origin (left, top) in widget coordinates in pixels."""
        return self._plotFrame.plotOrigin

    def plotSizeInPixels(self):
        """Plot area size (width, height) in pixels."""
        return self._plotFrame.plotSize

    # QGLWidget API #

    @staticmethod
    def _setBlendFuncGL():
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE)

    def initializeGL(self):
        testGL()

        glClearColor(1., 1., 1., 1.)
        glClearStencil(0)

        glEnable(GL_BLEND)
        self._setBlendFuncGL()

        # For lines
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # For points
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        glEnable(GL_POINT_SPRITE)  # OpenGL 2
        # glEnable(GL_PROGRAM_POINT_SIZE)

        # Building shader programs here failed on Mac OS X 10.7.5

    def _paintDirectGL(self):
        self._renderPlotAreaGL()
        self._plotFrame.render()
        self._renderMarkersGL()
        self._renderOverlayGL()

    def _paintFBOGL(self):
        context = getGLContext()
        plotFBOTex = self._plotFBOs.get(context)
        if (self._plotDirtyFlag or self._plotFrame.isDirty or
                plotFBOTex is None):
            self._plotDirtyFlag = False
            self._plotVertices = np.array(((-1., -1., 0., 0.),
                                           (1., -1., 1., 0.),
                                           (-1., 1., 0., 1.),
                                           (1., 1., 1., 1.)),
                                          dtype=np.float32)
            if plotFBOTex is None or \
               plotFBOTex.width != self._plotFrame.size[0] or \
               plotFBOTex.height != self._plotFrame.size[1]:
                if plotFBOTex is not None:
                    plotFBOTex.discard()
                plotFBOTex = FBOTexture(GL_RGBA,
                                        self._plotFrame.size[0],
                                        self._plotFrame.size[1],
                                        minFilter=GL_NEAREST,
                                        magFilter=GL_NEAREST,
                                        wrapS=GL_CLAMP_TO_EDGE,
                                        wrapT=GL_CLAMP_TO_EDGE)
                self._plotFBOs[context] = plotFBOTex

            with plotFBOTex:
                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
                self._renderPlotAreaGL()
                self._plotFrame.render()

        # Render plot in screen coords
        glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        self._progTex.use()
        texUnit = 0

        glUniform1i(self._progTex.uniforms['tex'], texUnit)
        glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, GL_TRUE,
                           mat4Identity())

        stride = self._plotVertices.shape[-1] * self._plotVertices.itemsize
        glEnableVertexAttribArray(self._progTex.attributes['position'])
        glVertexAttribPointer(self._progTex.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, self._plotVertices)

        texCoordsPtr = c_void_p(self._plotVertices.ctypes.data +
                                2 * self._plotVertices.itemsize)  # Better way?
        glEnableVertexAttribArray(self._progTex.attributes['texCoords'])
        glVertexAttribPointer(self._progTex.attributes['texCoords'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, texCoordsPtr)

        plotFBOTex.bind(texUnit)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(self._plotVertices))
        glBindTexture(GL_TEXTURE_2D, 0)

        self._renderMarkersGL()
        self._renderOverlayGL()

    def paintGL(self):
        # Release OpenGL resources
        for item in self._glGarbageCollector:
            item.discard()
        self._glGarbageCollector = []

        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # Check if window is large enough
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        # self._paintDirectGL()
        self._paintFBOGL()

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
        plotLeft, plotTop = self.plotOriginInPixels()
        plotWidth, plotHeight = self.plotSizeInPixels()

        corners = [(plotLeft, plotTop),
                   (plotLeft, plotTop + plotHeight),
                   (plotLeft + plotWidth, plotTop + plotHeight),
                   (plotLeft + plotWidth, plotTop)]
        corners = np.array([self.pixelToData(x, y, check=False)
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

                    if yCoord >= min(y0, y1) and yCoord < max(y0, y1):
                        xIntersect = (yCoord - y0) * (x1 - x0) / (y1 - y0) + x0

                        # Add text label
                        pixelPos = self.dataToPixel(
                            xIntersect, yCoord, check=False)

                        align, valign, offsets = textLayouts[border_name]

                        x = pixelPos[0] + offsets[0] * pixelOffset
                        y = pixelPos[1] + offsets[1] * pixelOffset
                        label = Text2D(marker['text'], x, y,
                                       color=marker['color'],
                                       bgColor=(1., 1., 1., 0.5),
                                       align=align, valign=valign)
                        break  # Stop at first intersection

            xMin, xMax = corners[:, 0].min(), corners[:, 0].max()
            vertices = np.array(
                ((xMin, yCoord), (xMax, yCoord)), dtype=np.float32)

        else:  # yCoord is None: vertical line in data space
            if marker['text'] is not None:
                # Find intersection of hline with borders in data
                # Order is important as it stops at first intersection
                for border_name in ('top', 'bottom', 'right', 'left'):
                    (x0, y0), (x1, y1) = borders[border_name]
                    if xCoord >= min(x0, x1) and xCoord < max(x0, x1):
                        yIntersect = (xCoord - x0) * (y1 - y0) / (x1 - x0) + y0

                        # Add text label
                        pixelPos = self.dataToPixel(
                            xCoord, yIntersect, check=False)

                        align, valign, offsets = textLayouts[border_name]

                        x = pixelPos[0] + offsets[0] * pixelOffset
                        y = pixelPos[1] + offsets[1] * pixelOffset
                        label = Text2D(marker['text'], x, y,
                                       color=marker['color'],
                                       bgColor=(1., 1., 1., 0.5),
                                       align=align, valign=valign)
                        break  # Stop at first intersection

            yMin, yMax = corners[:, 1].min(), corners[:, 1].max()
            vertices = np.array(
                ((xCoord, yMin), (xCoord, yMax)), dtype=np.float32)

        return vertices, label

    def _renderMarkersGL(self):
        if len(self._markers) == 0:
            return

        plotWidth, plotHeight = self.plotSizeInPixels()

        isXLog = self._plotFrame.xAxis.isLog
        isYLog = self._plotFrame.yAxis.isLog

        # Render in plot area
        glScissor(self._plotFrame.margins.left, self._plotFrame.margins.bottom,
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

        glViewport(self._plotFrame.margins.left,
                   self._plotFrame.margins.bottom,
                   plotWidth, plotHeight)

        # Prepare vertical and horizontal markers rendering
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self._plotFrame.transformedDataProjMat)
        glUniform2i(self._progBase.uniforms['isLog'], isXLog, isYLog)
        glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)
        posAttrib = self._progBase.attributes['position']

        labels = []
        pixelOffset = 3

        for marker in self._markers.values():
            xCoord, yCoord = marker['x'], marker['y']

            if ((isXLog and xCoord is not None and
                    xCoord < FLOAT32_MINPOS) or
                    (isYLog and yCoord is not None and
                     yCoord < FLOAT32_MINPOS)):
                # Do not render markers with negative coords on log axis
                continue

            if xCoord is None or yCoord is None:
                if not self.isDefaultBaseVectors():  # Non-orthogonal axes
                    vertices, label = self._nonOrthoAxesLineMarkerPrimitives(
                        marker, pixelOffset)
                    if label is not None:
                        labels.append(label)

                else:  # Orthogonal axes
                    pixelPos = self.dataToPixel(xCoord, yCoord, check=False)

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

                        xMin, xMax = self._plotFrame.dataRanges.x
                        vertices = np.array(((xMin, yCoord),
                                             (xMax, yCoord)),
                                            dtype=np.float32)

                    else:  # yCoord is None: vertical line in data space
                        if marker['text'] is not None:
                            x = pixelPos[0] + pixelOffset
                            y = self._plotFrame.margins.top + pixelOffset
                            label = Text2D(marker['text'], x, y,
                                           color=marker['color'],
                                           bgColor=(1., 1., 1., 0.5),
                                           align=LEFT, valign=TOP)
                            labels.append(label)

                        yMin, yMax = self._plotFrame.dataRanges.y
                        vertices = np.array(((xCoord, yMin),
                                             (xCoord, yMax)),
                                            dtype=np.float32)

                self._progBase.use()

                glUniform4f(self._progBase.uniforms['color'], *marker['color'])

                glEnableVertexAttribArray(posAttrib)
                glVertexAttribPointer(posAttrib,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      0, vertices)
                glLineWidth(1)
                glDrawArrays(GL_LINES, 0, len(vertices))

            else:
                pixelPos = self.dataToPixel(xCoord, yCoord, check=True)
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
                    np.array((xCoord,), dtype=np.float32),
                    np.array((yCoord,), dtype=np.float32),
                    marker=marker['symbol'],
                    markerColor=marker['color'],
                    markerSize=11)
                markerCurve.render(self._plotFrame.transformedDataProjMat,
                                   isXLog, isYLog)

        glViewport(0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

        # Render marker labels
        for label in labels:
            label.render(self.matScreenProj)

        glDisable(GL_SCISSOR_TEST)

    def _renderOverlayGL(self):
        # Render selection area and crosshair cursor
        if self._selectionAreas or self._crosshairCursor is not None:
            plotWidth, plotHeight = self.plotSizeInPixels()

            # Scissor to plot area
            glScissor(self._plotFrame.margins.left,
                      self._plotFrame.margins.bottom,
                      plotWidth, plotHeight)
            glEnable(GL_SCISSOR_TEST)

            self._progBase.use()
            glUniform2i(self._progBase.uniforms['isLog'],
                        self._plotFrame.xAxis.isLog,
                        self._plotFrame.yAxis.isLog)
            glUniform1f(self._progBase.uniforms['tickLen'], 0.)
            posAttrib = self._progBase.attributes['position']
            matrixUnif = self._progBase.uniforms['matrix']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']

            # Render selection area in plot area
            if self._selectionAreas:
                glViewport(self._plotFrame.margins.left,
                           self._plotFrame.margins.bottom,
                           plotWidth, plotHeight)

                glUniformMatrix4fv(matrixUnif, 1, GL_TRUE,
                                   self._plotFrame.transformedDataProjMat)

                for shape in self._selectionAreas.values():
                    if shape.isVideoInverted:
                        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO)

                    shape.render(posAttrib, colorUnif, hatchStepUnif)

                    if shape.isVideoInverted:
                        self._setBlendFuncGL()

            # Render crosshair cursor is screen frame but with scissor
            if (self._crosshairCursor is not None and
                    self._mousePosInPixels is not None):
                glViewport(
                    0, 0, self._plotFrame.size[0], self._plotFrame.size[1])

                glUniformMatrix4fv(matrixUnif, 1, GL_TRUE,
                                   self.matScreenProj)

                color, lineWidth = self._crosshairCursor
                glUniform4f(colorUnif, *color)
                glUniform1i(hatchStepUnif, 0)

                xPixel, yPixel = self._mousePosInPixels
                xPixel, yPixel = xPixel + 0.5, yPixel + 0.5
                vertices = np.array(((0., yPixel),
                                     (self._plotFrame.size[0], yPixel),
                                     (xPixel, 0.),
                                     (xPixel, self._plotFrame.size[1])),
                                    dtype=np.float32)

                glEnableVertexAttribArray(posAttrib)
                glVertexAttribPointer(posAttrib,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      0, vertices)
                glLineWidth(lineWidth)
                glDrawArrays(GL_LINES, 0, len(vertices))

            glDisable(GL_SCISSOR_TEST)

    def _renderPlotAreaGL(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        self._plotFrame.renderGrid()

        glScissor(self._plotFrame.margins.left,
                  self._plotFrame.margins.bottom,
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

        # Matrix
        trBounds = self._plotFrame.transformedDataRanges
        if trBounds.x[0] == trBounds.x[1] or \
           trBounds.y[0] == trBounds.y[1]:
            return

        isXLog = self._plotFrame.xAxis.isLog
        isYLog = self._plotFrame.yAxis.isLog

        glViewport(self._plotFrame.margins.left,
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
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self._plotFrame.transformedDataProjMat)
        glUniform2i(self._progBase.uniforms['isLog'],
                    self._plotFrame.xAxis.isLog,
                    self._plotFrame.yAxis.isLog)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)

        for item in self._items.values():
            shape2D = item.get('_shape2D')
            if shape2D is None:
                shape2D = Shape2D(tuple(zip(item['x'], item['y'])),
                                  fill=item['fill'],
                                  fillColor=item['color'],
                                  stroke=True,
                                  strokeColor=item['color'])
                item['_shape2D'] = shape2D

            if ((isXLog and shape2D.xMin < FLOAT32_MINPOS) or
                    (isYLog and shape2D.yMin < FLOAT32_MINPOS)):
                # Ignore items <= 0. on log axes
                continue

            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            shape2D.render(posAttrib, colorUnif, hatchStepUnif)

        glDisable(GL_SCISSOR_TEST)

    def resizeGL(self, width, height):
        self._plotFrame.size = width, height

        self.matScreenProj = mat4Ortho(0, self._plotFrame.size[0],
                                       self._plotFrame.size[1], 0,
                                       1, -1)

        (xMin, xMax), (yMin, yMax), (y2Min, y2Max) = \
            self._plotFrame.dataRanges
        self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

    # PlotBackend API #

    def insertMarker(self, x, y, legend=None, text=None, color='k',
                     selectable=False, draggable=False,
                     symbol=None, constraint=None,
                     **kw):
        if symbol is not None:
            warnings.warn("insertMarker ignores the symbol parameter",
                          RuntimeWarning)
        if kw:
            warnings.warn("insertMarker ignores additional parameters",
                          RuntimeWarning)
        if legend is None:
            legend = self._UNNAMED_ITEM

        if symbol is None:
            symbol = '+'

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

        if x is not None and self._plotFrame.xAxis.isLog and x <= 0.:
            raise RuntimeError(
                'Cannot add marker with X <= 0 with X axis log scale')
        if y is not None and self._plotFrame.yAxis.isLog and y <= 0.:
            raise RuntimeError(
                'Cannot add marker with Y <= 0 with Y axis log scale')

        self._markers[legend] = {
            'x': x,
            'y': y,
            'legend': legend,
            'text': text,
            'color': rgba(color, PlotBackend.COLORDICT),
            'behaviors': behaviors,
            'constraint': constraint if isConstraint else None,
            'symbol': symbol,
        }

        self._plotDirtyFlag = True

        return legend

    def insertXMarker(self, x, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kw):
        if kw:
            warnings.warn("insertXMarker ignores additional parameters",
                          RuntimeWarning)
        return self.insertMarker(x, None, legend, text, color,
                                 selectable, draggable, **kw)

    def insertYMarker(self, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kw):
        if kw:
            warnings.warn("insertYMarker ignores additional parameters",
                          RuntimeWarning)
        return self.insertMarker(None, y, legend, text, color,
                                 selectable, draggable, **kw)

    def removeMarker(self, legend, replot=True):
        try:
            del self._markers[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearMarkers(self):
        self._markers = MiniOrderedDict()
        self._plotDirtyFlag = True

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=0,
                 selectable=False, draggable=False,
                 colormap=None, **kw):
        if info is not None:
            warnings.warn("Ignore info parameter of addImage",
                          RuntimeWarning)
        if kw:
            warnings.warn("addImage ignores additional parameters",
                          RuntimeWarning)

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        if legend is None:
            legend = self._UNNAMED_ITEM

        oldImage = self._plotContent.get('image', legend)
        if oldImage is not None and oldImage.data.shape != data.shape:
            oldImage = None
            self.removeImage(legend)

        if replace:
            self.clearImages()

        if xScale is None:
            xScale = (0, 1)
        if yScale is None:
            yScale = (0, 1)

        if len(data.shape) == 2:
            # Ensure array is contiguous and eventually convert its type
            if np.dtype(data.dtype).kind == 'f' and data.dtype != np.float32:
                warnings.warn(
                    'addImage: Convert %s data to float32' % str(data.dtype),
                    RuntimeWarning)
                data = np.array(data, dtype=np.float32, order='C')
            else:
                data = np.array(data, copy=False, order='C')
            assert data.dtype in (np.float32, np.uint8, np.uint16)

            if colormap is None:
                colormap = self.getDefaultColormap()

            if colormap['normalization'] not in ('linear', 'log'):
                raise NotImplementedError(
                    "Normalisation: {0}".format(colormap['normalization']))
            if colormap['colors'] != 256:
                raise NotImplementedError(
                    "Colors: {0}".format(colormap['colors']))

            colormapIsLog = colormap['normalization'].startswith('log')

            if colormap['autoscale']:
                cmapRange = None
            else:
                cmapRange = colormap['vmin'], colormap['vmax']
                assert cmapRange[0] <= cmapRange[1]

            if oldImage is not None:  # TODO check if benefit
                image = oldImage
                image.origin = xScale[0], yScale[0]
                image.scale = xScale[1], yScale[1]
                image.colormap = colormap['name'][:]
                image.cmapIsLog = colormapIsLog
                image.cmapRange = cmapRange
                image.updateData(data)
            else:
                image = GLPlotColormap(data,
                                       (xScale[0], yScale[0]),  # origin
                                       (xScale[1], yScale[1]),  # scale
                                       colormap['name'][:],
                                       colormapIsLog,
                                       cmapRange)
            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }
            self._plotContent.add(image)

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert data.shape[2] in (3, 4)
            assert data.dtype in (np.float32, np.uint8)

            if oldImage is not None:
                image = oldImage
                image.origin = xScale[0], yScale[0]
                image.scale = xScale[1], yScale[1]
                image.updateData(data)
            else:
                image = GLPlotRGBAImage(data,
                                        origin=(xScale[0], yScale[0]),
                                        scale=(xScale[1], yScale[1]))

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

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend  # This is the 'handle'

    def removeImage(self, legend, replot=True):
        if legend is None:
            legend = self._UNNAMED_ITEM

        image = self._plotContent.pop('image', legend)
        if image is not None:
            self._glGarbageCollector.append(image)
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearImages(self):
        # Copy keys as it removes primitives from the dict
        for legend in list(self._plotContent.primitiveKeys('image')):
            self.removeImage(legend, replot=False)

    def addItem(self, xList, yList, legend=None, info=None,
                replace=False, replot=True,
                shape="polygon", fill=True, color=None, **kw):
        # info is ignored
        if shape not in ('polygon', 'rectangle', 'line', 'vline', 'hline'):
            raise NotImplementedError("Unsupported shape {0}".format(shape))
        if kw:
            warnings.warn("addItem ignores additional parameters",
                          RuntimeWarning)

        if legend is None:
            legend = self._UNNAMED_ITEM

        if replace:
            self.clearItems()

        colorCode = color if color is not None else 'black'

        if shape == 'rectangle':
            xMin, xMax = xList
            xList = np.array((xMin, xMin, xMax, xMax))
            yMin, yMax = yList
            yList = np.array((yMin, yMax, yMax, yMin))
        else:
            xList = np.array(xList, copy=False)
            yList = np.array(yList, copy=False)

        if self._plotFrame.xAxis.isLog and xList.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with X <= 0 with X axis log scale')
        if self._plotFrame.yAxis.isLog and yList.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with Y <= 0 with Y axis log scale')

        self._items[legend] = {
            'shape': shape,
            'color': rgba(colorCode, PlotBackend.COLORDICT),
            'fill': 'hatch' if fill else None,
            'x': xList,
            'y': yList
        }
        self._plotDirtyFlag = True

        if replot:
            self.replot()
        return legend  # this is the 'handle'

    def removeItem(self, legend, replot=True):
        if legend is None:
            legend = self._UNNAMED_ITEM

        try:
            del self._items[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearItems(self):
        self._items = MiniOrderedDict()
        self._plotDirtyFlag = True

    def addCurve(self, x, y, legend=None, info=None,
                 replace=False, replot=True,
                 color=None, symbol=None, linewidth=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=1, selectable=True,
                 fill=None, **kw):
        if kw:
            warnings.warn("addCurve ignores additional parameters",
                          RuntimeWarning)

        if legend is None:
            legend = self._UNNAMED_ITEM

        x = np.array(x, dtype=np.float32, copy=False, order='C')
        y = np.array(y, dtype=np.float32, copy=False, order='C')
        if xerror is not None:
            xerror = np.array(xerror, dtype=np.float32, copy=False, order='C')
            assert np.all(xerror >= 0.)
        if yerror is not None:
            yerror = np.array(yerror, dtype=np.float32, copy=False, order='C')
            assert np.all(yerror >= 0.)

        behaviors = set()
        if selectable:
            behaviors.add('selectable')

        wasActiveCurve = (legend == self._activeCurveLegend)
        oldCurve = self._plotContent.get('curve', legend)
        if oldCurve is not None:
            self.removeCurve(legend)

        if replace:
            self.clearCurves()

        if color is None:
            color = self._activeCurveColor

        if isinstance(color, np.ndarray) and len(color) > 4:
            colorArray = color
            color = None
        else:
            colorArray = None
            color = rgba(color, PlotBackend.COLORDICT)

        if fill is None and info is not None:  # To make it run with Plot.py
            fill = info.get('plot_fill', False)

        curve = GLPlotCurve2D(x, y, colorArray,
                              xError=xerror,
                              yError=yerror,
                              lineStyle=linestyle,
                              lineColor=color,
                              lineWidth=1 if linewidth is None else linewidth,
                              marker=symbol,
                              markerColor=color,
                              fillColor=color if fill else None)
        curve.info = {
            'legend': legend,
            'zOrder': z,
            'behaviors': behaviors,
            'xLabel': xlabel,
            'yLabel': ylabel,
            'yAxis': 'left' if yaxis is None else yaxis,
        }

        if yaxis == "right":
            self._plotFrame.isY2Axis = True

        self._plotContent.add(curve)

        self._plotDirtyFlag = True
        self._resetZoom()

        if wasActiveCurve:
            self.setActiveCurve(legend, replot=False)

        if replot:
            self.replot()

        return legend

    def removeCurve(self, legend, replot=True):
        if legend is None:
            legend = self._UNNAMED_ITEM

        curve = self._plotContent.pop('curve', legend)
        if curve is not None:
            # Check if some curves remains on the right Y axis
            y2AxisItems = (item for item in self._plotContent.primitives()
                           if item.info.get('yAxis', 'left') == 'right')
            self._plotFrame.isY2Axis = (next(y2AxisItems, None) is not None)

            self._glGarbageCollector.append(curve)
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearCurves(self):
        # Copy keys as dict is changed
        for legend in list(self._plotContent.primitiveKeys('curve')):
            self.removeCurve(legend, replot=False)

    def setActiveCurve(self, legend, replot=True):
        if not self._activeCurveHandling:
            return

        if legend is None:
            legend = self._UNNAMED_ITEM

        curve = self._plotContent.get('curve', legend)
        if curve is None:
            raise KeyError("Curve %s not found" % legend)

        if self._activeCurveLegend is not None:
            activeCurve = self._plotContent.get('curve',
                                                self._activeCurveLegend)
            # _inactiveState might not exists as
            # _activeCurveLegend is not reset when curve is removed.
            inactiveState = getattr(activeCurve, '_inactiveState', None)
            if inactiveState is not None:
                del activeCurve._inactiveState
                activeCurve.lineColor = inactiveState['lineColor']
                activeCurve.markerColor = inactiveState['markerColor']
                activeCurve.useColorVboData = inactiveState['useColorVbo']
                self.setGraphXLabel(inactiveState['xLabel'])
                self.setGraphYLabel(inactiveState['yLabel'])

        curve._inactiveState = {'lineColor': curve.lineColor,
                                'markerColor': curve.markerColor,
                                'useColorVbo': curve.useColorVboData,
                                'xLabel': self.getGraphXLabel(),
                                'yLabel': self.getGraphYLabel()}

        if curve.info['xLabel'] is not None:
            self.setGraphXLabel(curve.info['xLabel'])
        if curve.info['yAxis'] == 'left' and curve.info['yLabel'] is not None:
            self.setGraphYLabel(curve.info['yLabel'])

        color = rgba(self._activeCurveColor, PlotBackend.COLORDICT)
        curve.lineColor = color
        curve.markerColor = color
        curve.useColorVboData = False
        self._activeCurveLegend = legend

        if replot:
            self.replot()

    def clear(self):
        self.clearCurves()
        self.clearImages()
        self.clearItems()
        self.clearMarkers()

    def replot(self):
        self.postRedisplay()

    # Interaction modes #
    def getInteractiveMode(self):
        return self.eventHandler.getInteractiveMode()

    def setInteractiveMode(self, mode, color=None,
                           shape='polygon', label=None):
        self.eventHandler.setInteractiveMode(mode, color, shape, label)

    def isDrawModeEnabled(self):
        return self.getInteractiveMode()['mode'] == 'draw'

    def setDrawModeEnabled(self, flag=True, shape='polygon', label=None,
                           color=None, **kwargs):
        if kwargs:
            warnings.warn('setDrawModeEnabled ignores additional parameters',
                          RuntimeWarning)

        if flag:
            self.setInteractiveMode('draw', shape=shape,
                                    label=label, color=color)
        elif self.getInteractiveMode()['mode'] == 'draw':
            self.setInteractiveMode('select')

    def getDrawMode(self):
        mode = self.getInteractiveMode()
        return mode if mode['mode'] == 'draw' else None

    def isZoomModeEnabled(self):
        return self.getInteractiveMode()['mode'] == 'zoom'

    def setZoomModeEnabled(self, flag=True, color=None):
        if flag:
            self.setInteractiveMode('zoom', color=color)
        elif self.getInteractiveMode()['mode'] == 'zoom':
            self.setInteractiveMode('select')

    # Zoom #

    def isXAxisAutoScale(self):
        return self._xAutoScale

    def setXAxisAutoScale(self, flag=True):
        self._xAutoScale = flag

    def isYAxisAutoScale(self):
        return self._yAutoScale

    def setYAxisAutoScale(self, flag=True):
        self._yAutoScale = flag

    def _resetZoom(self, dataMargins=None, forceAutoscale=False):
        dataBounds = self._plotContent.getBounds(
            self.isXAxisLogarithmic(), self.isYAxisLogarithmic())

        if forceAutoscale:
            isXAuto, isYAuto = True, True
        else:
            isXAuto, isYAuto = self.isXAxisAutoScale(), self.isYAxisAutoScale()

        xMin, xMax, yMin, yMax, y2Min, y2Max = _utils.addMarginsToLimits(
            dataMargins,
            self.isXAxisLogarithmic(), self.isYAxisLogarithmic(),
            dataBounds.xAxis.min_, dataBounds.xAxis.max_,
            dataBounds.yAxis.min_, dataBounds.yAxis.max_,
            dataBounds.y2Axis.min_, dataBounds.y2Axis.max_)

        if isXAuto and isYAuto:
            self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        elif isXAuto:
            self.setGraphXLimits(xMin, xMax)

        elif isYAuto:
            xMin, xMax = self.getGraphXLimits()
            self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

    def resetZoom(self, dataMargins=None):
        self._resetZoom(dataMargins)
        self.replot()

    # Limits #

    def _setDataRanges(self, x=None, y=None, y2=None):
        """Set the visible range of data in the plot frame.

        This clips the ranges to possible values (takes care of float32
        range + positive range for log).
        This also takes care of non-orthogonal axes.

        This should be moved to PlotFrame.
        """
        # Update axes range with a clipped range if too wide
        self._plotFrame.setDataRanges(x, y, y2)

        if not self.isDefaultBaseVectors():
            # Update axes range with axes bounds in data coords
            plotLeft, plotTop = self.plotOriginInPixels()
            plotWidth, plotHeight = self.plotSizeInPixels()

            self._plotFrame.xAxis.dataRange = sorted([
                self.pixelToData(x, y, check=False)[0] for (x, y) in
                ((plotLeft, plotTop + plotHeight),
                 (plotLeft + plotWidth, plotTop + plotHeight))])

            self._plotFrame.yAxis.dataRange = sorted([
                self.pixelToData(x, y, check=False)[1] for (x, y) in
                ((plotLeft, plotTop + plotHeight),
                 (plotLeft, plotTop))])

            self._plotFrame.y2Axis.dataRange = sorted([
                self.pixelToData(x, y, axis='right', check=False)[1]
                for (x, y) in
                ((plotLeft + plotWidth, plotTop + plotHeight),
                 (plotLeft + plotWidth, plotTop))])

    def _ensureAspectRatio(self, keepDim=None):
        """Update plot bounds in order to keep aspect ratio.

        Warning: keepDim on right Y axis is not implemented !

        :param str keepDim: The dimension to maintain: 'x', 'y' or None.
            If None (the default), the dimension with the largest range.
        """
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if keepDim is None:
            dataBounds = self._plotContent.getBounds(
                self.isXAxisLogarithmic(), self.isYAxisLogarithmic())
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
        self._setDataRanges(x=(xMin, xMax), y=(yMin, yMax), y2=(y2Min, y2Max))

    def _setPlotBounds(self, xRange=None, yRange=None, y2Range=None,
                       keepDim=None):
        # Update axes range with a clipped range if too wide
        self._setDataRanges(x=xRange, y=yRange, y2=y2Range)

        # Keep data aspect ratio
        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio(keepDim)

        # Raise dirty flags
        self._plotDirtyFlag = True

        # Send limits changed to callback
        dataRanges = self._plotFrame.dataRanges

        eventDict = prepareLimitsChangedSignal(
            self.getWidgetHandle(),
            dataRanges.x,
            dataRanges.y,
            dataRanges.y2 if self._plotFrame.isY2Axis else None)
        self.sendEvent(eventDict)

    def isKeepDataAspectRatio(self):
        if self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog:
            return False
        else:
            return self._keepDataAspectRatio

    def keepDataAspectRatio(self, flag=True):
        if flag and (self._plotFrame.xAxis.isLog or
                     self._plotFrame.yAxis.isLog):
            warnings.warn("KeepDataAspectRatio is ignored with log axes",
                          RuntimeWarning)
        if flag and not self.isDefaultBaseVectors():
            warnings.warn(
                "keepDataAspectRatio ignored because baseVectors are set",
                RuntimeWarning)

        self._keepDataAspectRatio = flag

        self.resetZoom()

    def getGraphXLimits(self):
        return self._plotFrame.dataRanges.x

    def setGraphXLimits(self, xMin, xMax):
        assert xMin < xMax
        self._setPlotBounds(xRange=(xMin, xMax), keepDim='x')

    def getGraphYLimits(self, axis="left"):
        assert axis in ("left", "right")
        if axis == "left":
            return self._plotFrame.dataRanges.y
        else:
            return self._plotFrame.dataRanges.y2

    def setGraphYLimits(self, yMin, yMax, axis="left"):
        assert yMin < yMax
        assert axis in ("left", "right")

        if axis == "left":
            self._setPlotBounds(yRange=(yMin, yMax), keepDim='y')
        else:
            self._setPlotBounds(y2Range=(yMin, yMax), keepDim='y')

    def setLimits(self, xMin, xMax, yMin, yMax, y2Min=None, y2Max=None):
        assert xMin < xMax
        assert yMin < yMax

        if y2Min is None or y2Max is None:
            y2Range = None
        else:
            assert y2Min < y2Max
            y2Range = y2Min, y2Max
        self._setPlotBounds((xMin, xMax), (yMin, yMax), y2Range)

    def invertYAxis(self, flag=True):
        if flag != self._plotFrame.isYAxisInverted:
            self._plotFrame.isYAxisInverted = flag
            self._plotDirtyFlag = True

    def isYAxisInverted(self):
        return self._plotFrame.isYAxisInverted

    # Log axis #

    def setXAxisLogarithmic(self, flag=True):
        if flag != self._plotFrame.xAxis.isLog:
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            if flag and not self.isDefaultBaseVectors():
                warnings.warn(
                    "setXAxisLogarithmic ignored because baseVectors are set",
                    RuntimeWarning)
                return

            self._plotFrame.xAxis.isLog = flag

            # With log axis on, force autoscale to avoid limits <= 0
            if flag:
                self._resetZoom(forceAutoscale=True)

    def setYAxisLogarithmic(self, flag=True):
        if (flag != self._plotFrame.yAxis.isLog or
                flag != self._plotFrame.y2Axis.isLog):
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            if flag and not self.isDefaultBaseVectors():
                warnings.warn(
                    "setYAxisLogarithmic ignored because baseVectors are set",
                    RuntimeWarning)
                return

            self._plotFrame.yAxis.isLog = flag
            self._plotFrame.y2Axis.isLog = flag

            # With log axis on, force autoscale to avoid limits <= 0
            if flag:
                self._resetZoom(forceAutoscale=True)

    def isXAxisLogarithmic(self):
        return self._plotFrame.xAxis.isLog

    def isYAxisLogarithmic(self):
        return self._plotFrame.yAxis.isLog

    # Non orthogonal axes

    def setBaseVectors(self, x=(1., 0.), y=(0., 1.)):
        """Set base vectors.

        Useful for non-orthogonal axes.
        If an axis is in log scale, skew is applied to log transformed values.

        Base vector does not work well with log axes, to investi
        """
        if x != (1., 0.) and y != (0., 1.):
            if self.isXAxisLogarithmic():
                warnings.warn("setBaseVectors disables X axis logarithmic.",
                              RuntimeWarning)
                self.setXAxisLogarithmic(False)
            if self.isYAxisLogarithmic():
                warnings.warn("setBaseVectors disables Y axis logarithmic.",
                              RuntimeWarning)
                self.setYAxisLogarithmic(False)

            if self.isKeepDataAspectRatio():
                warnings.warn("setBaseVectors disables keepDataAspectRatio.",
                              RuntimeWarning)
                self.keepDataAspectRatio(False)

        self._plotFrame.baseVectors = x, y
        self._plotDirtyFlag = True
        self.resetZoom()

    def getBaseVectors(self):
        return self._plotFrame.baseVectors

    def isDefaultBaseVectors(self):
        return self._plotFrame.baseVectors == \
            self._plotFrame.DEFAULT_BASE_VECTORS

    # Title, Labels

    def setGraphTitle(self, title=""):
        self._plotFrame.title = title

    def getGraphTitle(self):
        return self._plotFrame.title

    def setGraphXLabel(self, label="X"):
        self._plotFrame.xAxis.title = label
        self._plotDirtyFlag = True

    def getGraphXLabel(self):
        return self._plotFrame.xAxis.title

    def setGraphYLabel(self, label="Y"):
        self._plotFrame.yAxis.title = label
        self._plotDirtyFlag = True

    def getGraphYLabel(self):
        return self._plotFrame.yAxis.title

    def showGrid(self, flag=True):
        self._plotFrame.grid = flag
        self._plotDirtyFlag = True
        self.replot()

    # Cursor

    def setGraphCursor(self, flag=True, color=None,
                       linewidth=1, linestyle=None):
        if linestyle is not None:
            warnings.warn(
                "OpenGLBackend.setGraphCursor linestyle parameter ignored",
                RuntimeWarning)

        if flag:
            # Default values
            if color is None:
                color = 'black'
            if linewidth is None:
                linewidth = 1

            color = rgba(color, PlotBackend.COLORDICT)
            crosshairCursor = color, linewidth
        else:
            crosshairCursor = None

        if crosshairCursor != self._crosshairCursor:
            self._crosshairCursor = crosshairCursor
            self.replot()

    def getGraphCursor(self):
        return self._crosshairCursor

    # Save
    def saveGraph(self, fileName, fileFormat='svg', dpi=None, **kw):
        """Save the graph as an image to a file.

        WARNING: This method is performing some OpenGL calls.
        It must be called from the main thread.
        """
        if dpi is not None:
            warnings.warn("saveGraph ignores dpi parameter",
                          RuntimeWarning)
        if kw:
            warnings.warn("saveGraph ignores additional parameters",
                          RuntimeWarning)

        if fileFormat not in ['png', 'ppm', 'svg', 'tiff']:
            raise NotImplementedError('Unsupported format: %s' % fileFormat)

        self.makeCurrent()

        data = np.empty((self._plotFrame.size[1], self._plotFrame.size[0], 3),
                        dtype=np.uint8, order='C')

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(0, 0, self._plotFrame.size[0], self._plotFrame.size[1],
                     GL_RGB, GL_UNSIGNED_BYTE, data)

        # glReadPixels gives bottom to top,
        # while images are stored as top to bottom
        data = np.flipud(data)

        # fileName is either a file-like object or a str
        saveImageToFile(data, fileName, fileFormat)
