# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
__authors__ = ["V.A. Sole - ESRF Data Analysis", "T. Vincent"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Matplotlib Plot backend.
"""

import weakref

import matplotlib

# blitting enabled by default
# it provides faster response at the cost of missing minor updates
# during movement (only the bounding box of the moving object is updated)
# For instance, when moving a marker, the label is not updated during the
# movement.
BLITTING = True

import numpy
from numpy import vstack as numpyvstack

from numpy import nanmax, nanmin

import sys

from matplotlib import cm

from matplotlib.widgets import Cursor

# TODO from .. import qt
# if qt.BINDING

# TODO make all this simpler and somewhere else
if ('PySide' in sys.modules) or ('PySide' in sys.argv):
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    matplotlib.rcParams['backend.qt4'] = 'PySide'
    from PySide import QtCore, QtGui
elif ("PyQt4" in sys.modules) or ('PyQt4' in sys.argv):
    from PyQt4 import QtCore, QtGui
    matplotlib.rcParams['backend'] = 'Qt4Agg'
elif ('PyQt5' in sys.modules):
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    from PyQt5 import QtCore, QtGui, QtWidgets
    QtGui.QApplication = QtWidgets.QApplication
else:
    try:
        from PyQt4 import QtCore, QtGui
        matplotlib.rcParams['backend'] = 'Qt4Agg'
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
            QtGui.QApplication = QtWidgets.QApplication
            matplotlib.rcParams['backend'] = 'Qt5Agg'
        except ImportError:
            from PySide import QtCore, QtGui

if ("PyQt4" in sys.modules) or ("PySide" in sys.modules):
    from matplotlib.backends.backend_qt4agg import \
        FigureCanvasQTAgg as FigureCanvas
elif "PyQt5" in sys.modules:
    from matplotlib.backends.backend_qt5agg import \
        FigureCanvasQTAgg as FigureCanvas


from matplotlib.figure import Figure
import matplotlib.patches as patches
Rectangle = patches.Rectangle
Polygon = patches.Polygon

from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from matplotlib.image import AxesImage
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import time

from . import _utils
from .ModestImage import ModestImage

import logging


logging.basicConfig()
logger = logging.getLogger(__name__)


class MatplotlibGraph(FigureCanvas):
    def __init__(self, parent=None, **kw):
        self.fig = Figure()
        self._originalCursorShape = QtCore.Qt.ArrowCursor
        FigureCanvas.__init__(self, self.fig)

        self.fig.set_facecolor("w")

        self.ax = self.fig.add_axes([.15, .15, .75, .75], label="left")
        self.ax2 = self.ax.twinx()
        self.ax2.set_label("right")

        # critical for picking!!!!
        self.ax2.set_zorder(0)
        self.ax2.set_autoscaley_on(True)
        self.ax.set_zorder(1)
        # this works but the figure color is left
        self.ax.set_axis_bgcolor('none')
        self.fig.sca(self.ax)

        # This should be independent of Qt
        if ("PyQt4" in sys.modules) or ("PySide" in sys.modules):
            FigureCanvas.setSizePolicy(
                self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.__lastMouseClick = ["middle", time.time()]
        self._zoomEnabled = False
        self._zoomColor = "black"
        self.__zooming = False
        self.__picking = False
        self._background = None
        self.__markerMoving = False
        self._zoomStack = []

        # info text
        self._infoText = None

        # drawingmode handling
        self.setDrawModeEnabled(False)
        self.__drawModeList = \
            ['line', 'hline', 'vline', 'rectangle', 'polygon']
        self.__drawing = False
        self._drawingPatch = None
        self._drawModePatch = 'line'

        # event handling
        self._callback = self._dummyCallback
        self._x0 = None
        self._y0 = None
        self._zoomRectangle = None
        self.fig.canvas.mpl_connect('button_press_event',
                                    self.onMousePressed)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.onMouseReleased)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.onMouseMoved)
        self.fig.canvas.mpl_connect('scroll_event',
                                    self.onMouseWheel)
        self.fig.canvas.mpl_connect('pick_event',
                                    self.onPick)

    def _dummyCallback(self, ddict):
        logger.info(str(ddict))

    def setCallback(self, callbackFuntion):
        self._callback = callbackFuntion

    def onPick(self, event):
        # Unfortunately only the artists on the top axes
        # can be picked -> A legend handling widget is
        # needed
        middleButton = 2
        rightButton = 3
        button = event.mouseevent.button

        if button == middleButton:
            # do nothing with the midle button
            return
        elif button == rightButton:
            button = "right"
        else:
            button = "left"

        if self._drawModeEnabled:
            # forget about picking or zooming
            # should one disconnect when setting the mode?
            return

        self.__picking = False
        self._pickingInfo = {}

        if isinstance(event.artist, Line2D) or \
           isinstance(event.artist, PathCollection):
            # we only handle curves and markers for the time being
            self.__picking = True
            artist = event.artist
            label = artist.get_label()
            ind = event.ind
            self._pickingInfo['artist'] = artist
            self._pickingInfo['event_ind'] = ind

            if label.startswith("__MARKER__"):
                label = label[10:]
                self._pickingInfo['type'] = 'marker'
                self._pickingInfo['label'] = label
                if 'draggable' in artist._plot_options:
                    self._pickingInfo['draggable'] = True
                else:
                    self._pickingInfo['draggable'] = False
                if 'selectable' in artist._plot_options:
                    self._pickingInfo['selectable'] = True
                else:
                    self._pickingInfo['selectable'] = False
                if hasattr(artist, "_infoText"):
                    self._pickingInfo['infoText'] = artist._infoText
                else:
                    self._pickingInfo['infoText'] = None

            elif isinstance(event.artist, PathCollection):
                # almost identical to line 2D
                self._pickingInfo['type'] = 'curve'
                self._pickingInfo['label'] = label
                self._pickingInfo['artist'] = artist
                data = artist.get_offsets()
                xdata = data[:, 0]
                ydata = data[:, 1]
                self._pickingInfo['xdata'] = xdata[ind]
                self._pickingInfo['ydata'] = ydata[ind]
                self._pickingInfo['infoText'] = None

            else:
                # line2D
                self._pickingInfo['type'] = 'curve'
                self._pickingInfo['label'] = label
                self._pickingInfo['artist'] = artist
                xdata = artist.get_xdata()
                ydata = artist.get_ydata()
                self._pickingInfo['xdata'] = xdata[ind]
                self._pickingInfo['ydata'] = ydata[ind]
                self._pickingInfo['infoText'] = None

            if self._pickingInfo['infoText'] is None:
                if self._infoText is None:
                    self._infoText = self.ax.text(event.mouseevent.xdata,
                                                  event.mouseevent.ydata,
                                                  label)
                else:
                    self._infoText.set_position((event.mouseevent.xdata,
                                                event.mouseevent.ydata))
                    self._infoText.set_text(label)
                self._pickingInfo['infoText'] = self._infoText

            self._pickingInfo['infoText'].set_visible(True)

            logger.debug("%s %s selected",
                         self._pickingInfo['type'].upper(),
                         self._pickingInfo['label'])

        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            logger.debug('onPick patch: %s', str(patch.get_path()))

        elif isinstance(event.artist, Text):
            text = event.artist
            logger.debug('onPick text: %s', text.get_text())

        elif isinstance(event.artist, AxesImage):
            self.__picking = True
            artist = event.artist
            self._pickingInfo['artist'] = artist
            label = artist.get_label()
            self._pickingInfo['type'] = 'image'
            self._pickingInfo['label'] = label
            self._pickingInfo['draggable'] = False
            self._pickingInfo['selectable'] = False
            if hasattr(artist, "_plot_options"):
                if 'draggable' in artist._plot_options:
                    self._pickingInfo['draggable'] = True
                else:
                    self._pickingInfo['draggable'] = False
                if 'selectable' in artist._plot_options:
                    self._pickingInfo['selectable'] = True
                else:
                    self._pickingInfo['selectable'] = False
        else:
            logger.warning("unhandled event %s", str(event.artist))

    def setDrawModeEnabled(self, flag=True, shape="polygon", label=None,
                           color=None, **kw):
        if flag:
            shape = shape.lower()
            if shape not in self.__drawModeList:
                self._drawModeEnabled = False
                raise ValueError("Unsupported shape %s" % shape)
            else:
                self._drawModeEnabled = True
                self.setZoomModeEnabled(False)
                self._drawModePatch = shape
            self._drawingParameters = kw
            if color is not None:
                self._drawingParameters['color'] = color
            self._drawingParameters['shape'] = shape
            self._drawingParameters['label'] = label
        else:
            self._drawModeEnabled = False

    def setZoomModeEnabled(self, flag=True, color=None):
        if color is None:
            color = self._zoomColor
        if len(color) == 4:
            if type(color[3]) in [type(1), numpy.uint8, numpy.int8]:
                color = numpy.array(color, dtype=numpy.float)/255.
        self._zoomColor = color
        if flag:
            self._zoomEnabled = True
            self.setDrawModeEnabled(False)
        else:
            self._zoomEnabled = False

    def isZoomModeEnabled(self):
        return self._zoomEnabled

    def isDrawModeEnabled(self):
        return self._drawModeEnabled

    def getDrawMode(self):
        if self.isDrawModeEnabled():
            return self._drawingParameters
        else:
            return None

    def onMousePressed(self, event):
        logger.debug("onMousePressed, event = %f %f", event.xdata, event.ydata)
        logger.debug("Mouse button = %s", str(event.button))
        self.__time0 = -1.0
        if event.inaxes != self.ax:
            logger.debug("RETURNING")
            return
        button = event.button
        leftButton = 1
        middleButton = 2
        rightButton = 3

        self._x0 = event.xdata
        self._y0 = event.ydata

        if button == middleButton:
            # by default, do nothing with the middle button
            return

        self._x0Pixel = event.x
        self._y0Pixel = event.y
        self._x1 = event.xdata
        self._y1 = event.ydata
        self._x1Pixel = event.x
        self._y1Pixel = event.y

        self.__movingMarker = 0
        # picking handling
        if self.__picking:
            logger.debug("PICKING, Ignoring zoom")
            self.__zooming = False
            self.__drawing = False
            self.__markerMoving = False
            if self._pickingInfo['type'] == "marker":
                if button == rightButton:
                    # only selection or movement
                    self._pickingInfo = {}
                    return
                artist = self._pickingInfo['artist']
                if button == leftButton:
                    if self._pickingInfo['draggable']:
                        self.__markerMoving = True
                    if self._pickingInfo['selectable']:
                        self.__markerMoving = False
                    if self.__markerMoving:
                        if 'xmarker' in artist._plot_options:
                            artist.set_xdata(event.xdata)
                        elif 'ymarker' in artist._plot_options:
                            artist.set_ydata(event.ydata)
                        else:
                            xData, yData = event.xdata, event.ydata
                            if artist._constraint is not None:
                                # Apply marker constraint
                                xData, yData = artist._constraint(xData, yData)
                            artist.set_xdata(xData)
                            artist.set_ydata(yData)
                    if BLITTING:
                        canvas = artist.figure.canvas
                        axes = artist.axes
                        artist.set_animated(True)
                        canvas.draw()
                        self._background = canvas.copy_from_bbox(axes.bbox)
                        axes.draw_artist(artist)
                        canvas.blit(axes.bbox)
                    else:
                        self.fig.canvas.draw()

                    ddict = {}
                    ddict['label'] = self._pickingInfo['label']
                    ddict['type'] = self._pickingInfo['type']
                    ddict['draggable'] = self._pickingInfo['draggable']
                    ddict['selectable'] = self._pickingInfo['selectable']
                    ddict['xpixel'] = self._x0Pixel
                    ddict['ypixel'] = self._y0Pixel
                    ddict['xdata'] = artist.get_xdata()
                    ddict['ydata'] = artist.get_ydata()

                    if self.__markerMoving:
                        ddict['event'] = "markerMoving"
                        ddict['x'] = self._x0
                        ddict['y'] = self._y0
                    else:
                        ddict['event'] = "markerClicked"
                        if hasattr(ddict['xdata'], "__len__"):
                            ddict['x'] = ddict['xdata'][-1]
                        else:
                            ddict['x'] = ddict['xdata']
                        if hasattr(ddict['ydata'], "__len__"):
                            ddict['y'] = ddict['ydata'][-1]
                        else:
                            ddict['y'] = ddict['ydata']

                    if button == leftButton:
                        ddict['button'] = "left"
                    else:
                        ddict['button'] = "right"
                    self._callback(ddict)
                return

            elif self._pickingInfo['type'] == "curve":
                ddict = {}
                ddict['event'] = "curveClicked"
                ddict['label'] = self._pickingInfo['label']
                ddict['type'] = self._pickingInfo['type']
                ddict['x'] = self._x0
                ddict['y'] = self._y0
                ddict['xpixel'] = self._x0Pixel
                ddict['ypixel'] = self._y0Pixel
                ddict['xdata'] = self._pickingInfo['xdata']
                ddict['ydata'] = self._pickingInfo['ydata']
                if button == leftButton:
                    ddict['button'] = "left"
                else:
                    ddict['button'] = "right"
                self._callback(ddict)
                return

            elif self._pickingInfo['type'] == "image":
                artist = self._pickingInfo['artist']
                ddict = {}
                ddict['event'] = "imageClicked"
                ddict['label'] = self._pickingInfo['label']
                ddict['type'] = self._pickingInfo['type']
                ddict['x'] = self._x0
                ddict['y'] = self._y0
                ddict['xpixel'] = self._x0Pixel
                ddict['ypixel'] = self._y0Pixel
                xScale = artist._plot_info['xScale']
                yScale = artist._plot_info['yScale']
                col = (ddict['x'] - xScale[0])/float(xScale[1])
                row = (ddict['y'] - yScale[0])/float(yScale[1])
                ddict['row'] = int(row)
                ddict['col'] = int(col)
                if button == leftButton:
                    ddict['button'] = "left"
                else:
                    ddict['button'] = "right"
                self.__picking = False
                self._callback(ddict)

        if event.button == rightButton:
            # right click
            self.__zooming = False
            if self._drawingPatch is not None:
                self._emitDrawingSignal("drawingFinished")
            return

        self.__time0 = time.time()
        self.__zooming = self._zoomEnabled
        self._zoomRect = None
        self._xmin, self._xmax = self.ax.get_xlim()
        self._ymin, self._ymax = self.ax.get_ylim()
        # deal with inverted axis
        if self._xmin > self._xmax:
            tmpValue = self._xmin
            self._xmin = self._xmax
            self._xmax = tmpValue
        if self._ymin > self._ymax:
            tmpValue = self._ymin
            self._ymin = self._ymax
            self._ymax = tmpValue

        if self.ax.get_aspect() != 'auto':
            self._ratio = (self._ymax - self._ymin) / (self._xmax - self._xmin)

        self.__drawing = self._drawModeEnabled
        if self.__drawing:
            if self._drawModePatch in ['hline', 'vline']:
                if self._drawingPatch is None:
                    self._mouseData = numpy.zeros((2, 2), numpy.float32)
                    if self._drawModePatch == "hline":
                        self._mouseData[0, 0] = self._xmin
                        self._mouseData[0, 1] = self._y0
                        self._mouseData[1, 0] = self._xmax
                        self._mouseData[1, 1] = self._y0
                    else:
                        self._mouseData[0, 0] = self._x0
                        self._mouseData[0, 1] = self._ymin
                        self._mouseData[1, 0] = self._x0
                        self._mouseData[1, 1] = self._ymax
                    color = self._getDrawingColor()
                    self._drawingPatch = Polygon(self._mouseData,
                                                 closed=True,
                                                 fill=False,
                                                 color=color)
                    self.ax.add_patch(self._drawingPatch)

    def _getDrawingColor(self):
        color = "black"
        if "color" in self._drawingParameters:
            color = self._drawingParameters["color"]
            if len(color) == 4:
                if type(color[3]) in [type(1), numpy.uint8, numpy.int8]:
                    color = numpy.array(color, dtype=numpy.float)/255.
        return color

    def onMouseMoved(self, event):
        logger.debug("onMouseMoved, event = %f %f", event.xdata, event.ydata)
        if event.inaxes != self.ax:
            logger.debug("RETURNING")
            return

        button = event.button
        if button == 1:
            button = "left"
        elif button == 2:
            button = "middle"
        elif button == 3:
            button = "right"
        else:
            button = None
        # as default, export the mouse in graph coordenates
        self._x1 = event.xdata
        self._y1 = event.ydata
        self._x1Pixel = event.x
        self._y1Pixel = event.y
        ddict = {
            'event': 'mouseMoved',
            'x': self._x1,
            'y': self._y1,
            'xpixel': self._x1Pixel,
            'ypixel': self._y1Pixel,
            'button': button,
        }
        self._callback(ddict)

        if button == "middle":
            return

        # should this be made by Plot1D with the previous call???
        # The problem is Plot1D does not know if one is zooming or drawing
        if not (self.__zooming or self.__drawing or self.__picking):
            # this corresponds to moving without click
            marker = None
            for artist in self.ax.lines:
                label = artist.get_label()
                if label.startswith("__MARKER__"):
                    # data = artist.get_xydata()[0:1]
                    x, y = artist.get_xydata()[-1]
                    pixels = self.ax.transData.transform(numpyvstack([x, y]).T)
                    xPixel, yPixel = pixels.T
                    if 'xmarker' in artist._plot_options:
                        if abs(xPixel-event.x) < 5:
                            marker = artist
                    elif 'ymarker' in artist._plot_options:
                        if abs(yPixel-event.y) < 5:
                            marker = artist
                    elif (abs(xPixel-event.x) < 5) and \
                         (abs(yPixel-event.y) < 5):
                        marker = artist
                if marker is not None:
                    break

            oldShape = self.cursor().shape()
            if oldShape not in [QtCore.Qt.SizeHorCursor,
                                QtCore.Qt.SizeVerCursor,
                                QtCore.Qt.PointingHandCursor,
                                QtCore.Qt.OpenHandCursor,
                                QtCore.Qt.SizeAllCursor]:
                self._originalCursorShape = oldShape

            if marker is not None:
                ddict = {}
                ddict['event'] = 'hover'
                ddict['type'] = 'marker'
                ddict['label'] = marker.get_label()[10:]
                if 'draggable' in marker._plot_options:
                    ddict['draggable'] = True
                    if 'ymarker' in artist._plot_options:
                        self.setCursor(
                            QtGui.QCursor(QtCore.Qt.SizeVerCursor))
                    elif 'xmarker' in artist._plot_options:
                        self.setCursor(
                            QtGui.QCursor(QtCore.Qt.SizeHorCursor))
                    else:
                        self.setCursor(
                            QtGui.QCursor(QtCore.Qt.SizeAllCursor))

                else:
                    ddict['draggable'] = False
                if 'selectable' in marker._plot_options:
                    ddict['selectable'] = True
                    self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                else:
                    ddict['selectable'] = False
                ddict['x'] = self._x1
                ddict['y'] = self._y1
                ddict['xpixel'] = self._x1Pixel
                ddict['ypixel'] = self._y1Pixel
                self._callback(ddict)
            else:
                cursors = (QtCore.Qt.SizeHorCursor,
                           QtCore.Qt.SizeVerCursor,
                           QtCore.Qt.PointingHandCursor,
                           QtCore.Qt.OpenHandCursor,
                           QtCore.Qt.SizeAllCursor)
                if self._originalCursorShape in cursors:
                    self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
                else:
                    self.setCursor(QtGui.QCursor(self._originalCursorShape))
            return

        if self.__picking:
            if self.__markerMoving:
                artist = self._pickingInfo['artist']
                infoText = self._pickingInfo['infoText']
                if 'xmarker' in artist._plot_options:
                    artist.set_xdata(event.xdata)
                    ymin, ymax = self.ax.get_ylim()
                    delta = abs(ymax - ymin)
                    ymax = max(ymax, ymin) - 0.005 * delta
                    if infoText is not None:
                        infoText.set_position((event.xdata, ymax))
                elif 'ymarker' in artist._plot_options:
                    artist.set_ydata(event.ydata)
                    if infoText is not None:
                        infoText.set_position((event.xdata, event.ydata))
                else:
                    xData, yData = event.xdata, event.ydata
                    if artist._constraint is not None:
                        # Apply marker constraint
                        xData, yData = artist._constraint(xData, yData)
                    artist.set_xdata(xData)
                    artist.set_ydata(yData)
                    if infoText is not None:
                        xtmp, ytmp = self.ax.transData.transform_point((xData,
                                                                        yData))
                        inv = self.ax.transData.inverted()
                        xtmp, ytmp = inv.transform_point((xtmp, ytmp + 15))
                        infoText.set_position((xData, ytmp))
                if BLITTING and (self._background is not None):
                    canvas = artist.figure.canvas
                    axes = artist.axes
                    artist.set_animated(True)
                    canvas.restore_region(self._background)
                    axes.draw_artist(artist)
                    canvas.blit(axes.bbox)
                else:
                    self.fig.canvas.draw()
                ddict = {}
                ddict['event'] = "markerMoving"
                ddict['button'] = "left"
                ddict['label'] = self._pickingInfo['label']
                ddict['type'] = self._pickingInfo['type']
                ddict['draggable'] = self._pickingInfo['draggable']
                ddict['selectable'] = self._pickingInfo['selectable']
                ddict['x'] = self._x1
                ddict['y'] = self._y1
                ddict['xpixel'] = self._x1Pixel
                ddict['ypixel'] = self._y1Pixel
                ddict['xdata'] = artist.get_xdata()
                ddict['ydata'] = artist.get_ydata()
                self._callback(ddict)
            return
        if not self.__zooming and not self.__drawing:
            return

        if self._x0 is None:
            # this happened when using the middle button
            return

        if (self.__zooming or
                (self.__drawing and self._drawModePatch == 'rectangle')):
            if self._x1 < self._xmin:
                self._x1 = self._xmin
            elif self._x1 > self._xmax:
                self._x1 = self._xmax
            if self._y1 < self._ymin:
                self._y1 = self._ymin
            elif self._y1 > self._ymax:
                self._y1 = self._ymax

            if self._x1 < self._x0:
                x = self._x1
                w = self._x0 - self._x1
            else:
                x = self._x0
                w = self._x1 - self._x0
            if self._y1 < self._y0:
                y = self._y1
                h = self._y0 - self._y1
            else:
                y = self._y0
                h = self._y1 - self._y0
            if w == 0:
                return
            if (not self.__drawing) and (self.ax.get_aspect() != 'auto'):
                if (h / w) > self._ratio:
                    h = w * self._ratio
                else:
                    w = h / self._ratio
                if self._x1 > self._x0:
                    x = self._x0
                else:
                    x = self._x0 - w
                if self._y1 > self._y0:
                    y = self._y0
                else:
                    y = self._y0 - h

            if self.__zooming:
                if self._zoomRectangle is None:
                    self._zoomRectangle = Rectangle(xy=(x, y),
                                                    width=w,
                                                    height=h,
                                                    color=self._zoomColor,
                                                    fill=False)
                    self.ax.add_patch(self._zoomRectangle)
                else:
                    self._zoomRectangle.set_bounds(x, y, w, h)
                if BLITTING:
                    artist = self._zoomRectangle
                    canvas = artist.figure.canvas
                    axes = artist.axes
                    artist.set_animated(True)
                    if self._background is None:
                        self._background = canvas.copy_from_bbox(axes.bbox)
                    canvas.restore_region(self._background)
                    axes.draw_artist(artist)
                    canvas.blit(axes.bbox)
                else:
                    self.fig.canvas.draw()
                return
            else:
                if self._drawingPatch is None:
                    color = self._getDrawingColor()
                    self._drawingPatch = Rectangle(xy=(x, y),
                                                   width=w,
                                                   height=h,
                                                   fill=False,
                                                   color=color)
                    self._drawingPatch.set_hatch('.')
                    self.ax.add_patch(self._drawingPatch)
                else:
                    self._drawingPatch.set_bounds(x, y, w, h)

        if self.__drawing:
            if self._drawingPatch is None:
                self._mouseData = numpy.zeros((2, 2), numpy.float32)
                self._mouseData[0, 0] = self._x0
                self._mouseData[0, 1] = self._y0
                self._mouseData[1, 0] = self._x1
                self._mouseData[1, 1] = self._y1
                color = self._getDrawingColor()
                self._drawingPatch = Polygon(self._mouseData,
                                             closed=True,
                                             fill=False,
                                             color=color)
                self.ax.add_patch(self._drawingPatch)
            elif self._drawModePatch == 'rectangle':
                # already handled, just for compatibility
                self._mouseData = numpy.zeros((2, 2), numpy.float32)
                self._mouseData[0, 0] = self._x0
                self._mouseData[0, 1] = self._y0
                self._mouseData[1, 0] = self._x1
                self._mouseData[1, 1] = self._y1
            elif self._drawModePatch == 'line':
                self._mouseData[0, 0] = self._x0
                self._mouseData[0, 1] = self._y0
                self._mouseData[1, 0] = self._x1
                self._mouseData[1, 1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'hline':
                xmin, xmax = self.ax.get_xlim()
                self._mouseData[0, 0] = xmin
                self._mouseData[0, 1] = self._y1
                self._mouseData[1, 0] = xmax
                self._mouseData[1, 1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'vline':
                ymin, ymax = self.ax.get_ylim()
                self._mouseData[0, 0] = self._x1
                self._mouseData[0, 1] = ymin
                self._mouseData[1, 0] = self._x1
                self._mouseData[1, 1] = ymax
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'polygon':
                self._mouseData[-1, 0] = self._x1
                self._mouseData[-1, 1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
                if matplotlib.__version__.startswith('1.1.1'):
                    # Patch for Debian 7
                    # Workaround matplotlib issue with closed path
                    # Need to toggle closed path to rebuild points
                    self._drawingPatch.set_closed(False)
                self._drawingPatch.set_closed(True)
                self._drawingPatch.set_hatch('/')
            if BLITTING:
                if self._background is None:
                    artist = self._drawingPatch
                    canvas = artist.figure.canvas
                    axes = artist.axes
                    self._background = canvas.copy_from_bbox(axes.bbox)
                artist = self._drawingPatch
                canvas = artist.figure.canvas
                axes = artist.axes
                artist.set_animated(True)
                canvas.restore_region(self._background)
                axes.draw_artist(artist)
                canvas.blit(axes.bbox)
            else:
                self.fig.canvas.draw()
            self._emitDrawingSignal(event='drawingProgress')

    def onMouseReleased(self, event):
        logger.debug("onMouseReleased, event = %f %f",
                     event.xdata, event.ydata)
        if self._infoText in self.ax.texts:
            self._infoText.set_visible(False)
        if self.__picking:
            self.__picking = False
            if self.__markerMoving:
                self.__markerMoving = False
                artist = self._pickingInfo['artist']
                if BLITTING:
                    artist.set_animated(False)
                    self._background = None
                    artist.figure.canvas.draw()
                ddict = {}
                ddict['event'] = "markerMoved"
                ddict['label'] = self._pickingInfo['label']
                ddict['type'] = self._pickingInfo['type']
                ddict['draggable'] = self._pickingInfo['draggable']
                ddict['selectable'] = self._pickingInfo['selectable']
                # use this and not the current mouse position because
                # it has to agree with the marker position
                ddict['x'] = artist.get_xdata()
                ddict['y'] = artist.get_ydata()
                ddict['xdata'] = artist.get_xdata()
                ddict['ydata'] = artist.get_ydata()
                self._callback(ddict)
            return

        if not hasattr(self, "__zoomstack"):
            self.__zoomstack = []

        if event.button == 3:
            # right click
            if self.__drawing:
                self.__drawing = False
                # self._drawingPatch = None
                ddict = {}
                ddict['event'] = 'drawingFinished'
                ddict['type'] = '%s' % self._drawModePatch
                ddict['data'] = self._mouseData * 1
                self._emitDrawingSignal(event='drawingFinished')
                return

            self.__zooming = False
            if len(self._zoomStack):
                xmin, xmax, ymin, ymax, y2min, y2max = self._zoomStack.pop()
                self.setLimits(xmin, xmax, ymin, ymax, y2min, y2max)
                self.draw()

        if self.__drawing and (self._drawingPatch is not None):
            nrows, ncols = self._mouseData.shape
            if self._drawModePatch in ['polygon']:
                self._mouseData = numpy.resize(self._mouseData, (nrows+1, 2))
            self._mouseData[-1, 0] = self._x1
            self._mouseData[-1, 1] = self._y1
            self._drawingPatch.set_xy(self._mouseData)
            if self._drawModePatch not in ['polygon']:
                self._emitDrawingSignal("drawingFinished")

        if self._x0 is None:
            if event.inaxes != self.ax:
                logger.debug("on MouseReleased RETURNING")
            else:
                logger.warning("How can it be here???")
            return
        if self._zoomRectangle is None:
            currentTime = time.time()
            deltaT = currentTime - self.__time0
            if (deltaT < 0.150 or self.__time0 < 0 or not self.__zooming or
                    (self._x1 == self._x0 and self._y1 == self._y0)):
                # single or double click, no zooming
                self.__zooming = False
                ddict = {'x': event.xdata,
                         'y': event.ydata,
                         'xpixel': event.x,
                         'ypixel': event.y}
                middleButton = 2
                rightButton = 3
                button = event.button
                if button == rightButton:
                    ddict['button'] = "right"
                elif button == middleButton:
                    ddict['button'] = "middle"
                else:
                    ddict['button'] = "left"
                if (button == self.__lastMouseClick[0]) and\
                   ((currentTime - self.__lastMouseClick[1]) < 0.6):
                    ddict['event'] = "mouseDoubleClicked"
                else:
                    ddict['event'] = "mouseClicked"
                self.__lastMouseClick = [button, time.time()]
                self._callback(ddict)
                return

        if self._zoomRectangle is not None:
            x, y = self._zoomRectangle.get_xy()
            w = self._zoomRectangle.get_width()
            h = self._zoomRectangle.get_height()
            self._zoomRectangle.remove()
            self._x0 = None
            self._y0 = None
            if BLITTING:
                artist = self._zoomRectangle
                artist.set_animated(False)
                self._background = None
            self._zoomRectangle = None
            if (w != 0) and (h != 0):
                # don't do anything
                xmin, xmax = self.ax.get_xlim()
                ymin, ymax = self.ax.get_ylim()
                if ymax < ymin:
                    ymin, ymax = ymax, ymin

                if not self.ax2.get_yaxis().get_visible():
                    y2min, y2max = None, None
                    newY2Min, newY2Max = None, None
                else:
                    bottom, top = self.ax2.get_ylim()
                    y2min, y2max = min(bottom, top), max(bottom, top)

                    # Convert corners from ax data to window
                    pt0 = self.ax.transData.transform_point((x, y))
                    pt1 = self.ax.transData.transform_point((x + w, y + h))
                    # Convert corners from window to ax2 data
                    pt0 = self.ax2.transData.inverted().transform_point(pt0)
                    pt1 = self.ax2.transData.inverted().transform_point(pt1)

                    # Get min and max on right Y axis
                    newY2Min, newY2Max = pt0[1], pt1[1]
                    if newY2Max < newY2Min:
                        newY2Min, newY2Max = newY2Max, newY2Min

                self._zoomStack.append((xmin, xmax, ymin, ymax, y2min, y2max))
                self.setLimits(x, x+w, y, y+h, newY2Min, newY2Max)
            self.draw()

    @staticmethod
    def _newZoomRange(min_, max_, center, scale, isLog):
        if isLog:
            if min_ > 0.:
                oldMin = numpy.log10(min_)
            else:
                # Happens when autoscale is off and switch to log scale
                # while displaying area < 0.
                oldMin = numpy.log10(numpy.nextafter(0, 1))

            if center > 0.:
                center = numpy.log10(center)
            else:
                center = numpy.log10(numpy.nextafter(0, 1))

            if max_ > 0.:
                oldMax = numpy.log10(max_)
            else:
                # Should not happen
                oldMax = 0.
        else:
            oldMin, oldMax = min_, max_

        offset = (center - oldMin) / (oldMax - oldMin)
        range_ = (oldMax - oldMin) / scale
        newMin = center - offset * range_
        newMax = center + (1. - offset) * range_
        if isLog:
            try:
                newMin, newMax = 10. ** float(newMin), 10. ** float(newMax)
            except OverflowError:  # Limit case
                newMin, newMax = min_, max_
            if newMin <= 0. or newMax <= 0.:  # Limit case
                newMin, newMax = min_, max_
        return newMin, newMax

    def onMouseWheel(self, event):
        if not self.isZoomModeEnabled():
            return

        if event.xdata is None or event.ydata is None:
            return

        scaleF = 1.1 if event.step > 0 else 1 / 1.1

        xLim = self.ax.get_xlim()
        xMin, xMax = min(xLim), max(xLim)
        isXLog = (self.ax.get_xscale() == 'log')

        yLim = self.ax.get_ylim()
        yMin, yMax = min(yLim), max(yLim)
        isYLog = (self.ax.get_yscale() == 'log')

        # If negative limit and log scale,
        # try to get a positive limit from the data limits
        if (isXLog and xMin <= 0.) or (isYLog and yMin <= 0.):
            bounds = self.getDataLimits()
            if isXLog:
                if xMin <= 0. and bounds[0] > 0.:
                    xMin = bounds[0]
                if xMax <= 0. and bounds[1] > 0.:
                    xMax = bounds[1]

            if isYLog:
                if yMin <= 0. and bounds[2] > 0.:
                    yMin = bounds[2]
                if yMax <= 0. and bounds[3] > 0.:
                    yMax = bounds[3]

        xMin, xMax = self._newZoomRange(xMin, xMax,
                                        event.xdata, scaleF, isXLog)

        yMin, yMax = self._newZoomRange(yMin, yMax,
                                        event.ydata, scaleF, isYLog)

        if self.ax2.get_yaxis().get_visible():
            # Get y position in right axis coords
            x, y2Data = self.ax2.transData.inverted().transform_point(
                (event.x, event.y))

            y2Lim = self.ax2.get_ylim()
            y2Min, y2Max = min(y2Lim), max(y2Lim)
            isY2Log = (self.ax2.get_yscale() == 'log')

            # If negative limit and log scale,
            # try to get a positive limit from the data limits
            if isY2Log and y2Min <= 0.:
                bounds = self.getDataLimits('right')
                if yMin <= 0. and bounds[2] > 0.:
                    y2Min = bounds[2]
                if yMax <= 0. and bounds[3] > 0.:
                    y2Max = bounds[3]

            y2Min, y2Max = self._newZoomRange(y2Min, y2Max,
                                              y2Data, scaleF, isYLog)
            self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
        else:
            self.setLimits(xMin, xMax, yMin, yMax)

        self.draw()

    def _emitDrawingSignal(self, event="drawingFinished"):
        ddict = {}
        ddict['event'] = event
        ddict['type'] = '%s' % self._drawModePatch
        a = self._drawingPatch.get_xy()
        ddict['points'] = numpy.array(a)
        ddict['points'].shape = -1, 2
        ddict['xdata'] = ddict['points'][:, 0]
        ddict['ydata'] = ddict['points'][:, 1]
        if self._drawModePatch in ["rectangle", "circle"]:
            # we need the rectangle containing it
            ddict['x'] = ddict['points'][:, 0].min()
            ddict['y'] = ddict['points'][:, 1].min()
            ddict['width'] = self._drawingPatch.get_width()
            ddict['height'] = self._drawingPatch.get_height()
        elif self._drawModePatch in ["ellipse"]:
            # we need the rectangle but given the four corners
            pass
        ddict['parameters'] = {}
        for key in self._drawingParameters.keys():
            ddict['parameters'][key] = self._drawingParameters[key]
        if event == "drawingFinished":
            self.__drawingParameters = None
            self.__drawing = False
            if self._drawingPatch is not None:
                if BLITTING:
                    artist = self._drawingPatch
                    artist.set_animated(False)
                    self._background = None
            self._drawingPatch.remove()
            self._drawingPatch = None
            self.draw()
        self._callback(ddict)

    def emitLimitsChangedSignal(self):
        # Send event about limits changed
        left, right = self.ax.get_xlim()
        xRange = (left, right) if left < right else (right, left)

        bottom, top = self.ax.get_ylim()
        yRange = (bottom, top) if bottom < top else (top, bottom)

        if hasattr(self.ax2, "get_visible") and self.ax2.get_visible():
            bottom2, top2 = self.ax2.get_ylim()
            y2Range = (bottom2, top2) if bottom2 < top2 else (top2, bottom2)
        else:
            y2Range = None

        if hasattr(self, "get_tk_widget"):
            sourceObj = self.get_tk_widget()
        else:
            sourceObj = self

        eventDict = {
            'event': 'limitsChanged',
            'source': id(sourceObj),
            'xdata': xRange,
            'ydata': yRange,
            'y2data': y2Range,
        }
        self._callback(eventDict)

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        self.ax.set_xlim(xmin, xmax)
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)

        if y2min is not None and y2max is not None:
            if y2max < y2min:
                y2min, y2max = y2max, y2min
            if self.ax2.yaxis_inverted():
                bottom, top = y2max, y2min
            else:
                bottom, top = y2min, y2max
            self.ax2.set_ylim(bottom, top)

        self.emitLimitsChangedSignal()

    def resetZoom(self, dataMargins=None):
        xmin, xmax, ymin, ymax = self.getDataLimits('left')
        if hasattr(self.ax2, "get_visible"):
            if self.ax2.get_visible():
                xmin2, xmax2, ymin2, ymax2 = self.getDataLimits('right')
            else:
                xmin2 = None
                xmax2 = None
        else:
            xmin2, xmax2, ymin2, ymax2 = self.getDataLimits('right')

        if (xmin2 is not None) and ((xmin2 != 0) or (xmax2 != 1)):
            xmin = min(xmin, xmin2)
            xmax = max(xmax, xmax2)

        # Add margins around data inside the plot area
        if xmin2 is None:
            newLimits = _utils.addMarginsToLimits(
                dataMargins,
                self.ax.get_xscale() == 'log', self.ax.get_yscale() == 'log',
                xmin, xmax, ymin, ymax)

            self.setLimits(*newLimits)
        else:
            newLimits = _utils.addMarginsToLimits(
                dataMargins,
                self.ax.get_xscale() == 'log', self.ax.get_yscale() == 'log',
                xmin, xmax, ymin, ymax, ymin2, ymax2)

            self.setLimits(*newLimits)

        self._zoomStack = []

    def getDataLimits(self, axesLabel='left'):
        if axesLabel == 'right':
            axes = self.ax2
        else:
            axes = self.ax
        logger.debug("CALCULATING limits %s", axes.get_label())
        xmin = None
        for line2d in axes.lines:
            label = line2d.get_label()
            if label.startswith("__MARKER__"):
                # it is a marker
                continue
            lineXMin = None
            if hasattr(line2d, "_plot_info"):
                if line2d._plot_info["axes"] != axesLabel:
                    continue
                if "xmin" in line2d._plot_info:
                    lineXMin = line2d._plot_info["xmin"]
                    lineXMax = line2d._plot_info["xmax"]
                    lineYMin = line2d._plot_info["ymin"]
                    lineYMax = line2d._plot_info["ymax"]
            if lineXMin is None:
                x = line2d.get_xdata()
                y = line2d.get_ydata()
                if not len(x) or not len(y):
                    continue
                lineXMin = nanmin(x)
                lineXMax = nanmax(x)
                lineYMin = nanmin(y)
                lineYMax = nanmax(y)
            if xmin is None:
                xmin = lineXMin
                xmax = lineXMax
                ymin = lineYMin
                ymax = lineYMax
                continue
            xmin = min(xmin, lineXMin)
            xmax = max(xmax, lineXMax)
            ymin = min(ymin, lineYMin)
            ymax = max(ymax, lineYMax)

        for line2d in axes.collections:
            label = line2d.get_label()
            if label.startswith("__MARKER__"):
                # it is a marker
                continue
            lineXMin = None
            if hasattr(line2d, "_plot_info"):
                if line2d._plot_info["axes"] != axesLabel:
                    continue
                if "xmin" in line2d._plot_info:
                    lineXMin = line2d._plot_info["xmin"]
                    lineXMax = line2d._plot_info["xmax"]
                    lineYMin = line2d._plot_info["ymin"]
                    lineYMax = line2d._plot_info["ymax"]
            if lineXMin is None:
                logger.warning("CANNOT CALCULATE LIMITS")
                continue
            if xmin is None:
                xmin = lineXMin
                xmax = lineXMax
                ymin = lineYMin
                ymax = lineYMax
                continue
            xmin = min(xmin, lineXMin)
            xmax = max(xmax, lineXMax)
            ymin = min(ymin, lineYMin)
            ymax = max(ymax, lineYMax)

        for artist in axes.images:
            x0, x1, y0, y1 = artist.get_extent()
            if (xmin is None):
                xmin = x0
                xmax = x1
                ymin = min(y0, y1)
                ymax = max(y0, y1)
            xmin = min(xmin, x0)
            xmax = max(xmax, x1)
            ymin = min(ymin, y0)
            ymax = max(ymax, y1)

        for artist in axes.artists:
            label = artist.get_label()
            if label.startswith("__IMAGE__"):
                if hasattr(artist, 'get_image_extent'):
                    x0, x1, y0, y1 = artist.get_image_extent()
                else:
                    x0, x1, y0, y1 = artist.get_extent()
                if (xmin is None):
                    xmin = x0
                    xmax = x1
                    ymin = min(y0, y1)
                    ymax = max(y0, y1)
                ymin = min(ymin, y0, y1)
                ymax = max(ymax, y1, y0)
                xmin = min(xmin, x0)
                xmax = max(xmax, x1)

        if xmin is None:
            xmin = 0
            xmax = 1
            ymin = 0
            ymax = 1
            if axesLabel == 'right':
                return None, None, None, None

        xSize = float(xmax - xmin)
        ySize = float(ymax - ymin)
        A = self.ax.get_aspect()
        if A != 'auto':
            figW, figH = self.ax.get_figure().get_size_inches()
            figAspect = figH / figW

            dataRatio = (ySize / xSize) * A

            y_expander = dataRatio - figAspect
            # If y_expander > 0, the dy/dx viewLim ratio needs to increase
            if abs(y_expander) < 0.005:
                # good enough
                pass
            else:
                # this works for any data ratio
                if y_expander < 0:
                    deltaY = xSize * (figAspect / A) - ySize
                    yc = 0.5 * (ymin + ymax)
                    ymin = yc - (ySize + deltaY) * 0.5
                    ymax = yc + (ySize + deltaY) * 0.5
                else:
                    deltaX = ySize * (A / figAspect) - xSize
                    xc = 0.5 * (xmin + xmax)
                    xmin = xc - (xSize + deltaX) * 0.5
                    xmax = xc + (xSize + deltaX) * 0.5
        logger.debug("CALCULATED LIMITS = %f %f %f %f", xmin, xmax, ymin, ymax)
        return xmin, xmax, ymin, ymax

    def resizeEvent(self, ev):
        # we have to get rid of the copy of the underlying image
        self._background = None
        FigureCanvas.resizeEvent(self, ev)

    def draw(self):
        logger.debug("Draw called")
        super(MatplotlibGraph, self).draw()


class MatplotlibBackend(object):
    """Matplotlib backend.

    See :class:`Backend.Backend` for the documentation of the public API.
    """

    def __init__(self, plot, parent=None):
        self._setPlot(plot)

        self.graph = MatplotlibGraph(parent)
        self.ax2 = self.graph.ax2
        self.ax = self.graph.ax

        self._parent = parent
        self._colormaps = {}

        self._graphCursor = None
        self._graphCursorConfiguration = None
        self.matplotlibVersion = matplotlib.__version__

        self.setGraphXLimits(0., 100.)
        self.setGraphYLimits(0., 100.)

        self._enableAxis('right', False)

    # TODO inherit from BackendBase.py
    @property
    def _plot(self):
        if self._plotRef is None:
            raise RuntimeError('This backend is not attached to a Plot')

        plot = self._plotRef()
        if plot is None:
            raise RuntimeError('This backend is no more attached to a Plot')
        return plot

    def _setPlot(self, plot):
        self._plotRef = weakref.ref(plot)

    # Add methods

    def addCurve(self, x, y, legend,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill):
        assert None not in (x, y, legend, color, symbol, linewidth, linestyle,
                            yaxis, z, selectable, fill)
        assert yaxis in ('left', 'right')

        if (len(color) == 4 and
                type(color[3]) in [type(1), numpy.uint8, numpy.int8]):
            color = numpy.array(color, dtype=numpy.float)/255.

        if yaxis == "right":
            axes = self.ax2
            self._enableAxis("right", True)
        else:
            axes = self.ax

        picker = 3 if selectable else None

        if hasattr(color, 'dtype') and len(color) == len(x):
            # scatter plot
            if color.dtype not in [numpy.float32, numpy.float]:
                actualColor = color / 255.
            else:
                actualColor = color

            pathObject = axes.scatter(x, y,
                                      label=legend,
                                      color=actualColor,
                                      marker=symbol,
                                      picker=picker)

            if linestyle not in [" ", None]:
                # scatter plot with an actual line ...
                # we need to assign a color ...
                curveList = axes.plot(x, y, label=legend,
                                      linestyle=linestyle,
                                      color=actualColor[0],
                                      linewidth=linewidth,
                                      picker=picker,
                                      marker=None)
                # TODO what happen to curveList????

            # scatter plot is a collection
            curveList = [pathObject]

        else:
            curveList = axes.plot(x, y,
                                  label=legend,
                                  linestyle=linestyle,
                                  color=color,
                                  linewidth=linewidth,
                                  picker=picker)

        # errorbar is a container?
        # axes.errorbar(x,y, label=legend,yerr=numpy.sqrt(y),
        #               linestyle=" ",color='b')

        if fill:
            axes.fill_between(x, 1.0e-8, y)

        if hasattr(curveList[-1], "set_marker"):
            curveList[-1].set_marker(symbol)

        curveList[-1]._plot_info = {
            'axes': yaxis,
            # this is needed for scatter plots because I do not know
            # how to recover the data yet, it can speed up limits too
            'xmin': nanmin(x),
            'xmax': nanmax(x),
            'ymin': nanmin(y),
            'ymax': nanmax(y),
        }

        curveList[-1].axes = axes
        curveList[-1].set_zorder(z)

        return curveList[-1]

    def addImage(self, data, legend,
                 xScale, yScale, z,
                 selectable, draggable,
                 colormap):
        # Non-uniform image
        # http://wiki.scipy.org/Cookbook/Histograms
        # Non-linear axes
        # http://stackoverflow.com/questions/11488800/non-linear-axes-for-imshow-in-matplotlib
        assert None not in (data, legend, xScale, yScale, z,
                            selectable, draggable)

        h, w = data.shape[0:2]
        xmin = xScale[0]
        xmax = xmin + xScale[1] * w
        ymin = yScale[0]
        ymax = ymin + yScale[1] * h
        extent = (xmin, xmax, ymax, ymin)

        picker = (selectable or draggable)

        # the normalization can be a source of time waste
        # Two possibilities, we receive data or a ready to show image
        if len(data.shape) == 3:
            if data.shape[-1] == 4:
                # force alpha? data[:,:,3] = 255
                pass

            # RGBA image
            # TODO: Possibility to mirror the image
            # in case of pixmaps just setting
            # extend = (xmin, xmax, ymax, ymin)
            # instead of (xmin, xmax, ymin, ymax)
            extent = (xmin, xmax, ymin, ymax)
            if tuple(xScale) != (0., 1.) or tuple(yScale) != (0., 1.):
                # for the time being not properly handled
                imageClass = AxesImage
            elif (data.shape[0] * data.shape[1]) > 5.0e5:
                imageClass = ModestImage
            else:
                imageClass = AxesImage
            image = imageClass(self.ax,
                               label="__IMAGE__"+legend,
                               interpolation='nearest',
                               picker=picker,
                               zorder=z)
            if image.origin == 'upper':
                image.set_extent((xmin, xmax, ymax, ymin))
            else:
                image.set_extent((xmin, xmax, ymin, ymax))
            image.set_data(data)

        else:
            assert colormap is not None
            cmap = self.__getColormap(colormap['name'])
            if colormap['normalization'].startswith('log'):
                vmin, vmax = None, None
                if not colormap['autoscale']:
                    if colormap['vmin'] > 0.:
                        vmin = colormap['vmin']
                    if colormap['vmax'] > 0.:
                        vmax = colormap['vmax']

                    if vmin is None or vmax is None:
                        logger.warning('Log colormap with negative bounds, ' +
                                       'changing bounds to positive ones.')
                    elif vmin > vmax:
                        logger.warning('Colormap bounds are inverted.')
                        vmin, vmax = vmax, vmin

                # Set unset/negative bounds to positive bounds
                if vmin is None or vmax is None:
                    posData = data[data > 0]
                    if vmax is None:
                        # 1. as an ultimate fallback
                        vmax = posData.max() if posData.size > 0 else 1.
                    if vmin is None:
                        vmin = posData.min() if posData.size > 0 else vmax
                    if vmin > vmax:
                        vmin = vmax

                norm = LogNorm(vmin, vmax)

            else:  # Linear normalization
                if colormap['autoscale']:
                    vmin = data.min()
                    vmax = data.max()
                else:
                    vmin = colormap['vmin']
                    vmax = colormap['vmax']
                    if vmin > vmax:
                        logger.warning('Colormap bounds are inverted.')
                        vmin, vmax = vmax, vmin

                norm = Normalize(vmin, vmax)

            # try as data
            if tuple(xScale) != (0., 1.) or tuple(yScale) != (0., 1.):
                # for the time being not properly handled
                imageClass = AxesImage
            elif (data.shape[0] * data.shape[1]) > 5.0e5:
                imageClass = ModestImage
            else:
                imageClass = AxesImage
            image = imageClass(self.ax,
                               label="__IMAGE__" + legend,
                               interpolation='nearest',
                               cmap=cmap,
                               extent=extent,
                               picker=picker,
                               zorder=z,
                               norm=norm)

            if image.origin == 'upper':
                image.set_extent((xmin, xmax, ymax, ymin))
            else:
                image.set_extent((xmin, xmax, ymin, ymax))

            image.set_data(data)

        self.ax.add_artist(image)

        image._plot_info = {'xScale': xScale, 'yScale': yScale}
        image._plot_options = []
        if draggable:
            image._plot_options.append('draggable')
        if selectable:
            image._plot_options.append('selectable')

        return image

    def addItem(self, x, y, legend, shape, color, fill):
        xView = numpy.array(x, copy=False)
        yView = numpy.array(y, copy=False)

        if shape == "hline":  # TODO this code was not active before
            if hasattr(y, "__len__"):
                y = y[-1]
            line = self.ax.axhline(y, label=legend, color=color)
            return line

        elif shape == "vline":  # TODO this code was not active before
            if hasattr(x, "__len__"):
                x = x[-1]
            line = self.ax.axvline(x, label=legend, color=color)
            return line

        elif shape == 'rectangle':
            xMin = nanmin(xView)
            xMax = nanmax(xView)
            yMin = nanmin(yView)
            yMax = nanmax(yView)
            w = xMax - xMin
            h = yMax - yMin
            item = Rectangle(xy=(xMin, yMin),
                             width=w,
                             height=h,
                             fill=False,
                             color=color)
            if fill:
                item.set_hatch('.')

            self.ax.add_patch(item)
            return item

        elif shape == 'polygon':
            xView.shape = 1, -1
            yView.shape = 1, -1
            item = Polygon(numpyvstack((xView, yView)).T,
                           closed=True,
                           fill=False,
                           label=legend,
                           color=color)
            if fill:
                item.set_hatch('/')

            self.ax.add_patch(item)
            return item

        else:
            raise NotImplementedError("Unsupported item shape %s" % shape)

    def addMarker(self, x, y, legend, text, color,
                  selectable, draggable,
                  symbol, constraint):
        legend = "__MARKER__" + legend  # TODO useful?

        # Apply constraint to provided position
        if draggable and constraint is not None:
            x, y = constraint(x, y)

        line = self.ax.plot(x, y, label=legend,
                            linestyle=" ",
                            color=color,
                            marker=symbol,
                            markersize=10.)[-1]

        if selectable or draggable:
            line.set_picker(5)

        if text is not None:
            xtmp, ytmp = self.ax.transData.transform((x, y))
            inv = self.ax.transData.inverted()
            xtmp, ytmp = inv.transform((xtmp, ytmp + 15))
            text = " " + text
            line._infoText = self.ax.text(x, ytmp, text,
                                          color=color,
                                          horizontalalignment='left',
                                          verticalalignment='top')

        line._constraint = constraint if draggable else None

        line._plot_options = ["marker"]
        if selectable:
            line._plot_options.append('selectable')
        if draggable:
            line._plot_options.append('draggable')

        return line

    def addXMarker(self, x, legend, text,
                   color, selectable, draggable):
        legend = "__MARKER__" + legend  # TODO useful?

        line = self.ax.axvline(x, label=legend, color=color)
        if selectable or draggable:
            line.set_picker(5)

        if text is not None:
            text = " " + text
            ymin, ymax = self.getGraphYLimits()
            delta = abs(ymax - ymin)
            if ymin > ymax:
                ymax = ymin
            ymax -= 0.005 * delta
            line._infoText = self.ax.text(x, ymax, text,
                                          color=color,
                                          horizontalalignment='left',
                                          verticalalignment='top')

        line._plot_options = ["xmarker"]
        if selectable:
            line._plot_options.append('selectable')
        if draggable:
            line._plot_options.append('draggable')

        return line

    def addYMarker(self, y, legend=None, text=None,
                   color='k', selectable=False, draggable=False):
        legend = "__MARKER__" + legend  # TODO useful?

        line = self.ax.axhline(y, label=legend, color=color)
        if selectable or draggable:
            line.set_picker(5)

        if text is not None:
            text = " " + text
            xmin, xmax = self.getGraphXLimits()
            delta = abs(xmax - xmin)
            if xmin > xmax:
                xmax = xmin
            xmax -= 0.005 * delta
            line._infoText = self.ax.text(y, xmax, text,
                                          color=color,
                                          horizontalalignment='left',
                                          verticalalignment='top')

        line._plot_options = ["ymarker"]
        if selectable:
            line._plot_options.append('selectable')
        if draggable:
            line._plot_options.append('draggable')

        return line

    # Remove methods

    def clear(self):
        self.ax.clear()
        self.ax2.clear()

    def remove(self, item):
        # Warning: It also needs to remove extra stuff if added as for markers
        if hasattr(item, "_infoText"):  # For markers text
            item._infoText.remove()
            item._infoText = None
        item.remove()

    # Interaction methods

    def isZoomModeEnabled(self, *args, **kwargs):
        self.graph.isZoomModeEnabled(*args, **kwargs)

    def setZoomModeEnabled(self, *args, **kwargs):
        self.graph.setZoomModeEnabled(*args, **kwargs)

    def isDrawModeEnabled(self, *args, **kwargs):
        self.graph.isDrawModeEnabled(*args, **kwargs)

    def setDrawModeEnabled(self, *args, **kwargs):
        self.graph.setDrawModeEnabled(*args, **kwargs)

    def getDrawMode(self, *args, **kwargs):
        self.graph.getDrawMode(*args, **kwargs)

    def getGraphCursor(self):
        if self._graphCursor is None or not self._graphCursor.visible:
            return None
        else:
            return self._graphCursorConfiguration

    def setGraphCursor(self, flag=True, color='black',
                       linewidth=1, linestyle='-'):
        self._graphCursorConfiguration = color, linewidth, linestyle

        if flag:
            if self._graphCursor is None:
                self._graphCursor = Cursor(self.ax,
                                           useblit=False,
                                           color=color,
                                           linewidth=linewidth,
                                           linestyle=linestyle)
            self._graphCursor.visible = True
        else:
            if self._graphCursor is not None:
                self._graphCursor.visible = False

    # Active curve

    def setCurveColor(self, curve, color):
        curve.set_color(color)

    # Misc.

    def getWidgetHandle(self):
        return self.graph

    def _enableAxis(self, axis, flag=True):
        """Show/hide Y axis

        :param str axis: Axis name: 'left' or 'right'
        :param bool flag: Default, True
        """
        assert axis in ('right', 'left')
        axes = self.ax2 if axis == 'right' else self.ax
        axes.get_yaxis().set_visible(flag)

    def replot(self):
        # TODO images, markers? scatter plot? move in remove?
        # Right Y axis only support curve for now
        # Hide right Y axis if no line is present
        if not self.ax2.lines:
            self._enableAxis('right', False)
        self.graph.draw()

    def saveGraph(self, fileName, fileFormat, dpi=None):
        # fileName can be also a StringIO or file instance
        if dpi is not None:
            self.ax.figure.savefig(fileName, format=fileFormat, dpi=dpi)
        else:
            self.ax.figure.savefig(fileName, format=fileFormat)

    def setCallback(self, callbackFunction):
        self.graph.setCallback(callbackFunction)
        # Should I call the base to keep a copy?
        # It does not seem necessary since the graph will do it.

    # Graph labels

    def setGraphTitle(self, title):
        self.ax.set_title(title)

    def setGraphXLabel(self, label):
        self.ax.set_xlabel(label)

    def setGraphYLabel(self, label):
        self.ax.set_ylabel(label)

    # Graph limits

    def resetZoom(self, dataMargins=None):
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits()
        xAuto = self._plot.isXAxisAutoScale()
        yAuto = self._plot.isYAxisAutoScale()

        if xAuto and yAuto:
            self.graph.resetZoom(dataMargins)
        elif yAuto:
            self.graph.resetZoom(dataMargins)
            self.setGraphXLimits(xmin, xmax)
        elif xAuto:
            self.graph.resetZoom(dataMargins)
            self.setGraphYLimits(ymin, ymax)
        else:
            logger.debug("Nothing to autoscale")

        self._zoomStack = []

    def setLimits(self, xmin, xmax, ymin, ymax):
        self.ax.set_xlim(xmin, xmax)
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)
        self.graph.emitLimitsChangedSignal()

    def getGraphXLimits(self):
        vmin, vmax = self.ax.get_xlim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def setGraphXLimits(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)
        self.graph.emitLimitsChangedSignal()

    def getGraphYLimits(self, axis="left"):
        assert axis in ('left', 'right')
        ax = self.ax2 if axis == 'right' else self.ax

        vmin, vmax = ax.get_ylim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def setGraphYLimits(self, ymin, ymax):
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)
        self.graph.emitLimitsChangedSignal()

    # Graph axes

    def setXAxisLogarithmic(self, flag):
        self.ax2.set_xscale('log' if flag else 'linear')
        self.ax.set_xscale('log' if flag else 'linear')

    def setYAxisLogarithmic(self, flag):
        self.ax2.set_yscale('log' if flag else 'linear')
        self.ax.set_yscale('log' if flag else 'linear')

    def invertYAxis(self, flag):
        if self.ax.yaxis_inverted() != bool(flag):
            self.ax.invert_yaxis()

    def isYAxisInverted(self):
        return self.ax.yaxis_inverted()

    def isKeepDataAspectRatio(self):
        return self.ax.get_aspect() in (1.0, 'equal')

    def keepDataAspectRatio(self, flag=True):
        self.ax.set_aspect(1.0 if flag else 'auto')
        self.resetZoom()

    def showGrid(self, flag=True):
        self.ax.grid(False, which='both')  # Disable all grid first
        if flag:
            self.ax.grid(True, which='both' if flag == 2 else 'major')

    # colormap

    def getSupportedColormaps(self):
        default = [
            'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue']
        maps = [m for m in cm.datad]
        maps.sort()
        return default + maps

    def __getColormap(self, name):
        if not self._colormaps:  # Lazy initialization of own colormaps
            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}
            self._colormaps['red'] = LinearSegmentedColormap(
                'red', cdict, 256)

            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 1.0, 1.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}
            self._colormaps['green'] = LinearSegmentedColormap(
                'green', cdict, 256)

            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0))}
            self._colormaps['blue'] = LinearSegmentedColormap(
                'blue', cdict, 256)

            # Temperature as defined in spslut
            cdict = {'red': ((0.0, 0.0, 0.0),
                             (0.5, 0.0, 0.0),
                             (0.75, 1.0, 1.0),
                             (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (0.25, 1.0, 1.0),
                               (0.75, 1.0, 1.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 1.0, 1.0),
                              (0.25, 1.0, 1.0),
                              (0.5, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}
            # but limited to 256 colors for a faster display (of the colorbar)
            self._colormaps['temperature'] = LinearSegmentedColormap(
                'temperature', cdict, 256)

            # reversed gray
            cdict = {'red':     ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0)),
                     'green':   ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0)),
                     'blue':    ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0))}

            self._colormaps['reversed gray'] = LinearSegmentedColormap(
                'yerg', cdict, 256)

        if name in self._colormaps:
            return self._colormaps[name]
        else:
            # matplotlib built-in
            return cm.get_cmap(name)

    # Data <-> Pixel coordinates conversion

    def dataToPixel(self, x=None, y=None, axis="left"):
        assert axis in ("left", "right")
        ax = self.ax2 if "axis" == "right" else self.ax

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if x is None:
            x = 0.5 * (xmax - xmin)
        if y is None:
            y = 0.5 * (ymax - ymin)

        if x > xmax or x < xmin:
            return None

        if y > ymax or y < ymin:
            return None

        pixels = ax.transData.transform([x, y])
        xPixel, yPixel = pixels.T
        return xPixel, yPixel

    def pixelToData(self, x=None, y=None, axis="left"):
        assert axis in ("left", "right")
        ax = self.ax2 if "axis" == "right" else self.ax

        inv = ax.transData.inverted()
        x, y = inv.transform((x, y))

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if x > xmax or x < xmin:
            return None

        if y > ymax or y < ymin:
            return None

        return x, y
