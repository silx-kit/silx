#/*##########################################################################
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Matplotlib Plot backend.
"""
from matplotlib import cbook
import matplotlib
# blitting enabled by default
# it provides faster response at the cost of missing minor updates
# during movement (only the bounding box of the moving object is updated)
# For instance, when moving a marker, the label is not updated during the
# movement.
BLITTING = True
import numpy
from numpy import vstack as numpyvstack
# Problem on debian6 numpy version 1.4.1 unsigned longs give infinity
#from numpy import nanmax, nanmin
def nanmax(x):
    try:
        return x[numpy.isfinite(x)].max()
    except:
        return numpy.nanmax(x)

def nanmin(x):
    try:
        return x[numpy.isfinite(x)].min()
    except:
        return numpy.nanmin(x)

import sys
import types
try:
    from .. import PlotBackend
except ImportError:
    from PyMca5.PyMca import PlotBackend
from matplotlib import cm
from matplotlib.font_manager import FontProperties
try:
    from matplotlib.widgets import Cursor
except:
    print("matplotlib.widgets Cursor not available")
# This should be independent of Qt
TK = False
if ("tk" in sys.argv) or ("Tkinter" in sys.modules) or ("tkinter" in sys.modules):
    TK = True
if TK and ("PyQt4" not in sys.modules) and ("PyQt5" not in sys.modules) and\
    ("PySide" not in sys.modules):
    if sys.version < '3.0':
        import Tkinter as Tk
    else:
        import tkinter as Tk
elif ('PySide' in sys.modules) or ('PySide' in sys.argv) :
    matplotlib.rcParams['backend']='Qt4Agg'
    matplotlib.rcParams['backend.qt4']='PySide'
    from PySide import QtCore, QtGui
elif ("PyQt4" in sys.modules) or ('PyQt4' in sys.argv):
    from PyQt4 import QtCore, QtGui
    matplotlib.rcParams['backend']='Qt4Agg'
elif ('PyQt5' in sys.modules):
    matplotlib.rcParams['backend']='Qt5Agg'
    from PyQt5 import QtCore, QtGui, QtWidgets
    QtGui.QApplication = QtWidgets.QApplication
else:
    try:
        from PyQt4 import QtCore, QtGui
        matplotlib.rcParams['backend']='Qt4Agg'
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
            QtGui.QApplication = QtWidgets.QApplication
            matplotlib.rcParams['backend']='Qt5Agg'
        except ImportError:
            from PySide import QtCore, QtGui
if ("PyQt4" in sys.modules) or ("PySide" in sys.modules):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    TK = False
    QT = True
elif "PyQt5" in sys.modules:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    TK = False
    QT = True
elif ("Tkinter" in sys.modules) or ("tkinter") in sys.modules:
    TK = True
    QT = False
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas

from matplotlib.figure import Figure
import matplotlib.patches as patches
Rectangle = patches.Rectangle
Polygon = patches.Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from matplotlib.image import AxesImage, NonUniformImage
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import time

try:
    from . import _utils
except ImportError:
    from PyMca5.PyMcaGraph.backends import _utils

DEBUG = 0

class ModestImage(AxesImage):
    """

Customization of https://github.com/ChrisBeaumont/ModestImage to allow
extent support.

Computationally modest image class.

ModestImage is an extension of the Matplotlib AxesImage class
better suited for the interactive display of larger images. Before
drawing, ModestImage resamples the data array based on the screen
resolution and view window. This has very little affect on the
appearance of the image, but can substantially cut down on
computation since calculations of unresolved or clipped pixels
are skipped.

The interface of ModestImage is the same as AxesImage. However, it
does not currently support setting the 'extent' property. There
may also be weird coordinate warping operations for images that
I'm not aware of. Don't expect those to work either.
"""
    def __init__(self, *args, **kwargs):
        self._full_res = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        self._origExtent = None
        super(ModestImage, self).__init__(*args, **kwargs)
        if 'extent' in kwargs and kwargs['extent'] is not None:
            self.set_extent(kwargs['extent'])

    def set_extent(self, extent):
        super(ModestImage, self).set_extent(extent)
        if self._origExtent is None:
            self._origExtent = self.get_extent()

    def get_image_extent(self):
        """Returns the extent of the whole image.

        get_extent returns the extent of the drawn area and not of the full
        image.

        :return: Bounds of the image (x0, x1, y0, y1).
        :rtype: Tuple of 4 floats.
        """
        if self._origExtent is not None:
            return self._origExtent
        else:
            return self.get_extent()

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """

        self._full_res = A
        self._A = A

        if self._A.dtype != numpy.uint8 and not numpy.can_cast(self._A.dtype,
                                                         numpy.float):
            raise TypeError("Image data can not convert to float")

        if (self._A.ndim not in (2, 3) or
            (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
            raise TypeError("Invalid dimensions for image data")

        self._imcache =None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None

    def get_array(self):
        """Override to return the full-resolution array"""
        return self._full_res

    def _scale_to_res(self):
        """ Change self._A and _extent to render an image whose
resolution is matched to the eventual rendering."""
        #extent has to be set BEFORE set_data
        if self._origExtent is None:
            if self.origin == "upper":
                self._origExtent = 0, self._full_res.shape[1], \
                                    self._full_res.shape[0], 0
            else:
                self._origExtent = 0, self._full_res.shape[1], \
                                    0, self._full_res.shape[0]

        if self.origin == "upper":
            origXMin, origXMax, origYMax, origYMin =\
                           self._origExtent[0:4]
        else:
            origXMin, origXMax, origYMin, origYMax =\
                           self._origExtent[0:4]
        ax = self.axes
        ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
        #print("PIXELS H = ", ext[0], "PIXELS V = ", ext[1])
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        #print("BEFORE AXES LIMITS X", xlim)
        #print("BEFORE AXES LIMITS Y", ylim)
        xlim = max(xlim[0], origXMin), min(xlim[1], origXMax)
        if ylim[0] > ylim[1]:
            ylim = max(ylim[1], origYMin), min(ylim[0], origYMax)
        else:
            ylim = max(ylim[0], origYMin), min(ylim[1], origYMax)
        #print("AXES LIMITS X", xlim)
        #print("AXES LIMITS Y", ylim)
        #print("THOSE LIMITS ARE TO BE COMPARED WITH THE EXTENT")
        #print("IN ORDER TO KNOW WHAT IT IS LIMITING THE DISPLAY")
        #print("IF THE AXES OR THE EXTENT")
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        y0 = max(0, ylim[0] - 5)
        y1 = min(self._full_res.shape[0], ylim[1] + 5)
        x0 = max(0, xlim[0] - 5)
        x1 = min(self._full_res.shape[1], xlim[1] + 5)
        y0, y1, x0, x1 = [int(a) for a in [y0, y1, x0, x1]]

        sy = int(max(1, min((y1 - y0) / 5., numpy.ceil(dy / ext[1]))))
        sx = int(max(1, min((x1 - x0) / 5., numpy.ceil(dx / ext[0]))))

        # have we already calculated what we need?
        if (self._sx is not None) and (self._sy is not None):
            if sx >= self._sx and sy >= self._sy and \
                x0 >= self._bounds[0] and x1 <= self._bounds[1] and \
                y0 >= self._bounds[2] and y1 <= self._bounds[3]:
                return

        self._A = self._full_res[y0:y1:sy, x0:x1:sx]
        self._A = cbook.safe_masked_invalid(self._A)
        x1 = x0 + self._A.shape[1] * sx
        y1 = y0 + self._A.shape[0] * sy

        if self.origin == "upper":
            self.set_extent([x0, x1, y1, y0])
        else:
            self.set_extent([x0, x1, y0, y1])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        self._scale_to_res()
        super(ModestImage, self).draw(renderer, *args, **kwargs)

class MatplotlibGraph(FigureCanvas):
    def __init__(self, parent=None, **kw):
        #self.figure = Figure(figsize=size, dpi=dpi) #in inches
        self.fig = Figure()
        if TK:
            self._canvas = FigureCanvas.__init__(self, self.fig, master=parent)
        else:
            self._originalCursorShape = QtCore.Qt.ArrowCursor
            self._canvas = FigureCanvas.__init__(self, self.fig)
            # get the default widget color
            color = self.palette().color(self.backgroundRole())
            color = "#%x" % color.rgb()
            if len(color) == 9:
                color = "#" + color[3:]
            #self.fig.set_facecolor(color)
            self.fig.set_facecolor("w")
            # that's it
        if 1:
            #this almost works
            """
        def twinx(self):
            call signature::

              ax = twinx()

            create a twin of Axes for generating a plot with a sharex
            x-axis but independent y axis.  The y-axis of self will have
            ticks on left and the returned axes will have ticks on the
            right
            ax2 = self.fig.add_axes(self.get_position(True), sharex=self,
                frameon=False)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.set_offset_position('right')
            self.ax.yaxis.tick_left()
            ax2.xaxis.set_visible(False)
            return ax2
            """
            self.ax = self.fig.add_axes([.15, .15, .75, .75], label="left")
            self.ax2 = self.ax.twinx()
            self.ax2.set_label("right")

            # critical for picking!!!!
            self.ax2.set_zorder(0)
            self.ax2.set_autoscaley_on(True)
            self.ax.set_zorder(1)
            #this works but the figure color is left
            self.ax.set_axis_bgcolor('none')
            self.fig.sca(self.ax)
        else:
            #this almost works
            self.ax2 = self.fig.add_axes([.15, .15, .75, .75],
                                         axisbg="w",
                                         label="right",
                                         frameon=False)
            self.ax = self.fig.add_axes(self.ax2.get_position(),
                                        sharex=self.ax2,
                                        label="left",
                                        frameon=True)
            self.ax2.yaxis.tick_right()
            self.ax2.xaxis.set_visible(False)
            self.ax2.yaxis.set_label_position('right')
            self.ax2.yaxis.set_offset_position('right')
            self.ax.set_axis_bgcolor('none')

        # this respects aspect size
        # self.ax = self.fig.add_subplot(111, aspect='equal')
        # This should be independent of Qt
        if ("PyQt4" in sys.modules) or ("PySide" in sys.modules):
            FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        self.__lastMouseClick = ["middle", time.time()]
        self._zoomEnabled = False
        self._zoomColor = "black"
        self.__zooming = False
        self.__picking = False
        self._background = None
        self.__markerMoving = False
        self._zoomStack = []
        self.xAutoScale = True
        self.yAutoScale = True

        #info text
        self._infoText = None

        #drawingmode handling
        self.setDrawModeEnabled(False)
        self.__drawModeList = ['line', 'hline', 'vline', 'rectangle', 'polygon']
        self.__drawing = False
        self._drawingPatch = None
        self._drawModePatch = 'line'

        #event handling
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
        if DEBUG:
            print(ddict)

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
            #xdata = thisline.get_xdata()
            #ydata = thisline.get_ydata()
            #print('onPick line:', zip(numpy.take(xdata, ind),
            #                           numpy.take(ydata, ind)))
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
            if DEBUG:
                print("%s %s selected" % (self._pickingInfo['type'].upper(),
                                          self._pickingInfo['label']))
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onPick patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onPick text:', text.get_text())
        elif isinstance(event.artist, AxesImage):
            self.__picking = True
            artist = event.artist
            #print dir(artist)
            self._pickingInfo['artist'] = artist
            #self._pickingInfo['event_ind'] = ind
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
            print("unhandled event", event.artist)

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
        if DEBUG:
            print("onMousePressed, event = ",event.xdata, event.ydata)
            print("Mouse button = ", event.button)
        self.__time0 = -1.0
        if event.inaxes != self.ax:
            if DEBUG:
                print("RETURNING")
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
            if DEBUG:
                print("PICKING, Ignoring zoom")
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
                #ddict['event'] = "legendClicked"
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
                #ddict['event'] = "legendClicked"
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
            #right click
            self.__zooming = False
            if self._drawingPatch is not None:
                self._emitDrawingSignal("drawingFinished")
            return

        self.__time0 = time.time()
        self.__zooming = self._zoomEnabled
        self._zoomRect = None
        self._xmin, self._xmax  = self.ax.get_xlim()
        self._ymin, self._ymax  = self.ax.get_ylim()
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
                    self._mouseData = numpy.zeros((2,2), numpy.float32)
                    if self._drawModePatch == "hline":
                        self._mouseData[0,0] = self._xmin
                        self._mouseData[0,1] = self._y0
                        self._mouseData[1,0] = self._xmax
                        self._mouseData[1,1] = self._y0
                    else:
                        self._mouseData[0,0] = self._x0
                        self._mouseData[0,1] = self._ymin
                        self._mouseData[1,0] = self._x0
                        self._mouseData[1,1] = self._ymax
                    color=self._getDrawingColor()
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
        if DEBUG:
            print("onMouseMoved, event = ",event.xdata, event.ydata)
        if event.inaxes != self.ax:
            if DEBUG:
                print("RETURNING")
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
        #as default, export the mouse in graph coordenates
        self._x1 = event.xdata
        self._y1 = event.ydata
        self._x1Pixel = event.x
        self._y1Pixel = event.y
        ddict= {'event':'mouseMoved',
              'x':self._x1,
              'y':self._y1,
              'xpixel':self._x1Pixel,
              'ypixel':self._y1Pixel,
              'button':button,
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
                    #data = artist.get_xydata()[0:1]
                    x, y = artist.get_xydata()[-1]
                    pixels = self.ax.transData.transform(numpyvstack([x,y]).T)
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
            if QT:
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
                    if QT:
                        if 'ymarker' in artist._plot_options:
                            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeVerCursor))
                        elif 'xmarker' in artist._plot_options:
                            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
                        else:
                            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))
                else:
                    ddict['draggable'] = False
                if 'selectable' in marker._plot_options:
                    ddict['selectable'] = True
                    if QT:
                        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                else:
                    ddict['selectable'] = False
                ddict['x'] = self._x1
                ddict['y'] = self._y1
                ddict['xpixel'] = self._x1Pixel
                ddict['ypixel'] = self._y1Pixel
                self._callback(ddict)
            elif QT:
                if self._originalCursorShape in [QtCore.Qt.SizeHorCursor,
                                QtCore.Qt.SizeVerCursor,
                                QtCore.Qt.PointingHandCursor,
                                QtCore.Qt.OpenHandCursor,
                                QtCore.Qt.SizeAllCursor]:
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
        if (not self.__zooming) and (not self.__drawing):
            return

        if self._x0 is None:
            # this happened when using the middle button
            return

        if self.__zooming or \
           (self.__drawing and (self._drawModePatch == 'rectangle')):
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
                    self._zoomRectangle = Rectangle(xy=(x,y),
                                                   width=w,
                                                   height=h,
                                                   color=self._zoomColor,
                                                   fill=False)
                    self.ax.add_patch(self._zoomRectangle)
                else:
                    self._zoomRectangle.set_bounds(x, y, w, h)
                    #self._zoomRectangle._update_patch_transform()
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
                    self._drawingPatch = Rectangle(xy=(x,y),
                                                   width=w,
                                                   height=h,
                                                   fill=False,
                                                   color=color)
                    self._drawingPatch.set_hatch('.')
                    self.ax.add_patch(self._drawingPatch)
                else:
                    self._drawingPatch.set_bounds(x, y, w, h)
                    #self._zoomRectangle._update_patch_transform()
        if self.__drawing:
            if self._drawingPatch is None:
                self._mouseData = numpy.zeros((2,2), numpy.float32)
                self._mouseData[0,0] = self._x0
                self._mouseData[0,1] = self._y0
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = self._y1
                color = self._getDrawingColor()
                self._drawingPatch = Polygon(self._mouseData,
                                             closed=True,
                                             fill=False,
                                             color=color)
                self.ax.add_patch(self._drawingPatch)
            elif self._drawModePatch == 'rectangle':
                # already handled, just for compatibility
                self._mouseData = numpy.zeros((2,2), numpy.float32)
                self._mouseData[0,0] = self._x0
                self._mouseData[0,1] = self._y0
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = self._y1
            elif self._drawModePatch == 'line':
                self._mouseData[0,0] = self._x0
                self._mouseData[0,1] = self._y0
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'hline':
                xmin, xmax = self.ax.get_xlim()
                self._mouseData[0,0] = xmin
                self._mouseData[0,1] = self._y1
                self._mouseData[1,0] = xmax
                self._mouseData[1,1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'vline':
                ymin, ymax = self.ax.get_ylim()
                self._mouseData[0,0] = self._x1
                self._mouseData[0,1] = ymin
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = ymax
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'polygon':
                self._mouseData[-1,0] = self._x1
                self._mouseData[-1,1] = self._y1
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
        if DEBUG:
            print("onMouseReleased, event = ",event.xdata, event.ydata)
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
            #right click
            if self.__drawing:
                self.__drawing = False
                #self._drawingPatch = None
                ddict = {}
                ddict['event'] = 'drawingFinished'
                ddict['type']  = '%s' % self._drawModePatch
                ddict['data']  = self._mouseData * 1
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
                self._mouseData = numpy.resize(self._mouseData, (nrows+1,2))
            self._mouseData[-1,0] = self._x1
            self._mouseData[-1,1] = self._y1
            self._drawingPatch.set_xy(self._mouseData)
            if self._drawModePatch not in ['polygon']:
                self._emitDrawingSignal("drawingFinished")

        if self._x0 is None:
            if event.inaxes != self.ax:
                if DEBUG:
                    print("on MouseReleased RETURNING")
            else:
                print("How can it be here???")
            return
        if self._zoomRectangle is None:
            currentTime = time.time()
            deltaT =  currentTime - self.__time0
            if (deltaT < 0.150) or (self.__time0 < 0) or (not self.__zooming) or\
               ((self._x1 == self._x0) and (self._y1 == self._y0)):
                # single or double click, no zooming
                self.__zooming = False
                ddict = {'x':event.xdata,
                         'y':event.ydata,
                         'xpixel':event.x,
                         'ypixel':event.y}
                leftButton = 1
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
                axes = artist.axes
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
        #ddict['xdata'] = numpy.array(self._drawingPatch.get_x())
        #ddict['ydata'] = numpy.array(self._drawingPatch.get_y())
        #print(dir(self._drawingPatch))
        a = self._drawingPatch.get_xy()
        ddict['points'] = numpy.array(a)
        ddict['points'].shape = -1, 2
        ddict['xdata'] = ddict['points'][:, 0]
        ddict['ydata'] = ddict['points'][:, 1]
        #print(numpyvstack(a))
        #pixels = self.ax.transData.transform(numpyvstack(a).T)
        #xPixel, yPixels = pixels.T
        if self._drawModePatch in ["rectangle", "circle"]:
            # we need the rectangle containing it
            ddict['x'] = ddict['points'][:, 0].min()
            ddict['y'] = ddict['points'][:, 1].min()
            ddict['width'] = self._drawingPatch.get_width()
            ddict['height'] = self._drawingPatch.get_height()
        elif self._drawModePatch in ["ellipse"]:
            #we need the rectangle but given the four corners
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
        current = self.ax.get_ylim()
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
            #top, bottom = current
        else:
            self.ax.set_ylim(ymin, ymax)
            #bottom, top = current

        if y2min is not None and y2max is not None:
            if y2max < y2min:
                y2min, y2max = y2max, y2min
            if self.ax2.yaxis_inverted():
                bottom, top = y2max, y2min
            else:
                bottom, top = y2min, y2max
            self.ax2.set_ylim(bottom, top)

        # if second axis was not properly initialized, this does not work
        #if 0 and hasattr(self.ax2, "get_visible") and self.ax2.get_visible():
        #    #print("BOTTOM, TOP = ", bottom, top)
        #    bottom2, top2 = self.ax2.get_ylim()
        #    #print("BOTTOM2, TO2 = ", bottom2, top2)
        #    i2Range = top2 - bottom2
        #    if i2Range > 0:
        #        ymin2 = bottom2 + i2Range * (ymin - bottom)/(top - bottom)
        #        ymax2 = bottom2 + i2Range * (ymax - bottom)/(top - bottom)
        #        #print("OBTAINED = ", ymin2, ymax2)
        #        if self.ax2.yaxis_inverted():
        #            self.ax2.set_ylim(ymax2, ymin2)
        #        else:
        #            self.ax2.set_ylim(ymin2, ymax2)
        # Next line forces a square display region
        #self.ax.set_aspect((xmax-xmin)/float(ymax-ymin))
        #self.draw()
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
        #self.ax2.set_ylim(ymin2, ymax2)
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

        #self.ax2.set_autoscaley_on(True)
        self._zoomStack = []

    def getDataLimits(self, axesLabel='left'):
        if axesLabel == 'right':
            axes = self.ax2
        else:
            axes = self.ax
        if DEBUG:
            print("CALCULATING limits ", axes.get_label())
        xmin = None
        for line2d in axes.lines:
            label = line2d.get_label()
            if label.startswith("__MARKER__"):
                #it is a marker
                continue
            lineXMin = None
            if hasattr(line2d, "_plot_info"):
                if line2d._plot_info.get("axes", "left") != axesLabel:
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
                #it is a marker
                continue
            lineXMin = None
            if hasattr(line2d, "_plot_info"):
                if line2d._plot_info.get("axes", "left") != axesLabel:
                    continue
                if "xmin" in line2d._plot_info:
                    lineXMin = line2d._plot_info["xmin"]
                    lineXMax = line2d._plot_info["xmax"]
                    lineYMin = line2d._plot_info["ymin"]
                    lineYMax = line2d._plot_info["ymax"]
            if lineXMin is None:
                print("CANNOT CALCULATE LIMITS")
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

            #l, b, w, h = self.ax.get_position(original=True).bounds
            #box_aspect = figAspect * (h / float(w))

            #dataRatio = box_aspect / A
            dataRatio = (ySize / xSize) * A

            y_expander = dataRatio - figAspect
            # If y_expander > 0, the dy/dx viewLim ratio needs to increase
            if abs(y_expander) < 0.005:
                #good enough
                pass
            else:
                # this works for any data ratio
                if y_expander < 0:
                    #print("adjust_y")
                    deltaY = xSize * (figAspect / A) - ySize
                    yc = 0.5 * (ymin + ymax)
                    ymin = yc - (ySize + deltaY) * 0.5
                    ymax = yc + (ySize + deltaY) * 0.5
                else:
                    #print("ADJUST X")
                    deltaX = ySize * (A / figAspect) - xSize
                    xc = 0.5 * (xmin + xmax)
                    xmin = xc - (xSize + deltaX) * 0.5
                    xmax = xc + (xSize + deltaX) * 0.5
        if DEBUG:
            print("CALCULATED LIMITS = ", xmin, xmax, ymin, ymax)
        return xmin, xmax, ymin, ymax

    def resizeEvent(self, ev):
        # we have to get rid of the copy of the underlying image
        self._background = None
        FigureCanvas.resizeEvent(self, ev)

    if DEBUG:
        def draw(self):
            print("Draw called")
            super(MatplotlibGraph, self).draw()

class MatplotlibBackend(PlotBackend.PlotBackend):
    def __init__(self, parent=None, **kw):
       	#self.figure = Figure(figsize=size, dpi=dpi) #in inches
        self.graph = MatplotlibGraph(parent, **kw)
        self.ax2 = self.graph.ax2
        self.ax = self.graph.ax
        PlotBackend.PlotBackend.__init__(self, parent)
        self._parent = parent
        self._logX = False
        self._logY = False
        self.setZoomModeEnabled = self.graph.setZoomModeEnabled
        self.setDrawModeEnabled = self.graph.setDrawModeEnabled
        self.isZoomModeEnabled = self.graph.isZoomModeEnabled
        self.isDrawModeEnabled = self.graph.isDrawModeEnabled
        self.getDrawMode = self.graph.getDrawMode
        self._oldActiveCurve = None
        self._oldActiveCurveLegend = None
        # should one have two methods, for enable and for show
        self._rightAxisEnabled = False
        self.enableAxis('right', False)
        self._graphCursor = None
        self.matplotlibVersion = matplotlib.__version__

    def setGraphCursor(self, flag=True, color=None, linewidth=None, linestyle=None):
        if color is None:
            color = "black"
        if linewidth is None:
            linewidth = 1
        if linestyle is None:
            linestyle="-"
        self._graphCursorConfiguration = (color, linewidth, linestyle)
        if flag:
            if self._graphCursor is None:
                self._graphCursor = Cursor(self.ax,
                                       useblit=True,
                                       color=color,
                                       linewidth=linewidth,
                                       linestyle=linestyle)
            self._graphCursor.visible = True
        else:
            if self._graphCursor is not None:
                self._graphCursor.visible = False

    def getGraphCursor(self):
        if self._graphCursor is None:
            return None
        elif not self._graphCursor.visible:
            return None
        else:
            return self._graphCursorConfiguration * 1            

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linewidth=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=1, selectable=True, **kw):
        if legend is None:
            legend = "Unnamed curve"
        if replace:
            self.clearCurves()
        else:
            self.removeCurve(legend, replot=False)
        if color is None:
            color = self._activeCurveColor
        if len(color) == 4:
                if type(color[3]) in [type(1), numpy.uint8, numpy.int8]:
                    color = numpy.array(color, dtype=numpy.float)/255.

        brush = color
        style = linestyle
        if linewidth is None:
            linewidth = 1
        if yaxis == "right":
            axisId = yaxis
        else:
            axisId = "left"
        fill = kw.get('plot_fill', False)
        if axisId == "right":
            axes = self.ax2
            if self._rightAxisEnabled is None:
                # never initialized
                self.enableAxis(axisId, True)
        else:
            axes = self.ax
        if selectable:
            picker = 3
        else:
            picker = None
        scatterPlot = False
        if hasattr(color, "dtype"):
            if len(color) == len(x):
                scatterPlot = True
        if scatterPlot:
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

            if style not in [" ", None]:
                # scatter plot with an actual line ...
                # we need to assign a color ...
                curveList = axes.plot(x, y, label=legend,
                                      linestyle=style,
                                      color=actualColor[0],
                                      linewidth=linewidth,
                                      picker=picker,
                                      marker=None,
                                      **kw)
                curveList[-1]._plot_info = {'color':actualColor,
                                              'linewidth':linewidth,
                                              'brush':brush,
                                              'style':style,
                                              'symbol':symbol,
                                              'label':legend,
                                              'axes':axisId,
                                              'fill':fill,
                                              'xlabel':xlabel,
                                              'ylabel':ylabel}
                if hasattr(x, "min") and hasattr(y, "min"):
                    curveList[-1]._plot_info['xmin'] = nanmin(x)
                    curveList[-1]._plot_info['xmax'] = nanmax(x)
                    curveList[-1]._plot_info['ymin'] = nanmin(y)
                    curveList[-1]._plot_info['ymax'] = nanmax(y)
            # scatter plot is a collection
            curveList = [pathObject]
            if self._logY:
                axes.set_yscale('log')
        elif self._logY:
            curveList = axes.semilogy( x, y, label=legend,
                                          linestyle=style,
                                          color=color,
                                          linewidth=linewidth,
                                          picker=picker,
                                          **kw)
        else:
            curveList = axes.plot( x, y, label=legend,
                                  linestyle=style,
                                  color=color,
                                  linewidth=linewidth,
                                  picker=picker,
                                  **kw)

        # errorbar is a container?
        #axes.errorbar(x,y, label=legend,yerr=numpy.sqrt(y), linestyle=" ",color='b')

        # nice effects:
        #curveList[-1].set_drawstyle('steps-mid')
        if fill:
            axes.fill_between(x, 1.0e-8, y)
        #curveList[-1].set_fillstyle('bottom')
        if hasattr(curveList[-1], "set_marker"):
            curveList[-1].set_marker(symbol)
        curveList[-1]._plot_info = {'color':color,
                                      'linewidth':linewidth,
                                      'brush':brush,
                                      'style':style,
                                      'symbol':symbol,
                                      'label':legend,
                                      'axes':axisId,
                                      'fill':fill,
                                      'xlabel':xlabel,
                                      'ylabel':ylabel}
        if hasattr(x, "min") and hasattr(y, "min"):
            # this is needed for scatter plots because I do not know
            # how to recover the data yet, it can speed up limits too
            curveList[-1]._plot_info['xmin'] = nanmin(x)
            curveList[-1]._plot_info['xmax'] = nanmax(x)
            curveList[-1]._plot_info['ymin'] = nanmin(y)
            curveList[-1]._plot_info['ymax'] = nanmax(y)
        if self._activeCurveHandling:
            if self._oldActiveCurve in self.ax.lines:
                if self._oldActiveCurve.get_label() == legend:
                    curveList[-1].set_color(self._activeCurveColor)
            elif self._oldActiveCurveLegend == legend:
                curveList[-1].set_color(self._activeCurveColor)
        curveList[-1].axes = axes # set_axes(axes) deprecated in version 1.5
        curveList[-1].set_zorder(z)
        if replot:
            self.replot()
        # If I return the instance, later on cannot make a copy.deepcopy
        # of the info and asks me to use "frozen" instead
        #return curveList[-1]
        return legend

    def addItem(self, x, y, legend, info=None, replace=False, replot=True, **kw):
        if replace:
            self.clearItems()
        else:
            # make sure we do not cummulate images with same name
            self.removeItem(legend, replot=False)
        item = None
        shape = kw.get('shape', "polygon")
        if shape not in ['line', 'hline', 'vline', 'rectangle', 'polygon']:
            raise NotImplemented("Unsupported item shape %s" % shape)
        label = kw.get('label', legend)
        color = kw.get('color', 'black')
        fill = kw.get('fill', True)
        xView = numpy.array(x, copy=False)
        yView = numpy.array(y, copy=False)
        label = "__ITEM__" + label
        if shape in ["line", "hline", "vline"]:
            print("Not implemented")
            return legend
        elif shape in ["hline"]:
            if hasattr(y, "__len__"):
                y = y[-1]
            line = self.ax.axhline(y, label=label, color=color)
            return line
        elif shape in ["vline"]:
            if hasattr(x, "__len__"):
                x = x[-1]
            line = self.ax.axvline(x, label=label, color=color)
            return line
        elif shape in ['rectangle']:
            xMin = nanmin(xView)
            xMax = nanmax(xView)
            yMin = nanmin(yView)
            yMax = nanmax(yView)
            w = xMax - xMin
            h = yMax - yMin
            item = Rectangle(xy=(xMin,yMin),
                             width=w,
                             height=h,
                             fill=False,
                             color=color)
            if fill:
                item.set_hatch('.')
        elif shape in ['polygon']:
            xView.shape = 1, -1
            yView.shape = 1, -1
            item = Polygon(numpyvstack((xView, yView)).T,
                            closed=True,
                            fill=False,
                            label=label,
                            color=color)
            if fill:
                #item.set_hatch('+')
                item.set_hatch('/')
        if item is None:
            print("Undefined item")
            print("shape = ", shape)
            print("legend = ", legend)
            return
        self.ax.add_patch(item)
        if replot:
            self.ax.figure.canvas.draw()
        return item

    def clear(self):
        """
        Clear all curvers and other items from the plot
        """
        n = list(range(len(self.ax.lines)))
        n.reverse()
        for i in n:
            line2d = self.ax.lines[i]
            line2d.remove()
            del line2d
        self.ax.clear()

    def clearImages(self):
        n = list(range(len(self.ax.images)))
        n.reverse()
        for i in n:
            image = self.ax.images[i]
            image.remove()
            del image
            del self.ax.images[i]

        n = list(range(len(self.ax.artists)))
        n.reverse()
        for i in n:
            artist = self.ax.artists[i]
            label = artist.get_label()
            if label.startswith("__IMAGE__"):
                artist.remove()
                del artist

    def clearCurves(self):
        """
        Clear all curves from the plot. Not the markers!!
        """
        for axes in [self.ax, self.ax2]:
            n = list(range(len(axes.lines)))
            n.reverse()
            for i in n:
                line2d = axes.lines[i]
                label = line2d.get_label()
                if label.startswith("__MARKER__"):
                    #it is a marker
                    continue
                line2d.remove()
                del line2d

    def clearMarkers(self):
        """
        Clear all markers from the plot. Not the curves!!
        """
        for axes in [self.ax, self.ax2]:
            n = list(range(len(axes.lines)))
            n.reverse()
            for i in n:
                line2d = axes.lines[i]
                label = line2d.get_label()
                if label.startswith("__MARKER__"):
                    #it is a marker
                    if hasattr(line2d, "_infoText"):
                        line2d._infoText.remove()
                    line2d.remove()
                    del line2d

    def clearItems(self):
        """
        Clear items, not markers, not curves
        """
        for axes in [self.ax, self.ax2]:
            n = list(range(len(axes.patches)))
            n.reverse()
            for i in n:
                item = axes.patches[i]
                label = item.get_label()
                if label.startswith("__ITEM__"):
                    item.remove()
                    del item

    def removeItem(self, handle, replot=True):
        if hasattr(handle, "remove"):
            for axes in [self.ax, self.ax2]:
                if handle in axes.patches:
                    handle.remove()
                    del handle
                    break
        else:
            # we have received a legend!
            # we have received a legend!
            legend = handle
            handle = None
            for axes in [self.ax, self.ax2]:
                if handle is not None:
                    break
                for item in axes.patches:
                    label = item.get_label()
                    if label == ("__ITEM__"+legend):
                        handle = item
                        break
            if handle is not None:
                handle.remove()
                del handle
        if replot:
            self.replot()

    def getGraphXLimits(self):
        """
        Get the graph X (bottom) limits.
        :return:  Minimum and maximum values of the X axis
        """
        vmin, vmax = self.ax.get_xlim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def getGraphYLimits(self, axis="left"):
        if axis == "right":
            ax = self.ax2
        else:
            ax = self.ax
        vmin, vmax = ax.get_ylim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def getWidgetHandle(self):
        """
        :return: Backend widget.
        """
        if hasattr(self.graph, "get_tk_widget"):
            return self.graph.get_tk_widget()
        else:
            return self.graph

    def enableAxis(self, axis, flag=True):
        """
        :param axis: Axis to be shown or hidden
        :type axis: string (left, right, top, bottom)
        :param flag: Boolean (default, True)
        """
        if axis == "right":
            self.ax2.get_yaxis().set_visible(flag)
            self._rightAxisEnabled = flag
        elif axis == "left":
            self.ax.get_yaxis().set_visible(flag)
        else:
            print("unhandled axis %s" % axis)

    def insertMarker(self, x, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False, replot=True,
                      symbol=None, constraint=None,
                      **kw):
        """
        :param x: Horizontal position of the marker in graph coordenates
        :type x: float
        :param y: Vertical position of the marker in graph coordenates
        :type y: float
        :param label: Legend associated to the marker
        :type label: string
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :param str symbol: Symbol representing the marker
        :param constraint: None or a function filtering marker displacements.
        :return: Handle used by the backend to univocally access the marker
        """
        #line = self.ax.axvline(x, picker=True)
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits()
        if x is None:
            x = 0.5 * (xmax + xmin)
        if y is None:
            y = 0.5 * (ymax + ymin)

        # Apply constraint to provided position
        if draggable and constraint is not None:
            x, y = constraint(x, y)

        if legend is None:
            legend = "Unnamed marker"
        if text is None:
            text = kw.get("label", None)
            if text is not None:
                print("Deprecation warning: use 'text' instead of 'label'")
        self.removeMarker(legend, replot=False)

        legend = "__MARKER__" + legend
        if symbol is None:
            symbol = "+"
        markersize=10.
        if selectable or draggable:
            line = self.ax.plot(x, y, label=legend,
                                      linestyle=" ",
                                      color=color,
                                      picker=5,
                                      marker=symbol,
                                      markersize=markersize)[-1]
        else:
            line = self.ax.plot(x, y, label=legend,
                                      linestyle=" ",
                                      color=color,
                                      marker=symbol,
                                      markersize=markersize)[-1]
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

        #line.set_ydata(numpy.array([1.0, 10.], dtype=numpy.float32))
        line._plot_options = ["marker"]
        if selectable:
            line._plot_options.append('selectable')
        if draggable:
            line._plot_options.append('draggable')
        if replot:
            self.replot()
        return line

    def insertXMarker(self, x, legend=None, text=None,
                      color='k', selectable=False, draggable=False,
                      replot=True, **kw):
        """
        :param x: Horizontal position of the marker in graph coordenates
        :type x: float
        :param legend: Legend associated to the marker
        :type legend: string
        :param label: Text associated to the marker
        :type label: string or None
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :return: Handle used by the backend to univocally access the marker
        """
        #line = self.ax.axvline(x, picker=True)
        if legend is None:
            legend = "Unnamed marker"
        if text is None:
            text = kw.get("label", None)
            if text is not None:
                print("Deprecation warning: use 'text' instead of 'label'")
        self.removeMarker(legend, replot=False)
        legend = "__MARKER__" + legend
        if selectable or draggable:
            line = self.ax.axvline(x, label=legend, color=color, picker=5)
        else:
            line = self.ax.axvline(x, label=legend, color=color)
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
        #line.set_ydata(numpy.array([1.0, 10.], dtype=numpy.float32))
        line._plot_options = ["xmarker"]
        if selectable:
            line._plot_options.append('selectable')
        if draggable:
            line._plot_options.append('draggable')
        if replot:
            self.replot()
        return line

    def insertYMarker(self, y, legend=None, text=None,
                      color='k', selectable=False, draggable=False,
                      replot=True, **kw):
        """
        :param y: Vertical position of the marker in graph coordenates
        :type y: float
        :param legend: Legend associated to the marker
        :type legend: string
        :param label: Text associated to the marker
        :type label: string or None
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :return: Handle used by the backend to univocally access the marker
        """
        if legend is None:
            legend = "Unnamed marker"
        if text is None:
            text = kw.get("label", None)
            if text is not None:
                print("Deprecation warning: use 'text' instead of 'label'")
        legend = "__MARKER__" + legend
        if selectable or draggable:
            line = self.ax.axhline(y, label=legend, color=color, picker=5)
        else:
            line = self.ax.axhline(y, label=legend, color=color)
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
        if replot:
            self.replot()
        return line

    def isXAxisAutoScale(self):
        if self._xAutoScale:
            return True
        else:
            return False

    def isYAxisAutoScale(self):
        if self._yAutoScale:
            return True
        else:
            return False

    def removeCurve(self, handle, replot=True):
        if hasattr(handle, "remove"):
            if handle in self.ax.lines:
                handle.remove()
            if handle in self.ax2.lines:
                handle.remove()
            if handle in self.ax.collections:
                handle.remove()
            if handle in self.ax2.collections:
                handle.remove()
        else:
            # we have received a legend!
            legend = handle
            testLists = [self.ax.lines, self.ax2.lines,
                         self.ax.collections, self.ax2.collections]
            for container in testLists:
                for line2d in container:
                    handle = None
                    label = line2d.get_label()
                    if label == legend:
                        handle = line2d
                    if handle is not None:
                        handle.remove()
                        del handle
        if replot:
            self.replot()

    def removeImage(self, handle, replot=True):
        if hasattr(handle, "remove"):
            if (handle in self.ax.images) or (handle in self.ax.artists):
                handle.remove()
        else:
            # we have received a legend!
            legend = handle
            handle = None
            for item in self.ax.artists:
                label = item.get_label()
                if label == ("__IMAGE__" + legend):
                    handle = item
            if handle is None:
                for item in self.ax.images:
                    label = item.get_label()
                    if label == legend:
                        handle = item
            if handle is not None:
                handle.remove()
                del handle
        if replot:
            self.replot()

    def removeMarker(self, handle, replot=True):
        if hasattr(handle, "remove"):
            self._removeInfoText(handle)
            handle.remove()
            del handle
        else:
            # we have received a legend!
            legend = handle
            done = False
            for axes in [self.ax, self.ax2]:
                for line2d in axes.lines:
                    if done:
                        break
                    label = line2d.get_label()
                    if label == ("__MARKER__"+legend):
                        if hasattr(line2d, "_infoText"):
                            line2d._infoText.remove()
                        line2d.remove()
                        del line2d
                        done = True
        if replot:
            self.replot()

    def _removeInfoText(self, handle):
        if hasattr(handle, "_infoText"):
            t = handle._infoText
            handle._infoText = None
            t.remove()
            del t

    def resetZoom(self, dataMargins=None):
        """
        It should autoscale any axis that is in autoscale mode
        """
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits()
        xAuto = self.isXAxisAutoScale()
        yAuto = self.isYAxisAutoScale()
        if xAuto and yAuto:
            self.graph.resetZoom(dataMargins)
        elif yAuto:
            self.graph.resetZoom(dataMargins)
            self.setGraphXLimits(xmin, xmax)
        elif xAuto:
            self.graph.resetZoom(dataMargins)
            self.setGraphYLimits(ymin, ymax)
        else:
            if DEBUG:
                print("Nothing to autoscale")
        #xmin2, xmax2, ymin2, ymax2 = self.graph.getDataLimits('right')
        #self.ax2.figure.sca(self.ax2)
        #self.ax2.set_ylim(10., 100.)
        #self.ax2.figure.sca(self.ax)
        self._zoomStack = []

        self.replot()
        return

    def replot(self):
        """
        Update plot
        """
        if self._rightAxisEnabled is not None:
            # the right axis was initialized at a certain point
            # so, we have to see if there is something still mapped
            # to that axis.
            # For the time being we only check lines
            if not len(self.ax2.lines):
                self.enableAxis('right', False)
                self._rightAxisEnabled = None
        #print("Calling draw")
        self.graph.draw()
        #print("Back from draw")
        """
        if QT:
            w = self.getWidgetHandle()
            QtGui.qApp.postEvent(w, QtGui.QResizeEvent(w.size(),
                                                   w.size()))
        """
        return

    def saveGraph(self, fileName, fileFormat='svg', dpi=None , **kw):
        # fileName can be also a StringIO or file instance
        fig = self.ax.figure
        if dpi is not None:
            fig.savefig(fileName, format=fileFormat, dpi=dpi)
        else:
            fig.savefig(fileName, format=fileFormat)
        fig = None
        return

    def setActiveCurve(self, legend, replot=True):
        if not self._activeCurveHandling:
            return
        if hasattr(legend, "_plot_info"):
            # we have received an actual item
            handle = legend
        else:
            # we have received a legend
            handle = None
            for line2d in self.ax.lines:
                label = line2d.get_label()
                if label.startswith("__MARKER__"):
                    continue
                if label == legend:
                    handle = line2d
                    axes = self.ax
                    break
            if handle is None:
                for line2d in self.ax2.lines:
                    label = line2d.get_label()
                    if label.startswith("__MARKER__"):
                        continue
                    if label == legend:
                        handle = line2d
                        axes = self.ax2
                        break
        if handle is not None:
            handle.set_color(self._activeCurveColor)
        else:
            raise KeyError("Curve %s not found" % legend)
        if self._oldActiveCurve in self.ax.lines:
            if self._oldActiveCurve._plot_info['label'] != legend:
                color = self._oldActiveCurve._plot_info['color']
                self._oldActiveCurve.set_color(color)
        elif self._oldActiveCurve in self.ax2.lines:
            if self._oldActiveCurve._plot_info['label'] != legend:
                color = self._oldActiveCurve._plot_info['color']
                self._oldActiveCurve.set_color(color)
        elif self._oldActiveCurveLegend is not None:
            if self._oldActiveCurveLegend != handle._plot_info['label']:
                done = False
                for line2d in self.ax.lines:
                    label = line2d.get_label()
                    if label == self._oldActiveCurveLegend:
                        color = line2d._plot_info['color']
                        line2d.set_color(color)
                        done = True
                        break
                if not done:
                    for line2d in self.ax2.lines:
                        label = line2d.get_label()
                        if label == self._oldActiveCurveLegend:
                            color = line2d._plot_info['color']
                            line2d.set_color(color)
                            break
        #update labels according to active curve???
        if hasattr(handle, "_plot_info"):
            xLabel = handle._plot_info.get("xlabel", None)
            yLabel = handle._plot_info.get("ylabel", None)
            if (xLabel is not None) and (yLabel is not None):
                axes.set_xlabel(xLabel)
                axes.set_ylabel(yLabel)
        self._oldActiveCurve = handle
        self._oldActiveCurveLegend = handle.get_label()
        if replot:
            self.replot()

    def setCallback(self, callbackFunction):
        self.graph.setCallback(callbackFunction)
        # Should I call the base to keep a copy?
        # It does not seem necessary since the graph will do it.

    def getGraphTitle(self):
        return self.ax.get_title()

    def getGraphXLabel(self):
        return self.ax.get_xlabel()

    def getGraphYLabel(self):
        return self.ax.get_ylabel()

    def setGraphTitle(self, title=""):
        self.ax.set_title(title)

    def setGraphXLabel(self, label="X"):
        self.ax.set_xlabel(label)

    def setGraphXLimits(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)
        self.graph.emitLimitsChangedSignal()

    def setGraphYLabel(self, label="Y"):
        self.ax.set_ylabel(label)

    def setGraphYLimits(self, ymin, ymax):
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)
        self.graph.emitLimitsChangedSignal()

    def setLimits(self, xmin, xmax, ymin, ymax):
        # Overrides PlotBackend to send a single limitsChanged event.
        self.ax.set_xlim(xmin, xmax)
        if self.ax.yaxis_inverted():
            self.ax.set_ylim(ymax, ymin)
        else:
            self.ax.set_ylim(ymin, ymax)
        self.graph.emitLimitsChangedSignal()

    def setXAxisAutoScale(self, flag=True):
        if flag:
            self._xAutoScale = True
        else:
            self._xAutoScale = False

    def setXAxisLogarithmic(self, flag):
        if flag:
            self._logX = True
            if hasattr(self.ax2, "get_visible"):
                if self.ax2.get_visible():
                    self.ax2.set_xscale('log')
            self.ax.set_xscale('log')
        else:
            self._logX = False
            if hasattr(self.ax2, "get_visible"):
                if self.ax2.get_visible():
                    self.ax2.set_xscale('linear')
            self.ax.set_xscale('linear')

    def setYAxisAutoScale(self, flag=True):
        if flag:
            self._yAutoScale = True
        else:
            self._yAutoScale = False

    def setYAxisLogarithmic(self, flag):
        """
        :param flag: If True, the left axis will use a log scale
        :type flag: boolean
        """
        if flag:
            self._logY = True
            if hasattr(self.ax2, "get_visible"):
                if self.ax2.get_visible():
                    self.ax2.set_yscale('log')
            self.ax.set_yscale('log')
        else:
            self._logY = False
            if hasattr(self.ax2, "get_visible"):
                if self.ax2.get_visible():
                    self.ax2.set_yscale('linear')
            self.ax.set_yscale('linear')

    def addImage(self, data, legend=None, info=None,
                    replace=True, replot=True,
                    xScale=None, yScale=None, z=0,
                    selectable=False, draggable=False,
                    colormap=None, **kw):
        """
        :param data: (nrows, ncolumns) data or (nrows, ncolumns, RGBA) ubyte array
        :type data: numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the image
        :type info: dict or None
        :param replace: Flag to indicate if already existing images are to be deleted
        :type replace: boolean default True
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        :param xScale: Two floats defining the x scale
        :type xScale: list or numpy.ndarray
        :param yScale: Two floats defining the y scale
        :type yScale: list or numpy.ndarray
        :param z: level at which the image is to be located (to allow overlays).
        :type z: A number bigger than or equal to zero (default)
        :param selectable: Flag to indicate if the image can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the image can be moved
        :type draggable: boolean, default False
        :param colormap: Dictionary describing the colormap to use (or None)
        :type colormap: Dictionnary or None (default). Ignored if data is RGB(A)
        :returns: The legend/handle used by the backend to univocally access it.
        """
        # Non-uniform image
        #http://wiki.scipy.org/Cookbook/Histograms
        # Non-linear axes
        #http://stackoverflow.com/questions/11488800/non-linear-axes-for-imshow-in-matplotlib
        if legend is None:
            legend = 'Unnamed image'

        if replace:
            self.clearImages()
        else:
            # make sure we do not cummulate images with same name
            self.removeImage(legend, replot=False)

        if xScale is None:
            xScale = [0.0, 1.0]
        if yScale is None:
            yScale = [0.0, 1.0]
        h, w = data.shape[0:2]
        xmin = xScale[0]
        xmax = xmin + xScale[1] * w
        ymin = yScale[0]
        ymax = ymin + yScale[1] * h
        extent = (xmin, xmax, ymax, ymin)

        if selectable or draggable:
            picker = True
        else:
            picker = None
        shape = data.shape
        if 0:
            # this supports non regularly spaced coordenates!!!!
            x = xmin + numpy.arange(w) * xScale[1]
            y = ymin + numpy.arange(h) * yScale[1]
            image = NonUniformImage(self.ax,
                                    interpolation='nearest',
                                    #aspect='auto',
                                    extent=extent,
                                    picker=picker,
                                    cmap=cmap)



            image.set_data(x, y, data)
            xmin, xmax = self.getGraphXLimits()
            ymin, ymax = self.getGraphYLimits()
            self.ax.images.append(image)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        elif 1:
            #the normalization can be a source of time waste
            # Two possibilities, we receive data or a ready to show image
            if len(data.shape) == 3:
                if data.shape[-1] == 4:
                    # force alpha(?)
                    # data[:,:,3] = 255
                    pass
            if len(shape) == 3:
                # RGBA image
                # TODO: Possibility to mirror the image
                # in case of pixmaps just setting
                # extend = (xmin, xmax, ymax, ymin)
                # instead of (xmin, xmax, ymin, ymax)
                extent = (xmin, xmax, ymin, ymax)
                if tuple(xScale) != (0., 1.) or tuple(yScale) != (0., 1.):
                    # for the time being not properly handled
                    imageClass = AxesImage
                elif (shape[0] * shape[1]) > 5.0e5:
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
                if colormap is None:
                    colormap = self.getDefaultColormap()
                cmap = self.__getColormap(colormap['name'])
                if colormap['normalization'].startswith('log'):
                    vmin, vmax = None, None
                    if not colormap['autoscale']:
                        if colormap['vmin'] > 0.:
                            vmin = colormap['vmin']
                        if colormap['vmax'] > 0.:
                            vmax = colormap['vmax']

                        if vmin is None or vmax is None:
                            print('Warning: ' +
                                  'Log colormap with negative bounds, ' +
                                  'changing bounds to positive ones.')
                        elif vmin > vmax:
                            print('Warning: Colormap bounds are inverted.')
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
                            print('Warning: Colormap bounds are inverted.')
                            vmin, vmax = vmax, vmin

                    norm = Normalize(vmin, vmax)

                # try as data
                if tuple(xScale) != (0., 1.) or tuple(yScale) != (0., 1.):
                    # for the time being not properly handled
                    imageClass = AxesImage
                elif (shape[0] * shape[1]) > 5.0e5:
                    imageClass = ModestImage
                else:
                    imageClass = AxesImage
                image = imageClass(self.ax,
                              label="__IMAGE__"+legend,
                              interpolation='nearest',
                              #origin=
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
            #self.ax.draw_artist(image)
        image._plot_info = {'label':legend,
                            'type':'image',
                            'xScale':xScale,
                            'yScale':yScale,
                            'z':z}
        image._plot_options = []
        if draggable:
            image._plot_options.append('draggable')
        if selectable:
            image._plot_options.append('selectable')
        return image

    def invertYAxis(self, flag=True):
        if flag:
            if not self.ax.yaxis_inverted():
                self.ax.invert_yaxis()
        else:
            if self.ax.yaxis_inverted():
                self.ax.invert_yaxis()

    def isYAxisInverted(self):
        return self.ax.yaxis_inverted()

    def showGrid(self, flag=True):
        if flag == 1:
            if hasattr(self.ax.xaxis, "set_tick_params"):
                self.ax.xaxis.set_tick_params(which='major')
                self.ax.yaxis.set_tick_params(which='major')
            self.ax.grid(which='major')
        elif flag == 2:
            if hasattr(self.ax.xaxis, "set_tick_params"):
                self.ax.xaxis.set_tick_params(which='both')
                self.ax.yaxis.set_tick_params(which='both')
            self.ax.grid(which='both')
        elif flag:
            if hasattr(self.ax.xaxis, "set_tick_params"):
                self.ax.xaxis.set_tick_params(which='major')
                self.ax.yaxis.set_tick_params(which='major')
            self.ax.grid(True)
        else:
            self.ax.grid(False)
        self.replot()

    def keepDataAspectRatio(self, flag=True):
        """
        :param flag:  True to respect data aspect ratio
        :type flag: Boolean, default True
        """
        if flag:
            for axes in [self.ax]:
                if axes.get_aspect() not in [1.0]:
                    axes.set_aspect(1.0)
                    self.resetZoom()
        else:
            for axes in [self.ax]:
                if axes.get_aspect() not in ['auto', None]:
                    axes.set_aspect('auto')
                    self.resetZoom()

    def isKeepDataAspectRatio(self):
        return self.ax.get_aspect() in (1.0, 'equal')

    def setDefaultColormap(self, colormap=None):
        if colormap is None:
            colormap = {'name': 'gray', 'normalization':'linear',
                        'autoscale':True, 'vmin':0.0, 'vmax':1.0,
                        'colors':256}
        self._defaultColormap = colormap

    def getDefaultColormap(self):
        if not hasattr(self, "_defaultColormap"):
            self.setDefaultColormap(None)
        return self._defaultColormap

    def getSupportedColormaps(self):
        default = ['gray', 'reversed gray', 'temperature', 'red', 'green', 'blue']
        maps = [m for m in cm.datad]
        maps.sort()
        return default + maps

    def __getColormap(self, name):
        if not hasattr(self, "__temperatureCmap"):
            #initialize own colormaps
            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}
            self.__redCmap = LinearSegmentedColormap('red',cdict,256)

            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 1.0, 1.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 0.0, 0.0))}
            self.__greenCmap = LinearSegmentedColormap('green',cdict,256)

            cdict = {'red': ((0.0, 0.0, 0.0),
                             (1.0, 0.0, 0.0)),
                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),
                     'blue': ((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0))}
            self.__blueCmap = LinearSegmentedColormap('blue',cdict,256)

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
            #but limited to 256 colors for a faster display (of the colorbar)
            self.__temperatureCmap = LinearSegmentedColormap('temperature',
                                                             cdict, 256)

            #reversed gray
            cdict = {'red':     ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0)),
                     'green':   ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0)),
                     'blue':    ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0))}

            self.__reversedGrayCmap = LinearSegmentedColormap('yerg', cdict, 256)

        if name == "reversed gray":
            return self.__reversedGrayCmap
        elif name == "temperature":
            return self.__temperatureCmap
        elif name == "red":
            return self.__redCmap
        elif name == "green":
            return self.__greenCmap
        elif name == "blue":
            return self.__blueCmap
        else:
            # built in
            return cm.get_cmap(name)

    def dataToPixel(self, x=None, y=None, axis="left"):
        """
        Convert a position in data space to a position in pixels in the widget.

        :param float x: The X coordinate in data space. If None (default)
                            the middle position of the displayed data is used.
        :param float y: The Y coordinate in data space. If None (default)
                            the middle position of the displayed data is used.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :returns: The corresponding position in pixels or
                  None if the data position is not in the displayed area.
        :rtype: A tuple of 2 floats: (xPixel, yPixel) or None.
        """
        assert axis in ("left", "right")
        if "axis" == "right":
            ax = self.ax2
        else:
            ax = self.ax
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if x is None:
            x = 0.5 * (xmax - xmin)
        if y is None:
            y = 0.5 * (ymax - ymin)

        if (x > xmax)  or (x < xmin):
            return None

        if (y > ymax)  or (y < ymin):
            return None

        pixels = ax.transData.transform([x, y])
        xPixel, yPixel = pixels.T
        return xPixel, yPixel

    def pixelToData(self, x=None, y=None, axis="left"):
        assert axis in ("left", "right")
        if "axis" == "right":
            ax = self.ax2
        else:
            ax = self.ax
        inv = ax.transData.inverted()
        x, y = inv.transform((x, y))

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if (x > xmax)  or (x < xmin):
            return None

        if (y > ymax)  or (y < ymin):
            return None

        return x, y

def main(parent=None):
    from .. import Plot
    x = numpy.arange(100.)
    y = x * x
    plot = Plot.Plot(parent, backend=MatplotlibBackend)
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x + 100, -x * x, "To set Active")
    #info = {}
    #info['plot_yaxis'] = 'right'
    #plot.addCurve(x + 100, -x * x + 500, "RIGHT", info=info)
    #print("Active curve = ", plot.getActiveCurve())
    print("X Limits) = ", plot.getGraphXLimits())
    print("Y Limits = ", plot.getGraphYLimits())
    #print("All curves = ", plot.getAllCurves())
    #plot.removeCurve("dummy")
    plot.setActiveCurve("To set Active")
    #print("All curves = ", plot.getAllCurves())
    #plot.resetZoom()
    return plot

if __name__ == "__main__":
    if "tkinter" in sys.modules or "Tkinter" in sys.modules:
        root = Tk.Tk()
        parent=root
        #w = MatplotlibGraph(root)
        #Tk.mainloop()
        #sys.exit(0)
        w = main(parent)
        widget = w._plot.graph
    else:
        app = QtGui.QApplication([])
        parent=None
        w = main(parent)
        widget = w.getWidgetHandle()
    #w.invertYAxis(True)
    w.replot()
    #w.invertYAxis(True)
    data = numpy.arange(1000.*1000)
    data.shape = 10000,100
    #plot.replot()
    #w.invertYAxis(True)
    #w.replot()
    #w.widget.show()
    w.addImage(data, legend="image 0", xScale=(25, 1.0) , yScale=(-1000, 1.0),
                  selectable=True)
    w.removeImage("image 0")
    #w.invertYAxis(True)
    #w.replot()
    w.addImage(data, legend="image 1", xScale=(25, 1.0) , yScale=(-1000, 1.0),
                  replot=False, selectable=True)
    #w.invertYAxis(True)
    widget.ax.axis('auto') # appropriate for curves, no aspect ratio
    #w.widget.ax.axis('equal') # candidate for keepting aspect ratio
    #w.widget.ax.axis('scaled') # candidate for keepting aspect ratio
    w.insertXMarker(50., text="Label", color='pink', draggable=True)
    w.insertMarker(25, -5000, text="Label\n", color='pink', draggable=True)
    w.resetZoom()
    #print(w.widget.ax.get_images())
    #print(w.widget.ax.get_lines())
    if "tkinter" in sys.modules or "Tkinter" in sys.modules:
        tkWidget = w.getWidgetHandle()
        tkWidget.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        Tk.mainloop()
    else:
        widget.show()
        app.exec_()
