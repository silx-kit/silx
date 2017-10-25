# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""Matplotlib Plot backend."""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent, H. Payno"]
__license__ = "MIT"
__date__ = "16/08/2017"


import logging

import numpy


_logger = logging.getLogger(__name__)


from ... import qt

# First of all init matplotlib and set its backend
from ..matplotlib import Colormap as MPLColormap
from ..matplotlib import FigureCanvasQTAgg
import matplotlib
from matplotlib.container import Container
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, LineCollection

from ..matplotlib.ModestImage import ModestImage
from . import BackendBase
from .._utils import FLOAT32_MINPOS


def _preserveAxisLimits(method):
    """If method may change axes limits (e.g. call self.ax.cla()),
    this wrapper ensures the previous limits are restored.
    """
    def inner(self, *args, **kwargs):
        xmin, xmax = self.getGraphXLimits()
        ymin_left, ymax_left = self.getGraphYLimits('left')
        ymin_right, ymax_right = self.getGraphYLimits('right')

        method(self, *args, **kwargs)

        if not self._plot.isXAxisAutoScale():
            self.setGraphXLimits(xmin, xmax)

        if not self._plot.isYAxisAutoScale():
            self.setGraphYLimits(ymin_left, ymax_left, 'left')
            self.setGraphYLimits(ymin_right, ymax_right, 'right')

    return inner


class BackendMatplotlib(BackendBase.BackendBase):
    """Base class for Matplotlib backend without a FigureCanvas.

    For interactive on screen plot, see :class:`BackendMatplotlibQt`.

    See :class:`BackendBase.BackendBase` for public API documentation.
    """

    def __init__(self, plot, parent=None):
        super(BackendMatplotlib, self).__init__(plot, parent)

        # matplotlib is handling keep aspect ratio at draw time
        # When keep aspect ratio is on, and one changes the limits and
        # ask them *before* next draw has been performed he will get the
        # limits without applying keep aspect ratio.
        # This attribute is used to ensure consistent values returned
        # when getting the limits at the expense of a replot
        self._dirtyLimits = True
        self._axesDisplayed = True

        self.fig = Figure()
        self.fig.set_facecolor("w")

        self.ax = self.fig.add_axes([.15, .15, .75, .75], label="left")
        self.ax2 = self.ax.twinx()
        self.ax2.set_label("right")

        # disable the use of offsets
        try:
            self.ax.get_yaxis().get_major_formatter().set_useOffset(False)
            self.ax.get_xaxis().get_major_formatter().set_useOffset(False)
            self.ax2.get_yaxis().get_major_formatter().set_useOffset(False)
            self.ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        except:
            _logger.warning('Cannot disabled axes offsets in %s ' \
                            % matplotlib.__version__)
        
        # critical for picking!!!!
        self.ax2.set_zorder(0)
        self.ax2.set_autoscaley_on(True)
        self.ax.set_zorder(1)
        # this works but the figure color is left
        if matplotlib.__version__[0] < '2':
            self.ax.set_axis_bgcolor('none')
        else:
            self.ax.set_facecolor('none')
        self.fig.sca(self.ax)

        self._overlays = set()
        self._background = None

        self._colormaps = {}

        self._graphCursor = tuple()
        self.matplotlibVersion = matplotlib.__version__

        self._enableAxis('right', False)

    # Add methods

    def addCurve(self, x, y, legend,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill, alpha, symbolsize):
        for parameter in (x, y, legend, color, symbol, linewidth, linestyle,
                          yaxis, z, selectable, fill, alpha, symbolsize):
            assert parameter is not None
        assert yaxis in ('left', 'right')

        if (len(color) == 4 and
                type(color[3]) in [type(1), numpy.uint8, numpy.int8]):
            color = numpy.array(color, dtype=numpy.float) / 255.

        if yaxis == "right":
            axes = self.ax2
            self._enableAxis("right", True)
        else:
            axes = self.ax

        picker = 3 if selectable else None

        artists = []  # All the artists composing the curve

        # First add errorbars if any so they are behind the curve
        if xerror is not None or yerror is not None:
            if hasattr(color, 'dtype') and len(color) == len(x):
                errorbarColor = 'k'
            else:
                errorbarColor = color

            # On Debian 7 at least, Nx1 array yerr does not seems supported
            if (isinstance(yerror, numpy.ndarray) and yerror.ndim == 2 and
                    yerror.shape[1] == 1 and len(x) != 1):
                yerror = numpy.ravel(yerror)

            errorbars = axes.errorbar(x, y, label=legend,
                                      xerr=xerror, yerr=yerror,
                                      linestyle=' ', color=errorbarColor)
            artists += list(errorbars.get_children())

        if hasattr(color, 'dtype') and len(color) == len(x):
            # scatter plot
            if color.dtype not in [numpy.float32, numpy.float]:
                actualColor = color / 255.
            else:
                actualColor = color

            if linestyle not in ["", " ", None]:
                # scatter plot with an actual line ...
                # we need to assign a color ...
                curveList = axes.plot(x, y, label=legend,
                                      linestyle=linestyle,
                                      color=actualColor[0],
                                      linewidth=linewidth,
                                      picker=picker,
                                      marker=None)
                artists += list(curveList)

            scatter = axes.scatter(x, y,
                                   label=legend,
                                   color=actualColor,
                                   marker=symbol,
                                   picker=picker,
                                   s=symbolsize)
            artists.append(scatter)

            if fill:
                artists.append(axes.fill_between(
                    x, FLOAT32_MINPOS, y, facecolor=actualColor[0], linestyle=''))

        else:  # Curve
            curveList = axes.plot(x, y,
                                  label=legend,
                                  linestyle=linestyle,
                                  color=color,
                                  linewidth=linewidth,
                                  marker=symbol,
                                  picker=picker,
                                  markersize=symbolsize)
            artists += list(curveList)

            if fill:
                artists.append(
                    axes.fill_between(x, FLOAT32_MINPOS, y, facecolor=color))

        for artist in artists:
            artist.set_zorder(z)
            if alpha < 1:
                artist.set_alpha(alpha)

        return Container(artists)

    def addImage(self, data, legend,
                 origin, scale, z,
                 selectable, draggable,
                 colormap, alpha):
        # Non-uniform image
        # http://wiki.scipy.org/Cookbook/Histograms
        # Non-linear axes
        # http://stackoverflow.com/questions/11488800/non-linear-axes-for-imshow-in-matplotlib
        for parameter in (data, legend, origin, scale, z,
                          selectable, draggable):
            assert parameter is not None

        origin = float(origin[0]), float(origin[1])
        scale = float(scale[0]), float(scale[1])
        height, width = data.shape[0:2]

        picker = (selectable or draggable)

        # Debian 7 specific support
        # No transparent colormap with matplotlib < 1.2.0
        # Add support for transparent colormap for uint8 data with
        # colormap with 256 colors, linear norm, [0, 255] range
        if matplotlib.__version__ < '1.2.0':
            if (len(data.shape) == 2 and colormap.getName() is None and
                    colormap.getColormapLUT() is not None):
                colors = colormap.getColormapLUT()
                if (colors.shape[-1] == 4 and
                        not numpy.all(numpy.equal(colors[3], 255))):
                    # This is a transparent colormap
                    if (colors.shape == (256, 4) and
                            colormap.getNormalization() == 'linear' and
                            not colormap.isAutoscale() and
                            colormap.getVMin() == 0 and
                            colormap.getVMax() == 255 and
                            data.dtype == numpy.uint8):
                        # Supported case, convert data to RGBA
                        data = colors[data.reshape(-1)].reshape(
                            data.shape + (4,))
                    else:
                        _logger.warning(
                            'matplotlib %s does not support transparent '
                            'colormap.', matplotlib.__version__)

        if ((height * width) > 5.0e5 and
                origin == (0., 0.) and scale == (1., 1.)):
            imageClass = ModestImage
        else:
            imageClass = AxesImage

        # the normalization can be a source of time waste
        # Two possibilities, we receive data or a ready to show image
        if len(data.shape) == 3:  # RGBA image
            image = imageClass(self.ax,
                               label="__IMAGE__" + legend,
                               interpolation='nearest',
                               picker=picker,
                               zorder=z,
                               origin='lower')

        else:
            # Convert colormap argument to matplotlib colormap
            scalarMappable = MPLColormap.getScalarMappable(colormap, data)

            # try as data
            image = imageClass(self.ax,
                               label="__IMAGE__" + legend,
                               interpolation='nearest',
                               cmap=scalarMappable.cmap,
                               picker=picker,
                               zorder=z,
                               norm=scalarMappable.norm,
                               origin='lower')
        if alpha < 1:
            image.set_alpha(alpha)

        # Set image extent
        xmin = origin[0]
        xmax = xmin + scale[0] * width
        if scale[0] < 0.:
            xmin, xmax = xmax, xmin

        ymin = origin[1]
        ymax = ymin + scale[1] * height
        if scale[1] < 0.:
            ymin, ymax = ymax, ymin

        image.set_extent((xmin, xmax, ymin, ymax))

        # Set image data
        if scale[0] < 0. or scale[1] < 0.:
            # For negative scale, step by -1
            xstep = 1 if scale[0] >= 0. else -1
            ystep = 1 if scale[1] >= 0. else -1
            data = data[::ystep, ::xstep]

        image.set_data(data)

        self.ax.add_artist(image)

        return image

    def addItem(self, x, y, legend, shape, color, fill, overlay, z):
        xView = numpy.array(x, copy=False)
        yView = numpy.array(y, copy=False)

        if shape == "line":
            item = self.ax.plot(x, y, label=legend, color=color,
                                linestyle='-', marker=None)[0]

        elif shape == "hline":
            if hasattr(y, "__len__"):
                y = y[-1]
            item = self.ax.axhline(y, label=legend, color=color)

        elif shape == "vline":
            if hasattr(x, "__len__"):
                x = x[-1]
            item = self.ax.axvline(x, label=legend, color=color)

        elif shape == 'rectangle':
            xMin = numpy.nanmin(xView)
            xMax = numpy.nanmax(xView)
            yMin = numpy.nanmin(yView)
            yMax = numpy.nanmax(yView)
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

        elif shape in ('polygon', 'polylines'):
            xView = xView.reshape(1, -1)
            yView = yView.reshape(1, -1)
            item = Polygon(numpy.vstack((xView, yView)).T,
                           closed=(shape == 'polygon'),
                           fill=False,
                           label=legend,
                           color=color)
            if fill and shape == 'polygon':
                item.set_hatch('/')

            self.ax.add_patch(item)

        else:
            raise NotImplementedError("Unsupported item shape %s" % shape)

        item.set_zorder(z)

        if overlay:
            item.set_animated(True)
            self._overlays.add(item)

        return item

    def addMarker(self, x, y, legend, text, color,
                  selectable, draggable,
                  symbol, constraint, overlay):
        legend = "__MARKER__" + legend

        if x is not None and y is not None:
            line = self.ax.plot(x, y, label=legend,
                                linestyle=" ",
                                color=color,
                                marker=symbol,
                                markersize=10.)[-1]

            if text is not None:
                xtmp, ytmp = self.ax.transData.transform_point((x, y))
                inv = self.ax.transData.inverted()
                xtmp, ytmp = inv.transform_point((xtmp, ytmp))

                if symbol is None:
                    valign = 'baseline'
                else:
                    valign = 'top'
                    text = "  " + text

                line._infoText = self.ax.text(x, ytmp, text,
                                              color=color,
                                              horizontalalignment='left',
                                              verticalalignment=valign)

        elif x is not None:
            line = self.ax.axvline(x, label=legend, color=color)
            if text is not None:
                text = " " + text
                ymin, ymax = self.getGraphYLimits(axis='left')
                delta = abs(ymax - ymin)
                if ymin > ymax:
                    ymax = ymin
                ymax -= 0.005 * delta
                line._infoText = self.ax.text(x, ymax, text,
                                              color=color,
                                              horizontalalignment='left',
                                              verticalalignment='top')

        elif y is not None:
            line = self.ax.axhline(y, label=legend, color=color)

            if text is not None:
                text = " " + text
                xmin, xmax = self.getGraphXLimits()
                delta = abs(xmax - xmin)
                if xmin > xmax:
                    xmax = xmin
                xmax -= 0.005 * delta
                line._infoText = self.ax.text(xmax, y, text,
                                              color=color,
                                              horizontalalignment='right',
                                              verticalalignment='top')

        else:
            raise RuntimeError('A marker must at least have one coordinate')

        if selectable or draggable:
            line.set_picker(5)

        if overlay:
            line.set_animated(True)
            self._overlays.add(line)

        return line

    # Remove methods

    def remove(self, item):
        # Warning: It also needs to remove extra stuff if added as for markers
        if hasattr(item, "_infoText"):  # For markers text
            item._infoText.remove()
            item._infoText = None
        self._overlays.discard(item)
        try:
            item.remove()
        except ValueError:
            pass  # Already removed e.g., in set[X|Y]AxisLogarithmic

    # Interaction methods

    def setGraphCursor(self, flag, color, linewidth, linestyle):
        if flag:
            lineh = self.ax.axhline(
                self.ax.get_ybound()[0], visible=False, color=color,
                linewidth=linewidth, linestyle=linestyle)
            lineh.set_animated(True)

            linev = self.ax.axvline(
                self.ax.get_xbound()[0], visible=False, color=color,
                linewidth=linewidth, linestyle=linestyle)
            linev.set_animated(True)

            self._graphCursor = lineh, linev
        else:
            if self._graphCursor is not None:
                lineh, linev = self._graphCursor
                lineh.remove()
                linev.remove()
                self._graphCursor = tuple()

    # Active curve

    def setCurveColor(self, curve, color):
        # Store Line2D and PathCollection
        for artist in curve.get_children():
            if isinstance(artist, (Line2D, LineCollection)):
                artist.set_color(color)
            elif isinstance(artist, PathCollection):
                artist.set_facecolors(color)
                artist.set_edgecolors(color)
            else:
                _logger.warning(
                    'setActiveCurve ignoring artist %s', str(artist))

    # Misc.

    def getWidgetHandle(self):
        return self.fig.canvas

    def _enableAxis(self, axis, flag=True):
        """Show/hide Y axis

        :param str axis: Axis name: 'left' or 'right'
        :param bool flag: Default, True
        """
        assert axis in ('right', 'left')
        axes = self.ax2 if axis == 'right' else self.ax
        axes.get_yaxis().set_visible(flag)

    def replot(self):
        """Do not perform rendering.

        Override in subclass to actually draw something.
        """
        # TODO images, markers? scatter plot? move in remove?
        # Right Y axis only support curve for now
        # Hide right Y axis if no line is present
        self._dirtyLimits = False
        if not self.ax2.lines:
            self._enableAxis('right', False)

    def saveGraph(self, fileName, fileFormat, dpi):
        # fileName can be also a StringIO or file instance
        if dpi is not None:
            self.fig.savefig(fileName, format=fileFormat, dpi=dpi)
        else:
            self.fig.savefig(fileName, format=fileFormat)
        self._plot._setDirtyPlot()

    # Graph labels

    def setGraphTitle(self, title):
        self.ax.set_title(title)

    def setGraphXLabel(self, label):
        self.ax.set_xlabel(label)

    def setGraphYLabel(self, label, axis):
        axes = self.ax if axis == 'left' else self.ax2
        axes.set_ylabel(label)

    # Graph limits

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        # Let matplotlib taking care of keep aspect ratio if any
        self._dirtyLimits = True
        self.ax.set_xlim(min(xmin, xmax), max(xmin, xmax))

        if y2min is not None and y2max is not None:
            if not self.isYAxisInverted():
                self.ax2.set_ylim(min(y2min, y2max), max(y2min, y2max))
            else:
                self.ax2.set_ylim(max(y2min, y2max), min(y2min, y2max))

        if not self.isYAxisInverted():
            self.ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
        else:
            self.ax.set_ylim(max(ymin, ymax), min(ymin, ymax))

    def getGraphXLimits(self):
        if self._dirtyLimits and self.isKeepDataAspectRatio():
            self.replot()  # makes sure we get the right limits
        return self.ax.get_xbound()

    def setGraphXLimits(self, xmin, xmax):
        self._dirtyLimits = True
        self.ax.set_xlim(min(xmin, xmax), max(xmin, xmax))

    def getGraphYLimits(self, axis):
        assert axis in ('left', 'right')
        ax = self.ax2 if axis == 'right' else self.ax

        if not ax.get_visible():
            return None

        if self._dirtyLimits and self.isKeepDataAspectRatio():
            self.replot()  # makes sure we get the right limits

        return ax.get_ybound()

    def setGraphYLimits(self, ymin, ymax, axis):
        ax = self.ax2 if axis == 'right' else self.ax
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        self._dirtyLimits = True

        if self.isKeepDataAspectRatio():
            # matplotlib keeps limits of shared axis when keeping aspect ratio
            # So x limits are kept when changing y limits....
            # Change x limits first by taking into account aspect ratio
            # and then change y limits.. so matplotlib does not need
            # to make change (to y) to keep aspect ratio
            xmin, xmax = ax.get_xbound()
            curYMin, curYMax = ax.get_ybound()

            newXRange = (xmax - xmin) * (ymax - ymin) / (curYMax - curYMin)
            xcenter = 0.5 * (xmin + xmax)
            ax.set_xlim(xcenter - 0.5 * newXRange, xcenter + 0.5 * newXRange)

        if not self.isYAxisInverted():
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(ymax, ymin)

    # Graph axes

    @_preserveAxisLimits
    def setXAxisLogarithmic(self, flag):
        if matplotlib.__version__ >= "2.0.0":
            self.ax.cla()
            self.ax2.cla()
        self.ax2.set_xscale('log' if flag else 'linear')
        self.ax.set_xscale('log' if flag else 'linear')

        # Make sur Y axis scale is set (it is reset with mpl 2.1.0)
        isYLog = self._plot.isYAxisLogarithmic()
        self.ax2.set_yscale('log' if isYLog else 'linear')
        self.ax.set_yscale('log' if isYLog else 'linear')

    @_preserveAxisLimits
    def setYAxisLogarithmic(self, flag):
        if matplotlib.__version__ >= "2.0.0":
            self.ax.cla()
            self.ax2.cla()
        self.ax2.set_yscale('log' if flag else 'linear')
        self.ax.set_yscale('log' if flag else 'linear')

        # Make sur X axis scale is set (it is reset with mpl 2.1.0)
        isXLog = self._plot.isXAxisLogarithmic()
        self.ax2.set_xscale('log' if isXLog else 'linear')
        self.ax.set_xscale('log' if isXLog else 'linear')

    def setYAxisInverted(self, flag):
        if self.ax.yaxis_inverted() != bool(flag):
            self.ax.invert_yaxis()

    def isYAxisInverted(self):
        return self.ax.yaxis_inverted()

    def isKeepDataAspectRatio(self):
        return self.ax.get_aspect() in (1.0, 'equal')

    def setKeepDataAspectRatio(self, flag):
        self.ax.set_aspect(1.0 if flag else 'auto')
        self.ax2.set_aspect(1.0 if flag else 'auto')

    def setGraphGrid(self, which):
        self.ax.grid(False, which='both')  # Disable all grid first
        if which is not None:
            self.ax.grid(True, which=which)

    # Data <-> Pixel coordinates conversion

    def dataToPixel(self, x, y, axis):
        ax = self.ax2 if axis == "right" else self.ax

        pixels = ax.transData.transform_point((x, y))
        xPixel, yPixel = pixels.T
        return xPixel, yPixel

    def pixelToData(self, x, y, axis, check):
        ax = self.ax2 if axis == "right" else self.ax

        inv = ax.transData.inverted()
        x, y = inv.transform_point((x, y))

        if check:
            xmin, xmax = self.getGraphXLimits()
            ymin, ymax = self.getGraphYLimits(axis=axis)

            if x > xmax or x < xmin or y > ymax or y < ymin:
                return None  # (x, y) is out of plot area

        return x, y

    def getPlotBoundsInPixels(self):
        bbox = self.ax.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted())
        dpi = self.fig.dpi
        # Warning this is not returning int...
        return (bbox.bounds[0] * dpi, bbox.bounds[1] * dpi,
                bbox.bounds[2] * dpi, bbox.bounds[3] * dpi)

    def setAxesDisplayed(self, displayed):
        """Display or not the axes.

        :param bool displayed: If `True` axes are displayed. If `False` axes
            are not anymore visible and the margin used for them is removed.
        """
        BackendBase.BackendBase.setAxesDisplayed(self, displayed)
        if displayed:
            # show axes and viewbox rect
            self.ax.set_axis_on()
            self.ax2.set_axis_on()
            # set the default margins
            self.ax.set_position([.15, .15, .75, .75])
            self.ax2.set_position([.15, .15, .75, .75])
        else:
            # hide axes and viewbox rect
            self.ax.set_axis_off()
            self.ax2.set_axis_off()
            # remove external margins
            self.ax.set_position([0, 0, 1, 1])
            self.ax2.set_position([0, 0, 1, 1])
        self._plot._setDirtyPlot()


class BackendMatplotlibQt(FigureCanvasQTAgg, BackendMatplotlib):
    """QWidget matplotlib backend using a QtAgg canvas.

    It adds fast overlay drawing and mouse event management.
    """

    _sigPostRedisplay = qt.Signal()
    """Signal handling automatic asynchronous replot"""

    def __init__(self, plot, parent=None):
        BackendMatplotlib.__init__(self, plot, parent)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(
            self, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        # Make postRedisplay asynchronous using Qt signal
        self._sigPostRedisplay.connect(
            super(BackendMatplotlibQt, self).postRedisplay,
            qt.Qt.QueuedConnection)

        self._picked = None

        self.mpl_connect('button_press_event', self._onMousePress)
        self.mpl_connect('button_release_event', self._onMouseRelease)
        self.mpl_connect('motion_notify_event', self._onMouseMove)
        self.mpl_connect('scroll_event', self._onMouseWheel)

    def contextMenuEvent(self, event):
        """Override QWidget.contextMenuEvent to implement the context menu"""
        # Makes sure it is overridden (issue with PySide)
        BackendBase.BackendBase.contextMenuEvent(self, event)

    def postRedisplay(self):
        self._sigPostRedisplay.emit()

    # Mouse event forwarding

    _MPL_TO_PLOT_BUTTONS = {1: 'left', 2: 'middle', 3: 'right'}

    def _onMousePress(self, event):
        self._plot.onMousePress(
            event.x, event.y, self._MPL_TO_PLOT_BUTTONS[event.button])

    def _onMouseMove(self, event):
        if self._graphCursor:
            lineh, linev = self._graphCursor
            if event.inaxes != self.ax and lineh.get_visible():
                lineh.set_visible(False)
                linev.set_visible(False)
                self._plot._setDirtyPlot(overlayOnly=True)
            else:
                linev.set_visible(True)
                linev.set_xdata((event.xdata, event.xdata))
                lineh.set_visible(True)
                lineh.set_ydata((event.ydata, event.ydata))
                self._plot._setDirtyPlot(overlayOnly=True)
            # onMouseMove must trigger replot if dirty flag is raised

        self._plot.onMouseMove(event.x, event.y)

    def _onMouseRelease(self, event):
        self._plot.onMouseRelease(
            event.x, event.y, self._MPL_TO_PLOT_BUTTONS[event.button])

    def _onMouseWheel(self, event):
        self._plot.onMouseWheel(event.x, event.y, event.step)

    def leaveEvent(self, event):
        """QWidget event handler"""
        self._plot.onMouseLeaveWidget()

    # picking

    def _onPick(self, event):
        # TODO not very nice and fragile, find a better way?
        # Make a selection according to kind
        if self._picked is None:
            _logger.error('Internal picking error')
            return

        label = event.artist.get_label()
        if label.startswith('__MARKER__'):
            self._picked.append({'kind': 'marker', 'legend': label[10:]})

        elif label.startswith('__IMAGE__'):
            self._picked.append({'kind': 'image', 'legend': label[9:]})

        else:  # it's a curve, item have no picker for now
            if not isinstance(event.artist, (PathCollection, Line2D)):
                _logger.info('Unsupported artist, ignored')
                return

            self._picked.append({'kind': 'curve', 'legend': label,
                                 'indices': event.ind})

    def pickItems(self, x, y):
        self._picked = []

        # Weird way to do an explicit picking: Simulate a button press event
        mouseEvent = MouseEvent('button_press_event', self, x, y)
        cid = self.mpl_connect('pick_event', self._onPick)
        self.fig.pick(mouseEvent)
        self.mpl_disconnect(cid)
        picked = self._picked
        self._picked = None

        return picked

    # replot control

    def resizeEvent(self, event):
        FigureCanvasQTAgg.resizeEvent(self, event)
        if self._overlays or self._graphCursor:
            # This is needed with matplotlib 1.5.x and 2.0.x
            self._plot._setDirtyPlot()

    def _drawOverlays(self):
        """Draw overlays if any."""
        if self._overlays or self._graphCursor:
            # There is some overlays or crosshair

            # This assume that items are only on left/bottom Axes
            for item in self._overlays:
                self.ax.draw_artist(item)

            for item in self._graphCursor:
                self.ax.draw_artist(item)

    def draw(self):
        """Overload draw

        It performs a full redraw (including overlays) of the plot.
        It also resets background and emit limits changed signal.

        This is directly called by matplotlib for widget resize.
        """
        # Store previous limits
        xLimits = self.ax.get_xbound()
        yLimits = self.ax.get_ybound()
        yRightLimits = self.ax2.get_ybound()

        FigureCanvasQTAgg.draw(self)
        if self._overlays or self._graphCursor:
            # Save background
            self._background = self.copy_from_bbox(self.fig.bbox)
        else:
            self._background = None  # Reset background

        # Check if limits changed due to a resize of the widget
        if xLimits != self.ax.get_xbound():
            self._plot.getXAxis()._emitLimitsChanged()
        if yLimits != self.ax.get_ybound():
            self._plot.getYAxis(axis='left')._emitLimitsChanged()
        if yRightLimits != self.ax2.get_ybound():
            self._plot.getYAxis(axis='left')._emitLimitsChanged()

        self._drawOverlays()

    def replot(self):
        BackendMatplotlib.replot(self)

        dirtyFlag = self._plot._getDirtyPlot()

        if dirtyFlag == 'overlay':
            # Only redraw overlays using fast rendering path
            if self._background is None:
                self._background = self.copy_from_bbox(self.fig.bbox)
            self.restore_region(self._background)
            self._drawOverlays()
            self.blit(self.fig.bbox)

        elif dirtyFlag:  # Need full redraw
            self.draw()


    # cursor

    _QT_CURSORS = {
        None: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        BackendBase.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        BackendBase.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        BackendBase.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setGraphCursorShape(self, cursor):
        cursor = self._QT_CURSORS[cursor]

        FigureCanvasQTAgg.setCursor(self, qt.QCursor(cursor))
