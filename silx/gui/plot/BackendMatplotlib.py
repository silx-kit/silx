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

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "18/01/2017"


import logging

import numpy


_logger = logging.getLogger(__name__)


from .. import qt

from ._matplotlib import FigureCanvasQTAgg
import matplotlib
from matplotlib import cm
from matplotlib.container import Container
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, LineCollection

from . import _utils
from .ModestImage import ModestImage
from . import BackendBase
from . import Colors


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

        self.fig = Figure()
        self.fig.set_facecolor("w")

        self.ax = self.fig.add_axes([.15, .15, .75, .75], label="left")
        self.ax2 = self.ax.twinx()
        self.ax2.set_label("right")

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

        self.setGraphXLimits(0., 100.)
        self.setGraphYLimits(0., 100., axis='right')
        self.setGraphYLimits(0., 100., axis='left')

        self._enableAxis('right', False)

    # Add methods

    def addCurve(self, x, y, legend,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill):
        for parameter in (x, y, legend, color, symbol, linewidth, linestyle,
                          yaxis, z, selectable, fill):
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
            if (yerror is not None and yerror.ndim == 2 and
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
                                   picker=picker)
            artists.append(scatter)

            if fill:
                artists.append(axes.fill_between(
                    x, 1.0e-8, y, facecolor=actualColor[0], linestyle=''))

        else:  # Curve
            curveList = axes.plot(x, y,
                                  label=legend,
                                  linestyle=linestyle,
                                  color=color,
                                  linewidth=linewidth,
                                  marker=symbol,
                                  picker=picker)
            artists += list(curveList)

            if fill:
                artists.append(
                    axes.fill_between(x, 1.0e-8, y,
                                      facecolor=color, linewidth=0))

        for artist in artists:
            artist.set_zorder(z)

        return Container(artists)

    def addImage(self, data, legend,
                 origin, scale, z,
                 selectable, draggable,
                 colormap):
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
            if (len(data.shape) == 2 and colormap['name'] is None and
                    'colors' in colormap):
                colors = numpy.array(colormap['colors'], copy=False)
                if (colors.shape[-1] == 4 and
                        not numpy.all(numpy.equal(colors[3], 255))):
                    # This is a transparent colormap
                    if (colors.shape == (256, 4) and
                            colormap['normalization'] == 'linear' and
                            not colormap['autoscale'] and
                            colormap['vmin'] == 0 and
                            colormap['vmax'] == 255 and
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
            scalarMappable = Colors.getMPLScalarMappable(colormap, data)

            # try as data
            image = imageClass(self.ax,
                               label="__IMAGE__" + legend,
                               interpolation='nearest',
                               cmap=scalarMappable.cmap,
                               picker=picker,
                               zorder=z,
                               norm=scalarMappable.norm,
                               origin='lower')

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
        item.remove()

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

    def setActiveCurve(self, curve, active, color=None):
        # Store Line2D and PathCollection
        for artist in curve.get_children():
            if active:
                if isinstance(artist, (Line2D, LineCollection)):
                    artist._initialColor = artist.get_color()
                    artist.set_color(color)
                elif isinstance(artist, PathCollection):
                    artist._initialColor = artist.get_facecolors()
                    artist.set_facecolors(color)
                    artist.set_edgecolors(color)
                else:
                    _logger.warning(
                        'setActiveCurve ignoring artist %s', str(artist))
            else:
                if hasattr(artist, '_initialColor'):
                    if isinstance(artist, (Line2D, LineCollection)):
                        artist.set_color(artist._initialColor)
                    elif isinstance(artist, PathCollection):
                        artist.set_facecolors(artist._initialColor)
                        artist.set_edgecolors(artist._initialColor)
                    else:
                        _logger.info(
                            'setActiveCurve ignoring artist %s', str(artist))
                    del artist._initialColor

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

    def resetZoom(self, dataMargins):
        xAuto = self._plot.isXAxisAutoScale()
        yAuto = self._plot.isYAxisAutoScale()

        if not xAuto and not yAuto:
            _logger.debug("Nothing to autoscale")
        else:  # Some axes to autoscale
            xLimits = self.getGraphXLimits()
            yLimits = self.getGraphYLimits(axis='left')
            y2Limits = self.getGraphYLimits(axis='right')

            # Get data range
            ranges = self._plot.getDataRange()
            xmin, xmax = (1., 100.) if ranges.x is None else ranges.x
            ymin, ymax = (1., 100.) if ranges.y is None else ranges.y
            if ranges.yright is None:
                ymin2, ymax2 = None, None
            else:
                ymin2, ymax2 = ranges.yright

            # Add margins around data inside the plot area
            newLimits = list(_utils.addMarginsToLimits(
                dataMargins,
                self.ax.get_xscale() == 'log',
                self.ax.get_yscale() == 'log',
                xmin, xmax, ymin, ymax, ymin2, ymax2))

            if self.isKeepDataAspectRatio():
                # Compute bbox wth figure aspect ratio
                figW, figH = self.fig.get_size_inches()
                figureRatio = figH / figW

                dataRatio = (ymax - ymin) / (xmax - xmin)
                if dataRatio < figureRatio:
                    # Increase y range
                    ycenter = 0.5 * (newLimits[3] + newLimits[2])
                    yrange = (xmax - xmin) * figureRatio
                    newLimits[2] = ycenter - 0.5 * yrange
                    newLimits[3] = ycenter + 0.5 * yrange

                elif dataRatio > figureRatio:
                    # Increase x range
                    xcenter = 0.5 * (newLimits[1] + newLimits[0])
                    xrange_ = (ymax - ymin) / figureRatio
                    newLimits[0] = xcenter - 0.5 * xrange_
                    newLimits[1] = xcenter + 0.5 * xrange_

            self.setLimits(*newLimits)

            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                if y2Limits is not None:
                    self.setGraphYLimits(
                        y2Limits[0], y2Limits[1], axis='right')
                if yLimits is not None:
                    self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')

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

    def setXAxisLogarithmic(self, flag):
        self.ax2.set_xscale('log' if flag else 'linear')
        self.ax.set_xscale('log' if flag else 'linear')

    def setYAxisLogarithmic(self, flag):
        self.ax2.set_yscale('log' if flag else 'linear')
        self.ax.set_yscale('log' if flag else 'linear')

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

    # colormap

    def getSupportedColormaps(self):
        default = super(BackendMatplotlib, self).getSupportedColormaps()
        maps = [m for m in cm.datad]
        maps.sort()
        return default + tuple(maps)

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


class BackendMatplotlibQt(FigureCanvasQTAgg, BackendMatplotlib):
    """QWidget matplotlib backend using a QtAgg canvas.

    It adds fast overlay drawing and mouse event management.
    """

    _sigPostRedisplay = qt.Signal()
    """Signal handling automatic asynchronous replot"""

    def __init__(self, plot, parent=None):
        self._insideResizeEventMethod = False

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

        self.mpl_connect('button_press_event', self._onMousePress)
        self.mpl_connect('button_release_event', self._onMouseRelease)
        self.mpl_connect('motion_notify_event', self._onMouseMove)
        self.mpl_connect('scroll_event', self._onMouseWheel)

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
        label = event.artist.get_label()
        if label.startswith('__MARKER__'):
            self._picked.append({'kind': 'marker', 'legend': label[10:]})

        elif label.startswith('__IMAGE__'):
            self._picked.append({'kind': 'image', 'legend': label[9:]})

        else:  # it's a curve, item have no picker for now
            if isinstance(event.artist, PathCollection):
                data = event.artist.get_offsets()[event.ind, :]
                xdata, ydata = data[:, 0], data[:, 1]
            elif isinstance(event.artist, Line2D):
                xdata = event.artist.get_xdata()[event.ind]
                ydata = event.artist.get_ydata()[event.ind]
            else:
                _logger.info('Unsupported artist, ignored')
                return

            self._picked.append({'kind': 'curve', 'legend': label,
                                 'xdata': xdata, 'ydata': ydata})

    def pickItems(self, x, y):
        self._picked = []

        # Weird way to do an explicit picking: Simulate a button press event
        mouseEvent = MouseEvent('button_press_event', self, x, y)
        cid = self.mpl_connect('pick_event', self._onPick)
        self.fig.pick(mouseEvent)
        self.mpl_disconnect(cid)
        picked = self._picked
        del self._picked

        return picked

    # replot control

    def resizeEvent(self, event):
        self._insideResizeEventMethod = True
        # Need to dirty the whole plot on resize.
        self._plot._setDirtyPlot()
        FigureCanvasQTAgg.resizeEvent(self, event)
        self._insideResizeEventMethod = False

    def draw(self):
        """Override canvas draw method to support faster draw of overlays."""
        if self._plot._getDirtyPlot():  # Need a full redraw
            FigureCanvasQTAgg.draw(self)
            self._background = None  # Any saved background is dirty

        if (self._overlays or self._graphCursor or
                self._plot._getDirtyPlot() == 'overlay'):
            # There are overlays or crosshair, or they is just no more overlays

            # Specific case: called from resizeEvent:
            # avoid store/restore background, just draw the overlay
            if not self._insideResizeEventMethod:
                if self._background is None:  # First store the background
                    self._background = self.copy_from_bbox(self.fig.bbox)

                self.restore_region(self._background)

            # This assume that items are only on left/bottom Axes
            for item in self._overlays:
                self.ax.draw_artist(item)

            for item in self._graphCursor:
                self.ax.draw_artist(item)

            self.blit(self.fig.bbox)

    def replot(self):
        BackendMatplotlib.replot(self)
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
