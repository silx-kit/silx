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
# ###########################################################################*/
"""Matplotlib Plot backend."""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "18/02/2016"


import logging
import sys

import numpy


logging.basicConfig()
_logger = logging.getLogger(__name__)

if 'matplotlib' in sys.modules:
    _logger.warning(
        'matplotlib already loaded, setting its backend may not work')


from .. import qt

import matplotlib

if qt.BINDING == 'PySide':
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    matplotlib.rcParams['backend.qt4'] = 'PySide'
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

elif qt.BINDING == 'PyQt4':
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

elif qt.BINDING == 'PyQt5':
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.image import AxesImage
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

from . import _utils
from .ModestImage import ModestImage
from . import BackendBase


class BackendMatplotlib(BackendBase.BackendBase):
    """Base class for Matplotlib backend without a FigureCanvas.

    For interactive on screen plot, see :class:`BackendMatplotlibQt`.

    See :class:`BackendBase.BackendBase` for public API documentation.
    """

    def __init__(self, plot, parent=None):
        super(BackendMatplotlib, self).__init__(plot, parent)

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
        self.ax.set_axis_bgcolor('none')
        self.fig.sca(self.ax)

        self._overlays = set()
        self._background = None

        self._colormaps = {}

        self._graphCursor = tuple()
        self.matplotlibVersion = matplotlib.__version__

        self.setGraphXLimits(0., 100.)
        self.setGraphYLimits(0., 100., axis='left')
        self.setGraphYLimits(0., 100., axis='right')

        self._enableAxis('right', False)

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
            'xmin': numpy.nanmin(x),
            'xmax': numpy.nanmax(x),
            'ymin': numpy.nanmin(y),
            'ymax': numpy.nanmax(y),
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
                        _logger.warning('Log colormap with negative bounds, ' +
                                        'changing bounds to positive ones.')
                    elif vmin > vmax:
                        _logger.warning('Colormap bounds are inverted.')
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
                        _logger.warning('Colormap bounds are inverted.')
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

        return image

    def addItem(self, x, y, legend, shape, color, fill, overlay):
        xView = numpy.array(x, copy=False)
        yView = numpy.array(y, copy=False)

        if shape == "hline":
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

        elif shape == 'polygon':
            xView.shape = 1, -1
            yView.shape = 1, -1
            item = Polygon(numpy.vstack((xView, yView)).T,
                           closed=True,
                           fill=False,
                           label=legend,
                           color=color)
            if fill:
                item.set_hatch('/')

            self.ax.add_patch(item)

        else:
            raise NotImplementedError("Unsupported item shape %s" % shape)

        if overlay:
            item.set_animated(True)
            self._overlays.add(item)

        return item

    def addMarker(self, x, y, legend, text, color,
                  selectable, draggable,
                  symbol, constraint, overlay):
        legend = "__MARKER__" + legend  # TODO useful?

        # Apply constraint to provided position
        if draggable and constraint is not None:
            x, y = constraint(x, y)

        # TODO issues with text placement when changing limits..
        if x is not None and y is not None:
            line = self.ax.plot(x, y, label=legend,
                                linestyle=" ",
                                color=color,
                                marker=symbol,
                                markersize=10.)[-1]

            if text is not None:
                xtmp, ytmp = self.ax.transData.transform((x, y))
                inv = self.ax.transData.inverted()
                xtmp, ytmp = inv.transform((xtmp, ytmp + 15))
                text = " " + text
                line._infoText = self.ax.text(x, ytmp, text,
                                              color=color,
                                              horizontalalignment='left',
                                              verticalalignment='top')

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

    def clear(self):
        self.ax.clear()
        self.ax2.clear()

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

    def setCurveColor(self, curve, color):
        curve.set_color(color)

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
        if not self.ax2.lines:
            self._enableAxis('right', False)

    def saveGraph(self, fileName, fileFormat, dpi):
        # fileName can be also a StringIO or file instance
        if dpi is not None:
            self.fig.savefig(fileName, format=fileFormat, dpi=dpi)
        else:
            self.fig.savefig(fileName, format=fileFormat)

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

            xmin, xmax, ymin, ymax = self._getDataLimits('left')
            if hasattr(self.ax2, "get_visible"):
                if self.ax2.get_visible():
                    xmin2, xmax2, ymin2, ymax2 = self._getDataLimits('right')
                else:
                    xmin2 = None
                    xmax2 = None
            else:
                xmin2, xmax2, ymin2, ymax2 = self._getDataLimits('right')

            if (xmin2 is not None) and ((xmin2 != 0) or (xmax2 != 1)):
                xmin = min(xmin, xmin2)
                xmax = max(xmax, xmax2)

            # Add margins around data inside the plot area
            if xmin2 is None:
                newLimits = _utils.addMarginsToLimits(
                    dataMargins,
                    self.ax.get_xscale() == 'log',
                    self.ax.get_yscale() == 'log',
                    xmin, xmax, ymin, ymax)

            else:
                newLimits = _utils.addMarginsToLimits(
                    dataMargins,
                    self.ax.get_xscale() == 'log',
                    self.ax.get_yscale() == 'log',
                    xmin, xmax, ymin, ymax, ymin2, ymax2)

            self.setLimits(*newLimits)

            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')
                self.setGraphYLimits(y2Limits[0], y2Limits[1], axis='right')

    def _getDataLimits(self, axesLabel='left'):
        """Returns the bounds of the data.

        :param str axesLabel: The Y axis to consider in 'left', 'right'
        :return: The data bounds
        :rtype: 4-tuple of float (xmin, xmax, ymin, ymax)
        """
        if axesLabel == 'right':
            axes = self.ax2
        else:
            axes = self.ax
        _logger.debug("CALCULATING limits %s", axes.get_label())
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
                lineXMin = numpy.nanmin(x)
                lineXMax = numpy.nanmax(x)
                lineYMin = numpy.nanmin(y)
                lineYMax = numpy.nanmax(y)
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
                _logger.warning("CANNOT CALCULATE LIMITS")
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
            figW, figH = self.fig.get_size_inches()
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
        _logger.debug(
            "CALCULATED LIMITS = %f %f %f %f", xmin, xmax, ymin, ymax)
        return xmin, xmax, ymin, ymax

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        self.setGraphXLimits(xmin, xmax)
        self.setGraphYLimits(ymin, ymax, axis='left')

        if y2min is not None and y2max is not None:
            self.setGraphYLimits(ymin, ymax, axis='right')

    def getGraphXLimits(self):
        vmin, vmax = self.ax.get_xlim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def setGraphXLimits(self, xmin, xmax):
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        self.ax.set_xlim(xmin, xmax)

    def getGraphYLimits(self, axis):
        assert axis in ('left', 'right')
        ax = self.ax2 if axis == 'right' else self.ax

        if not ax.get_visible():
            return None

        vmin, vmax = ax.get_ylim()
        if vmin > vmax:
            return vmax, vmin
        else:
            return vmin, vmax

    def setGraphYLimits(self, ymin, ymax, axis):
        ax = self.ax2 if axis == 'right' else self.ax
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if ax.yaxis_inverted():
            ax.set_ylim(ymax, ymin)
        else:
            ax.set_ylim(ymin, ymax)

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

    def keepDataAspectRatio(self, flag):
        self.ax.set_aspect(1.0 if flag else 'auto')

    def showGrid(self, flag):
        self.ax.grid(False, which='both')  # Disable all grid first
        if flag:
            self.ax.grid(True, which='both' if flag == 2 else 'major')

    # colormap

    def getSupportedColormaps(self):
        default = super(BackendMatplotlib, self).getSupportedColormaps()
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

    def dataToPixel(self, x, y, axis):
        ax = self.ax2 if "axis" == "right" else self.ax

        pixels = ax.transData.transform_point((x, y))
        xPixel, yPixel = pixels.T
        return xPixel, yPixel

    def pixelToData(self, x, y, axis, check):
        ax = self.ax2 if "axis" == "right" else self.ax

        inv = ax.transData.inverted()
        x, y = inv.transform_point((x, y))

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if check and (x > xmax or x < xmin or y > ymax or y < ymin):
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

    def __init__(self, plot, parent=None):
        self._insideResizeEventMethod = False

        BackendMatplotlib.__init__(self, plot, parent)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(
            self, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.mpl_connect('button_press_event', self._onMousePress)
        self.mpl_connect('button_release_event', self._onMouseRelease)
        self.mpl_connect('motion_notify_event', self._onMouseMove)
        self.mpl_connect('scroll_event', self._onMouseWheel)

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
