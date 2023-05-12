# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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

__authors__ = ["V.A. Sole", "T. Vincent, H. Payno"]
__license__ = "MIT"
__date__ = "21/12/2018"


import logging
import datetime as dt
from typing import Tuple, Union
import numpy

from pkg_resources import parse_version as _parse_version


_logger = logging.getLogger(__name__)


from ... import qt

# First of all init matplotlib and set its backend
from ...utils.matplotlib import FigureCanvasQTAgg
import matplotlib
from matplotlib.container import Container
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.ticker import Formatter, ScalarFormatter, Locator
from matplotlib.tri import Triangulation
from matplotlib.collections import TriMesh
from matplotlib import path as mpath

from . import BackendBase
from .. import items
from .._utils import FLOAT32_MINPOS
from .._utils.dtime_ticklayout import calcTicks, bestFormatString, timestamp
from ...qt import inspect as qt_inspect

_PATCH_LINESTYLE = {
    "-": 'solid',
    "--": 'dashed',
    '-.': 'dashdot',
    ':': 'dotted',
    '': "solid",
    None: "solid",
}
"""Patches do not uses the same matplotlib syntax"""

_MARKER_PATHS = {}
"""Store cached extra marker paths"""

_SPECIAL_MARKERS = {
    'tickleft': 0,
    'tickright': 1,
    'tickup': 2,
    'tickdown': 3,
    'caretleft': 4,
    'caretright': 5,
    'caretup': 6,
    'caretdown': 7,
}


def normalize_linestyle(linestyle):
    """Normalize known old-style linestyle, else return the provided value."""
    return _PATCH_LINESTYLE.get(linestyle, linestyle)

def get_path_from_symbol(symbol):
    """Get the path representation of a symbol, else None if
    it is not provided.

    :param str symbol: Symbol description used by silx
    :rtype: Union[None,matplotlib.path.Path]
    """
    if symbol == u'\u2665':
        path = _MARKER_PATHS.get(symbol, None)
        if path is not None:
            return path
        vertices = numpy.array([
            [0,-99],
            [31,-73], [47,-55], [55,-46],
            [63,-37], [94,-2], [94,33],
            [94,69], [71,89], [47,89],
            [24,89], [8,74], [0,58],
            [-8,74], [-24,89], [-47,89],
            [-71,89], [-94,69], [-94,33],
            [-94,-2], [-63,-37], [-55,-46],
            [-47,-55], [-31,-73], [0,-99],
            [0,-99]])
        codes = [mpath.Path.CURVE4] * len(vertices)
        codes[0] = mpath.Path.MOVETO
        codes[-1] = mpath.Path.CLOSEPOLY
        path = mpath.Path(vertices, codes)
        _MARKER_PATHS[symbol] = path
        return path
    return None

class NiceDateLocator(Locator):
    """
    Matplotlib Locator that uses Nice Numbers algorithm (adapted to dates)
    to find the tick locations. This results in the same number behaviour
    as when using the silx Open GL backend.

    Expects the data to be posix timestampes (i.e. seconds since 1970)
    """
    def __init__(self, numTicks=5, tz=None):
        """
        :param numTicks: target number of ticks
        :param datetime.tzinfo tz: optional time zone. None is local time.
        """
        super(NiceDateLocator, self).__init__()
        self.numTicks = numTicks

        self._spacing = None
        self._unit = None
        self.tz = tz

    @property
    def spacing(self):
        """ The current spacing. Will be updated when new tick value are made"""
        return self._spacing

    @property
    def unit(self):
        """ The current DtUnit. Will be updated when new tick value are made"""
        return self._unit

    def __call__(self):
        """Return the locations of the ticks"""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        """ Calculates tick values
        """
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # vmin and vmax should be timestamps (i.e. seconds since 1 Jan 1970)
        try:
            dtMin = dt.datetime.fromtimestamp(vmin, tz=self.tz)
            dtMax = dt.datetime.fromtimestamp(vmax, tz=self.tz)
        except ValueError:
            _logger.warning("Data range cannot be displayed with time axis")
            return []

        dtTicks, self._spacing, self._unit = \
            calcTicks(dtMin, dtMax, self.numTicks)

        # Convert datetime back to time stamps.
        ticks = [timestamp(dtTick) for dtTick in dtTicks]
        return ticks


class NiceAutoDateFormatter(Formatter):
    """
    Matplotlib FuncFormatter that is linked to a NiceDateLocator and gives the
    best possible formats given the locators current spacing an date unit.
    """

    def __init__(self, locator, tz=None):
        """
        :param niceDateLocator: a NiceDateLocator object
        :param datetime.tzinfo tz: optional time zone. None is local time.
        """
        super(NiceAutoDateFormatter, self).__init__()
        self.locator = locator
        self.tz = tz

    @property
    def formatString(self):
        if self.locator.spacing is None or self.locator.unit is None:
            # Locator has no spacing or units yet. Return elaborate fmtString
            return "Y-%m-%d %H:%M:%S"
        else:
            return bestFormatString(self.locator.spacing, self.locator.unit)

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*
           Expects x to be a POSIX timestamp (seconds since 1 Jan 1970)
        """
        dateTime = dt.datetime.fromtimestamp(x, tz=self.tz)
        tickStr = dateTime.strftime(self.formatString)
        return tickStr


class _PickableContainer(Container):
    """Artists container with a :meth:`contains` method"""

    def __init__(self, *args, **kwargs):
        Container.__init__(self, *args, **kwargs)
        self.__zorder = None

    @property
    def axes(self):
        """Mimin Artist.axes"""
        for child in self.get_children():
            if hasattr(child, 'axes'):
                return child.axes
        return None

    def draw(self, *args, **kwargs):
        """artist-like draw to broadcast draw to children"""
        for child in self.get_children():
            child.draw(*args, **kwargs)

    def get_zorder(self):
        """Mimic Artist.get_zorder"""
        return self.__zorder

    def set_zorder(self, z):
        """Mimic Artist.set_zorder to broadcast to children"""
        if z != self.__zorder:
            self.__zorder = z
            for child in self.get_children():
                child.set_zorder(z)

    def contains(self, mouseevent):
        """Mimic Artist.contains, and call it on all children.

        :param mouseevent:
        :return: Picking status and associated information as a dict
        :rtype: (bool,dict)
        """
        # Goes through children from front to back and return first picked one.
        for child in reversed(self.get_children()):
            picked, info = child.contains(mouseevent)
            if picked:
                return picked, info
        return False, {}


class _TextWithOffset(Text):
    """Text object which can be displayed at a specific position
    of the plot, but with a pixel offset"""

    def __init__(self, *args, **kwargs):
        Text.__init__(self, *args, **kwargs)
        self.pixel_offset = (0, 0)
        self.__cache = None

    def draw(self, renderer):
        self.__cache = None
        return Text.draw(self, renderer)

    def __get_xy(self):
        if self.__cache is not None:
            return self.__cache

        align = self.get_horizontalalignment()
        if align == "left":
            xoffset = self.pixel_offset[0]
        elif align == "right":
            xoffset = -self.pixel_offset[0]
        else:
            xoffset = 0

        align = self.get_verticalalignment()
        if align == "top":
            yoffset = -self.pixel_offset[1]
        elif align == "bottom":
            yoffset = self.pixel_offset[1]
        else:
            yoffset = 0

        trans = self.get_transform()
        x = super(_TextWithOffset, self).convert_xunits(self._x)
        y = super(_TextWithOffset, self).convert_xunits(self._y)
        pos = x, y

        try:
            invtrans = trans.inverted()
        except numpy.linalg.LinAlgError:
            # Cannot inverse transform, fallback: pos without offset
            self.__cache = None
            return pos

        proj = trans.transform_point(pos)
        proj = proj + numpy.array((xoffset, yoffset))
        pos = invtrans.transform_point(proj)
        self.__cache = pos
        return pos

    def convert_xunits(self, x):
        """Return the pixel position of the annotated point."""
        return self.__get_xy()[0]

    def convert_yunits(self, y):
        """Return the pixel position of the annotated point."""
        return self.__get_xy()[1]


class _MarkerContainer(_PickableContainer):
    """Marker artists container supporting draw/remove and text position update

    :param artists:
        Iterable with either one Line2D or a Line2D and a Text.
        The use of an iterable if enforced by Container being
        a subclass of tuple that defines a specific __new__.
    :param x: X coordinate of the marker (None for horizontal lines)
    :param y: Y coordinate of the marker (None for vertical lines)
    """

    def __init__(self, artists, symbol, x, y, yAxis):
        self.line = artists[0]
        self.text = artists[1] if len(artists) > 1 else None
        self.symbol = symbol
        self.x = x
        self.y = y
        self.yAxis = yAxis

        _PickableContainer.__init__(self, artists)

    def draw(self, *args, **kwargs):
        """artist-like draw to broadcast draw to line and text"""
        self.line.draw(*args, **kwargs)
        if self.text is not None:
            self.text.draw(*args, **kwargs)

    def updateMarkerText(self, xmin, xmax, ymin, ymax, yinverted):
        """Update marker text position and visibility according to plot limits

        :param xmin: X axis lower limit
        :param xmax: X axis upper limit
        :param ymin: Y axis lower limit
        :param ymax: Y axis upper limit
        :param yinverted: True if the y axis is inverted
        """
        if self.text is not None:
            visible = ((self.x is None or xmin <= self.x <= xmax) and
                       (self.y is None or ymin <= self.y <= ymax))
            self.text.set_visible(visible)

            if self.x is not None and self.y is not None:
                if self.symbol is None:
                    valign = 'baseline'
                else:
                    if yinverted:
                        valign = 'bottom'
                    else:
                        valign = 'top'
                self.text.set_verticalalignment(valign)

            elif self.y is None:  # vertical line
                # Always display it on top
                center = (ymax + ymin) * 0.5
                pos = (ymax - ymin) * 0.5 * 0.99
                if yinverted:
                    pos = -pos
                self.text.set_y(center + pos)

            elif self.x is None:  # Horizontal line
                delta = abs(xmax - xmin)
                if xmin > xmax:
                    xmax = xmin
                xmax -= 0.005 * delta
                self.text.set_x(xmax)

    def contains(self, mouseevent):
        """Mimic Artist.contains, and call it on the line Artist.

        :param mouseevent:
        :return: Picking status and associated information as a dict
        :rtype: (bool,dict)
        """
        return self.line.contains(mouseevent)


class _DoubleColoredLinePatch(matplotlib.patches.Patch):
    """Matplotlib patch to display any patch using double color."""

    def __init__(self, patch):
        super(_DoubleColoredLinePatch, self).__init__()
        self.__patch = patch
        self.linebgcolor = None

    def __getattr__(self, name):
        return getattr(self.__patch, name)

    def draw(self, renderer):
        oldLineStype = self.__patch.get_linestyle()
        if self.linebgcolor is not None and oldLineStype != "solid":
            oldLineColor = self.__patch.get_edgecolor()
            oldHatch = self.__patch.get_hatch()
            self.__patch.set_linestyle("solid")
            self.__patch.set_edgecolor(self.linebgcolor)
            self.__patch.set_hatch(None)
            self.__patch.draw(renderer)
            self.__patch.set_linestyle(oldLineStype)
            self.__patch.set_edgecolor(oldLineColor)
            self.__patch.set_hatch(oldHatch)
        self.__patch.draw(renderer)

    def set_transform(self, transform):
        self.__patch.set_transform(transform)

    def get_path(self):
        return self.__patch.get_path()

    def contains(self, mouseevent, radius=None):
        return self.__patch.contains(mouseevent, radius)

    def contains_point(self, point, radius=None):
        return self.__patch.contains_point(point, radius)


class Image(AxesImage):
    """An AxesImage with a fast path for uint8 RGBA images.

    :param List[float] silx_origin: (ox, oy) Offset of the image.
    :param List[float] silx_scale: (sx, sy) Scale of the image.
    """

    def __init__(self, *args,
                 silx_origin=(0., 0.),
                 silx_scale=(1., 1.),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.__silx_origin = silx_origin
        self.__silx_scale = silx_scale

    def contains(self, mouseevent):
        """Overridden to fill 'ind' with row and column"""
        inside, info = super().contains(mouseevent)
        if inside:
            x, y = mouseevent.xdata, mouseevent.ydata
            ox, oy = self.__silx_origin
            sx, sy = self.__silx_scale
            height, width = self.get_size()
            column = numpy.clip(int((x - ox) / sx), 0, width - 1)
            row = numpy.clip(int((y - oy) / sy), 0, height - 1)
            info['ind'] = (row,), (column,)
        return inside, info

    def set_data(self, A):
        """Overridden to add a fast path for RGBA unit8 images"""
        A = numpy.array(A, copy=False)
        if A.ndim != 3 or A.shape[2] != 4 or A.dtype != numpy.uint8:
            super(Image, self).set_data(A)
        else:
            # Call AxesImage.set_data with small data to set attributes
            super(Image, self).set_data(numpy.zeros((2, 2, 4), dtype=A.dtype))
            self._A = A  # Override stored data


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
        self._matplotlibVersion = _parse_version(matplotlib.__version__)

        self.fig = Figure()
        self.fig.set_facecolor("w")

        self.ax = self.fig.add_axes([.15, .15, .75, .75], label="left")
        self.ax2 = self.ax.twinx()
        self.ax2.set_label("right")
        # Make sure background of Axes is displayed
        self.ax2.patch.set_visible(False)
        self.ax.patch.set_visible(True)

        # Set axis zorder=0.5 so grid is displayed at 0.5
        self.ax.set_axisbelow(True)

        # disable the use of offsets
        try:
            axes = [
                self.ax.get_yaxis().get_major_formatter(),
                self.ax.get_xaxis().get_major_formatter(),
                self.ax2.get_yaxis().get_major_formatter(),
                self.ax2.get_xaxis().get_major_formatter(),
            ]
            for axis in axes:
                axis.set_useOffset(False)
                axis.set_scientific(False)
        except:
            _logger.warning('Cannot disabled axes offsets in %s '
                            % matplotlib.__version__)

        self.ax2.set_autoscaley_on(True)

        # this works but the figure color is left
        if self._matplotlibVersion < _parse_version('2'):
            self.ax.set_axis_bgcolor('none')
        else:
            self.ax.set_facecolor('none')
        self.fig.sca(self.ax)

        self._background = None

        self._colormaps = {}

        self._graphCursor = tuple()

        self._enableAxis('right', False)
        self._isXAxisTimeSeries = False

    def getItemsFromBackToFront(self, condition=None):
        """Order as BackendBase + take into account matplotlib Axes structure"""
        def axesOrder(item):
            if item.isOverlay():
                return 2
            elif isinstance(item, items.YAxisMixIn) and item.getYAxis() == 'right':
                return 1
            else:
                return 0

        return sorted(
            BackendBase.BackendBase.getItemsFromBackToFront(
                self, condition=condition),
            key=axesOrder)

    def _overlayItems(self):
        """Generator of backend renderer for overlay items"""
        for item in self._plot.getItems():
            if (item.isOverlay() and
                    item.isVisible() and
                    item._backendRenderer is not None):
                yield item._backendRenderer

    def _hasOverlays(self):
        """Returns whether there is an overlay layer or not.

        The overlay layers contains overlay items and the crosshair.

        :rtype: bool
        """
        if self._graphCursor:
            return True  # There is the crosshair

        for item in self._overlayItems():
            return True  # There is at least one overlay item
        return False

    # Add methods

    def _getMarkerFromSymbol(self, symbol):
        """Returns a marker that can be displayed by matplotlib.

        :param str symbol: A symbol description used by silx
        :rtype: Union[str,int,matplotlib.path.Path]
        """
        path = get_path_from_symbol(symbol)
        if path is not None:
            return path
        num = _SPECIAL_MARKERS.get(symbol, None)
        if num is not None:
            return num
        # This symbol must be supported by matplotlib
        return symbol

    def addCurve(self, x, y,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror,
                 fill, alpha, symbolsize, baseline):
        for parameter in (x, y, color, symbol, linewidth, linestyle,
                          yaxis, fill, alpha, symbolsize):
            assert parameter is not None
        assert yaxis in ('left', 'right')

        if (len(color) == 4 and
                type(color[3]) in [type(1), numpy.uint8, numpy.int8]):
            color = numpy.array(color, dtype=numpy.float64) / 255.

        if yaxis == "right":
            axes = self.ax2
            self._enableAxis("right", True)
        else:
            axes = self.ax

        pickradius = 3

        artists = []  # All the artists composing the curve

        # First add errorbars if any so they are behind the curve
        if xerror is not None or yerror is not None:
            if hasattr(color, 'dtype') and len(color) == len(x):
                errorbarColor = 'k'
            else:
                errorbarColor = color

            # Nx1 error array deprecated in matplotlib >=3.1 (removed in 3.3)
            if (isinstance(xerror, numpy.ndarray) and xerror.ndim == 2 and
                        xerror.shape[1] == 1):
                xerror = numpy.ravel(xerror)
            if (isinstance(yerror, numpy.ndarray) and yerror.ndim == 2 and
                    yerror.shape[1] == 1):
                yerror = numpy.ravel(yerror)

            errorbars = axes.errorbar(x, y,
                                      xerr=xerror, yerr=yerror,
                                      linestyle=' ', color=errorbarColor)
            artists += list(errorbars.get_children())

        if hasattr(color, 'dtype') and len(color) == len(x):
            # scatter plot
            if color.dtype not in [numpy.float32, numpy.float64]:
                actualColor = color / 255.
            else:
                actualColor = color

            if linestyle not in ["", " ", None]:
                # scatter plot with an actual line ...
                # we need to assign a color ...
                curveList = axes.plot(x, y,
                                      linestyle=linestyle,
                                      color=actualColor[0],
                                      linewidth=linewidth,
                                      picker=True,
                                      pickradius=pickradius,
                                      marker=None)
                artists += list(curveList)

            marker = self._getMarkerFromSymbol(symbol)
            scatter = axes.scatter(x, y,
                                   color=actualColor,
                                   marker=marker,
                                   picker=True,
                                   pickradius=pickradius,
                                   s=symbolsize**2)
            artists.append(scatter)

            if fill:
                if baseline is None:
                    _baseline = FLOAT32_MINPOS
                else:
                    _baseline = baseline
                artists.append(axes.fill_between(
                    x, _baseline, y, facecolor=actualColor[0], linestyle=''))

        else:  # Curve
            curveList = axes.plot(x, y,
                                  linestyle=linestyle,
                                  color=color,
                                  linewidth=linewidth,
                                  marker=symbol,
                                  picker=True,
                                  pickradius=pickradius,
                                  markersize=symbolsize)
            artists += list(curveList)

            if fill:
                if baseline is None:
                    _baseline = FLOAT32_MINPOS
                else:
                    _baseline = baseline
                artists.append(
                    axes.fill_between(x, _baseline, y, facecolor=color))

        for artist in artists:
            if alpha < 1:
                artist.set_alpha(alpha)

        return _PickableContainer(artists)

    def addImage(self, data, origin, scale, colormap, alpha):
        # Non-uniform image
        # http://wiki.scipy.org/Cookbook/Histograms
        # Non-linear axes
        # http://stackoverflow.com/questions/11488800/non-linear-axes-for-imshow-in-matplotlib
        for parameter in (data, origin, scale):
            assert parameter is not None

        origin = float(origin[0]), float(origin[1])
        scale = float(scale[0]), float(scale[1])
        height, width = data.shape[0:2]

        # All image are shown as RGBA image
        image = Image(self.ax,
                      interpolation='nearest',
                      picker=True,
                      origin='lower',
                      silx_origin=origin,
                      silx_scale=scale)

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

        if data.ndim == 2:  # Data image, convert to RGBA image
            data = colormap.applyToData(data)
        elif data.dtype == numpy.uint16:
            # Normalize uint16 data to have a similar behavior as opengl backend
            data = data.astype(numpy.float32)
            data /= 65535
        
        image.set_data(data)
        self.ax.add_artist(image)
        return image

    def addTriangles(self, x, y, triangles, color, alpha):
        for parameter in (x, y, triangles, color, alpha):
            assert parameter is not None

        color = numpy.array(color, copy=False)
        assert color.ndim == 2 and len(color) == len(x)

        if color.dtype not in [numpy.float32, numpy.float64]:
            color = color.astype(numpy.float32) / 255.

        collection = TriMesh(
            Triangulation(x, y, triangles),
            alpha=alpha,
            pickradius=0)  # 0 enables picking on filled triangle
        collection.set_color(color)
        self.ax.add_collection(collection)

        return collection

    def addShape(self, x, y, shape, color, fill, overlay,
                 linestyle, linewidth, linebgcolor):
        if (linebgcolor is not None and
                shape not in ('rectangle', 'polygon', 'polylines')):
            _logger.warning(
                'linebgcolor not implemented for %s with matplotlib backend',
                shape)
        xView = numpy.array(x, copy=False)
        yView = numpy.array(y, copy=False)

        linestyle = normalize_linestyle(linestyle)

        if shape == "line":
            item = self.ax.plot(x, y, color=color,
                                linestyle=linestyle, linewidth=linewidth,
                                marker=None)[0]

        elif shape == "hline":
            if hasattr(y, "__len__"):
                y = y[-1]
            item = self.ax.axhline(y, color=color,
                                   linestyle=linestyle, linewidth=linewidth)

        elif shape == "vline":
            if hasattr(x, "__len__"):
                x = x[-1]
            item = self.ax.axvline(x, color=color,
                                   linestyle=linestyle, linewidth=linewidth)

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
                             color=color,
                             linestyle=linestyle,
                             linewidth=linewidth)
            if fill:
                item.set_hatch('.')

            if linestyle != "solid" and linebgcolor is not None:
                item = _DoubleColoredLinePatch(item)
                item.linebgcolor = linebgcolor

            self.ax.add_patch(item)

        elif shape in ('polygon', 'polylines'):
            points = numpy.array((xView, yView)).T
            if shape == 'polygon':
                closed = True
            else:  # shape == 'polylines'
                closed = numpy.all(numpy.equal(points[0], points[-1]))
            item = Polygon(points,
                           closed=closed,
                           fill=False,
                           color=color,
                           linestyle=linestyle,
                           linewidth=linewidth)
            if fill and shape == 'polygon':
                item.set_hatch('/')

            if linestyle != "solid" and linebgcolor is not None:
                item = _DoubleColoredLinePatch(item)
                item.linebgcolor = linebgcolor

            self.ax.add_patch(item)

        else:
            raise NotImplementedError("Unsupported item shape %s" % shape)

        if overlay:
            item.set_animated(True)

        return item

    def addMarker(self, x, y, text, color,
                  symbol, linestyle, linewidth, constraint, yaxis):
        textArtist = None

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=yaxis)

        if yaxis == 'left':
            ax = self.ax
        elif yaxis == 'right':
            ax = self.ax2
        else:
            assert(False)

        marker = self._getMarkerFromSymbol(symbol)
        if x is not None and y is not None:
            line = ax.plot(x, y,
                           linestyle=" ",
                           color=color,
                           marker=marker,
                           markersize=10.)[-1]

            if text is not None:
                textArtist = _TextWithOffset(x, y, text,
                                             color=color,
                                             horizontalalignment='left')
                if symbol is not None:
                    textArtist.pixel_offset = 10, 3
        elif x is not None:
            line = ax.axvline(x,
                              color=color,
                              linewidth=linewidth,
                              linestyle=linestyle)
            if text is not None:
                # Y position will be updated in updateMarkerText call
                textArtist = _TextWithOffset(x, 1., text,
                                             color=color,
                                             horizontalalignment='left',
                                             verticalalignment='top')
                textArtist.pixel_offset = 5, 3
        elif y is not None:
            line = ax.axhline(y,
                              color=color,
                              linewidth=linewidth,
                              linestyle=linestyle)

            if text is not None:
                # X position will be updated in updateMarkerText call
                textArtist = _TextWithOffset(1., y, text,
                                             color=color,
                                             horizontalalignment='right',
                                             verticalalignment='top')
                textArtist.pixel_offset = 5, 3
        else:
            raise RuntimeError('A marker must at least have one coordinate')

        line.set_picker(True)
        line.set_pickradius(5)

        # All markers are overlays
        line.set_animated(True)
        if textArtist is not None:
            ax.add_artist(textArtist)
            textArtist.set_animated(True)

        artists = [line] if textArtist is None else [line, textArtist]
        container = _MarkerContainer(artists, symbol, x, y, yaxis)
        container.updateMarkerText(xmin, xmax, ymin, ymax, self.isYAxisInverted())

        return container

    def _updateMarkers(self):
        xmin, xmax = self.ax.get_xbound()
        ymin1, ymax1 = self.ax.get_ybound()
        ymin2, ymax2 = self.ax2.get_ybound()
        yinverted = self.isYAxisInverted()
        for item in self._overlayItems():
            if isinstance(item, _MarkerContainer):
                if item.yAxis == 'left':
                    item.updateMarkerText(xmin, xmax, ymin1, ymax1, yinverted)
                else:
                    item.updateMarkerText(xmin, xmax, ymin2, ymax2, yinverted)

    # Remove methods

    def remove(self, item):
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
            if self._graphCursor:
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
        with self._plot._paintContext():
            self._replot()

    def _replot(self):
        """Call from subclass :meth:`replot` to handle updates"""
        # TODO images, markers? scatter plot? move in remove?
        # Right Y axis only support curve for now
        # Hide right Y axis if no line is present
        self._dirtyLimits = False
        if not self.ax2.lines:
            self._enableAxis('right', False)

    def _drawOverlays(self):
        """Draw overlays if any."""
        def condition(item):
            return (item.isVisible() and
                    item._backendRenderer is not None and
                    item.isOverlay())

        for item in self.getItemsFromBackToFront(condition=condition):
            if (isinstance(item, items.YAxisMixIn) and
                    item.getYAxis() == 'right'):
                axes = self.ax2
            else:
                axes = self.ax
            axes.draw_artist(item._backendRenderer)

        for item in self._graphCursor:
            self.ax.draw_artist(item)

    def updateZOrder(self):
        """Reorder all items with z order from 0 to 1"""
        items = self.getItemsFromBackToFront(
            lambda item: item.isVisible() and item._backendRenderer is not None)
        count = len(items)
        for index, item in enumerate(items):
            if item.getZValue() < 0.5:
                # Make sure matplotlib z order is below the grid (with z=0.5)
                zorder = 0.5 * index / count
            else:  # Make sure matplotlib z order is above the grid (> 0.5)
                zorder = 1. + index / count
            if zorder != item._backendRenderer.get_zorder():
                item._backendRenderer.set_zorder(zorder)

    def saveGraph(self, fileName, fileFormat, dpi):
        self.updateZOrder()

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

        self._updateMarkers()

    def getGraphXLimits(self):
        if self._dirtyLimits and self.isKeepDataAspectRatio():
            self.ax.apply_aspect()
            self.ax2.apply_aspect()
            self._dirtyLimits = False
        return self.ax.get_xbound()

    def setGraphXLimits(self, xmin, xmax):
        self._dirtyLimits = True
        self.ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
        self._updateMarkers()

    def getGraphYLimits(self, axis):
        assert axis in ('left', 'right')
        ax = self.ax2 if axis == 'right' else self.ax

        if not ax.get_visible():
            return None

        if self._dirtyLimits and self.isKeepDataAspectRatio():
            self.ax.apply_aspect()
            self.ax2.apply_aspect()
            self._dirtyLimits = False

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

        self._updateMarkers()

    # Graph axes

    def setXAxisTimeZone(self, tz):
        super(BackendMatplotlib, self).setXAxisTimeZone(tz)

        # Make new formatter and locator with the time zone.
        self.setXAxisTimeSeries(self.isXAxisTimeSeries())

    def isXAxisTimeSeries(self):
        return self._isXAxisTimeSeries

    def setXAxisTimeSeries(self, isTimeSeries):
        self._isXAxisTimeSeries = isTimeSeries
        if self._isXAxisTimeSeries:
            # We can't use a matplotlib.dates.DateFormatter because it expects
            # the data to be in datetimes. Silx works internally with
            # timestamps (floats).
            locator = NiceDateLocator(tz=self.getXAxisTimeZone())
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(
                NiceAutoDateFormatter(locator, tz=self.getXAxisTimeZone()))
        else:
            try:
                scalarFormatter = ScalarFormatter(useOffset=False)
            except:
                _logger.warning('Cannot disabled axes offsets in %s ' %
                                matplotlib.__version__)
                scalarFormatter = ScalarFormatter()
            self.ax.xaxis.set_major_formatter(scalarFormatter)

    def setXAxisLogarithmic(self, flag):
        # Workaround for matplotlib 2.1.0 when one tries to set an axis
        # to log scale with both limits <= 0
        # In this case a draw with positive limits is needed first
        if flag and self._matplotlibVersion >= _parse_version('2.1.0'):
            xlim = self.ax.get_xlim()
            if xlim[0] <= 0 and xlim[1] <= 0:
                self.ax.set_xlim(1, 10)
                self.draw()

        self.ax2.set_xscale('log' if flag else 'linear')
        self.ax.set_xscale('log' if flag else 'linear')

    def setYAxisLogarithmic(self, flag):
        # Workaround for matplotlib 2.0 issue with negative bounds
        # before switching to log scale
        if flag and self._matplotlibVersion >= _parse_version('2.0.0'):
            redraw = False
            for axis, dataRangeIndex in ((self.ax, 1), (self.ax2, 2)):
                ylim = axis.get_ylim()
                if ylim[0] <= 0 or ylim[1] <= 0:
                    dataRange = self._plot.getDataRange()[dataRangeIndex]
                    if dataRange is None:
                        dataRange = 1, 100  # Fallback
                    axis.set_ylim(*dataRange)
                    redraw = True
            if redraw:
                self.draw()

        self.ax2.set_yscale('log' if flag else 'linear')
        self.ax.set_yscale('log' if flag else 'linear')

    def setYAxisInverted(self, flag):
        if self.ax.yaxis_inverted() != bool(flag):
            self.ax.invert_yaxis()
            self._updateMarkers()

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

    def _getDevicePixelRatio(self) -> float:
        """Compatibility wrapper for devicePixelRatioF"""
        return 1.

    def _mplToQtPosition(
        self,
        x: Union[float,numpy.ndarray],
        y: Union[float,numpy.ndarray]
    ) -> Tuple[Union[float,numpy.ndarray], Union[float,numpy.ndarray]]:
        """Convert matplotlib "display" space coord to Qt widget logical pixel
        """
        ratio = self._getDevicePixelRatio()
        # Convert from matplotlib origin (bottom) to Qt origin (top)
        # and apply device pixel ratio
        return x / ratio, (self.fig.get_window_extent().height - y) / ratio

    def _qtToMplPosition(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Qt widget logical pixel to matplotlib "display" space coord
        """
        ratio = self._getDevicePixelRatio()
        # Apply device pixel ration and
        # convert from Qt origin (top) to matplotlib origin (bottom)
        return x * ratio, self.fig.get_window_extent().height - (y * ratio)

    def dataToPixel(self, x, y, axis):
        ax = self.ax2 if axis == "right" else self.ax
        points = numpy.transpose((x, y))
        displayPos = ax.transData.transform(points).transpose()
        return self._mplToQtPosition(*displayPos)

    def pixelToData(self, x, y, axis):
        ax = self.ax2 if axis == "right" else self.ax
        displayPos = self._qtToMplPosition(x, y)
        return tuple(ax.transData.inverted().transform_point(displayPos))

    def getPlotBoundsInPixels(self):
        bbox = self.ax.get_window_extent()
        # Warning this is not returning int...
        ratio = self._getDevicePixelRatio()
        return tuple(int(value / ratio) for value in (
            bbox.xmin,
            self.fig.get_window_extent().height - bbox.ymax,
            bbox.width,
            bbox.height))

    def setAxesMargins(self, left: float, top: float, right: float, bottom: float):
        width, height = 1. - left - right, 1. - top - bottom
        position = left, bottom, width, height

        # Toggle display of axes and viewbox rect
        isFrameOn = position != (0., 0., 1., 1.)
        self.ax.set_frame_on(isFrameOn)
        self.ax2.set_frame_on(isFrameOn)

        self.ax.set_position(position)
        self.ax2.set_position(position)

        self._synchronizeBackgroundColors()
        self._synchronizeForegroundColors()
        self._plot._setDirtyPlot()

    def _synchronizeBackgroundColors(self):
        backgroundColor = self._plot.getBackgroundColor().getRgbF()

        dataBackgroundColor = self._plot.getDataBackgroundColor()
        if dataBackgroundColor.isValid():
            dataBackgroundColor = dataBackgroundColor.getRgbF()
        else:
            dataBackgroundColor = backgroundColor

        if self.ax.get_frame_on():
            self.fig.patch.set_facecolor(backgroundColor)
            if self._matplotlibVersion < _parse_version('2'):
                self.ax.set_axis_bgcolor(dataBackgroundColor)
            else:
                self.ax.set_facecolor(dataBackgroundColor)
        else:
            self.fig.patch.set_facecolor(dataBackgroundColor)

    def _synchronizeForegroundColors(self):
        foregroundColor = self._plot.getForegroundColor().getRgbF()

        gridColor = self._plot.getGridColor()
        if gridColor.isValid():
            gridColor = gridColor.getRgbF()
        else:
            gridColor = foregroundColor

        for axes in (self.ax, self.ax2):
            if axes.get_frame_on():
                axes.spines['bottom'].set_color(foregroundColor)
                axes.spines['top'].set_color(foregroundColor)
                axes.spines['right'].set_color(foregroundColor)
                axes.spines['left'].set_color(foregroundColor)
                axes.tick_params(axis='x', colors=foregroundColor)
                axes.tick_params(axis='y', colors=foregroundColor)
                axes.yaxis.label.set_color(foregroundColor)
                axes.xaxis.label.set_color(foregroundColor)
                axes.title.set_color(foregroundColor)

                for line in axes.get_xgridlines():
                    line.set_color(gridColor)

                for line in axes.get_ygridlines():
                    line.set_color(gridColor)
                # axes.grid().set_markeredgecolor(gridColor)

    def setBackgroundColors(self, backgroundColor, dataBackgroundColor):
        self._synchronizeBackgroundColors()

    def setForegroundColors(self, foregroundColor, gridColor):
        self._synchronizeForegroundColors()


class BackendMatplotlibQt(BackendMatplotlib, FigureCanvasQTAgg):
    """QWidget matplotlib backend using a QtAgg canvas.

    It adds fast overlay drawing and mouse event management.
    """

    _sigPostRedisplay = qt.Signal()
    """Signal handling automatic asynchronous replot"""

    def __init__(self, plot, parent=None):
        BackendMatplotlib.__init__(self, plot, parent)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        self._limitsBeforeResize = None

        FigureCanvasQTAgg.setSizePolicy(
            self, qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        # Make postRedisplay asynchronous using Qt signal
        self._sigPostRedisplay.connect(
            self.__deferredReplot, qt.Qt.QueuedConnection)

        self._picked = None

        self.mpl_connect('button_press_event', self._onMousePress)
        self.mpl_connect('button_release_event', self._onMouseRelease)
        self.mpl_connect('motion_notify_event', self._onMouseMove)
        self.mpl_connect('scroll_event', self._onMouseWheel)

    def postRedisplay(self):
        self._sigPostRedisplay.emit()

    def __deferredReplot(self):
        # Since this is deferred, makes sure it is still needed
        plot = self._plotRef()
        if (plot is not None and
                plot._getDirtyPlot() and
                plot.getBackend() is self):
            self.replot()

    def _getDevicePixelRatio(self) -> float:
        """Compatibility wrapper for devicePixelRatioF"""
        if hasattr(self, 'devicePixelRatioF'):
            ratio = self.devicePixelRatioF()
        else:  # Qt < 5.6 compatibility
            ratio = float(self.devicePixelRatio())
        # Safety net: avoid returning 0
        return ratio if ratio != 0. else 1.

    # Mouse event forwarding

    _MPL_TO_PLOT_BUTTONS = {1: 'left', 2: 'middle', 3: 'right'}

    def _onMousePress(self, event):
        button = self._MPL_TO_PLOT_BUTTONS.get(event.button, None)
        if button is not None:
            x, y = self._mplToQtPosition(event.x, event.y)
            self._plot.onMousePress(int(x), int(y), button)

    def _onMouseMove(self, event):
        x, y = self._mplToQtPosition(event.x, event.y)
        if self._graphCursor:
            position = self._plot.pixelToData(
                x, y, axis='left', check=True)
            lineh, linev = self._graphCursor
            if position is not None:
                linev.set_visible(True)
                linev.set_xdata((position[0], position[0]))
                lineh.set_visible(True)
                lineh.set_ydata((position[1], position[1]))
                self._plot._setDirtyPlot(overlayOnly=True)
            elif lineh.get_visible():
                    lineh.set_visible(False)
                    linev.set_visible(False)
                    self._plot._setDirtyPlot(overlayOnly=True)
            # onMouseMove must trigger replot if dirty flag is raised

        self._plot.onMouseMove(int(x), int(y))

    def _onMouseRelease(self, event):
        button = self._MPL_TO_PLOT_BUTTONS.get(event.button, None)
        if button is not None:
            x, y = self._mplToQtPosition(event.x, event.y)
            self._plot.onMouseRelease(int(x), int(y), button)

    def _onMouseWheel(self, event):
        x, y = self._mplToQtPosition(event.x, event.y)
        self._plot.onMouseWheel(int(x), int(y), event.step)

    def leaveEvent(self, event):
        """QWidget event handler"""
        try:
            plot = self._plot
        except RuntimeError:
            pass
        else:
            plot.onMouseLeaveWidget()

    # picking

    def pickItem(self, x, y, item):
        xDisplay, yDisplay = self._qtToMplPosition(x, y)
        mouseEvent = MouseEvent(
            'button_press_event', self, int(xDisplay), int(yDisplay))
        # Override axes and data position with the axes
        mouseEvent.inaxes = item.axes
        mouseEvent.xdata, mouseEvent.ydata = self.pixelToData(
            x, y, axis='left' if item.axes is self.ax else 'right')
        picked, info = item.contains(mouseEvent)

        if not picked:
            return None

        elif isinstance(item, TriMesh):
            # Convert selected triangle to data point indices
            triangulation = item._triangulation
            indices = triangulation.get_masked_triangles()[info['ind'][0]]

            # Sort picked triangle points by distance to mouse
            # from furthest to closest to put closest point last
            # This is to be somewhat consistent with last scatter point
            # being the top one.
            xdata, ydata = self.pixelToData(x, y, axis='left')
            dists = ((triangulation.x[indices] - xdata) ** 2 +
                     (triangulation.y[indices] - ydata) ** 2)
            return indices[numpy.flip(numpy.argsort(dists), axis=0)]

        else:  # Returns indices if any
            return info.get('ind', ())

    # replot control

    def resizeEvent(self, event):
        # Store current limits
        self._limitsBeforeResize = (
            self.ax.get_xbound(), self.ax.get_ybound(), self.ax2.get_ybound())

        FigureCanvasQTAgg.resizeEvent(self, event)
        if self.isKeepDataAspectRatio() or self._hasOverlays():
            # This is needed with matplotlib 1.5.x and 2.0.x
            self._plot._setDirtyPlot()

    def draw(self):
        """Overload draw

        It performs a full redraw (including overlays) of the plot.
        It also resets background and emit limits changed signal.

        This is directly called by matplotlib for widget resize.
        """
        if self.size().isEmpty():
            return  # Skip rendering of 0-sized canvas

        self.updateZOrder()

        # Starting with mpl 2.1.0, toggling autoscale raises a ValueError
        # in some situations. See #1081, #1136, #1163,
        if self._matplotlibVersion >= _parse_version("2.0.0"):
            try:
                FigureCanvasQTAgg.draw(self)
            except ValueError as err:
                _logger.debug(
                    "ValueError caught while calling FigureCanvasQTAgg.draw: "
                    "'%s'", err)
        else:
            FigureCanvasQTAgg.draw(self)

        if self._hasOverlays():
            # Save background
            self._background = self.copy_from_bbox(self.fig.bbox)
        else:
            self._background = None  # Reset background

        # Check if limits changed due to a resize of the widget
        if self._limitsBeforeResize is not None:
            xLimits, yLimits, yRightLimits = self._limitsBeforeResize
            self._limitsBeforeResize = None

            if (xLimits != self.ax.get_xbound() or
                    yLimits != self.ax.get_ybound()):
                self._updateMarkers()

            if xLimits != self.ax.get_xbound():
                self._plot.getXAxis()._emitLimitsChanged()
            if yLimits != self.ax.get_ybound():
                self._plot.getYAxis(axis='left')._emitLimitsChanged()
            if yRightLimits != self.ax2.get_ybound():
                self._plot.getYAxis(axis='right')._emitLimitsChanged()

        self._drawOverlays()

    def replot(self):
        if not qt_inspect.isValid(self):
            _logger.info("replot requested but widget no longer exists")
            return

        with self._plot._paintContext():
            BackendMatplotlib._replot(self)

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

            # Workaround issue of rendering overlays with some matplotlib versions
            if (_parse_version('1.5') <= self._matplotlibVersion < _parse_version('2.1') and
                    not hasattr(self, '_firstReplot')):
                self._firstReplot = False
                if self._hasOverlays():
                    qt.QTimer.singleShot(0, self.draw)  # Request async draw

    # cursor

    _QT_CURSORS = {
        BackendBase.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        BackendBase.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        BackendBase.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        BackendBase.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        BackendBase.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setGraphCursorShape(self, cursor):
        if cursor is None:
            FigureCanvasQTAgg.unsetCursor(self)
        else:
            cursor = self._QT_CURSORS[cursor]
            FigureCanvasQTAgg.setCursor(self, qt.QCursor(cursor))
