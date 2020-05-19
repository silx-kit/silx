# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
"""Module containing several widgets associated to a colormap.
"""

__authors__ = ["H. Payno", "T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import logging
import weakref
import numpy

from ._utils import ticklayout
from .. import qt
from silx.gui import colors

_logger = logging.getLogger(__name__)


class ColorBarWidget(qt.QWidget):
    """Colorbar widget displaying a colormap

    It uses a description of colormap as dict compatible with :class:`Plot`.

    .. image:: img/linearColorbar.png
        :width: 80px
        :align: center

    To run the following sample code, a QApplication must be initialized.

    >>> from silx.gui.plot import Plot2D
    >>> from silx.gui.plot.ColorBar import ColorBarWidget

    >>> plot = Plot2D()  # Create a plot widget
    >>> plot.show()

    >>> colorbar = ColorBarWidget(plot=plot, legend='Colormap')  # Associate the colorbar with it
    >>> colorbar.show()

    Initializer parameters:

    :param parent: See :class:`QWidget`
    :param plot: PlotWidget the colorbar is attached to (optional)
    :param str legend: the label to set to the colorbar
    """
    sigVisibleChanged = qt.Signal(bool)
    """Emitted when the property `visible` have changed."""

    def __init__(self, parent=None, plot=None, legend=None):
        self._isConnected = False
        self._plotRef = None
        self._colormap = None
        self._data = None

        super(ColorBarWidget, self).__init__(parent)

        self.__buildGUI()
        self.setLegend(legend)
        self.setPlot(plot)

    def __buildGUI(self):
        self.setLayout(qt.QHBoxLayout())

        # create color scale widget
        self._colorScale = ColorScaleBar(parent=self,
                                         colormap=None)
        self.layout().addWidget(self._colorScale)

        # legend (is the right group)
        self.legend = _VerticalLegend('', self)
        self.layout().addWidget(self.legend)

        self.layout().setSizeConstraint(qt.QLayout.SetMinAndMaxSize)

    def getPlot(self):
        """Returns the :class:`Plot` associated to this widget or None"""
        return None if self._plotRef is None else self._plotRef()

    def setPlot(self, plot):
        """Associate a plot to the ColorBar

        :param plot: the plot to associate with the colorbar.
                     If None will remove any connection with a previous plot.
        """
        self._disconnectPlot()
        self._plotRef = None if plot is None else weakref.ref(plot)
        self._connectPlot()

    def _disconnectPlot(self):
        """Disconnect from Plot signals"""
        plot = self.getPlot()
        if plot is not None and self._isConnected:
            self._isConnected = False
            plot.sigActiveImageChanged.disconnect(
                self._activeImageChanged)
            plot.sigActiveScatterChanged.disconnect(
                self._activeScatterChanged)
            plot.sigPlotSignal.disconnect(self._defaultColormapChanged)

    def _connectPlot(self):
        """Connect to Plot signals"""
        plot = self.getPlot()
        if plot is not None and not self._isConnected:
            activeImageLegend = plot.getActiveImage(just_legend=True)
            activeScatterLegend = plot._getActiveItem(
                kind='scatter', just_legend=True)
            if activeImageLegend is None and activeScatterLegend is None:
                # Show plot default colormap
                self._syncWithDefaultColormap()
            elif activeImageLegend is not None:  # Show active image colormap
                self._activeImageChanged(None, activeImageLegend)
            elif activeScatterLegend is not None:  # Show active scatter colormap
                self._activeScatterChanged(None, activeScatterLegend)

            plot.sigActiveImageChanged.connect(self._activeImageChanged)
            plot.sigActiveScatterChanged.connect(self._activeScatterChanged)
            plot.sigPlotSignal.connect(self._defaultColormapChanged)
            self._isConnected = True

    def setVisible(self, isVisible):
        # isHidden looks to be always synchronized, while isVisible is not
        wasHidden = self.isHidden()
        qt.QWidget.setVisible(self, isVisible)
        if wasHidden != self.isHidden():
            self.sigVisibleChanged.emit(not self.isHidden())

    def showEvent(self, event):
        self._connectPlot()

    def hideEvent(self, event):
        self._disconnectPlot()

    def getColormap(self):
        """Returns the colormap displayed in the colorbar.

        :rtype: ~silx.gui.colors.Colormap
        """
        return self.getColorScaleBar().getColormap()

    def setColormap(self, colormap, data=None):
        """Set the colormap to be displayed.

        :param ~silx.gui.colors.Colormap colormap:
            The colormap to apply on the ColorBarWidget
        :param Union[numpy.ndarray,~silx.gui.plot.items.ColormapMixin] data:
            The data to display or item, needed if the colormap require an autoscale
        """
        self._data = data
        self.getColorScaleBar().setColormap(colormap=colormap,
                                            data=data)
        if self._colormap is not None:
            self._colormap.sigChanged.disconnect(self._colormapHasChanged)
        self._colormap = colormap
        if self._colormap is not None:
            self._colormap.sigChanged.connect(self._colormapHasChanged)

    def _colormapHasChanged(self):
        """handler of the Colormap.sigChanged signal
        """
        assert self._colormap is not None
        self.setColormap(colormap=self._colormap,
                         data=self._data)

    def setLegend(self, legend):
        """Set the legend displayed along the colorbar

        :param str legend: The label
        """
        if legend is None or legend == "":
            self.legend.hide()
            self.legend.setText("")
        else:
            assert type(legend) is str
            self.legend.show()
            self.legend.setText(legend)

    def getLegend(self):
        """
        Returns the legend displayed along the colorbar

        :return: return the legend displayed along the colorbar
        :rtype: str
        """
        return self.legend.text()

    def _activeScatterChanged(self, previous, legend):
        """Handle plot active scatter changed"""
        plot = self.getPlot()

        # Do not handle active scatter while there is an image
        if plot.getActiveImage() is not None:
            return

        if legend is None:  # No active scatter, display no colormap
            self.setColormap(colormap=None)
            return

        # Sync with active scatter
        scatter = plot._getActiveItem(kind='scatter')

        self.setColormap(colormap=scatter.getColormap(),
                         data=scatter)

    def _activeImageChanged(self, previous, legend):
        """Handle plot active image changed"""
        plot = self.getPlot()

        if legend is None:  # No active image, try with active scatter
            activeScatterLegend = plot._getActiveItem(
                kind='scatter', just_legend=True)
            # No more active image, use active scatter if any
            self._activeScatterChanged(None, activeScatterLegend)
        else:
            # Sync with active image
            image = plot.getActiveImage()

            # RGB(A) image, display default colormap
            array = image.getData(copy=False)
            if array.ndim != 2:
                self.setColormap(colormap=None)
                return

            # data image, sync with image colormap
            # do we need the copy here : used in the case we are changing
            # vmin and vmax but should have already be done by the plot
            self.setColormap(colormap=image.getColormap(), data=image)

    def _defaultColormapChanged(self, event):
        """Handle plot default colormap changed"""
        if event['event'] == 'defaultColormapChanged':
            plot = self.getPlot()
            if (plot is not None and
                    plot.getActiveImage() is None and
                    plot._getActiveItem(kind='scatter') is None):
                # No active item, take default colormap update into account
                self._syncWithDefaultColormap()

    def _syncWithDefaultColormap(self):
        """Update colorbar according to plot default colormap"""
        self.setColormap(self.getPlot().getDefaultColormap())

    def getColorScaleBar(self):
        """

        :return: return the :class:`ColorScaleBar` used to display ColorScale
            and ticks"""
        return self._colorScale


class _VerticalLegend(qt.QLabel):
    """Display vertically the given text
    """
    def __init__(self, text, parent=None):
        """

        :param text: the legend
        :param parent: the Qt parent if any
        """
        qt.QLabel.__init__(self, text, parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

    def paintEvent(self, event):
        painter = qt.QPainter(self)
        painter.setFont(self.font())

        painter.translate(0, self.rect().height())
        painter.rotate(270)
        newRect = qt.QRect(0, 0, self.rect().height(), self.rect().width())

        painter.drawText(newRect, qt.Qt.AlignHCenter, self.text())

        fm = qt.QFontMetrics(self.font())
        preferedHeight = fm.width(self.text())
        preferedWidth = fm.height()
        self.setFixedWidth(preferedWidth)
        self.setMinimumHeight(preferedHeight)


class ColorScaleBar(qt.QWidget):
    """This class is making the composition of a :class:`_ColorScale` and a
    :class:`_TickBar`.

    It is the simplest widget displaying ticks and colormap gradient.

    .. image:: img/colorScaleBar.png
        :width: 150px
        :align: center

    To run the following sample code, a QApplication must be initialized.

    >>> colormap = Colormap(name='gray',
    ...                     norm='log',
    ...                     vmin=1,
    ...                     vmax=100000,
    ...             )
    >>> colorscale = ColorScaleBar(parent=None,
    ...                            colormap=colormap )
    >>> colorscale.show()

    Initializer parameters :

    :param colormap: the colormap to be displayed
    :param parent: the Qt parent if any
    :param displayTicksValues: display the ticks value or only the '-'
    """

    _TEXT_MARGIN = 5
    """The tick bar need a margin to display all labels at the correct place.
    So the ColorScale should have the same margin in order for both to fit"""

    def __init__(self, parent=None, colormap=None, data=None,
                 displayTicksValues=True):
        super(ColorScaleBar, self).__init__(parent)

        self.minVal = None
        """Value set to the _minLabel"""
        self.maxVal = None
        """Value set to the _maxLabel"""

        self.setLayout(qt.QGridLayout())

        # create the left side group (ColorScale)
        self.colorScale = _ColorScale(colormap=colormap,
                                      data=data,
                                      parent=self,
                                      margin=ColorScaleBar._TEXT_MARGIN)
        if colormap:
            vmin, vmax = colormap.getColormapRange(data)
            normalizer = colormap._getNormalizer()
        else:
            vmin, vmax = colors.DEFAULT_MIN_LIN, colors.DEFAULT_MAX_LIN
            normalizer = None

        self.tickbar = _TickBar(vmin=vmin,
                                vmax=vmax,
                                normalizer=normalizer,
                                parent=self,
                                displayValues=displayTicksValues,
                                margin=ColorScaleBar._TEXT_MARGIN)

        self.layout().addWidget(self.tickbar, 1, 0, 1, 1, qt.Qt.AlignRight)
        self.layout().addWidget(self.colorScale, 1, 1, qt.Qt.AlignLeft)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # max label
        self._maxLabel = qt.QLabel(str(1.0), parent=self)
        self._maxLabel.setToolTip(str(0.0))
        self.layout().addWidget(self._maxLabel, 0, 0, 1, 2, qt.Qt.AlignRight)

        # min label
        self._minLabel = qt.QLabel(str(0.0), parent=self)
        self._minLabel.setToolTip(str(0.0))
        self.layout().addWidget(self._minLabel, 2, 0, 1, 2, qt.Qt.AlignRight)

        self.layout().setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
        self.layout().setColumnStretch(0, 1)
        self.layout().setRowStretch(1, 1)

    def getTickBar(self):
        """

        :return: the instanciation of the :class:`_TickBar`
        """
        return self.tickbar

    def getColorScale(self):
        """

        :return: the instanciation of the :class:`_ColorScale`
        """
        return self.colorScale

    def getColormap(self):
        """

        :returns: the colormap.
        :rtype: :class:`.Colormap`
        """
        return self.colorScale.getColormap()

    def setColormap(self, colormap, data=None):
        """Set the new colormap to be displayed

        :param Colormap colormap: the colormap to set
        :param Union[numpy.ndarray,~silx.gui.plot.items.Item] data:
            The data or item to display, needed if the colormap requires an autoscale
        """
        self.colorScale.setColormap(colormap, data)

        if colormap is not None:
            vmin, vmax = colormap.getColormapRange(data)
            normalizer = colormap._getNormalizer()
        else:
            vmin, vmax = None, None
            normalizer = None

        self.tickbar.update(vmin=vmin,
                            vmax=vmax,
                            normalizer=normalizer)
        self._setMinMaxLabels(vmin, vmax)

    def setMinMaxVisible(self, val=True):
        """Change visibility of the min label and the max label

        :param val: if True, set the labels visible, otherwise set it not visible
        """
        self._minLabel.setVisible(val)
        self._maxLabel.setVisible(val)

    def _updateMinMax(self):
        """Update the min and max label if we are in the case of the
        configuration 'minMaxValueOnly'"""
        if self.minVal is None:
            text, tooltip = '', ''
        else:
            if self.minVal == 0 or 0 <= numpy.log10(abs(self.minVal)) < 7:
                text = '%.7g' % self.minVal
            else:
                text = '%.2e' % self.minVal
            tooltip = repr(self.minVal)

        self._minLabel.setText(text)
        self._minLabel.setToolTip(tooltip)

        if self.maxVal is None:
            text, tooltip = '', ''
        else:
            if self.maxVal == 0 or 0 <= numpy.log10(abs(self.maxVal)) < 7:
                text = '%.7g' % self.maxVal
            else:
                text = '%.2e' % self.maxVal
            tooltip = repr(self.maxVal)

        self._maxLabel.setText(text)
        self._maxLabel.setToolTip(tooltip)

    def _setMinMaxLabels(self, minVal, maxVal):
        """Change the value of the min and max labels to be displayed.

        :param minVal: the minimal value of the TickBar (not str)
        :param maxVal: the maximal value of the TickBar (not str)
        """
        # bad hack to try to display has much information as possible
        self.minVal = minVal
        self.maxVal = maxVal
        self._updateMinMax()

    def resizeEvent(self, event):
        qt.QWidget.resizeEvent(self, event)
        self._updateMinMax()


class _ColorScale(qt.QWidget):
    """Widget displaying the colormap colorScale.

    Show matching value between the gradient color (from the colormap) at mouse
    position and value.

    .. image:: img/colorScale.png
        :width: 20px
        :align: center


    To run the following sample code, a QApplication must be initialized.

    >>> colormap = Colormap(name='viridis',
    ...                     norm='log',
    ...                     vmin=1,
    ...                     vmax=100000,
    ...             )
    >>> colorscale = ColorScale(parent=None,
    ...                         colormap=colormap)
    >>> colorscale.show()

    Initializer parameters :

    :param colormap: the colormap to be displayed
    :param parent: the Qt parent if any
    :param int margin: the top and left margin to apply.
    :param Union[None,numpy.ndarray,~silx.gui.plot.items.ColormapMixin] data:
        The data or item to use for getting the range for autoscale colormap.

    .. warning:: Value drawing will be
        done at the center of ticks. So if no margin is done your values
        drawing might not be fully done for extrems values.
    """

    _NB_CONTROL_POINTS = 256

    def __init__(self, colormap, parent=None, margin=5, data=None):
        qt.QWidget.__init__(self, parent)
        self._colormap = None
        self.margin = margin
        self.setColormap(colormap, data)

        self.setLayout(qt.QVBoxLayout())
        self.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Expanding)
        # needed to get the mouse event without waiting for button click
        self.setMouseTracking(True)
        self.setMargin(margin)
        self.setContentsMargins(0, 0, 0, 0)

        self.setMinimumHeight(self._NB_CONTROL_POINTS // 2 + 2 * self.margin)
        self.setFixedWidth(25)

    def setColormap(self, colormap, data=None):
        """Set the new colormap to be displayed

        :param dict colormap: the colormap to set
        :param Union[None,numpy.ndarray,~silx.gui.plot.items.ColormapMixin] data:
            Optional data for which to compute colormap range.
        """
        self._colormap = colormap
        self.setEnabled(colormap is not None)

        if colormap is None:
            self.vmin, self.vmax = None, None
        else:
            assert colormap.getNormalization() in colors.Colormap.NORMALIZATIONS
            self.vmin, self.vmax = self._colormap.getColormapRange(data=data)
        self._updateColorGradient()
        self.update()

    def getColormap(self):
        """Returns the colormap

        :rtype: :class:`.Colormap`
        """
        return None if self._colormap is None else self._colormap

    def _updateColorGradient(self):
        """Compute the color gradient"""
        colormap = self.getColormap()
        if colormap is None:
            return

        indices = numpy.linspace(0., 1., self._NB_CONTROL_POINTS)
        colors = colormap.getNColors(nbColors=self._NB_CONTROL_POINTS)
        self._gradient = qt.QLinearGradient(0, 1, 0, 0)
        self._gradient.setCoordinateMode(qt.QGradient.StretchToDeviceMode)
        self._gradient.setStops(
            [(i, qt.QColor(*color)) for i, color in zip(indices, colors)]
        )

    def paintEvent(self, event):
        """"""
        painter = qt.QPainter(self)
        if self.getColormap() is not None:
            painter.setBrush(self._gradient)
            penColor = self.palette().color(qt.QPalette.Active,
                                            qt.QPalette.Foreground)
        else:
            penColor = self.palette().color(qt.QPalette.Disabled,
                                            qt.QPalette.Foreground)
        painter.setPen(penColor)

        painter.drawRect(qt.QRect(
            0,
            self.margin,
            self.width() - 1,
            self.height() - 2 * self.margin - 1))

    def mouseMoveEvent(self, event):
        tooltip = str(self.getValueFromRelativePosition(
            self._getRelativePosition(event.y())))
        qt.QToolTip.showText(event.globalPos(), tooltip, self)
        super(_ColorScale, self).mouseMoveEvent(event)

    def _getRelativePosition(self, yPixel):
        """yPixel : pixel position into _ColorScale widget reference
        """
        # widgets are bottom-top referencial but we display in top-bottom referential
        return 1. - (yPixel - self.margin) / float(self.height() - 2 * self.margin)

    def getValueFromRelativePosition(self, value):
        """Return the value in the colorMap from a relative position in the
        ColorScaleBar (y)

        :param value: float value in [0, 1]
        :return: the value in [colormap['vmin'], colormap['vmax']]
        """
        colormap = self.getColormap()
        if colormap is None:
            return

        value = numpy.clip(value, 0., 1.)
        normalizer = colormap._getNormalizer()
        normMin, normMax = normalizer.apply([self.vmin, self.vmax], self.vmin, self.vmax)

        return normalizer.revert(
            normMin + (normMax - normMin) * value, self.vmin, self.vmax)

    def setMargin(self, margin):
        """Define the margin to fit with a TickBar object.
        This is needed since we can only paint on the viewport of the widget.
        Didn't work with a simple setContentsMargins

        :param int margin: the margin to apply on the top and bottom.
        """
        self.margin = int(margin)
        self.update()


class _TickBar(qt.QWidget):
    """Bar grouping the ticks displayed

    To run the following sample code, a QApplication must be initialized.

    >>> bar = _TickBar(1, 1000, norm='log', parent=None, displayValues=True)
    >>> bar.show()

    .. image:: img/tickbar.png
        :width: 40px
        :align: center

    :param int vmin: smaller value of the range of values
    :param int vmax: higher value of the range of values
    :param normalizer: Normalization object.
    :param parent: the Qt parent if any
    :param bool displayValues: if True display the values close to the tick,
        Otherwise only signal it by '-'
    :param int nticks: the number of tick we want to display. Should be an
        unsigned int ot None. If None, let the Tick bar find the optimal
        number of ticks from the tick density.
    :param int margin: margin to set on the top and bottom
    """
    _WIDTH_DISP_VAL = 45
    """widget width when displayed with ticks labels"""
    _WIDTH_NO_DISP_VAL = 10
    """widget width when displayed without ticks labels"""
    _FONT_SIZE = 10
    """font size for ticks labels"""
    _LINE_WIDTH = 10
    """width of the line to mark a tick"""

    DEFAULT_TICK_DENSITY = 0.015

    def __init__(self, vmin, vmax, normalizer, parent=None, displayValues=True,
                 nticks=None, margin=5):
        super(_TickBar, self).__init__(parent)
        self.margin = margin
        self._nticks = None
        self.ticks = ()
        self.subTicks = ()
        self._forcedDisplayType = None
        self.ticksDensity = _TickBar.DEFAULT_TICK_DENSITY

        self._vmin = vmin
        self._vmax = vmax
        self._normalizer = normalizer
        self.displayValues = displayValues
        self.setTicksNumber(nticks)

        self.setMargin(margin)
        self.setContentsMargins(0, 0, 0, 0)

        self._resetWidth()

    def setTicksValuesVisible(self, val):
        self.displayValues = val
        self._resetWidth()

    def _resetWidth(self):
        width = self._WIDTH_DISP_VAL if self.displayValues else self._WIDTH_NO_DISP_VAL
        self.setFixedWidth(width)

    def update(self, vmin, vmax, normalizer):
        self._vmin = vmin
        self._vmax = vmax
        self._normalizer = normalizer
        self.computeTicks()
        qt.QWidget.update(self)

    def setMargin(self, margin):
        """Define the margin to fit with a _ColorScale object.
        This is needed since we can only paint on the viewport of the widget

        :param int margin: the margin to apply on the top and bottom.
        """
        self.margin = margin

    def setTicksNumber(self, nticks):
        """Set the number of ticks to display.

        :param nticks: the number of tick to be display. Should be an
            unsigned int ot None. If None, let the :class:`_TickBar` find the
            optimal number of ticks from the tick density.
        """
        self._nticks = nticks
        self.computeTicks()
        qt.QWidget.update(self)

    def setTicksDensity(self, density):
        """If you let :class:`_TickBar` deal with the number of ticks
        (nticks=None) then you can specify a ticks density to be displayed.
        """
        if density < 0.0:
            raise ValueError('Density should be a positive value')
        self.ticksDensity = density

    def computeTicks(self):
        """This function compute ticks values labels. It is called at each
        update and each resize event.
        Deal only with linear and log scale.
        """
        nticks = self._nticks
        if nticks is None:
            nticks = self._getOptimalNbTicks()

        if self._vmin == self._vmax:
            # No range: no ticks
            self.ticks = ()
            self.subTicks = ()
        elif isinstance(self._normalizer, colors._LogarithmicNormalization):
            self._computeTicksLog(nticks)
        else:  # Fallback: use linear
            self._computeTicksLin(nticks)

        # update the form
        font = qt.QFont()
        font.setPixelSize(_TickBar._FONT_SIZE)

        self.form = self._getFormat(font)

    def _computeTicksLog(self, nticks):
        logMin = numpy.log10(self._vmin)
        logMax = numpy.log10(self._vmax)
        lowBound, highBound, spacing, self._nfrac = ticklayout.niceNumbersForLog10(logMin,
                                                                                   logMax,
                                                                                   nticks)
        self.ticks = numpy.power(10., numpy.arange(lowBound, highBound, spacing))
        if spacing == 1:
            self.subTicks = ticklayout.computeLogSubTicks(ticks=self.ticks,
                                                          lowBound=numpy.power(10., lowBound),
                                                          highBound=numpy.power(10., highBound))
        else:
            self.subTicks = []

    def resizeEvent(self, event):
        qt.QWidget.resizeEvent(self, event)
        self.computeTicks()

    def _computeTicksLin(self, nticks):
        _min, _max, _spacing, self._nfrac = ticklayout.niceNumbers(self._vmin,
                                                                   self._vmax,
                                                                   nticks)

        self.ticks = numpy.arange(_min, _max, _spacing)
        self.subTicks = []

    def _getOptimalNbTicks(self):
        return max(2, int(round(self.ticksDensity * self.rect().height())))

    def paintEvent(self, event):
        painter = qt.QPainter(self)
        font = painter.font()
        font.setPixelSize(_TickBar._FONT_SIZE)
        painter.setFont(font)

        # paint ticks
        for val in self.ticks:
            self._paintTick(val, painter, majorTick=True)

        # paint subticks
        for val in self.subTicks:
            self._paintTick(val, painter, majorTick=False)

    def _getRelativePosition(self, val):
        """Return the relative position of val according to min and max value
        """
        if self._normalizer is None:
            return 0.
        normMin, normMax, normVal = self._normalizer.apply(
            [self._vmin, self._vmax, val],
            self._vmin,
            self._vmax)

        if normMin == normMax:
            return 0.
        else:
            return 1. - (normVal - normMin) / (normMax - normMin)

    def _paintTick(self, val, painter, majorTick=True):
        """

        :param bool majorTick: if False will never draw text and will set a line
            with a smaller width
        """
        fm = qt.QFontMetrics(painter.font())
        viewportHeight = self.rect().height() - self.margin * 2 - 1
        relativePos = self._getRelativePosition(val)
        height = int(viewportHeight * relativePos + self.margin)
        lineWidth = _TickBar._LINE_WIDTH
        if majorTick is False:
            lineWidth /= 2

        painter.drawLine(qt.QLine(int(self.width() - lineWidth),
                                  height,
                                  self.width(),
                                  height))

        if self.displayValues and majorTick is True:
            painter.drawText(qt.QPoint(0, int(height + fm.height() / 2)),
                             self.form.format(val))

    def setDisplayType(self, disType):
        """Set the type of display we want to set for ticks labels

        :param str disType: The type of display we want to set. disType values
            can be :

            - 'std' for standard, meaning only a formatting on the number of
                digits is done
            - 'e' for scientific display
            - None to let the _TickBar guess the best display for this kind of data.
        """
        if disType not in (None, 'std', 'e'):
            raise ValueError("display type not recognized, value should be in (None, 'std', 'e'")
        self._forcedDisplayType = disType

    def _getStandardFormat(self):
        return "{0:.%sf}" % self._nfrac

    def _getFormat(self, font):
        if self._forcedDisplayType is None:
            return self._guessType(font)
        elif self._forcedDisplayType == 'std':
            return self._getStandardFormat()
        elif self._forcedDisplayType == 'e':
            return self._getScientificForm()
        else:
            err = 'Forced type for display %s is not recognized' % self._forcedDisplayType
            raise ValueError(err)

    def _getScientificForm(self):
        return "{0:.0e}"

    def _guessType(self, font):
        """Try fo find the better format to display the tick's labels

        :param QFont font: the font we want to use during the painting
        """
        form = self._getStandardFormat()

        fm = qt.QFontMetrics(font)
        width = 0
        for tick in self.ticks:
            width = max(fm.boundingRect(form.format(tick)).width(), width)

        # if the length of the string are too long we are moving to scientific
        # display
        if width > _TickBar._WIDTH_DISP_VAL - _TickBar._LINE_WIDTH:
            return self._getScientificForm()
        else:
            return form
