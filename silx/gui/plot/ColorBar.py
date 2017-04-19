# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__date__ = "11/04/2017"


import logging
import numpy
from ._utils import ticklayout
from ._utils import clipColormapLogRange


from .. import qt
from silx.gui.plot import Colors

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
    :param str legend: the label to set to the colormap
    """

    def __init__(self, parent=None, plot=None, legend=None):
        super(ColorBarWidget, self).__init__(parent)
        self._plot = None

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
        self.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def getPlot(self):
        """Returns the :class:`Plot` associated to this widget or None"""
        return self._plot

    def setPlot(self, plot):
        """Associate a plot to the ColorBar

        :param plot: the plot to associate with the colorbar. If None will remove
            any connection with a previous plot.
        """
        # removing previous plot if any
        if self._plot is not None:
            self._plot.sigActiveImageChanged.disconnect(self._activeImageChanged)

        # setting the new plot
        self._plot = plot
        if self._plot is not None:
            self._plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged(self._plot.getActiveImage(just_legend=True))

    def getColormap(self):
        """Return the colormap displayed in the colorbar as a dict.

        It returns None if no colormap is set.
        See :class:`silx.gui.plot.Plot` documentation for the description of the colormap
        dict description.
        """
        return self._colormap.copy()

    def setColormap(self, colormap):
        """Set the colormap to be displayed.

        :param dict colormap: The colormap to apply on the ColorBarWidget
        """
        self._colormap = colormap
        if self._colormap is None:
            return

        if self._colormap['normalization'] not in ('log', 'linear'):
            raise ValueError('Wrong normalization %s' % self._colormap['normalization'])

        if self._colormap['normalization'] is 'log':
            if self._colormap['vmin'] < 1. or self._colormap['vmax'] < 1.:
                _logger.warning('Log colormap with bound <= 1: changing bounds.')
            clipColormapLogRange(colormap)

        self.getColorScaleBar().setColormap(self._colormap)

    def setLegend(self, legend):
        """Set the legend displayed along the colorbar

        :param str legend: The label
        """
        if legend is None or legend == "":
            self.legend.hide()
            self.legend.setText("")
        else:
            assert(type(legend) is str)
            self.legend.show()
            self.legend.setText(legend)

    def getLegend(self):
        """
        Returns the legend displayed along the colorbar

        :return: return the legend displayed along the colorbar
        :rtype: str
        """
        return self.legend.getText()

    def _activeImageChanged(self, legend):
        """Handle plot active curve changed"""
        if legend is None:  # No active image, display default colormap
            self._syncWithDefaultColormap()
            return

        # Sync with active image
        image = self._plot.getActiveImage().getData(copy=False)

        # RGB(A) image, display default colormap
        if image.ndim != 2:
            self._syncWithDefaultColormap()
            return

        # data image, sync with image colormap
        # do we need the copy here : used in the case we are changing
        # vmin and vmax but should have already be done by the plot
        cmap = self._plot.getActiveImage().getColormap().copy()
        if cmap['autoscale']:
            if cmap['normalization'] == 'log':
                data = image[
                    numpy.logical_and(image > 0, numpy.isfinite(image))]
            else:
                data = image[numpy.isfinite(image)]
            cmap['vmin'], cmap['vmax'] = data.min(), data.max()

        self.setColormap(cmap)

    def _defaultColormapChanged(self):
        """Handle plot default colormap changed"""
        if self._plot.getActiveImage() is None:
            # No active image, take default colormap update into account
            self._syncWithDefaultColormap()

    def _syncWithDefaultColormap(self):
        """Update colorbar according to plot default colormap"""
        self.setColormap(self._plot.getDefaultColormap())

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

    >>> colormap={'name':'gray',
    ...       'normalization':'log',
    ...       'vmin':1,
    ...       'vmax':100000,
    ...       'autoscale':False
    ...       }
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

    _MIN_LIM_SCI_FORM = -1000
    """Used for the min and max label to know when we should display it under
    the scientific form"""

    _MAX_LIM_SCI_FORM = 1000
    """Used for the min and max label to know when we should display it under
    the scientific form"""

    def __init__(self, parent=None, colormap=None, displayTicksValues=True):
        super(ColorScaleBar, self).__init__(parent)

        self.minVal = None
        """Value set to the _minLabel"""
        self.maxVal = None
        """Value set to the _maxLabel"""

        self.setLayout(qt.QGridLayout())

        # create the left side group (ColorScale)
        self.colorScale = _ColorScale(colormap=colormap,
                                     parent=self,
                                     margin=ColorScaleBar._TEXT_MARGIN)

        self.tickbar = _TickBar(vmin=colormap['vmin'] if colormap else 0.0,
                               vmax=colormap['vmax'] if colormap else 1.0,
                               norm=colormap['normalization'] if colormap else 'linear',
                               parent=self,
                               displayValues=displayTicksValues,
                               margin=ColorScaleBar._TEXT_MARGIN)

        self.layout().addWidget(self.tickbar, 1, 0)
        self.layout().addWidget(self.colorScale, 1, 1)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # max label
        self._maxLabel = qt.QLabel(str(1.0), parent=self)
        self._maxLabel.setAlignment(qt.Qt.AlignHCenter)
        self._maxLabel.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        self.layout().addWidget(self._maxLabel, 0, 1)

        # min label
        self._minLabel = qt.QLabel(str(0.0), parent=self)
        self._minLabel.setAlignment(qt.Qt.AlignHCenter)
        self._minLabel.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        self.layout().addWidget(self._minLabel, 2, 1)

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

    def setColormap(self, colormap):
        """Set the new colormap to be displayed

        :param dict colormap: the colormap to set
        """
        if colormap is not None:
            self.colorScale.setColormap(colormap)

            self.tickbar.update(vmin=colormap['vmin'],
                                vmax=colormap['vmax'],
                                norm=colormap['normalization'])

            self._setMinMaxLabels(colormap['vmin'], colormap['vmax'])

    def setMinMaxVisible(self, val=True):
        """Change visibility of the min label and the max label

        :param val: if True, set the labels visible, otherwise set it not visible
        """
        self._maxLabel.show() if val is True else self._maxLabel.hide()
        self._minLabel.show() if val is True else self._minLabel.hide()

    def _updateMinMax(self):
        """Update the min and max label if we are in the case of the
        configuration 'minMaxValueOnly'"""
        if self._minLabel is not None and self._maxLabel is not None:
            if self.minVal is not None:
                if ColorScaleBar._MIN_LIM_SCI_FORM <= self.minVal <= ColorScaleBar._MAX_LIM_SCI_FORM:
                    self._minLabel.setText(str(self.minVal))
                else:
                    self._minLabel.setText("{0:.0e}".format(self.minVal))
            if self.maxVal is not None:
                if ColorScaleBar._MIN_LIM_SCI_FORM <= self.maxVal <= ColorScaleBar._MAX_LIM_SCI_FORM:
                    self._maxLabel.setText(str(self.maxVal))
                else:
                    self._maxLabel.setText("{0:.0e}".format(self.maxVal))

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

    >>> colormap={'name':'viridis',
    ...       'normalization':'log',
    ...       'vmin':1,
    ...       'vmax':100000,
    ...       'autoscale':False
    ...       }
    >>> colorscale = ColorScale(parent=None,
    ...                         colormap=colormap)
    >>> colorscale.show()

    Initializer parameters :

    :param colormap: the colormap to be displayed
    :param parent: the Qt parent if any
    :param int margin: the top and left margin to apply.

    .. warning:: Value drawing will be
        done at the center of ticks. So if no margin is done your values
        drawing might not be fully done for extrems values.
    """

    _NB_CONTROL_POINTS = 256

    def __init__(self, colormap, parent=None, margin=5):
        qt.QWidget.__init__(self, parent)
        self.colormap = None
        self.setColormap(colormap)

        self.setLayout(qt.QVBoxLayout())
        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        # needed to get the mouse event without waiting for button click
        self.setMouseTracking(True)
        self.setMargin(margin)
        self.setContentsMargins(0, 0, 0, 0)

    def setColormap(self, colormap):
        """Set the new colormap to be displayed

        :param dict colormap: the colormap to set
        """
        if colormap is None:
            return

        if colormap['normalization'] not in ('log', 'linear'):
            raise ValueError("Unrecognized normalization, should be 'linear' or 'log'")

        if colormap['normalization'] is 'log':
            if not (colormap['vmin'] > 0 and colormap['vmax'] > 0):
                raise ValueError('vmin and vmax should be positives')
        self.colormap = colormap
        self._computeColorPoints()

    def _computeColorPoints(self):
        """Compute the color points for the gradient
        """
        if self.colormap is None:
            return

        vmin = self.colormap['vmin']
        vmax = self.colormap['vmax']
        steps = (vmax - vmin)/float(_ColorScale._NB_CONTROL_POINTS)
        self.ctrPoints = numpy.arange(vmin, vmax, steps)
        self.colorsCtrPts = Colors.applyColormapToData(self.ctrPoints,
                                                       name=self.colormap['name'],
                                                       normalization='linear',
                                                       autoscale=self.colormap['autoscale'],
                                                       vmin=vmin,
                                                       vmax=vmax)

    def paintEvent(self, event):
        """"""
        qt.QWidget.paintEvent(self, event)
        if self.colormap is None:
            return

        vmin = self.colormap['vmin']
        vmax = self.colormap['vmax']

        painter = qt.QPainter(self)
        gradient = qt.QLinearGradient(0, 0, 0, self.rect().height() - 2*self.margin)
        for iPt, pt in enumerate(self.ctrPoints):
            colormapPosition = 1 - (pt-vmin) / (vmax-vmin)
            assert(colormapPosition >= 0.0)
            assert(colormapPosition <= 1.0)
            gradient.setColorAt(colormapPosition, qt.QColor(*(self.colorsCtrPts[iPt])))

        painter.setBrush(gradient)
        painter.drawRect(
            qt.QRect(0, self.margin, self.width(), self.height() - 2.*self.margin))

    def mouseMoveEvent(self, event):
        """"""
        self.setToolTip(str(self.getValueFromRelativePosition(self._getRelativePosition(event.y()))))
        super(_ColorScale, self).mouseMoveEvent(event)

    def _getRelativePosition(self, yPixel):
        """yPixel : pixel position into _ColorScale widget reference
        """
        # widgets are bottom-top referencial but we display in top-bottom referential
        return 1 - float(yPixel)/float(self.height() - 2*self.margin)

    def getValueFromRelativePosition(self, value):
        """Return the value in the colorMap from a relative position in the
        ColorScaleBar (y)

        :param value: float value in [0, 1]
        :return: the value in [colormap['vmin'], colormap['vmax']]
        """
        value = max(0.0, value)
        value = min(value, 1.0)
        vmin = self.colormap['vmin']
        vmax = self.colormap['vmax']
        if self.colormap['normalization'] is 'linear':
            return vmin + (vmax - vmin) * value
        elif self.colormap['normalization'] is 'log':
            rpos = (numpy.log10(vmax) - numpy.log10(vmin)) * value + numpy.log10(vmin)
            return numpy.power(10., rpos)
        else:
            err = "normalization type (%s) is not managed by the _ColorScale Widget" % self.colormap['normalization']
            raise ValueError(err)

    def setMargin(self, margin):
        """Define the margin to fit with a TickBar object.
        This is needed since we can only paint on the viewport of the widget.
        Didn't work with a simple setContentsMargins

        :param int margin: the margin to apply on the top and bottom.
        """
        self.margin = margin


class _TickBar(qt.QWidget):
    """Bar grouping the ticks displayed

    To run the following sample code, a QApplication must be initialized.

    >>> bar = TickBar(1, 1000, norm='log', parent=None, displayValues=True)
    >>> bar.show()

    .. image:: img/tickbar.png
        :width: 40px
        :align: center

    :param int vmin: smaller value of the range of values
    :param int vmax: higher value of the range of values
    :param str norm: normalization type to be displayed. Valid values are
        'linear' and 'log'
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

    def __init__(self, vmin, vmax, norm, parent=None, displayValues=True,
                 nticks=None, margin=5):
        super(_TickBar, self).__init__(parent)
        self._forcedDisplayType = None
        self.ticksDensity = _TickBar.DEFAULT_TICK_DENSITY

        self._vmin = vmin
        self._vmax = vmax
        # TODO : should be grouped into a global function, called by all
        # logScale displayer to make sure we have the same behavior everywhere
        if self._vmin < 1. or self._vmax < 1.:
            _logger.warning(
                'Log colormap with bound <= 1: changing bounds.')
            self._vmin, self._vmax = 1., 10.

        self._norm = norm
        self.displayValues = displayValues
        self.setTicksNumber(nticks)
        self.setMargin(margin)

        self.setLayout(qt.QVBoxLayout())
        self.setMargin(margin)
        self.setContentsMargins(0, 0, 0, 0)

        self._resetWidth()

    def setTicksValuesVisible(self, val):
        self.displayValues = val
        self._resetWidth()

    def _resetWidth(self):
        self.width = _TickBar._WIDTH_DISP_VAL if self.displayValues else _TickBar._WIDTH_NO_DISP_VAL
        self.setFixedWidth(self.width)

    def update(self, vmin, vmax, norm):
        self._vmin = vmin
        self._vmax = vmax
        self._norm = norm
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
        self.ticks = None
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

        if self._norm == 'log':
            self._computeTicksLog(nticks)
        elif self._norm == 'linear':
            self._computeTicksLin(nticks)
        else:
            err = 'TickBar - Wrong normalization %s' % self._norm
            raise ValueError(err)
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
        if self.ticks is not None:
            for val in self.ticks:
                self._paintTick(val, painter, majorTick=True)

            # paint subticks
            for val in self.subTicks:
                self._paintTick(val, painter, majorTick=False)

        qt.QWidget.paintEvent(self, event)

    def _getRelativePosition(self, val):
        """Return the relative position of val according to min and max value
        """
        if self._norm == 'linear':
            return 1 - (val - self._vmin) / (self._vmax - self._vmin)
        elif self._norm == 'log':
            return 1 - (numpy.log10(val) - numpy.log10(self._vmin))/(numpy.log10(self._vmax) - numpy.log(self._vmin))
        else:
            raise ValueError('Norm is not recognized')

    def _paintTick(self, val, painter, majorTick=True):
        """

        :param bool majorTick: if False will never draw text and will set a line
            with a smaller width
        """
        fm = qt.QFontMetrics(painter.font())
        viewportHeight = self.rect().height() - self.margin * 2
        relativePos = self._getRelativePosition(val)
        height = viewportHeight * relativePos
        height += self.margin
        lineWidth = _TickBar._LINE_WIDTH
        if majorTick is False:
            lineWidth /= 2

        painter.drawLine(qt.QLine(self.width - lineWidth,
                                  height,
                                  self.width,
                                  height))

        if self.displayValues and majorTick is True:
            painter.drawText(qt.QPoint(0.0, height + (fm.height() / 2)),
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
        elif self._forcedDisplayType is 'std':
            return self._getStandardFormat()
        elif self._forcedDisplayType is 'e':
            return self._getScientificForm()
        else:
            err = 'Forced type for display %s is not recognized' % self._forcedDisplayType
            raise ValueError(err)

    def _getScientificForm(self):
        return "{0:.0e}"

    def _guessType(self, font):
        """Try fo find the better format to display the tick's labels

        :param QFont font: the font we want want to use durint the painting
        """
        assert(type(self._vmin) == type(self._vmax))
        form = self._getStandardFormat()

        fm = qt.QFontMetrics(font)
        width = 0
        for tick in self.ticks:
            width = max(fm.width(form.format(tick)), width)

        # if the length of the string are too long we are mooving to scientific
        # display
        if width > _TickBar._WIDTH_DISP_VAL - _TickBar._LINE_WIDTH:
            return self._getScientificForm()
        else:
            return form
