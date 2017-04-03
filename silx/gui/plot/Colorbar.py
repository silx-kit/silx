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
"""A widget displaying a colorbar linked to a :class:`PlotWidget`.

It is a wrapper over matplotlib :class:`ColorbarBase`.

It uses a description of colormaps as dict compatible with :class:`Plot`.

To run the following sample code, a QApplication must be initialized.

>>> import numpy
>>> from silx.gui.plot import Plot2D
>>> from silx.gui.plot.Colorbar import ColorbarWidget

>>> plot = Plot2D()  # Create a plot widget
>>> plot.show()

>>> colorbar = ColorbarWidget(plot=plot)  # Associate the colorbar with it
>>> colorbar._setLabel('Colormap')
>>> colorbar.show()
"""

__authors__ = ["H. Payno", "T. Vincent"]
__license__ = "MIT"
__date__ = "10/03/2017"


import logging
import numpy
from silx.gui.plot import PlotWidget
from ._utils import ticklayout


from .. import qt
from silx.gui.plot import Colors

_logger = logging.getLogger(__name__)


class ColorbarWidget(qt.QWidget):
    """Colorbar widget displaying a colormap
    """
    configuration=('withTicksValue', 'minMaxValueOnly')

    def __init__(self, parent=None, plot=None, legend=None, hideNorm=True, 
        hideAutoscale=True, config=configuration[0]):
        """

        :param parent: See :class:`QWidget`
        :param plot: PlotWidget the colorbar is attached to (optional)
        :param str legend: the label to set to the colormap
        :param bool hideNorm: if True hide the normalization groupbox (optional)
        :param bool hideAutoscale: if True hide the autoscale checkbox (optional)
        """
        super(ColorbarWidget, self).__init__(parent)
        self._plot = plot
        self._configuration = config
        self._label = ''  # Text label to display
        self.hideNorm = hideNorm
        self.hideAutoscale = hideAutoscale

        self.__buildGUI()
        if legend is not None:
            assert(type(legend) is str)
            self.setLegend(legend)

    def __buildGUI(self):
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        self.layout().addWidget(self.__buildMainColorMap())
        self.layout().addWidget(self.__buildAutoscale())
        self.layout().addWidget(self.__buildNorm())

        if self.hideNorm is True:
            self._groupNorm.hide()
        if self.hideAutoscale is True:
            self._autoscaleCB.hide()

        if self._plot is not None:
            self._plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged(
                None, self._plot.getActiveImage(just_legend=True))

        self.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Expanding)
        self.layout().setContentsMargins(0, 0, 0, 0)

    def __buildMainColorMap(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QVBoxLayout())
        widget.layout().addWidget(self.__buildGradationAndLegend())
        return widget

    def __buildNorm(self):
        # group definition
        self._groupNorm = qt.QGroupBox('Normalization', parent=self)
        self._groupNorm.setLayout(qt.QHBoxLayout())
        self._groupNorm.setEnabled(False)
        # adding linear option
        self._linearNorm = qt.QRadioButton('linear', self._groupNorm)
        self._groupNorm.layout().addWidget(self._linearNorm)
        # adding lof option
        self._logNorm = qt.QRadioButton('log', self._groupNorm)
        self._groupNorm.layout().addWidget(self._logNorm)

        return self._groupNorm

    def __buildAutoscale(self):
        self._autoscaleCB = qt.QCheckBox('autoscale', parent=self)
        self._autoscaleCB.setEnabled(False)
        return self._autoscaleCB
        
    def __buildGradationAndLegend(self):
        if self._configuration is ColorbarWidget.configuration[0]:
            return self.__buildGradationAndLegendWithTicksValue()
        if self._configuration is ColorbarWidget.configuration[1]:
            return self.__buildGradationAndLegendMinMax()

        msg = 'Given configuration is not recognize, can\'t create Gradation'
        raise ValueError(msg)

    def __buildGradationAndLegendWithTicksValue(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QHBoxLayout())
        widget.layout().setContentsMargins(0, 0, 0, 0)
        # create gradation
        self._gradation = GradationBar(parent=widget,
                                       colormap=self._plot.getDefaultColormap())
        widget.layout().addWidget(self._gradation)

        self.legend = VerticalLegend('', self)
        widget.layout().addWidget(self.legend)

        widget.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Preferred)
        return widget

    def __buildGradationAndLegendMinMax(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QHBoxLayout())

        widgetLeftGroup = qt.QWidget(widget)
        widgetLeftGroup.setLayout(qt.QVBoxLayout())
        widget.layout().addWidget(widgetLeftGroup)

        # min label
        self.__minLabel = qt.QLabel(str(0.0), parent=self)
        self.__minLabel.setAlignment(qt.Qt.AlignHCenter)
        self.__minLabel.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        widgetLeftGroup.layout().addWidget(self.__minLabel)

        # create gradation widget
        self._gradation = GradationBar(parent=widget, 
                                       colormap=self._plot.getDefaultColormap(),
                                       displayTicksValues=False)
        widgetLeftGroup.layout().addWidget(self._gradation)

        # max label
        self.__maxLabel = qt.QLabel(str(1.0), parent=self)
        self.__maxLabel.setAlignment(qt.Qt.AlignHCenter)
        self.__maxLabel.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        widgetLeftGroup.layout().addWidget(self.__maxLabel)

        # legend (is the right group)
        self.legend = VerticalLegend('', self)
        widget.layout().addWidget(self.legend)

        widget.layout().setContentsMargins(0, 0, 0, 0)
        widget.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Preferred)
        return widget

    def getColormap(self):
        """Return the colormap displayed in the colorbar as a dict.

        It returns None if no colormap is set.
        See :class:`Plot` documentation for the description of the colormap
        dict description.
        """
        return self._colormap.copy()

    def setColormap(self, name, normalization='linear',
                    vmin=0., vmax=1., colors=None, autoscale=False):
        """Set the colormap to display in the colorbar.

        :param str name: The name of the colormap or None
        :param str normalization: Normalization to use: 'linear' or 'log'
        :param float vmin: The value to bind to the beginning of the colormap
        :param float vmax: The value to bind to the end of the colormap
        :param colors: Array of RGB(A) colors to use as colormap
        :type colors: numpy.ndarray
        """
        if name is None and colors is None:
            self._colormap = None
            return

        if normalization == 'linear':
            self._setLinearNorm()
        elif normalization == 'log':
            self._setLogNorm()
            if vmin <= 0 or vmax <= 0:
                _logger.warning(
                    'Log colormap with bound <= 0: changing bounds.')
                print('SETTING a lof scale !!!!!!! ')
                vmin, vmax = 1., 10.
            pass
        else:
            raise ValueError('Wrong normalization %s' % normalization)

        self._setAutoscale(autoscale)

        self._colormap = {'name': name,
                          'normalization': normalization,
                          'autoscale': autoscale,
                          'vmin': vmin,
                          'vmax': vmax,
                          'colors': colors}

        print('will change colormap for ')
        print(self._colormap)

        self._gradation.setColormap(self._colormap)

    def _setLogNorm(self):
        self._logNorm.setChecked(True)

    def _setLinearNorm(self):
        self._linearNorm.setChecked(True)

    def _setAutoscale(self, b):
        self._autoscaleCB.setChecked(b)

    def setLegend(self, legend):
        """Set the legend displayed along the colorbar

        :param str legend: The label
        """
        self.legend.setText(legend)

    def getLegend(self):
        """

        :return: return the legend displayed along the colorbar
        :rtype: str 
        """
        return self.legend.getText()

    def _activeImageChanged(self, previous, legend):
        """Handle plot active curve changed"""
        if legend is None:  # No active image, display default colormap
            self._syncWithDefaultColormap()
            return

        # Sync with active image
        image = self._plot.getActiveImage().getData()

        # RGB(A) image, display default colormap
        if image.ndim != 2:
            self._syncWithDefaultColormap()
            return

        # data image, sync with image colormap
        cmap = self._plot.getActiveImage().getColormap()
        if cmap['autoscale']:
            if cmap['normalization'] == 'log':
                data = image[
                    numpy.logical_and(image > 0, numpy.isfinite(image))]
            else:
                data = image[numpy.isfinite(image)]
            vmin, vmax = data.min(), data.max()
        else:  # No autoscale
            vmin, vmax = cmap['vmin'], cmap['vmax']

        self.setColormap(name=cmap['name'],
                         normalization=cmap['normalization'],
                         vmin=vmin,
                         vmax=vmax,
                         colors=cmap.get('colors', None))

    def _defaultColormapChanged(self):
        """Handle plot default colormap changed"""
        if self._plot.getActiveImage() is None:
            # No active image, take default colormap update into account
            self._syncWithDefaultColormap()

    def _syncWithDefaultColormap(self):
        """Update colorbar according to plot default colormap"""
        cmap = self._plot.getDefaultColormap()
        if cmap['autoscale']:  # Makes sure range is OK
            vmin, vmax = 1., 10.
        else:
            vmin, vmax = cmap['vmin'], cmap['vmax']

        self.setColormap(name=cmap['name'],
                         normalization=cmap['normalization'],
                         vmin=vmin,
                         vmax=vmax,
                         colors=cmap.get('colors', None))


class VerticalLegend(qt.QLabel):
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

    def paintEvent(self, event ):
        painter = qt.QPainter(self)
        painter.setFont(self.font())

        painter.translate(0, self.rect().height())
        painter.rotate(270)
        newRect = qt.QRect(0, 0, self.rect().height(), self.rect().width())
        # painter.drawText(self.rect(),
        painter.drawText(newRect,
                         qt.Qt.AlignHCenter,self.text())

        fm = qt.QFontMetrics(self.font())
        preferedHeight = fm.width(self.text())
        preferedWidth = fm.height()
        self.setFixedWidth(preferedWidth)
        self.setMinimumHeight(preferedHeight)

class GradationBar(qt.QWidget):
    """The object grouping the Gradation and ticks associated to the Gradation
    """
    def __init__(self, colormap, parent=None, displayTicksValues=True):
        """

        :param colormap: the colormap to be displayed
        :param parent: the Qt parent if any
        :param displayTicksValues: display the ticks value or only the '-'
        """
        super(GradationBar, self).__init__(parent)

        self.setLayout(qt.QHBoxLayout())
        self.textMargin = 5

        # create the left side group (Gradation)
        self.gradation = Gradation(colormap=colormap, parent=self)
        self.tickbar = TickBar(vmin=colormap['vmin'],
                               vmax=colormap['vmax'],
                               norm=colormap['normalization'],
                               parent=self,
                               displayValues=displayTicksValues)

        self.gradation.setMargin(self.textMargin)
        self.tickbar.setMargin(self.textMargin)
        self.layout().addWidget(self.tickbar)
        self.layout().addWidget(self.gradation)

    def getTickBar(self):
        """

        :return: :class:`TickBar`
        """
        return self.tickbar

    def getGradation(self):
        """

        :return: :class:`Gradation`
        """
        return self.gradation

    def setColormap(self, colormap):
        self.gradation.setColormap(colormap)
        self.tickbar._norm = colormap['normalization']
        self.tickbar.vmin = colormap['vmin']
        self.tickbar.vmax = colormap['vmax']


class Gradation(qt.QWidget):
    """Simple widget wich display the colormap gradation and update the tooltip
    to return the value equivalence for the color
    """
    def __init__(self, colormap, parent=None):
        """

        :param colormap: the colormap to be displayed
        :param parent: the Qt parent if any
        """
        print('building')
        qt.QWidget.__init__(self, parent)
        if colormap is not None:
            self.setColormap(colormap)

        self.setLayout(qt.QVBoxLayout())
        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        # needed to get the mouse event without waiting for button click
        self.setMouseTracking(True)
        self.setMargin(0)
        self.setContentsMargins(0, 0, 0, 0)
        print('end building')


    def setColormap(self, colormap):
        """Create a _MyColorMap elemtent from the given silx colormap.
        In the future the _MyColorMap should be removed
        """
        self.colormap = _MyColorMap(colormap)
        print("setting colormap")
        print(colormap)

    def paintEvent(self, event):
        
        qt.QWidget.paintEvent(self, event)

        painter = qt.QPainter(self)
        gradient = qt.QLinearGradient(0, 0, 0, self.rect().height() - 2*self.margin);
        steps = (self.colormap.vmax - self.colormap.vmin)/256
        for pt in numpy.arange(self.colormap.vmin, self.colormap.vmax, steps):
            colormapPosition = (pt-self.colormap.vmin) / (self.colormap.vmax-self.colormap.vmin)
            assert(colormapPosition >= 0.0 )
            assert(colormapPosition <= 1.0 )

            qtGradientPosition = pt / self.colormap.vmax
            # self.colormap.vmax - position because Qt is in the top-bottom reference
            gradient.setColorAt( qtGradientPosition, self.colormap.getColor(pt))

        painter.setBrush(gradient)
        painter.drawRect(
            qt.QRect(0, self.margin, self.width(), self.height() - 2.*self.margin))

    def mouseMoveEvent(self, event):
        self.setToolTip(str(self.getValueFromRelativePosition(self._getRelativePosition(event.y()))))
        super(Gradation, self).mouseMoveEvent(event)

    def _getRelativePosition(self, yPixel):
        """yPixel : pixel position into Gradation widget reference
        """
        # widgets are bottom-top referencial but we display in top-bottom referential
        return float(yPixel)/float(self.height())

    def getValueFromRelativePosition(self, value):
        """Return the value in the colorMap from a relative position in the 
        GradationBar (y)

        :param val: float value in [0, 1]
        :return: the value in [colormap['vmin'], colormap['vmax']]
        """
        print("vmin is %s"%self.colormap.vmin)
        print("vmax is %s"%self.colormap.vmax)
        if not ((value >=0) and (value <=1)):
            raise ValueError('invalid value given, should be in [0.0, 1.0]')
        if self.colormap.normalization is 'linear':
            return self.colormap.vmin + (self.colormap.vmax-self.colormap.vmin)*value
        elif self.colormap.normalization is 'log':
            rpos = (numpy.log10(self.colormap.vmax)-numpy.log10(self.colormap.vmin))*value
            return self.colormap.vmin + numpy.exp(rpos)
        else:
            err = "normalization type (%s) is not managed by the Gradation Widget"%self.colormap.normalization
            raise ValueError(err)


    def setMargin(self, margin):
        """Define the margin to fit with a TickBar object.
        This is needed since we can only paint on the viewport of the widget.
        Didn't work with a simple setContentsMargins

        :param int margin: the margin to apply on the top and bottom.
        """
        self.margin = margin


class TickBar(qt.QWidget):
    _widthDisplayVal = 45
    """widget width when displayed with ticks labels"""
    _widthNoDisplayVal = 10
    """widget width when displayed without ticks labels"""
    _fontSize = 10
    """font size for ticks labels"""
    _lineWidth = 10
    """width of the line to mark a tickk"""

    def __init__(self, vmin, vmax, norm, parent=None, displayValues=True):
        """Bar grouping the tickes displayed

        :param vmin: minimal value on the colormap
        :param vmax: maximal value on the colormap
        :param str norm: the normalization of the colormap
        :param parent: the Qt parent if any
        :param displayValues: if True display the values close to the tick,
            Otherwise only signal it by '-'
        """
        super(TickBar, self).__init__(parent)
        self._forcedDisplayType = None
        self.displayValues = displayValues
        self.nticks = 5
        self._norm = norm
        self.vmin = vmin
        self.vmax = vmax
        self.setLayout(qt.QVBoxLayout())
        self.setMargin(0)
        self.setContentsMargins(0, 0, 0, 0)

        self.width = self._widthDisplayVal if self.displayValues else self._widthNoDisplayVal
        self.setFixedWidth(self.width)

    def setMargin(self, margin):
        """Define the margin to fit with a Gradation object.
        This is needed since we can only paint on the viewport of the widget

        :param int margin: the margin to apply on the top and bottom.
        """
        self.margin = margin

    def _computeTicks(self):
        'Deal only with linear and log for now'
        if self._norm == 'log':
            return self._computeTicksLog()
        elif self._norm == 'linear':
            return self._computeTicksLin()
        else:
            err = 'TickBar - Wrong normalization %s'%normalization
            raise ValueError(err)

    def _computeTicksLog(self):
        self.tickMin, self.tickMax, self.step, self.nfrac = ticklayout.niceNumbersForLog10(numpy.log10(self.vmin),
                                                                                           numpy.log10(self.vmax), 
                                                                                           self.nticks)
        _dtype = numpy.int64 if type(self.vmin) in (int, numpy.integer) else numpy.float64
        self.ticks = numpy.linspace(self.tickMin, self.tickMax, self.nticks).astype(_dtype)
        # self.ticks = numpy.exp(self.ticks).astype(_dtype)
        self.ticks = 10**self.ticks
        return self.ticks

    def _computeTicksLin(self):
        self.tickMin, self.tickMax, self.step, self.nfrac = ticklayout.niceNumbers(self.vmin,
                                                                                   self.vmax, 
                                                                                   self.nticks)
        _dtype = numpy.int64 if type(self.vmin) in (int, numpy.integer) else numpy.float64
        self.ticks = numpy.linspace(self.tickMin, self.tickMax, self.nticks).astype(_dtype)
        return self.ticks

    def paintEvent(self, event):
        self._computeTicks()
        painter = qt.QPainter(self)
        font = painter.font()
        font.setPixelSize(self._fontSize)
        fm = qt.QFontMetrics(font)
        painter.setFont(font)

        viewportHeight = self.rect().height() - self.margin * 2
        form = self._getFormat(font)
        for iTick, val in enumerate(self.ticks):
            height = (viewportHeight * iTick) / (self.nticks -1)
            height += self.margin
            painter.drawLine(qt.QLine(self.width - self._lineWidth,
                                      height,
                                      self.width,
                                      height))
            if self.displayValues:
                painter.drawText(qt.QPoint(0.0, height + (fm.height() / 2)),
                                 form.format(val));
        qt.QWidget.paintEvent(self, event)

    def setDisplayType(self, disType):
        """Set the type of display we want to set for ticks labels

        :param str disType: The type of display we want to set. disType values
            can be : 
            - 'std' for standard, meaning only a formatting on the number of
                digits is done
            - 'e' for scientific display
            - None to let the TickBar guess the best display for this kind of data.
        """
        if not disType in (None, 'std', 'e'):
            raise ValueError("display type not recognized, value should be in (None, 'std', 'e'")
        self._forcedDisplayType = disType

    def _getStandardFormat(self, val):
        if type(val) in (int, numpy.integer):
            return "{0:d}"
        elif type(val) in (numpy.inexact, float, long):
            return "{0:.%sf}"%self.nfrac
        else:
            err = "type %s is not managed by the TickBar"
            raise ValueError(err)

    def _getFormat(self, font):
        if self._forcedDisplayType is None:
            return self._guessType(font)
        elif self._forcedDisplayType is 'std':
            return self._getStandardFormat()
        elif self._forcedDisplayType is 'e':
            return self._getScientificForm()
        else:
            err = 'Forced type for display %s is not recognized'%self._forcedDisplayType
            raise ValueError(err)

    def _getScientificForm(self):
        return "{0:.%se}"%self.nfrac

    def _guessType(self, font):
        """Try fo find the better format to display the tick's labels

        :param QFont font: the font we want want to use durint the painting
        """
        assert(type(self.vmin) == type(self.vmax))
        form = self._getStandardFormat(self.vmin)
        
        fm = qt.QFontMetrics(font)
        width = max(fm.width(form.format(self.vmin)), fm.width(form.format(self.vmax)))
        painter = None
        # if the length of the string are too long we are mooving to scientific
        # display
        if width > self._widthDisplayVal - self._lineWidth:
            return self._getScientificForm()
        else:
            return form


class _MyColorMap(object):
    """
    Temporaty object, will be removed soon
    This widget is using matplotlib.
    """
    def __init__(self, colormap):
        self.name = colormap['name']
        self.normalization = colormap['normalization']
        self.autoscale = colormap['autoscale']
        self.vmin = colormap['vmin']
        self.vmax = colormap['vmax']

        # for now only deal with matplotlib colorbar
        from silx.gui.plot import Colors
        cmap = Colors.getMPLColormap(self.name)
        import matplotlib.cm
        norm = matplotlib.colors.Normalize(self.vmin, self.vmax)
        self.scalarMappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    def getColor(self, val):
        color = self.scalarMappable.to_rgba(val)
        return qt.QColor.fromRgbF(*color)
