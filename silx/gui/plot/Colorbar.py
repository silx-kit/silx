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
>>> colorbar.setLabel('Colormap')
>>> colorbar.show()
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "18/10/2016"


import logging
import numpy
from silx.gui.plot import PlotWidget


from .. import qt
from silx.gui.plot import Colors

_logger = logging.getLogger(__name__)


# TODO:
# - Add a button to edit the colormap
# - Handle width and ticks labels
# - Store data min and max somewhere in common with plot instead of recomputing
# - Doc + tests
# - Add get/setOrientation?

class ColorbarWidget(qt.QWidget):
    """Colorbar widget displaying a colormap

    This widget is using matplotlib.

    :param parent: See :class:`QWidget`
    :param plot: PlotWidget the colorbar is attached to (optional)
    """

    def __init__(self, parent=None, plot=None):
        super(ColorbarWidget, self).__init__(parent)
        self._plot = plot
        self.setContentsMargins(0, 0, 0, 0);
        self.setEnabled(False)

        self.colorbar = None  # matplotlib colorbar this will be an object

        self._label = ''  # Text label to display

        self.setFixedWidth(150)
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        self.layout().addWidget(self.__buildMainColorMap())

        if self._plot is not None:
            self._plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged(
                None, self._plot.getActiveImage(just_legend=True))
            # self._plot.sigSetDefaultColormap.connect(
            #     self._defaultColormapChanged)


    def __buildMainColorMap(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QVBoxLayout())

        widget.layout().addWidget(self.__buildMin())
        widget.layout().addWidget(self.__buildGradationAndLegend())
        widget.layout().addWidget(self.__buildMax())

        return widget
        
    def __buildMin(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QHBoxLayout())
        self._minLabel = qt.QLabel('min', widget)
        widget.layout().addWidget(self._minLabel)
        widget.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Preferred)
        widget.setContentsMargins(0, 0, 0, 0);
        # TODO : reduce margin at least
        return widget

    def __buildMax(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QHBoxLayout())
        self._maxLabel = qt.QLabel('max', widget)
        widget.layout().addWidget(self._maxLabel)
        widget.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Preferred)
        widget.setContentsMargins(0, 0, 0, 0);
        # TODO : reduce margin at least
        return widget

    def __buildGradationAndLegend(self):
        widget = qt.QWidget(self)
        widget.setLayout(qt.QHBoxLayout())
        # create gradation
        self._gradation = GradationV2(parent=widget, colormap=self._plot.getDefaultColormap())
        widget.layout().addWidget(self._gradation)

        # create legend# TODO : center this
        self.legend = VerticalLegend('test 1 2 3 4 5 6 7 8', self)
        widget.layout().addWidget(self.legend)

        widget.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Preferred)
        widget.setContentsMargins(0, 0, 0, 0);
        return widget

    def getColormap(self):
        """Return the colormap displayed in the colorbar as a dict.

        It returns None if no colormap is set.
        See :class:`Plot` documentation for the description of the colormap
        dict description.
        """
        return self._colormap.copy()

    def setColormap(self, name, normalization='linear',
                    vmin=0., vmax=1., colors=None):
        """Set the colormap to display in the colorbar.

        :param str name: The name of the colormap or None
        :param str normalization: Normalization to use: 'linear' or 'log'
        :param float vmin: The value to bind to the beginning of the colormap
        :param float vmax: The value to bind to the end of the colormap
        :param colors: Array of RGB(A) colors to use as colormap
        :type colors: numpy.ndarray
        """
        print('colormap setted')
        if name is None and colors is None:
            self.colorbar = None
            self._colormap = None
            return

        if normalization == 'linear':
            pass
        elif normalization == 'log':
            if vmin <= 0 or vmax <= 0:
                _logger.warning(
                    'Log colormap with bound <= 0: changing bounds.')
                vmin, vmax = 1., 10.
            pass
        else:
            raise ValueError('Wrong normalization %s' % normalization)

        self.colorbar = None # TODO : get colormap

        self.legend.setText(self._label)
        # if normalization == 'linear':
        #     formatter = matplotlib.ticker.FormatStrFormatter('%.4g')
        #     self.colorbar.formatter = formatter
        #     self.colorbar.update_ticks()

        self._colormap = {'name': name,
                          'normalization': normalization,
                          'autoscale': False,
                          'vmin': vmin,
                          'vmax': vmax,
                          'colors': colors}

        self.setMinVal(vmin)
        self.setMaxVal(vmax)

    def setMinVal(self, val):
        self._minLabel.setText(str(val))

    def setMaxVal(self, val):
        self._maxLabel.setText(str(val))

    def getLabel(self):
        """Return the label of the colorbar (str)"""
        return self._label

    def setLabel(self, label):
        """Set the label displayed along the colorbar

        :param str label: The label
        """
        self._label = str(label)
        if self.colorbar is not None:
            self.colorbar.set_label(self._label)

    def _activeImageChanged(self, previous, legend):
        """Handle plot active curve changed"""
        if legend is None:  # No active image, display default colormap
            self._syncWithDefaultColormap()
            return

        # Sync with active image
        image = self._plot.getActiveImage()[0]

        # RGB(A) image, display default colormap
        if image.ndim != 2:
            self._syncWithDefaultColormap()
            return

        # data image, sync with image colormap
        cmap = self._plot.getActiveImage()[4]['colormap']
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
    """Display vertically the given text"""
    def __init__(self, text, parent=None):
        qt.QLabel.__init__(self, text, parent)

    def paintEvent(self, event ):
        painter = qt.QPainter(self)
        painter.setFont(self.font())

        painter.translate(0, self.sizeHint().height())
        painter.rotate(270)
        painter.drawText(qt.QRect(qt.QPoint(0,0),
                                  self.sizeHint()),
                         qt.Qt.AlignHCenter,self.text())


    def sizeHint(self):
        s = qt.QLabel.sizeHint(self)
        # return qt.QSize(self.rect().bottom(), self.rect().right())
        return qt.QSize(self.rect().right(), self.rect().bottom())

class Gradation(qt.QWidget):

    def __init__(self, colormap, parent=None, backend=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self.colorMap = MyColorMap(colormap)

        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        # self.setContentsMargins(0, 0, 0, 0);
        self.xs = numpy.arange(0, 256, 1)
        self.xs = self.xs.reshape(self.xs.shape[0], 1)
        self._display = PlotWidget(parent=self)
        self._display.addImage(self.xs)
        
        self.layout().addWidget(self._display)
        # self._display.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        # TODO : issue : if not in, plot won't display
        self.layout().addWidget(qt.QLabel('test', parent=self))


class GradationV2(qt.QWidget):

    def __init__(self, colormap, parent=None, backend=None):
        qt.QWidget.__init__(self, parent)
        self.setLayout(qt.QVBoxLayout())
        self.colorMap = MyColorMap(colormap)

        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

    def paintEvent(self, event):
        
        qt.QWidget.paintEvent(self, event)

        painter = qt.QPainter(self)
        gradient = qt.QLinearGradient(0, 0, 0, self.rect().height());
        for pt in numpy.arange(0, 256):
            position = pt/256.0
            gradient.setColorAt( position, self.colorMap.getColor(position))

        painter.setBrush(gradient)
        painter.drawRect(self.rect())


class MyColorMap(object):
    def __init__(self, colormap):
        self.name = colormap['name']
        self.normalization = colormap['normalization']
        self.autoscale = colormap['autoscale']
        self.vmin = colormap['vmin']
        self.vmax = colormap['vmax']

        # for now only deal with matplotlib colorbar
        from silx.gui.plot import Colors
        cmap = Colors.getMPLColormap(self.name)
        print(type(cmap))
        # res = matplotlib.colors.LinearSegmentedColormap('blue', cmap, 256)
        import matplotlib.cm
        norm = matplotlib.colors.Normalize(0, 255)
        self.scalarMappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    def getColor(self, val):
        color = self.scalarMappable.to_rgba(val*255)
        return qt.QColor(color[0]*255, color[1]*255, color[2]*255)
