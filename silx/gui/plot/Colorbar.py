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


from .. import qt

import matplotlib
import matplotlib.ticker
from ._matplotlib import FigureCanvasQTAgg

from .Colors import getMPLColormap


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
        self.colorbar = None  # matplotlib colorbar
        self._colormap = None  # PlotWidget compatible colormap

        self._label = ''  # Text label to display

        self._fig = matplotlib.figure.Figure()
        self._fig.set_facecolor("w")

        self._canvas = FigureCanvasQTAgg(self._fig)

        super(ColorbarWidget, self).__init__(parent)
        self.setFixedWidth(150)
        layout = qt.QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

        self._plot = plot
        if self._plot is not None:
            self._plot.sigActiveImageChanged.connect(self._activeImageChanged)
            self._activeImageChanged(
                None, self._plot.getActiveImage(just_legend=True))
            self._plot.sigSetDefaultColormap.connect(
                self._defaultColormapChanged)

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
        if name is None and colors is None:
            self._fig.clear()
            self.colorbar = None
            self._colormap = None
            self._canvas.draw()
            return

        if normalization == 'linear':
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        elif normalization == 'log':
            if vmin <= 0 or vmax <= 0:
                _logger.warning(
                    'Log colormap with bound <= 0: changing bounds.')
                vmin, vmax = 1., 10.
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise ValueError('Wrong normalization %s' % normalization)

        self._fig.clear()
        ax = self._fig.add_axes((0.03, 0.15, 0.3, 0.75))
        self.colorbar = matplotlib.colorbar.ColorbarBase(
            ax, cmap=getMPLColormap(name), norm=norm, orientation='vertical')
        self.colorbar.set_label(self._label)
        if normalization == 'linear':
            formatter = matplotlib.ticker.FormatStrFormatter('%.4g')
            self.colorbar.formatter = formatter
            self.colorbar.update_ticks()
        self._canvas.draw()

        self._colormap = {'name': name,
                          'normalization': normalization,
                          'autoscale': False,
                          'vmin': vmin,
                          'vmax': vmax,
                          'colors': colors}

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
            self._canvas.draw()

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
