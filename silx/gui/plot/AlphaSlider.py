# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module defines slider widgets interacting with the transparency
of an image on a :class:`PlotWidget`

Classes:
--------

- :class:`BaseAlphaSlider` (abstract class)
- :class:`NamedImageAlphaSlider`
- :class:`ActiveImageAlphaSlider`

Example:
--------

This widget can, for instance, be added to a plot toolbar.

.. code-block:: python

    import numpy
    from silx.gui import qt
    from silx.gui.plot import PlotWidget
    from silx.gui.plot.ImageAlphaSlider import NamedImageAlphaSlider

    app = qt.QApplication([])
    pw = PlotWidget()

    img0 = numpy.arange(200*150).reshape((200, 150))
    pw.addImage(img0, legend="my background", z=0, origin=(50, 50))

    x, y = numpy.meshgrid(numpy.linspace(-10, 10, 200),
                          numpy.linspace(-10, 5, 150),
                          indexing="ij")
    img1 = numpy.asarray(numpy.sin(x * y) / (x * y),
                        dtype='float32')

    pw.addImage(img1, legend="my data", z=1,
                replace=False)

    alpha_slider = NamedImageAlphaSlider(parent=pw,
                                         plot=pw,
                                         legend="my data")
    alpha_slider.setOrientation(qt.Qt.Horizontal)

    toolbar = qt.QToolBar("plot", pw)
    toolbar.addWidget(alpha_slider)
    pw.addToolBar(toolbar)

    pw.show()
    app.exec_()

"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "24/03/2017"

import logging

from silx.gui import qt

_logger = logging.getLogger(__name__)


class BaseAlphaSlider(qt.QSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of a plot primitive (image, scatter or curve).

    Internally, the slider stores its state as an integer between
    0 and 255. This is the value emitted by the :attr:`valueChanged`
    signal.

    The method :meth:`getAlpha` returns the corresponding opacity/alpha
    as a float between 0. and 1. (with a step of :math:`\frac{1}{255}`).

    You must subclass this class and implement :meth:`getItem`.
    """
    sigAlphaChanged = qt.Signal(float)
    """Emits the alpha value when the slider's value changes,
    as a float between 0. and 1."""

    def __init__(self, parent=None, plot=None):
        """

        :param parent: Parent QWidget
        :param plot: Parent plot widget
        """
        assert plot is not None
        super(BaseAlphaSlider, self).__init__(parent)

        self.plot = plot

        self.setRange(0, 255)

        # if already connected to an item, use its alpha as initial value
        if self.getItem() is None:
            self.setValue(255)
            self.setEnabled(False)
        else:
            alpha = self.getItem().getAlpha()
            self.setValue(round(255*alpha))

        self.valueChanged.connect(self._valueChanged)

    def getItem(self):
        """You must implement this class to define which item
        to work on. It must return an item that inherits
        :class:`silx.gui.plot.items.core.AlphaMixIn`.

        :return: Item on which to operate, or None
        :rtype: :class:`silx.plot.items.Item`
        """
        raise NotImplementedError(
                "BaseAlphaSlider must be subclassed to " +
                "implement getItem()")

    def getAlpha(self):
        """Get the opacity, as a float between 0. and 1.

        :return: Alpha value in [0., 1.]
        :rtype: float
        """
        return self.value() / 255.

    def _valueChanged(self, value):
        self._updateItem()
        self.sigAlphaChanged.emit(value / 255.)

    def _updateItem(self):
        """Update the item's alpha channel.
        """
        item = self.getItem()
        if item is not None:
            item.setAlpha(self.getAlpha())


class NamedImageAlphaSlider(BaseAlphaSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of an image (defined by its legend).

    :param parent: Parent QWidget
    :param plot: Plot on which to operate
    :param str legend: Legend of image whose transparency is to be
        controlled.
        An image with this legend should exist at all times, or this
        widget should be manually deactivated whenever the image does not
        exist.

    See documentation of :class:`BaseAlphaSlider`
    """
    def __init__(self, parent=None, plot=None, legend=None):
        self._image_legend = legend
        super(NamedImageAlphaSlider, self).__init__(parent, plot)
        if legend is not None and self.plot.getImage(legend) is not None:
            self.setEnabled(True)
        else:
            self.setEnabled(False)

    def getItem(self):
        if self._image_legend is None:
            return None
        return self.plot.getImage(self._image_legend)

    def setLegend(self, legend):
        """Associate a different image on the same plot to the slider.

        :param legend: New legend of image whose transparency is to be
            controlled.
        """
        self._image_legend = legend
        if self.plot.getImage(legend) is not None:
            self.setEnabled(True)
        else:
            self.setEnabled(False)

    def getLegend(self):
        """Return legend of the image currently controlled by this slider.

        :return: Image legend associated to the slider
        """
        return self._image_legend


class ActiveImageAlphaSlider(BaseAlphaSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of the **active image**.

    :param parent: Parent QWidget
    :param plot: Plot on which to operate

    See documentation of :class:`BaseAlphaSlider`
    """
    def __init__(self, parent=None, plot=None):
        """

        :param parent: Parent QWidget
        :param plot: Plot widget on which to operate
        """
        super(ActiveImageAlphaSlider, self).__init__(parent, plot)
        plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def getItem(self):
        return self.plot.getActiveImage()

    def _activeImageChanged(self, previous, new):
        """Activate or deactivate slider depending on presence of a new
        active image.
        Apply transparency value to new active image.

        :param previous: Legend of previous active image, or None
        :param new: Legend of new active image, or None
        """
        if new is not None and not self.isEnabled():
            self.setEnabled(True)
        elif new is None and self.isEnabled():
            self.setEnabled(False)

        self._updateItem()


class NamedScatterAlphaSlider(BaseAlphaSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of a scatter (defined by its legend).

    :param parent: Parent QWidget
    :param plot: Plot on which to operate
    :param str legend: Legend of scatter whose transparency is to be
        controlled.
        An scatter with this legend should exist at all times, or this
        widget should be manually deactivated whenever the scatter does not
        exist.

    See documentation of :class:`BaseAlphaSlider`
    """
    def __init__(self, parent=None, plot=None, legend=None):
        self._scatter_legend = legend
        super(NamedScatterAlphaSlider, self).__init__(parent, plot)
        if legend is not None and self.plot.getScatter(legend) is not None:
            self.setEnabled(True)
        else:
            self.setEnabled(False)

    def getItem(self):
        if self._scatter_legend is None:
            return None
        return self.plot.getScatter(self._scatter_legend)

    def setLegend(self, legend):
        """Associate a different scatter on the same plot to the slider.

        :param legend: Legend of new scatter whose transparency is to be
            controlled.
        """
        self._scatter_legend = legend
        if self.plot.getScatter(legend) is not None:
            self.setEnabled(True)
        else:
            self.setEnabled(False)

    def getLegend(self):
        """Return legend of the scatter currently controlled by this slider.

        :return: Scatter legend associated to the slider
        """
        return self._scatter_legend
