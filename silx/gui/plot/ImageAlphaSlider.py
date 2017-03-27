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

- :class:`BaseImageAlphaSlider`
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
                                         legend="my data",
                                         label="My data's opacity")
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
import numpy

from silx.gui import qt
from silx.gui.plot.Colors import getMPLScalarMappable

_logger = logging.getLogger(__name__)


class BaseImageAlphaSlider(qt.QWidget):
    """Slider widget to be used in a plot toolbar to control the
    transparency of an image.

    The slider range is 0 to 255.
    0 means no opacity (100% transparency), and 255 means total opacity.

    You must subclass this class and implement :meth:`getImage`.
    """
    sigValueChanged = qt.Signal(int)
    """Emits the slider's current value, between 0 and 255."""

    def __init__(self, parent=None, plot=None, label=None):
        """

        :param parent: Parent QWidget
        :param plot: Parent plot widget
        :param label: Optional text used as a label in front of the slider
        """
        assert plot is not None
        super(BaseImageAlphaSlider, self).__init__(parent)

        self.plot = plot

        layout = qt.QHBoxLayout(self)

        if label is not None:
            label_widget = qt.QLabel(label)
            layout.addWidget(label_widget)

        self.slider = qt.QSlider(qt.Qt.Horizontal, self)
        self.slider.setRange(0, 255)
        self.slider.valueChanged.connect(self._valueChanged)
        self.slider.setValue(255)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def getImage(self):
        """You must implement this class to define which image
        to work on.

        :return: Image on which to operate, or None
        :rtype: :class:`silx.plot.items.Image`
        """
        raise NotImplementedError(
                "BaseImageAlphaSlider must be subclassed to " +
                "implement getImage()")

    def _valueChanged(self, value):
        self._updateImage()
        self.sigValueChanged.emit(value)

    def _updateImage(self):
        """Get active image's colormap, update its alpha channel.
        """
        img = self.getImage()
        if img is None:
            _logger.warning("ImageAlphaSlider not connected to an image")
            return
        cmap = img.getColormap()
        cmap_name = cmap.get("name")

        if cmap_name is None:
            rgba_colors = cmap.get("colors")
            if rgba_colors is None:
                raise RuntimeError(
                        "Could not get active image's colormap as RGBA array")
        else:
            # rgba vector for colormap must be computed from name
            if cmap.get("autoscale"):
                # use image data for vmin vmax calculation
                vmin = numpy.nanmin(img.getData(copy=False))
                vmax = numpy.nanmax(img.getData(copy=False))
            else:
                vmin = cmap.get("vmin")
                assert vmin is not None
                vmax = cmap.get("vmax")
                assert vmax is not None

            sm = getMPLScalarMappable(cmap,
                                      data=numpy.array([vmin, vmax]))

            rgba_colors = sm.to_rgba(numpy.linspace(vmin, vmax, 255))

        rgba_colors = self._change_alpha(rgba_colors,
                                         self.slider.value())

        # set custom colormap based on previous one with alpha changed
        cmap["name"] = None
        cmap["colors"] = rgba_colors
        img.setColormap(cmap)

    def _change_alpha(self, rgba_colors, alpha):
        """Set the alpha channel to a constant value for the entire
         colormap vector.

        :param rgba_colors: Nx3 or Nx4 numpy array of RGB(A) colors.
            Type can be uint8 or float in [0, 1] (will be converted to uint8)
        :param int alpha: Alpha value in range [0, 255]
        :return: Nx4 numpy array of RGBA colors, as uint8.
        """
        rgba_colors = self._normalize_rgba_colors(rgba_colors)
        rgba_colors[:, -1] = alpha
        return rgba_colors

    @staticmethod
    def _normalize_rgba_colors(rgba_colors):
        """Return colormap vector as an (N, 4) array of uint8.

        :param rgba_colors: Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 in [0, 255] or float in [0, 1].
        :return: Nx4 numpy array of RGBA colors, as uint8.
            If input array does not have an alpha channel, it is set to 255
        """
        if (rgba_colors.dtype.kind != "f" and
                    rgba_colors.dtype != numpy.uint8):
            raise TypeError("rgba array must be float in [0, 1] or uint8")
        elif rgba_colors.dtype.kind == "f":
            # convert to uint in [range 0, 255]
            rgba_colors = numpy.array(rgba_colors * 255, dtype=numpy.uint8)

        if len(rgba_colors.shape) != 2 or rgba_colors.shape[1] not in (3, 4):
            raise TypeError("rgb(a) array shape must (N, 3) or (N, 4)")
        elif rgba_colors.shape[1] == 3:
            # add alpha channel
            new_array = numpy.empty((rgba_colors.shape[0], 4))
            new_array[:, :-1] = rgba_colors
            new_array[:, -1] = 255
            rgba_colors = new_array

        return rgba_colors


class NamedImageAlphaSlider(BaseImageAlphaSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of an image (defined by its legend).

    :param parent: Parent QWidget
    :param plot: Plot on which to operate
    :param str legend: Legend of image whose transparency is to be
        controlled.
        An image with this legend should exist at all times, or this
        widget should be manually deactivated whenever the image does not
        exist.
    :param str label: Optional label put in front of the slider.

    See documentation of :class:`BaseImageAlphaSlider`
    """
    def __init__(self, parent=None, plot=None, legend=None, label=None):
        self._image_legend = legend
        super(NamedImageAlphaSlider, self).__init__(parent, plot, label)

    def getImage(self):
        return self.plot.getImage(self._image_legend)

    def setLegend(self, legend):
        """Associate a different image on the same plot to the slider.

        :param legend: New legend of image whose transparency is to be
            controlled.
        """
        self._image_legend = legend

    def getLegend(self):
        """Return legend of the image currently controlled by this slider.

        :return: Image legend associated to the slider
        """
        return self._image_legend


class ActiveImageAlphaSlider(BaseImageAlphaSlider):
    """Slider widget to be used in a plot toolbar to control the
    transparency of the **active image**.

    :param parent: Parent QWidget
    :param plot: Plot on which to operate
    :param str label: Optional label put in front of the slider.

    See documentation of :class:`BaseImageAlphaSlider`
    """
    def __init__(self, parent=None, plot=None, label=None):
        """

        :param parent: Parent QWidget
        :param plot: Plot widget on which to operate
        """
        super(ActiveImageAlphaSlider, self).__init__(parent, plot, label)
        plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def getImage(self):
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

        self._updateImage()
