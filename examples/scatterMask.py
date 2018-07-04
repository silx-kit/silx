# coding: utf-8
# /*##########################################################################
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""
This example demonstrates how to use ScatterMaskToolsWidget
and NamedScatterAlphaSlider with a PlotWidget.
"""

import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget

from silx.gui.plot.AlphaSlider import NamedScatterAlphaSlider

from silx.gui.plot import ScatterMaskToolsWidget


class MaskScatterWidget(qt.QMainWindow):
    """Simple plot widget designed to display a scatter plot on top
    of a background image.

    A transparency slider is provided to adjust the transparency of the
    scatter points.

    A mask tools widget is provided to select/mask points of the scatter
    plot.
    """
    def __init__(self, parent=None):
        super(MaskScatterWidget, self).__init__(parent=parent)
        self._activeScatterLegend = "active scatter"
        self._bgImageLegend = "background image"

        # widgets
        centralWidget = qt.QWidget(self)

        self._plot = PlotWidget(parent=centralWidget)

        self._maskToolsWidget = ScatterMaskToolsWidget.ScatterMaskToolsWidget(
            plot=self._plot, parent=centralWidget)

        self._alphaSlider = NamedScatterAlphaSlider(parent=self, plot=self._plot)
        self._alphaSlider.setOrientation(qt.Qt.Horizontal)
        self._alphaSlider.setToolTip("Adjust scatter opacity")

        # layout
        layout = qt.QVBoxLayout(centralWidget)
        layout.addWidget(self._plot)
        layout.addWidget(self._alphaSlider)
        layout.addWidget(self._maskToolsWidget)
        centralWidget.setLayout(layout)

        self.setCentralWidget(centralWidget)

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
                    Mask type: array of uint8 of dimension 1,
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 1-tuple if successful.
        """
        return self._maskToolsWidget.setSelectionMask(mask,
                                                      copy=copy)

    def getSelectionMask(self, copy=True):
        """Get the current mask as a 1D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the scatter data.
                 If there is no scatter data, None is returned.
        :rtype: 1D numpy.ndarray of uint8
        """
        return self._maskToolsWidget.getSelectionMask(copy=copy)

    def setBackgroundImage(self, image, xscale=(0, 1.), yscale=(0, 1.),
                           colormap=None):
        """Set a background image

        :param image: 2D image, array of shape (nrows, ncolumns)
            or (nrows, ncolumns, 3) or (nrows, ncolumns, 4) RGB(A) pixmap
        :param xscale: Factors for polynomial scaling  for x-axis,
            *(a, b)* such as :math:`x \mapsto a + bx`
        :param yscale: Factors for polynomial scaling  for y-axis
        """
        self._plot.addImage(image, legend=self._bgImageLegend,
                            origin=(xscale[0], yscale[0]),
                            scale=(xscale[1], yscale[1]),
                            z=0,
                            colormap=colormap)

    def setScatter(self, x, y, v=None, info=None, colormap=None):
        """Set the scatter data, by providing its data as a 1D
        array or as a pixmap.

        The scatter plot set through this method is associated
        with the transparency slider.

        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :param v: Array of values for each point, represented as the color
             of the point on the plot.
        """
        self._plot.addScatter(x, y, v, legend=self._activeScatterLegend,
                              info=info, colormap=colormap)
        # the mask is associated with the active scatter
        self._plot._setActiveItem(kind="scatter",
                                  legend=self._activeScatterLegend)

        self._alphaSlider.setLegend(self._activeScatterLegend)


if __name__ == "__main__":
    app = qt.QApplication([])
    msw = MaskScatterWidget()

    # create a synthetic bg image
    bg_img = numpy.arange(200*150).reshape((200, 150))
    bg_img[75:125, 80:120] = 1000

    # create synthetic data for a scatter plot
    twopi = numpy.pi * 2
    x = 50 + 80 * numpy.linspace(0, twopi, num=100) / twopi * numpy.cos(numpy.linspace(0, twopi, num=100))
    y = 150 + 150 * numpy.linspace(0, twopi, num=100) / twopi * numpy.sin(numpy.linspace(0, twopi, num=100))
    v = numpy.arange(100) / 3.14

    msw.setScatter(x, y, v=v)
    msw.setBackgroundImage(bg_img)
    msw.show()
    app.exec_()
