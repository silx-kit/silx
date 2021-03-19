# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""A QComboBox to display prefered colormaps
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "27/11/2018"


import logging
import numpy

from .. import qt
from .. import colors as colors_mdl

_logger = logging.getLogger(__name__)


_colormapIconPreview = {}


class ColormapNameComboBox(qt.QComboBox):
    def __init__(self, parent=None):
        qt.QComboBox.__init__(self, parent)
        self.__initItems()

    LUT_NAME = qt.Qt.UserRole + 1
    LUT_COLORS = qt.Qt.UserRole + 2

    def __initItems(self):
        for colormapName in colors_mdl.preferredColormaps():
            index = self.count()
            self.addItem(str.title(colormapName))
            self.setItemIcon(index, self.getIconPreview(name=colormapName))
            self.setItemData(index, colormapName, role=self.LUT_NAME)

    def getIconPreview(self, name=None, colors=None):
        """Return an icon preview from a LUT name.

        This icons are cached into a global structure.

        :param str name: Name of the LUT
        :param numpy.ndarray colors: Colors identify the LUT
        :rtype: qt.QIcon
        """
        if name is not None:
            iconKey = name
        else:
            iconKey = tuple(colors)
        icon = _colormapIconPreview.get(iconKey, None)
        if icon is None:
            icon = self.createIconPreview(name, colors)
            _colormapIconPreview[iconKey] = icon
        return icon

    def createIconPreview(self, name=None, colors=None):
        """Create and return an icon preview from a LUT name.

        This icons are cached into a global structure.

        :param str name: Name of the LUT
        :param numpy.ndarray colors: Colors identify the LUT
        :rtype: qt.QIcon
        """
        colormap = colors_mdl.Colormap(name)
        size = 32
        if name is not None:
            lut = colormap.getNColors(size)
        else:
            lut = colors
            if len(lut) > size:
                # Down sample
                step = int(len(lut) / size)
                lut = lut[::step]
            elif len(lut) < size:
                # Over sample
                indexes = numpy.arange(size) / float(size) * (len(lut) - 1)
                indexes = indexes.astype("int")
                lut = lut[indexes]
        if lut is None or len(lut) == 0:
            return qt.QIcon()

        pixmap = qt.QPixmap(size, size)
        painter = qt.QPainter(pixmap)
        for i in range(size):
            rgb = lut[i]
            r, g, b = rgb[0], rgb[1], rgb[2]
            painter.setPen(qt.QColor(r, g, b))
            painter.drawPoint(qt.QPoint(i, 0))

        painter.drawPixmap(0, 1, size, size - 1, pixmap, 0, 0, size, 1)
        painter.end()

        return qt.QIcon(pixmap)

    def getCurrentName(self):
        return self.itemData(self.currentIndex(), self.LUT_NAME)

    def getCurrentColors(self):
        return self.itemData(self.currentIndex(), self.LUT_COLORS)

    def findLutName(self, name):
        return self.findData(name, role=self.LUT_NAME)

    def findLutColors(self, lut):
        for index in range(self.count()):
            if self.itemData(index, role=self.LUT_NAME) is not None:
                continue
            colors = self.itemData(index, role=self.LUT_COLORS)
            if colors is None:
                continue
            if numpy.array_equal(colors, lut):
                return index
        return -1

    def setCurrentLut(self, colormap):
        name = colormap.getName()
        if name is not None:
            self._setCurrentName(name)
        else:
            lut = colormap.getColormapLUT()
            self._setCurrentLut(lut)

    def _setCurrentLut(self, lut):
        index = self.findLutColors(lut)
        if index == -1:
            index = self.count()
            self.addItem("Custom")
            self.setItemIcon(index, self.getIconPreview(colors=lut))
            self.setItemData(index, None, role=self.LUT_NAME)
            self.setItemData(index, lut, role=self.LUT_COLORS)
        self.setCurrentIndex(index)

    def _setCurrentName(self, name):
        index = self.findLutName(name)
        if index < 0:
            index = self.count()
            self.addItem(str.title(name))
            self.setItemIcon(index, self.getIconPreview(name=name))
            self.setItemData(index, name, role=self.LUT_NAME)
        self.setCurrentIndex(index)
