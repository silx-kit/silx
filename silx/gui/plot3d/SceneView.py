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
"""This module provides a window to view data sets in 3D."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/10/2017"

import logging

from .. import qt
from ..plot.Colors import rgba

from .Plot3DWindow import Plot3DWindow
from .scene import axes
from .items import GroupItem


_logger = logging.getLogger(__name__)


class SceneView(Plot3DWindow):
    """Widget displaying data sets in 3D"""

    def __init__(self, parent=None):
        super(SceneView, self).__init__(parent)
        self._items = []

        self._foregroundColor = 1., 1., 1., 1.
        self._highlightColor = 0.7, 0.7, 0., 1.

        self._sceneGroup = GroupItem(parent=self)

        self._bbox = axes.LabelledAxes()
        self._bbox.children = [self._sceneGroup._getScenePrimitive()]
        self.getPlot3DWidget().viewport.scene.children.append(self._bbox)

    def getSceneGroup(self):
        """Returns the root group of the scene

        :rtype: GroupItem
        """
        return self._sceneGroup

    # Axes labels

    def isBoundingBoxVisible(self):
        """Returns axes labels, grid and bounding box visibility.

        :rtype: bool
        """
        return self._bbox.boxVisible

    def setBoundingBoxVisible(self, visible):
        """Set axes labels, grid and bounding box visibility.

        :param bool visible: True to show axes, False to hide
        """
        self._bbox.boxVisible = bool(visible)

    def setAxesLabels(self, xlabel=None, ylabel=None, zlabel=None):
        """Set the text labels of the axes.

        :param str xlabel: Label of the X axis, None to leave unchanged.
        :param str ylabel: Label of the Y axis, None to leave unchanged.
        :param str zlabel: Label of the Z axis, None to leave unchanged.
        """
        if xlabel is not None:
            self._bbox.xlabel = xlabel

        if ylabel is not None:
            self._bbox.ylabel = ylabel

        if zlabel is not None:
            self._bbox.zlabel = zlabel

    class _Labels(tuple):
        """Return type of :meth:`getAxesLabels`"""

        def getXLabel(self):
            """Label of the X axis (str)"""
            return self[0]

        def getYLabel(self):
            """Label of the Y axis (str)"""
            return self[1]

        def getZLabel(self):
            """Label of the Z axis (str)"""
            return self[2]

    def getAxesLabels(self):
        """Returns the text labels of the axes

        >>> widget = SceneView()
        >>> widget.setAxesLabels(xlabel='X')

        You can get the labels either as a 3-tuple:

        >>> xlabel, ylabel, zlabel = widget.getAxesLabels()

        Or as an object with methods getXLabel, getYLabel and getZLabel:

        >>> labels = widget.getAxesLabels()
        >>> labels.getXLabel()
        ... 'X'

        :return: object describing the labels
        """
        return self._Labels((self._bbox.xlabel,
                             self._bbox.ylabel,
                             self._bbox.zlabel))

    # Colors

    def _updateColors(self):
        """Update item depending on foreground/highlight color"""
        self._bbox.tickColor = self._foregroundColor
        self._bbox.color = self._highlightColor

    def getForegroundColor(self):
        """Return color used for text and bounding box (QColor)"""
        return qt.QColor.fromRgbF(*self._foregroundColor)

    def setForegroundColor(self, color):
        """Set the foreground color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._foregroundColor:
            self._foregroundColor = color
            self._updateColors()

    def getHighlightColor(self):
        """Return color used for highlighted item bounding box (QColor)"""
        return qt.QColor.fromRgbF(*self._highlightColor)

    def setHighlightColor(self, color):
        """Set highlighted item color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._highlightColor:
            self._highlightColor = color
            self._updateColors()
