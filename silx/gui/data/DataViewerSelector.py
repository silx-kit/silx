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
"""This module defines a widget to be able to select the available view
of the DataViewer.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "10/01/2017"

from collections import OrderedDict
import functools
import silx.gui.icons
from silx.gui import qt
from silx.gui.data.DataViewer import DataViewer
import silx.utils.weakref


class DataViewerSelector(qt.QWidget):
    """Widget to be able to select a custom view from the DataViewer"""

    def __init__(self, parent=None, dataViewer=None):
        """Constructor

        :param QWidget parent: The parent of the widget
        :param DataViewer dataViewer: The connected `DataViewer`
        """
        super(DataViewerSelector, self).__init__(parent)

        self.__buttons = {}
        self.__dataViewer = None
        self.__group = qt.QButtonGroup(self)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        iconSize = qt.QSize(16, 16)

        buttons = OrderedDict()
        buttons[DataViewer.PLOT1D_MODE] = ("Curve", "view-1d")
        buttons[DataViewer.PLOT2D_MODE] = ("Image", "view-2d")
        buttons[DataViewer.PLOT3D_MODE] = ("Cube", "view-3d")
        buttons[DataViewer.ARRAY_MODE] = ("Raw", "view-raw")
        buttons[DataViewer.RECORD_MODE] = ("Raw", "view-raw")
        buttons[DataViewer.TEXT_MODE] = ("Text", "view-text")
        buttons[DataViewer.STACK_MODE] = ("Image stack", "view-2d-stack")

        for modeId, state in buttons.items():
            text, iconName = state
            button = qt.QPushButton(text)
            button.setIcon(silx.gui.icons.getQIcon(iconName))
            button.setIconSize(iconSize)
            button.setCheckable(True)
            # the weakmethod is needed to be able to destroy the widget safely
            callback = functools.partial(silx.utils.weakref.WeakMethodProxy(self.__setDisplayMode), modeId)
            button.clicked.connect(callback)
            self.layout().addWidget(button)
            self.__group.addButton(button)
            self.__buttons[modeId] = button

        button = qt.QPushButton("Dummy")
        button.setCheckable(True)
        button.setVisible(False)
        self.layout().addWidget(button)
        self.__group.addButton(button)
        self.__buttonDummy = button

        self.layout().addStretch(1)

        if dataViewer is not None:
            self.setDataViewer(dataViewer)

    def setDataViewer(self, dataViewer):
        """Define the dataviewer connected to this status bar

        :param DataViewer dataViewer: The connected `DataViewer`
        """
        if self.__dataViewer is dataViewer:
            return
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.disconnect(self.__dataChanged)
            self.__dataViewer.displayModeChanged.disconnect(self.__displayModeChanged)
        self.__dataViewer = dataViewer
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.connect(self.__dataChanged)
            self.__dataViewer.displayModeChanged.connect(self.__displayModeChanged)
            self.__displayModeChanged(self.__dataViewer.displayMode())
        self.__dataChanged()

    def setFlat(self, isFlat):
        """Set the flat state of all the buttons.

        :param bool isFlat: True to display the buttons flatten.
        """
        for b in self.__buttons.values():
            b.setFlat(isFlat)
        self.__buttonDummy.setFlat(isFlat)

    def __displayModeChanged(self, mode):
        """Called on display mode changed"""
        selectedButton = self.__buttons.get(mode, self.__buttonDummy)
        selectedButton.setChecked(True)

    def __setDisplayMode(self, modeId, clickEvent=None):
        """Display a data using requested mode

        :param int modeId: Requested mode id
        :param clickEvent: Event sent by the clicked event
        """
        if self.__dataViewer is None:
            return
        self.__dataViewer.setDisplayMode(modeId)

    def __dataChanged(self):
        """Called on data changed"""
        if self.__dataViewer is None:
            for b in self.__buttons.values():
                b.setVisible(False)
        else:
            availableViews = self.__dataViewer.currentAvailableViews()
            availableModes = set([v.modeId() for v in availableViews])
            for modeId, button in self.__buttons.items():
                button.setVisible(modeId in availableModes)
