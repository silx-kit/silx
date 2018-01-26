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
__date__ = "23/01/2018"

import weakref
import functools
from silx.gui import qt
import silx.utils.weakref


class DataViewerSelector(qt.QWidget):
    """Widget to be able to select a custom view from the DataViewer"""

    def __init__(self, parent=None, dataViewer=None):
        """Constructor

        :param QWidget parent: The parent of the widget
        :param DataViewer dataViewer: The connected `DataViewer`
        """
        super(DataViewerSelector, self).__init__(parent)

        self.__group = None
        self.__buttons = {}
        self.__buttonLayout = None
        self.__buttonDummy = None
        self.__dataViewer = None

        # Create the fixed layout
        self.setLayout(qt.QHBoxLayout())
        layout = self.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.__buttonLayout = qt.QHBoxLayout()
        self.__buttonLayout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.__buttonLayout)
        layout.addStretch(1)

        if dataViewer is not None:
            self.setDataViewer(dataViewer)

    def __updateButtons(self):
        if self.__group is not None:
            self.__group.deleteLater()

        # Clean up
        for _, b in self.__buttons.items():
            b.deleteLater()
        if self.__buttonDummy is not None:
            self.__buttonDummy.deleteLater()
            self.__buttonDummy = None
        self.__buttons = {}
        self.__buttonDummy = None

        self.__group = qt.QButtonGroup(self)
        if self.__dataViewer is None:
            return

        iconSize = qt.QSize(16, 16)

        for view in self.__dataViewer.availableViews():
            label = view.label()
            icon = view.icon()
            button = qt.QPushButton(label)
            button.setIcon(icon)
            button.setIconSize(iconSize)
            button.setCheckable(True)
            # the weak objects are needed to be able to destroy the widget safely
            weakView = weakref.ref(view)
            weakMethod = silx.utils.weakref.WeakMethodProxy(self.__setDisplayedView)
            callback = functools.partial(weakMethod, weakView)
            button.clicked.connect(callback)
            self.__buttonLayout.addWidget(button)
            self.__group.addButton(button)
            self.__buttons[view] = button

        button = qt.QPushButton("Dummy")
        button.setCheckable(True)
        button.setVisible(False)
        self.__buttonLayout.addWidget(button)
        self.__group.addButton(button)
        self.__buttonDummy = button

        self.__updateButtonsVisibility()
        self.__displayedViewChanged(self.__dataViewer.displayedView())

    def setDataViewer(self, dataViewer):
        """Define the dataviewer connected to this status bar

        :param DataViewer dataViewer: The connected `DataViewer`
        """
        if self.__dataViewer is dataViewer:
            return
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.disconnect(self.__updateButtonsVisibility)
            self.__dataViewer.displayedViewChanged.disconnect(self.__displayedViewChanged)
        self.__dataViewer = dataViewer
        if self.__dataViewer is not None:
            self.__dataViewer.dataChanged.connect(self.__updateButtonsVisibility)
            self.__dataViewer.displayedViewChanged.connect(self.__displayedViewChanged)
        self.__updateButtons()

    def setFlat(self, isFlat):
        """Set the flat state of all the buttons.

        :param bool isFlat: True to display the buttons flatten.
        """
        for b in self.__buttons.values():
            b.setFlat(isFlat)
        self.__buttonDummy.setFlat(isFlat)

    def __displayedViewChanged(self, view):
        """Called on displayed view changes"""
        selectedButton = self.__buttons.get(view, self.__buttonDummy)
        selectedButton.setChecked(True)

    def __setDisplayedView(self, refView, clickEvent=None):
        """Display a data using the requested view

        :param DataView view: Requested view
        :param clickEvent: Event sent by the clicked event
        """
        if self.__dataViewer is None:
            return
        view = refView()
        if view is None:
            return
        self.__dataViewer.setDisplayedView(view)

    def __checkAvailableButtons(self):
        views = set(self.__dataViewer.availableViews())
        if views == set(self.__buttons.keys()):
            return
        # Recreate all the buttons
        # TODO: We dont have to create everything again
        # We expect the views stay quite stable
        self.__updateButtons()

    def __updateButtonsVisibility(self):
        """Called on data changed"""
        if self.__dataViewer is None:
            for b in self.__buttons.values():
                b.setVisible(False)
        else:
            self.__checkAvailableButtons()
            availableViews = set(self.__dataViewer.currentAvailableViews())
            for view, button in self.__buttons.items():
                button.setVisible(view in availableViews)
