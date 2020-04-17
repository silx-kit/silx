# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module provides tool bar helper.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"


import logging
import weakref

from silx.gui import qt
from . import manager


_logger = logging.getLogger(__name__)


class ProfileToolBar(qt.QToolBar):
    """Tool bar to provide profile for a plot.
    
    It is an helper class. For a dedicated application it would be better to
    use an own tool bar in order in order have more flexibility.
    """
    def __init__(self, parent=None, plot=None):
        super(ProfileToolBar, self).__init__(parent=parent)
        self.__scheme = None
        self.__manager = None
        self.__plot = weakref.ref(plot)

    def getPlotWidget(self):
        """The :class:`~silx.gui.plot.PlotWidget` associated to the toolbar.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        if self.__plot is None:
            return None
        plot = self.__plot()
        if self.__plot is None:
            self.__plot = None
        return plot

    def setScheme(self, scheme):
        """Initialize the tool bar using a configuration scheme.

        It have to be done once and only once.

        :param str scheme: One of "scatter", "image", "imagestack"
        """
        assert self.__scheme is None
        self.__scheme = scheme

        plot = self.getPlotWidget()
        self.__manager = manager.ProfileManager(self, plot)

        if scheme == "image":
            self.__manager.setItemType(image=True)
            self.__manager.setActiveItemTracking(True)
            for action in self.__manager.createImageActions(self):
                self.addAction(action)
            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)
        elif scheme == "scatter":
            self.__manager.setItemType(scatter=True)
            self.__manager.setActiveItemTracking(True)
            for action in self.__manager.createScatterActions(self):
                self.addAction(action)
            for action in self.__manager.createScatterSliceActions(self):
                self.addAction(action)
            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)
        elif scheme == "imagestack":
            self.__manager.setItemType(image=True)
            self.__manager.setActiveItemTracking(True)
            for action in self.__manager.createImageStackActions(self):
                self.addAction(action)
            cleanAction = self.__manager.createClearAction(self)
            self.addAction(cleanAction)
            editorAction = self.__manager.createEditorAction(self)
            self.addAction(editorAction)
        else:
            raise ValueError("Toolbar scheme %s unsupported" % scheme)
