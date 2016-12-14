# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""This module contains a DataViewer all in one in a frame.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/12/2016"

from silx.gui import qt
from .DataViewer import DataViewer
from .DataViewerSelector import DataViewerSelector


class DataViewerFrame(qt.QWidget):
    """
    A DataViewer with a view selector
    """

    dataChanged = qt.Signal()
    """Emitted when the data changed"""

    def __init__(self, parent=None):
        super(DataViewerFrame, self).__init__(parent)

        self.__dataViewer = DataViewer(self)
        self.__dataViewer.setFrameShape(qt.QFrame.StyledPanel)
        self.__dataViewer.setFrameShadow(qt.QFrame.Sunken)
        self.__dataViewerSelector = DataViewerSelector(self, self.__dataViewer)
        self.__dataViewerSelector.setFlat(True)

        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addWidget(self.__dataViewer, 1)
        self.layout().addWidget(self.__dataViewerSelector)

        self.__dataViewer.dataChanged.connect(self.__dataChanged)

    def __dataChanged(self):
        self.dataChanged.emit()

    def setData(self, data):
        self.__dataViewer.setData(data)

    def data(self):
        return self.__dataViewer.data()
