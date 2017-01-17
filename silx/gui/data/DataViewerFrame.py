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
"""This module contains a DataViewer with a view selector.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "15/12/2016"

from silx.gui import qt
from .DataViewer import DataViewer
from .DataViewerSelector import DataViewerSelector


class DataViewerFrame(qt.QWidget):
    """
    A :class:`DataViewer` with a view selector.

    .. image:: img/DataViewerFrame.png

    This widget provides the same API as :class:`DataViewer`. Therefore, for more
    documentation, take a look at the documentation of the class
    :class:`DataViewer`.

    .. code-block:: python

        import numpy
        data = numpy.random.rand(500,500)
        viewer = DataViewerFrame()
        viewer.setData(data)
        viewer.setVisible(True)

    """

    displayModeChanged = qt.Signal(int)
    """Emitted when the display mode change"""

    dataChanged = qt.Signal()
    """Emitted when the data changed"""

    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent:
        """
        super(DataViewerFrame, self).__init__(parent)

        self.__dataViewer = DataViewer(self)
        self.__dataViewer.setFrameShape(qt.QFrame.StyledPanel)
        self.__dataViewer.setFrameShadow(qt.QFrame.Sunken)
        self.__dataViewerSelector = DataViewerSelector(self, self.__dataViewer)
        self.__dataViewerSelector.setFlat(True)

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.__dataViewer, 1)
        layout.addWidget(self.__dataViewerSelector)
        self.setLayout(layout)

        self.__dataViewer.dataChanged.connect(self.__dataChanged)
        self.__dataViewer.displayModeChanged.connect(self.__displayModeChanged)

    def __dataChanged(self):
        """Called when the data is changed"""
        self.dataChanged.emit()

    def __displayModeChanged(self, modeId):
        """Called when the display mode changed"""
        self.displayModeChanged.emit(modeId)

    def setData(self, data):
        """Set the data to view.

        It mostly can be a h5py.Dataset or a numpy.ndarray. Other kind of
        objects will be displayed as text rendering.

        :param numpy.ndarray data: The data.
        """
        self.__dataViewer.setData(data)

    def data(self):
        """Returns the data"""
        return self.__dataViewer.data()

    def displayMode(self):
        return self.__dataViewer.displayMode()

    def setDisplayMode(self, modeId):
        """Set the displayed view using display mode.

        Change the displayed view according to the requested mode.

        :param int modeId:  Display mode, one of

            - `EMPTY_MODE`: display nothing
            - `PLOT1D_MODE`: display the data as a curve
            - `PLOT2D_MODE`: display the data as an image
            - `TEXT_MODE`: display the data as a text
            - `ARRAY_MODE`: display the data as a table
        """
        return self.__dataViewer.setDisplayMode(modeId)
