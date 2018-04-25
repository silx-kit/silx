# coding: utf-8
# /*##########################################################################
#
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
"""This module contains a DataViewer with a view selector.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/04/2018"

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

    displayedViewChanged = qt.Signal(object)
    """Emitted when the displayed view changes"""

    dataChanged = qt.Signal()
    """Emitted when the data changes"""

    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent:
        """
        super(DataViewerFrame, self).__init__(parent)

        class _DataViewer(DataViewer):
            """Overwrite methods to avoid to create views while the instance
            is not created. `initializeViews` have to be called manually."""

            def _initializeViews(self):
                pass

            def initializeViews(self):
                """Avoid to create views while the instance is not created."""
                super(_DataViewer, self)._initializeViews()

            def _createDefaultViews(self, parent):
                """Expose the original `createDefaultViews` function"""
                return super(_DataViewer, self).createDefaultViews()

            def createDefaultViews(self, parent=None):
                """Allow the DataViewerFrame to override this function"""
                return self.parent().createDefaultViews(parent)

        self.__dataViewer = _DataViewer(self)
        # initialize views when `self.__dataViewer` is set
        self.__dataViewer.initializeViews()
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
        self.__dataViewer.displayedViewChanged.connect(self.__displayedViewChanged)

    def __dataChanged(self):
        """Called when the data is changed"""
        self.dataChanged.emit()

    def __displayedViewChanged(self, view):
        """Called when the displayed view changes"""
        self.displayedViewChanged.emit(view)

    def setGlobalHooks(self, hooks):
        """Set a data view hooks for all the views

        :param DataViewHooks context: The hooks to use
        """
        self.__dataViewer.setGlobalHooks(hooks)

    def availableViews(self):
        """Returns the list of registered views

        :rtype: List[DataView]
        """
        return self.__dataViewer.availableViews()

    def currentAvailableViews(self):
        """Returns the list of available views for the current data

        :rtype: List[DataView]
        """
        return self.__dataViewer.currentAvailableViews()

    def createDefaultViews(self, parent=None):
        """Create and returns available views which can be displayed by default
        by the data viewer. It is called internally by the widget. It can be
        overwriten to provide a different set of viewers.

        :param QWidget parent: QWidget parent of the views
        :rtype: List[silx.gui.data.DataViews.DataView]
        """
        return self.__dataViewer._createDefaultViews(parent)

    def addView(self, view):
        """Allow to add a view to the dataview.

        If the current data support this view, it will be displayed.

        :param DataView view: A dataview
        """
        return self.__dataViewer.addView(view)

    def removeView(self, view):
        """Allow to remove a view which was available from the dataview.

        If the view was displayed, the widget will be updated.

        :param DataView view: A dataview
        """
        return self.__dataViewer.removeView(view)

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

    def setDisplayedView(self, view):
        self.__dataViewer.setDisplayedView(view)

    def displayedView(self):
        return self.__dataViewer.displayedView()

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

    def getViewFromModeId(self, modeId):
        """See :meth:`DataViewer.getViewFromModeId`"""
        return self.__dataViewer.getViewFromModeId(modeId)

    def replaceView(self, modeId, newView):
        """Replace one of the builtin data views with a custom view.
        See :meth:`DataViewer.replaceView` for more documentation.

        :param DataViews.DataView newView: New data view
        :return: True if replacement was successful, else False
        """
        return self.__dataViewer.replaceView(modeId, newView)
