# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "16/06/2017"


from .. import qt
from .Hdf5TreeModel import Hdf5TreeModel


class Hdf5HeaderView(qt.QHeaderView):
    """
    Default HDF5 header

    Manage auto-resize and context menu to display/hide columns
    """

    def __init__(self, orientation, parent=None):
        """
        Constructor

        :param orientation qt.Qt.Orientation: Orientation of the header
        :param parent qt.QWidget: Parent of the widget
        """
        super(Hdf5HeaderView, self).__init__(orientation, parent)
        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__createContextMenu)

        # default initialization done by QTreeView for it's own header
        self.setSectionsClickable(True)
        self.setSectionsMovable(True)
        self.setDefaultAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self.setStretchLastSection(True)

        self.__auto_resize = True
        self.__hide_columns_popup = True

    def setModel(self, model):
        """Override model to configure view when a model is expected

        `qt.QHeaderView.setSectionResizeMode` expect already existing columns
        to work.

        :param model qt.QAbstractItemModel: A model
        """
        super(Hdf5HeaderView, self).setModel(model)
        self.__updateAutoResize()

    def __updateAutoResize(self):
        """Update the view according to the state of the auto-resize"""
        if self.__auto_resize:
            self.setSectionResizeMode(Hdf5TreeModel.NAME_COLUMN, qt.QHeaderView.ResizeToContents)
            self.setSectionResizeMode(Hdf5TreeModel.TYPE_COLUMN, qt.QHeaderView.ResizeToContents)
            self.setSectionResizeMode(Hdf5TreeModel.SHAPE_COLUMN, qt.QHeaderView.ResizeToContents)
            self.setSectionResizeMode(Hdf5TreeModel.VALUE_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.DESCRIPTION_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.NODE_COLUMN, qt.QHeaderView.ResizeToContents)
            self.setSectionResizeMode(Hdf5TreeModel.LINK_COLUMN, qt.QHeaderView.ResizeToContents)
        else:
            self.setSectionResizeMode(Hdf5TreeModel.NAME_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.TYPE_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.SHAPE_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.VALUE_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.DESCRIPTION_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.NODE_COLUMN, qt.QHeaderView.Interactive)
            self.setSectionResizeMode(Hdf5TreeModel.LINK_COLUMN, qt.QHeaderView.Interactive)

    def setAutoResizeColumns(self, autoResize):
        """Enable/disable auto-resize. When auto-resized, the header take care
        of the content of the column to set fixed size of some of them, or to
        auto fix the size according to the content.

        :param autoResize bool: Enable/disable auto-resize
        """
        if self.__auto_resize == autoResize:
            return
        self.__auto_resize = autoResize
        self.__updateAutoResize()

    def hasAutoResizeColumns(self):
        """Is auto-resize enabled.

        :rtype: bool
        """
        return self.__auto_resize

    autoResizeColumns = qt.Property(bool, hasAutoResizeColumns, setAutoResizeColumns)
    """Property to enable/disable auto-resize."""

    def setEnableHideColumnsPopup(self, enablePopup):
        """Enable/disable a popup to allow to hide/show each column of the
        model.

        :param bool enablePopup: Enable/disable popup to hide/show columns
        """
        self.__hide_columns_popup = enablePopup

    def hasHideColumnsPopup(self):
        """Is popup to hide/show columns is enabled.

        :rtype: bool
        """
        return self.__hide_columns_popup

    enableHideColumnsPopup = qt.Property(bool, hasHideColumnsPopup, setAutoResizeColumns)
    """Property to enable/disable popup allowing to hide/show columns."""

    def __genHideSectionEvent(self, column):
        """Generate a callback which change the column visibility according to
        the event parameter

        :param int column: logical id of the column
        :rtype: callable
        """
        return lambda checked: self.setSectionHidden(column, not checked)

    def __createContextMenu(self, pos):
        """Callback to create and display a context menu

        :param pos qt.QPoint: Requested position for the context menu
        """
        if not self.__hide_columns_popup:
            return

        model = self.model()
        if model.columnCount() > 1:
            menu = qt.QMenu(self)
            menu.setTitle("Display/hide columns")

            action = qt.QAction("Display/hide column", self)
            action.setEnabled(False)
            menu.addAction(action)

            for column in range(model.columnCount()):
                if column == 0:
                    # skip the main column
                    continue
                text = model.headerData(column, qt.Qt.Horizontal, qt.Qt.DisplayRole)
                action = qt.QAction("%s displayed" % text, self)
                action.setCheckable(True)
                action.setChecked(not self.isSectionHidden(column))
                action.toggled.connect(self.__genHideSectionEvent(column))
                menu.addAction(action)

            menu.popup(self.viewport().mapToGlobal(pos))

    def setSections(self, logicalIndexes):
        """
        Defines order of visible sections by logical indexes.

        Use `Hdf5TreeModel.NAME_COLUMN` to set the list.

        :param list logicalIndexes: List of logical indexes to display
        """
        for pos, column_id in enumerate(logicalIndexes):
            current_pos = self.visualIndex(column_id)
            self.moveSection(current_pos, pos)
            self.setSectionHidden(column_id, False)
        for column_id in set(range(self.model().columnCount())) - set(logicalIndexes):
            self.setSectionHidden(column_id, True)
