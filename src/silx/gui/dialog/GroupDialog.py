# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides a dialog widget to select a HDF5 group in a
tree.

.. autoclass:: GroupDialog
   :members: addFile, addGroup, getSelectedDataUrl, setMode

"""
from silx.gui import qt
from silx.gui.hdf5.Hdf5TreeView import Hdf5TreeView
import silx.io
from silx.io.url import DataUrl

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/03/2018"


class _Hdf5ItemSelectionDialog(qt.QDialog):
    SaveMode = 1
    """Mode used to set the HDF5 item selection dialog to *save* mode.
    This adds a text field to type in a new item name."""

    LoadMode = 2
    """Mode used to set the HDF5 item selection dialog to *load* mode.
    Only existing items of the HDF5 file can be selected in this mode."""

    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("HDF5 item selection")

        self._tree = Hdf5TreeView(self)
        self._tree.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self._tree.activated.connect(self._onActivation)
        self._tree.selectionModel().selectionChanged.connect(
            self._onSelectionChange)

        self._model = self._tree.findHdf5TreeModel()

        self._header = self._tree.header()

        self._newItemWidget = qt.QWidget(self)
        newItemLayout = qt.QVBoxLayout(self._newItemWidget)
        self._labelNewItem = qt.QLabel(self._newItemWidget)
        self._labelNewItem.setText("Create new item in selected group (optional):")
        self._lineEditNewItem = qt.QLineEdit(self._newItemWidget)
        self._lineEditNewItem.setToolTip(
                "Specify the name of a new item "
                "to be created in the selected group.")
        self._lineEditNewItem.textChanged.connect(
                self._onNewItemNameChange)
        newItemLayout.addWidget(self._labelNewItem)
        newItemLayout.addWidget(self._lineEditNewItem)

        _labelSelectionTitle = qt.QLabel(self)
        _labelSelectionTitle.setText("Current selection")
        self._labelSelection = qt.QLabel(self)
        self._labelSelection.setStyleSheet("color: gray")
        self._labelSelection.setWordWrap(True)
        self._labelSelection.setText("Select an item")

        buttonBox = qt.QDialogButtonBox()
        self._okButton = buttonBox.addButton(qt.QDialogButtonBox.Ok)
        self._okButton.setEnabled(False)
        buttonBox.addButton(qt.QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        vlayout = qt.QVBoxLayout(self)
        vlayout.addWidget(self._tree)
        vlayout.addWidget(self._newItemWidget)
        vlayout.addWidget(_labelSelectionTitle)
        vlayout.addWidget(self._labelSelection)
        vlayout.addWidget(buttonBox)
        self.setLayout(vlayout)

        self.setMinimumWidth(400)

        self._selectedUrl = None

    def _onSelectionChange(self, old, new):
        self._updateUrl()

    def _onNewItemNameChange(self, text):
        self._updateUrl()

    def _onActivation(self, idx):
        # double-click or enter press
        self.accept()

    def setMode(self, mode):
        """Set dialog mode DatasetDialog.SaveMode or DatasetDialog.LoadMode

        :param mode: DatasetDialog.SaveMode or DatasetDialog.LoadMode
        """
        if mode == self.LoadMode:
            # hide "Create new item" field
            self._lineEditNewItem.clear()
            self._newItemWidget.hide()
        elif mode == self.SaveMode:
            self._newItemWidget.show()
        else:
            raise ValueError("Invalid DatasetDialog mode %s" % mode)

    def addFile(self, path):
        """Add a HDF5 file to the tree.
        All groups it contains will be selectable in the dialog.

        :param str path: File path
        """
        self._model.insertFile(path)

    def addGroup(self, group):
        """Add a HDF5 group to the tree. This group and all its subgroups
        will be selectable in the dialog.

        :param h5py.Group group: HDF5 group
        """
        self._model.insertH5pyObject(group)

    def _updateUrl(self):
        nodes = list(self._tree.selectedH5Nodes())
        subgroupName = self._lineEditNewItem.text()
        if nodes:
            node = nodes[0]
            data_path = node.local_name
            if subgroupName.lstrip("/"):
                if not data_path.endswith("/"):
                    data_path += "/"
                data_path += subgroupName.lstrip("/")
            self._selectedUrl = DataUrl(file_path=node.local_filename,
                                        data_path=data_path)
            self._okButton.setEnabled(True)
            self._labelSelection.setText(
                    self._selectedUrl.path())

    def getSelectedDataUrl(self):
        """Return a :class:`DataUrl` with a file path and a data path.
        Return None if the dialog was cancelled.

        :return: :class:`silx.io.url.DataUrl` object pointing to the
            selected HDF5 item.
        """
        return self._selectedUrl


class GroupDialog(_Hdf5ItemSelectionDialog):
    """This :class:`QDialog` uses a :class:`silx.gui.hdf5.Hdf5TreeView` to
    provide a HDF5 group selection dialog.

    The information identifying the selected node is provided as a
    :class:`silx.io.url.DataUrl`.

    Example:

    .. code-block:: python

        dialog = GroupDialog()
        dialog.addFile(filepath1)
        dialog.addFile(filepath2)

        if dialog.exec_():
            print("File path: %s" % dialog.getSelectedDataUrl().file_path())
            print("HDF5 group path : %s " % dialog.getSelectedDataUrl().data_path())
        else:
            print("Operation cancelled :(")

    """
    def __init__(self, parent=None):
        _Hdf5ItemSelectionDialog.__init__(self, parent)

        # customization for groups
        self.setWindowTitle("HDF5 group selection")

        self._header.setSections([self._model.NAME_COLUMN,
                                  self._model.NODE_COLUMN,
                                  self._model.LINK_COLUMN])

    def _onActivation(self, idx):
        # double-click or enter press: filter for groups
        nodes = list(self._tree.selectedH5Nodes())
        node = nodes[0]
        if silx.io.is_group(node.h5py_object):
            self.accept()

    def _updateUrl(self):
        # overloaded to filter for groups
        nodes = list(self._tree.selectedH5Nodes())
        subgroupName = self._lineEditNewItem.text()
        if nodes:
            node = nodes[0]
            if silx.io.is_group(node.h5py_object):
                data_path = node.local_name
                if subgroupName.lstrip("/"):
                    if not data_path.endswith("/"):
                        data_path += "/"
                    data_path += subgroupName.lstrip("/")
                self._selectedUrl = DataUrl(file_path=node.local_filename,
                                            data_path=data_path)
                self._okButton.setEnabled(True)
                self._labelSelection.setText(
                        self._selectedUrl.path())
            else:
                self._selectedUrl = None
                self._okButton.setEnabled(False)
                self._labelSelection.setText("Select a group")
