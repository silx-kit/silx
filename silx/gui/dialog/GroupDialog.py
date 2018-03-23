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
   :show-inheritance:
   :members:


"""
from silx.gui import qt
from silx.gui.hdf5.Hdf5TreeView import Hdf5TreeView
import silx.io
from silx.io.url import DataUrl

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/03/2018"


class GroupDialog(qt.QDialog):
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
            print("File path: %s" % dialog.selected_url.file_path())
            print("HDF5 group path : %s " % dialog.selected_url.data_path())
        else:
            print("Operation cancelled :(")

    """
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("HDF5 group selection")

        self.tree = Hdf5TreeView(self)
        self.tree.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.model = self.tree.findHdf5TreeModel()

        self.tree.activated.connect(self._onActivation)
        self.tree.selectionModel().selectionChanged.connect(
            self._onSelectionChange)

        buttonBox = qt.QDialogButtonBox()
        self.okButton = buttonBox.addButton(qt.QDialogButtonBox.Ok)
        self.okButton.setEnabled(False)
        buttonBox.addButton(qt.QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.statusBar = qt.QStatusBar()
        self.statusBar.setStyleSheet("color: gray")
        self.statusBar.showMessage("Select a group")

        vlayout = qt.QVBoxLayout(self)
        vlayout.addWidget(self.tree)
        vlayout.addWidget(buttonBox)
        vlayout.addWidget(self.statusBar)
        self.setLayout(vlayout)

        self.selected_url = None
        """None or a :class:`DataUrl` with a file path and a data path."""

    def addFile(self, path):
        """Add a HDF5 file to the tree.
        All groups it contains will be selectable in the dialog.

        :param str path: File path
        """
        self.model.insertFile(path)

    def addGroup(self, group):
        """Add a HDF5 group to the tree. This group and all its subgroups
        will be selectable in the dialog.

        :param h5py.Group group: HDF5 group
        :return:
        """
        self.model.insertH5pyObject(group)

    def _onActivation(self, idx):
        # double-click or enter press
        nodes = list(self.tree.selectedH5Nodes())
        node = nodes[0]
        if silx.io.is_group(node.h5py_object):
            self.accept()

    def _onSelectionChange(self, old, new):
        nodes = list(self.tree.selectedH5Nodes())
        if nodes:
            node = nodes[0]
            if silx.io.is_group(node.h5py_object):
                self.selected_url = DataUrl(file_path=node.local_filename,
                                            data_path=node.local_name)
                self.okButton.setEnabled(True)
                self.statusBar.showMessage(
                        "DataUrl('%s')" % self.selected_url.path())
            else:
                self.selected_url = None
                self.okButton.setEnabled(False)
                self.statusBar.showMessage("Select a group")
