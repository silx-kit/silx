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
"""This module provides a dialog widget to select a HDF5 dataset in a
tree.

.. autoclass:: DatasetDialog
   :members: addFile, addGroup, getSelectedDataUrl, setMode

"""
from .GroupDialog import _Hdf5ItemSelectionDialog
import silx.io
from silx.io.url import DataUrl


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/09/2018"


class DatasetDialog(_Hdf5ItemSelectionDialog):
    """This :class:`QDialog` uses a :class:`silx.gui.hdf5.Hdf5TreeView` to
    provide a HDF5 dataset selection dialog.

    The information identifying the selected node is provided as a
    :class:`silx.io.url.DataUrl`.

    Example:

    .. code-block:: python

        dialog = DatasetDialog()
        dialog.addFile(filepath1)
        dialog.addFile(filepath2)

        if dialog.exec_():
            print("File path: %s" % dialog.getSelectedDataUrl().file_path())
            print("HDF5 dataset path : %s " % dialog.getSelectedDataUrl().data_path())
        else:
            print("Operation cancelled :(")

    """
    def __init__(self, parent=None):
        _Hdf5ItemSelectionDialog.__init__(self, parent)

        # customization for groups
        self.setWindowTitle("HDF5 dataset selection")

        self._header.setSections([self._model.NAME_COLUMN,
                                  self._model.NODE_COLUMN,
                                  self._model.LINK_COLUMN,
                                  self._model.TYPE_COLUMN,
                                  self._model.SHAPE_COLUMN])
        self._selectDatasetStatusText = "Select a dataset or type a new dataset name"

    def setMode(self, mode):
        """Set dialog mode DatasetDialog.SaveMode or DatasetDialog.LoadMode

        :param mode: DatasetDialog.SaveMode or DatasetDialog.LoadMode
        """
        _Hdf5ItemSelectionDialog.setMode(self, mode)
        if mode == DatasetDialog.SaveMode:
            self._selectDatasetStatusText = "Select a dataset or type a new dataset name"
        elif mode == DatasetDialog.LoadMode:
            self._selectDatasetStatusText = "Select a dataset"

    def _onActivation(self, idx):
        # double-click or enter press: filter for datasets
        nodes = list(self._tree.selectedH5Nodes())
        node = nodes[0]
        if silx.io.is_dataset(node.h5py_object):
            self.accept()

    def _updateUrl(self):
        # overloaded to filter for datasets
        nodes = list(self._tree.selectedH5Nodes())
        newDatasetName = self._lineEditNewItem.text()
        isDatasetSelected = False
        if nodes:
            node = nodes[0]
            if silx.io.is_dataset(node.h5py_object):
                data_path = node.local_name
                isDatasetSelected = True
            elif silx.io.is_group(node.h5py_object):
                data_path = node.local_name
                if newDatasetName.lstrip("/"):
                    if not data_path.endswith("/"):
                        data_path += "/"
                    data_path += newDatasetName.lstrip("/")
                    isDatasetSelected = True

            if isDatasetSelected:
                self._selectedUrl = DataUrl(file_path=node.local_filename,
                                            data_path=data_path)
                self._okButton.setEnabled(True)
                self._labelSelection.setText(
                        self._selectedUrl.path())
            else:
                self._selectedUrl = None
                self._okButton.setEnabled(False)
                self._labelSelection.setText(self._selectDatasetStatusText)
