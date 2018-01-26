#!/usr/bin/env python
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
"""Qt Hdf5 widget examples
"""

import logging
import sys
import tempfile
import numpy
import h5py

logging.basicConfig()
_logger = logging.getLogger("customHdf5TreeModel")
"""Module logger"""

from silx.gui import qt
import silx.gui.hdf5
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.hdf5.Hdf5TreeModel import Hdf5TreeModel


class CustomTooltips(qt.QIdentityProxyModel):
    """Custom the tooltip of the model by composition.

    It is a very stable way to custom it cause it uses the Qt API. Then it will
    not change according to the version of Silx.

    But it is not well integrated if you only want to add custom fields to the
    default tooltips.
    """

    def data(self, index, role=qt.Qt.DisplayRole):
        if role == qt.Qt.ToolTipRole:

            # Reach information from the node
            sourceIndex = self.mapToSource(index)
            sourceModel = self.sourceModel()
            originalTooltip = sourceModel.data(sourceIndex, qt.Qt.ToolTipRole)
            originalH5pyObject = sourceModel.data(sourceIndex, Hdf5TreeModel.H5PY_OBJECT_ROLE)

            # We can filter according to the column
            if sourceIndex.column() == Hdf5TreeModel.TYPE_COLUMN:
                return super(CustomTooltips, self).data(index, role)

            # Let's create our own tooltips
            template = u"""<html>
            <dl>
            <dt><b>Original</b></dt><dd>{original}</dd>
            <dt><b>Parent name</b></dt><dd>{parent_name}</dd>
            <dt><b>Name</b></dt><dd>{name}</dd>
            <dt><b>Power of 2</b></dt><dd>{pow_of_2}</dd>
            </dl>
            </html>
            """

            try:
                data = originalH5pyObject[()]
                if data.size <= 10:
                    result = data ** 2
                else:
                    result = "..."
            except Exception:
                result = "NA"

            info = dict(
                original=originalTooltip,
                parent_name=originalH5pyObject.parent.name,
                name=originalH5pyObject.name,
                pow_of_2=str(result)
            )
            return template.format(**info)

        return super(CustomTooltips, self).data(index, role)


_file_cache = {}


def get_hdf5_with_all_types():
    ID = "alltypes"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("arrays")
    g.create_dataset("scalar", data=10)
    g.create_dataset("list", data=numpy.arange(10))
    base_image = numpy.arange(10**2).reshape(10, 10)
    images = [base_image,
              base_image.T,
              base_image.size - 1 - base_image,
              base_image.size - 1 - base_image.T]
    dtype = images[0].dtype
    data = numpy.empty((10 * 10, 10, 10), dtype=dtype)
    for i in range(10 * 10):
        data[i] = images[i % 4]
    data.shape = 10, 10, 10, 10
    g.create_dataset("image", data=data[0, 0])
    g.create_dataset("cube", data=data[0])
    g.create_dataset("hypercube", data=data)
    g = h5.create_group("dtypes")
    g.create_dataset("int32", data=numpy.int32(10))
    g.create_dataset("int64", data=numpy.int64(10))
    g.create_dataset("float32", data=numpy.float32(10))
    g.create_dataset("float64", data=numpy.float64(10))
    g.create_dataset("string_", data=numpy.string_("Hi!"))
    # g.create_dataset("string0",data=numpy.string0("Hi!\x00"))

    g.create_dataset("bool", data=True)
    g.create_dataset("bool2", data=False)
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


class Hdf5TreeViewExample(qt.QMainWindow):
    """
    This window show an example of use of a Hdf5TreeView.

    The tree is initialized with a list of filenames. A panel allow to play
    with internal property configuration of the widget, and a text screen
    allow to display events.
    """

    def __init__(self, filenames=None):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx HDF5 widget example")

        self.__asyncload = False
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        """Silx HDF5 TreeView"""

        self.__sourceModel = self.__treeview.model()
        """Store the source model"""

        self.__text = qt.QTextEdit(self)
        """Widget displaying information"""

        self.__dataViewer = DataViewerFrame(self)
        vSpliter = qt.QSplitter(qt.Qt.Vertical)
        vSpliter.addWidget(self.__dataViewer)
        vSpliter.addWidget(self.__text)
        vSpliter.setSizes([10, 0])

        spliter = qt.QSplitter(self)
        spliter.addWidget(self.__treeview)
        spliter.addWidget(vSpliter)
        spliter.setStretchFactor(1, 1)

        main_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(spliter)
        layout.addWidget(self.createTreeViewConfigurationPanel(self, self.__treeview))
        layout.setStretchFactor(spliter, 1)
        main_panel.setLayout(layout)

        self.setCentralWidget(main_panel)

        # append all files to the tree
        for file_name in filenames:
            self.__treeview.findHdf5TreeModel().appendFile(file_name)

        self.__treeview.activated.connect(self.displayData)

    def displayData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            # data is a hdf5.H5Node object
            # data.h5py_object is a Group/Dataset object (from h5py, spech5, fabioh5)
            # The dataviewer can display both
            self.__dataViewer.setData(data)

    def __fileCreated(self, filename):
        if self.__asyncload:
            self.__treeview.findHdf5TreeModel().insertFileAsync(filename)
        else:
            self.__treeview.findHdf5TreeModel().insertFile(filename)

    def __hdf5ComboChanged(self, index):
        function = self.__hdf5Combo.itemData(index)
        self.__createHdf5Button.setCallable(function)

    def __edfComboChanged(self, index):
        function = self.__edfCombo.itemData(index)
        self.__createEdfButton.setCallable(function)

    def __useCustomLabel(self):
        customModel = CustomTooltips(self.__treeview)
        customModel.setSourceModel(self.__sourceModel)
        self.__treeview.setModel(customModel)

    def __useOriginalModel(self):
        self.__treeview.setModel(self.__sourceModel)

    def createTreeViewConfigurationPanel(self, parent, treeview):
        """Create a configuration panel to allow to play with widget states"""
        panel = qt.QWidget(parent)
        panel.setLayout(qt.QHBoxLayout())

        content = qt.QGroupBox("Create HDF5", panel)
        content.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(content)

        combo = qt.QComboBox()
        combo.addItem("Containing all types", get_hdf5_with_all_types)
        combo.activated.connect(self.__hdf5ComboChanged)
        content.layout().addWidget(combo)

        button = ThreadPoolPushButton(content, text="Create")
        button.setCallable(combo.itemData(combo.currentIndex()))
        button.succeeded.connect(self.__fileCreated)
        content.layout().addWidget(button)

        self.__hdf5Combo = combo
        self.__createHdf5Button = button

        content.layout().addStretch(1)

        option = qt.QGroupBox("Custom model", panel)
        option.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(option)

        button = qt.QPushButton("Original model")
        button.clicked.connect(self.__useOriginalModel)
        option.layout().addWidget(button)

        button = qt.QPushButton("Custom tooltips by composition")
        button.clicked.connect(self.__useCustomLabel)
        option.layout().addWidget(button)

        option.layout().addStretch(1)

        panel.layout().addStretch(1)
        return panel


def main(filenames):
    """
    :param filenames: list of file paths
    """
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    window = Hdf5TreeViewExample(filenames)
    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    sys.exit(result)


if __name__ == "__main__":
    main(sys.argv[1:])
