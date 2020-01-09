#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""
Example for the use of the ImageFileDialog.
"""

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/02/2018"

import enum
import logging
from silx.gui import qt
from silx.gui.dialog.ImageFileDialog import ImageFileDialog
from silx.gui.dialog.DataFileDialog import DataFileDialog
import silx.io


logging.basicConfig()


class Mode(enum.Enum):
    DEFAULT_FILEDIALOG = 0
    IMAGEFILEDIALOG = 1
    DATAFILEDIALOG = 2
    DATAFILEDIALOG_DATASET = 3
    DATAFILEDIALOG_GROUP = 4
    DATAFILEDIALOG_NXENTRY = 5


class DialogExample(qt.QMainWindow):

    def __init__(self, parent=None):
        super(DialogExample, self).__init__(parent)

        self.__state = {}

        centralWidget = qt.QWidget(self)
        layout = qt.QHBoxLayout()
        centralWidget.setLayout(layout)

        options = self.createOptions()
        layout.addWidget(options)

        buttonGroup = qt.QGroupBox()
        buttonGroup.setTitle("Create dialog")
        layout.addWidget(buttonGroup)
        buttonLayout = qt.QVBoxLayout()
        buttonGroup.setLayout(buttonLayout)

        # ImageFileDialog

        b1 = qt.QPushButton(self)
        b1.setMinimumHeight(50)
        b1.setText("Open a dialog")
        b1.clicked.connect(self.openDialog)
        buttonLayout.addWidget(b1)

        b2 = qt.QPushButton(self)
        b2.setMinimumHeight(50)
        b2.setText("Open a dialog with state stored")
        b2.clicked.connect(self.openDialogStoredState)
        buttonLayout.addWidget(b2)

        b3 = qt.QPushButton(self)
        b3.setMinimumHeight(50)
        b3.setText("Open a dialog at home")
        b3.clicked.connect(self.openDialogAtHome)
        buttonLayout.addWidget(b3)

        b4 = qt.QPushButton(self)
        b4.setMinimumHeight(50)
        b4.setText("Open a dialog at computer root")
        b4.clicked.connect(self.openDialogAtComputer)
        buttonLayout.addWidget(b4)

        self.setCentralWidget(centralWidget)

    def createOptions(self):
        panel = qt.QGroupBox()
        panel.setTitle("Options")
        layout = qt.QVBoxLayout()
        panel.setLayout(layout)
        group = qt.QButtonGroup(panel)

        radio = qt.QRadioButton(panel)
        radio.setText("Qt QFileDialog")
        radio.setProperty("Mode", Mode.DEFAULT_FILEDIALOG)
        group.addButton(radio)
        layout.addWidget(radio)

        radio = qt.QRadioButton(panel)
        radio.setText("silx ImageFileDialog")
        radio.setProperty("Mode", Mode.IMAGEFILEDIALOG)
        group.addButton(radio)
        layout.addWidget(radio)

        radio = qt.QRadioButton(panel)
        radio.setChecked(True)
        radio.setText("silx DataFileDialog")
        radio.setProperty("Mode", Mode.DATAFILEDIALOG)
        group.addButton(radio)
        layout.addWidget(radio)

        radio = qt.QRadioButton(panel)
        radio.setText("silx DataFileDialog (filter=dataset)")
        radio.setProperty("Mode", Mode.DATAFILEDIALOG_DATASET)
        group.addButton(radio)
        layout.addWidget(radio)

        radio = qt.QRadioButton(panel)
        radio.setText("silx DataFileDialog (filter=group)")
        radio.setProperty("Mode", Mode.DATAFILEDIALOG_GROUP)
        group.addButton(radio)
        layout.addWidget(radio)

        radio = qt.QRadioButton(panel)
        radio.setText("silx DataFileDialog (filter=NXentry)")
        radio.setProperty("Mode", Mode.DATAFILEDIALOG_NXENTRY)
        group.addButton(radio)
        layout.addWidget(radio)

        self.__options = group
        return panel

    def printResult(self, dialog, result):
        if not result:
            print("Nothing selected")
            return

        print("Selection:")
        if isinstance(dialog, qt.QFileDialog):
            print("- Files: %s" % dialog.selectedFiles())
        elif isinstance(dialog, ImageFileDialog):
            print("- File: %s" % dialog.selectedFile())
            print("- URL: %s" % dialog.selectedUrl())
            print("- Data URL: %s" % dialog.selectedDataUrl())
            image = dialog.selectedImage()
            print("- Image: <dtype: %s, shape: %s>" % (image.dtype, image.shape))
        elif isinstance(dialog, DataFileDialog):
            print("- File: %s" % dialog.selectedFile())
            print("- URL: %s" % dialog.selectedUrl())
            print("- Data URL: %s" % dialog.selectedDataUrl())
            try:
                data = dialog.selectedData()
                print("- Data: <dtype: %s, shape: %s>" % (data.dtype, data.shape))
            except Exception as e:
                print("- Data: %s" % e)

            url = dialog.selectedDataUrl()
            with silx.io.open(url.file_path()) as h5:
                node = h5[url.data_path()]
                print("- Node: %s" % node)
        else:
            assert(False)

    def createDialog(self):
        print("")
        print("-------------------------")
        print("----- Create dialog -----")
        print("-------------------------")
        button = self.__options.checkedButton()
        mode = button.property("Mode")
        if mode == Mode.DEFAULT_FILEDIALOG:
            dialog = qt.QFileDialog(self)
            dialog.setAcceptMode(qt.QFileDialog.AcceptOpen)
        elif mode == Mode.IMAGEFILEDIALOG:
            dialog = ImageFileDialog(self)
        elif mode == Mode.DATAFILEDIALOG:
            dialog = DataFileDialog(self)
        elif mode == Mode.DATAFILEDIALOG_DATASET:
            dialog = DataFileDialog(self)
            dialog.setFilterMode(DataFileDialog.FilterMode.ExistingDataset)
        elif mode == Mode.DATAFILEDIALOG_GROUP:
            dialog = DataFileDialog(self)
            dialog.setFilterMode(DataFileDialog.FilterMode.ExistingGroup)
        elif mode == Mode.DATAFILEDIALOG_NXENTRY:
            def customFilter(obj):
                if "NX_class" in obj.attrs:
                    return obj.attrs["NX_class"] in [b"NXentry", u"NXentry"]
                return False
            dialog = DataFileDialog(self)
            dialog.setFilterMode(DataFileDialog.FilterMode.ExistingGroup)
            dialog.setFilterCallback(customFilter)
        else:
            assert(False)
        return dialog

    def openDialog(self):
        # Clear the dialog
        dialog = self.createDialog()

        # Execute the dialog as modal
        result = dialog.exec_()
        self.printResult(dialog, result)

    def openDialogStoredState(self):
        # Clear the dialog
        dialog = self.createDialog()
        if dialog.__class__ in self.__state:
            dialog.restoreState(self.__state[dialog.__class__])

        # Execute the dialog as modal
        result = dialog.exec_()
        self.__state[dialog.__class__] = dialog.saveState()
        self.printResult(dialog, result)

    def openDialogAtHome(self):
        # Clear the dialog
        path = qt.QDir.homePath()
        dialog = self.createDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()
        self.printResult(dialog, result)

    def openDialogAtComputer(self):
        # Clear the dialog
        path = ""
        dialog = self.createDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()
        self.printResult(dialog, result)


def main():
    app = qt.QApplication([])
    example = DialogExample()
    example.show()
    app.exec_()


if __name__ == "__main__":
    main()
