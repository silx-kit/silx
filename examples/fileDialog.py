#!/usr/bin/env python
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
"""
Example for the use of the ImageFileDialog.
"""

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "08/02/2018"

import logging
from silx.gui import qt
from silx.gui.dialog.ImageFileDialog import ImageFileDialog
from silx.gui.dialog.DataFileDialog import DataFileDialog


logging.basicConfig(level=logging.DEBUG)


class DialogExample(qt.QMainWindow):

    def __init__(self, parent=None):
        super(DialogExample, self).__init__(parent)

        widget = qt.QWidget(self)
        layout = qt.QGridLayout()
        widget.setLayout(layout)

        self.__stateImage = None
        self.__stateData = None

        layout.addWidget(qt.QLabel("<center>Qt QFileDialog</center>", self), 0, 0)
        layout.addWidget(qt.QLabel("<center>silx ImageFileDialog</center>", self), 0, 1)
        layout.addWidget(qt.QLabel("<center>silx DataFileDialog</center>", self), 0, 2)

        # Default dialog

        b0 = qt.QPushButton(self)
        b0.setMinimumHeight(50)
        b0.setText("Open a dialog")
        b0.clicked.connect(self.__openDefaultFileDialog)
        layout.addWidget(b0, 1, 0)

        # ImageFileDialog

        b1 = qt.QPushButton(self)
        b1.setMinimumHeight(50)
        b1.setText("Open a dialog")
        b1.clicked.connect(self.__openImageDialog)
        layout.addWidget(b1, 1, 1)

        b2 = qt.QPushButton(self)
        b2.setMinimumHeight(50)
        b2.setText("Open a dialog with state stored")
        b2.clicked.connect(self.__openImageDialogStoredState)
        layout.addWidget(b2, 2, 1)

        b3 = qt.QPushButton(self)
        b3.setMinimumHeight(50)
        b3.setText("Open a dialog at home")
        b3.clicked.connect(self.__openImageDialogAtHome)
        layout.addWidget(b3, 3, 1)

        b4 = qt.QPushButton(self)
        b4.setMinimumHeight(50)
        b4.setText("Open a dialog at computer root")
        b4.clicked.connect(self.openImageDialogAtComputer)
        layout.addWidget(b4, 4, 1)

        # DataFileDialog

        b1 = qt.QPushButton(self)
        b1.setMinimumHeight(50)
        b1.setText("Open a dialog")
        b1.clicked.connect(self.__openDataDialog)
        layout.addWidget(b1, 1, 2)

        b2 = qt.QPushButton(self)
        b2.setMinimumHeight(50)
        b2.setText("Open a dialog with state stored")
        b2.clicked.connect(self.__openDataDialogStoredState)
        layout.addWidget(b2, 2, 2)

        b3 = qt.QPushButton(self)
        b3.setMinimumHeight(50)
        b3.setText("Open a dialog at home")
        b3.clicked.connect(self.__openDataDialogAtHome)
        layout.addWidget(b3, 3, 2)

        b4 = qt.QPushButton(self)
        b4.setMinimumHeight(50)
        b4.setText("Open a dialog at computer root")
        b4.clicked.connect(self.openDataDialogAtComputer)
        layout.addWidget(b4, 4, 2)

        self.setCentralWidget(widget)

    def __printSelection(self, dialog):
        print("Selected file:", dialog.selectedFile())
        print("Selected data:", dialog.selectedData())
        print("Selected URL:", dialog.selectedUrl())

    def __openDefaultFileDialog(self):
        # Clear the dialog
        dialog = qt.QFileDialog(self)
        dialog.setAcceptMode(qt.QFileDialog.AcceptOpen)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            print(dialog.selectedFiles())
        else:
            print("Nothing selected")

    def __openImageDialog(self):
        # Clear the dialog
        dialog = ImageFileDialog(self)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def __openImageDialogStoredState(self):
        # Clear the dialog
        dialog = ImageFileDialog()
        if self.__stateImage is not None:
            dialog.restoreState(self.__stateImage)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        self.__stateImage = dialog.saveState()
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def __openImageDialogAtHome(self):
        # Clear the dialog
        path = qt.QDir.homePath()
        dialog = ImageFileDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def openImageDialogAtComputer(self):
        # Clear the dialog
        path = ""
        dialog = ImageFileDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def __openDataDialog(self):
        # Clear the dialog
        dialog = DataFileDialog(self)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def __openDataDialogStoredState(self):
        # Clear the dialog
        dialog = DataFileDialog()
        if self.__stateData is not None:
            dialog.restoreState(self.__stateData)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        self.__stateData = dialog.saveState()
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def __openDataDialogAtHome(self):
        # Clear the dialog
        path = qt.QDir.homePath()
        dialog = DataFileDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")

    def openDataDialogAtComputer(self):
        # Clear the dialog
        path = ""
        dialog = DataFileDialog()
        dialog.setDirectory(path)

        # Execute the dialog as modal
        result = dialog.exec_()

        # Reach the result
        if result:
            print("Selection:")
            self.__printSelection(dialog)
        else:
            print("Nothing selected")


def main():
    app = qt.QApplication([])
    example = DialogExample()
    example.show()
    app.exec_()


if __name__ == "__main__":
    main()
