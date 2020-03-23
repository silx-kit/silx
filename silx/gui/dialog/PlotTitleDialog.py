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
"""This module provides a dialog widget to set a title for the shown plot.

.. autoclass:: PlotTitleDialog
   :members: getCustomTitleTextArea, getFontSize, setHdf5data

"""

from silx.gui import qt

__authors__ = ["P. Kügler"]
__license__ = "MIT"
__date__ = "12/03/2020"


class PlotTitleDialog(qt.QDialog):
    def __init__(self, hdf5data, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Set Plot Title")
        app = qt.QApplication.instance()
        self.setWindowIcon(app.windowIcon())

        self._hdf5data = hdf5data
        fontSizes = ["8", "10", "12", "14", "16", "32"]

        buttonBox = qt.QDialogButtonBox()
        self._okButton = buttonBox.addButton(qt.QDialogButtonBox.Ok)
        self._okButton.setEnabled(True)
        buttonBox.addButton(qt.QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        showTitleLabel = qt.QLabel("Show title")
        self._showTitleRadioButton = qt.QCheckBox()
        showTitleLayout = qt.QHBoxLayout()
        showTitleLayout.addWidget(self._showTitleRadioButton)
        showTitleLayout.addWidget(showTitleLabel)
        showTitleLayout.addStretch(1)
        self._showTitleRadioButton.setChecked(True)
        self._showTitleRadioButton.toggled.connect(lambda: self._showTitleToggled())

        fileNameLabel = qt.QLabel("Include file name")
        self._fileNameRadioButton = qt.QCheckBox()
        self._fileNameLayout = qt.QHBoxLayout()
        self._fileNameLayout.addWidget(self._fileNameRadioButton)
        self._fileNameLayout.addWidget(fileNameLabel)
        self._fileNameLayout.addStretch(1)
        self._fileNameRadioButton.toggled.connect(lambda: self._fileNameToggled())

        fileDirectoryLabel = qt.QLabel("Include file directory")
        self._fileDirectoryRadioButton = qt.QCheckBox()
        self._fileDirectoryLayout = qt.QHBoxLayout()
        self._fileDirectoryLayout.addWidget(self._fileDirectoryRadioButton)
        self._fileDirectoryLayout.addWidget(fileDirectoryLabel)
        self._fileDirectoryLayout.addStretch(1)
        self._fileDirectoryRadioButton.toggled.connect(lambda: self._fileDirectoryToggled())

        datasetPathLabel = qt.QLabel("Include dataset path")
        self._datasetPathRadioButton = qt.QCheckBox()
        self._datasetPathLayout = qt.QHBoxLayout()
        self._datasetPathLayout.addWidget(self._datasetPathRadioButton)
        self._datasetPathLayout.addWidget(datasetPathLabel)
        self._datasetPathLayout.addStretch(1)
        self._datasetPathRadioButton.toggled.connect(lambda: self._dataSetPathToggled())

        customeTitleLabel = qt.QLabel("Use custom title")
        self._customeTitleRadioButton = qt.QCheckBox()
        self._customeTitleLayout = qt.QHBoxLayout()
        self._customeTitleLayout.addWidget(self._customeTitleRadioButton)
        self._customeTitleLayout.addWidget(customeTitleLabel)
        self._customeTitleLayout.addStretch(1)
        self._customeTitleRadioButton.toggled.connect(lambda: self._customeTitleToggeld())

        self._customTitleTextArea = qt.QTextEdit()
        self._customTitleTextArea.setEnabled(False)

        fontSizeLabel = qt.QLabel("Fontsize")
        self._fontSizeCombobox = qt.QComboBox()
        self._fontSizeCombobox.addItems(fontSizes)
        self._fontSizeLayout = qt.QHBoxLayout()
        self._fontSizeLayout.addWidget(fontSizeLabel)
        self._fontSizeLayout.addWidget(self._fontSizeCombobox)

        vlayout = qt.QVBoxLayout(self)
        vlayout.addLayout(showTitleLayout)
        vlayout.addLayout(self._fileNameLayout)
        vlayout.addLayout(self._fileDirectoryLayout)
        vlayout.addLayout(self._datasetPathLayout)
        vlayout.addLayout(self._customeTitleLayout)
        vlayout.addWidget(self._customTitleTextArea)
        vlayout.addLayout(self._fontSizeLayout)
        vlayout.addWidget(buttonBox)
        self.setLayout(vlayout)
        self._datasetPathRadioButton.setChecked(True)
        self._setLayoutsEnabled(False)
        self._setLayoutsEnabled(self._showTitleRadioButton.isChecked())

    def getCustomTitleTextArea(self):
        """Returns the current text for the plot title.

        :return: string for the plot title
        """
        return self._customTitleTextArea

    def getFontSize(self):
        """Returns the current chosen size of the font
        for the plot title.

        :return: int for the font size
        """
        return int(self._fontSizeCombobox.currentText())

    def setHdf5data(self, data):
        """Sets the hdf5data to get information for the plot title

        :param data: h5py.Dataset
        """
        self._hdf5data = data

    def _showTitleToggled(self):
        self._setLayoutsEnabled(self._showTitleRadioButton.isChecked())
        if not self._showTitleRadioButton.isChecked():
            self._uncheckRadioButtons()
            self._customTitleTextArea.setText("")

    def _fileNameToggled(self):
        if self._fileNameRadioButton.isChecked():
            self._customTitleTextArea.setText( self._getFileName() + self._customTitleTextArea.toPlainText())
            for index in range(self._fileDirectoryLayout.count() - 1):
                self._fileDirectoryLayout.itemAt(index).widget().setEnabled(True)
        else:
            self._fileDirectoryRadioButton.setChecked(False)
            if self._getFileName() in self._customTitleTextArea.toPlainText():
                self._customTitleTextArea.setText(
                    self._cutOut(self._customTitleTextArea.toPlainText(), self._getFileName()))
            for index in range(self._fileDirectoryLayout.count() - 1):
                self._fileDirectoryLayout.itemAt(index).widget().setEnabled(False)

    def _fileDirectoryToggled(self):
        if self._fileDirectoryRadioButton.isChecked():
            self._customTitleTextArea.setText(self._getFileDirectory() + self._customTitleTextArea.toPlainText())
        elif self._getFileDirectory() in self._customTitleTextArea.toPlainText():
            self._customTitleTextArea.setText(
                self._cutOut(self._customTitleTextArea.toPlainText(), self._getFileDirectory()))

    def _dataSetPathToggled(self):
        if self._datasetPathRadioButton.isChecked():
            self._customTitleTextArea.setText(self._customTitleTextArea.toPlainText() + str(self._hdf5data.name))
        elif self._hdf5data.name in self._customTitleTextArea.toPlainText():
            self._customTitleTextArea.setText(
                self._cutOut(self._customTitleTextArea.toPlainText(), self._hdf5data.name))

    def _customeTitleToggeld(self):
        if self._customeTitleRadioButton.isChecked():
            self._customTitleTextArea.setEnabled(True)
        else:
            self._customTitleTextArea.setEnabled(False)

    def _setLayoutsEnabled(self, value):
        for index in range(self._fileNameLayout.count() - 1):
            self._fileNameLayout.itemAt(index).widget().setEnabled(value)
            self._customeTitleLayout.itemAt(index).widget().setEnabled(value)
            self._datasetPathLayout.itemAt(index).widget().setEnabled(value)
            if not value:
                self._fileDirectoryLayout.itemAt(index).widget().setEnabled(value)

    def _uncheckRadioButtons(self):
        self._fileNameRadioButton.setChecked(False)
        self._fileDirectoryRadioButton.setChecked(False)
        self._datasetPathRadioButton.setChecked(False)
        self._customeTitleRadioButton.setChecked(False)

    def _getFileName(self):
        filepath = str(self._hdf5data.file.filename)
        for index in range(len(filepath)):
            if filepath[-index - 1] == '/':
                return filepath[-index - 1:len(filepath)]

    def _getFileDirectory(self):
        filepath = str(self._hdf5data.file.filename)
        for index in range(len(filepath)):
            if filepath[-index - 1] == '/':
                return filepath[:-index - 1]

    def _cutOut(self, title, cutting):
        prefix = title[:title.index(cutting)]
        suffix = title[title.index(cutting) + len(cutting):]
        return prefix + suffix
