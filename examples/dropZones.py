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
Example of drop zone supporting application/x-silx-uri
"""

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/01/2019"

import logging
import silx.io
from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)
logging.basicConfig()


class DropPlotWidget(PlotWidget):

    def __init__(self, parent=None, backend=None):
        PlotWidget.__init__(self, parent=parent, backend=backend)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        silxUrl = byteString.data().decode("utf-8")
        with silx.io.open(silxUrl) as h5:
            if silx.io.is_dataset(h5):
                dataset = h5[...]
            else:
                _logger.error("Unsupported URI")
                dataset = None

        if dataset is not None:
            if dataset.ndim == 1:
                self.clear()
                self.addCurve(y=dataset, x=range(dataset.size))
                event.acceptProposedAction()
            elif dataset.ndim == 2:
                self.clear()
                self.addImage(data=dataset)
                event.acceptProposedAction()
            else:
                _logger.error("Unsupported dataset")


class DropLabel(qt.QLabel):

    def __init__(self, parent=None, backend=None):
        qt.QLabel.__init__(self)
        self.setAcceptDrops(True)
        self.setText("Drop something here")

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        silxUrl = byteString.data().decode("utf-8")
        url = silx.io.url.DataUrl(silxUrl)
        self.setText(url.path())

        toolTipTemplate = ("<html><ul>"
                           "<li><b>file_path</b>: {file_path}</li>"
                           "<li><b>data_path</b>: {data_path}</li>"
                           "<li><b>data_slice</b>: {data_slice}</li>"
                           "<li><b>scheme</b>: {scheme}</li>"
                           "</html>"
                           "</ul></html>"
                           )

        toolTip = toolTipTemplate.format(
            file_path=url.file_path(),
            data_path=url.data_path(),
            data_slice=url.data_slice(),
            scheme=url.scheme())

        self.setToolTip(toolTip)
        event.acceptProposedAction()


class DropExample(qt.QMainWindow):

    def __init__(self, parent=None):
        super(DropExample, self).__init__(parent)
        centralWidget = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        centralWidget.setLayout(layout)
        layout.addWidget(DropPlotWidget(parent=self))
        layout.addWidget(DropLabel(parent=self))
        self.setCentralWidget(centralWidget)


def main():
    app = qt.QApplication([])
    example = DropExample()
    example.show()
    app.exec_()


if __name__ == "__main__":
    main()
