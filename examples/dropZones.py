#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
Example of drop zone supporting application/x-silx-uri.

This example illustrates the support of drag&drop of silx URLs.
It provides 2 URLs (corresponding to 2 datasets) that can be dragged to
either a :class:`PlotWidget` or a QLable displaying the URL information.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/01/2019"

import logging
import os
import tempfile

import h5py
import numpy

import silx.io
from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)
logging.basicConfig()


class DropPlotWidget(PlotWidget):
    """PlotWidget accepting drop of silx URLs"""

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
    """Label widget accepting drop of silx URLs"""

    DEFAULT_TEXT = "Drop an URL here to display information"

    def __init__(self, parent=None):
        qt.QLabel.__init__(self, parent)
        self.setAcceptDrops(True)
        self.setUrl(silx.io.url.DataUrl())

    def setUrl(self, url):
        template = ("<html>URL information (drop an URL here to parse its information):<ul>"
                    "<li><b>file_path</b>: {file_path}</li>"
                    "<li><b>data_path</b>: {data_path}</li>"
                    "<li><b>data_slice</b>: {data_slice}</li>"
                    "<li><b>scheme</b>: {scheme}</li>"
                    "</ul></html>"
                    )

        text = template.format(
            file_path=url.file_path(),
            data_path=url.data_path(),
            data_slice=url.data_slice(),
            scheme=url.scheme())
        self.setText(text)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
        self.setUrl(url)
        event.acceptProposedAction()


class DragLabel(qt.QLabel):
    """Label widget providing a silx URL to drag"""

    def __init__(self, parent=None, url=None):
        self._url = url
        qt.QLabel.__init__(self, parent)
        self.setText('-' if url is None else "- " + self._url.path())

    def mousePressEvent(self, event):
        if event.button() == qt.Qt.LeftButton and self._url is not None:
            mimeData = qt.QMimeData()
            mimeData.setText(self._url.path())
            mimeData.setData(
                "application/x-silx-uri",
                self._url.path().encode(encoding='utf-8'))
            drag = qt.QDrag(self)
            drag.setMimeData(mimeData)
            dropAction = drag.exec()


class DragAndDropExample(qt.QMainWindow):
    """Main window of the example"""

    def __init__(self, parent=None, urls=()):
        super(DragAndDropExample, self).__init__(parent)
        centralWidget = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        centralWidget.setLayout(layout)
        layout.addWidget(qt.QLabel(
            "Drag and drop one of the following URLs on the plot or on the URL information zone:",
            self))
        for url in urls:
            layout.addWidget(DragLabel(parent=self, url=url))

        layout.addWidget(DropPlotWidget(parent=self))
        layout.addWidget(DropLabel(parent=self))

        self.setCentralWidget(centralWidget)


def main():
    app = qt.QApplication([])
    with tempfile.TemporaryDirectory() as tempdir:
        # Create temporary file with datasets
        filename = os.path.join(tempdir, "file.h5")
        with h5py.File(filename, "w") as f:
            f['image'] = numpy.arange(10000.).reshape(100, 100)
            f['curve'] = numpy.sin(numpy.linspace(0, 2*numpy.pi, 1000))

        # Create widgets
        example = DragAndDropExample(urls=(
            silx.io.url.DataUrl(file_path=filename, data_path='/image', scheme="silx"),
            silx.io.url.DataUrl(file_path=filename, data_path='/curve', scheme="silx")))
        example.setWindowTitle("Drag&Drop URLs sample code")
        example.show()
        app.exec()


if __name__ == "__main__":
    main()
