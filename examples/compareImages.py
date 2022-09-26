#!/usr/bin/env python
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
"""Example demonstrating the use of the widget CompareImages
"""

import sys
import logging
import numpy
import argparse
import os

import silx.io
from silx.gui import qt
import silx.test.utils
from silx.io.url import DataUrl
from silx.gui.plot.CompareImages import CompareImages
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable

_logger = logging.getLogger(__name__)

import fabio

try:
    import PIL
except ImportError:
    _logger.debug("Backtrace", exc_info=True)
    PIL = None


class CompareImagesSelection(qt.QMainWindow):
    def __init__(self, backend):
        qt.QMainWindow.__init__(self, parent=None)
        self._plot = CompareImages(parent=self, backend=backend)

        self._selectionTable = UrlSelectionTable(parent=self)
        self._dockWidgetMenu = qt.QDockWidget(parent=self)
        self._dockWidgetMenu.layout().setContentsMargins(0, 0, 0, 0)
        self._dockWidgetMenu.setFeatures(qt.QDockWidget.DockWidgetMovable)
        self._dockWidgetMenu.setWidget(self._selectionTable)
        self.addDockWidget(qt.Qt.LeftDockWidgetArea, self._dockWidgetMenu)

        self.setCentralWidget(self._plot)

        self._selectionTable.sigImageAChanged.connect(self._updateImageA)
        self._selectionTable.sigImageBChanged.connect(self._updateImageB)

    def setUrls(self, urls):
        for url in urls:
            self._selectionTable.addUrl(url)

    def setFiles(self, files):
        urls = list()
        for _file in files:
            if os.path.isfile(_file):
                urls.append(DataUrl(file_path=_file, scheme=None))
        urls.sort(key=lambda url: url.path())
        window.setUrls(urls)
        window._selectionTable.setSelection(url_img_a=urls[0].path(),
                                            url_img_b=urls[1].path())

    def clear(self):
        self._plot.clear()
        self._selectionTable.clear()

    def _updateImageA(self, urlpath):
        self._updateImage(urlpath, self._plot.setImage1)

    def _updateImage(self, urlpath, fctptr):
        def getData():
            _url = silx.io.url.DataUrl(path=urlpath)
            for scheme in ('silx', 'fabio'):
                try:
                    dataImg = silx.io.utils.get_data(
                        silx.io.url.DataUrl(file_path=_url.file_path(),
                                            data_slice=_url.data_slice(),
                                            data_path=_url.data_path(),
                                            scheme=scheme))
                except:
                    _logger.debug("Error while loading image with %s" % scheme,
                                  exc_info=True)
                else:
                    # TODO: check is an image
                    return dataImg
            return None

        data = getData()
        if data is not None:
            fctptr(data)

    def _updateImageB(self, urlpath):
        self._updateImage(urlpath, self._plot.setImage2)


def createTestData():
    data = numpy.arange(100 * 100)
    data = (data % 100) / 5.0
    data = numpy.sin(data)
    data1 = data.copy()
    data1.shape = 100, 100
    data2 = silx.test.utils.add_gaussian_noise(data, stdev=0.1)
    data2.shape = 100, 100
    return data1, data2


def loadImage(filename):
    try:
        return silx.io.get_data(filename)
    except Exception:
        _logger.debug("Error while loading image with silx.io", exc_info=True)

    try:
        return fabio.open(filename).data
    except Exception:
        _logger.debug("Error while loading image with fabio", exc_info=True)

    if PIL is not None:
        try:
            return numpy.asarray(PIL.Image.open(filename))
        except Exception:
            _logger.debug("Error while loading image with PIL", exc_info=True)

    raise Exception("Impossible to load '%s' with the available image libraries" % filename)


file_description = """
Image data to compare (HDF5 file with path, EDF files, JPEG/PNG image files).
Data from HDF5 files can be accessed using dataset path and slicing as an URL: silx:../my_file.h5?path=/entry/data&slice=10
EDF file frames also can can be accessed using URL: fabio:../my_file.edf?slice=10
Using URL in command like usually have to be quoted: "URL".
"""


def createParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'files',
        nargs=argparse.ZERO_OR_MORE,
        help=file_description)
    parser.add_argument(
        '--debug',
        dest="debug",
        action="store_true",
        default=False,
        help='Set logging system in debug mode')
    parser.add_argument(
        '--testdata',
        dest="testdata",
        action="store_true",
        default=False,
        help='Use synthetic images to test the application')
    parser.add_argument(
        '--use-opengl-plot',
        dest="use_opengl_plot",
        action="store_true",
        default=False,
        help='Use OpenGL for plots (instead of matplotlib)')
    return parser


if __name__ == "__main__":
    parser = createParser()
    options = parser.parse_args(sys.argv[1:])

    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    if options.use_opengl_plot:
        backend = "gl"
    else:
        backend = "mpl"

    app = qt.QApplication([])
    if options.testdata or len(options.files) == 2:
        if options.testdata:
            _logger.info("Generate test data")
            data1, data2 = createTestData()
        else:
            data1 = loadImage(options.files[0])
            data2 = loadImage(options.files[1])
        window = CompareImages(backend=backend)
        window.setData(data1, data2)
    else:
        data = options.files
        window = CompareImagesSelection(backend=backend)
        window.setFiles(options.files)

    window.setVisible(True)
    app.exec()
