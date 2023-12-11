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
"""GUI to compare images"""

import sys
import logging
import argparse
import silx
from silx.gui import qt
from silx.app.utils import parseutils
from silx.app.compare.CompareImagesWindow import CompareImagesWindow

_logger = logging.getLogger(__name__)


file_description = """
Image data to compare (HDF5 file with path, EDF files, JPEG/PNG image files).
Data from HDF5 files can be accessed using dataset path and slicing as an URL: silx:../my_file.h5?path=/entry/data&slice=10
EDF file frames also can can be accessed using URL: fabio:../my_file.edf?slice=10
Using URL in command like usually have to be quoted: "URL".
"""


def createParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs=argparse.ZERO_OR_MORE, help=file_description)
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Set logging system in debug mode",
    )
    parser.add_argument(
        "--use-opengl-plot",
        dest="use_opengl_plot",
        action="store_true",
        default=False,
        help="Use OpenGL for plots (instead of matplotlib)",
    )
    return parser


def mainQt(options):
    """Part of the main depending on Qt"""
    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    if options.use_opengl_plot:
        backend = "gl"
    else:
        backend = "mpl"

    settings = qt.QSettings(
        qt.QSettings.IniFormat, qt.QSettings.UserScope, "silx", "silx-compare", None
    )

    urls = list(parseutils.filenames_to_dataurls(options.files))

    if options.use_opengl_plot:
        # It have to be done after the settings (after the Viewer creation)
        silx.config.DEFAULT_PLOT_BACKEND = "opengl"

    app = qt.QApplication([])
    window = CompareImagesWindow(backend=backend, settings=settings)
    window.setAttribute(qt.Qt.WA_DeleteOnClose, True)

    # Note: Have to be before setUrls to have a proper resetZoom
    window.setVisible(True)

    window.setUrls(urls)

    app.exec()


def main(argv):
    parser = createParser()
    options = parser.parse_args(argv[1:])
    mainQt(options)


if __name__ == "__main__":
    main(sys.argv)
