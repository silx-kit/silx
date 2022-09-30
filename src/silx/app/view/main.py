# /*##########################################################################
# Copyright (C) 2016-2022 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Module containing launcher of the `silx view` application"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/01/2019"

import argparse
import glob
import logging
import os
import signal
import sys
from typing import Generator, Iterable


_logger = logging.getLogger(__name__)
"""Module logger"""


def createParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'files',
        nargs=argparse.ZERO_OR_MORE,
        help='Data file to show (h5 file, edf files, spec files)')
    parser.add_argument(
        '--debug',
        dest="debug",
        action="store_true",
        default=False,
        help='Set logging system in debug mode')
    parser.add_argument(
        '--use-opengl-plot',
        dest="use_opengl_plot",
        action="store_true",
        default=False,
        help='Use OpenGL for plots (instead of matplotlib)')
    parser.add_argument(
        '-f', '--fresh',
        dest="fresh_preferences",
        action="store_true",
        default=False,
        help='Start the application using new fresh user preferences')
    parser.add_argument(
        '--hdf5-file-locking',
        dest="hdf5_file_locking",
        action="store_true",
        default=False,
        help='Start the application with HDF5 file locking enabled (it is disabled by default)')
    return parser


def filesArgToUrls(filenames: Iterable[str]) -> Generator[object, None, None]:
    """Expand filenames and HDF5 data path in files input argument"""
    # Imports here so they are performed after setting HDF5_USE_FILE_LOCKING and logging level
    import silx.io
    from silx.io.utils import match
    from silx.io.url import DataUrl
    import silx.utils.files

    for filename in filenames:
        url = DataUrl(filename)

        for file_path in sorted(silx.utils.files.expand_filenames([url.file_path()])):
            if url.data_path() is not None and glob.has_magic(url.data_path()):
                try:
                    with silx.io.open(file_path) as f:
                        data_paths = list(match(f, url.data_path()))
                except BaseException as e:
                    _logger.error(
                        f"Error searching HDF5 path pattern '{url.data_path()}' in file '{file_path}': Ignored")
                    _logger.error(e.args[0])
                    _logger.debug("Backtrace", exc_info=True)
                    continue
            else:
                data_paths = [url.data_path()]

            for data_path in data_paths:
                yield DataUrl(
                    file_path=file_path,
                    data_path=data_path,
                    data_slice=url.data_slice(),
                    scheme=url.scheme(),
                )


def createWindow(parent, settings):
    from .Viewer import Viewer
    window = Viewer(parent=None, settings=settings)
    return window


def mainQt(options):
    """Part of the main depending on Qt"""
    if options.debug:
        logging.root.setLevel(logging.DEBUG)

    #
    # Import most of the things here to be sure to use the right logging level
    #

    # Use max opened files hard limit as soft limit
    try:
        import resource
    except ImportError:
        _logger.debug("No resource module available")
    else:
        if hasattr(resource, 'RLIMIT_NOFILE'):
            try:
                hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard_nofile, hard_nofile))
            except (ValueError, OSError):
                _logger.warning("Failed to retrieve and set the max opened files limit")
            else:
                _logger.debug("Set max opened files to %d", hard_nofile)

    # This needs to be done prior to load HDF5
    hdf5_file_locking = 'TRUE' if options.hdf5_file_locking else 'FALSE'
    _logger.info('Set HDF5_USE_FILE_LOCKING=%s', hdf5_file_locking)
    os.environ['HDF5_USE_FILE_LOCKING'] = hdf5_file_locking

    try:
        # it should be loaded before h5py
        import hdf5plugin  # noqa
    except ImportError:
        _logger.debug("Backtrace", exc_info=True)

    import h5py

    import silx
    from silx.gui import qt
    # Make sure matplotlib is configured
    # Needed for Debian 8: compatibility between Qt4/Qt5 and old matplotlib
    import silx.gui.utils.matplotlib  # noqa

    app = qt.QApplication([])
    qt.QLocale.setDefault(qt.QLocale.c())

    def sigintHandler(*args):
        """Handler for the SIGINT signal."""
        qt.QApplication.quit()

    signal.signal(signal.SIGINT, sigintHandler)
    sys.excepthook = qt.exceptionHandler

    timer = qt.QTimer()
    timer.start(500)
    # Application have to wake up Python interpreter, else SIGINT is not
    # catched
    timer.timeout.connect(lambda: None)

    settings = qt.QSettings(qt.QSettings.IniFormat,
                            qt.QSettings.UserScope,
                            "silx",
                            "silx-view",
                            None)
    if options.fresh_preferences:
        settings.clear()

    window = createWindow(parent=None, settings=settings)
    window.setAttribute(qt.Qt.WA_DeleteOnClose, True)

    if options.use_opengl_plot:
        # It have to be done after the settings (after the Viewer creation)
        silx.config.DEFAULT_PLOT_BACKEND = "opengl"


    for url in filesArgToUrls(options.files):
        # TODO: Would be nice to add a process widget and a cancel button
        try:
            window.appendFile(url.path())
        except IOError as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)

    window.show()
    result = app.exec()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    return result


def main(argv):
    """
    Main function to launch the viewer as an application

    :param argv: Command line arguments
    :returns: exit status
    """
    parser = createParser()
    options = parser.parse_args(argv[1:])
    mainQt(options)


if __name__ == '__main__':
    main(sys.argv)
