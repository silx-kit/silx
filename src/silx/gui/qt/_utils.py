# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
"""This module provides convenient functions related to Qt."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/11/2016"


import logging
import traceback

from . import _qt

_logger = logging.getLogger(__name__)


def getMouseEventPosition(event):
    """Qt5/Qt6 compatibility wrapper to access QMouseEvent position

    :param QMouseEvent event:
    :returns: (x, y) as a tuple of float
    """
    if _qt.BINDING == "PyQt5":
        return float(event.x()), float(event.y())
    # Qt6
    position = event.position()
    return position.x(), position.y()


def supportedImageFormats():
    """Return a set of string of file format extensions supported by the
    Qt runtime."""
    formats = _qt.QImageReader.supportedImageFormats()
    return {str(data, "ascii") for data in formats}


__globalThreadPoolInstance = None
"""Store the own silx global thread pool"""


def silxGlobalThreadPool():
    """Manage an own QThreadPool to avoid issue on Qt5 Windows with the
    default Qt global thread pool.

    A thread pool is create in lazy loading. With a maximum of 4 threads.
    Else `qt.Thread.idealThreadCount()` is used.

    :rtype: qt.QThreadPool
    """
    global __globalThreadPoolInstance
    if __globalThreadPoolInstance is None:
        tp = _qt.QThreadPool()
        # Setting maxThreadCount fixes a segfault with PyQt 5.9.1 on Windows
        maxThreadCount = min(4, tp.maxThreadCount())
        tp.setMaxThreadCount(maxThreadCount)
        __globalThreadPoolInstance = tp
    return __globalThreadPoolInstance


def exceptionHandler(type_, value, trace):
    """
    This exception handler prevents quitting to the command line when there is
    an unhandled exception while processing a Qt signal.

    The script/application willing to use it should implement code similar to:

    .. code-block:: python

        if __name__ == "__main__":
            sys.excepthook = qt.exceptionHandler

    """
    _logger.error("%s %s %s", type_, value, "".join(traceback.format_tb(trace)))
    msg = _qt.QMessageBox()
    msg.setWindowTitle("Unhandled exception")
    msg.setIcon(_qt.QMessageBox.Critical)
    msg.setInformativeText(f"{type_} {value}\nPlease report details")
    msg.setDetailedText(("%s " % value) + "".join(traceback.format_tb(trace)))
    msg.raise_()
    msg.exec()
