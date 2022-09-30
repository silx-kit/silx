# /*##########################################################################
#
# Copyright (c) 2004-2021 European Synchrotron Radiation Facility
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
"""This module provides convenient functions related to Qt.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/11/2016"


from . import _qt


def getMouseEventPosition(event):
    """Qt5/Qt6 compatibility wrapper to access QMouseEvent position

    :param QMouseEvent event:
    :returns: (x, y) as a tuple of float
    """
    if _qt.BINDING in ("PyQt5", "PySide2"):
        return float(event.x()), float(event.y())
    # Qt6
    position = event.position()
    return position.x(), position.y()


def supportedImageFormats():
    """Return a set of string of file format extensions supported by the
    Qt runtime."""
    if _qt.BINDING == 'PySide2':
        def convert(data):
            return str(data.data(), 'ascii')
    else:
        convert = lambda data: str(data, 'ascii')
    formats = _qt.QImageReader.supportedImageFormats()
    return set([convert(data) for data in formats])


__globalThreadPoolInstance = None
"""Store the own silx global thread pool"""


def silxGlobalThreadPool():
    """"Manage an own QThreadPool to avoid issue on Qt5 Windows with the
    default Qt global thread pool.

    A thread pool is create in lazy loading. With a maximum of 4 threads.
    Else `qt.Thread.idealThreadCount()` is used.

    :rtype: qt.QThreadPool
    """
    global __globalThreadPoolInstance
    if __globalThreadPoolInstance is  None:
        tp = _qt.QThreadPool()
        # Setting maxThreadCount fixes a segfault with PyQt 5.9.1 on Windows
        maxThreadCount = min(4, tp.maxThreadCount())
        tp.setMaxThreadCount(maxThreadCount)
        __globalThreadPoolInstance = tp
    return __globalThreadPoolInstance
