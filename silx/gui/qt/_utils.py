# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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

import sys
from . import _qt as qt


def supportedImageFormats():
    """Return a set of string of file format extensions supported by the
    Qt runtime."""
    if sys.version_info[0] < 3 or qt.BINDING == 'PySide':
        convert = str
    elif qt.BINDING == 'PySide2':
        def convert(data):
            return str(data.data(), 'ascii')
    else:
        convert = lambda data: str(data, 'ascii')
    formats = qt.QImageReader.supportedImageFormats()
    return set([convert(data) for data in formats])


__globalThreadPoolInstance = None
"""Store the own silx global thread pool"""


def silxGlobalThreadPool():
    """"Manage an own QThreadPool to avoid issue on Qt5 Windows with the
    default Qt global thread pool.

    :rtype: qt.QThreadPool
    """
    global __globalThreadPoolInstance
    if __globalThreadPoolInstance is  None:
        tp = qt.QThreadPool()
        # This pointless command fixes a segfault with PyQt 5.9.1 on Windows
        tp.setMaxThreadCount(tp.maxThreadCount())
        __globalThreadPoolInstance = tp
    return __globalThreadPoolInstance
