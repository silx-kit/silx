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
This module contains utilitaries used by other dialog modules.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/10/2017"

import os
import sys
import types
from silx.gui import qt
from silx.third_party import six


def samefile(path1, path2):
    """Portable :func:`os.path.samepath` function.

    :param str path1: A path to a file
    :param str path2: Another path to a file
    :rtype: bool
    """
    if six.PY2 and sys.platform == "win32":
        path1 = os.path.normcase(path1)
        path2 = os.path.normcase(path2)
        return path1 == path2
    if path1 == path2:
        return True
    if path1 == "":
        return False
    if path2 == "":
        return False
    return os.path.samefile(path1, path2)


def findClosestSubPath(hdf5Object, path):
    """Find the closest existing path from the hdf5Object using a subset of the
    provided path.

    Returns None if no path found. It is possible if the path is a relative
    path.

    :param h5py.Node hdf5Object: An HDF5 node
    :param str path: A path
    :rtype: str
    """
    if path in ["", "/"]:
        return "/"
    names = path.split("/")
    if path[0] == "/":
        names.pop(0)
    for i in range(len(names)):
        n = len(names) - i
        path2 = "/".join(names[0:n])
        if path2 == "":
            return ""
        if path2 in hdf5Object:
            return path2

    if path[0] == "/":
        return "/"
    return None


def patchToConsumeReturnKey(widget):
    """
    Monkey-patch a widget to consume the return key instead of propagating it
    to the dialog.
    """
    assert(not hasattr(widget, "_oldKeyPressEvent"))

    def keyPressEvent(self, event):
        k = event.key()
        result = self._oldKeyPressEvent(event)
        if k in [qt.Qt.Key_Return, qt.Qt.Key_Enter]:
            event.accept()
        return result

    widget._oldKeyPressEvent = widget.keyPressEvent
    widget.keyPressEvent = types.MethodType(keyPressEvent, widget)
