# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""Miscellaneous helpers for Qt"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/03/2018"


import contextlib as _contextlib


@_contextlib.contextmanager
def blockSignals(*objs):
    """Context manager blocking signals of QObjects.

    It restores previous state when leaving.

    :param qt.QObject objs: QObjects for which to block signals
    """
    blocked = [(obj, obj.blockSignals(True)) for obj in objs]
    try:
        yield
    finally:
        for obj, previous in blocked:
            obj.blockSignals(previous)


class LockReentrant():
    """Context manager to lock a code block and check the state.
    """
    def __init__(self):
        self.__locked = False

    def __enter__(self):
        self.__locked = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__locked = False

    def locked(self):
        """Returns True if the code block is locked"""
        return self.__locked


def getQEventName(eventType):
    """
    Returns the name of a QEvent.

    :param Union[int,qt.QEvent] eventType: A QEvent or a QEvent type.
    :returns: str
    """
    from . import qtutils
    return qtutils.getQEventName(eventType)
