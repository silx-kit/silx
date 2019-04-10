# coding: utf-8
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
"""This module provides functions to access Qt C++ object state:

- :func:`isValid` to check whether a QObject C++ pointer is valid.
- :func:`createdByPython` to check if a QObject was created from Python.
- :func:`ownedByPython` to check if a QObject is currently owned by Python.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/10/2018"


from . import _qt as qt


if qt.BINDING in ('PyQt4', 'PyQt5'):
    if qt.BINDING == 'PyQt5':
        try:
            from PyQt5.sip import isdeleted as _isdeleted  # noqa
            from PyQt5.sip import ispycreated as createdByPython  # noqa
            from PyQt5.sip import ispyowned as ownedByPython  # noqa
        except ImportError:
            from sip import isdeleted as _isdeleted  # noqa
            from sip import ispycreated as createdByPython  # noqa
            from sip import ispyowned as ownedByPython  # noqa

    else:  # PyQt4
        from sip import isdeleted as _isdeleted  # noqa
        from sip import ispycreated as createdByPython  # noqa
        from sip import ispyowned as ownedByPython  # noqa

    def isValid(obj):
        """Returns True if underlying C++ object is valid.

        :param QObject obj:
        :rtype: bool
        """
        return not _isdeleted(obj)

elif qt.BINDING == 'PySide2':
    try:
        from PySide2.shiboken2 import isValid  # noqa
        from PySide2.shiboken2 import createdByPython  # noqa
        from PySide2.shiboken2 import ownedByPython  # noqa
    except ImportError:
        from shiboken2 import isValid  # noqa
        from shiboken2 import createdByPython  # noqa
        from shiboken2 import ownedByPython  # noqa

elif qt.BINDING == 'PySide':
    try:  # Available through PySide
        from PySide.shiboken import isValid  # noqa
        from PySide.shiboken import createdByPython  # noqa
        from PySide.shiboken import ownedByPython  # noqa
    except ImportError:  # Available through standalone shiboken package
        from Shiboken.shiboken import isValid  # noqa
        from Shiboken.shiboken import createdByPython  # noqa
        from Shiboken.shiboken import ownedByPython  # noqa

else:
    raise ImportError("Unsupported Qt binding %s" % qt.BINDING)

__all__ = ['isValid', 'createdByPython', 'ownedByPython']
