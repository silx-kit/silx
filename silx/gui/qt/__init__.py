# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""Common wrapper over Python Qt bindings:

- `PyQt5 <http://pyqt.sourceforge.net/Docs/PyQt5/>`_
- `PySide2 <https://wiki.qt.io/Qt_for_Python>`_
- `PyQt4 <http://pyqt.sourceforge.net/Docs/PyQt4/>`_

If a Qt binding is already loaded, it will use it, otherwise the different
Qt bindings are tried in this order: PyQt5, PyQt4, PySide2.

The name of the loaded Qt binding is stored in the BINDING variable.

This module provides a flat namespace over Qt bindings by importing
all symbols from **QtCore** and **QtGui** packages and if available
from **QtOpenGL** and **QtSvg** packages.
For **PyQt5**, it also imports all symbols from **QtWidgets** and
**QtPrintSupport** packages.

Example of using :mod:`silx.gui.qt` module:

>>> from silx.gui import qt
>>> app = qt.QApplication([])
>>> widget = qt.QWidget()

For an alternative solution providing a structured namespace,
see `qtpy <https://pypi.python.org/pypi/QtPy/>`_ which
provides the namespace of PyQt5 over PyQt4, PySide and PySide2.
"""

from ._qt import *  # noqa
from ._utils import *  # noqa


if sys.platform == "darwin":
    if BINDING in ["PySide", "PyQt4"]:
        from . import _macosx
        _macosx.patch_QUrl_toLocalFile()
