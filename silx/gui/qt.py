# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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

- `PyQt5 <http://pyqt.sourceforge.net/Docs/PyQt5/>`_,
- `PyQt4 <http://pyqt.sourceforge.net/Docs/PyQt4/>`_ or
- `PySide <http://www.pyside.org>`_.

If a Qt binding is already loaded, it will use it, otherwise the different
Qt bindings are tried in this order: PyQt4, PySide, PyQt5.

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
provides the namespace of PyQt5 over PyQt4 and PySide.
"""

__authors__ = ["V.A. Sole - ESRF Data Analysis"]
__license__ = "MIT"
__date__ = "01/09/2016"


import logging
import sys


_logger = logging.getLogger(__name__)


BINDING = None
"""The name of the Qt binding in use: 'PyQt5', 'PyQt4' or 'PySide'."""

QtBinding = None
"""The Qt binding module in use: PyQt5, PyQt4 or PySide."""

HAS_SVG = False
"""True if Qt provides support for Scalable Vector Graphics (QtSVG)."""

# First check for an already loaded wrapper
if 'PySide' in sys.modules:
    BINDING = 'PySide'

elif 'PyQt5' in sys.modules:
    BINDING = 'PyQt5'

elif 'PyQt4' in sys.modules:
    BINDING = 'PyQt4'

else:  # Then try Qt bindings
    try:
        import PyQt4  # noqa
    except ImportError:
        try:
            import PySide  # noqa
        except ImportError:
            try:
                import PyQt5  # noqa
            except ImportError:
                raise ImportError(
                    'No Qt wrapper found. Install PyQt4, PyQt5 or PySide.')
            else:
                BINDING = 'PyQt5'
        else:
            BINDING = 'PySide'
    else:
        BINDING = 'PyQt4'


if BINDING == 'PyQt4':
    _logger.debug('Using PyQt4 bindings')

    if sys.version < "3.0.0":
        try:
            import sip

            sip.setapi("QString", 2)
            sip.setapi("QVariant", 2)
        except:
            _logger.warning("Cannot set sip API")

    import PyQt4 as QtBinding  # noqa

    from PyQt4.QtCore import *  # noqa
    from PyQt4.QtGui import *  # noqa

    try:
        from PyQt4.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PyQt4.QtOpenGL not available")

    try:
        from PyQt4.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PyQt4.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    from PyQt4.uic import loadUi  # noqa

    Signal = pyqtSignal

    Property = pyqtProperty

    Slot = pyqtSlot

elif BINDING == 'PySide':
    _logger.debug('Using PySide bindings')

    import PySide as QtBinding  # noqa

    from PySide.QtCore import *  # noqa
    from PySide.QtGui import *  # noqa

    try:
        from PySide.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PySide.QtOpenGL not available")

    try:
        from PySide.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PySide.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    pyqtSignal = Signal

    # Import loadUi wrapper for PySide
    from ._pyside_dynamic import loadUi  # noqa

elif BINDING == 'PyQt5':
    _logger.debug('Using PyQt5 bindings')

    import PyQt5 as QtBinding  # noqa

    from PyQt5.QtCore import *  # noqa
    from PyQt5.QtGui import *  # noqa
    from PyQt5.QtWidgets import *  # noqa
    from PyQt5.QtPrintSupport import *  # noqa

    try:
        from PyQt5.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PyQt5.QtOpenGL not available")

    try:
        from PyQt5.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PyQt5.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    from PyQt5.uic import loadUi  # noqa

    Signal = pyqtSignal

    Property = pyqtProperty

    Slot = pyqtSlot

else:
    raise ImportError('No Qt wrapper found. Install PyQt4, PyQt5 or PySide')


def supportedImageFormats():
    """Return a set of string of file format extensions supported by the
    Qt runtime."""
    if sys.version_info[0] < 3:
        convert = str
    else:
        convert = lambda data: str(data, 'ascii')
    formats = QImageReader.supportedImageFormats()
    return set([convert(data) for data in formats])
