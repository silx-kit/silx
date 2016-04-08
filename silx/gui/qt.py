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
"""Common wrapper over Python Qt bindings: PyQt5, PyQt4, PySide.

This module provides a flattened namespace over Qt bindings.

If a Qt bindings is already loaded, it will be used, otherwise the different
bindings are tried in this order: PyQt4, PySide, PyQt5.

The name of the loaded Qt bindings is stored in the BINDING variable.

For an alternative solution providing a structured namespace,
see `qtpy <https://pypi.python.org/pypi/QtPy/>`_ which
provides the namespace of PyQt5 over PyQt4 and PySide.
"""

__authors__ = ["V.A. Sole - ESRF Data Analysis"]
__license__ = "MIT"
__date__ = "16/02/2016"


import logging
import sys


_logger = logging.getLogger(__name__)


BINDING = None
"""The Python Qt binding that is used (One of: 'PySide', 'PyQt5', 'PyQt4')."""

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

    Signal = pyqtSignal

elif BINDING == 'PySide':
    _logger.debug('Using PySide bindings')

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

elif BINDING == 'PyQt5':
    _logger.debug('Using PyQt5 bindings')

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

    Signal = pyqtSignal

else:
    raise ImportError('No Qt wrapper found. Install PyQt4, PyQt5 or PySide')
