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
"""Load Qt binding"""

__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "23/05/2018"


import logging
import sys
import traceback

from ...utils.deprecation import deprecated_warning


_logger = logging.getLogger(__name__)


BINDING = None
"""The name of the Qt binding in use: PyQt5, PyQt4 or PySide2."""

QtBinding = None  # noqa
"""The Qt binding module in use: PyQt5, PyQt4 or PySide2."""

HAS_SVG = False
"""True if Qt provides support for Scalable Vector Graphics (QtSVG)."""

HAS_OPENGL = False
"""True if Qt provides support for OpenGL (QtOpenGL)."""

# First check for an already loaded wrapper
if 'PySide2.QtCore' in sys.modules:
    BINDING = 'PySide2'

elif 'PySide.QtCore' in sys.modules:
    BINDING = 'PySide'

elif 'PyQt5.QtCore' in sys.modules:
    BINDING = 'PyQt5'

elif 'PyQt4.QtCore' in sys.modules:
    BINDING = 'PyQt4'

else:  # Then try Qt bindings
    try:
        import PyQt5  # noqa
    except ImportError:
        try:
            import PyQt4  # noqa
        except ImportError:
            try:
                import PySide2  # noqa
            except ImportError:
                try:
                    import PySide  # noqa
                except ImportError:
                    raise ImportError(
                        'No Qt wrapper found. Install PyQt5, PyQt4 or PySide2.')
                else:
                    BINDING = 'PySide'
            else:
                BINDING = 'PySide2'
        else:
            BINDING = 'PyQt4'
    else:
        BINDING = 'PyQt5'


if BINDING == 'PyQt4':
    _logger.debug('Using PyQt4 bindings')
    deprecated_warning("Qt Binding", "PyQt4",
                       replacement='PyQt5',
                       since_version='0.9.0')

    if sys.version_info < (3, ):
        try:
            import sip

            sip.setapi("QString", 2)
            sip.setapi("QVariant", 2)
            sip.setapi('QDate', 2)
            sip.setapi('QDateTime', 2)
            sip.setapi('QTextStream', 2)
            sip.setapi('QTime', 2)
            sip.setapi('QUrl', 2)
        except:
            _logger.warning("Cannot set sip API")

    import PyQt4 as QtBinding  # noqa

    from PyQt4.QtCore import *  # noqa
    from PyQt4.QtGui import *  # noqa

    try:
        from PyQt4.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PyQt4.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

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
    deprecated_warning("Qt Binding", "PySide",
                       replacement='PySide2',
                       since_version='0.9.0')

    import PySide as QtBinding  # noqa

    from PySide.QtCore import *  # noqa
    from PySide.QtGui import *  # noqa

    try:
        from PySide.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PySide.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

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

    # Import missing classes
    if not hasattr(locals(), "QIdentityProxyModel"):
        from ._pyside_missing import QIdentityProxyModel  # noqa

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
        _logger.info("PySide.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

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

elif BINDING == 'PySide2':
    _logger.debug('Using PySide2 bindings')
    _logger.warning(
        'Using PySide2 Qt binding: PySide2 support in silx is experimental!')

    import PySide2 as QtBinding  # noqa

    from PySide2.QtCore import *  # noqa
    from PySide2.QtGui import *  # noqa
    from PySide2.QtWidgets import *  # noqa
    from PySide2.QtPrintSupport import *  # noqa

    try:
        from PySide2.QtOpenGL import *  # noqa
    except ImportError:
        _logger.info("PySide2.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

    try:
        from PySide2.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PySide2.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    # Import loadUi wrapper for PySide2
    from ._pyside_dynamic import loadUi  # noqa

    pyqtSignal = Signal

else:
    raise ImportError('No Qt wrapper found. Install PyQt4, PyQt5, PySide2')


# provide a exception handler but not implement it by default
def exceptionHandler(type_, value, trace):
    """
    This exception handler prevents quitting to the command line when there is
    an unhandled exception while processing a Qt signal.

    The script/application willing to use it should implement code similar to:

    .. code-block:: python

        if __name__ == "__main__":
            sys.excepthook = qt.exceptionHandler

    """
    _logger.error("%s %s %s", type_, value, ''.join(traceback.format_tb(trace)))
    msg = QMessageBox()
    msg.setWindowTitle("Unhandled exception")
    msg.setIcon(QMessageBox.Critical)
    msg.setInformativeText("%s %s\nPlease report details" % (type_, value))
    msg.setDetailedText(("%s " % value) + ''.join(traceback.format_tb(trace)))
    msg.raise_()
    msg.exec_()
