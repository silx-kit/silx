# coding: utf-8
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
"""Load Qt binding"""

__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "23/05/2018"


import logging
import sys
import traceback


_logger = logging.getLogger(__name__)


BINDING = None
"""The name of the Qt binding in use: PyQt5, PySide2, PySide6."""

QtBinding = None  # noqa
"""The Qt binding module in use: PyQt5, PySide2, PySide6."""

HAS_SVG = False
"""True if Qt provides support for Scalable Vector Graphics (QtSVG)."""

HAS_OPENGL = False
"""True if Qt provides support for OpenGL (QtOpenGL)."""

# First check for an already loaded wrapper
for _binding in ('PySide2', 'PyQt5', 'PySide6'):
    if _binding + '.QtCore' in sys.modules:
        BINDING = _binding
        break
else:  # Then try Qt bindings
    try:
        import PyQt5.QtCore  # noqa
    except ImportError:
        if 'PyQt5' in sys.modules:
            del sys.modules["PyQt5"]
        try:
            import PySide2.QtCore  # noqa
        except ImportError:
            if 'PySide2' in sys.modules:
                del sys.modules["PySide2"]
            try:
                import PySide6.QtCore  # noqa
            except ImportError:
                if 'PySide6' in sys.modules:
                    del sys.modules["PySide6"]
                raise ImportError(
                    'No Qt wrapper found. Install PyQt5, PySide2, PySide6.')
            else:
                BINDING = 'PySide6'
        else:
            BINDING = 'PySide2'
    else:
        BINDING = 'PyQt5'


if BINDING == 'PyQt5':
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

    # Disable PyQt5's cooperative multi-inheritance since other bindings do not provide it.
    # See https://www.riverbankcomputing.com/static/Docs/PyQt5/multiinheritance.html?highlight=inheritance
    class _Foo(object): pass
    class QObject(QObject, _Foo): pass


elif BINDING == 'PySide2':
    _logger.debug('Using PySide2 bindings')

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

    pyqtSignal = Signal

    # Qt6 compatibility:
    # with PySide2 `exec` method has a special behavior
    class _ExecMixIn:
        """Mix-in class providind `exec` compatibility"""
        def exec(self, *args, **kwargs):
            return super().exec_(*args, **kwargs)

    # QtWidgets
    class QApplication(_ExecMixIn, QApplication): pass
    class QColorDialog(_ExecMixIn, QColorDialog): pass
    class QDialog(_ExecMixIn, QDialog): pass
    class QErrorMessage(_ExecMixIn, QErrorMessage): pass
    class QFileDialog(_ExecMixIn, QFileDialog): pass
    class QFontDialog(_ExecMixIn, QFontDialog): pass
    class QInputDialog(_ExecMixIn, QInputDialog): pass
    class QMenu(_ExecMixIn, QMenu): pass
    class QMessageBox(_ExecMixIn, QMessageBox): pass
    class QProgressDialog(_ExecMixIn, QProgressDialog): pass
    #QtCore
    class QCoreApplication(_ExecMixIn, QCoreApplication): pass
    class QEventLoop(_ExecMixIn, QEventLoop): pass
    if hasattr(QTextStreamManipulator, "exec_"):
        # exec_ only wrapped in PySide2 and NOT in PyQt5
        class QTextStreamManipulator(_ExecMixIn, QTextStreamManipulator): pass
    class QThread(_ExecMixIn, QThread): pass


elif BINDING == 'PySide6':
    _logger.debug('Using PySide6 bindings')

    import PySide6 as QtBinding  # noqa

    from PySide6.QtCore import *  # noqa
    from PySide6.QtGui import *  # noqa
    from PySide6.QtWidgets import *  # noqa
    from PySide6.QtPrintSupport import *  # noqa

    try:
        from PySide6.QtOpenGL import *  # noqa
        from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa
    except ImportError:
        _logger.info("PySide6.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

    try:
        from PySide6.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PySide6.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    pyqtSignal = Signal

else:
    raise ImportError('No Qt wrapper found. Install PyQt5, PySide2 or PySide6')


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
    msg.exec()
