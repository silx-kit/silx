# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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
__date__ = "12/01/2022"


import importlib
import logging
import os
import sys
import traceback

from packaging.version import Version
from silx.utils import deprecation

_logger = logging.getLogger(__name__)


BINDING = None
"""The name of the Qt binding in use: PyQt5, PySide6, PyQt6."""

QtBinding = None  # noqa
"""The Qt binding module in use: PyQt5, PySide6, PyQt6."""

HAS_SVG = False
"""True if Qt provides support for Scalable Vector Graphics (QtSVG)."""

HAS_OPENGL = False
"""True if Qt provides support for OpenGL (QtOpenGL)."""


def _select_binding() -> str:
    """Select and load a Qt binding

    Qt binding is selected according to:
    - Already loaded binding
    - QT_API environment variable
    - Bindings order of priority

    :raises ImportError:
    :returns: Loaded binding
    """
    bindings = "PyQt5", "PySide6", "PyQt6"

    envvar = os.environ.get("QT_API", "").lower()

    # First check for an already loaded binding
    for binding in bindings:
        if f"{binding}.QtCore" in sys.modules:
            if envvar and envvar != binding.lower():
                _logger.warning(
                    f"Cannot satisfy QT_API={envvar} environment variable, {binding} is already loaded"
                )
            return binding

    # Check if QT_API can be satisfied
    if envvar:
        selection = [b for b in bindings if envvar == b.lower()]
        if not selection:
            _logger.warning(f"Environment variable QT_API={envvar} is not supported")
        else:
            binding = selection[0]
            try:
                importlib.import_module(f"{binding}.QtCore")
            except ImportError:
                _logger.warning(
                    f"Cannot import {binding} specified by QT_API environment variable"
                )
            else:
                return binding

    # Try to load binding
    for binding in bindings:
        try:
            importlib.import_module(f"{binding}.QtCore")
        except ImportError:
            if binding in sys.modules:
                del sys.modules[binding]
        else:
            return binding

    raise ImportError("No Qt wrapper found. Install PyQt5, PySide6, PyQt6.")


BINDING = _select_binding()


if BINDING == "PyQt5":
    _logger.debug("Using PyQt5 bindings")
    from PyQt5 import QtCore

    if sys.version_info >= (3, 10) and QtCore.PYQT_VERSION < 0x50E02:
        raise RuntimeError(
            "PyQt5 v%s is not supported, please upgrade it." % QtCore.PYQT_VERSION_STR
        )

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
    class _Foo(object):
        pass

    class QObject(QObject, _Foo):
        pass

elif BINDING == "PySide6":
    _logger.debug("Using PySide6 bindings")

    import PySide6 as QtBinding  # noqa

    if Version(QtBinding.__version__) < Version("6.4"):
        raise RuntimeError(
            f"PySide6 v{QtBinding.__version__} is not supported, please upgrade it."
        )

    from PySide6.QtCore import *  # noqa
    from PySide6.QtGui import *  # noqa
    from PySide6.QtWidgets import *  # noqa
    from PySide6.QtPrintSupport import *  # noqa

    try:
        from PySide6.QtOpenGL import *  # noqa
        from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa
    except ImportError:
        _logger.info("PySide6's QtOpenGL or QtOpenGLWidgets not available")
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


elif BINDING == "PyQt6":
    _logger.debug("Using PyQt6 bindings")

    # Monkey-patch module to expose enum values for compatibility
    # All Qt modules loaded here should be patched.
    from . import _pyqt6
    from PyQt6 import QtCore

    if QtCore.PYQT_VERSION < int("0x60300", 16):
        raise RuntimeError(
            "PyQt6 v%s is not supported, please upgrade it." % QtCore.PYQT_VERSION_STR
        )

    from PyQt6 import QtGui, QtWidgets, QtPrintSupport, QtOpenGL, QtSvg
    from PyQt6 import QtTest as _QtTest

    _pyqt6.patch_enums(
        QtCore, QtGui, QtWidgets, QtPrintSupport, QtOpenGL, QtSvg, _QtTest
    )

    import PyQt6 as QtBinding  # noqa

    from PyQt6.QtCore import *  # noqa
    from PyQt6.QtGui import *  # noqa
    from PyQt6.QtWidgets import *  # noqa
    from PyQt6.QtPrintSupport import *  # noqa

    try:
        from PyQt6.QtOpenGL import *  # noqa
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa
    except ImportError:
        _logger.info("PyQt6's QtOpenGL or QtOpenGLWidgets not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

    try:
        from PyQt6.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PyQt6.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    from PyQt6.uic import loadUi  # noqa

    Signal = pyqtSignal

    Property = pyqtProperty

    Slot = pyqtSlot

    # Disable PyQt6 cooperative multi-inheritance since other bindings do not provide it.
    # See https://www.riverbankcomputing.com/static/Docs/PyQt6/multiinheritance.html?highlight=inheritance
    class _Foo(object):
        pass

    class QObject(QObject, _Foo):
        pass

else:
    raise ImportError("No Qt wrapper found. Install PyQt5, PySide6 or PyQt6")


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
    _logger.error("%s %s %s", type_, value, "".join(traceback.format_tb(trace)))
    msg = QMessageBox()
    msg.setWindowTitle("Unhandled exception")
    msg.setIcon(QMessageBox.Critical)
    msg.setInformativeText("%s %s\nPlease report details" % (type_, value))
    msg.setDetailedText(("%s " % value) + "".join(traceback.format_tb(trace)))
    msg.raise_()
    msg.exec()
