# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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

from __future__ import absolute_import

"""This module inits matplotlib and setups the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2018"


import sys
import logging


_logger = logging.getLogger(__name__)

_matplotlib_already_loaded = 'matplotlib' in sys.modules
"""If true, matplotlib was already loaded"""

import matplotlib
from ... import qt


def _configure(backend, backend_qt4=None, backend_qt5=None, check=False):
    """Configure matplotlib using a specific backend.

    It initialize `matplotlib.rcParams` using the requested backend, or check
    if it is already configured as requested.

    :param bool check: If true, the function only check that matplotlib
        is already initialized as request. If not a warning is emitted.
        If `check` is false, matplotlib is initialized.
    """
    if check:
        valid = matplotlib.rcParams['backend'] == backend
        if backend_qt4 is not None:
            valid = valid and matplotlib.rcParams['backend.qt4'] == backend_qt4
        if backend_qt5 is not None:
            valid = valid and matplotlib.rcParams['backend.qt5'] == backend_qt5

        if not valid:
            _logger.warning('matplotlib already loaded, setting its backend may not work')
    else:
        matplotlib.rcParams['backend'] = backend
        if backend_qt4 is not None:
            matplotlib.rcParams['backend.qt4'] = backend_qt4
        if backend_qt5 is not None:
            matplotlib.rcParams['backend.qt5'] = backend_qt5


if qt.BINDING == 'PySide':
    _configure('Qt4Agg', backend_qt4='PySide', check=_matplotlib_already_loaded)
    import matplotlib.backends.backend_qt4agg as backend

elif qt.BINDING == 'PyQt4':
    _configure('Qt4Agg', check=_matplotlib_already_loaded)
    import matplotlib.backends.backend_qt4agg as backend

elif qt.BINDING == 'PySide2':
    _configure('Qt5Agg', backend_qt5="PySide2", check=_matplotlib_already_loaded)
    import matplotlib.backends.backend_qt5agg as backend

elif qt.BINDING == 'PyQt5':
    _configure('Qt5Agg', check=_matplotlib_already_loaded)
    import matplotlib.backends.backend_qt5agg as backend

else:
    backend = None

if backend is not None:
    FigureCanvasQTAgg = backend.FigureCanvasQTAgg  # noqa
