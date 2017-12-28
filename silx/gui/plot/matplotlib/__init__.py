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
"""This module inits matplotlib and setups the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "04/05/2017"


import sys
import logging


_logger = logging.getLogger(__name__)

if 'matplotlib' in sys.modules:
    _logger.warning(
        'matplotlib already loaded, setting its backend may not work')


from ... import qt

import matplotlib

if qt.BINDING == 'PySide':
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    matplotlib.rcParams['backend.qt4'] = 'PySide'
    import matplotlib.backends.backend_qt4agg as backend

elif qt.BINDING == 'PyQt4':
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    import matplotlib.backends.backend_qt4agg as backend

elif qt.BINDING == 'PySide2':
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    matplotlib.rcParams['backend.qt5'] = 'PySide2'
    import matplotlib.backends.backend_qt5agg as backend

elif qt.BINDING == 'PyQt5':
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    import matplotlib.backends.backend_qt5agg as backend

else:
    backend = None

if backend is not None:
    FigureCanvasQTAgg = backend.FigureCanvasQTAgg  # noqa
