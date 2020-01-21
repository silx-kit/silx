# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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

"""This module initializes matplotlib and sets-up the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2018"


from pkg_resources import parse_version
import matplotlib

from ... import qt


def _matplotlib_use(backend, force):
    """Wrapper of `matplotlib.use` to set-up backend.

     It adds extra initialization for PySide and PySide2 with matplotlib < 2.2.
    """
    # This is kept for compatibility with matplotlib < 2.2
    if parse_version(matplotlib.__version__) < parse_version('2.2'):
        if qt.BINDING == 'PySide':
            matplotlib.rcParams['backend.qt4'] = 'PySide'
        if qt.BINDING == 'PySide2':
            matplotlib.rcParams['backend.qt5'] = 'PySide2'

    matplotlib.use(backend, force=force)


if qt.BINDING in ('PyQt4', 'PySide'):
    _matplotlib_use('Qt4Agg', force=False)
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg  # noqa

elif qt.BINDING in ('PyQt5', 'PySide2'):
    _matplotlib_use('Qt5Agg', force=False)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa

else:
    raise ImportError("Unsupported Qt binding: %s" % qt.BINDING)
