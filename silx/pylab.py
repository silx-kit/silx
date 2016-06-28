# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Convenient module to use main features of silx from the console.

Usage from python or ipython console and ipython/jupyter notebook:

>>> from silx.pylab import *
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/06/2016"


import sys as _sys


# Probe module loaded from console
_IS_CONSOLE = hasattr(_sys, 'ps1')

# Probe ipython
try:
    __IPYTHON__
except NameError:
    __IPYTHON__ = False

# Probe ipython/jupyter notebook
if __IPYTHON__:
    from IPython import get_ipython

    # Notebook detection probably fragile
    _IS_NOTEBOOK = ('parent_appname' in get_ipython().config['IPKernelApp'] or
                    hasattr(get_ipython(), 'kernel'))
else:
    _IS_NOTEBOOK = False


if not _IS_NOTEBOOK:
    # Load Qt and widgets only if running from console
    from silx.gui import qt

    if _IS_CONSOLE:
        _qapp = qt.QApplication.instance() or qt.QApplication([])

    # Makes sure we set-up matplotlib first
    import silx.gui.plot.BackendMatplotlib as _BackendMatplotlib  # noqa

    from silx.gui.plot import *  # noqa

if __IPYTHON__:
    # %pylab
    if _IS_NOTEBOOK:
        get_ipython().enable_pylab(gui='inline', import_all=False)
    else:
        get_ipython().enable_pylab(gui='qt', import_all=False)

else:  # pylab equivalent
    import numpy
    import matplotlib  # noqa
    from matplotlib import pylab, mlab, pyplot  # noqa
    np = numpy
    plt = pyplot

    # import_all=True equivalent
    # from pylab import *
    # from numpy import *


# Load modules
from silx.image import *  # noqa
from silx.io import *  # noqa
from silx.math import *  # noqa
