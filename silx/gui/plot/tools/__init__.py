# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This package provides a set of widgets working with :class:`PlotWidget`.

It provides some QToolBar and QWidget:

- :class:`InteractiveModeToolBar`
- :class:`OutputToolBar`
- :class:`ImageToolBar`
- :class:`CurveToolBar`
- :class:`LimitsToolBar`
- :class:`PositionInfo`

It also provides a :mod:`~silx.gui.plot.tools.roi` module to handle
interactive region of interest on a :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/03/2018"


from .toolbars import InteractiveModeToolBar  # noqa
from .toolbars import OutputToolBar  # noqa
from .toolbars import ImageToolBar, CurveToolBar, ScatterToolBar  # noqa

from .LimitsToolBar import LimitsToolBar  # noqa
from .PositionInfo import PositionInfo  # noqa
