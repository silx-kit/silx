# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""Depracted module linking old PlotAction with the actions.xxx"""


__author__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "01/06/2017"

from silx.utils.deprecation import deprecated_warning

deprecated_warning(type_='module',
                   name=__file__,
                   reason='PlotActions refactoring',
                   replacement='plot.actions',
                   since_version='0.6')

from .actions import PlotAction

from .actions.io import CopyAction
from .actions.io import PrintAction
from .actions.io import SaveAction

from .actions.control import ColormapAction
from .actions.control import CrosshairAction
from .actions.control import CurveStyleAction
from .actions.control import GridAction
from .actions.control import KeepAspectRatioAction
from .actions.control import PanWithArrowKeysAction
from .actions.control import ResetZoomAction
from .actions.control import XAxisAutoScaleAction
from .actions.control import XAxisLogarithmicAction
from .actions.control import YAxisAutoScaleAction
from .actions.control import YAxisLogarithmicAction
from .actions.control import YAxisInvertedAction
from .actions.control import ZoomInAction
from .actions.control import ZoomOutAction

from .actions.medfilt import MedianFilter1DAction
from .actions.medfilt import MedianFilter2DAction
from .actions.medfilt import MedianFilterAction

from .actions.histogram import PixelIntensitiesHistoAction

from .actions.fit import FitAction
