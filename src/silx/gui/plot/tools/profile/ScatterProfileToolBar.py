# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module profile tools for scatter plots.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


from silx.utils import deprecation
from . import toolbar


class ScatterProfileToolBar(toolbar.ProfileToolBar):
    """QToolBar providing scatter plot profiling tools

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`~silx.gui.plot.PlotWidget` on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    def __init__(self, parent=None, plot=None, title=None):
        super(ScatterProfileToolBar, self).__init__(parent, plot)
        if title is not None:
            deprecation.deprecated_warning("Attribute",
                                          name="title",
                                          reason="removed",
                                          since_version="0.13.0",
                                          only_once=True,
                                          skip_backtrace_count=1)
        self.setScheme("scatter")
