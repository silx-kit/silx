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
"""This package provides a set of Qt widgets for plotting curves and images.

The plotting API is inherited from the `PyMca <http://pymca.sourceforge.net/>`_
plot API and is mostly compatible with it.

Those widgets supports interaction (e.g., zoom, pan, selections).

List of Qt widgets:

.. currentmodule:: silx.gui.plot

- :mod:`.PlotWidget`: A widget displaying a single plot.
- :mod:`.PlotWindow`: A :mod:`.PlotWidget` with a configurable set of tools.
- :class:`.Plot1D`: A widget with tools for curves.
- :class:`.Plot2D`: A widget with tools for images.

List of functions:

- :func:`.plot1D`: A function to plot curves from the (i)Python console.
- :func:`.plot2D`: A function to plot an image from the (i)Python console.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/02/2016"


from .PlotWidget import PlotWidget  # noqa
from .PlotWindow import PlotWindow, Plot1D, Plot2D, plot1D, plot2D  # noqa
from .PlotActions import PlotAction  # noqa

__all__ = ['PlotWidget', 'PlotWindow', 'Plot1D', 'Plot2D', 'plot1D', 'plot2D', 'PlotAction']
