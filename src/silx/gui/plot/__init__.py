# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
- :class:`.ScatterView`: A widget with tools for scatter plot.
- :class:`.ImageView`: A widget with tools for images and a side histogram.
- :class:`.StackView`: A widget with tools for a stack of images.

By default, those widget are using matplotlib_.
They can optionally use a faster OpenGL-based rendering (beta feature),
which is enabled by setting the ``backend`` argument to ``'gl'``
when creating the widgets (See :class:`.PlotWidget`).

.. note::

    This package depends on matplotlib_.
    The OpenGL backend further depends on
    `PyOpenGL <http://pyopengl.sourceforge.net/>`_ and OpenGL >= 2.1.

.. _matplotlib: http://matplotlib.org/
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/05/2017"


from .PlotWidget import PlotWidget  # noqa
from .PlotWindow import PlotWindow, Plot1D, Plot2D  # noqa
from .items.axis import TickMode
from .ImageView import ImageView  # noqa
from .StackView import StackView  # noqa
from .ScatterView import ScatterView  # noqa

__all__ = ['ImageView', 'PlotWidget', 'PlotWindow', 'Plot1D', 'Plot2D',
           'StackView', 'ScatterView', 'TickMode']
