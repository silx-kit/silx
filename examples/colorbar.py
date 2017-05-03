#!/usr/bin/env python
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
"""
Example to show the use of `ColorBarWidget` widget.
It can be associated to a plot.

In this exqmple the `ColorBarWidget` widget will display the colormap of the
active image.

To change the active image slick on the image you want to set active.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "03/05/2017"


from silx.gui import qt
import numpy
from silx.gui.plot import Plot2D
from silx.gui.plot.ColorBar import ColorBarWidget

image = numpy.exp(numpy.random.rand(100, 100) * 10)

app = qt.QApplication([])

plot = Plot2D()
colorbar = ColorBarWidget(parent=None, plot=plot)
colorbar.setLegend('my colormap')
colorbar.show()
plot.show()

clm = plot.getDefaultColormap()
clm['normalization'] = 'log'
clm['name'] = 'viridis'
plot.addImage(data=image, colormap=clm, legend='image')
plot.setActiveImage('image')

app.exec_()
