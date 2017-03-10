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
"""This script is a simple example to illustrate how to use the StackView
widget.
"""
import numpy
import sys
from silx.gui import qt
# from silx.gui.plot import StackView
from silx.gui.plot.StackView import StackViewMainWindow

app = qt.QApplication(sys.argv[1:])
    
x, y, z = numpy.meshgrid(numpy.linspace(-10, 10, 200),
                         numpy.linspace(-10, 5, 150),
                         numpy.linspace(-5, 10, 150))
mystack = numpy.asarray(numpy.sin(x * y * z) / (x * y * z),
                        dtype='float32')

# sv = StackView()
sv = StackViewMainWindow()
sv.setColormap("jet", autoscale=True)
sv.setStack(mystack)
sv.setLabels(["1st dim (0-99)", "2nd dim (0-199)",
              "3rd dim (0-299)"])
sv.show()

app.exec_()
