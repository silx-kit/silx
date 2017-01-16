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
sys.excepthook = qt.exceptionHandler
    
# synthetic data, stack of 100 images of size 200x300
mystack = numpy.fromfunction(
    lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
    (100, 200, 300)
)

# sv = StackView()
sv = StackViewMainWindow()
sv.setColormap("jet", autoscale=True)
sv.setStack(mystack)
sv.setLabels(["1st dim (0-99)", "2nd dim (0-199)",
              "3rd dim (0-299)"])
sv.show()

app.exec_()
sys.excepthook = sys.__excepthook__
