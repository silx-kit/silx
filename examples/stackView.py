#!/usr/bin/env python
# coding: utf-8
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
"""This script is a simple example to illustrate how to use the
:mod:`~silx.gui.plot.StackView` widget.
"""
import numpy
import sys
from silx.gui import qt
from silx.gui.plot.StackView import StackViewMainWindow

app = qt.QApplication(sys.argv[1:])
    
a, b, c = numpy.meshgrid(numpy.linspace(-10, 10, 200),
                         numpy.linspace(-10, 5, 150),
                         numpy.linspace(-5, 10, 120),
                         indexing="ij")
mystack = numpy.asarray(numpy.sin(a * b * c) / (a * b * c),
                        dtype='float32')

# linear calibrations (a, b), x -> a + bx
dim0_calib = (-10., 20. / 200.)
dim1_calib = (-10., 15. / 150.)
dim2_calib = (-5., 15. / 120.)

# sv = StackView()
sv = StackViewMainWindow()
sv.setColormap("jet", autoscale=True)
sv.setStack(mystack,
            calibrations=[dim0_calib, dim1_calib, dim2_calib])
sv.setLabels(["dim0: -10 to 10 (200 samples)",
              "dim1: -10 to 5 (150 samples)",
              "dim2: -5 to 10 (120 samples)"])
sv.show()

app.exec_()
