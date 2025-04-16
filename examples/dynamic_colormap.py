# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
"""This script shows how to dyamically adjust the colormap to a small region 
around the cursor position. The DynamicColormapMode can be activated either by
the icon in the widget toolbar or by simply pressing the w-key.
The image has 4 regions of different gaussian distribution. When activated, the 
DynamicColormap mode will adjust the colormap to enhance the contrast in the 
region close to the cursor.

The pan and zoom modes (the two other interaction modes) can be activated either 
by their respective icon or by pressing the P- and Z-key respectively.
"""


import numpy
from silx.gui import qt
from silx.gui.plot import Plot2D


def main():
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    x = numpy.zeros((100, 100),dtype=numpy.float32)
    x[:50,:50] = numpy.random.randn(50,50)
    x[:50,50:] = 10 * numpy.random.randn(50,50)
    x[50:,:50] = 100 * numpy.random.randn(50,50)
    x[50:,50:] = 5 * numpy.random.randn(50,50)

    example = Plot2D()
    example.addImage(x)
    example.show()

    app.exec()


if __name__ == "__main__":
    main()
