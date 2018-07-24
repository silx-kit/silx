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
"""This script is a simple example of how to add your own statistic to a
:class:`~silx.gui.plot.statsWidget.StatsWidget` from customs
:class:`~silx.gui.plot.stats.Stats` and display it.

On this example we will:

   - show sum of values for each type
   - compute curve integrals (only for 'curve').
   - compute center of mass for all possible items

.. note:: for now the possible types manged by the Stats are ('curve', 'image',
          'scatter' and 'histogram')
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "24/07/2018"


from silx.gui import qt
from silx.gui.plot import Plot1D
from silx.gui.plot.stats.stats import StatBase
import numpy


class Integral(StatBase):
    """
    Simple calculation of the line integral
    """
    def __init__(self):
        StatBase.__init__(self, name='integral', compatibleKinds=('curve',))

    def calculate(self, context):
        xData, yData = context.data
        return numpy.trapz(x=xData, y=yData)


class COM(StatBase):
    """
    Compute data center of mass
    """
    def __init__(self):
        StatBase.__init__(self, name='COM', description="Center of mass")

    def calculate(self, context):
        if context.kind in ('curve', 'histogram'):
            xData, yData = context.data
            deno = numpy.sum(yData).astype(numpy.float32)
            if deno == 0.0:
                return 0.0
            else:
                return numpy.sum(xData * yData).astype(numpy.float32) / deno
        elif context.kind == 'scatter':
            xData, yData, values = context.data
            values = values.astype(numpy.float64)
            deno = numpy.sum(values)
            if deno == 0.0:
                return float('inf'), float('inf')
            else:
                comX = numpy.sum(xData * values) / deno
                comY = numpy.sum(yData * values) / deno
                return comX, comY


def main():
    app = qt.QApplication([])

    plot = Plot1D()

    x = numpy.arange(21)
    y = numpy.arange(21)
    plot.addCurve(x=x, y=y, legend='myCurve')
    plot.addCurve(x=x, y=(y + 5), legend='myCurve2')

    plot.setActiveCurve('myCurve')

    plot.addScatter(x=[0, 2, 5, 5, 12, 20],
                    y=[2, 3, 4, 20, 15, 6],
                    value=[5, 6, 7, 10, 90, 20],
                    legend='myScatter')

    stats = [
        ('sum', numpy.sum),
        Integral(),
        (COM(), '{0:.2f}'),
    ]

    plot.getStatsWidget().setStats(stats)
    plot.getStatsWidget().parent().setVisible(True)

    plot.show()
    app.exec_()


if __name__ == '__main__':
    main()
