#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This example illustrates some usage possible with the baseline parameter
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "12/09/2019"


from silx.gui import qt
from silx.gui.plot import Plot1D
import numpy
import sys
import argparse


def stacked_histogran(plot, edges, histograms, colors, legend):
    # check that we have the same number of histogram, color and baseline
    current_baseline = numpy.zeros_like(edges)

    for histogram, color, layer_index in zip(histograms, colors, range(len(colors))):
        stacked_histo = histogram + current_baseline
        plot.addHistogram(histogram=stacked_histo,
                          edges=edges,
                          legend='_'.join((legend, str(layer_index))),
                          color=color,
                          baseline=current_baseline,
                          z=len(histograms)-layer_index,
                          fill=True)
        current_baseline = stacked_histo


def get_plot_std(backend):
    x = numpy.arange(0, 10, step=0.1)
    my_sin = numpy.sin(x)
    y = numpy.arange(-4, 6, step=0.1) + my_sin
    mean = numpy.arange(-5, 5, step=0.1) + my_sin
    baseline = numpy.arange(-6, 4, step=0.1) + my_sin
    edges = x[y >= 3.0]
    histo = mean[y >= 3.0] - 1.8

    plot = Plot1D(backend=backend)
    plot.addCurve(x=x, y=y, baseline=baseline, color='grey',
                  legend='std-curve', fill=True)
    plot.addCurve(x=x, y=mean, color='red', legend='mean')
    plot.addHistogram(histogram=histo, edges=edges, color='red',
                      legend='mean2', fill=True)
    return plot


def get_plot_stacked_histogram(backend):
    plot = Plot1D(backend=backend)
    # first histogram
    edges = numpy.arange(-6, 6, step=0.5)
    histo_1 = numpy.random.random(len(edges))
    histo_2 = numpy.random.random(len(edges))
    histo_3 = numpy.random.random(len(edges))
    histo_4 = numpy.random.random(len(edges))
    stacked_histogran(plot=plot,
                      edges=edges,
                      histograms=(histo_1, histo_2, histo_3, histo_4),
                      colors=('blue', 'green', 'red', 'yellow'),
                      legend='first_stacked_histo')

    # second histogram
    edges = numpy.arange(10, 25, step=1.0)
    histo_1 = -numpy.random.random(len(edges))
    histo_2 = -numpy.random.random(len(edges))
    stacked_histogran(plot=plot, histograms=(histo_1, histo_2),
                      edges=edges,
                      colors=('gray', 'black'),
                      legend='second_stacked_histo')

    # last histogram
    edges = [30, 40]
    histograms = [
        [0.2, 0.3],
        [0.0, 1.0],
        [0.1, 0.4],
        [0.2, 0.0],
        [0.6, 0.4],
    ]
    stacked_histogran(plot=plot,
                      histograms=histograms,
                      edges=edges,
                      colors=('blue', 'green', 'red', 'yellow', 'cyan'),
                      legend='third_stacked_histo')

    return plot


def get_plot_mean_baseline(backend):
    plot = Plot1D(backend=backend)
    x = numpy.arange(0, 10, step=0.1)
    y = numpy.sin(x)
    plot.addCurve(x=x, y=y, baseline=0, fill=True)
    plot.setYAxisLogarithmic(True)
    return plot


def get_plot_log(backend):
    plot = Plot1D(backend=backend)
    x = numpy.arange(0, 10, step=0.01)
    y = numpy.exp2(x)
    baseline = numpy.exp(x)
    plot.addCurve(x=x, y=y, baseline=baseline, fill=True)
    plot.setYAxisLogarithmic(True)
    return plot


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--backend',
        dest="backend",
        action="store",
        default=None,
        help='Set plot backend. Should be "matplotlib" (default) or "opengl"')

    options = parser.parse_args(argv[1:])
    assert options.backend in (None, 'matplotlib', 'opengl')
    qapp = qt.QApplication([])

    plot_std = get_plot_std(backend=options.backend)
    plot_std.show()

    plot_mean = get_plot_mean_baseline(backend=options.backend)
    plot_mean.show()

    plot_stacked_histo = get_plot_stacked_histogram(backend=options.backend)
    plot_stacked_histo.show()

    plot_log = get_plot_log(backend=options.backend)
    plot_log.show()

    qapp.exec_()


if __name__ == '__main__':
    main(sys.argv)
