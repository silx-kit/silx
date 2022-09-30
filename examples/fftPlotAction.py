#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
"""This script is a simple example of how to create a :class:`~silx.gui.plot.PlotWindow`
with a custom :class:`~silx.gui.plot.actions.PlotAction` added to the toolbar.

The action computes the FFT of all curves and plots their amplitude spectrum.
It also performs the reverse transform.

This example illustrates:
   - how to create a checkable action
   - how to store user info with a curve in a PlotWindow
   - how to modify the graph title and axes labels
   - how to add your own icon as a PNG file

See shiftPlotAction.py for a simpler example with more basic comments.

"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "27/06/2017"

import numpy
import os
import sys

from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.gui.plot.actions import PlotAction

# Custom icon
# make sure there is a "fft.png" file saved in the same folder as this script
scriptdir = os.path.dirname(os.path.realpath(__file__))
my_icon = os.path.join(scriptdir, "fft.png")


class FftAction(PlotAction):
    """QAction performing a Fourier transform on all curves when checked,
    and reverse transform when unchecked.

    :param plot: PlotWindow on which to operate
    :param parent: See documentation of :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        PlotAction.__init__(
                self,
                plot,
                icon=qt.QIcon(my_icon),
                text='FFT',
                tooltip='Perform Fast Fourier Transform on all curves',
                triggered=self.fftAllCurves,
                checkable=True,
                parent=parent)

    def _rememberGraphLabels(self):
        """Store labels and title as attributes"""
        self.original_title = self.plot.getGraphTitle()
        self.original_xlabel = self.plot.getXAxis().getLabel()
        self.original_ylabel = self.plot.getYAxis().getLabel()

    def fftAllCurves(self, checked=False):
        """Get all curves from our PlotWindow, compute the amplitude spectrum
        using a Fast Fourier Transform, replace all curves with their
        amplitude spectra.

        When un-checking the button, do the reverse transform.

        :param checked: Boolean parameter signaling whether the action
            has been checked or unchecked.
        """
        allCurves = self.plot.getAllCurves(withhidden=True)

        if checked:
            # remember original labels
            self._rememberGraphLabels()
            # change them
            self.plot.setGraphTitle("Amplitude spectrum")
            self.plot.getXAxis().setLabel("Frequency")
            self.plot.getYAxis().setLabel("Amplitude")
        else:
            # restore original labels
            self.plot.setGraphTitle(self.original_title)
            self.plot.getXAxis().setLabel(self.original_xlabel)
            self.plot.getYAxis().setLabel(self.original_ylabel)

        self.plot.clearCurves()

        for curve in allCurves:
            x = curve.getXData()
            y = curve.getYData()
            legend = curve.getName()
            info = curve.getInfo()
            if info is None:
                info = {}

            if checked:
                # FAST FOURIER TRANSFORM
                fft_y = numpy.fft.fft(y)
                # amplitude spectrum
                A = numpy.abs(fft_y)

                # sampling frequency (samples per X unit)
                Fs = len(x) / (max(x) - min(x))
                # frequency array (abscissa of new curve)
                F = [k * Fs / len(x) for k in range(len(A))]

                # we need to store  the complete transform (complex data) to be
                # able to perform the reverse transform.
                info["complex fft"] = fft_y
                info["original x"] = x

                # plot the amplitude spectrum
                self.plot.addCurve(F, A, legend="FFT of " + legend,
                                   info=info)

            else:
                # INVERSE FFT
                fft_y = info["complex fft"]
                # we keep only the real part because we know the imaginary
                # part is 0 (our original data was real numbers)
                y1 = numpy.real(numpy.fft.ifft(fft_y))

                # recover original info
                x1 = info["original x"]
                legend1 = legend[7:]    # remove "FFT of "

                # remove restored data from info dict
                for key in ["complex fft", "original x"]:
                    del info[key]

                # plot the original data
                self.plot.addCurve(x1, y1, legend=legend1,
                                   info=info)

        self.plot.resetZoom()


app = qt.QApplication([])

sys.excepthook = qt.exceptionHandler

plotwin = PlotWindow(control=True)
toolbar = qt.QToolBar("My toolbar")
plotwin.addToolBar(toolbar)

myaction = FftAction(plotwin)
toolbar.addAction(myaction)

# x range: 0 -- 10 (1000 points)
x = numpy.arange(1000) * 0.01

twopi = 2 * numpy.pi
# Sum of sine functions with frequencies 3, 20 and 42 Hz
y1 = numpy.sin(twopi * 3 * x) + 1.5 * numpy.sin(twopi * 20 * x) + 2 * numpy.sin(twopi * 42 * x)
# Cosine with frequency 7 Hz and phase pi / 3
y2 = numpy.cos(twopi * 7 * (x - numpy.pi / 3))
# 5 periods of square wave, amplitude 2
y3 = numpy.zeros_like(x)
for i in [0, 2, 4, 6, 8]:
    y3[i * len(x) // 10:(i + 1) * len(x) // 10] = 2

plotwin.addCurve(x, y1, legend="sin")
plotwin.addCurve(x, y2, legend="cos")
plotwin.addCurve(x, y3, legend="square wave")

plotwin.setGraphTitle("Original data")
plotwin.getYAxis().setLabel("amplitude")
plotwin.getXAxis().setLabel("time")

plotwin.show()
app.exec()
sys.excepthook = sys.__excepthook__
