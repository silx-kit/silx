# /*##########################################################################
# Copyright (C) 2017-2021 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Tests of the median filter"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "02/05/2017"

from silx.gui import qt
from silx.math.medianfilter import medfilt2d as medfilt2d_silx
import numpy
import numpy.random
from timeit import Timer
from silx.gui.plot import Plot1D
import logging

try:
    import scipy
except:
    scipy = None
else:
    import scipy.ndimage

try:
    import PyMca5.PyMca as pymca
except:
    pymca = None
else:
    from PyMca5.PyMca.median import medfilt2d as medfilt2d_pymca

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BenchmarkMedianFilter(object):
    """Simple benchmark of the median fiter silx vs scipy"""

    NB_ITER = 3

    def __init__(self, imageWidth, kernels):
        self.img = numpy.random.rand(imageWidth, imageWidth)
        self.kernels = kernels

        self.run()

    def run(self):
        self.execTime = {}
        for kernel in self.kernels:
            self.execTime[kernel] = self.bench(kernel)

    def bench(self, width):
        def execSilx():
            medfilt2d_silx(self.img, width)

        def execScipy():
            scipy.ndimage.median_filter(input=self.img,
                                        size=width,
                                        mode='nearest')

        def execPymca():
            medfilt2d_pymca(self.img, width)

        execTime = {}

        t = Timer(execSilx)
        execTime["silx"] = t.timeit(BenchmarkMedianFilter.NB_ITER)
        logger.info(
            'exec time silx (kernel size = %s) is %s' % (width, execTime["silx"]))

        if scipy is not None:
            t = Timer(execScipy)
            execTime["scipy"] = t.timeit(BenchmarkMedianFilter.NB_ITER)
            logger.info(
                'exec time scipy (kernel size = %s) is %s' % (width, execTime["scipy"]))
        if pymca is not None:
            t = Timer(execPymca)
            execTime["pymca"] = t.timeit(BenchmarkMedianFilter.NB_ITER)
            logger.info(
                'exec time pymca (kernel size = %s) is %s' % (width, execTime["pymca"]))

        return execTime

    def getExecTimeFor(self, id):
        res = []
        for k in self.kernels:
            res.append(self.execTime[k][id])
        return res


app = qt.QApplication([])
kernels = [3, 5, 7, 11, 15]
benchmark = BenchmarkMedianFilter(imageWidth=1000, kernels=kernels)
plot = Plot1D()
plot.addCurve(x=kernels, y=benchmark.getExecTimeFor("silx"), legend='silx')
if scipy is not None:
    plot.addCurve(x=kernels, y=benchmark.getExecTimeFor("scipy"), legend='scipy')
if pymca is not None:
    plot.addCurve(x=kernels, y=benchmark.getExecTimeFor("pymca"), legend='pymca')
plot.show()
app.exec()
del app
