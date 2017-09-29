# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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

__authors__ = ["N. Vigano", "H. Payno", "P. Paleo"]
__license__ = "MIT"
__date__ = "19/09/2017"

import numpy


class PhantomGenerator(object):
    """
    Class for generating Phantoms
    """

    class _Ellipsoid:
        def __init__(self, a, b, c, x0, y0, z0, alpha, mu):
            self.a = a
            self.b = b
            self.c = c
            self.x0 = x0
            self.y0 = y0
            self.z0 = z0
            self.alpha = alpha * numpy.pi / 180.0
            self.mu = mu
            self.cosAlpha = numpy.cos(self.alpha)
            self.sinAlpha = numpy.sin(self.alpha)

    SHEPP_LOGAN = [
        #         a       b      c       x0     y0      z0     alpha  mu
        _Ellipsoid(0.69, 0.92, 0.90, 0.0, 0.0, 0.0, 0.0, 0.10),
        _Ellipsoid(0.6624, 0.874, 0.88, 0.0, -0.02, 0.0, 0.0, -0.08),
        _Ellipsoid(0.11, 0.31, 0.21, 0.22, -0.0, 0.0, -18.0, -0.02),
        _Ellipsoid(0.16, 0.41, 0.22, -0.22, 0.0, -0.25, 18.0, -0.02),
        _Ellipsoid(0.21, 0.25, 0.35, 0.0, 0.35, -0.25, 0.0, 0.03),
        _Ellipsoid(0.046, 0.046, 0.046, 0.0, 0.10, -0.25, 0.0, 0.01),
        _Ellipsoid(0.046, 0.046, 0.02, 0.0, -0.10, -0.25, 0.0, 0.01),
        _Ellipsoid(0.046, 0.023, 0.02, -0.08, -0.605, -0.25, 0.0, 0.01),
        _Ellipsoid(0.023, 0.023, 0.10, 0.0, -0.605, -0.25, 0.0, 0.01),
        _Ellipsoid(0.023, 0.046, 0.10, 0.06, -0.605, -0.25, 0.0, 0.01)
    ]

    @staticmethod
    def get2DPhantomSheppLogan(n, ellipsoidID=None):
        """
        generate a classical 2D shepp logan phantom.

        :param n: The width (and height) of the phantom to generate
        :param ellipsoidID: The Id of the ellipsoid to pick. If None will
                            produce every ellipsoid
        :return numpy.ndarray: shepp logan phantom
        """
        assert(ellipsoidID is None or (ellipsoidID >= 0 and ellipsoidID < len(PhantomGenerator.SHEPP_LOGAN)))
        if ellipsoidID is None:
            area = PhantomGenerator._get2DPhantom(n,
                                                  PhantomGenerator.SHEPP_LOGAN)
        else:
            area = PhantomGenerator._get2DPhantom(n,
                                                  [PhantomGenerator.SHEPP_LOGAN[ellipsoidID]])

        indices = numpy.abs(area) > 0
        area[indices] = numpy.multiply(area[indices] + 0.1, 5)
        return area / 100.0

    @staticmethod
    def _get2DPhantom(n, phantomSpec):
        area = numpy.ndarray(shape=(n, n))
        area.fill(0.)

        count = 0
        for ell in phantomSpec:
            count = count+1
            for x in range(n):
                sumSquareXandY = PhantomGenerator._getSquareXandYsum(n, x, ell)
                indices = sumSquareXandY <= 1
                area[indices, x] = ell.mu
        return area

    @staticmethod
    def _getSquareXandYsum(n, x, ell):
        supportX1 = numpy.ndarray(shape=(n, ))
        supportX2 = numpy.ndarray(shape=(n, ))
        support_consts = numpy.ndarray(shape=(n, ))

        xScaled = float(2*x-n)/float(n)
        xCos = xScaled * ell.cosAlpha
        xSin = -xScaled * ell.sinAlpha
        supportX1.fill(xCos)
        supportX2.fill(xSin)

        supportY1 = numpy.arange(n)
        support_consts.fill(2.)
        supportY1 = numpy.multiply(support_consts, supportY1)
        support_consts.fill(n)
        supportY1 = numpy.subtract(supportY1, support_consts)
        support_consts.fill(n)
        supportY1 = numpy.divide(supportY1, support_consts)
        supportY2 = numpy.array(supportY1)

        support_consts.fill(ell.sinAlpha)
        supportY1 = numpy.add(supportX1,
                              numpy.multiply(supportY1, support_consts))
        support_consts.fill(ell.cosAlpha)
        supportY2 = numpy.add(supportX2,
                              numpy.multiply(supportY2, support_consts))

        support_consts.fill(ell.x0)
        supportY1 = numpy.subtract(supportY1, support_consts)
        support_consts.fill(ell.y0)
        supportY2 = numpy.subtract(supportY2, support_consts)

        support_consts.fill(ell.a)
        supportY1 = numpy.power((numpy.divide(supportY1, support_consts)),
                                2)
        support_consts.fill(ell.b)
        supportY2 = numpy.power(numpy.divide(supportY2, support_consts),
                                2)

        return numpy.add(supportY1, supportY2)

    @staticmethod
    def _getSquareZ(n, ell):
        supportZ1 = numpy.arange(n)
        support_consts = numpy.ndarray(shape=(n, ))
        support_consts.fill(2.)
        supportZ1 = numpy.multiply(support_consts, supportZ1)
        support_consts.fill(n)
        supportZ1 = numpy.subtract(supportZ1, support_consts)
        support_consts.fill(n)
        supportZ1 = numpy.divide(supportZ1, support_consts)

        support_consts.fill(ell.z0)
        supportZ1 = numpy.subtract(supportZ1, ell.z0)

        support_consts.fill(ell.c)
        return numpy.power(numpy.divide(supportZ1, support_consts),
                           2)

