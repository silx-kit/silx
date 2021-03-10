# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest
import numpy

from silx.utils.testutils import ParametricTestCase

from silx.gui.plot._utils import ticklayout


class TestTickLayout(ParametricTestCase):
    """Test ticks layout algorithms"""

    def testTicks(self):
        """Test of :func:`ticks`"""
        tests = {  # (vmin, vmax): ref_ticks
            (1., 1.): (1.,),
            (0.5, 10.5): (2.0, 4.0, 6.0, 8.0, 10.0),
            (0.001, 0.005): (0.001, 0.002, 0.003, 0.004, 0.005)
            }

        for (vmin, vmax), ref_ticks in tests.items():
            with self.subTest(vmin=vmin, vmax=vmax):
                ticks, labels = ticklayout.ticks(vmin, vmax)
                self.assertTrue(numpy.allclose(ticks, ref_ticks))

    def testNiceNumbers(self):
        """Minimalistic tests of :func:`niceNumbers`"""
        tests = {  # (vmin, vmax): ref_ticks
            (0.5, 10.5): (0.0, 12.0, 2.0, 0),
            (10000., 10000.5): (10000.0, 10000.5, 0.1, 1),
            (0.001, 0.005): (0.001, 0.005, 0.001, 3)
            }

        for (vmin, vmax), ref_ticks in tests.items():
            with self.subTest(vmin=vmin, vmax=vmax):
                ticks = ticklayout.niceNumbers(vmin, vmax)
                self.assertEqual(ticks, ref_ticks)

    def testNiceNumbersLog(self):
        """Minimalistic tests of :func:`niceNumbersForLog10`"""
        tests = {  # (log10(min), log10(max): ref_ticks
            (0., 3.): (0, 3, 1, 0),
            (-3., 3): (-3, 3, 1, 0),
            (-32., 0.): (-36, 0, 6, 0)
        }

        for (vmin, vmax), ref_ticks in tests.items():
            with self.subTest(vmin=vmin, vmax=vmax):
                ticks = ticklayout.niceNumbersForLog10(vmin, vmax)
                self.assertEqual(ticks, ref_ticks)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestTickLayout))
    return testsuite


if __name__ == '__main__':
    unittest.main()
