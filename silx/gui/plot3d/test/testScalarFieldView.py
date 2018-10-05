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
# ###########################################################################*/
"""Test ScalarFieldView widget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import logging
import unittest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d.SFViewParamTree import TreeView


_logger = logging.getLogger(__name__)


class TestScalarFieldView(TestCaseQt, ParametricTestCase):
    """Tests of ScalarFieldView widget."""

    def setUp(self):
        super(TestScalarFieldView, self).setUp()
        self.widget = ScalarFieldView()
        self.widget.show()

        paramTreeWidget = TreeView()
        paramTreeWidget.setSfView(self.widget)

        dock = qt.QDockWidget()
        dock.setWidget(paramTreeWidget)
        self.widget.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)

        # Commented as it slows down the tests
        # self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.qapp.processEvents()
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget
        super(TestScalarFieldView, self).tearDown()

    @staticmethod
    def _buildData(size):
        """Make a 3D dataset"""
        coords = numpy.linspace(-10, 10, size)
        z = coords.reshape(-1, 1, 1)
        y = coords.reshape(1, -1, 1)
        x = coords.reshape(1, 1, -1)
        return numpy.sin(x * y * z) / (x * y * z)

    def testSimple(self):
        """Set the data and an isosurface"""
        data = self._buildData(size=32)

        self.widget.setData(data)
        self.widget.addIsosurface(0.5, (1., 0., 0., 0.5))
        self.widget.addIsosurface(0.7, qt.QColor('green'))
        self.qapp.processEvents()

    def testNotFinite(self):
        """Test with NaN and inf in data set"""

        # Some NaNs and inf
        data = self._buildData(size=32)
        data[8, :, :] = numpy.nan
        data[16, :, :] = numpy.inf
        data[24, :, :] = - numpy.inf

        self.widget.addIsosurface(0.5, 'red')
        self.widget.setData(data, copy=True)
        self.qapp.processEvents()
        self.widget.setData(None)

        # All NaNs or inf
        data = numpy.empty((4, 4, 4), dtype=numpy.float32)
        for value in (numpy.nan, numpy.inf):
            with self.subTest(value=str(value)):
                data[:] = value
                self.widget.setData(data, copy=True)
                self.qapp.processEvents()

    def testIsoSliderNormalization(self):
        """Test set TreeView with a different isoslider normalization"""
        data = self._buildData(size=32)

        self.widget.setData(data)
        self.widget.addIsosurface(0.5, (1., 0., 0., 0.5))
        self.widget.addIsosurface(0.7, qt.QColor('green'))
        self.qapp.processEvents()

        # Add a second TreeView
        paramTreeWidget = TreeView(self.widget)
        paramTreeWidget.setIsoLevelSliderNormalization('arcsinh')
        paramTreeWidget.setSfView(self.widget)

        dock = qt.QDockWidget()
        dock.setWidget(paramTreeWidget)
        self.widget.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestScalarFieldView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
