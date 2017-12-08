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
"""Basic tests for ColormapDialog"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/12/2016"


import doctest
import unittest

from silx.gui.test.utils import qWaitForWindowExposedAndActivate
from silx.gui import qt
from silx.gui.plot import ColormapDialog
from silx.gui.test.utils import TestCaseQt
from silx.gui.plot.Colormap import Colormap
from silx.test.utils import ParametricTestCase


# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


def _tearDownQt(docTest):
    """Tear down to use for test from docstring.

    Checks that dialog widget is displayed
    """
    dialogWidget = docTest.globs['dialog']
    qWaitForWindowExposedAndActivate(dialogWidget)
    dialogWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
    dialogWidget.close()
    del dialogWidget
    _qapp.processEvents()


cmapDocTestSuite = doctest.DocTestSuite(ColormapDialog, tearDown=_tearDownQt)
"""Test suite of tests from the module's docstrings."""


class TestColormapDialog(TestCaseQt, ParametricTestCase):
    def setUp(self):
        self.colormap = Colormap(name='gray', vmin=0.0, vmax=1.0,
                                 normalization='linear')

        self.colormapDiag = ColormapDialog.ColormapDialog()
        self.colormapDiag.setAttribute(qt.Qt.WA_DeleteOnClose)

    def tearDown(self):
        self.colormapDiag.close()
        del self.colormapDiag

    def testGUIEdition(self):
        """Make sure the colormap is correctly edited"""
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag._rangeAutoscaleButton.setChecked(False)
        self.colormapDiag._comboBoxColormap.setCurrentIndex(3)
        self.colormapDiag._normButtonLog.setChecked(True)
        self.colormapDiag.setDataRange(10.0, 20.0)
        self.assertTrue(self.colormap.getName() == 'red')
        self.assertTrue(self.colormapDiag.getColormap().getName() == 'red')
        self.assertTrue(self.colormap.getNormalization() == 'log')
        self.assertTrue(self.colormap.getVMin() == 10)
        self.assertTrue(self.colormap.getVMax() == 20)

    def testGUIAccept(self):
        """Make sure the colormap is modify if go through accept"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.show()
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag._rangeAutoscaleButton.setChecked(True)
        self.colormapDiag.accept()
        self.assertTrue(self.colormap.isAutoscale() is True)

    def testGUIReject(self):
        """Make sure the colormap is modify if go through reject"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.show()
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag._rangeAutoscaleButton.setChecked(True)
        self.colormapDiag.reject()
        self.assertTrue(self.colormap.isAutoscale() is False)

    def testGUIClose(self):
        """Make sure the colormap is modify if go through reject"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.show()
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag._rangeAutoscaleButton.setChecked(True)
        self.colormapDiag.close()
        self.assertTrue(self.colormap.isAutoscale() is False)

    def testSetColormapIsCorrect(self):
        """Make sure the interface fir the colormap when set a new colormap"""
        self.colormap.setName('red')
        for norm in (Colormap.NORMALIZATIONS):
            for autoscale in (True, False):
                if autoscale is True:
                    self.colormap.setVRange(None, None)
                else:
                    self.colormap.setVRange(11, 101)
                self.colormap.setNormalization(norm)
                with self.subTest(colormap=self.colormap):
                    self.colormapDiag.setColormap(self.colormap)
                    self.assertTrue(
                        self.colormapDiag._normButtonLinear.isChecked() == (norm is Colormap.LINEAR))
                    self.assertTrue(
                        self.colormapDiag._comboBoxColormap.currentText() == 'Red')
                    self.assertTrue(
                        self.colormapDiag._rangeAutoscaleButton.isChecked() == autoscale)
                    if autoscale is False:
                        self.assertTrue(self.colormapDiag._minValue.value() == 11)
                        self.assertTrue(self.colormapDiag._maxValue.value() == 101)
                        self.assertFalse(self.colormapDiag._minValue.isEnabled())
                        self.assertFalse(self.colormapDiag._maxValue.isEnabled())
                    else:
                        self.assertTrue(self.colormapDiag._minValue.isEnabled())
                        self.assertTrue(self.colormapDiag._maxValue.isEnabled())

    def testColormapDel(self):
        """Check behavior if the colormap has been deleted outside"""
        pass

    def testColormapEditedOutside(self):
        """Make sure the GUI is still up to date if the colormap is modified
        outside"""
        pass


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(cmapDocTestSuite)
    for testClass in (TestColormapDialog, ):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
            testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
