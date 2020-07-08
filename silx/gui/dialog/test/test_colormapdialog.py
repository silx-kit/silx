# coding: utf-8
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
"""Basic tests for ColormapDialog"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/11/2018"


import unittest

from silx.gui import qt
from silx.gui.dialog import ColormapDialog
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.colors import Colormap, preferredColormaps
from silx.utils.testutils import ParametricTestCase
from silx.gui.plot.PlotWindow import PlotWindow
from silx.gui.plot.items.image import ImageData

import numpy.random


class TestColormapDialog(TestCaseQt, ParametricTestCase):
    """Test the ColormapDialog."""
    def setUp(self):
        TestCaseQt.setUp(self)
        ParametricTestCase.setUp(self)
        self.colormap = Colormap(name='gray', vmin=10.0, vmax=20.0,
                                 normalization='linear')

        self.colormapDiag = ColormapDialog.ColormapDialog()

    def tearDown(self):
        self.qapp.processEvents()
        colormapDiag = self.colormapDiag
        self.colormapDiag = None
        if colormapDiag is not None:
            colormapDiag.close()
            colormapDiag.deleteLater()
            colormapDiag = None
        self.qapp.processEvents()
        ParametricTestCase.tearDown(self)
        TestCaseQt.tearDown(self)

    def testGUIEdition(self):
        """Make sure the colormap is correctly edited and also that the
        modification are correctly updated if an other colormapdialog is
        editing the same colormap"""
        colormapDiag2 = ColormapDialog.ColormapDialog()
        colormapDiag2.setColormap(self.colormap)
        colormapDiag2.show()
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag.show()
        self.qapp.processEvents()

        self.colormapDiag._comboBoxColormap._setCurrentName('red')
        self.colormapDiag._comboBoxNormalization.setCurrentIndex(
            self.colormapDiag._comboBoxNormalization.findData(Colormap.LOGARITHM))
        self.assertTrue(self.colormap.getName() == 'red')
        self.assertTrue(self.colormapDiag.getColormap().getName() == 'red')
        self.assertTrue(self.colormap.getNormalization() == 'log')
        self.assertTrue(self.colormap.getVMin() == 10)
        self.assertTrue(self.colormap.getVMax() == 20)
        # checked second colormap dialog
        self.assertTrue(colormapDiag2._comboBoxColormap.getCurrentName() == 'red')
        self.assertEqual(colormapDiag2._comboBoxNormalization.currentData(),
                         Colormap.LOGARITHM)
        self.assertTrue(int(colormapDiag2._minValue.getValue()) == 10)
        self.assertTrue(int(colormapDiag2._maxValue.getValue()) == 20)
        colormapDiag2.close()

    def testGUIModalOk(self):
        """Make sure the colormap is modified if gone through accept"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.setModal(True)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.setValue(None)
        self.assertTrue(self.colormap.getVMin() is None)
        self.colormapDiag._maxValue.setValue(None)
        self.mouseClick(
            widget=self.colormapDiag._buttonsModal.button(qt.QDialogButtonBox.Ok),
            button=qt.Qt.LeftButton
        )
        self.assertTrue(self.colormap.getVMin() is None)
        self.assertTrue(self.colormap.getVMax() is None)
        self.assertTrue(self.colormap.isAutoscale() is True)

    def testGUIModalCancel(self):
        """Make sure the colormap is not modified if gone through reject"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.setModal(True)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.setValue(None)
        self.assertTrue(self.colormap.getVMin() is None)
        self.mouseClick(
            widget=self.colormapDiag._buttonsModal.button(qt.QDialogButtonBox.Cancel),
            button=qt.Qt.LeftButton
        )
        self.assertTrue(self.colormap.getVMin() is not None)

    def testGUIModalClose(self):
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.setModal(False)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.setValue(None)
        self.assertTrue(self.colormap.getVMin() is None)
        self.mouseClick(
            widget=self.colormapDiag._buttonsNonModal.button(qt.QDialogButtonBox.Close),
            button=qt.Qt.LeftButton
        )
        self.assertTrue(self.colormap.getVMin() is None)

    def testGUIModalReset(self):
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.setModal(False)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.setValue(None)
        self.assertTrue(self.colormap.getVMin() is None)
        self.mouseClick(
            widget=self.colormapDiag._buttonsNonModal.button(qt.QDialogButtonBox.Reset),
            button=qt.Qt.LeftButton
        )
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag.close()

    def testGUIClose(self):
        """Make sure the colormap is modify if go through reject"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.setValue(None)
        self.assertTrue(self.colormap.getVMin() is None)
        self.colormapDiag.close()
        self.qapp.processEvents()
        self.assertTrue(self.colormap.getVMin() is None)

    def testSetColormapIsCorrect(self):
        """Make sure the interface fir the colormap when set a new colormap"""
        self.colormap.setName('red')
        self.colormapDiag.show()
        self.qapp.processEvents()
        for norm in (Colormap.NORMALIZATIONS):
            for autoscale in (True, False):
                if autoscale is True:
                    self.colormap.setVRange(None, None)
                else:
                    self.colormap.setVRange(11, 101)
                self.colormap.setNormalization(norm)
                with self.subTest(colormap=self.colormap):
                    self.colormapDiag.setColormap(self.colormap)
                    self.assertEqual(
                        self.colormapDiag._comboBoxNormalization.currentData(), norm)
                    self.assertTrue(
                        self.colormapDiag._comboBoxColormap.getCurrentName() == 'red')
                    self.assertTrue(
                        self.colormapDiag._minValue.isAutoChecked() == autoscale)
                    self.assertTrue(
                        self.colormapDiag._maxValue.isAutoChecked() == autoscale)
                    if autoscale is False:
                        self.assertTrue(self.colormapDiag._minValue.getValue() == 11)
                        self.assertTrue(self.colormapDiag._maxValue.getValue() == 101)
                        self.assertTrue(self.colormapDiag._minValue.isEnabled())
                        self.assertTrue(self.colormapDiag._maxValue.isEnabled())
                    else:
                        self.assertFalse(self.colormapDiag._minValue._numVal.isEnabled())
                        self.assertFalse(self.colormapDiag._maxValue._numVal.isEnabled())

    def testColormapDel(self):
        """Check behavior if the colormap has been deleted outside. For now
        we make sure the colormap is still running and nothing more"""
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag.show()
        self.qapp.processEvents()
        del self.colormap
        self.assertTrue(self.colormapDiag.getColormap() is None)
        self.colormapDiag._comboBoxColormap._setCurrentName('blue')

    def testColormapEditedOutside(self):
        """Make sure the GUI is still up to date if the colormap is modified
        outside"""
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag.show()
        self.qapp.processEvents()

        self.colormap.setName('red')
        self.assertTrue(
            self.colormapDiag._comboBoxColormap.getCurrentName() == 'red')
        self.colormap.setNormalization(Colormap.LOGARITHM)
        self.assertEqual(self.colormapDiag._comboBoxNormalization.currentData(),
                         Colormap.LOGARITHM)
        self.colormap.setVRange(11, 201)
        self.assertTrue(self.colormapDiag._minValue.getValue() == 11)
        self.assertTrue(self.colormapDiag._maxValue.getValue() == 201)
        self.assertTrue(self.colormapDiag._minValue._numVal.isEnabled())
        self.assertTrue(self.colormapDiag._maxValue._numVal.isEnabled())
        self.assertFalse(self.colormapDiag._minValue.isAutoChecked())
        self.assertFalse(self.colormapDiag._maxValue.isAutoChecked())
        self.colormap.setVRange(None, None)
        self.assertFalse(self.colormapDiag._minValue._numVal.isEnabled())
        self.assertFalse(self.colormapDiag._maxValue._numVal.isEnabled())
        self.assertTrue(self.colormapDiag._minValue.isAutoChecked())
        self.assertTrue(self.colormapDiag._maxValue.isAutoChecked())

    def testSetColormapScenario(self):
        """Test of a simple scenario of a colormap dialog editing several
        colormap"""
        colormap1 = Colormap(name='gray', vmin=10.0, vmax=20.0,
                             normalization='linear')
        colormap2 = Colormap(name='red', vmin=10.0, vmax=20.0,
                             normalization='log')
        colormap3 = Colormap(name='blue', vmin=None, vmax=None,
                             normalization='linear')
        self.colormapDiag.setColormap(self.colormap)
        self.colormapDiag.setColormap(colormap1)
        del colormap1
        self.colormapDiag.setColormap(colormap2)
        del colormap2
        self.colormapDiag.setColormap(colormap3)
        del colormap3

    def testNotPreferredColormap(self):
        """Test that the colormapEditor is able to edit a colormap which is not
        part of the 'prefered colormap'
        """
        def getFirstNotPreferredColormap():
            cms = Colormap.getSupportedColormaps()
            preferred = preferredColormaps()
            for cm in cms:
                if cm not in preferred:
                    return cm
            return None

        colormapName = getFirstNotPreferredColormap()
        assert colormapName is not None
        colormap = Colormap(name=colormapName)
        self.colormapDiag.setColormap(colormap)
        self.colormapDiag.show()
        self.qapp.processEvents()
        cb = self.colormapDiag._comboBoxColormap
        self.assertTrue(cb.getCurrentName() == colormapName)
        cb.setCurrentIndex(0)
        index = cb.findLutName(colormapName)
        assert index != 0  # if 0 then the rest of the test has no sense
        cb.setCurrentIndex(index)
        self.assertTrue(cb.getCurrentName() == colormapName)

    def testColormapEditableMode(self):
        """Test that the colormapDialog is correctly updated when changing the
        colormap editable status"""
        colormap = Colormap(normalization='linear', vmin=1.0, vmax=10.0)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(colormap)
        for editable in (True, False):
            with self.subTest(editable=editable):
                colormap.setEditable(editable)
                self.assertTrue(
                    self.colormapDiag._comboBoxColormap.isEnabled() is editable)
                self.assertTrue(
                    self.colormapDiag._minValue.isEnabled() is editable)
                self.assertTrue(
                    self.colormapDiag._maxValue.isEnabled() is editable)
                self.assertTrue(
                    self.colormapDiag._comboBoxNormalization.isEnabled() is editable)

        # Make sure the reset button is also set to enable when edition mode is
        # False
        self.colormapDiag.setModal(False)
        colormap.setEditable(True)
        self.colormapDiag._comboBoxNormalization.setCurrentIndex(
            self.colormapDiag._comboBoxNormalization.findData(Colormap.LOGARITHM))
        resetButton = self.colormapDiag._buttonsNonModal.button(qt.QDialogButtonBox.Reset)
        self.assertTrue(resetButton.isEnabled())
        colormap.setEditable(False)
        self.assertFalse(resetButton.isEnabled())

    def testImageData(self):
        data = numpy.random.rand(5, 5)
        self.colormapDiag.setData(data)

    def testEmptyData(self):
        data = numpy.empty((10, 0))
        self.colormapDiag.setData(data)

    def testNoneData(self):
        data = numpy.random.rand(5, 5)
        self.colormapDiag.setData(data)
        self.colormapDiag.setData(None)

    def testImageItem(self):
        """Check that an ImageData plot item can be used"""
        dialog = self.colormapDiag
        colormap = Colormap(name='gray', vmin=None, vmax=None)
        data = numpy.arange(3**2).reshape(3, 3)
        item = ImageData()
        item.setData(data, copy=False)

        dialog.setColormap(colormap)
        dialog.show()
        self.qapp.processEvents()
        dialog.setItem(item)
        vrange = dialog._getFiniteColormapRange()
        self.assertEqual(vrange, (0, 8))

    def testItemDel(self):
        """Check that the plot items are not hard linked to the dialog"""
        dialog = self.colormapDiag
        colormap = Colormap(name='gray', vmin=None, vmax=None)
        data = numpy.arange(3**2).reshape(3, 3)
        item = ImageData()
        item.setData(data, copy=False)

        dialog.setColormap(colormap)
        dialog.show()
        self.qapp.processEvents()
        dialog.setItem(item)
        previousRange = dialog._getFiniteColormapRange()
        del item
        vrange = dialog._getFiniteColormapRange()
        self.assertNotEqual(vrange, previousRange)

    def testDataDel(self):
        """Check that the data are not hard linked to the dialog"""
        dialog = self.colormapDiag
        colormap = Colormap(name='gray', vmin=None, vmax=None)
        data = numpy.arange(5)

        dialog.setColormap(colormap)
        dialog.show()
        self.qapp.processEvents()
        dialog.setData(data)
        previousRange = dialog._getFiniteColormapRange()
        del data
        vrange = dialog._getFiniteColormapRange()
        self.assertNotEqual(vrange, previousRange)

    def testDeleteWhileExec(self):
        colormapDiag = self.colormapDiag
        self.colormapDiag = None
        qt.QTimer.singleShot(1000, colormapDiag.deleteLater)
        result = colormapDiag.exec_()
        self.assertEqual(result, 0)


class TestColormapAction(TestCaseQt):
    def setUp(self):
        TestCaseQt.setUp(self)
        self.plot = PlotWindow()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)

        self.colormap1 = Colormap(name='blue', vmin=0.0, vmax=1.0,
                                  normalization='linear')
        self.colormap2 = Colormap(name='red', vmin=10.0, vmax=100.0,
                                  normalization='log')
        self.defaultColormap = self.plot.getDefaultColormap()

        self.plot.getColormapAction()._actionTriggered(checked=True)
        self.colormapDialog = self.plot.getColormapAction()._dialog
        self.colormapDialog.setAttribute(qt.Qt.WA_DeleteOnClose)

    def tearDown(self):
        self.colormapDialog.close()
        self.plot.close()
        del self.colormapDialog
        del self.plot
        TestCaseQt.tearDown(self)

    def testActiveColormap(self):
        self.assertTrue(self.colormapDialog.getColormap() is self.defaultColormap)

        self.plot.addImage(data=numpy.random.rand(10, 10), legend='img1',
                           origin=(0, 0),
                           colormap=self.colormap1)
        self.plot.setActiveImage('img1')
        self.assertTrue(self.colormapDialog.getColormap() is self.colormap1)

        self.plot.addImage(data=numpy.random.rand(10, 10), legend='img2',
                           origin=(0, 0),
                           colormap=self.colormap2)
        self.plot.addImage(data=numpy.random.rand(10, 10), legend='img3',
                           origin=(0, 0))

        self.plot.setActiveImage('img3')
        self.assertTrue(self.colormapDialog.getColormap() is self.defaultColormap)
        self.plot.getActiveImage().setColormap(self.colormap2)
        self.assertTrue(self.colormapDialog.getColormap() is self.colormap2)

        self.plot.remove('img2')
        self.plot.remove('img3')
        self.plot.remove('img1')
        self.assertTrue(self.colormapDialog.getColormap() is self.defaultColormap)

    def testShowHideColormapDialog(self):
        self.plot.getColormapAction()._actionTriggered(checked=False)
        self.assertFalse(self.plot.getColormapAction().isChecked())
        self.plot.getColormapAction()._actionTriggered(checked=True)
        self.assertTrue(self.plot.getColormapAction().isChecked())
        self.plot.addImage(data=numpy.random.rand(10, 10), legend='img1',
                           origin=(0, 0),
                           colormap=self.colormap1)
        self.colormap1.setName('red')
        self.plot.getColormapAction()._actionTriggered()
        self.colormap1.setName('blue')
        self.colormapDialog.close()
        self.assertFalse(self.plot.getColormapAction().isChecked())


def suite():
    test_suite = unittest.TestSuite()
    for testClass in (TestColormapDialog, TestColormapAction):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
            testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
