# /*##########################################################################
#
# Copyright (c) 2016-2022 European Synchrotron Radiation Facility
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


import pytest
import weakref

from silx.gui import qt
from silx.gui.dialog import ColormapDialog
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.colors import Colormap, preferredColormaps
from silx.utils.testutils import ParametricTestCase
from silx.gui.plot.items.image import ImageData

import numpy


@pytest.fixture
def colormap():
    colormap = Colormap(name='gray',
                        vmin=10.0, vmax=20.0,
                        normalization='linear')
    yield colormap


@pytest.fixture
def colormapDialog(qapp):
    dialog = ColormapDialog.ColormapDialog()
    dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
    yield weakref.proxy(dialog)
    qapp.processEvents()
    from silx.gui.qt import inspect
    if inspect.isValid(dialog):
        dialog.close()
        del dialog
        qapp.processEvents()


@pytest.fixture
def colormap_class_attr(request, qapp_utils, colormap, colormapDialog):
    """Provides few fixtures to a class as class attribute

    Used as transition from TestCase to pytest
    """
    request.cls.qapp_utils = qapp_utils
    request.cls.colormap = colormap
    request.cls.colormapDiag = colormapDialog
    yield
    request.cls.qapp_utils = None
    request.cls.colormap = None
    request.cls.colormapDiag = None


@pytest.mark.usefixtures("colormap_class_attr")
class TestColormapDialog(TestCaseQt, ParametricTestCase):

    def testGUIEdition(self):
        """Make sure the colormap is correctly edited and also that the
        modification are correctly updated if an other colormapdialog is
        editing the same colormap"""
        colormapDiag2 = ColormapDialog.ColormapDialog()
        colormapDiag2.setAttribute(qt.Qt.WA_DeleteOnClose)
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
        del colormapDiag2
        self.qapp.processEvents()

    def testGUIModalOk(self):
        """Make sure the colormap is modified if gone through accept"""
        assert self.colormap.isAutoscale() is False
        self.colormapDiag.setModal(True)
        self.colormapDiag.show()
        self.qapp.processEvents()
        self.colormapDiag.setColormap(self.colormap)
        self.assertTrue(self.colormap.getVMin() is not None)
        self.colormapDiag._minValue.sigAutoScaleChanged.emit(True)
        self.assertTrue(self.colormap.getVMin() is None)
        self.colormapDiag._maxValue.sigAutoScaleChanged.emit(True)
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
        self.colormapDiag._minValue.sigAutoScaleChanged.emit(True)
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
        self.colormapDiag._minValue.sigAutoScaleChanged.emit(True)
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
        self.colormapDiag._minValue.sigAutoScaleChanged.emit(True)
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
        self.colormapDiag._minValue.sigAutoScaleChanged.emit(True)
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
                        self.assertTrue(self.colormapDiag._minValue._numVal.isReadOnly())
                        self.assertTrue(self.colormapDiag._maxValue._numVal.isReadOnly())

    def testColormapDel(self):
        """Check behavior if the colormap has been deleted outside. For now
        we make sure the colormap is still running and nothing more"""
        colormap = Colormap(name='gray')
        self.colormapDiag.setColormap(colormap)
        self.colormapDiag.show()
        self.qapp.processEvents()
        colormap = None
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
        self.assertFalse(self.colormapDiag._minValue._numVal.isReadOnly())
        self.assertFalse(self.colormapDiag._maxValue._numVal.isReadOnly())
        self.assertFalse(self.colormapDiag._minValue.isAutoChecked())
        self.assertFalse(self.colormapDiag._maxValue.isAutoChecked())
        self.colormap.setVRange(None, None)
        self.qapp.processEvents()
        self.assertTrue(self.colormapDiag._minValue._numVal.isReadOnly())
        self.assertTrue(self.colormapDiag._maxValue._numVal.isReadOnly())
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
        result = colormapDiag.exec()
        self.assertEqual(result, 0)
