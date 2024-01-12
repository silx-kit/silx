# /*##########################################################################
#
# Copyright (c) 2016-2024 European Synchrotron Radiation Facility
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

from silx.gui import qt
from silx.gui.dialog import ColormapDialog
from silx.gui.colors import Colormap, preferredColormaps
from silx.gui.plot.items.image import ImageData

import numpy


def testGUIEdition(qWidgetFactory):
    """Make sure the colormap is correctly edited and also that the
    modification are correctly updated if an other colormapdialog is
    editing the same colormap"""
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setColormap(colormap)
    dialog2 = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog2.setColormap(colormap)

    dialog._comboBoxColormap._setCurrentName("red")
    dialog._comboBoxNormalization.setCurrentIndex(
        dialog._comboBoxNormalization.findData(Colormap.LOGARITHM)
    )
    assert colormap.getName() == "red"
    assert dialog.getColormap().getName() == "red"
    assert colormap.getNormalization() == "log"
    assert colormap.getVMin() == 10
    assert colormap.getVMax() == 20
    # checked second colormap dialog
    assert dialog2._comboBoxColormap.getCurrentName() == "red"
    assert dialog2._comboBoxNormalization.currentData() == Colormap.LOGARITHM
    assert int(dialog2._minValue.getValue()) == 10
    assert int(dialog2._maxValue.getValue()) == 20


def testGUIModalOk(qapp, qapp_utils, qWidgetFactory):
    """Make sure the colormap is modified if gone through accept"""
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    assert colormap.isAutoscale() is False
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setModal(True)
    qapp.processEvents()

    dialog.setColormap(colormap)
    assert colormap.getVMin() is not None
    dialog._minValue.sigAutoScaleChanged.emit(True)
    assert colormap.getVMin() is None
    dialog._maxValue.sigAutoScaleChanged.emit(True)
    qapp_utils.mouseClick(
        widget=dialog._buttonsModal.button(qt.QDialogButtonBox.Ok),
        button=qt.Qt.LeftButton,
    )
    assert colormap.getVMin() is None
    assert colormap.getVMax() is None
    assert colormap.isAutoscale() is True


def testGUIModalCancel(qapp, qapp_utils, qWidgetFactory):
    """Make sure the colormap is not modified if gone through reject"""
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    assert colormap.isAutoscale() is False
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setModal(True)
    qapp.processEvents()

    dialog.setColormap(colormap)
    assert colormap.getVMin() is not None
    dialog._minValue.sigAutoScaleChanged.emit(True)
    assert colormap.getVMin() is None
    qapp_utils.mouseClick(
        widget=dialog._buttonsModal.button(qt.QDialogButtonBox.Cancel),
        button=qt.Qt.LeftButton,
    )
    assert colormap.getVMin() is not None


def testGUIModalClose(qapp, qapp_utils, qWidgetFactory):
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    assert colormap.isAutoscale() is False
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setModal(False)
    qapp.processEvents()

    dialog.setColormap(colormap)
    assert colormap.getVMin() is not None
    dialog._minValue.sigAutoScaleChanged.emit(True)
    assert colormap.getVMin() is None
    qapp_utils.mouseClick(
        widget=dialog._buttonsNonModal.button(qt.QDialogButtonBox.Close),
        button=qt.Qt.LeftButton,
    )
    assert colormap.getVMin() is None


def testGUIModalReset(qapp, qapp_utils, qWidgetFactory):
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    assert colormap.isAutoscale() is False
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setModal(False)
    dialog.show()
    qapp.processEvents()
    dialog.setColormap(colormap)
    assert colormap.getVMin() is not None
    dialog._minValue.sigAutoScaleChanged.emit(True)
    assert colormap.getVMin() is None
    qapp_utils.mouseClick(
        widget=dialog._buttonsNonModal.button(qt.QDialogButtonBox.Reset),
        button=qt.Qt.LeftButton,
    )
    assert colormap.getVMin() is not None
    dialog.close()


def testGUIClose(qapp, qWidgetFactory):
    """Make sure the colormap is modify if go through reject"""
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    assert colormap.isAutoscale() is False
    qapp.processEvents()

    dialog.setColormap(colormap)
    assert colormap.getVMin() is not None
    dialog._minValue.sigAutoScaleChanged.emit(True)
    assert colormap.getVMin() is None
    dialog.close()
    qapp.processEvents()
    assert colormap.getVMin() is None


@pytest.mark.parametrize("norm", Colormap.NORMALIZATIONS)
@pytest.mark.parametrize("autoscale", (True, False))
def testSetColormapIsCorrect(norm, autoscale, qapp, qWidgetFactory):
    """Make sure the interface fir the colormap when set a new colormap"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)

    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    colormap.setName("red")
    if autoscale is True:
        colormap.setVRange(None, None)
    else:
        colormap.setVRange(11, 101)
    colormap.setNormalization(norm)
    dialog.setColormap(colormap)
    qapp.processEvents()

    assert dialog._comboBoxNormalization.currentData() == norm
    assert dialog._comboBoxColormap.getCurrentName() == "red"
    assert dialog._minValue.isAutoChecked() == autoscale
    assert dialog._maxValue.isAutoChecked() == autoscale
    if autoscale is False:
        assert dialog._minValue.getValue() == 11
        assert dialog._maxValue.getValue() == 101
        assert dialog._minValue.isEnabled()
        assert dialog._maxValue.isEnabled()
    else:
        assert dialog._minValue._numVal.isReadOnly()
        assert dialog._maxValue._numVal.isReadOnly()


def testColormapDel(qapp, qWidgetFactory):
    """Check behavior if the colormap has been deleted outside. For now
    we make sure the colormap is still running and nothing more"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(name="gray")
    dialog.setColormap(colormap)
    qapp.processEvents()

    colormap = None
    assert dialog.getColormap() is None
    dialog._comboBoxColormap._setCurrentName("blue")


def testColormapEditedOutside(qapp, qWidgetFactory):
    """Make sure the GUI is still up to date if the colormap is modified
    outside"""
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    dialog.setColormap(colormap)
    qapp.processEvents()

    colormap.setName("red")
    assert dialog._comboBoxColormap.getCurrentName() == "red"
    colormap.setNormalization(Colormap.LOGARITHM)
    assert dialog._comboBoxNormalization.currentData() == Colormap.LOGARITHM
    colormap.setVRange(11, 201)
    assert dialog._minValue.getValue() == 11
    assert dialog._maxValue.getValue() == 201
    assert not (dialog._minValue._numVal.isReadOnly())
    assert not (dialog._maxValue._numVal.isReadOnly())
    assert not (dialog._minValue.isAutoChecked())
    assert not (dialog._maxValue.isAutoChecked())
    colormap.setVRange(None, None)
    qapp.processEvents()

    assert dialog._minValue._numVal.isReadOnly()
    assert dialog._maxValue._numVal.isReadOnly()
    assert dialog._minValue.isAutoChecked()
    assert dialog._maxValue.isAutoChecked()


def testSetColormapScenario(qWidgetFactory):
    """Test of a simple scenario of a colormap dialog editing several
    colormap"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    colormap1 = Colormap(name="gray", vmin=10.0, vmax=20.0, normalization="linear")
    colormap2 = Colormap(name="red", vmin=10.0, vmax=20.0, normalization="log")
    colormap3 = Colormap(name="blue", vmin=None, vmax=None, normalization="linear")

    dialog.setColormap(colormap)
    dialog.setColormap(colormap1)
    del colormap1
    dialog.setColormap(colormap2)
    del colormap2
    dialog.setColormap(colormap3)
    del colormap3


def testNotPreferredColormap(qapp, qWidgetFactory):
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

    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormapName = getFirstNotPreferredColormap()
    assert colormapName is not None
    colormap = Colormap(name=colormapName)
    dialog.setColormap(colormap)
    qapp.processEvents()

    cb = dialog._comboBoxColormap
    assert cb.getCurrentName() == colormapName
    cb.setCurrentIndex(0)
    index = cb.findLutName(colormapName)
    assert index != 0  # if 0 then the rest of the test has no sense
    cb.setCurrentIndex(index)
    assert cb.getCurrentName() == colormapName


def testColormapEditableMode(qWidgetFactory):
    """Test that the colormapDialog is correctly updated when changing the
    colormap editable status"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(normalization="linear", vmin=1.0, vmax=10.0)

    dialog.setColormap(colormap)

    for editable in (True, False):
        colormap.setEditable(editable)
        assert dialog._comboBoxColormap.isEnabled() is editable
        assert dialog._minValue.isEnabled() is editable
        assert dialog._maxValue.isEnabled() is editable
        assert dialog._comboBoxNormalization.isEnabled() is editable

    # Make sure the reset button is also set to enable when edition mode is
    # False
    dialog.setModal(False)
    colormap.setEditable(True)
    dialog._comboBoxNormalization.setCurrentIndex(
        dialog._comboBoxNormalization.findData(Colormap.LOGARITHM)
    )
    resetButton = dialog._buttonsNonModal.button(qt.QDialogButtonBox.Reset)
    assert resetButton.isEnabled()
    colormap.setEditable(False)
    assert not (resetButton.isEnabled())


def testImageData(qWidgetFactory):
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    data = numpy.random.rand(5, 5)
    dialog.setData(data)


def testEmptyData(qWidgetFactory):
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    data = numpy.empty((10, 0))
    dialog.setData(data)


def testNoneData(qWidgetFactory):
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    data = numpy.random.rand(5, 5)
    dialog.setData(data)
    dialog.setData(None)


def testImageItem(qapp, qWidgetFactory):
    """Check that an ImageData plot item can be used"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(name="gray", vmin=None, vmax=None)
    data = numpy.arange(3**2).reshape(3, 3)
    item = ImageData()
    item.setData(data, copy=False)

    dialog.setColormap(colormap)
    qapp.processEvents()

    dialog.setItem(item)
    vrange = dialog._getFiniteColormapRange()
    assert vrange == (0, 8)


def testItemDel(qapp, qWidgetFactory):
    """Check that the plot items are not hard linked to the dialog"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(name="gray", vmin=None, vmax=None)
    data = numpy.arange(3**2).reshape(3, 3)
    item = ImageData()
    item.setData(data, copy=False)

    dialog.setColormap(colormap)
    dialog.show()
    qapp.processEvents()
    dialog.setItem(item)
    previousRange = dialog._getFiniteColormapRange()
    del item
    vrange = dialog._getFiniteColormapRange()
    assert vrange != previousRange


def testDataDel(qapp, qWidgetFactory):
    """Check that the data are not hard linked to the dialog"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    colormap = Colormap(name="gray", vmin=None, vmax=None)
    data = numpy.arange(5)

    dialog.setColormap(colormap)
    qapp.processEvents()

    dialog.setData(data)
    previousRange = dialog._getFiniteColormapRange()
    del data
    vrange = dialog._getFiniteColormapRange()
    assert vrange != previousRange


def testDeleteWhileExec(qWidgetFactory):
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)
    qt.QTimer.singleShot(1000, dialog.deleteLater)
    result = dialog.exec()
    assert result == 0


def testUpdateImageData(qapp, qWidgetFactory):
    """Test that range/histogram takes into account item updates"""
    dialog = qWidgetFactory(ColormapDialog.ColormapDialog)

    item = ImageData()
    item.setColormap(Colormap())
    dialog.setItem(item)
    dialog.setColormap(item.getColormap())
    qapp.processEvents()

    assert dialog._histoWidget.getFiniteRange() == (0, 1)

    item.setData([(1, 2), (3, 4)])

    assert dialog._histoWidget.getFiniteRange() == (1, 4)
