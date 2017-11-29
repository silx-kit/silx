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
"""
This module provides a tree widget to set/view parameters of a ScalarFieldView.
"""

from __future__ import absolute_import

__authors__ = ["D. N."]
__license__ = "MIT"
__date__ = "02/10/2017"

import logging

import numpy

from silx.gui import qt
from silx.gui.icons import getQIcon
from silx.gui.plot.Colormap import Colormap
from silx.gui.widgets.FloatEdit import FloatEdit

from .ScalarFieldView import Isosurface
from . import params

_logger = logging.getLogger(__name__)


# View settings ###############################################################

class BoundingBoxItem(params.SubjectItem):
    """Bounding box, axes labels and grid visibility item.

    Item is checkable.
    """
    ITEM_NAME = 'Bounding Box'

    def _init(self):
        visible = self.getSubject().isBoundingBoxVisible()
        self.setCheckable(True)
        self.setCheckState(qt.Qt.Checked if visible else qt.Qt.Unchecked)

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != self.getSubject().isBoundingBoxVisible():
            self.getSubject().setBoundingBoxVisible(checked)


class OrientationIndicatorItem(params.SubjectItem):
    """Orientation indicator visibility item.

    Item is checkable.
    """
    ITEM_NAME = 'Axes indicator'

    def _init(self):
        plot3d = self.getSubject().getPlot3DWidget()
        visible = plot3d.isOrientationIndicatorVisible()
        self.setCheckable(True)
        self.setCheckState(qt.Qt.Checked if visible else qt.Qt.Unchecked)

    def leftClicked(self):
        plot3d = self.getSubject().getPlot3DWidget()
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != plot3d.isOrientationIndicatorVisible():
            plot3d.setOrientationIndicatorVisible(checked)


class ViewSettingsItem(qt.QStandardItem):
    """Viewport settings"""

    def __init__(self, subject, *args):

        super(ViewSettingsItem, self).__init__(*args)

        self.setEditable(False)

        classes = (params.BackgroundColorItem,
                   params.ForegroundColorItem,
                   params.HighlightColorItem,
                   BoundingBoxItem,
                   OrientationIndicatorItem)
        for cls in classes:
            titleItem = qt.QStandardItem(cls.ITEM_NAME)
            titleItem.setEditable(False)
            self.appendRow([titleItem, cls(subject)])

        nameItem = params.DirectionalLightItem(
            subject.getPlot3DWidget(), 'Light Direction')
        valueItem = qt.QStandardItem()
        self.appendRow([nameItem, valueItem])


# Data information ############################################################

class DataChangedItem(params.SubjectItem):
    """
    Base class for items listening to ScalarFieldView.sigDataChanged
    """

    def getSignals(self):
        subject = self.getSubject()
        if subject:
            return subject.sigDataChanged
        return None

    def _init(self):
        self._subjectChanged()


class DataTypeItem(DataChangedItem):
    itemName = 'dtype'

    def _pullData(self):
        data = self.getSubject().getData(copy=False)
        return ((data is not None) and str(data.dtype)) or 'N/A'


class DataShapeItem(DataChangedItem):
    itemName = 'size'

    def _pullData(self):
        data = self.getSubject().getData(copy=False)
        if data is None:
            return 'N/A'
        else:
            return str(list(reversed(data.shape)))


class OffsetItem(DataChangedItem):
    itemName = 'offset'

    def _pullData(self):
        offset = self.getSubject().getTranslation()
        return ((offset is not None) and str(offset)) or 'N/A'


class ScaleItem(DataChangedItem):
    itemName = 'scale'

    def _pullData(self):
        scale = self.getSubject().getScale()
        return ((scale is not None) and str(scale)) or 'N/A'


class DataSetItem(qt.QStandardItem):

    def __init__(self, subject, *args):

        super(DataSetItem, self).__init__(*args)

        self.setEditable(False)

        klasses = [DataTypeItem, DataShapeItem, OffsetItem, ScaleItem]
        for klass in klasses:
            titleItem = qt.QStandardItem(klass.itemName)
            titleItem.setEditable(False)
            self.appendRow([titleItem, klass(subject)])


# Isosurface ##################################################################

class IsoSurfaceRootItem(params.SubjectItem):
    """
    Root (i.e : column index 0) Isosurface item.
    """

    def getSignals(self):
        subject = self.getSubject()
        return [subject.sigColorChanged,
                subject.sigVisibilityChanged]

    def _subjectChanged(self, signalIdx=None, args=None, kwargs=None):
        if signalIdx == 0:
            color = self.getSubject().getColor()
            self.setData(color, qt.Qt.DecorationRole)
        elif signalIdx == 1:
            visible = args[0]
            self.setCheckState((visible and qt.Qt.Checked) or qt.Qt.Unchecked)

    def _init(self):
        self.setCheckable(True)

        isosurface = self.getSubject()
        color = isosurface.getColor()
        visible = isosurface.isVisible()
        self.setData(color, qt.Qt.DecorationRole)
        self.setCheckState((visible and qt.Qt.Checked) or qt.Qt.Unchecked)

        nameItem = qt.QStandardItem('Level')
        sliderItem = IsoSurfaceLevelSlider(self.getSubject())
        self.appendRow([nameItem, sliderItem])

        nameItem = qt.QStandardItem('Color')
        nameItem.setEditable(False)
        valueItem = IsoSurfaceColorItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Opacity')
        nameItem.setTextAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
        nameItem.setEditable(False)
        valueItem = IsoSurfaceAlphaItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem()
        nameItem.setEditable(False)
        valueItem = IsoSurfaceAlphaLegendItem(self.getSubject())
        valueItem.setEditable(False)
        self.appendRow([nameItem, valueItem])

    def queryRemove(self, view=None):
        buttons = qt.QMessageBox.Ok | qt.QMessageBox.Cancel
        ans = qt.QMessageBox.question(view,
                                      'Remove isosurface',
                                      'Remove the selected iso-surface?',
                                      buttons=buttons)
        if ans == qt.QMessageBox.Ok:
            sfview = self.getSubject().parent()
            if sfview:
                sfview.removeIsosurface(self.getSubject())
                return False
        return False

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        visible = self.getSubject().isVisible()
        if checked != visible:
            self.getSubject().setVisible(checked)


class IsoSurfaceLevelItem(params.SubjectItem):
    """
    Base class for the isosurface level items.
    """
    editable = True

    def getSignals(self):
        subject = self.getSubject()
        return [subject.sigLevelChanged,
                subject.sigVisibilityChanged]

    def getEditor(self, parent, option, index):
        return FloatEdit(parent)

    def setEditorData(self, editor):
        editor.setValue(self._pullData())
        return False

    def _setModelData(self, editor):
        self._pushData(editor.value())
        return True

    def _pullData(self):
        return self.getSubject().getLevel()

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.getSubject().setLevel(value)
        return self.getSubject().getLevel()


class _IsoLevelSlider(qt.QSlider):
    """QSlider used for iso-surface level"""

    def __init__(self, parent, subject):
        super(_IsoLevelSlider, self).__init__(parent=parent)
        self._subject = subject

        self.sliderReleased.connect(self.__sliderReleased)

        self._subject.sigLevelChanged.connect(self.setLevel)
        self._subject.parent().sigDataChanged.connect(self.__dataChanged)

    def setLevel(self, level):
        """Set slider from iso-surface level"""
        dataRange = self._subject.parent().getDataRange()

        if dataRange is not None:
            width = dataRange[-1] - dataRange[0]
            if width > 0:
                sliderWidth = self.maximum() - self.minimum()
                sliderPosition = sliderWidth * (level - dataRange[0]) / width
                self.setValue(sliderPosition)

    def __dataChanged(self):
        """Handles data update to refresh slider range if needed"""
        self.setLevel(self._subject.getLevel())

    def __sliderReleased(self):
        value = self.value()
        dataRange = self._subject.parent().getDataRange()
        if dataRange is not None:
            min_, _, max_ = dataRange
            width = max_ - min_
            sliderWidth = self.maximum() - self.minimum()
            level = min_ + width * value / sliderWidth
            self._subject.setLevel(level)


class IsoSurfaceLevelSlider(IsoSurfaceLevelItem):
    """
    Isosurface level item with a slider editor.
    """
    nTicks = 1000
    persistent = True

    def getEditor(self, parent, option, index):
        editor = _IsoLevelSlider(parent, self.getSubject())
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(0)
        editor.setMaximum(self.nTicks)

        editor.setSingleStep(1)

        editor.setLevel(self.getSubject().getLevel())
        return editor

    def setEditorData(self, editor):
        return True

    def _setModelData(self, editor):
        return True


class IsoSurfaceColorItem(params.SubjectItem):
    """
    Isosurface color item.
    """
    editable = True
    persistent = True

    def getSignals(self):
        return self.getSubject().sigColorChanged

    def getEditor(self, parent, option, index):
        editor = params.ColorEditor(parent)
        color = self.getSubject().getColor()
        color.setAlpha(255)
        editor.setColor(color)
        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.sigColorChanged.connect(
            lambda color: self.__editorChanged(color))
        return editor

    def __editorChanged(self, color):
        color.setAlpha(self.getSubject().getColor().alpha())
        self.getSubject().setColor(color)

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.getSubject().setColor(value)
        return self.getSubject().getColor()


class IsoSurfaceAlphaItem(params.SubjectItem):
    """
    Isosurface alpha item.
    """
    editable = True
    persistent = True

    def _init(self):
        pass

    def getSignals(self):
        return self.getSubject().sigColorChanged

    def getEditor(self, parent, option, index):
        editor = qt.QSlider(parent)
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(0)
        editor.setMaximum(255)

        color = self.getSubject().getColor()
        editor.setValue(color.alpha())

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.valueChanged.connect(
            lambda value: self.__editorChanged(value))

        return editor

    def __editorChanged(self, value):
        color = self.getSubject().getColor()
        color.setAlpha(value)
        self.getSubject().setColor(color)

    def setEditorData(self, editor):
        return True

    def _setModelData(self, editor):
        return True


class IsoSurfaceAlphaLegendItem(params.SubjectItem):
    """Legend to place under opacity slider"""

    editable = False
    persistent = True

    def getEditor(self, parent, option, index):
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(qt.QLabel('0'))
        layout.addStretch(1)
        layout.addWidget(qt.QLabel('1'))

        editor = qt.QWidget(parent)
        editor.setLayout(layout)
        return editor


class IsoSurfaceCount(params.SubjectItem):
    """
    Item displaying the number of isosurfaces.
    """

    def getSignals(self):
        subject = self.getSubject()
        return [subject.sigIsosurfaceAdded, subject.sigIsosurfaceRemoved]

    def _pullData(self):
        return len(self.getSubject().getIsosurfaces())


class IsoSurfaceAddRemoveWidget(qt.QWidget):

    sigViewTask = qt.Signal(str)
    """Signal for the tree view to perform some task"""

    def __init__(self, parent, item):
        super(IsoSurfaceAddRemoveWidget, self).__init__(parent)
        self._item = item
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        addBtn = qt.QToolButton()
        addBtn.setText('+')
        addBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(addBtn)
        addBtn.clicked.connect(self.__addClicked)

        removeBtn = qt.QToolButton()
        removeBtn.setText('-')
        removeBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(removeBtn)
        removeBtn.clicked.connect(self.__removeClicked)

        layout.addStretch(1)

    def __addClicked(self):
        sfview = self._item.getSubject()
        if not sfview:
            return
        dataRange = sfview.getDataRange()
        if dataRange is None:
            dataRange = [0, 1]

        sfview.addIsosurface(
            numpy.mean((dataRange[0], dataRange[-1])), '#0000FF')

    def __removeClicked(self):
        self.sigViewTask.emit('remove_iso')


class IsoSurfaceAddRemoveItem(params.SubjectItem):
    """
    Item displaying a simple QToolButton allowing to add an isosurface.
    """
    persistent = True

    def getEditor(self, parent, option, index):
        return IsoSurfaceAddRemoveWidget(parent, self)


class IsoSurfaceGroup(params.SubjectItem):
    """
    Root item for the list of isosurface items.
    """
    def getSignals(self):
        subject = self.getSubject()
        return [subject.sigIsosurfaceAdded, subject.sigIsosurfaceRemoved]

    def _subjectChanged(self, signalIdx=None, args=None, kwargs=None):
        if signalIdx == 0:
            if len(args) >= 1:
                isosurface = args[0]
                if not isinstance(isosurface, Isosurface):
                    raise ValueError('Expected an isosurface instance.')
                self.__addIsosurface(isosurface)
            else:
                raise ValueError('Expected an isosurface instance.')
        elif signalIdx == 1:
            if len(args) >= 1:
                isosurface = args[0]
                if not isinstance(isosurface, Isosurface):
                    raise ValueError('Expected an isosurface instance.')
                self.__removeIsosurface(isosurface)
            else:
                raise ValueError('Expected an isosurface instance.')

    def __addIsosurface(self, isosurface):
        valueItem = IsoSurfaceRootItem(subject=isosurface)
        nameItem = IsoSurfaceLevelItem(subject=isosurface)
        self.insertRow(max(0, self.rowCount() - 1), [valueItem, nameItem])

    def __removeIsosurface(self, isosurface):
        for row in range(self.rowCount()):
            child = self.child(row)
            getSubject = getattr(child, 'getSubject', None)
            if getSubject is not None and getSubject() == isosurface:
                self.takeRow(row)
                break

    def _init(self):
        nameItem = IsoSurfaceAddRemoveItem(self.getSubject())
        valueItem = qt.QStandardItem()
        valueItem.setEditable(False)
        self.appendRow([nameItem, valueItem])

        isosurfaces = self.getSubject().getIsosurfaces()
        for isosurface in isosurfaces:
            self.__addIsosurface(isosurface)


# Cutting Plane ###############################################################

class ColormapBase(params.SubjectItem):
    """
    Mixin class for colormap items.
    """

    def getSignals(self):
        return [self.getSubject().getCutPlanes()[0].sigColormapChanged]


class PlaneMinRangeItem(ColormapBase):
    """
    colormap minVal item.
    Editor is a QLineEdit with a QDoubleValidator
    """
    editable = True

    def _pullData(self):
        colormap = self.getSubject().getCutPlanes()[0].getColormap()
        auto = colormap.isAutoscale()
        if auto == self.isEnabled():
            self._enableRow(not auto)
        return colormap.getVMin()

    def _pushData(self, value, role=qt.Qt.UserRole):
        self._setVMin(value)

    def _setVMin(self, value):
        colormap = self.getSubject().getCutPlanes()[0].getColormap()
        vMin = value
        vMax = colormap.getVMax()

        if vMax is not None and value > vMax:
            vMin = vMax
            vMax = value
        colormap.setVRange(vMin, vMax)

    def getEditor(self, parent, option, index):
        return FloatEdit(parent)

    def setEditorData(self, editor):
        editor.setValue(self._pullData())
        return True

    def _setModelData(self, editor):
        value = editor.value()
        self._setVMin(value)
        return True


class PlaneMaxRangeItem(ColormapBase):
    """
    colormap maxVal item.
    Editor is a QLineEdit with a QDoubleValidator
    """
    editable = True

    def _pullData(self):
        colormap = self.getSubject().getCutPlanes()[0].getColormap()
        auto = colormap.isAutoscale()
        if auto == self.isEnabled():
            self._enableRow(not auto)
        return self.getSubject().getCutPlanes()[0].getColormap().getVMax()

    def _setVMax(self, value):
        colormap = self.getSubject().getCutPlanes()[0].getColormap()
        vMin = colormap.getVMin()
        vMax = value
        if vMin is not None and value < vMin:
            vMax = vMin
            vMin = value
        colormap.setVRange(vMin, vMax)

    def getEditor(self, parent, option, index):
        return FloatEdit(parent)

    def setEditorData(self, editor):
        editor.setText(str(self._pullData()))
        return True

    def _setModelData(self, editor):
        value = editor.value()
        self._setVMax(value)
        return True


class PlaneOrientationItem(params.SubjectItem):
    """
    Plane orientation item.
    Editor is a QComboBox.
    """
    editable = True

    _PLANE_ACTIONS = (
        ('3d-plane-normal-x', 'Plane 0',
         'Set plane perpendicular to red axis', (1., 0., 0.)),
        ('3d-plane-normal-y', 'Plane 1',
         'Set plane perpendicular to green axis', (0., 1., 0.)),
        ('3d-plane-normal-z', 'Plane 2',
         'Set plane perpendicular to blue axis', (0., 0., 1.)),
    )

    def getSignals(self):
        return [self.getSubject().getCutPlanes()[0].sigPlaneChanged]

    def _pullData(self):
        currentNormal = self.getSubject().getCutPlanes()[0].getNormal()
        for _, text, _, normal in self._PLANE_ACTIONS:
            if numpy.array_equal(normal, currentNormal):
                return text
        return ''

    def getEditor(self, parent, option, index):
        editor = qt.QComboBox(parent)
        for iconName, text, tooltip, normal in self._PLANE_ACTIONS:
            editor.addItem(getQIcon(iconName), text)

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.currentIndexChanged[int].connect(
            lambda index: self.__editorChanged(index))
        return editor

    def __editorChanged(self, index):
        normal = self._PLANE_ACTIONS[index][3]
        plane = self.getSubject().getCutPlanes()[0]
        plane.setNormal(normal)
        plane.moveToCenter()

    def setEditorData(self, editor):
        currentText = self._pullData()
        index = 0
        for normIdx, (_, text, _, _) in enumerate(self._PLANE_ACTIONS):
            if text == currentText:
                index = normIdx
                break
        editor.setCurrentIndex(index)
        return True

    def _setModelData(self, editor):
        return True


class PlaneInterpolationItem(params.SubjectItem):
    """Toggle cut plane interpolation method: nearest or linear.

    Item is checkable
    """

    def _init(self):
        interpolation = self.getSubject().getCutPlanes()[0].getInterpolation()
        self.setCheckable(True)
        self.setCheckState(
            qt.Qt.Checked if interpolation == 'linear' else qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def getSignals(self):
        return [self.getSubject().getCutPlanes()[0].sigInterpolationChanged]

    def leftClicked(self):
        checked = self.checkState() == qt.Qt.Checked
        self._setInterpolation('linear' if checked else 'nearest')

    def _pullData(self):
        interpolation = self.getSubject().getCutPlanes()[0].getInterpolation()
        self._setInterpolation(interpolation)
        return interpolation[0].upper() + interpolation[1:]

    def _setInterpolation(self, interpolation):
        self.getSubject().getCutPlanes()[0].setInterpolation(interpolation)


class PlaneDisplayBelowMinItem(params.SubjectItem):
    """Toggle whether to display or not values <= colormap min of the cut plane

    Item is checkable
    """

    def _init(self):
        display = self.getSubject().getCutPlanes()[0].getDisplayValuesBelowMin()
        self.setCheckable(True)
        self.setCheckState(
            qt.Qt.Checked if display else qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def getSignals(self):
        return [self.getSubject().getCutPlanes()[0].sigTransparencyChanged]

    def leftClicked(self):
        checked = self.checkState() == qt.Qt.Checked
        self._setDisplayValuesBelowMin(checked)

    def _pullData(self):
        display = self.getSubject().getCutPlanes()[0].getDisplayValuesBelowMin()
        self._setDisplayValuesBelowMin(display)
        return "Displayed" if display else "Hidden"

    def _setDisplayValuesBelowMin(self, display):
        self.getSubject().getCutPlanes()[0].setDisplayValuesBelowMin(display)


class PlaneColormapItem(ColormapBase):
    """
    colormap name item.
    Editor is a QComboBox
    """
    editable = True

    listValues = ['gray', 'reversed gray',
                  'temperature', 'red',
                  'green', 'blue',
                  'viridis', 'magma', 'inferno', 'plasma']

    def getEditor(self, parent, option, index):
        editor = qt.QComboBox(parent)
        editor.addItems(self.listValues)

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.currentIndexChanged[int].connect(
            lambda index: self.__editorChanged(index))

        return editor

    def __editorChanged(self, index):
        colormapName = self.listValues[index]
        colormap = self.getSubject().getCutPlanes()[0].getColormap()
        colormap.setName(colormapName)

    def setEditorData(self, editor):
        colormapName = self.getSubject().getCutPlanes()[0].getColormap().getName()
        try:
            index = self.listValues.index(colormapName)
        except ValueError:
            _logger.error('Unsupported colormap: %s', colormapName)
        else:
            editor.setCurrentIndex(index)
        return True

    def _setModelData(self, editor):
        self.__editorChanged(editor.currentIndex())
        return True

    def _pullData(self):
        return self.getSubject().getCutPlanes()[0].getColormap().getName()


class PlaneAutoScaleItem(ColormapBase):
    """
    colormap autoscale item.
    Item is checkable.
    """

    def _init(self):
        colorMap = self.getSubject().getCutPlanes()[0].getColormap()
        self.setCheckable(True)
        self.setCheckState((colorMap.isAutoscale() and qt.Qt.Checked)
                           or qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        self._setAutoScale(checked)

    def _setAutoScale(self, auto):
        view3d = self.getSubject()
        colormap = view3d.getCutPlanes()[0].getColormap()

        if auto != colormap.isAutoscale():
            if auto:
                vMin = vMax = None
            else:
                dataRange = view3d.getDataRange()
                if dataRange is None:
                    vMin = vMax = None
                else:
                    vMin, vMax = dataRange[0], dataRange[-1]
            colormap.setVRange(vMin, vMax)

    def _pullData(self):
        auto = self.getSubject().getCutPlanes()[0].getColormap().isAutoscale()
        self._setAutoScale(auto)
        if auto:
            data = 'Auto'
        else:
            data = 'User'
        return data


class NormalizationNode(ColormapBase):
    """
    colormap normalization item.
    Item is a QComboBox.
    """
    editable = True
    listValues = list(Colormap.NORMALIZATIONS)

    def getEditor(self, parent, option, index):
        editor = qt.QComboBox(parent)
        editor.addItems(self.listValues)

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.currentIndexChanged[int].connect(
            lambda index: self.__editorChanged(index))

        return editor

    def __editorChanged(self, index):
        colorMap = self.getSubject().getCutPlanes()[0].getColormap()
        normalization = self.listValues[index]
        self.getSubject().getCutPlanes()[0].setColormap(name=colorMap.getName(),
                                                        norm=normalization,
                                                        vmin=colorMap.getVMin(),
                                                        vmax=colorMap.getVMax())

    def setEditorData(self, editor):
        normalization = self.getSubject().getCutPlanes()[0].getColormap().getNormalization()
        index = self.listValues.index(normalization)
        editor.setCurrentIndex(index)
        return True

    def _setModelData(self, editor):
        self.__editorChanged(editor.currentIndex())
        return True

    def _pullData(self):
        return self.getSubject().getCutPlanes()[0].getColormap().getNormalization()


class PlaneGroup(params.SubjectItem):
    """
    Root Item for the plane items.
    """
    def _init(self):
        valueItem = qt.QStandardItem()
        valueItem.setEditable(False)
        nameItem = PlaneVisibleItem(self.getSubject(), 'Visible')
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Colormap')
        nameItem.setEditable(False)
        valueItem = PlaneColormapItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Normalization')
        nameItem.setEditable(False)
        valueItem = NormalizationNode(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Orientation')
        nameItem.setEditable(False)
        valueItem = PlaneOrientationItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Interpolation')
        nameItem.setEditable(False)
        valueItem = PlaneInterpolationItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Autoscale')
        nameItem.setEditable(False)
        valueItem = PlaneAutoScaleItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Min')
        nameItem.setEditable(False)
        valueItem = PlaneMinRangeItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Max')
        nameItem.setEditable(False)
        valueItem = PlaneMaxRangeItem(self.getSubject())
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Values<=Min')
        nameItem.setEditable(False)
        valueItem = PlaneDisplayBelowMinItem(self.getSubject())
        self.appendRow([nameItem, valueItem])


class PlaneVisibleItem(params.SubjectItem):
    """
    Plane visibility item.
    Item is checkable.
    """
    def _init(self):
        plane = self.getSubject().getCutPlanes()[0]
        self.setCheckable(True)
        self.setCheckState((plane.isVisible() and qt.Qt.Checked)
                           or qt.Qt.Unchecked)

    def leftClicked(self):
        plane = self.getSubject().getCutPlanes()[0]
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != plane.isVisible():
            plane.setVisible(checked)
            if plane.isVisible():
                plane.moveToCenter()


# Tree ########################################################################

class TreeView(params.TreeView):
    """
    TreeView displaying the SubjectItems for the ScalarFieldView.
    """

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        delegate = self.itemDelegate()
        delegate.sigDelegateEvent.connect(self.__delegateEvent)

    def setSfView(self, sfView):
        """
        Sets the ScalarFieldView this view is controlling.

        :param sfView: A `ScalarFieldView`
        """
        model = params.TreeViewModelBase()

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([ViewSettingsItem(sfView, 'Style'), item])

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([DataSetItem(sfView, 'Data'), item])

        item = IsoSurfaceCount(sfView)
        item.setEditable(False)
        model.appendRow([IsoSurfaceGroup(sfView, 'Isosurfaces'), item])

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([PlaneGroup(sfView, 'Cutting Plane'), item])

        self.setModel(model)

    def __removeIsosurfaces(self):
        model = self.model()
        selected = self.selectedIndexes()
        items = []
        # WARNING : the selection mode is set to single, so we re not
        # supposed to have more than one item here.
        # Multiple selection deletion has not been tested.
        # Watch out for index invalidation
        for index in selected:
            leftIndex = model.sibling(index.row(), 0, index)
            leftItem = model.itemFromIndex(leftIndex)
            if isinstance(leftItem, params.SubjectItem) and leftItem not in items:
                items.append(leftItem)

        isos = [item for item in items if isinstance(item, IsoSurfaceRootItem)]
        if isos:
            for iso in isos:
                if iso.queryRemove(self):
                    parentItem = iso.parent()
                    parentItem.removeRow(iso.row())
        else:
            qt.QMessageBox.information(
                self,
                'Remove isosurface',
                'Select an iso-surface to remove it')

    def __delegateEvent(self, task):
        if task == 'remove_iso':
            self.__removeIsosurfaces()
