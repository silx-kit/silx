# /*##########################################################################
#
# Copyright (c) 2015-2021 European Synchrotron Radiation Facility
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

__authors__ = ["D. N."]
__license__ = "MIT"
__date__ = "24/04/2018"

import logging
import sys
import weakref

import numpy

from silx.gui import qt
from silx.gui.icons import getQIcon
from silx.gui.colors import Colormap
from silx.gui.widgets.FloatEdit import FloatEdit

from .ScalarFieldView import Isosurface


_logger = logging.getLogger(__name__)


class ModelColumns(object):
    NameColumn, ValueColumn, ColumnMax = range(3)
    ColumnNames = ['Name', 'Value']


class SubjectItem(qt.QStandardItem):
    """
    Base class for observers items.

    Subclassing:
    ------------
    The following method can/should be reimplemented:
    - _init
    - _pullData
    - _pushData
    - _setModelData
    - _subjectChanged
    - getEditor
    - getSignals
    - leftClicked
    - queryRemove
    - setEditorData

    Also the following attributes are available:
    - editable
    - persistent

    :param subject: object that this item will be observing.
    """

    editable = False
    """ boolean: set to True to make the item editable. """

    persistent = False
    """
    boolean: set to True to make the editor persistent.
        See : Qt.QAbstractItemView.openPersistentEditor
    """

    def __init__(self, subject, *args):

        super(SubjectItem, self).__init__(*args)

        self.setEditable(self.editable)

        self.__subject = None
        self.subject = subject

    def setData(self, value, role=qt.Qt.UserRole, pushData=True):
        """
        Overloaded method from QStandardItem. The pushData keyword tells
        the item to push data to the subject if the role is equal to EditRole.
        This is useful to let this method know if the setData method was called
        internally or from the view.

        :param value: the value ti set to data
        :param role: role in the item
        :param pushData: if True push value in the existing data.
        """
        if role == qt.Qt.EditRole and pushData:
            setValue = self._pushData(value, role)
            if setValue != value:
                value = setValue
        super(SubjectItem, self).setData(value, role)

    @property
    def subject(self):
        """The subject this item is observing"""
        return None if self.__subject is None else self.__subject()

    @subject.setter
    def subject(self, subject):
        if self.__subject is not None:
            raise ValueError('Subject already set '
                             ' (subject change not supported).')
        if subject is None:
            self.__subject = None
        else:
            self.__subject = weakref.ref(subject)
        if subject is not None:
            self._init()
            self._connectSignals()

    def _connectSignals(self):
        """
        Connects the signals. Called when the subject is set.
        """

        def gen_slot(_sigIdx):
            def slotfn(*args, **kwargs):
                self._subjectChanged(signalIdx=_sigIdx,
                                     args=args,
                                     kwargs=kwargs)
            return slotfn

        if self.__subject is not None:
            self.__slots = slots = []

            signals = self.getSignals()

            if signals:
                if not isinstance(signals, (list, tuple)):
                    signals = [signals]
                for sigIdx, signal in enumerate(signals):
                    slot = gen_slot(sigIdx)
                    signal.connect(slot)
                    slots.append((signal, slot))

    def _disconnectSignals(self):
        """
        Disconnects all subject's signal
        """
        if self.__slots:
            for signal, slot in self.__slots:
                try:
                    signal.disconnect(slot)
                except TypeError:
                    pass

    def _enableRow(self, enable):
        """
        Set the enabled state for this cell, or for the whole row
        if this item has a parent.

        :param bool enable: True if we wan't to enable the cell
        """
        parent = self.parent()
        model = self.model()
        if model is None or parent is None:
            # no parent -> no siblings
            self.setEnabled(enable)
            return

        for col in range(model.columnCount()):
            sibling = parent.child(self.row(), col)
            sibling.setEnabled(enable)

    #################################################################
    # Overloadable methods
    #################################################################

    def getSignals(self):
        """
        Returns the list of this items subject's signals that
        this item will be listening to.

        :return: list.
        """
        return None

    def _subjectChanged(self, signalIdx=None, args=None, kwargs=None):
        """
        Called when one of the signals is triggered. Default implementation
        just calls _pullData, compares the result to the current value stored
        as Qt.EditRole, and stores the new value if it is different. It also
        stores its str representation as Qt.DisplayRole

        :param signalIdx: index of the triggered signal. The value passed
            is the same as the signal position in the list returned by
            SubjectItem.getSignals.
        :param args: arguments received from the signal
        :param kwargs: keyword arguments received from the signal
        """
        data = self._pullData()
        if data == self.data(qt.Qt.EditRole):
            return
        self.setData(data, role=qt.Qt.DisplayRole, pushData=False)
        self.setData(data, role=qt.Qt.EditRole, pushData=False)

    def _pullData(self):
        """
        Pulls data from the subject.

        :return: subject data
        """
        return None

    def _pushData(self, value, role=qt.Qt.UserRole):
        """
        Pushes data to the subject and returns the actual value that was stored

        :return: the value that was stored
        """
        return value

    def _init(self):
        """
        Called when the subject is set.
        :return:
        """
        self._subjectChanged()

    def getEditor(self, parent, option, index):
        """
        Returns the editor widget used to edit this item's data. The arguments
        are the one passed to the QStyledItemDelegate.createEditor method.

        :param parent: the Qt parent of the editor
        :param option:
        :param index:
        :return:
        """
        return None

    def setEditorData(self, editor):
        """
        This is called by the View's delegate just before the editor is shown,
        its purpose it to setup the editors contents. Return False to use
        the delegate's default behaviour.

        :param editor:
        :return:
        """
        return True

    def _setModelData(self, editor):
        """
        This is called by the View's delegate just before the editor is closed,
        its allows this item to update itself with data from the editor.

        :param editor:
        :return:
        """
        return False

    def queryRemove(self, view=None):
        """
        This is called by the view to ask this items if it (the view) can
        remove it. Return True to let the view know that the item can be
        removed.

        :param view:
        :return:
        """
        return False

    def leftClicked(self):
        """
        This method is called by the view when the item's cell if left clicked.

        :return:
        """
        pass


# View settings ###############################################################

class ColorItem(SubjectItem):
    """color item."""
    editable = True
    persistent = True

    def getEditor(self, parent, option, index):
        editor = QColorEditor(parent)
        editor.color = self.getColor()

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.sigColorChanged.connect(
            lambda color: self._editorSlot(color))
        return editor

    def _editorSlot(self, color):
        self.setData(color, qt.Qt.EditRole)

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.setColor(value)
        return self.getColor()

    def _pullData(self):
        self.getColor()

    def setColor(self, color):
        """Override to implement actual color setter"""
        pass


class BackgroundColorItem(ColorItem):
    itemName = 'Background'

    def setColor(self, color):
        self.subject.setBackgroundColor(color)

    def getColor(self):
        return self.subject.getBackgroundColor()


class ForegroundColorItem(ColorItem):
    itemName = 'Foreground'

    def setColor(self, color):
        self.subject.setForegroundColor(color)

    def getColor(self):
        return self.subject.getForegroundColor()


class HighlightColorItem(ColorItem):
    itemName = 'Highlight'

    def setColor(self, color):
        self.subject.setHighlightColor(color)

    def getColor(self):
        return self.subject.getHighlightColor()


class _LightDirectionAngleBaseItem(SubjectItem):
    """Base class for directional light angle item."""
    editable = True
    persistent = True

    def _init(self):
        pass

    def getSignals(self):
        """Override to provide signals to listen"""
        raise NotImplementedError("MUST be implemented in subclass")

    def _pullData(self):
        """Override in subclass to get current angle"""
        raise NotImplementedError("MUST be implemented in subclass")

    def _pushData(self, value, role=qt.Qt.UserRole):
        """Override in subclass to set the angle"""
        raise NotImplementedError("MUST be implemented in subclass")

    def getEditor(self, parent, option, index):
        editor = qt.QSlider(parent)
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(-90)
        editor.setMaximum(90)
        editor.setValue(int(self._pullData()))

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.valueChanged.connect(
            lambda value: self._pushData(value))

        return editor

    def setEditorData(self, editor):
        editor.setValue(int(self._pullData()))
        return True

    def _setModelData(self, editor):
        value = editor.value()
        self._pushData(value)
        return True


class LightAzimuthAngleItem(_LightDirectionAngleBaseItem):
    """Light direction azimuth angle item."""

    def getSignals(self):
        return self.subject.sigAzimuthAngleChanged

    def _pullData(self):
         return self.subject.getAzimuthAngle()

    def _pushData(self, value, role=qt.Qt.UserRole):
         self.subject.setAzimuthAngle(value)


class LightAltitudeAngleItem(_LightDirectionAngleBaseItem):
    """Light direction altitude angle item."""

    def getSignals(self):
        return self.subject.sigAltitudeAngleChanged

    def _pullData(self):
         return self.subject.getAltitudeAngle()

    def _pushData(self, value, role=qt.Qt.UserRole):
         self.subject.setAltitudeAngle(value)


class _DirectionalLightProxy(qt.QObject):
    """Proxy to handle directional light with angles rather than vector.
    """

    sigAzimuthAngleChanged = qt.Signal()
    """Signal sent when the azimuth angle has changed."""

    sigAltitudeAngleChanged = qt.Signal()
    """Signal sent when altitude angle has changed."""

    def __init__(self, light):
        super(_DirectionalLightProxy, self).__init__()
        self._light = light
        light.addListener(self._directionUpdated)
        self._azimuth = 0.
        self._altitude = 0.

    def getAzimuthAngle(self):
        """Returns the signed angle in the horizontal plane.

         Unit: degrees.
        The 0 angle corresponds to the axis perpendicular to the screen.

        :rtype: float
        """
        return self._azimuth

    def getAltitudeAngle(self):
        """Returns the signed vertical angle from the horizontal plane.

        Unit: degrees.
        Range: [-90, +90]

        :rtype: float
        """
        return self._altitude

    def setAzimuthAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zx plane in degrees.
        """
        if angle != self._azimuth:
            self._azimuth = angle
            self._updateLight()
            self.sigAzimuthAngleChanged.emit()

    def setAltitudeAngle(self, angle):
        """Set the horizontal angle.

        :param float angle: Angle from -z axis in zy plane in degrees.
        """
        if angle != self._altitude:
            self._altitude = angle
            self._updateLight()
            self.sigAltitudeAngleChanged.emit()

    def _directionUpdated(self, *args, **kwargs):
        """Handle light direction update in the scene"""
        # Invert direction to manipulate the 'source' pointing to
        # the center of the viewport
        x, y, z = - self._light.direction

        # Horizontal plane is plane xz
        azimuth = numpy.degrees(numpy.arctan2(x, z))
        altitude = numpy.degrees(numpy.pi/2. - numpy.arccos(y))

        if (abs(azimuth - self.getAzimuthAngle()) > 0.01 and
                abs(abs(altitude) - 90.) >= 0.001):  # Do not update when at zenith
            self.setAzimuthAngle(azimuth)

        if abs(altitude - self.getAltitudeAngle()) > 0.01:
            self.setAltitudeAngle(altitude)

    def _updateLight(self):
        """Update light direction in the scene"""
        azimuth = numpy.radians(self._azimuth)
        delta = numpy.pi/2. - numpy.radians(self._altitude)
        z = - numpy.sin(delta) * numpy.cos(azimuth)
        x = - numpy.sin(delta) * numpy.sin(azimuth)
        y = - numpy.cos(delta)
        self._light.direction = x, y, z


class DirectionalLightGroup(SubjectItem):
    """
    Root Item for the directional light
    """

    def __init__(self,subject, *args):
        self._light = _DirectionalLightProxy(
            subject.getPlot3DWidget().viewport.light)

        super(DirectionalLightGroup, self).__init__(subject, *args)

    def _init(self):

        nameItem = qt.QStandardItem('Azimuth')
        nameItem.setEditable(False)
        valueItem = LightAzimuthAngleItem(self._light)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Altitude')
        nameItem.setEditable(False)
        valueItem = LightAltitudeAngleItem(self._light)
        self.appendRow([nameItem, valueItem])


class BoundingBoxItem(SubjectItem):
    """Bounding box, axes labels and grid visibility item.

    Item is checkable.
    """
    itemName = 'Bounding Box'

    def _init(self):
        visible = self.subject.isBoundingBoxVisible()
        self.setCheckable(True)
        self.setCheckState(qt.Qt.Checked if visible else qt.Qt.Unchecked)

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != self.subject.isBoundingBoxVisible():
            self.subject.setBoundingBoxVisible(checked)


class OrientationIndicatorItem(SubjectItem):
    """Orientation indicator visibility item.

    Item is checkable.
    """
    itemName = 'Axes indicator'

    def _init(self):
        plot3d = self.subject.getPlot3DWidget()
        visible = plot3d.isOrientationIndicatorVisible()
        self.setCheckable(True)
        self.setCheckState(qt.Qt.Checked if visible else qt.Qt.Unchecked)

    def leftClicked(self):
        plot3d = self.subject.getPlot3DWidget()
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != plot3d.isOrientationIndicatorVisible():
            plot3d.setOrientationIndicatorVisible(checked)


class ViewSettingsItem(qt.QStandardItem):
    """Viewport settings"""

    def __init__(self, subject, *args):

        super(ViewSettingsItem, self).__init__(*args)

        self.setEditable(False)

        classes = (BackgroundColorItem,
                   ForegroundColorItem,
                   HighlightColorItem,
                   BoundingBoxItem,
                   OrientationIndicatorItem)
        for cls in classes:
            titleItem = qt.QStandardItem(cls.itemName)
            titleItem.setEditable(False)
            self.appendRow([titleItem, cls(subject)])

        nameItem = DirectionalLightGroup(subject, 'Light Direction')
        valueItem = qt.QStandardItem()
        self.appendRow([nameItem, valueItem])


# Data information ############################################################

class DataChangedItem(SubjectItem):
    """
    Base class for items listening to ScalarFieldView.sigDataChanged
    """

    def getSignals(self):
        subject = self.subject
        if subject:
            return subject.sigDataChanged, subject.sigTransformChanged
        return None

    def _init(self):
        self._subjectChanged()


class DataTypeItem(DataChangedItem):
    itemName = 'dtype'

    def _pullData(self):
        data = self.subject.getData(copy=False)
        return ((data is not None) and str(data.dtype)) or 'N/A'


class DataShapeItem(DataChangedItem):
    itemName = 'size'

    def _pullData(self):
        data = self.subject.getData(copy=False)
        if data is None:
            return 'N/A'
        else:
            return str(list(reversed(data.shape)))


class OffsetItem(DataChangedItem):
    itemName = 'offset'

    def _pullData(self):
        offset = self.subject.getTranslation()
        return ((offset is not None) and str(offset)) or 'N/A'


class ScaleItem(DataChangedItem):
    itemName = 'scale'

    def _pullData(self):
        scale = self.subject.getScale()
        return ((scale is not None) and str(scale)) or 'N/A'


class MatrixItem(DataChangedItem):

    def __init__(self, subject, row, *args):
        self.__row = row
        super(MatrixItem, self).__init__(subject, *args)

    def _pullData(self):
        matrix = self.subject.getTransformMatrix()
        return str(matrix[self.__row])


class DataSetItem(qt.QStandardItem):

    def __init__(self, subject, *args):

        super(DataSetItem, self).__init__(*args)

        self.setEditable(False)

        klasses = [DataTypeItem, DataShapeItem, OffsetItem]
        for klass in klasses:
            titleItem = qt.QStandardItem(klass.itemName)
            titleItem.setEditable(False)
            self.appendRow([titleItem, klass(subject)])

        matrixItem = qt.QStandardItem('matrix')
        matrixItem.setEditable(False)
        valueItem = qt.QStandardItem()
        self.appendRow([matrixItem, valueItem])

        for row in range(3):
            titleItem = qt.QStandardItem()
            titleItem.setEditable(False)
            valueItem = MatrixItem(subject, row)
            matrixItem.appendRow([titleItem, valueItem])

        titleItem = qt.QStandardItem(ScaleItem.itemName)
        titleItem.setEditable(False)
        self.appendRow([titleItem, ScaleItem(subject)])


# Isosurface ##################################################################

class IsoSurfaceRootItem(SubjectItem):
    """
    Root (i.e : column index 0) Isosurface item.
    """

    def __init__(self, subject, normalization, *args):
        self._isoLevelSliderNormalization = normalization
        super(IsoSurfaceRootItem, self).__init__(subject, *args)

    def getSignals(self):
        subject = self.subject
        return [subject.sigColorChanged,
                subject.sigVisibilityChanged]

    def _subjectChanged(self, signalIdx=None, args=None, kwargs=None):
        if signalIdx == 0:
            color = self.subject.getColor()
            self.setData(color, qt.Qt.DecorationRole)
        elif signalIdx == 1:
            visible = args[0]
            self.setCheckState((visible and qt.Qt.Checked) or qt.Qt.Unchecked)

    def _init(self):
        self.setCheckable(True)

        isosurface = self.subject
        color = isosurface.getColor()
        visible = isosurface.isVisible()
        self.setData(color, qt.Qt.DecorationRole)
        self.setCheckState((visible and qt.Qt.Checked) or qt.Qt.Unchecked)

        nameItem = qt.QStandardItem('Level')
        sliderItem = IsoSurfaceLevelSlider(self.subject,
                                           self._isoLevelSliderNormalization)
        self.appendRow([nameItem, sliderItem])

        nameItem = qt.QStandardItem('Color')
        nameItem.setEditable(False)
        valueItem = IsoSurfaceColorItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Opacity')
        nameItem.setTextAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
        nameItem.setEditable(False)
        valueItem = IsoSurfaceAlphaItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem()
        nameItem.setEditable(False)
        valueItem = IsoSurfaceAlphaLegendItem(self.subject)
        valueItem.setEditable(False)
        self.appendRow([nameItem, valueItem])

    def queryRemove(self, view=None):
        buttons = qt.QMessageBox.Ok | qt.QMessageBox.Cancel
        ans = qt.QMessageBox.question(view,
                                      'Remove isosurface',
                                      'Remove the selected iso-surface?',
                                      buttons=buttons)
        if ans == qt.QMessageBox.Ok:
            sfview = self.subject.parent()
            if sfview:
                sfview.removeIsosurface(self.subject)
                return False
        return False

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        visible = self.subject.isVisible()
        if checked != visible:
            self.subject.setVisible(checked)


class IsoSurfaceLevelItem(SubjectItem):
    """
    Base class for the isosurface level items.
    """
    editable = True

    def getSignals(self):
        subject = self.subject
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
        return self.subject.getLevel()

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.subject.setLevel(value)
        return self.subject.getLevel()


class _IsoLevelSlider(qt.QSlider):
    """QSlider used for iso-surface level with linear scale"""

    def __init__(self, parent, subject, normalization):
        super(_IsoLevelSlider, self).__init__(parent=parent)
        self.subject = subject

        if normalization == 'arcsinh':
            self.__norm = numpy.arcsinh
            self.__invNorm = numpy.sinh
        elif normalization == 'linear':
            self.__norm = lambda x: x
            self.__invNorm = lambda x: x
        else:
            raise ValueError(
                "Unsupported normalization %s", normalization)

        self.sliderReleased.connect(self.__sliderReleased)

        self.subject.sigLevelChanged.connect(self.setLevel)
        self.subject.parent().sigDataChanged.connect(self.__dataChanged)

    def setLevel(self, level):
        """Set slider from iso-surface level"""
        dataRange = self.subject.parent().getDataRange()

        if dataRange is not None:
            min_ = self.__norm(dataRange[0])
            max_ = self.__norm(dataRange[-1])

            width = max_ - min_
            if width > 0:
                sliderWidth = self.maximum() - self.minimum()
                sliderPosition = sliderWidth * (self.__norm(level) - min_) / width
                self.setValue(int(sliderPosition))

    def __dataChanged(self):
        """Handles data update to refresh slider range if needed"""
        self.setLevel(self.subject.getLevel())

    def __sliderReleased(self):
        value = self.value()
        dataRange = self.subject.parent().getDataRange()
        if dataRange is not None:
            min_ = self.__norm(dataRange[0])
            max_ = self.__norm(dataRange[-1])
            width = max_ - min_
            sliderWidth = self.maximum() - self.minimum()
            level = min_ + width * value / sliderWidth
            self.subject.setLevel(self.__invNorm(level))


class IsoSurfaceLevelSlider(IsoSurfaceLevelItem):
    """
    Isosurface level item with a slider editor.
    """
    nTicks = 1000
    persistent = True

    def __init__(self, subject, normalization):
        self.normalization = normalization
        super(IsoSurfaceLevelSlider, self).__init__(subject)

    def getEditor(self, parent, option, index):
        editor = _IsoLevelSlider(parent, self.subject, self.normalization)
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(0)
        editor.setMaximum(self.nTicks)

        editor.setSingleStep(1)

        editor.setLevel(self.subject.getLevel())
        return editor

    def setEditorData(self, editor):
        return True

    def _setModelData(self, editor):
        return True


class IsoSurfaceColorItem(SubjectItem):
    """
    Isosurface color item.
    """
    editable = True
    persistent = True

    def getSignals(self):
        return self.subject.sigColorChanged

    def getEditor(self, parent, option, index):
        editor = QColorEditor(parent)
        color = self.subject.getColor()
        color.setAlpha(255)
        editor.color = color
        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.sigColorChanged.connect(
            lambda color: self.__editorChanged(color))
        return editor

    def __editorChanged(self, color):
        color.setAlpha(self.subject.getColor().alpha())
        self.subject.setColor(color)

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.subject.setColor(value)
        return self.subject.getColor()


class QColorEditor(qt.QWidget):
    """
    QColor editor.
    """
    sigColorChanged = qt.Signal(object)

    color = property(lambda self: qt.QColor(self.__color))

    @color.setter
    def color(self, color):
        self._setColor(color)
        self.__previousColor = color

    def __init__(self, *args, **kwargs):
        super(QColorEditor, self).__init__(*args, **kwargs)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        button = qt.QToolButton()
        icon = qt.QIcon(qt.QPixmap(32, 32))
        button.setIcon(icon)
        layout.addWidget(button)
        button.clicked.connect(self.__showColorDialog)
        layout.addStretch(1)

        self.__color = None
        self.__previousColor = None

    def sizeHint(self):
        return qt.QSize(0, 0)

    def _setColor(self, qColor):
        button = self.findChild(qt.QToolButton)
        pixmap = qt.QPixmap(32, 32)
        pixmap.fill(qColor)
        button.setIcon(qt.QIcon(pixmap))
        self.__color = qColor

    def __showColorDialog(self):
        dialog = qt.QColorDialog(parent=self)
        if sys.platform == 'darwin':
            # Use of native color dialog on macos might cause problems
            dialog.setOption(qt.QColorDialog.DontUseNativeDialog, True)

        self.__previousColor = self.__color
        dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
        dialog.setModal(True)
        dialog.currentColorChanged.connect(self.__colorChanged)
        dialog.finished.connect(self.__dialogClosed)
        dialog.show()

    def __colorChanged(self, color):
        self.__color = color
        self._setColor(color)
        self.sigColorChanged.emit(color)

    def __dialogClosed(self, result):
        if result == qt.QDialog.Rejected:
            self.__colorChanged(self.__previousColor)
        self.__previousColor = None


class IsoSurfaceAlphaItem(SubjectItem):
    """
    Isosurface alpha item.
    """
    editable = True
    persistent = True

    def _init(self):
        pass

    def getSignals(self):
        return self.subject.sigColorChanged

    def getEditor(self, parent, option, index):
        editor = qt.QSlider(parent)
        editor.setOrientation(qt.Qt.Horizontal)
        editor.setMinimum(0)
        editor.setMaximum(255)

        color = self.subject.getColor()
        editor.setValue(color.alpha())

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.valueChanged.connect(
            lambda value: self.__editorChanged(value))

        return editor

    def __editorChanged(self, value):
        color = self.subject.getColor()
        color.setAlpha(value)
        self.subject.setColor(color)

    def setEditorData(self, editor):
        return True

    def _setModelData(self, editor):
        return True


class IsoSurfaceAlphaLegendItem(SubjectItem):
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


class IsoSurfaceCount(SubjectItem):
    """
    Item displaying the number of isosurfaces.
    """

    def getSignals(self):
        subject = self.subject
        return [subject.sigIsosurfaceAdded, subject.sigIsosurfaceRemoved]

    def _pullData(self):
        return len(self.subject.getIsosurfaces())


class IsoSurfaceAddRemoveWidget(qt.QWidget):

    sigViewTask = qt.Signal(str)
    """Signal for the tree view to perform some task"""

    def __init__(self, parent, item):
        super(IsoSurfaceAddRemoveWidget, self).__init__(parent)
        self._item = item
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        addBtn = qt.QToolButton(self)
        addBtn.setText('+')
        addBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(addBtn)
        addBtn.clicked.connect(self.__addClicked)

        removeBtn = qt.QToolButton(self)
        removeBtn.setText('-')
        removeBtn.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
        layout.addWidget(removeBtn)
        removeBtn.clicked.connect(self.__removeClicked)

        layout.addStretch(1)

    def __addClicked(self):
        sfview = self._item.subject
        if not sfview:
            return
        dataRange = sfview.getDataRange()
        if dataRange is None:
            dataRange = [0, 1]

        sfview.addIsosurface(
            numpy.mean((dataRange[0], dataRange[-1])), '#0000FF')

    def __removeClicked(self):
        self.sigViewTask.emit('remove_iso')


class IsoSurfaceAddRemoveItem(SubjectItem):
    """
    Item displaying a simple QToolButton allowing to add an isosurface.
    """
    persistent = True

    def getEditor(self, parent, option, index):
        return IsoSurfaceAddRemoveWidget(parent, self)


class IsoSurfaceGroup(SubjectItem):
    """
    Root item for the list of isosurface items.
    """

    def __init__(self, subject, normalization, *args):
        self._isoLevelSliderNormalization = normalization
        super(IsoSurfaceGroup, self).__init__(subject, *args)

    def getSignals(self):
        subject = self.subject
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
        valueItem = IsoSurfaceRootItem(
            subject=isosurface,
            normalization=self._isoLevelSliderNormalization)
        nameItem = IsoSurfaceLevelItem(subject=isosurface)
        self.insertRow(max(0, self.rowCount() - 1), [valueItem, nameItem])

    def __removeIsosurface(self, isosurface):
        for row in range(self.rowCount()):
            child = self.child(row)
            subject = getattr(child, 'subject', None)
            if subject == isosurface:
                self.takeRow(row)
                break

    def _init(self):
        nameItem = IsoSurfaceAddRemoveItem(self.subject)
        valueItem = qt.QStandardItem()
        valueItem.setEditable(False)
        self.appendRow([nameItem, valueItem])

        subject = self.subject
        isosurfaces = subject.getIsosurfaces()
        for isosurface in isosurfaces:
            self.__addIsosurface(isosurface)


# Cutting Plane ###############################################################

class ColormapBase(SubjectItem):
    """
    Mixin class for colormap items.
    """

    def getSignals(self):
        return [self.subject.getCutPlanes()[0].sigColormapChanged]


class PlaneMinRangeItem(ColormapBase):
    """
    colormap minVal item.
    Editor is a QLineEdit with a QDoubleValidator
    """
    editable = True

    def _pullData(self):
        colormap = self.subject.getCutPlanes()[0].getColormap()
        auto = colormap.isAutoscale()
        if auto == self.isEnabled():
            self._enableRow(not auto)
        return colormap.getVMin()

    def _pushData(self, value, role=qt.Qt.UserRole):
        self._setVMin(value)

    def _setVMin(self, value):
        colormap = self.subject.getCutPlanes()[0].getColormap()
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
        colormap = self.subject.getCutPlanes()[0].getColormap()
        auto = colormap.isAutoscale()
        if auto == self.isEnabled():
            self._enableRow(not auto)
        return self.subject.getCutPlanes()[0].getColormap().getVMax()

    def _setVMax(self, value):
        colormap = self.subject.getCutPlanes()[0].getColormap()
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


class PlaneOrientationItem(SubjectItem):
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
        return [self.subject.getCutPlanes()[0].sigPlaneChanged]

    def _pullData(self):
        currentNormal = self.subject.getCutPlanes()[0].getNormal(
            coordinates='scene')
        for _, text, _, normal in self._PLANE_ACTIONS:
            if numpy.allclose(normal, currentNormal):
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
        plane = self.subject.getCutPlanes()[0]
        plane.setNormal(normal, coordinates='scene')
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


class PlaneInterpolationItem(SubjectItem):
    """Toggle cut plane interpolation method: nearest or linear.

    Item is checkable
    """

    def _init(self):
        interpolation = self.subject.getCutPlanes()[0].getInterpolation()
        self.setCheckable(True)
        self.setCheckState(
            qt.Qt.Checked if interpolation == 'linear' else qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def getSignals(self):
        return [self.subject.getCutPlanes()[0].sigInterpolationChanged]

    def leftClicked(self):
        checked = self.checkState() == qt.Qt.Checked
        self._setInterpolation('linear' if checked else 'nearest')

    def _pullData(self):
        interpolation = self.subject.getCutPlanes()[0].getInterpolation()
        self._setInterpolation(interpolation)
        return interpolation[0].upper() + interpolation[1:]

    def _setInterpolation(self, interpolation):
        self.subject.getCutPlanes()[0].setInterpolation(interpolation)


class PlaneDisplayBelowMinItem(SubjectItem):
    """Toggle whether to display or not values <= colormap min of the cut plane

    Item is checkable
    """

    def _init(self):
        display = self.subject.getCutPlanes()[0].getDisplayValuesBelowMin()
        self.setCheckable(True)
        self.setCheckState(
            qt.Qt.Checked if display else qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def getSignals(self):
        return [self.subject.getCutPlanes()[0].sigTransparencyChanged]

    def leftClicked(self):
        checked = self.checkState() == qt.Qt.Checked
        self._setDisplayValuesBelowMin(checked)

    def _pullData(self):
        display = self.subject.getCutPlanes()[0].getDisplayValuesBelowMin()
        self._setDisplayValuesBelowMin(display)
        return "Displayed" if display else "Hidden"

    def _setDisplayValuesBelowMin(self, display):
        self.subject.getCutPlanes()[0].setDisplayValuesBelowMin(display)


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
        colormap = self.subject.getCutPlanes()[0].getColormap()
        colormap.setName(colormapName)

    def setEditorData(self, editor):
        colormapName = self.subject.getCutPlanes()[0].getColormap().getName()
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
        return self.subject.getCutPlanes()[0].getColormap().getName()


class PlaneAutoScaleItem(ColormapBase):
    """
    colormap autoscale item.
    Item is checkable.
    """

    def _init(self):
        colorMap = self.subject.getCutPlanes()[0].getColormap()
        self.setCheckable(True)
        self.setCheckState((colorMap.isAutoscale() and qt.Qt.Checked)
                           or qt.Qt.Unchecked)
        self.setData(self._pullData(), role=qt.Qt.DisplayRole, pushData=False)

    def leftClicked(self):
        checked = (self.checkState() == qt.Qt.Checked)
        self._setAutoScale(checked)

    def _setAutoScale(self, auto):
        view3d = self.subject
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
        auto = self.subject.getCutPlanes()[0].getColormap().isAutoscale()
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
        colorMap = self.subject.getCutPlanes()[0].getColormap()
        normalization = self.listValues[index]
        self.subject.getCutPlanes()[0].setColormap(name=colorMap.getName(),
                                                   norm=normalization,
                                                   vmin=colorMap.getVMin(),
                                                   vmax=colorMap.getVMax())

    def setEditorData(self, editor):
        normalization = self.subject.getCutPlanes()[0].getColormap().getNormalization()
        index = self.listValues.index(normalization)
        editor.setCurrentIndex(index)
        return True

    def _setModelData(self, editor):
        self.__editorChanged(editor.currentIndex())
        return True

    def _pullData(self):
        return self.subject.getCutPlanes()[0].getColormap().getNormalization()


class PlaneGroup(SubjectItem):
    """
    Root Item for the plane items.
    """
    def _init(self):
        valueItem = qt.QStandardItem()
        valueItem.setEditable(False)
        nameItem = PlaneVisibleItem(self.subject, 'Visible')
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Colormap')
        nameItem.setEditable(False)
        valueItem = PlaneColormapItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Normalization')
        nameItem.setEditable(False)
        valueItem = NormalizationNode(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Orientation')
        nameItem.setEditable(False)
        valueItem = PlaneOrientationItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Interpolation')
        nameItem.setEditable(False)
        valueItem = PlaneInterpolationItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Autoscale')
        nameItem.setEditable(False)
        valueItem = PlaneAutoScaleItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Min')
        nameItem.setEditable(False)
        valueItem = PlaneMinRangeItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Max')
        nameItem.setEditable(False)
        valueItem = PlaneMaxRangeItem(self.subject)
        self.appendRow([nameItem, valueItem])

        nameItem = qt.QStandardItem('Values<=Min')
        nameItem.setEditable(False)
        valueItem = PlaneDisplayBelowMinItem(self.subject)
        self.appendRow([nameItem, valueItem])


class PlaneVisibleItem(SubjectItem):
    """
    Plane visibility item.
    Item is checkable.
    """
    def _init(self):
        plane = self.subject.getCutPlanes()[0]
        self.setCheckable(True)
        self.setCheckState((plane.isVisible() and qt.Qt.Checked)
                           or qt.Qt.Unchecked)

    def leftClicked(self):
        plane = self.subject.getCutPlanes()[0]
        checked = (self.checkState() == qt.Qt.Checked)
        if checked != plane.isVisible():
            plane.setVisible(checked)
            if plane.isVisible():
                plane.moveToCenter()


# Tree ########################################################################

class ItemDelegate(qt.QStyledItemDelegate):
    """
    Delegate for the QTreeView filled with SubjectItems.
    """

    sigDelegateEvent = qt.Signal(str)

    def __init__(self, parent=None):
        super(ItemDelegate, self).__init__(parent)

    def createEditor(self, parent, option, index):
        item = index.model().itemFromIndex(index)
        if item:
            if isinstance(item, SubjectItem):
                editor = item.getEditor(parent, option, index)
                if editor:
                    editor.setAutoFillBackground(True)
                    if hasattr(editor, 'sigViewTask'):
                        editor.sigViewTask.connect(self.__viewTask)
                    return editor

        editor = super(ItemDelegate, self).createEditor(parent,
                                                        option,
                                                        index)
        return editor

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def setEditorData(self, editor, index):
        item = index.model().itemFromIndex(index)
        if item:
            if isinstance(item, SubjectItem) and item.setEditorData(editor):
                return
        super(ItemDelegate, self).setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        item = index.model().itemFromIndex(index)
        if isinstance(item, SubjectItem) and item._setModelData(editor):
            return
        super(ItemDelegate, self).setModelData(editor, model, index)

    def __viewTask(self, task):
        self.sigDelegateEvent.emit(task)


class TreeView(qt.QTreeView):
    """
    TreeView displaying the SubjectItems for the ScalarFieldView.
    """

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        self.__openedIndex = None
        self._isoLevelSliderNormalization = 'linear'

        self.setIconSize(qt.QSize(16, 16))

        header = self.header()
        header.setSectionResizeMode(qt.QHeaderView.ResizeToContents)

        delegate = ItemDelegate()
        self.setItemDelegate(delegate)
        delegate.sigDelegateEvent.connect(self.__delegateEvent)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.clicked.connect(self.__clicked)

    def setSfView(self, sfView):
        """
        Sets the ScalarFieldView this view is controlling.

        :param sfView: A `ScalarFieldView`
        """
        model = qt.QStandardItemModel()
        model.setColumnCount(ModelColumns.ColumnMax)
        model.setHorizontalHeaderLabels(['Name', 'Value'])

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([ViewSettingsItem(sfView, 'Style'), item])

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([DataSetItem(sfView, 'Data'), item])

        item = IsoSurfaceCount(sfView)
        item.setEditable(False)
        model.appendRow([IsoSurfaceGroup(sfView,
                                         self._isoLevelSliderNormalization,
                                         'Isosurfaces'),
                         item])

        item = qt.QStandardItem()
        item.setEditable(False)
        model.appendRow([PlaneGroup(sfView, 'Cutting Plane'), item])

        self.setModel(model)

    def setModel(self, model):
        """
        Reimplementation of the QTreeView.setModel method. It connects the
        rowsRemoved signal and opens the persistent editors.

        :param qt.QStandardItemModel model: the model
        """

        prevModel = self.model()
        if prevModel:
            self.__openPersistentEditors(qt.QModelIndex(), False)
            try:
                prevModel.rowsRemoved.disconnect(self.rowsRemoved)
            except TypeError:
                pass

        super(TreeView, self).setModel(model)
        model.rowsRemoved.connect(self.rowsRemoved)
        self.__openPersistentEditors(qt.QModelIndex())

    def __openPersistentEditors(self, parent=None, openEditor=True):
        """
        Opens or closes the items persistent editors.

        :param qt.QModelIndex parent: starting index, or None if the whole tree
            is to be considered.
        :param bool openEditor: True to open the editors, False to close them.
        """
        model = self.model()

        if not model:
            return

        if not parent or not parent.isValid():
            parent = self.model().invisibleRootItem().index()

        if openEditor:
            meth = self.openPersistentEditor
        else:
            meth = self.closePersistentEditor

        curParent = parent
        children = [model.index(row, 0, curParent)
                    for row in range(model.rowCount(curParent))]

        columnCount = model.columnCount()

        while len(children) > 0:
            curParent = children.pop(-1)

            children.extend([model.index(row, 0, curParent)
                             for row in range(model.rowCount(curParent))])

            for colIdx in range(columnCount):
                sibling = model.sibling(curParent.row(),
                                        colIdx,
                                        curParent)
                item = model.itemFromIndex(sibling)
                if isinstance(item, SubjectItem) and item.persistent:
                    meth(sibling)

    def rowsAboutToBeRemoved(self, parent, start, end):
        """
        Reimplementation of the QTreeView.rowsAboutToBeRemoved. Closes all
        persistent editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index (inclusive)
        :param int end: End index from parent index (inclusive)
        """
        self.__openPersistentEditors(parent, False)
        super(TreeView, self).rowsAboutToBeRemoved(parent, start, end)

    def rowsRemoved(self, parent, start, end):
        """
        Called when QTreeView.rowsRemoved is emitted. Opens all persistent
        editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index (inclusive)
        :param int end: End index from parent index (inclusive)
        """
        super(TreeView, self).rowsRemoved(parent, start, end)
        self.__openPersistentEditors(parent, True)

    def rowsInserted(self, parent, start, end):
        """
        Reimplementation of the QTreeView.rowsInserted. Opens all persistent
        editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index
        :param int end: End index from parent index
        """
        self.__openPersistentEditors(parent, False)
        super(TreeView, self).rowsInserted(parent, start, end)
        self.__openPersistentEditors(parent)

    def keyReleaseEvent(self, event):
        """
        Reimplementation of the QTreeView.keyReleaseEvent.
        At the moment only Key_Delete is handled. It calls the selected item's
        queryRemove method, and deleted the item if needed.

        :param qt.QKeyEvent event: A key event
        """

        # TODO : better filtering
        key = event.key()
        modifiers = event.modifiers()

        if key == qt.Qt.Key_Delete and modifiers == qt.Qt.NoModifier:
            self.__removeIsosurfaces()

        super(TreeView, self).keyReleaseEvent(event)

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
            if isinstance(leftItem, SubjectItem) and leftItem not in items:
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

    def __clicked(self, index):
        """
        Called when the QTreeView.clicked signal is emitted. Calls the item's
        leftClick method.

        :param qt.QIndex index: An index
        """
        item = self.model().itemFromIndex(index)
        if isinstance(item, SubjectItem):
            item.leftClicked()

    def __delegateEvent(self, task):
        if task == 'remove_iso':
            self.__removeIsosurfaces()

    def setIsoLevelSliderNormalization(self, normalization):
        """Set the normalization for iso level slider

        This MUST be called *before* :meth:`setSfView` to have an effect.

        :param str normalization: Either 'linear' or 'arcsinh'
        """
        assert normalization in ('linear', 'arcsinh')
        self._isoLevelSliderNormalization = normalization
