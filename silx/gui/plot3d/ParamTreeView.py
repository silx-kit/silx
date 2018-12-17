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
#
# ###########################################################################*/
"""
This module provides a :class:`QTreeView` dedicated to display plot3d models.

This module contains:
- :class:`ParamTreeView`: A QTreeView specific for plot3d parameters and scene.
- :class:`ParameterTreeDelegate`: The delegate for :class:`ParamTreeView`.
- A set of specific editors used by :class:`ParameterTreeDelegate`:
  :class:`FloatEditor`, :class:`Vector3DEditor`,
  :class:`Vector4DEditor`, :class:`IntSliderEditor`, :class:`BooleanEditor`
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/12/2017"


import numbers
import sys

import six

from .. import qt
from ..widgets.FloatEdit import FloatEdit as _FloatEdit
from ._model import visitQAbstractItemModel


class FloatEditor(_FloatEdit):
    """Editor widget for float.

    :param parent: The widget's parent
    :param float value: The initial editor value
    """

    valueChanged = qt.Signal(float)
    """Signal emitted when the float value has changed"""

    def __init__(self, parent=None, value=None):
        super(FloatEditor, self).__init__(parent, value)
        self.setAlignment(qt.Qt.AlignLeft)
        self.editingFinished.connect(self._emit)

    def _emit(self):
        self.valueChanged.emit(self.value)

    value = qt.Property(float,
                        fget=_FloatEdit.value,
                        fset=_FloatEdit.setValue,
                        user=True,
                        notify=valueChanged)
    """Qt user property of the float value this widget edits"""


class Vector3DEditor(qt.QWidget):
    """Editor widget for QVector3D.

    :param parent: The widget's parent
    :param flags: The widgets's flags
    """

    valueChanged = qt.Signal(qt.QVector3D)
    """Signal emitted when the QVector3D value has changed"""

    def __init__(self, parent=None, flags=qt.Qt.Widget):
        super(Vector3DEditor, self).__init__(parent, flags)
        layout = qt.QHBoxLayout(self)
        # layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self._xEdit = _FloatEdit(parent=self, value=0.)
        self._xEdit.setAlignment(qt.Qt.AlignLeft)
        # self._xEdit.editingFinished.connect(self._emit)
        self._yEdit = _FloatEdit(parent=self, value=0.)
        self._yEdit.setAlignment(qt.Qt.AlignLeft)
        # self._yEdit.editingFinished.connect(self._emit)
        self._zEdit = _FloatEdit(parent=self, value=0.)
        self._zEdit.setAlignment(qt.Qt.AlignLeft)
        # self._zEdit.editingFinished.connect(self._emit)
        layout.addWidget(qt.QLabel('x:'))
        layout.addWidget(self._xEdit)
        layout.addWidget(qt.QLabel('y:'))
        layout.addWidget(self._yEdit)
        layout.addWidget(qt.QLabel('z:'))
        layout.addWidget(self._zEdit)
        layout.addStretch(1)

    def _emit(self):
        vector = self.value
        self.valueChanged.emit(vector)

    def getValue(self):
        """Returns the QVector3D value of this widget

        :rtype: QVector3D
        """
        return qt.QVector3D(
            self._xEdit.value(), self._yEdit.value(), self._zEdit.value())

    def setValue(self, value):
        """Set the QVector3D value

        :param QVector3D value: The new value
        """
        self._xEdit.setValue(value.x())
        self._yEdit.setValue(value.y())
        self._zEdit.setValue(value.z())
        self.valueChanged.emit(value)

    value = qt.Property(qt.QVector3D,
                        fget=getValue,
                        fset=setValue,
                        user=True,
                        notify=valueChanged)
    """Qt user property of the QVector3D value this widget edits"""


class Vector4DEditor(qt.QWidget):
    """Editor widget for QVector4D.

    :param parent: The widget's parent
    :param flags: The widgets's flags
    """

    valueChanged = qt.Signal(qt.QVector4D)
    """Signal emitted when the QVector4D value has changed"""

    def __init__(self, parent=None, flags=qt.Qt.Widget):
        super(Vector4DEditor, self).__init__(parent, flags)
        layout = qt.QHBoxLayout(self)
        # layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self._xEdit = _FloatEdit(parent=self, value=0.)
        self._xEdit.setAlignment(qt.Qt.AlignLeft)
        # self._xEdit.editingFinished.connect(self._emit)
        self._yEdit = _FloatEdit(parent=self, value=0.)
        self._yEdit.setAlignment(qt.Qt.AlignLeft)
        # self._yEdit.editingFinished.connect(self._emit)
        self._zEdit = _FloatEdit(parent=self, value=0.)
        self._zEdit.setAlignment(qt.Qt.AlignLeft)
        # self._zEdit.editingFinished.connect(self._emit)
        self._wEdit = _FloatEdit(parent=self, value=0.)
        self._wEdit.setAlignment(qt.Qt.AlignLeft)
        # self._wEdit.editingFinished.connect(self._emit)
        layout.addWidget(qt.QLabel('x:'))
        layout.addWidget(self._xEdit)
        layout.addWidget(qt.QLabel('y:'))
        layout.addWidget(self._yEdit)
        layout.addWidget(qt.QLabel('z:'))
        layout.addWidget(self._zEdit)
        layout.addWidget(qt.QLabel('w:'))
        layout.addWidget(self._wEdit)
        layout.addStretch(1)

    def _emit(self):
        vector = self.value
        self.valueChanged.emit(vector)

    def getValue(self):
        """Returns the QVector4D value of this widget

        :rtype: QVector4D
        """
        return qt.QVector4D(self._xEdit.value(), self._yEdit.value(),
                            self._zEdit.value(), self._wEdit.value())

    def setValue(self, value):
        """Set the QVector4D value

        :param QVector4D value: The new value
        """
        self._xEdit.setValue(value.x())
        self._yEdit.setValue(value.y())
        self._zEdit.setValue(value.z())
        self._wEdit.setValue(value.w())
        self.valueChanged.emit(value)

    value = qt.Property(qt.QVector4D,
                        fget=getValue,
                        fset=setValue,
                        user=True,
                        notify=valueChanged)
    """Qt user property of the QVector4D value this widget edits"""


class IntSliderEditor(qt.QSlider):
    """Slider editor widget for integer.

    Note: Tracking is disabled.

    :param parent: The widget's parent
    """

    def __init__(self, parent=None):
        super(IntSliderEditor, self).__init__(parent)
        self.setOrientation(qt.Qt.Horizontal)
        self.setSingleStep(1)
        self.setRange(0, 255)
        self.setValue(0)


class BooleanEditor(qt.QCheckBox):
    """Checkbox editor for bool.

    This is a QCheckBox with white background.

    :param parent: The widget's parent
    """

    def __init__(self, parent=None):
        super(BooleanEditor, self).__init__(parent)
        self.setStyleSheet("background: white;")


class ParameterTreeDelegate(qt.QStyledItemDelegate):
    """TreeView delegate specific to plot3d scene and object parameter tree.

    It provides additional editors.

    :param parent: Delegate's parent
    """

    EDITORS = {
        bool: BooleanEditor,
        float: FloatEditor,
        qt.QVector3D: Vector3DEditor,
        qt.QVector4D: Vector4DEditor,
    }
    """Specific editors for different type of data"""

    def __init__(self, parent=None):
        super(ParameterTreeDelegate, self).__init__(parent)

    def _fixVariant(self, data):
        """Fix PyQt4 zero vectors being stored as QPyNullVariant.

        :param data: Data retrieved from the model
        :return: Corresponding object
        """
        if qt.BINDING == 'PyQt4' and isinstance(data, qt.QPyNullVariant):
            typeName = data.typeName()
            if typeName == 'QVector3D':
                data = qt.QVector3D()
            elif typeName == 'QVector4D':
                data = qt.QVector4D()
        return data

    def paint(self, painter, option, index):
        """See :meth:`QStyledItemDelegate.paint`"""
        data = index.data(qt.Qt.DisplayRole)
        data = self._fixVariant(data)

        if isinstance(data, (qt.QVector3D, qt.QVector4D)):
            if isinstance(data, qt.QVector3D):
                text = '(x: %g; y: %g; z: %g)' % (data.x(), data.y(), data.z())
            elif isinstance(data, qt.QVector4D):
                text = '(%g; %g; %g; %g)' % (data.x(), data.y(), data.z(), data.w())
            else:
                text = ''

            painter.save()
            painter.setRenderHint(qt.QPainter.Antialiasing, True)

            # Select palette color group
            colorGroup = qt.QPalette.Inactive
            if option.state & qt.QStyle.State_Active:
                colorGroup = qt.QPalette.Active
            if not option.state & qt.QStyle.State_Enabled:
                colorGroup = qt.QPalette.Disabled

            # Draw background if selected
            if option.state & qt.QStyle.State_Selected:
                brush = option.palette.brush(colorGroup,
                                             qt.QPalette.Highlight)
                painter.fillRect(option.rect, brush)

            # Draw text
            if option.state & qt.QStyle.State_Selected:
                colorRole = qt.QPalette.HighlightedText
            else:
                colorRole = qt.QPalette.WindowText
            color = option.palette.color(colorGroup, colorRole)
            painter.setPen(qt.QPen(color))
            painter.drawText(option.rect, qt.Qt.AlignLeft, text)

            painter.restore()

            # The following commented code does the same as QPainter based code
            # but it does not work with PySide
            # self.initStyleOption(option, index)
            # option.text = text
            # widget = option.widget
            # style = qt.QApplication.style() if not widget else widget.style()
            # style.drawControl(qt.QStyle.CE_ItemViewItem, option, painter, widget)

        else:
            super(ParameterTreeDelegate, self).paint(painter, option, index)

    def _commit(self, *args):
        """Commit data to the model from editors"""
        sender = self.sender()
        self.commitData.emit(sender)

    def editorEvent(self, event, model, option, index):
        """See :meth:`QStyledItemDelegate.editorEvent`"""
        if (event.type() == qt.QEvent.MouseButtonPress and
                isinstance(index.data(qt.Qt.EditRole), qt.QColor)):
            initialColor = index.data(qt.Qt.EditRole)

            def callback(color):
                theModel = index.model()
                theModel.setData(index, color, qt.Qt.EditRole)

            dialog = qt.QColorDialog(self.parent())
            # dialog.setOption(qt.QColorDialog.ShowAlphaChannel, True)
            if sys.platform == 'darwin':
                # Use of native color dialog on macos might cause problems
                dialog.setOption(qt.QColorDialog.DontUseNativeDialog, True)
            dialog.setCurrentColor(initialColor)
            dialog.currentColorChanged.connect(callback)
            if dialog.exec_() == qt.QDialog.Rejected:
                # Reset color
                dialog.setCurrentColor(initialColor)

            return True
        else:
            return super(ParameterTreeDelegate, self).editorEvent(
                event, model, option, index)

    def createEditor(self, parent, option, index):
        """See :meth:`QStyledItemDelegate.createEditor`"""
        data = index.data(qt.Qt.EditRole)
        data = self._fixVariant(data)
        editorHint = index.data(qt.Qt.UserRole)

        if callable(editorHint):
            editor = editorHint()
            assert isinstance(editor, qt.QWidget)
            editor.setParent(parent)

        elif isinstance(data, numbers.Number) and editorHint is not None:
            # Use a slider
            editor = IntSliderEditor(parent)
            range_ = editorHint
            editor.setRange(*range_)
            editor.sliderReleased.connect(self._commit)

        elif isinstance(data, six.string_types) and editorHint is not None:
            # Use a combo box
            editor = qt.QComboBox(parent)
            if data not in editorHint:
                editor.addItem(data)
            editor.addItems(editorHint)

            index = editor.findText(data)
            editor.setCurrentIndex(index)

            editor.currentIndexChanged.connect(self._commit)

        else:
            # Handle overridden editors from Python
            # Mimic Qt C++ implementation
            for type_, editorClass in self.EDITORS.items():
                if isinstance(data, type_):
                    editor = editorClass(parent)
                    metaObject = editor.metaObject()
                    userProperty = metaObject.userProperty()
                    if userProperty.isValid() and userProperty.hasNotifySignal():
                        notifySignal = userProperty.notifySignal()
                        if hasattr(notifySignal, 'signature'):  # Qt4
                            signature = notifySignal.signature()
                        else:
                            signature = notifySignal.methodSignature()
                            if qt.BINDING == 'PySide2':
                                signature = signature.data()
                            else:
                                signature = bytes(signature)

                        if hasattr(signature, 'decode'):  # For PySide with python3
                            signature = signature.decode('ascii')
                        signalName = signature.split('(')[0]

                        signal = getattr(editor, signalName)
                        signal.connect(self._commit)
                    break

            else:  # Default handling for default types
                return super(ParameterTreeDelegate, self).createEditor(
                    parent, option, index)

        editor.setAutoFillBackground(True)
        return editor

    def setModelData(self, editor, model, index):
        """See :meth:`QStyledItemDelegate.setModelData`"""
        if isinstance(editor, tuple(self.EDITORS.values())):
            # Special handling of Python classes
            # Translation of QStyledItemDelegate::setModelData to Python
            # To make it work with Python QVariant wrapping/unwrapping
            name = editor.metaObject().userProperty().name()
            if not name:
                pass  # TODO handle the case of missing user property
            if name:
                if hasattr(editor, name):
                    value = getattr(editor, name)
                else:
                    value = editor.property(name)
                model.setData(index, value, qt.Qt.EditRole)

        else:
            super(ParameterTreeDelegate, self).setModelData(editor, model, index)


class ParamTreeView(qt.QTreeView):
    """QTreeView specific to handle plot3d scene and object parameters.

    It provides additional editors and specific creation of persistent editors.

    :param parent: The widget's parent.
    """

    def __init__(self, parent=None):
        super(ParamTreeView, self).__init__(parent)

        header = self.header()
        header.setMinimumSectionSize(128)  # For colormap pixmaps
        if hasattr(header, 'setSectionResizeMode'):  # Qt5
            header.setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        else:  # Qt4
            header.setResizeMode(qt.QHeaderView.ResizeToContents)

        delegate = ParameterTreeDelegate()
        self.setItemDelegate(delegate)

        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.expanded.connect(self._expanded)

        self.setEditTriggers(qt.QAbstractItemView.CurrentChanged |
                             qt.QAbstractItemView.DoubleClicked)

        self.__persistentEditors = set()

    def _openEditorForIndex(self, index):
        """Check if it has to open a persistent editor for a specific cell.

        :param QModelIndex index: The cell index
        """
        if index.flags() & qt.Qt.ItemIsEditable:
            data = index.data(qt.Qt.EditRole)
            editorHint = index.data(qt.Qt.UserRole)
            if (isinstance(data, bool) or
                    callable(editorHint) or
                    (isinstance(data, numbers.Number) and editorHint)):
                self.openPersistentEditor(index)
                self.__persistentEditors.add(index)

    def _openEditors(self, parent=qt.QModelIndex()):
        """Open persistent editors in a subtree starting at parent.

        :param QModelIndex parent: The root of the subtree to process.
        """
        model = self.model()
        if model is not None:
            for index in visitQAbstractItemModel(model, parent):
                self._openEditorForIndex(index)

    def setModel(self, model):
        """Set the model this TreeView is displaying

        :param QAbstractItemModel model:
        """
        super(ParamTreeView, self).setModel(model)
        self._openEditors()

    def rowsInserted(self, parent, start, end):
        """See :meth:`QTreeView.rowsInserted`"""
        super(ParamTreeView, self).rowsInserted(parent, start, end)
        model = self.model()
        if model is not None:
            for row in range(start, end+1):
                self._openEditorForIndex(model.index(row, 1, parent))
                self._openEditors(model.index(row, 0, parent))

    def _expanded(self, index):
        """Handle QTreeView expanded signal"""
        name = index.data(qt.Qt.DisplayRole)
        if name == 'Transform':
            rotateIndex = self.model().index(1, 0, index)
            self.setExpanded(rotateIndex, True)

    def dataChanged(self, topLeft, bottomRight, roles=()):
        """Handle model dataChanged signal eventually closing editors"""
        if roles:  # Qt 5
            super(ParamTreeView, self).dataChanged(topLeft, bottomRight, roles)
        else:  # Qt4 compatibility
            super(ParamTreeView, self).dataChanged(topLeft, bottomRight)
        if not roles or qt.Qt.UserRole in roles:  # Check editorHint update
            for row in range(topLeft.row(), bottomRight.row() + 1):
                for column in range(topLeft.column(), bottomRight.column() + 1):
                    index = topLeft.sibling(row, column)
                    if index.isValid():
                        if self._isPersistentEditorOpen(index):
                            self.closePersistentEditor(index)
                        self._openEditorForIndex(index)

    def _isPersistentEditorOpen(self, index):
        """Returns True if a persistent editor is opened for index

        :param QModelIndex index:
        :rtype: bool
        """
        return index in self.__persistentEditors

    def selectionCommand(self, index, event=None):
        """Filter out selection of not selectable items"""
        if index.flags() & qt.Qt.ItemIsSelectable:
            return super(ParamTreeView, self).selectionCommand(index, event)
        else:
            return qt.QItemSelectionModel.NoUpdate
