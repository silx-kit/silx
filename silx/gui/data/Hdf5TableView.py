# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
This module  define model and widget to display 1D slices from numpy
array using compound data types or hdf5 databases.
"""
from __future__ import division

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "12/02/2019"

import collections
import functools
import os.path
import logging
import h5py
import numpy

from silx.gui import qt
import silx.io
from .TextFormatter import TextFormatter
import silx.gui.hdf5
from silx.gui.widgets import HierarchicalTableView
from ..hdf5.Hdf5Formatter import Hdf5Formatter
from ..hdf5._utils import htmlFromDict


_logger = logging.getLogger(__name__)


class _CellData(object):
    """Store a table item
    """
    def __init__(self, value=None, isHeader=False, span=None, tooltip=None):
        """
        Constructor

        :param str value: Label of this property
        :param bool isHeader: True if the cell is an header
        :param tuple span: Tuple of row, column span
        """
        self.__value = value
        self.__isHeader = isHeader
        self.__span = span
        self.__tooltip = tooltip

    def isHeader(self):
        """Returns true if the property is a sub-header title.

        :rtype: bool
        """
        return self.__isHeader

    def value(self):
        """Returns the value of the item.
        """
        return self.__value

    def span(self):
        """Returns the span size of the cell.

        :rtype: tuple
        """
        return self.__span

    def tooltip(self):
        """Returns the tooltip of the item.

        :rtype: tuple
        """
        return self.__tooltip

    def invalidateValue(self):
        self.__value = None

    def invalidateToolTip(self):
        self.__tooltip = None

    def data(self, role):
        return None


class _TableData(object):
    """Modelize a table with header, row and column span.

    It is mostly defined as a row based table.
    """

    def __init__(self, columnCount):
        """Constructor.

        :param int columnCount: Define the number of column of the table
        """
        self.__colCount = columnCount
        self.__data = []

    def rowCount(self):
        """Returns the number of rows.

        :rtype: int
        """
        return len(self.__data)

    def columnCount(self):
        """Returns the number of columns.

        :rtype: int
        """
        return self.__colCount

    def clear(self):
        """Remove all the cells of the table"""
        self.__data = []

    def cellAt(self, row, column):
        """Returns the cell at the row column location. Else None if there is
        nothing.

        :rtype: _CellData
        """
        if row < 0:
            return None
        if column < 0:
            return None
        if row >= len(self.__data):
            return None
        cells = self.__data[row]
        if column >= len(cells):
            return None
        return cells[column]

    def addHeaderRow(self, headerLabel):
        """Append the table with header on the full row.

        :param str headerLabel: label of the header.
        """
        item = _CellData(value=headerLabel, isHeader=True, span=(1, self.__colCount))
        self.__data.append([item])

    def addHeaderValueRow(self, headerLabel, value, tooltip=None):
        """Append the table with a row using the first column as an header and
        other cells as a single cell for the value.

        :param str headerLabel: label of the header.
        :param object value: value to store.
        """
        header = _CellData(value=headerLabel, isHeader=True)
        value = _CellData(value=value, span=(1, self.__colCount), tooltip=tooltip)
        self.__data.append([header, value])

    def addRow(self, *args):
        """Append the table with a row using arguments for each cells

        :param list[object] args: List of cell values for the row
        """
        row = []
        for value in args:
            if not isinstance(value, _CellData):
                value = _CellData(value=value)
            row.append(value)
        self.__data.append(row)


class _CellFilterAvailableData(_CellData):
    """Cell rendering for availability of a filter"""

    _states = {
        True: ("Available", qt.QColor(0x000000), None, None),
        False: ("Not available", qt.QColor(0xFFFFFF), qt.QColor(0xFF0000),
                "You have to install this filter on your system to be able to read this dataset"),
        "na": ("n.a.", qt.QColor(0x000000), None,
               "This version of h5py/hdf5 is not able to display the information"),
    }

    def __init__(self, filterId):
        if h5py.version.hdf5_version_tuple >= (1, 10, 2):
            # Previous versions only returns True if the filter was first used
            # to decode a dataset
            self.__availability = h5py.h5z.filter_avail(filterId)
        else:
            self.__availability = "na"
        _CellData.__init__(self)

    def value(self):
        state = self._states[self.__availability]
        return state[0]

    def tooltip(self):
        state = self._states[self.__availability]
        return state[3]

    def data(self, role=qt.Qt.DisplayRole):
        state = self._states[self.__availability]
        if role == qt.Qt.TextColorRole:
            return state[1]
        elif role == qt.Qt.BackgroundColorRole:
            return state[2]
        else:
            return None


class Hdf5TableModel(HierarchicalTableView.HierarchicalTableModel):
    """This data model provides access to HDF5 node content (File, Group,
    Dataset). Main info, like name, file, attributes... are displayed
    """

    def __init__(self, parent=None, data=None):
        """
        Constructor

        :param qt.QObject parent: Parent object
        :param object data: An h5py-like object (file, group or dataset)
        """
        super(Hdf5TableModel, self).__init__(parent)

        self.__obj = None
        self.__data = _TableData(columnCount=5)
        self.__formatter = None
        self.__hdf5Formatter = Hdf5Formatter(self)
        formatter = TextFormatter(self)
        self.setFormatter(formatter)
        self.setObject(data)

    def rowCount(self, parent_idx=None):
        """Returns number of rows to be displayed in table"""
        return self.__data.rowCount()

    def columnCount(self, parent_idx=None):
        """Returns number of columns to be displayed in table"""
        return self.__data.columnCount()

    def data(self, index, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method to access data values
        in the format ready to be displayed"""
        if not index.isValid():
            return None

        cell = self.__data.cellAt(index.row(), index.column())
        if cell is None:
            return None

        if role == self.SpanRole:
            return cell.span()
        elif role == self.IsHeaderRole:
            return cell.isHeader()
        elif role in (qt.Qt.DisplayRole, qt.Qt.EditRole):
            value = cell.value()
            if callable(value):
                try:
                    value = value(self.__obj)
                except Exception:
                    cell.invalidateValue()
                    raise
            return value
        elif role == qt.Qt.ToolTipRole:
            value = cell.tooltip()
            if callable(value):
                try:
                    value = value(self.__obj)
                except Exception:
                    cell.invalidateToolTip()
                    raise
            return value
        else:
            return cell.data(role)
        return None

    def isSupportedObject(self, h5pyObject):
        """
        Returns true if the provided object can be modelized using this model.
        """
        isSupported = False
        isSupported = isSupported or silx.io.is_group(h5pyObject)
        isSupported = isSupported or silx.io.is_dataset(h5pyObject)
        isSupported = isSupported or isinstance(h5pyObject, silx.gui.hdf5.H5Node)
        return isSupported

    def setObject(self, h5pyObject):
        """Set the h5py-like object exposed by the model

        :param h5pyObject: A h5py-like object. It can be a `h5py.Dataset`,
            a `h5py.File`, a `h5py.Group`. It also can be a,
            `silx.gui.hdf5.H5Node` which is needed to display some local path
            information.
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()

        if h5pyObject is None or self.isSupportedObject(h5pyObject):
            self.__obj = h5pyObject
        else:
            _logger.warning("Object class %s unsupported. Object ignored.", type(h5pyObject))
        self.__initProperties()

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()

    def __formatHdf5Type(self, dataset):
        """Format the HDF5 type"""
        return self.__hdf5Formatter.humanReadableHdf5Type(dataset)

    def __attributeTooltip(self, attribute):
        attributeDict = collections.OrderedDict()
        if hasattr(attribute, "shape"):
            attributeDict["Shape"] = self.__hdf5Formatter.humanReadableShape(attribute)
        attributeDict["Data type"] = self.__hdf5Formatter.humanReadableType(attribute, full=True)
        html = htmlFromDict(attributeDict, title="HDF5 Attribute")
        return html

    def __formatDType(self, dataset):
        """Format the numpy dtype"""
        return self.__hdf5Formatter.humanReadableType(dataset, full=True)

    def __formatShape(self, dataset):
        """Format the shape"""
        if dataset.shape is None or len(dataset.shape) <= 1:
            return self.__hdf5Formatter.humanReadableShape(dataset)
        size = dataset.size
        shape = self.__hdf5Formatter.humanReadableShape(dataset)
        return u"%s = %s" % (shape, size)

    def __formatChunks(self, dataset):
        """Format the shape"""
        chunks = dataset.chunks
        if chunks is None:
            return ""
        shape = " \u00D7 ".join([str(i) for i in chunks])
        sizes = numpy.product(chunks)
        text = "%s = %s" % (shape, sizes)
        return text

    def __initProperties(self):
        """Initialize the list of available properties according to the defined
        h5py-like object."""
        self.__data.clear()
        if self.__obj is None:
            return

        obj = self.__obj

        hdf5obj = obj
        if isinstance(obj, silx.gui.hdf5.H5Node):
            hdf5obj = obj.h5py_object

        if silx.io.is_file(hdf5obj):
            objectType = "File"
        elif silx.io.is_group(hdf5obj):
            objectType = "Group"
        elif silx.io.is_dataset(hdf5obj):
            objectType = "Dataset"
        else:
            objectType = obj.__class__.__name__
        self.__data.addHeaderRow(headerLabel="HDF5 %s" % objectType)

        SEPARATOR = "::"

        self.__data.addHeaderRow(headerLabel="Path info")
        if isinstance(obj, silx.gui.hdf5.H5Node):
            # helpful informations if the object come from an HDF5 tree
            self.__data.addHeaderValueRow("Basename", lambda x: x.local_basename)
            self.__data.addHeaderValueRow("Name", lambda x: x.local_name)
            local = lambda x: x.local_filename + SEPARATOR + x.local_name
            self.__data.addHeaderValueRow("Local", local)
            physical = lambda x: x.physical_filename + SEPARATOR + x.physical_name
            self.__data.addHeaderValueRow("Physical", physical)
        else:
            # it's a real H5py object
            self.__data.addHeaderValueRow("Basename", lambda x: os.path.basename(x.name))
            self.__data.addHeaderValueRow("Name", lambda x: x.name)
            if obj.file is not None:
                self.__data.addHeaderValueRow("File", lambda x: x.file.filename)

            if hasattr(obj, "path"):
                # That's a link
                if hasattr(obj, "filename"):
                    link = lambda x: x.filename + SEPARATOR + x.path
                else:
                    link = lambda x: x.path
                self.__data.addHeaderValueRow("Link", link)
            else:
                if silx.io.is_file(obj):
                    physical = lambda x: x.filename + SEPARATOR + x.name
                elif obj.file is not None:
                        physical = lambda x: x.file.filename + SEPARATOR + x.name
                else:
                    # Guess it is a virtual node
                    physical = "No physical location"
                self.__data.addHeaderValueRow("Physical", physical)

        if hasattr(obj, "dtype"):

            self.__data.addHeaderRow(headerLabel="Data info")

            if hasattr(obj, "id") and hasattr(obj.id, "get_type"):
                # display the HDF5 type
                self.__data.addHeaderValueRow("HDF5 type", self.__formatHdf5Type)
            self.__data.addHeaderValueRow("dtype", self.__formatDType)
            if hasattr(obj, "shape"):
                self.__data.addHeaderValueRow("shape", self.__formatShape)
            if hasattr(obj, "chunks") and obj.chunks is not None:
                self.__data.addHeaderValueRow("chunks", self.__formatChunks)

        # relative to compression
        # h5py expose compression, compression_opts but are not initialized
        # for external plugins, then we use id
        # h5py also expose fletcher32 and shuffle attributes, but it is also
        # part of the filters
        if hasattr(obj, "shape") and hasattr(obj, "id"):
            if hasattr(obj.id, "get_create_plist"):
                dcpl = obj.id.get_create_plist()
                if dcpl.get_nfilters() > 0:
                    self.__data.addHeaderRow(headerLabel="Compression info")
                    pos = _CellData(value="Position", isHeader=True)
                    hdf5id = _CellData(value="HDF5 ID", isHeader=True)
                    name = _CellData(value="Name", isHeader=True)
                    options = _CellData(value="Options", isHeader=True)
                    availability = _CellData(value="", isHeader=True)
                    self.__data.addRow(pos, hdf5id, name, options, availability)
                for index in range(dcpl.get_nfilters()):
                    filterId, name, options = self.__getFilterInfo(obj, index)
                    pos = _CellData(value=str(index))
                    hdf5id = _CellData(value=str(filterId))
                    name = _CellData(value=name)
                    options = _CellData(value=options)
                    availability = _CellFilterAvailableData(filterId=filterId)
                    self.__data.addRow(pos, hdf5id, name, options, availability)

        if hasattr(obj, "attrs"):
            if len(obj.attrs) > 0:
                self.__data.addHeaderRow(headerLabel="Attributes")
                for key in sorted(obj.attrs.keys()):
                    callback = lambda key, x: self.__formatter.toString(x.attrs[key])
                    callbackTooltip = lambda key, x: self.__attributeTooltip(x.attrs[key])
                    self.__data.addHeaderValueRow(headerLabel=key,
                                                  value=functools.partial(callback, key),
                                                  tooltip=functools.partial(callbackTooltip, key))

    def __getFilterInfo(self, dataset, filterIndex):
        """Get a tuple of readable info from dataset filters

        :param h5py.Dataset dataset: A h5py dataset
        :param int filterId:
        """
        try:
            dcpl = dataset.id.get_create_plist()
            info = dcpl.get_filter(filterIndex)
            filterId, _flags, cdValues, name = info
            name = self.__formatter.toString(name)
            options = " ".join([self.__formatter.toString(i) for i in cdValues])
            return (filterId, name, options)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
        return (None, None, None)

    def object(self):
        """Returns the internal object modelized.

        :rtype: An h5py-like object
        """
        return self.__obj

    def setFormatter(self, formatter):
        """Set the formatter object to be used to display data from the model

        :param TextFormatter formatter: Formatter to use
        """
        if formatter is self.__formatter:
            return

        self.__hdf5Formatter.setTextFormatter(formatter)

        if qt.qVersion() > "4.6":
            self.beginResetModel()

        if self.__formatter is not None:
            self.__formatter.formatChanged.disconnect(self.__formatChanged)

        self.__formatter = formatter
        if self.__formatter is not None:
            self.__formatter.formatChanged.connect(self.__formatChanged)

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()

    def getFormatter(self):
        """Returns the text formatter used.

        :rtype: TextFormatter
        """
        return self.__formatter

    def __formatChanged(self):
        """Called when the format changed.
        """
        self.reset()


class Hdf5TableItemDelegate(HierarchicalTableView.HierarchicalItemDelegate):
    """Item delegate the :class:`Hdf5TableView` with read-only text editor"""

    def createEditor(self, parent, option, index):
        """See :meth:`QStyledItemDelegate.createEditor`"""
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, qt.QLineEdit):
            editor.setReadOnly(True)
            editor.deselect()
            editor.textChanged.connect(self.__textChanged, qt.Qt.QueuedConnection)
            self.installEventFilter(editor)
        return editor

    def __textChanged(self, text):
        sender = self.sender()
        if sender is not None:
            sender.deselect()

    def eventFilter(self, watched, event):
        eventType = event.type()
        if eventType == qt.QEvent.FocusIn:
            watched.selectAll()
            qt.QTimer.singleShot(0, watched.selectAll)
        elif eventType == qt.QEvent.FocusOut:
            watched.deselect()
        return super().eventFilter(watched, event)


class Hdf5TableView(HierarchicalTableView.HierarchicalTableView):
    """A widget to display metadata about a HDF5 node using a table."""

    def __init__(self, parent=None):
        super(Hdf5TableView, self).__init__(parent)
        self.setModel(Hdf5TableModel(self))
        self.setItemDelegate(Hdf5TableItemDelegate(self))
        self.setSelectionMode(qt.QAbstractItemView.NoSelection)

    def isSupportedData(self, data):
        """
        Returns true if the provided object can be modelized using this model.
        """
        return self.model().isSupportedObject(data)

    def setData(self, data):
        """Set the h5py-like object exposed by the model

        :param data: A h5py-like object. It can be a `h5py.Dataset`,
            a `h5py.File`, a `h5py.Group`. It also can be a,
            `silx.gui.hdf5.H5Node` which is needed to display some local path
            information.
        """
        model = self.model()

        model.setObject(data)
        header = self.horizontalHeader()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.Fixed)
        setResizeMode(1, qt.QHeaderView.ResizeToContents)
        setResizeMode(2, qt.QHeaderView.Stretch)
        setResizeMode(3, qt.QHeaderView.ResizeToContents)
        setResizeMode(4, qt.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        for row in range(model.rowCount()):
            for column in range(model.columnCount()):
                index = model.index(row, column)
                if (index.isValid() and index.data(
                        HierarchicalTableView.HierarchicalTableModel.IsHeaderRole) is False):
                    self.openPersistentEditor(index)
