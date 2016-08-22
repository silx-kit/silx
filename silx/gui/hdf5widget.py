# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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

"""Qt tree model for a HDF5 file

.. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
    library, which is not a mandatory dependency for `silx`. You might need
    to install it if you don't already have it.
"""

import os
import sys
import numpy
import logging
from . import qt
from ..io import spech5


try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/08/2016"


_logger = logging.getLogger(__name__)


def is_hdf5_file(fname):
    """Return True if file is an HDF5 file, else False
    """
    if not os.path.isfile(fname):
        raise IOError("Parameter fname (%s) is not a valid file path" % fname)
    try:
        f = h5py.File(fname)
    except IOError:
        return False
    else:
        f.close()
        return True


def htmlFromDict(input):
    """Generate a readable HTML from a dictionary

    :param input dict: A Dictionary
    :rtype: str
    """
    result = "<html><ul>"
    for key, value in input.items():
        result += "<li><b>%s</b>: %s</li>" % (key, value)
    result += "</ul></html>"
    return result


class MultiColumnTreeItem(qt.QStandardItem):
    """A QStandardItem used to create an item tree,
    which is able to manage his item colums"""

    def __init__(self, text=None, icon=None):
        """Constructor

        :param text str: Text displayed by the item
        :param icon qtQIcon: Icon displayed by the item
        """
        if icon is not None:
            qt.QStandardItem.__init__(self, icon, text)
        elif text is not None:
            qt.QStandardItem.__init__(self, text)
        else:
            qt.QStandardItem.__init__(self)
        self.__row = [self]

    def setExtraColumns(self, *args):
        """Define other items of the row.

        :param args list of qt.QStandardItem: A list of items
        """
        row = [self]
        row.extend(args)
        self.__row = row

    def _getItemRow(self):
        """Return the item row. The one appended to the table
        
        :rtype: list
        """
        return self.__row

    def setChild(self, row, item):
        """"Override of default setChild to be able to set a full row
        instead of the single item.

        :param row int: An row index
        :param item qt.QStandardItem: An item
        """
        if isinstance(item, MultiColumnTreeItem):
            for column, columnItem in enumerate(item._getItemRow()):
                super(MultiColumnTreeItem, self).setChild(row, column, columnItem)
            item = item._getItemRow()
        else:
            super(MultiColumnTreeItem, self).setChild(row, item)

    def appendRow(self, item):
        """"Override of default appendRow to be able to append the full row
        instead of the single item.
        
        :param item qt.QStandardItem: An item
        """
        if isinstance(item, MultiColumnTreeItem):
            item = item._getItemRow()
        super(MultiColumnTreeItem, self).appendRow(item)


class MustBeLoadedItem(qt.QStandardItem):
    """A dummy item must be created, else parent from modelitem is not
    valid"""
    pass


class LazyLoadableItem(object):
    """A way to tag Item as lazy loadable item.

    Child can be lazy loaded by the class model by calling
    hasChildren and child methods. This methods are not virtual
    in QStandardItem, then it has to be called in python side.
    """

    def __init__(self):
        self._must_load_child = self.hasChildren()
        self._populateDummies()

    def hasChildren(self):
        """Override method to be able to generate chrildren on demand.
        The result is computed from the HDF5 model.

        :rtype: bool
        """
        raise NotImplementedError()

    def child(self, row, column):
        """Override method to be able to generate chrildren on demand.

        :rtype list of QStandardItem:
        """
        if self._must_load_child:
            self._populateChild()
            self._must_load_child = False

    def rowCount(self):
        """Override method to be able to generate chrildren on demand.

        :rtype list of QStandardItem:
        """
        raise NotImplementedError()

    def _populateDummies(self):
        """Called to populate child with dummy items.

        If no dummies are created, index parent is not valid.

        :rtype list of QStandardItem:
        """
        for row in range(self.rowCount()):
            self.setChild(row, MustBeLoadedItem())

    def _populateChild(self):
        """Called to populate child
        """
        raise NotImplementedError()


class Hdf5BrokenLinkItem(MultiColumnTreeItem):
    """Subclass of :class:`qt.QStandardItem` to represent a broken link
    in HDF5 tree structure.
    """

    def __init__(self, text, obj=None, message=None):
        """Constructor

        :param text str: Text displayed by the item
        :param obj h5py link: HDF5 object containing link informations
        :param message str: Message to display as description
        """
        super(Hdf5BrokenLinkItem, self).__init__(text)

        style = qt.QApplication.style()
        icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
        self.setIcon(icon)

        self.obj = obj
        if message is None:
            if isinstance(self.obj, h5py.ExternalLink):
                message = "External link broken. Path %s::%s does not exist" % (self.obj.filename, self.obj.path)
            elif isinstance(self.obj, h5py.SoftLink):
                message = "Soft link broken. Path %s does not exist" % (self.obj.path)
            else:
                name = obj.__class__.__name__.split(".")[-1].capitalize()
                message = "%s broken" % (name)
        self._item_description = qt.QStandardItem(message)
        self._item_type = qt.QStandardItem("")
        self.setExtraColumns(self._item_description, self._item_type)
        self._message = message

        self._setDefaultToolTip()

    def _setDefaultToolTip(self):
        input = {}
        if isinstance(self.obj, h5py.ExternalLink):
            input["linked path"] = self.obj.path
            input["linked file"] = self.obj.filename
        elif isinstance(self.obj, h5py.SoftLink):
            input["linked path"] = self.obj.path
        tooltip = htmlFromDict(input)
        self.setToolTip(tooltip)


class Hdf5Item(MultiColumnTreeItem, LazyLoadableItem):
    """Subclass of :class:`qt.QStandardItem` to represent an HDF5-like
    item (dataset, file, group or link) as an element of a HDF5-like
    tree structure.
    """

    def __init__(self, text, obj):
        super(Hdf5Item, self).__init__(text)

        self.basename = text
        """Name of the item: base filename, group name, or dataset name"""

        self.obj = obj
        """Pointer to data instance. Data can be an instance of one of
        the following classes:

            - :class:`h5py.File` (:attr:`itemtype` *file*)
            - :class:`h5py.Group` (:attr:`itemtype` *group*)
            - :class:`h5py.Dataset` (:attr:`itemtype` *dataset*)
            - :class:`h5py.SoftLink` (:attr:`itemtype` *soft link*)
            - :class:`h5py.ExternalLink`(:attr:`itemtype` *external link*)
            - :class:`silx.io.spech5.SpecH5` (:attr:`itemtype` *file*)
            - :class:`silx.io.spech5.SpecH5Group` (:attr:`itemtype` *group*)
            - :class:`silx.io.spech5.SpecH5Dataset` (:attr:`itemtype` *dataset*)
            - :class:`silx.io.spech5.SpecH5LinkToGroup` (:attr:`itemtype` *group*)
            - :class:`silx.io.spech5.SpecH5LinkToDataset` (:attr:`itemtype` *dataset*)
        """

        self.itemtype = self._getH5ClassName()
        """Type of item: 'file', 'group', 'dataset', 'soft link' or
        'external link'. For hard links, the type is the type of the target item."""

        self.hdf5name = self.obj.name
        """Name of group or dataset within the HDF5 file."""

        LazyLoadableItem.__init__(self)

        # store owned items
        self._item_type = self._createTypeItem()
        self._item_description = self._createDescriptionItem()
        self.setExtraColumns(self._item_description, self._item_type)

        self._setDefaultTooltip()

    def _getH5ClassName(self):
        if hasattr(self.obj, "h5py_class"):
            class_ = self.obj.h5py_class
        else:
            class_ = self.obj.__class__

        if issubclass(class_, h5py.File):
            return "file"
        elif issubclass(class_, h5py.SoftLink):
            return "soft link"
        elif issubclass(class_, h5py.ExternalLink):
            return "external link"
        # hard link type = target type (Group or Dataset)
        elif issubclass(class_, h5py.Group):
            return "group"
        elif issubclass(class_, h5py.Dataset):
            return "dataset"
        else:
            raise TypeError("Unsupported class '%s'" % class_)

    def hasChildren(self):
        """Override method to be able to generate chrildren on demand.
        The result is computed from the HDF5 model.

        :rtype: bool
        """
        if self.isGroupObj():
            return len(self.obj) > 0
        return super(Hdf5Item, self).hasChildren()

    def rowCount(self):
        if self.isGroupObj():
            return len(self.obj)
        return super(Hdf5Item, self).rowCount()

    def child(self, row, column):
        """Override method to be able to generate chrildren on demand."""
        LazyLoadableItem.child(self, row, column)
        return MultiColumnTreeItem.child(self, row, column)

    def _populateChild(self):
        """Recurse through an HDF5 structure to append groups an datasets
        into the tree model.
        :param h5item: Parent :class:`Hdf5Item` or
            :class:`Hdf5ItemModel` object
        :param gr_or_ds: h5py or spech5 object (h5py.File, h5py.Group,
            h5py.Dataset, spech5.SpecH5, spech5.SpecH5Group,
            spech5.SpecH5Dataset)
        """
        row = 0
        if self.isGroupObj():
            for child_gr_ds_name in self.obj:
                try:
                    child_gr_ds = self.obj.get(child_gr_ds_name)
                except RuntimeError as e:
                    _logger.error("Internal h5py error", exc_info=True)
                    link = self.obj.get(child_gr_ds_name, getlink=True)
                    item = Hdf5BrokenLinkItem(text=child_gr_ds_name, obj=link, message=e.args[0])
                else:
                    if child_gr_ds is None:
                        # that's a broken link
                        link = self.obj.get(child_gr_ds_name, getlink=True)
                        item = Hdf5BrokenLinkItem(text=child_gr_ds_name, obj=link)
                    else:
                        item = Hdf5Item(text=child_gr_ds_name, obj=child_gr_ds)
                self.setChild(row, item)
                #self.appendRow(item)
                row += 1

    def isGroupObj(self):
        """Is the hdf5 obj contains sub group or datasets"""
        return self.itemtype in ["group", "file"]

    def _setDefaultTooltip(self):
        """Set the default tooltip"""
        attrs = dict(self.obj.attrs)
        if self.itemtype == "dataset":
            if self.obj.shape == ():
                attrs["shape"] = "scalar"
            else:
                attrs["shape"] = self.obj.shape
            attrs["dtype"] = self.obj.dtype
            if self.obj.shape == ():
                attrs["value"] = self.obj.value
            else:
                attrs["value"] = "..."

        if len(attrs) > 0:
            tooltip = htmlFromDict(attrs)
        else:
            tooltip = ""

        self.setToolTip(tooltip)

    def _createTypeItem(self):
        """Create the item holding the type column"""
        if self.itemtype == "dataset":
            if self.obj.dtype.type == numpy.string_:
                text = "string"
            else:
                text = str(self.obj.dtype)

            for axes in self.obj.shape:
                text = u"%s \u00D7 %s" % (text, axes)
        else:
            text = ""

        return qt.QStandardItem(text)

    def _createDescriptionItem(self):
        """Create the item holding the description column"""
        text = self.itemtype.capitalize()
        if "desc" in self.obj.attrs:
            text += ": " + self.obj.attrs["desc"]

        return qt.QStandardItem(text)

    @property
    def filename(self):
        """Path to parent file in the filesystem"""
        # filename made a property rather than an attribute because
        # we don't know self.parent() in __init__.
        return self.getParentFile().filename

    def getParentFile(self):
        """Return reference to parent file handle (:class:`h5py.File` or
        :class:`silx.io.spech5.SpecH5` object), which is stored
        as attribute ``obj`` of the root level item with
        ``itemtype == "file"``

        You should only call this method if this :class:`Hdf5Item` is part
        of a valid :class:`Hdf5TreeModel` with a file item a the root level.
        """
        if self.itemtype == "file":
            return self.obj

        parent = self.parent()

        errmsg = "Cannot find parent of Hdf5Item %s"
        if parent is None:
            raise AttributeError(errmsg % self.basename)

        while parent.itemtype != "file":
            basename = parent.basename[:]
            parent = parent.parent()
            if parent is None:
                raise AttributeError(errmsg % basename)

        return parent.obj


class Hdf5TreeModel(qt.QStandardItemModel):
    """Data model for the content of an HDF5 file or a Specfile.
    This model is a hierarchical tree, whose nodes are :class:`Hdf5Item`
    objects in the first column and :class:`qt.QStandardItem` in the second
    column.
    The first column contains the data name and a pointer to the HDF5 data
    objects, while the second column is only a data description to be
    displayed in a tree view.
    """
    def __init__(self):
        """
        :param files: List of file handles/descriptors for a :class:`h5py.File`
            or a  :class:`spech5.SpecH5` object, or list of file pathes.
        """
        super(Hdf5TreeModel, self).__init__()
        self.setHorizontalHeaderLabels(['Name', 'Description', 'Type'])

    def itemFromIndex(self, index):
        """
        Override itemFromIndex to be able to call non virtual method
        Qt.QStandardItem.child

        :param index qt.QModelIndex: An index
        :rtype: qt.QStandardItem
        """
        item = qt.QStandardItemModel.itemFromIndex(self, index)
        if isinstance(item, MustBeLoadedItem):
            parent_index = self.parent(index)
            parent = qt.QStandardItemModel.itemFromIndex(self, parent_index)
            if isinstance(parent, LazyLoadableItem):
                item = parent.child(index.row(), index.column())
        return item

    def hasChildren(self, index):
        """
        Override hasChildren to be able to call non virtual method
        Qt.QStandardItem.hasChildren

        :param index qt.QModelIndex: An index
        :rtype: bool
        """
        item = self.itemFromIndex(index)
        if isinstance(item, LazyLoadableItem):
            return item.hasChildren()
        return super(Hdf5TreeModel, self).hasChildren(index)

    def rowCount(self, index):
        """
        Override rowCount to be able to call non virtual method
        Qt.QStandardItem.rowCount

        :param index qt.QModelIndex: An index
        :rtype: int
        """
        item = self.itemFromIndex(index)
        if isinstance(item, LazyLoadableItem):
            return item.rowCount()
        return super(Hdf5TreeModel, self).rowCount(index)

    def appendRow(self, items):
        # TODO it would be better to generate a self invisibleItem, but it looks to be impossible
        if isinstance(items, MultiColumnTreeItem):
            items = items._getItemRow()
        super(Hdf5TreeModel, self).appendRow(items)

    def appendH5pyObject(self, h5pyObject, text=None):
        """Append an HDF5 object from h5py to the tree.

        :param h5pyObject: File handle/descriptor for a :class:`h5py.File`
            or any other class of h5py file structure.
        """
        if text is None:
            if hasattr(h5pyObject, "h5py_class"):
                class_ = h5pyObject.h5py_class
            else:
                class_ = h5pyObject.__class__

            if class_ == h5py.File:
                text = os.path.basename(h5pyObject.filename)
            else:
                filename = os.path.basename(h5pyObject.file.filename)
                path = h5pyObject.name
                text = "%s::%s" % (filename, path)

        file_item = Hdf5Item(text=text, obj=h5pyObject)
        self.appendRow(file_item)

    def appendFile(self, filename):
        """Load a HDF5 file into the data model.

        :param filename: file path.
        """
        if not os.path.isfile(filename):
            raise IOError("Filename '%s' must be a file path" % filename)
        try:
            if is_hdf5_file(filename):
                fd = h5py.File(filename)
            else:
                # assume Specfile
                fd = spech5.SpecH5(filename)

            # add root level row with file name
            self.appendH5pyObject(fd)
        except IOError:
            _logger.debug("File '%s' can't be read.", filename, exc_info=True)
            raise IOError("File '%s' can't be read as HDF5 or SpecFile" % filename)


class Hdf5TreeView(qt.QTreeView):
    """TreeView which allow to browse HDF5 file structure.

    It provids columns width auto-resizing and additional
    signals.

    The default model is `Hdf5TreeModel`.
    """
    enterKeyPressed = qt.pyqtSignal()

    def __init__(self, parent=None):
        qt.QTreeView.__init__(self, parent)
        self.setModel(Hdf5TreeModel())
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)

        self.__autoResizeColumns = False
        self.lastMouse = None

    def setAutoResizeColumns(self, autoResizeColumns):
        """Enable/disable  auto-resize of columns headers when
        expanding/collapsing items.

        :param autoResizeColumns bool: True to enable the behaviour.
        """
        if self.__autoResizeColumns == autoResizeColumns:
            return
        self.__autoResizeColumns = autoResizeColumns
        if self.__autoResizeColumns:
            self.expanded.connect(self.resizeAllColumns)
            self.collapsed.connect(self.resizeAllColumns)
        else:
            self.expanded.disconnect(self.resizeAllColumns)
            self.collapsed.disconnect(self.resizeAllColumns)

    def getAutoResizeColumns(self):
        """Returns true if the auto-resize behaviour is enabled.

        :rtype: bool
        """
        return self.__autoResizeColumns

    autoResizeColumns = qt.pyqtProperty(bool, getAutoResizeColumns, setAutoResizeColumns)
    """Property to enable/disable the auto-resize behaviour when user expend
    of collapse items."""

    def resizeAllColumns(self):
        for i in range(0, self.model().columnCount()):
            self.resizeColumnToContents(i)

    def keyPressEvent(self, event):
        """Overload QTreeView.keyPressEvent to emit an enterKeyPressed
        signal when users press enter.
        """
        if event.key() in [qt.Qt.Key_Enter, qt.Qt.Key_Return]:
            self.enterKeyPressed.emit()
        qt.QTreeView.keyPressEvent(self, event)

    def mousePressEvent(self, event):
        """On mouse press events, remember which button was pressed
        in :attr:`lastMouse`. Make sure itemClicked signal
        is emitted.
        """
        self.lastMouse = event.button()
        super(Hdf5TreeView, self).mousePressEvent(event)
        if event.button() != qt.Qt.LeftButton:
            # Qt5 only sends itemClicked on left button mouse click
            if qt.qVersion() > "5":
                qindex = self.indexAt(event.pos())
                self.clicked.emit(qindex)
