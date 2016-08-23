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
__date__ = "23/08/2016"


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
        self.setExtraColumns(None, None, None, self._item_description, None)
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
        """
        :param text str: text displayed
        :param obj object: Pointer to h5py data. See the `obj` attribute.
        """
        super(Hdf5Item, self).__init__(text)

        self.obj = obj
        """Pointer to h5py data. Can be one of the following classes:

            - :class:`h5py.File` (:attr:`itemtype` *file*)
            - :class:`h5py.Group` (:attr:`itemtype` *group*)
            - :class:`h5py.Dataset` (:attr:`itemtype` *dataset*)
            - :class:`h5py.SoftLink` (:attr:`itemtype` *soft link*)
            - :class:`h5py.ExternalLink`(:attr:`itemtype` *external link*)
            - or a mimick version (in this case the class provide a property `h5py_class`)
        """

        LazyLoadableItem.__init__(self)

        # store owned items
        self._item_type = self._createTypeItem()
        self._item_shape = self._createShapeItem()
        self._item_value = self._createValueItem()
        self._item_description = self._createDescriptionItem()
        self._item_node = self._createNodeItem()
        self.setExtraColumns(
            self._item_type,
            self._item_shape,
            self._item_value,
            self._item_description,
            self._item_node)

        self._setDefaultTooltip()

    def _getH5pyClass(self):
        if hasattr(self.obj, "h5py_class"):
            class_ = self.obj.h5py_class
        else:
            class_ = self.obj.__class__
        return class_

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
        class_ = self._getH5pyClass()
        return issubclass(class_, (h5py.File, h5py.Group))

    def _setDefaultTooltip(self):
        """Set the default tooltip"""
        class_ = self._getH5pyClass()
        attrs = dict(self.obj.attrs)
        if issubclass(class_, h5py.Dataset):
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
        class_ = self._getH5pyClass()
        if issubclass(class_, h5py.Dataset):
            if self.obj.dtype.type == numpy.string_:
                text = "string"
            else:
                text = str(self.obj.dtype)
        else:
            text = ""

        return qt.QStandardItem(text)

    def _createShapeItem(self):
        """Create the item holding the type column"""
        class_ = self._getH5pyClass()
        if not issubclass(class_, h5py.Dataset):
            return None

        shape = [str(i) for i in self.obj.shape]
        text = u" \u00D7 ".join(shape)
        return qt.QStandardItem(text)

    def _createValueItem(self):
        """Create the item holding the type column"""
        class_ = self._getH5pyClass()
        if not issubclass(class_, h5py.Dataset):
            return None

        numpy_object = self.obj.value

        if self.obj.dtype.type == numpy.object_:
            text = str(numpy_object)
        elif self.obj.dtype.type == numpy.string_:
            text = str(numpy_object)
        else:
            size = 1
            for dim in numpy_object.shape:
                size = size * dim

            if size > 5:
                text = "..."
            else:
                text = str(numpy_object)

        return qt.QStandardItem(text)

    def _createDescriptionItem(self):
        """Create the item holding the description column"""
        if "desc" in self.obj.attrs:
            text = self.obj.attrs["desc"]
        else:
            return None
        item = qt.QStandardItem(text)
        item.setToolTip("Description:%s" % text)
        return item

    def _createNodeItem(self):
        """Create the item holding the description column"""
        class_ = self._getH5pyClass()
        text = class_.__name__.split(".")[-1]
        item = qt.QStandardItem(text)
        item.setToolTip("Class name: %s" % self.__class__)
        return item


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
        self.setHorizontalHeaderLabels([
            'Name',
            'Type',
            'Shape',
            'Value',
            'Description',
            'Node',
        ])

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


class Hdf5HeaderView(qt.QHeaderView):
    """
    Default HDF5 header

    Manage auto-resize and context menu to display/hide columns
    """

    def __init__(self, orientation, parent=None):
        """\
        Constructor

        :param orientation qt.Qt.Orientation: Orientation of the header
        :param parent qt.QWidget: Parent of the widget
        """
        super(Hdf5HeaderView, self).__init__(orientation, parent)
        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__createContextMenu)

        # default initialization done by QTreeView for it's own header
        self.setClickable(True)
        self.setMovable(True)
        self.setDefaultAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self.setStretchLastSection(True)

        self.__auto_resize = True

    def setModel(self, model):
        """Override model to configure view when a model is expected

        `qt.QHeaderView.setResizeMode` expect already existing columns
        to work.

        :param model qt.QAbstractItemModel: A model
        """
        super(Hdf5HeaderView, self).setModel(model)
        self.__updateAutoResize()

    def __updateAutoResize(self):
        """Update the view according to the state of the auto-resize"""
        if self.__auto_resize:
            self.setResizeMode(0, qt.QHeaderView.ResizeToContents)
            self.setResizeMode(1, qt.QHeaderView.ResizeToContents)
            self.setResizeMode(2, qt.QHeaderView.ResizeToContents)
            self.setResizeMode(3, qt.QHeaderView.Interactive)
            self.setResizeMode(4, qt.QHeaderView.Interactive)
            self.setResizeMode(5, qt.QHeaderView.ResizeToContents)
        else:
            self.setResizeMode(0, qt.QHeaderView.Interactive)
            self.setResizeMode(1, qt.QHeaderView.Interactive)
            self.setResizeMode(2, qt.QHeaderView.Interactive)
            self.setResizeMode(3, qt.QHeaderView.Interactive)
            self.setResizeMode(4, qt.QHeaderView.Interactive)
            self.setResizeMode(5, qt.QHeaderView.Interactive)

    def setAutoResizeColumns(self, autoResize):
        """Enable/disable auto-resize. When auto-resized, the header take care
        of the content of the column to set fixed size of some of them, or to
        auto fix the size according to the content.

        :param autoResize bool: Enable/disable auto-resize
        """
        if self.__auto_resize == autoResize:
            return
        self.__auto_resize = autoResize
        self.__updateAutoResize()

    def hasAutoResizeColumns(self):
        """Is auto-resize enabled.

        :rtype: bool
        """
        return self.__auto_resize

    autoResizeColumns = qt.pyqtProperty(bool, hasAutoResizeColumns, setAutoResizeColumns)
    """Property to enable/disable auto-resize."""

    def __createContextMenu(self, pos):
        """Callback to create and display a context menu

        :param pos qt.QPoint: Requested position for the context menu
        """
        model = self.model()
        if model.columnCount() > 1:
            menu = qt.QMenu(self)
            menu.setTitle("Display/hide columns")

            action = qt.QAction("Display/hide column", self)
            action.setEnabled(False)
            menu.addAction(action)

            for column in range(model.columnCount()):
                if column == 0:
                    # skip the main column
                    continue
                text = model.headerData(column, qt.Qt.Horizontal)
                action = qt.QAction("%s displayed" % text, self)
                action.setCheckable(True)
                action.setChecked(not self.isSectionHidden(column))
                gen_hide_section_event = lambda column: lambda checked: self.setSectionHidden(column, not checked)
                action.toggled.connect(gen_hide_section_event(column))
                menu.addAction(action)

            menu.popup(self.viewport().mapToGlobal(pos))


class Hdf5TreeView(qt.QTreeView):
    """TreeView which allow to browse HDF5 file structure.

    It provids columns width auto-resizing and additional
    signals.

    The default model is `Hdf5TreeModel` and the default header is
    `Hdf5HeaderView`.
    """
    def __init__(self, parent=None):
        qt.QTreeView.__init__(self, parent)
        self.setModel(Hdf5TreeModel())
        self.setHeader(Hdf5HeaderView(qt.Qt.Horizontal, self))
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        # optimise the rendering
        self.setUniformRowHeights(True)

    def selectedH5pyObjects(self, ignoreBrokenLinks=True):
        """Returns selected h5py objects like `h5py.File`, `h5py.Group`,
        `h5py.Dataset` or mimicked objects.
        :param ignoreBrokenLinks bool: Returns objects which are not not
            broken links.
        """
        result = []
        for index in self.selectedIndexes():
            item = self.model().itemFromIndex(index)
            if item is None:
                continue
            if isinstance(item, Hdf5Item):
                result.append(item.obj)
            if not ignoreBrokenLinks and isinstance(item, Hdf5BrokenLinkItem):
                result.append(item.obj)
        return result

    def mousePressEvent(self, event):
        """Override mousePressEvent to provide a consistante compatible API
        between Qt4 and Qt5
        """
        super(Hdf5TreeView, self).mousePressEvent(event)
        if event.button() != qt.Qt.LeftButton:
            # Qt5 only sends itemClicked on left button mouse click
            if qt.qVersion() > "5":
                qindex = self.indexAt(event.pos())
                self.clicked.emit(qindex)
