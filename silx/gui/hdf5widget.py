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
"""
import h5py
import os
import sys
import numpy
from silx.gui import qt
from silx.io import spech5

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "16/08/2016"


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


class Hdf5Item(qt.QStandardItem):
    """Subclass of :class:`qt.QStandardItem` to represent an HDF5-like
    item (dataset, file, group or link) as an element of a HDF5-like
    tree structure.
    """
    def __init__(self, text, obj):
        super(Hdf5Item, self).__init__(text)

        group_types = (h5py.File, h5py.Group,
                       spech5.SpecH5, spech5.SpecH5Group,
                       spech5.SpecH5LinkToGroup  # ~ hard link
                       )

        if isinstance(obj, h5py.File):
            itemtype = "file"
        elif isinstance(obj, h5py.SoftLink):
            itemtype = "soft link"
        elif isinstance(obj, h5py.SoftLink):
            itemtype = "soft link"
        elif isinstance(obj, h5py.ExternalLink):
            itemtype = "external link"
        # hard link type = target type (Group or Dataset)
        elif isinstance(obj, group_types):
            itemtype = "group"
        else:
            itemtype = "dataset"

        self.basename = text
        """Name of the item: base filename, group name, or dataset name"""

        self.itemtype = itemtype
        """Type of item: 'file', 'group', 'dataset', 'soft link' or
        'external link'

        For hard links, the type is the type of the target item.
        :class:`SpecH5LinkToGroup` and :class:`SpecH5LinkToDataset`
        items are considered to be hard links."""

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

        self.hdf5name = self.obj.name
        """Name of group or dataset within the HDF5 file."""

        # store owned items
        self._item_type = self._createTypeItem()
        self._item_description = self._createDescriptionItem()

        self._setDefaultTooltip()

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
            tooltip = self._htmlFromDict(attrs)
        else:
            tooltip = ""

        self.setToolTip(tooltip)

    def appendRow(self, list):
        if isinstance(list, Hdf5Item):
            list = list._createRow()
        super(Hdf5Item, self).appendRow(list)

    def _createRow(self):
        """Create the row where the first item is self and other
        items are content of columns"""
        return [self, self._description_type, self._item_type]

    def _htmlFromDict(self, input):
        """Generate a readable HTML from a dictionary

        :param input dict: A Dictionary
        :rtype: str
        """
        result = "<html><ul>"
        for key, value in input.items():
            result += "<li><b>%s</b>: %s</li>" % (key, value)
        result += "</ul></html>"
        return result

    def _createTypeItem(self):
        """Create the item holding the type column"""
        if self.itemtype == "dataset":
            if self.obj.dtype.type == numpy.string_:
                text = "string"
            else:
                text = str(self.obj.dtype)

            for axes in self.obj.shape:
                text += u" \u00D7 " + unicode(axes)
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

        errmsg = "Cannot find parent. Hdf5Item %s " % self.basename
        errmsg += "is not in a valid Hdf5TreeModel"
        if parent is None:
            raise AttributeError(errmsg)

        while parent.itemtype != "file":
            parent = parent.parent()
            if parent is None:
                raise AttributeError(errmsg)

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
    def __init__(self, files=None):
        """
        :param files: List of file handles/descriptors for a :class:`h5py.File`
            or a  :class:`spech5.SpecH5` object, or list of file pathes.
        """
        super(Hdf5TreeModel, self).__init__()
        self.setHorizontalHeaderLabels(['Name', 'Description', 'Type'])

        if files is not None:
            for file_ in files:
                self.load(file_)

    def appendRow(self, list):
        # FIXME it would be better to generate a self invisibleItem, but it looks to be impossible
        if isinstance(list, Hdf5Item):
            list = list._createRow()
        super(Hdf5TreeModel, self).appendRow(list)

    def load(self, file_):
        """Load a HDF5 file into the data model.

        :param file_: File handle/descriptor for a :class:`h5py.File`
            or a  :class:`spech5.SpecH5` object, or file path.
        """
        if isinstance(file_, (h5py.File, spech5.SpecH5)):
            filename = file_.filename
            fd = file_
        else:
            if not os.path.isfile(file_):
                raise IOError("Parameter file_ must be a file path, " +
                              "or a h5py.File object, " +
                              "or a spech5.SpecH5 object.")
            filename = file_
            try:
                if is_hdf5_file(file_):
                    fd = h5py.File(file_)
                else:
                    # assume Specfile
                    fd = spech5.SpecH5(file_)
            except IOError:
                # replace error message
                raise IOError("Parameter file_ must be a file path, " +
                              "or a h5py.File object, " +
                              "or a spech5.SpecH5 object.")

        # add root level row with file name
        file_item = Hdf5Item(text=os.path.basename(filename), obj=fd)
        file_item.setToolTip("File <%s>" % os.path.abspath(filename))
        self.appendRow(file_item)

        # fill HDF5 structure tree underneath file
        self._recursive_append_rows(h5item=file_item,
                                    gr_or_ds=fd,
                                    filename=os.path.abspath(filename))

    def _recursive_append_rows(self, h5item, gr_or_ds, filename):
        """Recurse through an HDF5 structure to append groups an datasets
        into the tree model.
        :param h5item: Parent :class:`Hdf5Item` or
            :class:`Hdf5ItemModel` object
        :param gr_or_ds: h5py or spech5 object (h5py.File, h5py.Group,
            h5py.Dataset, spech5.SpecH5, spech5.SpecH5Group,
            spech5.SpecH5Dataset)
        """

        group_types = (h5py.File, h5py.Group,
                       spech5.SpecH5, spech5.SpecH5Group,
                       spech5.SpecH5LinkToGroup  # ~ hard link
                       )

        if isinstance(gr_or_ds, group_types):
            for child_gr_ds_name in gr_or_ds:
                child_gr_ds = gr_or_ds[child_gr_ds_name]

                # actual item
                child_h5item = Hdf5Item(text=child_gr_ds_name, obj=child_gr_ds)
                h5item.appendRow(child_h5item)
                self._recursive_append_rows(child_h5item,
                                            child_gr_ds,
                                            filename)


class MyTreeView(qt.QTreeView):
    """QTreeView with column width auto-resizing and additional
    signals
    """
    enterKeyPressed = qt.pyqtSignal()

    def __init__(self, parent=None, auto_resize=True, model=None):
        qt.QTreeView.__init__(self, parent)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setUniformRowHeights(True)

        self.lastMouse = None
        if auto_resize:
            self.expanded.connect(self.resizeAllColumns)
            self.collapsed.connect(self.resizeAllColumns)

        if model is not None:
            self.setModel(model)

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
        button = event.button()
        if button == qt.Qt.LeftButton:
            self.lastMouse = "left"
        elif button == qt.Qt.RightButton:
            self.lastMouse = "right"
        elif button == qt.Qt.MidButton:
            self.lastMouse = "middle"
        else:
            self.lastMouse = "????"
        qt.QTreeView.mousePressEvent(self, event)
        if self.lastMouse != "left":
            # Qt5 only sends itemClicked on left button mouse click
            if qt.qVersion() > "5":
                event = "itemClicked"
                qindex = self.indexAt(event.pos())
                self.emitSignal(event, qindex)


class Hdf5TreeView(qt.QWidget):
    """
    This widget provides a tree view of one or several HDF5 files,
    with two columns *Name* and *Description*.

    When hovering the mouse cursor over the name column, you get a tooltip
    with a complete name.

    The columns automatically resize themselves to the needed width when
    expanding or collapsing a group.
    """
    sigHdf5TreeView = qt.pyqtSignal(object)
    """Signal emitted when clicking or pressing the ``Enter`` key. It
    broadcasts a dictionary of information about the event and the
    selected item.

    Dictionary keys:

    - ``event``: "itemClicked", "itemDoubleClicked",
            or "itemEnterKeyPressed"
    - ``filename``: name of HDF5 or Spec file
    - ``name``: path within the HDF5 structure
    - ``dtype``: dataset dtype, None if item is a group
    - ``shape``: dataset shape, None if item is a group
    - ``attr``: attributes dictionary of element
    """
    def __init__(self, parent=None, files_=None):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.model = Hdf5TreeModel(files_)
        """:class:`Hdf5TreeModel` in charge of loading and storing
        the HDF5 data structure"""

        self.treeview = MyTreeView(model=self.model)
        """Tree view widget displaying :attr:`model`"""
        layout.addWidget(self.treeview)

        # connect events to handler methods
        self.treeview.clicked.connect(self.itemClicked)
        self.treeview.doubleClicked.connect(self.itemDoubleClicked)
        self.treeview.enterKeyPressed.connect(self.itemEnterKeyPressed)

    def load(self, file_):
        """
        :param file_: HDF5 or Spec file (path or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instance)
        """
        self.model.load(file_)

    def itemClicked(self, modelIndex):
        """
        :param modelIndex: Index within the :class:`Hdf5TreeModel` of the
                           clicked item.
        :type modelIndex: :class:`qt.QModelIndex`
        """
        event = "itemClicked"
        self.emitSignal(event, modelIndex)

    def itemDoubleClicked(self, modelIndex):
        """
        :param modelIndex: Index within the :class:`Hdf5TreeModel` of the
                           clicked item.
        :type modelIndex: :class:`qt.QModelIndex`
        """
        event = "itemDoubleClicked"
        self.emitSignal(event, modelIndex)

    def itemEnterKeyPressed(self):
        """
        """
        event = "itemEnterKeyPressed"
        modelIndex = self.treeview.selectedIndexes()[0]
        self.emitSignal(event, modelIndex)

    def emitSignal(self, event, qindex):
        """
        Emits a ``sigHdf5TreeView`` signal to broadcast a dictionary of
        information about the selected row in the tree view.

        :param event: Type of event: "itemClicked", "itemDoubleClicked",
            or "itemEnterKeyPressed"
        :type event: string
        :param qindex: Index within the :class:`Hdf5TreeModel` of the
                           selected item.
        :type qindex: :class:`qt.QModelIndex`

        """
        # when selecting a row, we are interested in the first column
        # item, which has the pointer to the group/dataset
        this_row = qindex.row()
        if qindex.column() != 0:
            qindex = qindex.sibling(this_row, 0)

        item = self.model.itemFromIndex(qindex)

        if not "Clicked" in event:
            mouse_button = None
        else:
            mouse_button = self.treeview.lastMouse * 1

        ddict = {
            'event': event,
            'filename': item.filename,
            'basename': item.basename,
            'hdf5name': item.hdf5name,
            'mouse': mouse_button,
            'obj': item.obj,
            'dtype': getattr(item.obj, "dtype", None),
            'shape': getattr(item.obj, "shape", None),
            'attrs': getattr(item.obj, "attrs", None)
        }

        # FIXME: Maybe emit only {event, obj}
        self.sigHdf5TreeView.emit(ddict)


def main(filenames):
    """
    :param filenames: list of file paths
    """
    app = qt.QApplication([])

    view = Hdf5TreeView(files_=filenames)

    def my_slot(ddict):
        print(ddict)

    view.sigHdf5TreeView.connect(my_slot)
    view.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv[1:])

