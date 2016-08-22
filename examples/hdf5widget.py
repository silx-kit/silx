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

"""Qt Hdf5 widget examples

.. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
    library, which is not a mandatory dependency for `silx`. You might need
    to install it if you don't already have it.
"""

import os
import sys
import numpy
import logging
from silx.gui import qt
from silx.gui import hdf5widget


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

        self.model = hdf5widget.Hdf5TreeModel(files_)
        """:class:`Hdf5TreeModel` in charge of loading and storing
        the HDF5 data structure"""

        self.treeview = hdf5widget.Hdf5TreeView(model=self.model)
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
        if not isinstance(item, hdf5widget.Hdf5Item):
            return

        if "Clicked" in event:
            button = self.treeview.lastMouse
            if button == qt.Qt.LeftButton:
                mouse_button = "left"
            elif button == qt.Qt.RightButton:
                mouse_button = "right"
            elif button == qt.Qt.MidButton:
                mouse_button = "middle"
            else:
                mouse_button = "????"
        else:
            mouse_button = None

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
