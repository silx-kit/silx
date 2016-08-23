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
from fabio.eigerimage import h5py

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
import pprint
import html


class Hdf5TreeViewExample(qt.QMainWindow):
    """
    This window show an example of use of a Hdf5TreeView.

    The tree is initialized with a list of filenames. A panel allow to play
    with internal property configuration of the widget, and a text screen
    allow to display events.
    """

    def __init__(self, filenames=None):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx HDF5 widget example")

        self.__treeview = hdf5widget.Hdf5TreeView()
        """Silx HDF5 TreeView"""
        self.__text = qt.QTextEdit(self)
        """Widget displaying information"""

        tree_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(self.__treeview)
        layout.addWidget(self.createTreeViewConfigurationPanel(self, self.__treeview))
        tree_panel.setLayout(layout)

        spliter = qt.QSplitter()
        spliter.addWidget(tree_panel)
        spliter.addWidget(self.__text)
        spliter.setStretchFactor(1, 1)

        self.setCentralWidget(spliter)

        # append all files to the tree
        for file_name in filenames:
            self.__treeview.model().appendFile(file_name)

        self.__treeview.activated.connect(lambda index: self.displayEvent("activated", index))
        self.__treeview.clicked.connect(lambda index: self.displayEvent("clicked", index))
        self.__treeview.doubleClicked.connect(lambda index: self.displayEvent("doubleClicked", index))
        self.__treeview.entered.connect(lambda index: self.displayEvent("entered", index))
        self.__treeview.pressed.connect(lambda index: self.displayEvent("pressed", index))

    def displayEvent(self, eventName, index):

        def formatKey(name, value):
            name, value = html.escape(str(name)), html.escape(str(value))
            return "<li><b>%s</b>: %s</li>" % (name, value)

        text = "<html>"
        text += "<h1>Event</h1>"
        text += "<ul>"
        text += formatKey("name", eventName)
        text += formatKey("index", type(index))
        text += "</ul>"

        text += "<h1>Selected HDF5 objects</h1>"

        for h5py_obj in self.__treeview.selectedH5pyObjects():
            text += "<h2>HDF5 object</h2>"
            text += "<ul>"
            text += formatKey("filename", h5py_obj.file.filename)
            text += formatKey("basename", h5py_obj.name.split("/")[-1])
            text += formatKey("hdf5name", h5py_obj.name)
            text += formatKey("obj", type(h5py_obj))
            text += formatKey("dtype", getattr(h5py_obj, "dtype", None))
            text += formatKey("shape", getattr(h5py_obj, "shape", None))
            text += formatKey("attrs", getattr(h5py_obj, "attrs", None))
            if hasattr(h5py_obj, "attrs"):
                text += "<ul>"
                for key, value in h5py_obj.attrs.items():
                    text += formatKey(key, value)
                text += "</ul>"
            text += "</ul>"

        text += "</html>"
        self.__text.setHtml(text)

    def createTreeViewConfigurationPanel(self, parent, treeview):
        """Create a configuration panel to allow to play with widget states"""
        panel = qt.QGroupBox("Tree options", parent)
        layout = qt.QVBoxLayout()
        panel.setLayout(layout)

        autosize = qt.QCheckBox("Auto-size headers", panel)
        autosize.setChecked(treeview.header().hasAutoResizeColumns())
        autosize.toggled.connect(lambda: treeview.header().setAutoResizeColumns(autosize.isChecked()))
        layout.addWidget(autosize)

        multiselection = qt.QCheckBox("Multi-selection", panel)
        multiselection.setChecked(treeview.selectionMode() == qt.QAbstractItemView.MultiSelection)
        switch_selection = lambda: treeview.setSelectionMode(
                qt.QAbstractItemView.MultiSelection if multiselection.isChecked()
                else qt.QAbstractItemView.SingleSelection)
        multiselection.toggled.connect(switch_selection)
        layout.addWidget(multiselection)

        return panel


def main(filenames):
    """
    :param filenames: list of file paths
    """
    app = qt.QApplication([])
    window = Hdf5TreeViewExample(filenames)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv[1:])
