.. currentmodule:: silx.gui

Getting started with hdf5 widgets
=================================

Commented source code
---------------------

Import and create your tree view
++++++++++++++++++++++++++++++++

.. code-block:: python

   import silx.gui.hdf5
   treeview = silx.gui.hdf5.Hdf5TreeView()

Custom your tree view
+++++++++++++++++++++

.. code-block:: python

   # Sort content of files by time or name
   treeview.setSortingEnabled(True)

   # Avoid the user to drop file in the widget
   treeview.findHdf5TreeModel().setFileDropEnabled(False)

   # Allow the user to reorder files with drag-and-drop
   treeview.findHdf5TreeModel().setFileMoveEnabled(True)

   # Only display few columns
   column_ids = [treeview.findHdf5TreeModel().NAME_COLUMN]
   treeview.header().setSections(column_ids)

   # Do not allow the user to custom visible columns
   treeview.header().setEnableHideColumnsPopup(False)

Add a file by name
++++++++++++++++++

The tree view use the model througt a proxy. It is better to use
`findHdf5TreeModel`.

.. code-block:: python

   treeview.findHdf5TreeModel().insertFile("test.h5")

Add a file with h5py
++++++++++++++++++++

.. code-block:: python

   import h5py
   h5 = h5py.File("test.py")
   treeview.findHdf5TreeModel().insertH5pyObject(h5)

   # You also can add a dataset or a group
   treeview.findHdf5TreeModel().insertH5pyObject(h5["group1"])

Custom the context menu
+++++++++++++++++++++++

The Hdf5TreeView provides a callback API to populate context menu. The callback
receive an :class:`silx.gui.hdf5.Hdf5ContextMenuEvent` everytime the user
request the context menu. The event contains :class:`silx.gui.hdf5.H5Node`
objects which wrap h5py objects with extra-informations.

.. code-block:: python

   def my_action_callback(obj):
      # do what you want

   def my_callback(event):
      objects = event.source().selectedH5Nodes()
      obj = objects[0] # single selection

      if obj.ntype is h5py.Dataset:
         action = qt.QAction("My funky action on datasets only")
         action.triggered.connect(lambda: my_action_callback(obj))
         event.menu().addAction(action)

   treeview.addContextMenuCallback(my_callback)

Capture selection
+++++++++++++++++

The widget provides default Qt events, `activated`, `clicked`,
`doubleClicked`, `entered`, `pressed`.

.. code-block:: python

   def my_callback(index):
       objects = treeview.selectedH5Nodes()
       obj = objects[0]

       print(obj)

       print(obj.basename)             # not provided by h5py
       print(obj.name)
       print(obj.file.filename)

       print(obj.local_file.filename)  # not provided by h5py
       print(obj.local_basename)       # not provided by h5py
       print(obj.local_name)           # not provided by h5py

       print(obj.attrs)

       if obj.ntype is h5py.Dataset:
           print(obj.dtype)
           print(obj.shape)
           print(obj.value)

   treeview.clicked.connect(my_callback)

Example
-------

Silx source code provide an example to play with properties of the view, the
model and the header.

.. image:: img/Hdf5Example.png
   :height: 200px
   :width: 400 px
   :alt: Example for HDF5TreeView features
   :align: center

- https://github.com/silx-kit/silx/blob/master/examples/hdf5widget.py

After installing silx. You can use it like that.

.. code-block:: bash

   python examples/hdf5widget.py <files>

This example support to load files added to the command line, or files dropped
from the file system. It also provide a GUI to display few test files created
programatically.
