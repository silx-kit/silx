.. currentmodule:: silx.gui.hdf5

Getting started with HDF5 widgets
=================================

Silx provides an implementation of a tree model and a tree view for HDF5 files.
The aim of this tree is to provide a convenient read-only widget for a big
amount of data and supporting file formats often used in synchrotrons.

This page provides some source code to show how to use this widget.

Commented source code
---------------------

Import and create your tree view
++++++++++++++++++++++++++++++++

HDF5 widgets are all exposed by the package `silx.gui.hdf5`.

.. code-block:: python

   import silx.gui.hdf5
   treeview = silx.gui.hdf5.Hdf5TreeView()

Custom your tree view
+++++++++++++++++++++

The tree view can be customized to be sorted by default.

.. code-block:: python

   # Sort content of files by time or name
   treeview.setSortingEnabled(True)

The model can be customized to support mouse interaction.
A convenient method :meth:`Hdf5TreeView.findHdf5TreeModel` returns the main
HDF5 model used through proxy models.

.. code-block:: python

   model = treeview.findHdf5TreeModel()

   # Avoid the user to drop file in the widget
   model.setFileDropEnabled(False)

   # Allow the user to reorder files with drag-and-drop
   model.setFileMoveEnabled(True)

The tree view is also provided with a custom header which help to choose
visible columns.

.. code-block:: python

   header = treeview.header()

   # Select displayed columns
   column_ids = [treeview.findHdf5TreeModel().NAME_COLUMN]
   header.setSections(column_ids)

   # Do not allow the user to custom visible columns
   header.setEnableHideColumnsPopup(False)

Add a file by name
++++++++++++++++++

The model can be used to add HDF5. It is internally using
:func:`silx.io.utils.load`.

.. code-block:: python

   model.insertFile("test.h5")

Add a file with h5py
++++++++++++++++++++

The model internally uses :mod:`h5py` object API. We can use h5py file, group
and dataset as it is.

.. code-block:: python

   import h5py
   h5 = h5py.File("test.py")
   
   # We can use file
   model.insertH5pyObject(h5)

   # or group or dataset
   model.insertH5pyObject(h5["group1"])
   model.insertH5pyObject(h5["group1/dataset50"])

Add a file with silx
++++++++++++++++++++

Silx also provides an input API. It supports HDF5 files through :mod:`h5py`.

.. code-block:: python

   from silx.io.utils.load import load

   # We can load HDF5 files
   model.insertH5pyObject(load("test.h5"))

   # or Spec files
   model.insertH5pyObject(load("test.dat"))


Custom context menu
+++++++++++++++++++

The :class:`Hdf5TreeView` provides a callback API to populate the context menu.
The callback receives a :class:`Hdf5ContextMenuEvent` every time the user
requests the context menu. The event contains :class:`H5Node` objects which wrap
h5py objects with extra information.

.. code-block:: python

   def my_action_callback(obj):
      # do what you want

   def my_callback(event):
      objects = event.source().selectedH5Nodes()
      obj = objects[0]  # for single selection

      if obj.ntype is h5py.Dataset:
         action = qt.QAction("My funky action on datasets only")
         action.triggered.connect(lambda: my_action_callback(obj))
         event.menu().addAction(action)

   treeview.addContextMenuCallback(my_callback)

Capture selection
+++++++++++++++++

The :class:`Hdf5TreeView` widget provides default Qt signals inherited from
`QAbstractItemView`.

- `activated`:
      This signal is emitted when the item specified by index is
      activated by the user. How to activate items depends on the platform;
      e.g., by single- or double-clicking the item, or by pressing the
      Return or Enter key when the item is current.
- `clicked`:
      This signal is emitted when a mouse button is clicked. The item the mouse
      was clicked on is specified by index. The signal is only emitted when the
      index is valid.
- `doubleClicked`:
      This signal is emitted when a mouse button is double-clicked. The item
      the mouse was double-clicked on is specified by index. The signal is
      only emitted when the index is valid.
- `entered`:
      This signal is emitted when the mouse cursor enters the item specified by
      index. Mouse tracking needs to be enabled for this feature to work.
- `pressed`:
      This signal is emitted when a mouse button is pressed. The item the mouse
      was pressed on is specified by index. The signal is only emitted when the
      index is valid.

.. code-block:: python

   def my_callback(index):
       objects = treeview.selectedH5Nodes()
       obj = objects[0]  # for single selection

       print(obj)

       print(obj.basename)             # not provided by h5py
       print(obj.name)
       print(obj.file.filename)

       print(obj.local_basename)       # not provided by h5py
       print(obj.local_name)           # not provided by h5py
       print(obj.local_file.filename)  # not provided by h5py

       print(obj.attrs)

       if obj.ntype is h5py.Dataset:
           print(obj.dtype)
           print(obj.shape)
           print(obj.value)

   treeview.clicked.connect(my_callback)

Example
-------

.. toctree::
   :hidden:

   examples_hdf5widget.rst

The :doc:`examples_hdf5widget` sample code provides an example of properties of
the view, the model and the header.

.. image:: img/Hdf5Example.png
   :height: 200px
   :width: 400px
   :alt: Example for HDF5TreeView features
   :align: center

Source code: :doc:`examples_hdf5widget`.

After installing `silx` and downloading the script, you can start it from the
command prompt:

.. code-block:: bash

   python hdf5widget.py <files>

This example loads files added to the command line, or files dropped from the
file system. It also provides a GUI to display test files created
programmatically.
