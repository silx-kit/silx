. currentmodule:: silx.io

:mod:`commonh5`: Helpers for writing h5py-like API
--------------------------------------------------

.. automodule:: silx.io.commonh5

Classes
+++++++

.. autoclass:: Node
   :members:

.. autoclass:: File
   :show-inheritance:
   :members:

.. autoclass:: Group
    :show-inheritance:
    :undoc-members:
    :members: name, basename, file, attrs, h5py_class, parent,
        get, keys, values, items, visit, visititems
    :special-members: __getitem__, __len__, __contains__, __iter__
    :exclude-members: add_node

.. autoclass:: Dataset
   :show-inheritance:
   :members:
