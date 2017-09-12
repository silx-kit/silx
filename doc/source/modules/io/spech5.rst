
.. currentmodule:: silx.io

:mod:`spech5`: h5py-like API to SpecFile
-----------------------------------------

.. automodule:: silx.io.spech5


Classes
+++++++

- :class:`SpecH5`
- :class:`SpecH5Group`
- :class:`SpecH5Dataset`

.. autoclass:: SpecH5
    :members:
    :show-inheritance:
    :undoc-members:
    :inherited-members: name, basename, attrs, h5py_class, parent,
        get, keys, values, items,
    :special-members: __getitem__, __len__, __contains__, __enter__, __exit__, __iter__
    :exclude-members: add_node

.. autoclass:: SpecH5Group
    :show-inheritance:

.. autoclass:: silx.io.commonh5.Group
    :show-inheritance:
    :undoc-members:
    :members: name, basename, file, attrs, h5py_class, parent,
        get, keys, values, items, visit, visititems
    :special-members: __getitem__, __len__, __contains__, __iter__
    :exclude-members: add_node

.. autoclass:: SpecH5Dataset
    :show-inheritance:

.. autoclass:: SpecH5NodeDataset
    :members:
    :show-inheritance:
    :undoc-members:
    :inherited-members:
    :special-members: __getitem__, __len__, __iter__, __getattr__
