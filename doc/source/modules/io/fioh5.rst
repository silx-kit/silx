
.. currentmodule:: silx.io

:mod:`fioh5`: h5py-like API to FIO file
----------------------------------------

.. automodule:: silx.io.fioh5


Classes
+++++++

- :class:`FioH5`
- :class:`FioFile`

.. autoclass:: FioH5
    :members:
    :show-inheritance:
    :undoc-members:
    :inherited-members: name, basename, attrs, h5py_class, parent,
        get, keys, values, items,
    :special-members: __getitem__, __len__, __contains__, __enter__, __exit__, __iter__
    :exclude-members: add_node

.. autoclass:: FioFile

.. autofunction:: is_fiofile