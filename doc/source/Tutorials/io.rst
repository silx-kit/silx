
Getting started with silx.io
============================

This tutorial explains how to read data files using the :meth:`silx.io.open` function.

The target audience are developers without knowledge of the *h5py* library, and how
to use it to read HDF5 files.

If you are already familiar with *h5py*, you probably just need to know that
the :meth:`silx.io.open` function returns objects that mimic *h5py* file objects,
and that the main supported file formats are:

  - HDF5
  - all formats supported by the *FabIO* library
  - SPEC data files

Knowledge about the python *dictionary* type and the numpy *ndarray* type
are prerequisites for this tutorial.


Background
----------

In the past, it was necessary to learn multiple libraries to read multiple
data formats.

The library *FabIO* was designed to read images in many formats, but not to read
more heterogeneous formats, such as *HDF5* or *SPEC*.

To read *SPEC* data files in Python, a common solution was to use the module
:mod:`PyMca5.PyMcaIO.specfilewrapper`.

Regarding HDF5 files, the de-facto standard for reading them in Python is to
use the *h5py* library.

*silx* tries to address this situation by providing a unified way to read all
data formats supported at the ESRF.

Today, HDF5 is the preffered format to store data in many scientific institutions,
including most synchrotrons. So it was decided to provide tools for reading data
that mimic the *h5py* library's API.


Definitions
-----------

HDF5
++++

The *HDF5* format is a *hierarchical data format*, designed to store and
organize large amounts of data.

A HDF5 file contains a number of *datasets*, which are multidimensional arrays
of a homogeneous type.

These datasets are stored in container structures
called *groups*. Groups can also be stored in other groups, allowing to
define a hierarchical tree structure.

Both datasets and groups may have *attributes* attached to them. Attributes are
used to document the object. They are similar to datasets in several ways
(data container of homogeneous type), but they are typically much smaller.

It is a common analogy to compare a HDF5 file with a filesystem.
Groups are analogous to directories, while datasets are analogous to files,
and attributes are analogous to file metadata (creation date, last modification...).

.. todo: add an image to illustrate the filesystem analogy


h5py
++++

The *h5py* library is a Pythonic interface to the `HDF5`_ binary data format.

It exposes an HDF5 group as a python object that resembles a python
dictionary, and an HDF5 dataset or attribute as an object that resembles a
numpy array.

API description
---------------

All three main objects, File, Group and Dataset, share the following attributes:

 - :attr:`attrs`: Attributes, as a dictionary of metadata for the group or dataset.
 - :attr:`basename`: String giving the basename of this group or dataset.
 - :attr:`name`: String giving the full path to this group or dataset, relative
   to the root group (file).
 - :attr:`file`: File object at the root of the tree structure containing this
   group or dataset.
 - :attr:`parent`: Group object containing this group or dataset.

File object
+++++++++++

The API of the file objects returned by the :meth:`silx.io.open`
function tries to be as close as possible to the API of the :class:`h5py.File`
objects used to read HDF5 data.

A h5py file is a group with just a few extra attributes and methods.

The objects defined in `silx.io` implement a subset of these attributes and methods:

 - :attr:`filename`: Name of the file on disk.
 - :attr:`mode`: String indicating if the file is open in read mode ("r")
   or write mode ("w"). :meth:`silx.io.open` always returns objects in read mode.
 - :meth:`close`: Close this file. All open objects will become invalid.

The :attr:`parent` of a file is `None`, and its :attr:`name` is an empty string.

Group object
++++++++++++

Group objects behave much like python dictionaries.

You can iterate over a group's :meth:`keys`, which are the names of the objects
encapsulated by the group (datasets and sub-groups). The :meth:`values` method
returns an iterator over the encapsulated objects. The :meth:`items` method returns
an iterator over `(name, value)` pairs.

Groups provide a :meth:`get` method that retrieves an item, or information about an item.
Like standard python dictionaries, a `default` parameter can be used to specify
a value to be returned if the given name is not a member of the group.

Two methods are provided recursively visit all members of a group, :meth:`visit`
and :meth:`visititems`. The former takes as argument a *callable* with the signature
`callable(name) -> None or return value`. The latter  takes as argument a *callable*
with the signature `callable(name, object) -> None or return value` (`object` being a
a group or dataset instance.)




Additional resources
--------------------

- `h5py documentation <http://docs.h5py.org/en/latest/>`_
- `Formats supported by FabIO <http://www.silx.org/doc/fabio/dev/getting_started.html#list-of-file-formats-that-fabio-can-read-and-write>`_
- `Spec file h5py-like structure <http://www.silx.org/doc/silx/dev/modules/io/spech5.html#api-description>`_
- `HDF5 format documentation <https://support.hdfgroup.org/HDF5/>`_
