
Converting various data files to HDF5
=====================================

This document explains how to convert SPEC files, EDF files and various other data
formats into HDF5 files.

An understanding of the way these data formats are exposed by the :meth:`silx.io.open`
function is a prerequisite for this tutorial. You can learn more about this subject by
reading :doc:`io`.

Using the convert module
++++++++++++++++++++++++

The *silx* module :mod:`silx.io.convert` can be used to convert various data files into a
HDF5 file with the same structure as the one exposed by the :mod:`spech5` or :mod:`fabioh5` modules.

.. code-block:: python

    from silx.io.convert import convert

    convert("/home/pierre/myspecfile.dat", "myfile.h5")


You can then read the file with any HDF5 reader.


The function :func:`silx.io.convert.convert` is a simplified version of a
more flexible function :func:`silx.io.convert.write_to_h5`.

The latter allows you to write scans into a specific HDF5 group in the output directory.
You can also decide whether you want to overwrite an existing file, or append data to it.
You can specify whether existing data with the same name as input data should be overwritten
or ignored.

This allows you to repeatedly transfer new content of a SPEC file to an existing HDF5 file, in between
two scans.

The following script is an example of a command line interface to :func:`write_to_h5`.

.. literalinclude:: ../../../examples/writetoh5.py
   :lines: 44-

But the functionality implemented in this script (and much more) is already implemented
in the *silx convert* application.


Using the convert application
+++++++++++++++++++++++++++++

.. versionadded:: 0.6


*silx* also provides a ``silx convert`` command line application, which allows you to
perform standard conversions without having to write your own program.

Type ``silx convert --help`` in a terminal to see all available options.

.. note::

    The complete documentation for the *silx convert* command is available here:
    :doc:`../applications/convert`.

Converting single files
***********************

The simplest command to convert a single SPEC file to an HDF5 file would be::

    silx convert myspecfile.dat

As no output name is supplied, the output file name will be a timestamp with a *.h5* suffix
(e.g. *20180110-114930.h5*).

The following example allows you to append the content of a SPEC file to an
existing HDF5 file::

    silx convert myspecfile.dat -m a -o myhdf5file.h5

The ``-m a`` argument stands for *append mode*. The ``-o myhdf5file.h5``
argument is used to specify the output file name.

You could write the file into a specific group of the HDF5 file by providing
the complete URI in the format ``file_path::group_path``. For instance::

    silx convert myspecfile.dat -m a -o archive.h5::/2017-09-20/SPEC
