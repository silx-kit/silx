#!/usr/bin/env python 
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------
# Copyright (c) 2014, NeXpy Development Team.
#
# Author: Paul Kienzle, Ray Osborn
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING, distributed with this software.
#-----------------------------------------------------------------------------

"""
Python NeXus interface.

NeXus is a common data format for neutron, Xray and muon science.
The files contain multidimensional data elements grouped into a
hierarchical structure.  The data sets are self-describing, with
a description of the instrument configuration including the units
used as well as the data measured.

The NeXus file interface requires compiled libraries to read the
underlying HDF or XML files.  Binary packages are available for some
platforms from the NeXus site.  Details of where the nexus package
searches for the libraries are recorded in `nexus.napi`.

Example
=======

First we need to load the file structure::

    import nexus
    f = nexus.load('file.nxs')

We can examine the file structure using a number of commands::

    f.attrs             # Shows file name, date, user, and NeXus version
    f.tree()            # Lists the entire contents of the NeXus file
    f.NXentry             # Shows the list of datasets in the file
    f.NXentry[0].dir()  # Lists the fields in the first entry

Some files can even be plotted automatically::

    f.NXentry[0].data.plot()

We can create a copy of the file using write::

    nexus.save('copy.nxs', tree)

For a complete description of the features available in this tree view
of the NeXus data file, see `nexus.tree`.

NeXus API
=========

When converting code to python from other languages you do not
necessarily want to rewrite the file handling code using the
tree view.  The `nexus.napi` module provides an interface which
more closely follows the NeXus application programming
interface (NAPI_).

.. _Nexus: http://www.nexusformat.org/Introduction
.. _NAPI:  http://www.nexusformat.org/Application_Program_Interface
.. _HDF:   http://www.hdfgroup.org
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .tree import *
try:
    import h5pyd
    from .remote import *
except ImportError:
    pass
