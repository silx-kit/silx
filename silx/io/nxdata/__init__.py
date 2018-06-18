# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""
:mod:`nxdata`: NXdata parsing and validation
--------------------------------------------

To parse an existing NXdata group, use :class:`NXdata`.

Following functions help you check the validity of a existing NXdata group:
 - :func:`is_valid_nxdata`
 - :func:`is_NXentry_with_default_NXdata`
 - :func:`is_NXroot_with_default_NXdata`

To help you write a NXdata group, you can use :func:`save_NXdata`.

.. currentmodule:: silx.io.nxdata

Classes
+++++++

.. autoclass:: NXdata
    :members:


Functions
+++++++++

.. autofunction:: get_default

.. autofunction:: is_valid_nxdata

.. autofunction:: is_NXentry_with_default_NXdata

.. autofunction:: is_NXroot_with_default_NXdata

.. autofunction:: save_NXdata

"""
from .parse import NXdata, get_default, is_valid_nxdata, InvalidNXdataError, \
    is_NXentry_with_default_NXdata, is_NXroot_with_default_NXdata
from ._utils import get_attr_as_unicode, get_attr_as_string, nxdata_logger
from .write import save_NXdata
