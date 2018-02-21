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
"""Wrapper module for the `concurrent.futures` package.

The `concurrent.futures` package is available in python>=3.2.

For Python2, it tries to fill this module with the local copy
of `concurrent.futures` if it exists.
Otherwise it expects to have the `concurrent.futures` packaged
installed in the Python path.
For Python3, it uses the module from Python.

It should be used like that:

.. code-block::

    from silx.third_party import concurrent_futures

"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/02/2018"

import sys

if sys.version_info < (3,):  # Use Python 2 backport
    try:
        # try to import the local version
        from ._local.concurrent_futures import *  # noqa
    except ImportError:
        # else try to import it from the python path
        from concurrent.futures import *  # noqa
else:  # Use Python 3 standard library
    from concurrent.futures import *  # noqa
