# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This module contains wrapper from file format to h5py. The exposed layout is
as close as possible to the original file format.
"""
import numpy
from . import commonh5
import logging

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "21/09/2017"


_logger = logging.getLogger(__name__)


class _FreeDataset(commonh5.Dataset):

    def _check_data(self, data):
        """Release the constriants checked on types cause we can reach more
        types than the one available on h5py, and it is not supposed to be
        converted into h5py."""
        chartype = data.dtype.char
        if chartype in ["U", "O"]:
            msg = "Dataset '%s' uses an unsupported type '%s'."
            msg = msg % (self.name, data.dtype)
            _logger.warning(msg)


class NumpyFile(commonh5.File):
    """
    Expose a numpy file `npy`, or `npz` as an h5py.File-like.

    :param str name: Filename to load
    """
    def __init__(self, name=None):
        commonh5.File.__init__(self, name=name, mode="w")
        np_file = numpy.load(name)
        if hasattr(np_file, "close"):
            # For npz (created using  by numpy.savez, numpy.savez_compressed)
            for key, value in np_file.items():
                self[key] = _FreeDataset(None, data=value)
            np_file.close()
        else:
            # For npy (created using numpy.save)
            value = np_file
            dataset = _FreeDataset("data", data=value)
            self.add_node(dataset)
