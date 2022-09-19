# /*##########################################################################
# Copyright (C) 2022 European Synchrotron Radiation Facility
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
"""Provides a wrapper to expose a dataset slice as a `commonh5.Dataset`."""
from typing import Union
from . import commonh5
from . import utils

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "12/09/2022"


class DatasetSlice(commonh5.LazyLoadableDataset):
    """Lazy-loaded wrapper of a dataset's slice.

    :param h5file: h5py-like file containing the dataset
    :param dataset: h5py-like dataset from which to access a slice
    """
    def __init__(self, name: str, h5_file, dataset, data_slice: Union[int, slice]):
        if not utils.is_file(h5_file):
            raise ValueError(f"Unsupported h5_file '{h5_file}'")
        if not utils.is_dataset(dataset):
            raise ValueError(f"Unsupported dataset '{dataset}'")

        self.__file = h5_file
        self.__dataset = dataset
        self.__data_slice = data_slice
        super().__init__(name)

    def _create_data(self):
        return self.__dataset[self.__data_slice]

    @property
    def file(self):
        return self.__file

    @property
    def name(self):
        return self.basename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the file"""
        self.__file.close()
        self.__file = None
