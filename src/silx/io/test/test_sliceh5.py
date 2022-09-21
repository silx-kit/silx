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
import contextlib
from io import BytesIO

import h5py
import numpy
import pytest

import silx.io
from silx.io import commonh5
from silx.io._sliceh5 import DatasetSlice, _combine_indices


@contextlib.contextmanager
def h5py_file(filename, mode):
    with BytesIO() as buffer:
        with h5py.File(buffer, mode) as h5file:
            yield h5file


@pytest.fixture(params=[commonh5.File, h5py_file])
def temp_h5file(request):
    temp_file_context = request.param
    with temp_file_context("tempfile.h5", "w") as h5file:
        yield h5file


@pytest.mark.parametrize("indices", [1, slice(None), (1, slice(1, 4))])
def test_datasetslice(temp_h5file, indices):
    data = numpy.arange(50).reshape(10, 5)
    ref_data = numpy.array(data[indices], copy=False)
    name = "/group/dataset"

    h5dataset = temp_h5file.create_group("group").create_dataset("dataset", data=data)

    with DatasetSlice(name, temp_h5file, h5dataset, indices) as dset:
        assert silx.io.is_dataset(dset)
        assert dset.file is temp_h5file
        assert dset.shape == ref_data.shape
        assert dset.size == ref_data.size
        assert dset.dtype == ref_data.dtype
        assert len(dset) == len(ref_data)
        assert numpy.array_equal(dset[()], ref_data)
        assert dset.name == name


@pytest.mark.parametrize(
    "shape,outer_indices,indices",
    [
        ((2, 5, 10), (-1, slice(None), slice(None)), slice(None)),
        ((2, 5, 10), (-1, slice(None), slice(None)), Ellipsis),
        # negative strides
        ((5, 10), (slice(1, 5, 2), slice(2, 8)), (slice(2, 3), slice(4, None, -2))),
        (
            (5, 10),
            (slice(4, None, -1), slice(9, 3, -2)),
            (slice(1, 3), slice(3, 0, -1)),
        ),
        ((5, 10), (slice(1, 8, 2), slice(None)), slice(2, 8)),  # slice overflow
    ],
)
def test_combine_indices(shape, outer_indices, indices):
    data = numpy.arange(numpy.prod(shape)).reshape(shape)
    ref_data = data[outer_indices][indices]

    combined_indices = _combine_indices(shape, outer_indices, indices)

    assert numpy.array_equal(data[combined_indices], ref_data)
