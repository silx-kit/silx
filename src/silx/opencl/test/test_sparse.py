#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""Test of the sparse module"""

import numpy as np
import unittest
import logging
from itertools import product
from ..common import ocl
if ocl:
    import pyopencl.array as parray
    from silx.opencl.sparse import CSR
try:
    import scipy.sparse as sp
except ImportError:
    sp = None
logger = logging.getLogger(__name__)



def generate_sparse_random_data(
    shape=(1000,),
    data_min=0, data_max=100,
    density=0.1,
    use_only_integers=True,
    dtype="f"):
    """
    Generate random sparse data where.

    Parameters
    ------------
    shape: tuple
        Output data shape.
    data_min: int or float
        Minimum value of data
    data_max: int or float
        Maximum value of data
    density: float
        Density of non-zero elements in the output data.
        Low value of density mean low number of non-zero elements.
    use_only_integers: bool
        If set to True, the output data items will be primarily integers,
        possibly casted to float if dtype is a floating-point type.
        This can be used for ease of debugging.
    dtype: str or numpy.dtype
        Output data type
    """
    mask = np.random.binomial(1, density, size=shape)
    if use_only_integers:
        d = np.random.randint(data_min, high=data_max, size=shape)
    else:
        d = data_min + (data_max - data_min) * np.random.rand(*shape)
    return (d * mask).astype(dtype)



@unittest.skipUnless(ocl and sp, "PyOpenCl/scipy is missing")
class TestCSR(unittest.TestCase):
    """Test CSR format"""

    def setUp(self):
        # Test possible configurations
        input_on_device = [False, True]
        output_on_device = [False, True]
        dtypes = [np.float32, np.int32, np.uint16]
        self._test_configs = list(product(input_on_device, output_on_device, dtypes))


    def compute_ref_sparsification(self, array):
        ref_sparse = sp.csr_matrix(array)
        return ref_sparse


    def test_sparsification(self):
        for input_on_device, output_on_device, dtype in self._test_configs:
            self._test_sparsification(input_on_device, output_on_device, dtype)


    def _test_sparsification(self, input_on_device, output_on_device, dtype):
        current_config = "input on device: %s, output on device: %s, dtype: %s" % (
            str(input_on_device), str(output_on_device), str(dtype)
        )
        logger.debug("CSR: %s" % current_config)
        # Generate data and reference CSR
        array = generate_sparse_random_data(shape=(512, 511), dtype=dtype)
        ref_sparse = self.compute_ref_sparsification(array)
        # Sparsify on device
        csr = CSR(array.shape, dtype=dtype)
        if input_on_device:
            # The array has to be flattened
            arr = parray.to_device(csr.queue, array.ravel())
        else:
            arr = array
        if output_on_device:
            d_data = parray.empty_like(csr.data)
            d_indices = parray.empty_like(csr.indices)
            d_indptr = parray.empty_like(csr.indptr)
            d_data.fill(0)
            d_indices.fill(0)
            d_indptr.fill(0)
            output = (d_data, d_indices, d_indptr)
        else:
            output = None
        data, indices, indptr = csr.sparsify(arr, output=output)
        if output_on_device:
            data = data.get()
            indices = indices.get()
            indptr = indptr.get()
        # Compare
        nnz = ref_sparse.nnz
        self.assertTrue(
            np.allclose(data[:nnz], ref_sparse.data),
            "something wrong with sparsified data (%s)"
            % current_config
        )
        self.assertTrue(
            np.allclose(indices[:nnz], ref_sparse.indices),
            "something wrong with sparsified indices (%s)"
            % current_config
        )
        self.assertTrue(
            np.allclose(indptr, ref_sparse.indptr),
            "something wrong with sparsified indices pointers (indptr) (%s)"
            % current_config
        )


    def test_desparsification(self):
        for input_on_device, output_on_device, dtype in self._test_configs:
            self._test_desparsification(input_on_device, output_on_device, dtype)


    def _test_desparsification(self, input_on_device, output_on_device, dtype):
        current_config = "input on device: %s, output on device: %s, dtype: %s" % (
            str(input_on_device), str(output_on_device), str(dtype)
        )
        logger.debug("CSR: %s" % current_config)
        # Generate data and reference CSR
        array = generate_sparse_random_data(shape=(512, 511), dtype=dtype)
        ref_sparse = self.compute_ref_sparsification(array)
        # De-sparsify on device
        csr = CSR(array.shape, dtype=dtype, max_nnz=ref_sparse.nnz)
        if input_on_device:
            data = parray.to_device(csr.queue, ref_sparse.data)
            indices = parray.to_device(csr.queue, ref_sparse.indices)
            indptr = parray.to_device(csr.queue, ref_sparse.indptr)
        else:
            data = ref_sparse.data
            indices = ref_sparse.indices
            indptr = ref_sparse.indptr
        if output_on_device:
            d_arr = parray.empty_like(csr.array)
            d_arr.fill(0)
            output = d_arr
        else:
            output = None
        arr = csr.densify(data, indices, indptr, output=output)
        if output_on_device:
            arr = arr.get()
        # Compare
        self.assertTrue(
            np.allclose(arr.reshape(array.shape), array),
            "something wrong with densified data (%s)"
            % current_config
        )
