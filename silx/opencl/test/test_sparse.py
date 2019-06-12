#!/usr/bin/env python
# coding: utf-8
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
        self.array = generate_sparse_random_data(shape=(512, 511))
        # Compute reference sparsification
        a_s = sp.csr_matrix(self.array)
        self.ref_data = a_s.data
        self.ref_indices = a_s.indices
        self.ref_indptr = a_s.indptr
        self.ref_nnz = a_s.nnz
        # Test possible configurations
        input_on_device = [False, True]
        output_on_device = [False, True]
        self._test_configs = list(product(input_on_device, output_on_device))


    def test_sparsification(self):
        for input_on_device, output_on_device in self._test_configs:
            self._test_sparsification(input_on_device, output_on_device)


    def _test_sparsification(self, input_on_device, output_on_device):
        current_config = "input on device: %s, output on device: %s" % (
            str(input_on_device), str(output_on_device)
        )
        # Sparsify on device
        csr = CSR(self.array.shape)
        if input_on_device:
            # The array has to be flattened
            arr = parray.to_device(csr.queue, self.array.ravel())
        else:
            arr = self.array
        if output_on_device:
            d_data = parray.zeros_like(csr.data)
            d_indices = parray.zeros_like(csr.indices)
            d_indptr = parray.zeros_like(csr.indptr)
            output = (d_data, d_indices, d_indptr)
        else:
            output = None
        data, indices, indptr = csr.sparsify(arr, output=output)
        if output_on_device:
            data = data.get()
            indices = indices.get()
            indptr = indptr.get()
        # Compare
        nnz = self.ref_nnz
        self.assertTrue(
            np.allclose(data[:nnz], self.ref_data),
            "something wrong with sparsified data (%s)"
            % current_config
        )
        self.assertTrue(
            np.allclose(indices[:nnz], self.ref_indices),
            "something wrong with sparsified indices (%s)"
            % current_config
        )
        self.assertTrue(
            np.allclose(indptr, self.ref_indptr),
            "something wrong with sparsified indices pointers (indptr) (%s)"
            % current_config
        )


    def test_desparsification(self):
        for input_on_device, output_on_device in self._test_configs:
            self._test_desparsification(input_on_device, output_on_device)


    def _test_desparsification(self, input_on_device, output_on_device):
        current_config = "input on device: %s, output on device: %s" % (
            str(input_on_device), str(output_on_device)
        )
        # De-sparsify on device
        csr = CSR(self.array.shape, max_nnz=self.ref_nnz)
        if input_on_device:
            data = parray.to_device(csr.queue, self.ref_data)
            indices = parray.to_device(csr.queue, self.ref_indices)
            indptr = parray.to_device(csr.queue, self.ref_indptr)
        else:
            data = self.ref_data
            indices = self.ref_indices
            indptr = self.ref_indptr
        if output_on_device:
            d_arr = parray.zeros_like(csr.array)
            output = d_arr
        else:
            output = None
        arr = csr.densify(data, indices, indptr, output=output)
        if output_on_device:
            arr = arr.get()
        # Compare
        self.assertTrue(
            np.allclose(arr.reshape(self.array.shape), self.array),
            "something wrong with densified data (%s)"
            % current_config
        )



def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestCSR)
    )
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")


