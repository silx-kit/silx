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
from silx.opencl.sparse import CSR
from ..common import ocl
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


    def test_sparsification(self):
        # Sparsify on device
        csr = CSR(self.array.shape)
        data, ind, indptr = csr.sparsify(self.array)
        # Compute reference sparsification
        a_s = sp.csr_matrix(self.array)
        data_ref, ind_ref, indptr_ref = a_s.data, a_s.indices, a_s.indptr
        # Compare
        nnz = a_s.nnz
        self.assertTrue(
            np.allclose(data[:nnz], data_ref),
            "something wrong with sparsified data"
        )
        self.assertTrue(
            np.allclose(ind[:nnz], ind_ref),
            "something wrong with sparsified indices"
        )
        self.assertTrue(
            np.allclose(indptr, indptr_ref),
            "something wrong with sparsified indices pointers (indptr)"
        )



def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestCSR)
    )
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")


