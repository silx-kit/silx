#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Module for data sparsification on CPU/GPU."""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "07/06/2019"

import numpy as np
import pyopencl.array as parray
from pyopencl.algorithm import GenericScanKernel
from .common import pyopencl as cl
from .processing import OpenclProcessing, EventDescription, BufferDescription
mf = cl.mem_flags


# only float32 arrays are supported for now

class CSR(OpenclProcessing):
    kernel_files = ["sparse.cl"]

    def __init__(self, shape, max_nnz=None, ctx=None, devicetype="all",
                 platformid=None, deviceid=None, block_size=None, memory=None,
                 profile=False):
        """
        Compute Compressed Sparse Row format of an image (2D matrix).
        It is designed to be compatible with scipy.sparse.csr_matrix.

        :param shape: tuple
            Matrix shape.
        :param max_nnz: int, optional
            Maximum number of non-zero elements. By default, the arrays "data"
            and "indices" are allocated with prod(shape) elements, but
            in practice a much lesser space is needed.
            The number of non-zero items cannot be known in advance, but one can
            estimate an upper-bound with this parameter to save memory.

        Opencl processing parameters
        -----------------------------
        Please refer to the documentation of silx.opencl.processing.OpenclProcessing
        for information on the other parameters.
        """

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self._set_parameters(shape, max_nnz)
        self._allocate_memory()
        self._setup_kernels()


    def _set_parameters(self, shape, max_nnz):
        self.shape = shape
        self.size = np.prod(shape)
        self.indice_dtype = np.int32 #
        assert len(shape) == 2 #
        if max_nnz is None:
            self.max_nnz = np.prod(shape) # worst case
        else:
            self.max_nnz = int(max_nnz)


    def _allocate_memory(self):
        self.is_cpu = (self.device.type == "CPU") # move to OpenclProcessing ?
        self.buffers = [
            BufferDescription("array", (self.size,), np.float32, mf.READ_ONLY),
            BufferDescription("data", (self.max_nnz,), np.float32, mf.READ_WRITE),
            BufferDescription("indices", (self.max_nnz,), self.indice_dtype, mf.READ_WRITE),
            BufferDescription("indptr", (self.shape[0]+1,), self.indice_dtype, mf.READ_WRITE),
        ]
        self.allocate_buffers(use_array=True)
        for arr_name in ["array", "data", "indices", "indptr"]:
            setattr(self, arr_name, self.cl_mem[arr_name])
        self._old_array = self.array


    def _setup_kernels(self):
        self.scan_kernel = GenericScanKernel(
            self.ctx, self.indice_dtype,
            arguments="__global float* data, __global float *data_compacted, __global int *indices, __global int* indptr",
            input_expr="(fabs(data[i]) > 0.0f) ? 1 : 0",
            scan_expr="a+b", neutral="0",
            output_statement="""
                // item is the running sum of input_expr(i), i.e the cumsum of "nonzero"
                if (prev_item != item) {
                    data_compacted[item-1] = data[i];
                    indices[item-1] = GET_INDEX(i);
                }
                // The last cumsum element of each line of "nonzero" goes to inptr[i]
                if ((i+1) % IMAGE_WIDTH == 0) {
                    indptr[(i/IMAGE_WIDTH)+1] = item;
                }
                """,
            options="-DIMAGE_WIDTH=%d" % self.shape[1],
            preamble="#define GET_INDEX(i) (i % IMAGE_WIDTH)",
        )



    def _setup_decompaction_kernel(self):
        OpenclProcessing.compile_kernels(
            self,
            self.kernel_files,
            compile_options=["-DIMAGE_WIDTH=%d" % self.shape[1]]
        )
        self._decomp_wg = (32, 1) # TODO tune
        self._decomp_grid = (self._decomp_wg[0], self.shape[0])


    def densify(self, arr):

        evt = self.
        self.kernels.densify_csr(
            self.queue,
            self._decomp_grid,
            self._decomp_wg,
            data.data,
            ind.data,
            indptr.data,
            output.data,
            np.int32(self.shape[0]),
        )
        evt.wait()
        return evt














    def check_input(self, arr):
        assert arr.shape == self.shape
        assert arr.dtype == np.float32

    # TODO handle pyopencl Buffer
    def check_output(self, output):
        assert len(output) == 3
        data, ind, indptr = output
        for arr in [data, ind, indptr]:
            assert arr.ndim == 1
        assert data.size == self.nnz_max
        assert ind.size == self.nnz_max
        assert iptr.size == self.shape[0]+1
        assert data.dtype == np.float32
        assert ind.dtype == np.indice_dtype
        assert indptr.dtype == np.indice_dtype


    def set_array(self, arr):
        self.check_input(arr)
        # GenericScanKernel only supports 1D data
        if isinstance(arr, parray.Array):
            self._old_array = arr
            self.array = arr
        elif isinstance(arr, np.ndarray):
            self.array[:] = arr.ravel()[:]
        else:
            raise ValueError("Expected pyopencl array or numpy array")


    def get_output(self, output):
        """
        Get the output array of the previous processing.

        :param output: tuple or None
            tuple in the form (data, indices, indptr). The content of these
            arrays will be overwritten with the result of the previous
            computation.
        """
        self.array = self._old_array
        numels = self.max_nnz
        if output is None:
            data = self.data.get()[:numels]
            ind = self.indices.get()[:numels]
            indptr = self.indptr.get()
        else:
            self.check_output(output)
            data, ind, indptr = output
            data[:] = self.data[:]
            ind[:] = self.indices[:]
            indptr[:] = self.indptr[:]
        return (data, ind, indptr)


    def sparsify(self, arr, output=None):
        """
        Convert an image (2D matrix) into a CSR representation.

        :param arr: numpy.ndarray or pyopencl.array.Array
            Input array.
        :param output: tuple of pyopencl.array.Array, optional
            If provided, this must be a tuple of 3 arrays (data, indices, indptr).
            The content of each array is overwritten by the computation result.
        """
        self.set_array(arr)
        self.scan_kernel(
            self.array,
            self.data,
            self.indices,
            self.indptr,
        )
        return self.get_output(output)




















