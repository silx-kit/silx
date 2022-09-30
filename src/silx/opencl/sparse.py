#!/usr/bin/env python
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

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "07/06/2019"

import numpy
import pyopencl.array as parray
from collections import namedtuple
from pyopencl.scan import GenericScanKernel
from pyopencl.tools import dtype_to_ctype
from .common import pyopencl as cl
from .processing import OpenclProcessing, EventDescription, BufferDescription
mf = cl.mem_flags


CSRData = namedtuple("CSRData", ["data", "indices", "indptr"])

def tuple_to_csrdata(arrs):
    """
    Converts a 3-tuple to a CSRData namedtuple.
    """
    if arrs is None:
        return None
    return CSRData(data=arrs[0], indices=arrs[1], indptr=arrs[2])



class CSR(OpenclProcessing):
    kernel_files = ["sparse.cl"]

    def __init__(self, shape, dtype="f", max_nnz=None, idx_dtype=numpy.int32,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """
        Compute Compressed Sparse Row format of an image (2D matrix).
        It is designed to be compatible with scipy.sparse.csr_matrix.

        :param shape: tuple
            Matrix shape.
        :param dtype: str or numpy.dtype, optional
            Numeric data type. By default, sparse matrix data will be float32.
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
                                  block_size=block_size, memory=memory,
                                  profile=profile)
        self._set_parameters(shape, dtype, max_nnz, idx_dtype)
        self._allocate_memory()
        self._setup_kernels()

    # --------------------------------------------------------------------------
    # -------------------------- Initialization --------------------------------
    # --------------------------------------------------------------------------

    def _set_parameters(self, shape, dtype, max_nnz, idx_dtype):
        self.shape = shape
        self.size = numpy.prod(shape)
        self._set_idx_dtype(idx_dtype)
        assert len(shape) == 2 #
        if max_nnz is None:
            self.max_nnz = numpy.prod(shape) # worst case
        else:
            self.max_nnz = int(max_nnz)
        self._set_dtype(dtype)


    def _set_idx_dtype(self, idx_dtype):
        idx_dtype = numpy.dtype(idx_dtype)
        if idx_dtype.kind not in ["i", "u"]:
            raise ValueError("Not an integer type: %s" % idx_dtype)
        # scan value type must have size divisible by 4 bytes
        if idx_dtype.itemsize % 4 != 0:
            raise ValueError("Due to an internal pyopencl limitation, idx_dtype type must have size divisible by 4 bytes")
        self.indice_dtype = idx_dtype #


    def _set_dtype(self, dtype):
        self.dtype = numpy.dtype(dtype)
        if self.dtype.kind == "c":
            raise ValueError("Complex data is not supported")
        if self.dtype == numpy.dtype(numpy.float32):
            self._c_zero_str = "0.0f"
        elif self.dtype == numpy.dtype(numpy.float64):
            self._c_zero_str = "0.0"
        else: # assuming integer
            self._c_zero_str = "0"
        self.c_dtype = dtype_to_ctype(self.dtype)
        self.idx_c_dtype = dtype_to_ctype(self.indice_dtype)


    def _allocate_memory(self):
        self.is_cpu = (self.device.type == "CPU") # move to OpenclProcessing ?
        self.buffers = [
            BufferDescription("array", (self.size,), self.dtype, mf.READ_ONLY),
            BufferDescription("data", (self.max_nnz,), self.dtype, mf.READ_WRITE),
            BufferDescription("indices", (self.max_nnz,), self.indice_dtype, mf.READ_WRITE),
            BufferDescription("indptr", (self.shape[0]+1,), self.indice_dtype, mf.READ_WRITE),
        ]
        self.allocate_buffers(use_array=True)
        for arr_name in ["array", "data", "indices", "indptr"]:
            setattr(self, arr_name, self.cl_mem[arr_name])
            self.cl_mem[arr_name].fill(0) # allocate_buffers() uses empty()
        self._old_array = self.array
        self._old_data = self.data
        self._old_indices = self.indices
        self._old_indptr = self.indptr


    def _setup_kernels(self):
        self._setup_compaction_kernel()
        self._setup_decompaction_kernel()


    def _setup_compaction_kernel(self):
        kernel_signature = str(
            "__global %s *data, \
            __global %s *data_compacted, \
            __global %s *indices, \
            __global %s* indptr \
            """ % (self.c_dtype, self.c_dtype, self.idx_c_dtype, self.idx_c_dtype)
        )
        if self.dtype.kind == "f":
            map_nonzero_expr = "(fabs(data[i]) > %s) ? 1 : 0" % self._c_zero_str
        elif self.dtype.kind in ["u", "i"]:
            map_nonzero_expr = "(data[i] != %s) ? 1 : 0" % self._c_zero_str
        else:
            raise ValueError("Unknown data type")

        self.scan_kernel = GenericScanKernel(
            self.ctx, self.indice_dtype,
            arguments=kernel_signature,
            input_expr=map_nonzero_expr,
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
            options=["-DIMAGE_WIDTH=%d" % self.shape[1]],
            preamble="#define GET_INDEX(i) (i % IMAGE_WIDTH)",
        )


    def _setup_decompaction_kernel(self):
        OpenclProcessing.compile_kernels(
            self,
            self.kernel_files,
            compile_options=[
                "-DIMAGE_WIDTH=%d" % self.shape[1],
                "-DDTYPE=%s" % self.c_dtype,
                "-DIDX_DTYPE=%s" % self.idx_c_dtype,
            ]
        )
        device = self.ctx.devices[0]
        wg_x = min(
            device.max_work_group_size,
            32,
            self.kernels.max_workgroup_size("densify_csr")
        )
        self._decomp_wg = (wg_x, 1)
        self._decomp_grid = (self._decomp_wg[0], self.shape[0])


    # --------------------------------------------------------------------------
    # -------------------------- Array utils -----------------------------------
    # --------------------------------------------------------------------------

    # TODO handle pyopencl Buffer
    def check_array(self, arr):
        """
        Check that provided array is compatible with current context.

        :param arr: numpy.ndarray or pyopencl.array.Array
            2D array in dense format.
        """
        assert arr.size == self.size
        assert arr.dtype == self.dtype


    # TODO handle pyopencl Buffer
    def check_sparse_arrays(self, csr_data):
        """
        Check that the provided sparse arrays are compatible with the current
        context.

        :param arrays: namedtuple CSRData.
            It contains the arrays "data", "indices", "indptr"
        """
        assert isinstance(csr_data, CSRData)
        for arr in [csr_data.data, csr_data.indices, csr_data.indptr]:
            assert arr.ndim == 1
        assert csr_data.data.size <= self.max_nnz
        assert csr_data.indices.size <= self.max_nnz
        assert csr_data.indptr.size == self.shape[0]+1
        assert csr_data.data.dtype == self.dtype
        assert csr_data.indices.dtype == self.indice_dtype
        assert csr_data.indptr.dtype == self.indice_dtype


    def set_array(self, arr):
        """
        Set the provided array as the current context 2D matrix.

        :param arr: numpy.ndarray or pyopencl.array.Array
            2D array in dense format.
        """
        if arr is None:
            return
        self.check_array(arr)
        # GenericScanKernel only supports 1D data
        if isinstance(arr, parray.Array):
            self._old_array = self.array
            self.array = arr
        elif isinstance(arr, numpy.ndarray):
            self.array[:] = arr.ravel()[:]
        else:
            raise ValueError("Expected pyopencl array or numpy array")


    def set_sparse_arrays(self, csr_data):
        if csr_data is None:
            return
        self.check_sparse_arrays(csr_data)
        for name, arr in {"data": csr_data.data, "indices": csr_data.indices, "indptr": csr_data.indptr}.items():
            # The current array is a device array. Don't copy, use it directly
            if isinstance(arr, parray.Array):
                setattr(self, "_old_" + name, getattr(self, name))
                setattr(self, name, arr)
            # The current array is a numpy.ndarray: copy H2D
            elif isinstance(arr, numpy.ndarray):
                getattr(self, name)[:arr.size] = arr[:]
            else:
                raise ValueError("Unsupported array type: %s" % type(arr))


    def _recover_arrays_references(self):
        """
        Recover the previous arrays references, and return the references of the
        "current" arrays.
        """
        array = self.array
        data = self.data
        indices = self.indices
        indptr = self.indptr
        for name in ["array", "data", "indices", "indptr"]:
            # self.X = self._old_X
            setattr(self, name, getattr(self, "_old_" + name))
        return array, (data, indices, indptr)


    def get_sparse_arrays(self, output):
        """
        Get the 2D dense array of the current context.

        :param output: tuple or None
            tuple in the form (data, indices, indptr). These arrays have to be
            compatible with the current context (size and data type).
            The content of these arrays will be overwritten with the result of
            the previous computation.
        """
        numels = self.max_nnz
        if output is None:
            data = self.data.get()[:numels]
            ind = self.indices.get()[:numels]
            indptr = self.indptr.get()
            res = (data, ind, indptr)
        else:
            res = output
        return res


    def get_array(self, output):
        if output is None:
            res = self.array.get().reshape(self.shape)
        else:
            res = output
        return res

    # --------------------------------------------------------------------------
    # -------------------------- Compaction ------------------------------------
    # --------------------------------------------------------------------------

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
        self.set_sparse_arrays(tuple_to_csrdata(output))
        evt = self.scan_kernel(
            self.array,
            self.data,
            self.indices,
            self.indptr,
        )
        #~ evt.wait()
        self.profile_add(evt, "sparsification kernel")
        res = self.get_sparse_arrays(output)
        self._recover_arrays_references()
        return res

    # --------------------------------------------------------------------------
    # -------------------------- Decompaction ----------------------------------
    # --------------------------------------------------------------------------

    def densify(self, data, indices, indptr, output=None):
        self.set_sparse_arrays(
            CSRData(data=data, indices=indices, indptr=indptr)
        )
        self.set_array(output)
        evt = self.kernels.densify_csr(
            self.queue,
            self._decomp_grid,
            self._decomp_wg,
            self.data.data,
            self.indices.data,
            self.indptr.data,
            self.array.data,
            numpy.int32(self.shape[0]),
        )
        #~ evt.wait()
        self.profile_add(evt, "desparsification kernel")
        res = self.get_array(output)
        self._recover_arrays_references()
        return res

