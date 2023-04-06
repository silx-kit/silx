#!/usr/bin/env python
#
#    Project: Bitshuffle-LZ4 decompression in OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2022-2023  European Synchrotron Radiation Facility,
#                             Grenoble, France
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Test suite for byte-offset decompression 
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2022 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/11/2022"

import logging
import struct
import unittest
import numpy
try:
    import bitshuffle
except ImportError:
    bitshuffle = None
from silx.opencl.common import ocl, pyopencl
from silx.opencl.codec.bitshuffle_lz4 import BitshuffleLz4

logger = logging.getLogger(__name__)


@unittest.skipUnless(
    ocl and pyopencl and bitshuffle,
    "PyOpenCl or bitshuffle is missing",
)
class TestBitshuffle(unittest.TestCase):

    @staticmethod
    def _create_test_data(shape, lam=100, dtype="uint32"):
        """Create test (image, compressed stream) pair.

        :param shape: Shape of test image
        :param lam: Expectation of interval argument for numpy.random.poisson
        :return: (reference image array, compressed stream)
        """
        ref = numpy.random.poisson(lam, size=shape).astype(dtype)
        raw = struct.pack(">Q", ref.nbytes) +b"\x00"*4+bitshuffle.compress_lz4(ref).tobytes()
        return ref, raw

    def one_decompression(self, dtype, shape):
        """
        tests the byte offset decompression on GPU
        """
        ref, raw = self._create_test_data(shape=shape, dtype=dtype)
        bs = BitshuffleLz4(len(raw), numpy.prod(shape), dtype=dtype)
        res = bs.decompress(raw).get()
        self.assertEqual(numpy.all(res==ref.ravel()), True, "Checks decompression works")
        
    def test_decompress(self):
        """
        tests the byte offset decompression on GPU with various configuration
        """ 
        self.one_decompression("uint64", (103,503))
        self.one_decompression("int64", (101,509))
        self.one_decompression("uint32", (229,659))
        self.one_decompression("int32", (233,653))
        self.one_decompression("uint16", (743,647))
        self.one_decompression("int16", (751,643))
        self.one_decompression("uint8", (157,1373))
        self.one_decompression("int8", (163,1367))

    def test_decompress_from_buffer(self):
        """Test reading compressed data from pyopencl Buffer"""
        shape = 103, 503
        dtype = "uint32"

        ref, raw = self._create_test_data(shape=shape, dtype=dtype)

        bs = BitshuffleLz4(0, numpy.prod(shape), dtype=dtype)

        buffer = pyopencl.Buffer(
            bs.ctx,
            flags=pyopencl.mem_flags.COPY_HOST_PTR | pyopencl.mem_flags.READ_ONLY,
            hostbuf=raw,
        )

        res = bs.decompress(buffer, nbytes=buffer.size).get()
        self.assertEqual(numpy.all(res==ref.ravel()), True, "Checks decompression works")

    def test_decompress_from_array(self):
        """Test reading compressed data from pyopencl Array"""
        shape = 103, 503
        dtype = "uint32"

        ref, raw = self._create_test_data(shape=shape, dtype=dtype)

        bs = BitshuffleLz4(0, numpy.prod(shape), dtype=dtype)

        array = pyopencl.array.to_device(
            bs.queue,
            numpy.frombuffer(raw, dtype=numpy.uint8),
            array_queue=bs.queue,
        )

        res = bs.decompress(array).get()
        self.assertEqual(numpy.all(res==ref.ravel()), True, "Checks decompression works")
