#cython: embedsignature=True, language_level=3
## This is for optimization
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developing:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True

"""This module provides :func:`_unblock_lz4` which scans a dataset for different blocks.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "04/06/2026"

import os
cimport cython
from libc.stdint cimport uint8_t, uint32_t, uint64_t
import logging
import numpy

__all__ = ['_unblock_lz4']

_logger = logging.getLogger(__name__)

cdef inline uint64_t load64_at(const uint8_t[::1] src,
                               uint64_t pos) noexcept nogil :
    """Read a 64-bit BIG-endian integer at the given position in the stream.

    :param src: byte array containing the data
    :param pos: position in the stream
    :return: 64-bit unsigned integer
    """
    cdef uint64_t result
    result = (<uint64_t>(src[pos + 0]) << 56) | \
             (<uint64_t>(src[pos + 1]) << 48) | \
             (<uint64_t>(src[pos + 2]) << 40) | \
             (<uint64_t>(src[pos + 3]) << 32) | \
             (<uint64_t>(src[pos + 4]) << 24) | \
             (<uint64_t>(src[pos + 5]) << 16) | \
             (<uint64_t>(src[pos + 6]) << 8) | \
             (<uint64_t>(src[pos + 7]))
    return result


cdef inline uint32_t load32_at(const uint8_t[::1] src,
                      uint64_t pos) noexcept nogil :
    """Read a 32-bit BIG-endian integer at the given position in the stream.

    :param src: byte array containing the data
    :param pos: position in the stream
    :param swap: if True, perform byte-swap (for endianness conversion)
    :return: 32-bit unsigned integer
    """
    cdef uint32_t result
    result = (<uint32_t>(src[pos + 0]) << 24) | \
                (<uint32_t>(src[pos + 1]) << 16) | \
                (<uint32_t>(src[pos + 2]) <<  8) | \
                (<uint32_t>(src[pos + 3]))
    return result


def _unblock_lz4(bytes src):
    """
    Parse a compressed LZ4 stream and record the start of each block
    :param src: compressed LZ4 stream
    :return: list of start of block in the input stream
    """
    cdef:
        uint32_t size = len(src)
        uint64_t total_nbytes = load64_at(src, 0)
        uint32_t block_nbytes = load32_at(src, 8)
        const uint8_t[::1] buffer = src
        uint32_t block_max = min(size//4+1, 1<<32-1)
        uint64_t[::1] block_start = numpy.empty(block_max, dtype=numpy.uint64)
        uint32_t block_idx = 0
        uint64_t pos = 12
        uint64_t end = pos + 4
        uint64_t block_size
    with nogil:
      while ((end+4<size) and (block_idx<block_max)):
            block_size = load32_at(buffer, pos)
            block_start[block_idx] = end
            block_idx +=1
            pos = end + block_size
            end = pos + 4

    return numpy.asarray(block_start[:block_idx])
