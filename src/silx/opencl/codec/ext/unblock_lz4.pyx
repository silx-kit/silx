#cython: embedsignature=True, language_level=3
## This is for optimisation
##cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True

"""This module provides :func:`_unblock_lz4` which scans a dataset for different blocks.
"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "29/05/2026"

import os
cimport cython
from libc.stdint cimport uint8_t, uint32_t, uint64_t
import logging
import numpy

__all__ = ['_unblock_lz4']

_logger = logging.getLogger(__name__)

cdef bint SWAP_BE, SWAP_LE
if numpy.little_endian:
    SWAP_BE = True
    SWAP_LE = False
else:
    SWAP_BE = False
    SWAP_LE = True


cdef inline load64_at(uint8_t[::1] src,
               uint64_t pos,
               bint swap):
    """Read a 64-bit integer at the given position in the stream.

    :param src: byte array containing the data
    :param pos: position in the stream
    :param swap: if True, perform byte-swap (for endianness conversion)
    :return: 64-bit unsigned integer
    """
    cdef uint64_t result
    if swap:
        # Read bytes in reverse order (big-endian to little-endian or vice versa)
        result = (<uint64_t>(src[pos + 7]) << 56) | \
                 (<uint64_t>(src[pos + 6]) << 48) | \
                 (<uint64_t>(src[pos + 5]) << 40) | \
                 (<uint64_t>(src[pos + 4]) << 32) | \
                 (<uint64_t>(src[pos + 3]) << 24) | \
                 (<uint64_t>(src[pos + 2]) << 16) | \
                 (<uint64_t>(src[pos + 1]) << 8) | \
                 (<uint64_t>(src[pos + 0]))
    else:
        # Read bytes in native order (little-endian)
        result = (<uint64_t>(src[pos + 0]) << 56) | \
                 (<uint64_t>(src[pos + 1]) << 48) | \
                 (<uint64_t>(src[pos + 2]) << 40) | \
                 (<uint64_t>(src[pos + 3]) << 32) | \
                 (<uint64_t>(src[pos + 4]) << 24) | \
                 (<uint64_t>(src[pos + 5]) << 16) | \
                 (<uint64_t>(src[pos + 6]) << 8) | \
                 (<uint64_t>(src[pos + 7]))
    return result


cdef inline load32_at(uint8_t[::1] src,
               uint64_t pos,
               bint swap):
    """Read a 32-bit integer at the given position in the stream.

    :param src: byte array containing the data
    :param pos: position in the stream
    :param swap: if True, perform byte-swap (for endianness conversion)
    :return: 32-bit unsigned integer
    """
    cdef uint32_t result
    if swap:
        # Read bytes in reverse order (big-endian to little-endian or vice versa)
        result = (<uint32_t>(src[pos + 3]) << 24) | \
                 (<uint32_t>(src[pos + 2]) << 16) | \
                 (<uint32_t>(src[pos + 1]) << 8) | \
                 (<uint32_t>(src[pos + 0]))
    else:
        # Read bytes in native order (little-endian)
        result = (<uint32_t>(src[pos + 0]) << 24) | \
                 (<uint32_t>(src[pos + 1]) << 16) | \
                 (<uint32_t>(src[pos + 2]) <<  8) | \
                 (<uint32_t>(src[pos + 3]))
    return result


def _unblock_lz4(bytes src):
    """
    Parse a compressed LZ4 stream and record the start of each block
    :param src: compressed LZ4 stream
    """
    cdef:
        uint64_t size = len(src.size)
        uint8_t[::1] buffer = numpy.frombuffer(src, numpy.uint8)
        uint64_t max_blocks = block_start.size
        uint64_t total_nbytes = load64_at(src, 0, SWAP_BE)
        uint32_t block_nbytes = load32_at(src, 8, SWAP_BE)
        uint32_t block_idx = 0
        uint64_t pos = 12
        uint32_t block_size

    while ((pos+4<size) and (block_idx<max_blocks)):
        block_size = load32_at(src, pos, SWAP_BE)
        block_start[block_idx] = pos + 4
        block_idx +=1
        pos += 4 + block_size

    return numpy.uint32(block_idx)

