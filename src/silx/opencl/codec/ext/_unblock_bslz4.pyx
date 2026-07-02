#cython: embedsignature=True, language_level=3
## This is for optimization
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developing:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True

"""This module provides :func:`unblock_bslz4` which scans a dataset for different blocks.
"""


__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "02/07/2026"


from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport ceil
import numpy

# Define some constants for bitshuffle
cdef:
    const uint32_t BSHUF_MIN_RECOMMEND_BLOCK = 128
    const uint32_t BSHUF_BLOCKED_MULT = 8            # Block sizes must be multiple of this.
    const uint32_t BSHUF_TARGET_BLOCK_SIZE_B 8192


cdef inline uint64_t load64_BE(const uint8_t[::1] src,
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


cdef inline uint32_t load32_BE(const uint8_t[::1] src,
                      uint64_t pos) noexcept nogil :
    """Read a 32-bit BIG-endian integer at the given position in the stream.

    :param src: byte array containing the data
    :param pos: position in the stream
    :return: 32-bit unsigned integer
    """
    cdef uint32_t result
    result = (<uint32_t>(src[pos + 0]) << 24) | \
             (<uint32_t>(src[pos + 1]) << 16) | \
             (<uint32_t>(src[pos + 2]) <<  8) | \
             (<uint32_t>(src[pos + 3]))
    return result


cdef inline  uint32_t bshuf_default_block_size(uint32_t elem_size):
    """Calculate the proper block size

    This function needs to be absolutely stable between versions.
    Otherwise encoded data will not be decodable.
    """

    cdef uint32_t block_size = BSHUF_TARGET_BLOCK_SIZE_B / elem_size
    # Ensure it is a required multiple.
    block_size = (block_size // BSHUF_BLOCKED_MULT) * BSHUF_BLOCKED_MULT
    return max(block_size, BSHUF_MIN_RECOMMEND_BLOCK)
}


def unblock_bslz4(bytes src, uint8_t item_size=4):
    """
    Parse a compressed Bitshuffle-LZ4 stream and record the start of each LZ4-block

    :param src: compressed LZ4 stream
    :return: list of start of block in the input stream (as numpy array)
    """
    cdef:
        uint64_t size = len(src)
        uint64_t total_nbytes = load64_BE(src, 0)
        uint32_t block_nbytes = load32_BE(src, 8)
        const uint8_t[::1] buffer = src
        uint32_t block_max, block_max_size
        uint64_t[::1] block_start
        uint32_t block_idx = 0
        uint64_t pos = 12
        uint64_t end = pos + 4
        uint64_t block_size

    if block_nbytes == 0:
            block_nbytes = 1<<13  # 8k
            # This is a simplified version. One should take into account the item_size
    block_max_size = int(ceil(block_nbytes * 1.0046))
    # the actual (uncompressed) block size is 8184
    # but compressed block can be 0.5% larger in case of incompressible data

    block_max = (total_nbytes + block_nbytes - 1) // block_nbytes
    block_start = numpy.empty(block_max, dtype=numpy.uint64)
    with nogil:
        for block_idx in range(block_max):
            if end >= size:
                # This would read beyond the end of the stream -> invalid
                block_start = block_start[:block_idx]
                break
            block_size = load32_BE(buffer, pos)
            if block_size>=block_max_size:
                # This is an invalid block ... since it is larger than the compressed block size
                block_start = block_start[:block_idx]
                break
            block_start[block_idx] = end
            pos = end + block_size
            end = pos + 4

    return numpy.asarray(block_start)
