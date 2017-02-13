from cython.parallel import parallel, prange
cimport cython
cimport median_filter
import numpy
cimport numpy as cnumpy
cdef Py_ssize_t size = 10
from libcpp cimport bool

# TODO : add more types ?
def median_filter(input_buffer, kernel_dim, bool conditionnal, int nthread=8):
    output_buffer = numpy.zeros(input_buffer.shape, dtype=input_buffer.dtype)
    check(input_buffer, output_buffer)

    ranges = numpy.array([ int(input_buffer.shape[0] * x / nthread) for x in range(nthread+1) ], dtype=numpy.int32)

    if input_buffer.dtype == numpy.float64:
        _median_filter_float64(input_buffer, output_buffer, kernel_dim[0], kernel_dim[1], ranges, conditionnal)
    elif input_buffer.dtype == numpy.float32:
        _median_filter_float32(input_buffer, output_buffer, kernel_dim[0], kernel_dim[1], ranges, conditionnal)
    elif input_buffer.dtype == numpy.int32:
        _median_filter_int32(input_buffer, output_buffer, kernel_dim[0], kernel_dim[1], ranges, conditionnal)
    else:
        raise ValueError("%s type is not managed by the median filter"%input_buffer.dtype)

    return output_buffer

def check(input_buffer, output_buffer):
    if (input_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<input_buffer> must be a C_CONTIGUOUS numpy array.')

    if (output_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<output_buffer> must be a C_CONTIGUOUS numpy array.')

    if not (len(input_buffer.shape) == 2):
        raise ValueError('<input_buffer> dimension must be 2.')    

    if not (len(output_buffer.shape) == 2):
        raise ValueError('<output_buffer> dimension must be 2.')    

    if not(input_buffer.dtype == output_buffer.dtype):
        raise ValueError('input buffer and output_buffer must be of the same type')

    if not (input_buffer.shape[0] == output_buffer.shape[0] and
            input_buffer.shape[1] == output_buffer.shape[1]):
        raise ValueError('input buffer and output_buffer must be of the same dimension and same dimension')

def _median_filter_int32(cnumpy.ndarray[cnumpy.int32_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.int32_t, ndim=2, mode='c'] output_buffer not None,
                  int kernel_width,
                  int kernel_height,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will mange all the y
        for x in prange(nthread):
            median_filter.median_filter[int](<int*> input_buffer.data, 
                                             <int*> output_buffer.data, 
                                             kernel_width,
                                             kernel_height,
                                             input_buffer.shape[0],
                                             input_buffer.shape[1],
                                             ranges[x],
                                             ranges[x+1]-1,
                                             0,
                                             input_buffer.shape[1]-1,
                                             conditionnal);

def _median_filter_float32(cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] output_buffer not None,
                  int kernel_width,
                  int kernel_height,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will mange all the y
        for x in prange(nthread):
            median_filter.median_filter[float](<float*> input_buffer.data, 
                                               <float*> output_buffer.data, 
                                               kernel_width,
                                               kernel_height,
                                               input_buffer.shape[0],
                                               input_buffer.shape[1],
                                               ranges[x],
                                               ranges[x+1]-1,
                                               0,
                                               input_buffer.shape[1]-1,
                                               conditionnal);            


def _median_filter_float64(cnumpy.ndarray[cnumpy.float64_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.float64_t, ndim=2, mode='c'] output_buffer not None,
                  int kernel_width,
                  int kernel_height,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will mange all the y
        for x in prange(nthread):
            median_filter.median_filter[double](<double*> input_buffer.data, 
                                                <double*> output_buffer.data, 
                                                kernel_width,
                                                kernel_height,
                                                input_buffer.shape[0],
                                                input_buffer.shape[1],
                                                ranges[x],
                                                ranges[x+1]-1,
                                                0,
                                                input_buffer.shape[1]-1,
                                                conditionnal);
