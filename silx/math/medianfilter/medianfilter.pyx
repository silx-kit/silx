from cython.parallel import parallel, prange
cimport cython
cimport median_filter
import numpy
cimport numpy as cnumpy
cdef Py_ssize_t size = 10
from libcpp cimport bool

def median_filter(input_buffer, kernel_dim, bool conditionnal, int nthread=8):
    reshaped = False
    if len(input_buffer.shape) < 2 :
      input_buffer = input_buffer.reshape(input_buffer.shape[0], 1)
      reshaped = True

    # simple median filter apply into a 2D buffer
    output_buffer = numpy.zeros(input_buffer.shape, dtype=input_buffer.dtype)
    check2D(input_buffer, output_buffer)

    image_dim = numpy.array(input_buffer.shape, dtype=numpy.int32)

    if type(kernel_dim) in (tuple, list):
      if(len(kernel_dim) == 1):
          ker_dim = numpy.array([kernel_dim[0], 1], dtype=numpy.int32)
      else:
          ker_dim = numpy.array(kernel_dim, dtype=numpy.int32)
    else:
      ker_dim = numpy.array([kernel_dim, 1], dtype=numpy.int32)

    ranges = numpy.array([ int(input_buffer.shape[0] * x / nthread) for x in range(nthread+1) ], dtype=numpy.int32)

    if input_buffer.dtype == numpy.float64:
        medfilterfc = _median_filter_float64
    elif input_buffer.dtype == numpy.float32:
        medfilterfc = _median_filter_float32
    elif input_buffer.dtype == numpy.int32:
        medfilterfc = _median_filter_int32
    else:
        raise ValueError("%s type is not managed by the median filter"%input_buffer.dtype)

    medfilterfc(input_buffer=input_buffer,
                output_buffer=output_buffer, 
                kernel_dim=ker_dim,
                ranges=ranges,
                image_dim=image_dim,
                conditionnal=conditionnal)

    if reshaped : 
      input_buffer = input_buffer.reshape(input_buffer.shape[0])
      output_buffer = output_buffer.reshape(input_buffer.shape[0])

    return output_buffer

def check2D(input_buffer, output_buffer):
    if (input_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<input_buffer> must be a C_CONTIGUOUS numpy array.')

    if (output_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<output_buffer> must be a C_CONTIGUOUS numpy array.')

    if not (len(input_buffer.shape) <= 2):
        raise ValueError('<input_buffer> dimension must mo higher than 2.')    

    if not (len(output_buffer.shape) <= 2):
        raise ValueError('<output_buffer> dimension must mo higher than 2.')    

    if not(input_buffer.dtype == output_buffer.dtype):
        raise ValueError('input buffer and output_buffer must be of the same type')

    if not (input_buffer.shape == output_buffer.shape):
        raise ValueError('input buffer and output_buffer must be of the same dimension and same dimension')

def _median_filter_int32(cnumpy.ndarray[cnumpy.int32_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.int32_t, ndim=2, mode='c'] output_buffer not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] kernel_dim not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] image_dim not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will manage all the y
        for x in prange(nthread):
            median_filter.median_filter[int](<int*> input_buffer.data, 
                                             <int*> output_buffer.data, 
                                             <int*>kernel_dim.data,
                                             <int*>image_dim.data,
                                             ranges[x],
                                             ranges[x+1]-1,
                                             0,
                                             image_dim[1]-1,
                                             conditionnal);

def _median_filter_float32(cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] output_buffer not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] kernel_dim not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] image_dim not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will manage all the y
        for x in prange(nthread):
            median_filter.median_filter[float](<float*> input_buffer.data, 
                                               <float*> output_buffer.data, 
                                               <int*>kernel_dim.data,
                                               <int*>image_dim.data,
                                               ranges[x],
                                               ranges[x+1]-1,
                                               0,
                                               image_dim[1]-1,
                                               conditionnal);            

def _median_filter_float64(cnumpy.ndarray[cnumpy.float64_t, ndim=2, mode='c'] input_buffer not None, 
                  cnumpy.ndarray[cnumpy.float64_t, ndim=2, mode='c'] output_buffer not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] kernel_dim not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] ranges not None,
                  cnumpy.ndarray[cnumpy.int32_t, ndim=1, mode='c'] image_dim not None,
                  bool conditionnal):

    # init x range
    cdef int nthread = len(ranges) -1
    cdef int x = 0
    with nogil, parallel():
        # for now we are only deviding the image by coluumns (only in x). All threads will manage all the y
        for x in prange(nthread):
            median_filter.median_filter[double](<double*> input_buffer.data, 
                                                <double*> output_buffer.data, 
                                                <int*>kernel_dim.data,
                                                <int*>image_dim.data,
                                                ranges[x],
                                                ranges[x+1]-1,
                                                0,
                                                image_dim[1]-1,
                                                conditionnal);
