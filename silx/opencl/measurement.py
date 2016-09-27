#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
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


from __future__ import division, print_function

__authors__ = ["Henri Payno, Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/09/2016"

from . import ocl

if ocl:
    import pyopencl
    import pyopencl.array

from .utils import get_opencl_code

import numpy 

def measurement(device):
    """
    TODO
    """
    if (ocl is None) or (device is None) :
        return None

    if isinstance(device, pyopencl.Context):
        device = device.devices[0]
    elif isinstance(device, Device):
        device =  "jsdf"

    expected_wg = device.max_work_group_size

def measure_workgroup_size(device):
    """
    Function to mesure the maximal work group size of the given device
    """
    assert isinstance(device, pyopencl.Device)
    
    shape = 4096
    # get the context
    ctx = pyopencl.Context()
    assert(not ctx is None)
    queue = pyopencl.CommandQueue(ctx)

    max_valid_wg = 1
    data = numpy.random.random(shape).astype(numpy.float32)
    d_data = pyopencl.array.to_device(queue, data)
    d_data_1 = pyopencl.array.zeros_like(d_data) + 1

    program = pyopencl.Program(ctx, get_opencl_code("addition")).build()

    maxi = int(round(numpy.log2(shape)))
    for i in range(maxi):
        d_res = pyopencl.array.empty_like(d_data)
        wg = 1 << i
        try:
            evt = program.addition(queue, (shape,), (wg,),
                   d_data.data, d_data_1.data, d_res.data, numpy.int32(shape))
            evt.wait()
        except Exception as error:
            print("Error on WG=%s"%wg)
            program = queue = d_res = d_data_1 = d_data = None
            break;
        else:
            res = d_res.get()
            good = numpy.allclose(res, data + 1 )
            if good and wg>max_valid_wg:
                max_valid_wg = wg


    program = queue = d_res = d_data_1 = d_data = None
    return max_valid_wg