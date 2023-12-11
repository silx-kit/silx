#
#    Project: S I L X project
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2023 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
Utilities around atomic operation in OpenCL
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "14/06/2023"
__copyright__ = "2023-2023, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import numpy
import pyopencl
import pyopencl.array as cla


def check_atomic32(device):
    try:
        ctx = pyopencl.Context(devices=[device])
    except:
        return False, f"Unable to create context on {device}"
    else:
        queue = pyopencl.CommandQueue(ctx)
    src = """
kernel void check_atomic32(global int* ary){
int res = atom_inc(ary);
}
"""
    try:
        prg = pyopencl.Program(ctx, src).build()
    except Exception as err:
        return False, f"{type(err)}: {err}"
    a = numpy.zeros(1, numpy.int32)
    d = cla.to_device(queue, a)
    prg.check_atomic32(queue, (1024,), (32,), d.data).wait()
    value = d.get()[0]
    return value == 1024, f"Got the proper value 1024=={value}"


def check_atomic64(device):
    try:
        ctx = pyopencl.Context(devices=[device])
    except:
        return False, f"Unable to create context on {device}"
    else:
        queue = pyopencl.CommandQueue(ctx)
    if (
        device.platform.name == "Portable Computing Language"
        and "GPU" in pyopencl.device_type.to_string(device.type).upper()
    ):
        # this configuration is known to seg-fault
        return False, "PoCL + GPU do not support atomic64"
    src = """
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
kernel void check_atomic64(global long* ary){
long res = atom_inc(ary);
}
"""
    try:
        prg = pyopencl.Program(ctx, src).build()
    except Exception as err:
        return False, f"{type(err)}: {err}"
    a = numpy.zeros(1, numpy.int64)
    d = cla.to_device(queue, a)
    prg.check_atomic64(queue, (1024,), (32,), d.data).wait()
    value = d.get()[0]
    return value == 1024, f"Got the proper value 1024=={value}"
