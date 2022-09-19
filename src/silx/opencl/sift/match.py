#!/usr/bin/env python
#
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2018  European Synchrotron Radiation Facility, Grenoble, France
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

"""
Contains a class for creating a matching plan, allocating arrays, 
compiling kernels and other things like that
"""

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/01/2018"
__status__ = "production"


import logging
import numpy
from .param import par
from ..common import pyopencl, kernel_workgroup_size
from .utils import calc_size
from ..processing import OpenclProcessing, BufferDescription
logger = logging.getLogger(__name__)


class MatchPlan(OpenclProcessing):
    """Plan to compare sets of SIFT keypoint and find common ones.

    .. code-block:: python

        siftp = sift.MatchPlan(devicetype="ALL")
        commonkp = siftp.match(kp1,kp2)

    where kp1, kp2 is a n x 132 array. the second dimension is composed of x,y, 
    scale and angle as well as 128 floats describing the keypoint.
    commonkp is mx2 array of matching keypoints
    """
    kernels_size = {"matching_gpu": 64,
                    "matching_cpu": 16}

    dtype_kp = numpy.dtype([('x', numpy.float32),
                            ('y', numpy.float32),
                            ('scale', numpy.float32),
                            ('angle', numpy.float32),
                            ('desc', (numpy.uint8, 128))
                            ])

    def __init__(self, size=16384, devicetype="ALL", profile=False, device=None,
                 block_size=None, roi=None, ctx=None):
        """Constructor of the class:

        :param size: size of the input keypoint-list alocated on the GPU.
        :param devicetype: can be CPU or GPU
        :param profile: set to true to activate profiling information collection
        :param device: 2-tuple of integer, see clinfo
        :param block_size: CPU on MacOS, limit to 1. None by default to use default ones (max=128).
        :param roi: Region Of Interest: TODO
        :param context: Use an external context (discard devicetype and device options)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  block_size=block_size, profile=profile)
        self.kpsize = size
        self.octave_max = None
        self.red_size = None
        self.debug = []

        devicetype = self.device.type
        if (devicetype == "CPU"):
            self.USE_CPU = True
            matching_kernel = "matching_cpu"
        else:
            self.USE_CPU = False
            matching_kernel = "matching_gpu"
        wg_size = self.__class__.kernels_size[matching_kernel]
        self.compile_kernels(kernel_files=["sift/sift",
                                           "sift/memset",
                                           "sift/" + matching_kernel],
                             compile_options='-D WORKGROUP_SIZE=%s' % wg_size)

        self.roi = None
        if roi:
            self.set_roi(roi)

        buffers = [  # BufferDescription"name", "size", "dtype", "flags"
                   BufferDescription("Kp_1", self.kpsize, self.dtype_kp, flags=None),
                   BufferDescription("Kp_2", self.kpsize, dtype=self.dtype_kp, flags=None),
                   BufferDescription("match", (self.kpsize, 2), dtype=numpy.int32, flags=None),
                   BufferDescription("cnt", 1, numpy.int32, flags=None)]
        self.allocate_buffers(buffers, use_array=True)
        self.kernel_size = {}
        for name, kernel in self.kernels.get_kernels().items():
            self.kernel_size[name] = kernel_workgroup_size(self.program, kernel)

    def match(self, nkp1, nkp2, raw_results=False):
        """Calculate the matching of 2 keypoint list

        :param nkp1: numpy 1D recarray of keypoints or equivalent GPU buffer
        :param nkp2: numpy 1D recarray of keypoints or equivalent GPU buffer
        :param raw_results: if true return the 2D array of indexes of matching keypoints (not the actual keypoints)

        TODO: implement the ROI ...
        """
        assert len(nkp1.shape) == 1  # Nota: nkp1.ndim is not valid for gpu_arrays
        assert len(nkp2.shape) == 1
        valid_types = (numpy.ndarray, numpy.core.records.recarray, pyopencl.array.Array)
        assert isinstance(nkp1, valid_types)
        assert isinstance(nkp2, valid_types)
        result = None
        with self.sem:
            if isinstance(nkp1, pyopencl.array.Array):

                kpt1_gpu = nkp1
            else:
                if nkp1.size > self.cl_mem["Kp_1"].size:
                    logger.warning("increasing size of keypoint vector 1 to %i" % nkp1.size)
                    self.cl_mem["Kp_1"] = pyopencl.array.empty(self.queue, (nkp1.size,), dtype=self.dtype_kp)
                kpt1_gpu = self.cl_mem["Kp_1"]
                self._reset_buffer1()
                evt1 = pyopencl.enqueue_copy(self.queue, kpt1_gpu.data, nkp1)
                if self.profile:
                    self.events.append(("copy H->D KP_1", evt1))

            if isinstance(nkp2, pyopencl.array.Array):
                kpt2_gpu = nkp2
            else:
                if nkp2.size > self.cl_mem["Kp_2"].size:
                    logger.warning("increasing size of keypoint vector 2 to %i" % nkp2.size)
                    self.cl_mem["Kp_2"] = pyopencl.array.empty(self.queue, (nkp2.size,), dtype=self.dtype_kp)
                kpt2_gpu = self.cl_mem["Kp_2"]
                self._reset_buffer2()
                evt2 = pyopencl.enqueue_copy(self.queue, kpt2_gpu.data, nkp2)
                if self.profile:
                    self.events.append(("copy H->D KP_2", evt2))

            if min(kpt1_gpu.size, kpt2_gpu.size) > self.cl_mem["match"].shape[0]:
                self.kpsize = min(kpt1_gpu.size, kpt2_gpu.size)
                self.cl_mem["match"] = pyopencl.array.empty(self.queue, (self.kpsize, 2), dtype=numpy.int32)
            self._reset_output()
            wg = self.kernel_size["matching"]
            size = calc_size((nkp1.size,), (wg,))
            evt = self.kernels.matching(self.queue, size, (wg,),
                                        kpt1_gpu.data,
                                        kpt2_gpu.data,
                                        self.cl_mem["match"].data,
                                        self.cl_mem["cnt"].data,
                                        numpy.int32(self.kpsize),
                                        numpy.float32(par.MatchRatio * par.MatchRatio),
                                        numpy.int32(nkp1.size),
                                        numpy.int32(nkp2.size))
            if self.profile:
                self.events.append(("matching", evt))
            size = self.cl_mem["cnt"].get()[0]
            match = numpy.empty(shape=(size, 2), dtype=numpy.int32)
            if size > 0:
                cpyD2H = pyopencl.enqueue_copy(self.queue, match, self.cl_mem["match"].data)
            if self.profile:
                self.events.append(("copy D->H match", cpyD2H))
            if raw_results:
                result = match
            else:
                result = numpy.recarray(shape=(size, 2), dtype=self.dtype_kp)

                result[:, 0] = nkp1[match[:size, 0]]
                result[:, 1] = nkp2[match[:size, 1]]
        return result

    __call__ = match

    def _reset_buffer(self):
        """Reseet all buffers"""
        self._reset_buffer1()
        self._reset_buffer2()
        self._reset_output()

    def _reset_buffer1(self):
        wg = self.kernel_size["memset_kp"]
        size = calc_size((self.cl_mem["Kp_1"].size,), (wg,))
        ev1 = self.kernels.memset_kp(self.queue, size, (wg,),
                                     self.cl_mem["Kp_1"].data, numpy.float32(-1.0), numpy.uint8(0), numpy.int32(self.cl_mem["Kp_1"].size))
        if self.profile:
            self.events.append(("memset Kp1", ev1))

    def _reset_buffer2(self):
        wg = self.kernel_size["memset_kp"]
        size = calc_size((self.cl_mem["Kp_2"].size,), (wg,))
        ev2 = self.kernels.memset_kp(self.queue, size, (wg,),
                                     self.cl_mem["Kp_2"].data, numpy.float32(-1.0), numpy.uint8(0), numpy.int32(self.cl_mem["Kp_2"].size))
        if self.profile:
            self.events.append(("memset Kp2", ev2))

    def _reset_output(self):
        ev3 = self.kernels.memset_int(self.queue, calc_size((self.cl_mem["match"].size,), (self.kernel_size["memset_int"],)), (self.kernel_size["memset_int"],),
                                      self.cl_mem["match"].data, numpy.int32(-1), numpy.int32(self.cl_mem["match"].size))
        ev4 = self.kernels.memset_int(self.queue, (1,), (1,),
                                      self.cl_mem["cnt"].data, numpy.int32(0), numpy.int32(1))
        if self.profile:
            self.events += [("memset match", ev3),
                            ("memset cnt", ev4), ]

    reset_timer = OpenclProcessing.reset_log

    def set_roi(self, roi):
        """Defines the region of interest

        :param roi: region of interest as 2D numpy array with non zero where
                    valid pixels are
        """
        with self.sem:
            self.roi = numpy.ascontiguousarray(roi, numpy.int8)
            self.cl_mem["ROI"] = pyopencl.array.to_device(self.queue, self.roi)

    def unset_roi(self):
        """Unset the region of interest
        """
        with self.sem:
            self.roi = None
            self.cl_mem["ROI"] = None


def match_py(nkp1, nkp2, raw_results=False):
    """Pure numpy implementation of match:

    :param nkp1, nkp2: Numpy record array of keypoints with descriptors
    :param raw_results: return the indices of valid indexes instead of 
    :return: (2,n) 2D array of matching keypoints. 
    """
    assert len(nkp1.shape) == 1
    assert len(nkp2.shape) == 1
    valid_types = (numpy.ndarray, numpy.core.records.recarray)
    assert isinstance(nkp1, valid_types)
    assert isinstance(nkp2, valid_types)
    result = None

    desc1 = nkp1.desc
    desc2 = nkp2.desc
    big1 = desc1.astype(int)[:, numpy.newaxis, :]
    big2 = desc2.astype(int)[numpy.newaxis, :, :]
    big = abs(big1 - big2).sum(axis=-1)
    maxi = big.max(axis=-1)
    mini = big.min(axis=-1)
    amin = big.argmin(axis=-1)
    patched = big.copy()
    patched[numpy.arange(big.shape[0]), amin] = maxi
    mini2 = patched.min(axis=-1)
    ratio = mini.astype(float) / mini2
    ratio[mini2 == 0] = 1.0
    match_mask = ratio < (par.MatchRatio * par.MatchRatio)
    size = match_mask.sum()
    match = numpy.empty((size, 2), dtype=int)
    match[:, 0] = numpy.arange(nkp1.size)[match_mask]
    match[:, 1] = amin[match_mask]
    if raw_results:
        result = match
    else:
        result = numpy.recarray(shape=(size, 2), dtype=MatchPlan.dtype_kp)

        result[:, 0] = nkp1[match[:, 0]]
        result[:, 1] = nkp2[match[:, 1]]
    return result


def demo():
    import scipy.misc
    from .plan import SiftPlan
    if hasattr(scipy.misc, "ascent"):
        img1 = scipy.misc.ascent()
    else:
        img1 = scipy.misc.lena()

    splan = SiftPlan(template=img1)
    kp1 = splan(img1)
    img2 = numpy.zeros_like(img1)
    img2[5:, 8:] = img1[:-5, :-8]
    kp2 = splan(img2)
    mp = MatchPlan()
    match = mp(kp1, kp2)
    print(match.shape)
    print(numpy.median(match[:, 0].x - match[:, 1].x))
    print(numpy.median(match[:, 0].y - match[:, 1].y))

if __name__ == "__main__":
    demo()
