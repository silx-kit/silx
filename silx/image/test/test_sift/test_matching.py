#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for image kernels
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-05-28"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import time, os, logging
import numpy
import pyopencl, pyopencl.array
import scipy, scipy.misc, scipy.ndimage, pylab
import sys
import unittest
from utilstest import UtilsTest, getLogger, ctx
from test_image_functions import * #for Python implementation of tested functions
from test_image_setup import *
import sift_pyocl as sift
from sift_pyocl.utils import calc_size
logger = getLogger(__file__)
if logger.getEffectiveLevel() <= logging.INFO:
    PROFILE = True
    queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
    import pylab
else:
    PROFILE = False
    queue = pyopencl.CommandQueue(ctx)

SHOW_FIGURES = False
PRINT_KEYPOINTS = False
USE_CPU = False
USE_CPP_SIFT = True #use reference cplusplus implementation for descriptors comparison... not valid for (octsize,scale)!=(1,1)



print "working on %s" % ctx.devices[0].name

'''
For Python implementation of tested functions, see "test_image_functions.py"
'''



class test_matching(unittest.TestCase):
    def setUp(self):
        
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), ("matching_gpu.cl" if not(USE_CPU) else "matching_cpu.cl"))
        print kernel_path
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build() #.build('-D WORKGROUP_SIZE=%s' % wg_size)
        self.wg = (1, 128)



    def tearDown(self):
        self.mat = None
        self.program = None
        
        

    def test_matching(self):
        '''
        tests keypoints matching kernel
        '''    
        image = scipy.misc.lena().astype(numpy.float32)
        try:
            import feature
        except:
            logger.error("WARNING: feature module is not available to compare results with C++ implementation. Matching cannot be tested.")
            feature = None
        
        if (feature != None):
            #get the struct keypoints : (x,y,s,angle,[descriptors])
            sc = feature.SiftAlignment()
            ref_sift = sc.sift(image)
            ref_sift_2 = numpy.recarray((ref_sift.shape),dtype=ref_sift.dtype)
            ref_sift_2[:] = (ref_sift[::-1])
            t0_matching = time.time()
            siftmatch = feature.sift_match(ref_sift, ref_sift_2)
            t1_matching = time.time()
            ref = ref_sift.desc
            
            if (USE_CPU): wg = 1,
            else: wg = 64,
            shape = ref_sift.shape[0]*wg[0],
            
            ratio_th = numpy.float32(0.5329) #sift.cpp : 0.73*0.73
            keypoints_start, keypoints_end = 0, min(ref_sift.shape[0],ref_sift_2.shape[0])
            
            gpu_keypoints1 = pyopencl.array.to_device(queue, ref_sift)
            gpu_keypoints2 = pyopencl.array.to_device(queue, ref_sift_2)
            gpu_matchings = pyopencl.array.zeros(queue, (keypoints_end-keypoints_start,2),dtype=numpy.int32, order="C")
            keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
            nb_keypoints = numpy.int32(10000)
            counter = pyopencl.array.zeros(queue, (1,1),dtype=numpy.int32, order="C")

            t0 = time.time()
            k1 = self.program.matching(queue, shape, wg,
                    gpu_keypoints1.data, gpu_keypoints2.data, gpu_matchings.data, counter.data,
                    nb_keypoints, ratio_th, keypoints_end, keypoints_end)
            res = gpu_matchings.get()
            cnt = counter.get()
            t1 = time.time()

    #        ref_python, nb_match = my_matching(kp1, kp2, keypoints_start, keypoints_end)
            t2 = time.time()
            
            res_sort = res[numpy.argsort(res[:,1])]
    #        ref_sort = ref[numpy.argsort(ref[:,1])]
            
            print res[0:20]
            print ""
    #        print ref_sort[0:20]
            print("C++ Matching took %.3f ms" %(1000.0*(t1_matching-t0_matching)))
            print("OpenCL: %d match / C++ : %d match" %(cnt,siftmatch.shape[0]))


            #sort to compare added keypoints
            '''
            delta = abs(res_sort-ref_sort).max()
            self.assert_(delta == 0, "delta=%s" % (delta)) #integers
            logger.info("delta=%s" % delta)
            '''

            if PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Matching took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
                
                
            
            

def test_suite_matching():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_matching("test_matching"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_matching()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

