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
PRINT_KEYPOINTS = True
#USE_CPU = False
USE_CPP_SIFT = False #use reference cplusplus implementation for descriptors comparison... not valid for (octsize,scale)!=(1,1)



#print "working on %s" % ctx.devices[0].name

'''
For Python implementation of tested functions, see "test_image_functions.py"
'''


class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite


class test_keypoints(ParameterisedTestCase):
    def setUp(self):
        self.abort = False
        for kernel_file in self.param:
            if "cpu" in kernel_file:
                self.USE_CPU = True
            else:
                self.USE_CPU = False
            if kernel_file.startswith("orient"):
                self.wg_orient = self.param[kernel_file]
                kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), kernel_file + ".cl")
                kernel_src = open(kernel_path).read()
                try:
                    self.program_orient = pyopencl.Program(ctx, kernel_src).build()
                except:
                    logger.warning("Failed to compile kernel '%s': aborting" % kernel_file)
                    self.abort = True
                    return
            elif kernel_file.startswith("keypoint"):
                self.wg_keypoint = self.param[kernel_file]
                kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), kernel_file + ".cl")
                kernel_src = open(kernel_path).read()
                try:
                    self.program_keypoint = pyopencl.Program(ctx, kernel_src).build()
                except:
                    logger.warning("Failed to compile kernel '%s': aborting" % kernel_file)
                    self.abort = True
                    return

    def tearDown(self):
        self.mat = None
        self.program = None




    def test_orientation(self):
        '''
        #tests keypoints orientation assignment kernel
        '''
        if self.abort:
            return
        #orientation_setup :
        keypoints, nb_keypoints, updated_nb_keypoints, grad, ori, octsize = orientation_setup()
        keypoints, compact_cnt = my_compact(numpy.copy(keypoints),nb_keypoints)
        updated_nb_keypoints = compact_cnt
        
#        if (USE_CPU):
#            print("Using CPU-optimized kernels")
#            wg = 1,
#            shape = keypoints.shape[0]*wg[0],
#        else:
#            wg = 128, #FIXME : have to choose it for histograms #wg = max(self.wg),
        wg = self.wg_orient
        shape = keypoints.shape[0] * wg[0],  #shape = calc_size(keypoints.shape, self.wg)
        
        gpu_keypoints = pyopencl.array.to_device(queue, keypoints)
        actual_nb_keypoints = numpy.int32(updated_nb_keypoints)
        print("Number of keypoints before orientation assignment : %s" % actual_nb_keypoints)

        gpu_grad = pyopencl.array.to_device(queue, grad)
        gpu_ori = pyopencl.array.to_device(queue, ori)
        orisigma = numpy.float32(1.5) #SIFT
        grad_height, grad_width = numpy.int32(grad.shape)
        keypoints_start = numpy.int32(0)
        keypoints_end = numpy.int32(actual_nb_keypoints)
        counter = pyopencl.array.to_device(queue, keypoints_end) #actual_nb_keypoints)

        t0 = time.time()
        k1 = self.program_orient.orientation_assignment(queue, shape, wg,
        	gpu_keypoints.data, gpu_grad.data, gpu_ori.data, counter.data,
        	octsize, orisigma, nb_keypoints, keypoints_start, keypoints_end, grad_width, grad_height)
        res = gpu_keypoints.get()
        cnt = counter.get()
        t1 = time.time()
        
        if (USE_CPP_SIFT):
            import feature
            sc = feature.SiftAlignment()
            ref2 = sc.sift(scipy.misc.lena()) #ref2.x, ref2.y, ref2.scale, ref2.angle, ref2.desc --- ref2[numpy.argsort(ref2.y)]).desc
            ref = ref2.angle
            kp_ref = numpy.empty((ref2.size, 4), dtype=numpy.float32)
            kp_ref[:, 0] = ref2.x
            kp_ref[:, 1] = ref2.y
            kp_ref[:, 2] = ref2.scale
            kp_ref[:, 3] = ref2.angle
            
        else:
            ref, updated_nb_keypoints = my_orientation(keypoints, nb_keypoints, keypoints_start, keypoints_end, grad, ori, octsize, orisigma)
       
        t2 = time.time()
        
        if (PRINT_KEYPOINTS):
#            print("Keypoints after orientation assignment :")
#            print res[numpy.argsort(res[0:cnt,1])][0:cnt+10,3] #res[0:compact_cnt]
            print " "
#            print kp_ref[0:cnt+10]
#            print "Showing error (NOTE: significant error at position (i) should have its opposite at (i+1))"
#            print res[numpy.argsort(res[0:compact_cnt,1])][0:compact_cnt,3] - ref[0:compact_cnt]

#        print("Total keypoints for kernel : %s -- For Python : %s \t [octsize = %s]" % (cnt, updated_nb_keypoints, octsize))
#        print("Opencl found %s keypoints (%s added)" %(cnt,cnt-compact_cnt))
        
        #sort to compare added keypoints
        upbound = min(cnt,updated_nb_keypoints)
        d1, d2, d3, d4 = keypoints_compare(ref[0:upbound], res[0:upbound]) 
        self.assert_(d1 < 1e-4, "delta_cols=%s" % (d1))
        self.assert_(d2 < 1e-4, "delta_rows=%s" % (d2))
        self.assert_(d3 < 1e-4, "delta_sigma=%s" % (d3))
        self.assert_(d4 < 1e-1, "delta_angle=%s" % (d4)) #orientation has a poor precision
        logger.info("delta_cols=%s" % d1)
        logger.info("delta_rows=%s" % d2)
        logger.info("delta_sigma=%s" % d3)
        logger.info("delta_angle=%s" % d4)
        
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Orientation assignment took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
        

    def test_descriptor(self):
        '''
        #tests keypoints descriptors creation kernel
        '''
        if self.abort:
            return

        #descriptor_setup :
        keypoints_o, nb_keypoints, actual_nb_keypoints, grad, ori, octsize = descriptor_setup()
        #keypoints should be a compacted vector of keypoints
        keypoints_o, compact_cnt = my_compact(numpy.copy(keypoints_o),nb_keypoints)
        actual_nb_keypoints = compact_cnt
        keypoints_start, keypoints_end = 0, actual_nb_keypoints
        keypoints = keypoints_o[keypoints_start:keypoints_end+52] #to check if we actually stop at keypoints_end
        print("Working on keypoints : [%s,%s] (octave = %s)" % (keypoints_start, keypoints_end-1,int(numpy.log2(octsize)+1)))
        if not(USE_CPP_SIFT) and (100 < keypoints_end-keypoints_start): print "NOTE: Python implementation of descriptors is slow. Do not handle more than 100 keypoints, or grab a coffee..."
        
#        if (self.USE_CPU):
#            print "Using CPU-optimized kernels"
#            wg = 1,
#            shape = keypoints.shape[0]*wg[0],
#        else:
#            wg = (8, 8, 8)
#            shape = int(keypoints.shape[0]*wg[0]), 8, 8
#            wg = (4, 4, 8)
#            shape = int(keypoints.shape[0]*wg[0]), 4, 8
        wg = self.wg_keypoint
        if len(wg) == 1:
            shape = keypoints.shape[0] * wg[0],
        else:
            shape = keypoints.shape[0] * wg[0], wg[1], wg[2]
        gpu_keypoints = pyopencl.array.to_device(queue, keypoints)
        #NOTE: for the following line, use pyopencl.array.empty instead of pyopencl.array.zeros if the keypoints are compacted
        gpu_descriptors = pyopencl.array.zeros(queue, (keypoints_end - keypoints_start, 128), dtype=numpy.uint8, order="C")
        gpu_grad = pyopencl.array.to_device(queue, grad)
        gpu_ori = pyopencl.array.to_device(queue, ori)

        keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
        grad_height, grad_width = numpy.int32(grad.shape)
        counter = pyopencl.array.to_device(queue, keypoints_end)
        
        t0 = time.time()
        k1 = self.program_keypoint.descriptor(queue, shape, wg,
            gpu_keypoints.data, gpu_descriptors.data, gpu_grad.data, gpu_ori.data, numpy.int32(octsize),
            keypoints_start, counter.data, grad_width, grad_height)
        try:
            res = gpu_descriptors.get()
        except (pyopencl.LogicError, RuntimeError) as error:
            logger.warning("Segmentation fault like error (%s) on Descriptor for %s" % (error, self.param))
            return
        t1 = time.time()

        if (USE_CPP_SIFT):
            import feature
            sc = feature.SiftAlignment()
            ref2 = sc.sift(scipy.misc.lena()) #ref2.x, ref2.y, ref2.scale, ref2.angle, ref2.desc --- ref2[numpy.argsort(ref2.y)]).desc
            ref = ref2.desc
            ref_sort = ref
        else:
            ref = my_descriptor(keypoints_o, grad, ori, octsize, keypoints_start, keypoints_end)
            ref_sort = ref[numpy.argsort(keypoints[keypoints_start:keypoints_end,1])]

        t2 = time.time()
        
        if (PRINT_KEYPOINTS):
            res_sort = res[numpy.argsort(keypoints[keypoints_start:keypoints_end,1])]
            print res_sort[5:10]#keypoints_end-keypoints_start,0:15]
#            print res_sort[9]
            print ""
            print ref_sort[5:10]
#            numpy.savetxt("grrr_ocl_4_3.txt",res_sort,fmt='%d')
#            numpy.savetxt("grrr_cpp_4_3.txt",ref_sort,fmt='%d')
#            print ref[50:80,0:15]#[0:keypoints_end-keypoints_start,0:15]
            if (USE_CPP_SIFT and octsize == 1) or not(USE_CPP_SIFT): #this comparison is only relevant for the first keypoints
                print "Comparing descriptors (OpenCL and cpp) :"
                match, nulldesc = descriptors_compare(ref[keypoints_start:keypoints_end],res)
                print ("%s/%s match found" %(match,(keypoints_end-keypoints_start)-nulldesc))
#            print ref[1,:]
#            print res[1,:].sum(), ref[1,:].sum()

             #append to existing text file
#            f_handle = file('desc_by_test_keypoints.txt', 'a')
#            numpy.savetxt(f_handle,res_sort,fmt='%d')
#            f_handle.close()


        '''
            For now, the descriptor kernel is not precise enough to get exactly the same descriptors values 
        (we have several difference of 1, but it is OK for the SIFT matching).
            Use descriptors_compare(ref,res) to count how many descriptors are exactly the same.
        
        #sort to compare added keypoints
        delta = abs(res_sort-ref_sort).max()
        self.assert_(delta <= 1, "delta=%s" % (delta))
        logger.info("delta=%s" % delta)
        '''

        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Descriptors computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            
            
            
            

def test_suite_keypoints():
    testSuite = unittest.TestSuite()
    TESTCASES = [{"orientation_gpu":(128,), "keypoints_gpu2":(8, 8, 8)},
                 {"orientation_cpu":(1,), "keypoints_cpu":(1,)},
                  {"orientation_gpu":(128,), "keypoints_gpu1":(4, 4, 8)},

                  ]
    for param in TESTCASES:
        testSuite.addTest(ParameterisedTestCase.parameterise(
                test_keypoints, param))

#    testSuite.addTest(test_keypoints("test_orientation"))
#    testSuite.addTest(test_keypoints("test_descriptor"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_keypoints()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

