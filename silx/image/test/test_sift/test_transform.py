#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for transformation kernel
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
IMAGE_RESHAPE = True
USE_LENA = False

print "working on %s" % ctx.devices[0].name





class test_transform(unittest.TestCase):
    def setUp(self):
        
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "transform.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build() #.build('-D WORKGROUP_SIZE=%s' % wg_size)
        self.wg = (1, 128)



    def tearDown(self):
        self.program = None
        

    
    
    def image_reshape(self,img,output_height,output_width,image_height, image_width):
        '''
        Reshape the image to get a bigger image with the input image in the center
        
        '''
        image3 = numpy.zeros((output_height,output_width),dtype=numpy.float32)
        d1 = (output_width - image_width)/2
        d0 = (output_height - image_height)/2
        image3[d0:-d0,d1:-d1] = numpy.copy(img)
        image = image3
        image_height, image_width = output_height, output_width
        return image, image_height, image_width
    
    
    
    
    
    
    
    def matching_correction(self,image,image2):
        '''
        Computes keypoints for two images and try to align image2 on image1
        '''
        #computing keypoints matching
        s = sift.SiftPlan(template=image,devicetype="gpu")
        kp1 = s.keypoints(image)
        kp2 = s.keypoints(image2) #image2 and image must have the same size
        m = sift.MatchPlan(devicetype="GPU")
        matching = m.match(kp2,kp1)
        N = matching.shape[0]
        #solving normals equations for least square fit
        X = numpy.zeros((2*N,6))
        X[::2,2:] = 1,0,0,0
        X[::2,0] = matching.x[:,0]
        X[::2,1] = matching.y[:,0]
        X[1::2,0:3] = 0,0,0
        X[1::2,3] = matching.x[:,0]
        X[1::2,4] = matching.y[:,0]
        X[1::2,5] = 1
        y = numpy.zeros((2*N,1))
        y[::2,0] = matching.x[:,1]
        y[1::2,0] = matching.y[:,1]
        #A = numpy.dot(X.transpose(),X)
        #sol = numpy.dot(numpy.linalg.inv(A),numpy.dot(X.transpose(),y))
        sol = numpy.dot(numpy.linalg.pinv(X),y)
        MSE = numpy.linalg.norm(y - numpy.dot(X,sol))**2/N #value of the sum of residuals at "sol"
        return sol, MSE






    def test_transform(self):
        '''
        tests transform kernel
        '''    



        if (USE_LENA):
            #original image
            image = scipy.misc.lena().astype(numpy.float32)
            image = numpy.ascontiguousarray(image[0:512,0:512])
            image_height, image_width = image.shape
            #transformation
            angle = 1.9 #numpy.pi/5.0
    #        matrix = numpy.array([[numpy.cos(angle),-numpy.sin(angle)],[numpy.sin(angle),numpy.cos(angle)]],dtype=numpy.float32)
    #        offset_value = numpy.array([1000.0, 100.0],dtype=numpy.float32)
    #        matrix = numpy.array([[0.9,0.2],[-0.4,0.9]],dtype=numpy.float32)
    #        offset_value = numpy.array([-20.0,256.0],dtype=numpy.float32)
            matrix = numpy.array([[1.0,-0.75],[0.7,0.5]],dtype=numpy.float32)
            
            offset_value = numpy.array([250.0, -150.0],dtype=numpy.float32)
           
            image2 = scipy.ndimage.interpolation.affine_transform(image,matrix,offset=offset_value,order=1, mode="constant")
        
        else: #use images of a stack
            image = scipy.misc.imread("/home/paleo/Titanium/test/frame0.png")
            image2 = scipy.misc.imread("/home/paleo/Titanium/test/frame1.png")
            offset_value = numpy.array([0.0, 0.0],dtype=numpy.float32)
            image_height, image_width = image.shape
            image2_height, image2_width = image2.shape
            
        fill_value = numpy.float32(0.0)
        mode = numpy.int32(1)   
            
        if IMAGE_RESHAPE: #turns out that image should always be reshaped
            output_height, output_width = int(3000), int(3000)
            image, image_height, image_width = self.image_reshape(image,output_height,output_width,image_height,image_width)
            image2, image2_height, image2_width = self.image_reshape(image2,output_height,output_width,image2_height,image2_width) 
            
            
       
        else: output_height, output_width = int(image_height*numpy.sqrt(2)),int(image_width*numpy.sqrt(2))
        print "Image : (%s, %s) -- Output: (%s, %s)" %(image_height, image_width , output_height, output_width)
        
        
        
            
        
        
        
        
        
        
        
        #perform correction by least square
        sol, MSE = self.matching_correction(image,image2)
        print sol
        
        
        correction_matrix = numpy.zeros((2,2),dtype=numpy.float32)
        correction_matrix[0] = sol[0:2,0]
        correction_matrix[1] = sol[3:5,0]
        matrix_for_gpu = correction_matrix.reshape(4,1) #for float4 struct
        offset_value[0] = sol[2,0]
        offset_value[1] = sol[5,0]
        
        wg = 8,8
        shape = calc_size((output_width,output_height), wg)
        gpu_image = pyopencl.array.to_device(queue, image2)
        gpu_output = pyopencl.array.empty(queue, (output_height, output_width), dtype=numpy.float32, order="C")
        gpu_matrix = pyopencl.array.to_device(queue,matrix_for_gpu)
        gpu_offset = pyopencl.array.to_device(queue,offset_value)
        image_height, image_width = numpy.int32((image_height, image_width))
        output_height, output_width = numpy.int32((output_height, output_width))
        
        t0 = time.time()
        k1 = self.program.transform(queue, shape, wg,
                gpu_image.data, gpu_output.data, gpu_matrix.data, gpu_offset.data, 
                image_width, image_height, output_width, output_height, fill_value, mode)
        res = gpu_output.get()
        t1 = time.time()
#        print res[0,0]
        
        ref = scipy.ndimage.interpolation.affine_transform(image2,correction_matrix,
            offset=offset_value, output_shape=(output_height,output_width),order=1, mode="constant", cval=fill_value)
        t2 = time.time()
        
        delta = abs(res-image)
        delta_arg = delta.argmax()
        delta_max = delta.max()
#        delta_mse_res = ((res-image)**2).sum()/image.size
#        delta_mse_ref = ((ref-image)**2).sum()/image.size
        at_0, at_1 = delta_arg/output_width, delta_arg%output_width
        print("Max error: %f at (%d, %d)" %(delta_max, at_0, at_1))
#        print("Mean Squared Error Res/Original : %f" %(delta_mse_res))
#        print("Mean Squared Error Ref/Original: %f" %(delta_mse_ref))
        print("minimal MSE according to least squares : %f" %MSE)
#        print res[at_0,at_1]
#        print ref[at_0,at_1]
        
        SHOW_FIGURES = True
        if SHOW_FIGURES:
            fig = pylab.figure()
            sp1 = fig.add_subplot(221,title="Input image")
            sp1.imshow(image, interpolation="nearest")
            sp2 = fig.add_subplot(222,title="Image after deformation")
            sp2.imshow(image2, interpolation="nearest")
            sp2 = fig.add_subplot(223,title="Corrected image (OpenCL)")
            sp2.imshow(res, interpolation="nearest")
            sp2 = fig.add_subplot(224,title="Corrected image (Scipy)")
            sp2.imshow(ref, interpolation="nearest")
#            sp2.imshow(ref, interpolation="nearest")
#            sp3 = fig.add_subplot(223,title="delta (max = %f)" %delta_max)
#            sh3 = sp3.imshow(delta[:,:], interpolation="nearest")
#            cbar = fig.colorbar(sh3)
            fig.show()
            raw_input("enter")


        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Transformation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            










            
            
            

def test_suite_transform():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_transform("test_transform"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_transform()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

