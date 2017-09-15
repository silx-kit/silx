#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""
Simple example of the Backprojection utilisation
"""
__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "12/09/2017"

from silx.opencl import backprojection, projection
from silx.image.phantomgenerator import PhantomGenerator
import numpy


phantom_width = 128
angles = numpy.linspace(0, numpy.pi, num=180)

phantom = PhantomGenerator.get2DPhantomSheppLogan(phantom_width)

# Create the projection geometry
proj = projection.Projection(phantom.shape, angles=angles)
# Generate the sinogram using the forward projector
sino = proj.projection(phantom)

# Define the tomography geometry.
# By default, the angles series is [0, pi] with sino.shape[0] angles
# and the rotation center is (sino.shape[1]-1.)/2
tomo_geometry = backprojection.Backprojection(sino.shape)
# Reconstruct (fbp) with this geometry
result = tomo_geometry.filtered_backprojection(sino)
