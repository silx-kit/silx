# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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

from libcpp.vector cimport vector as std_vector
from libcpp cimport bool

cdef extern from "mc.hpp":
    cdef cppclass MarchingCubes[FloatIn, FloatOut]:
        MarchingCubes(FloatIn level) except +
        void process(FloatIn * data,
                     unsigned int depth,
                     unsigned int height,
                     unsigned int width) except +
        void set_slice_size(unsigned int height,
                            unsigned int width)
        void process_slice(FloatIn * slice0,
                           FloatIn * slice1) except +
        void finish_process()
        void reset()

        unsigned int depth
        unsigned int height
        unsigned int width
        unsigned int sampling[3]
        FloatIn isolevel
        bool invert_normals
        std_vector[FloatOut] vertices
        std_vector[FloatOut] normals
        std_vector[unsigned int] indices
