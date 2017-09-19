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
Simple usage example of the SIRT algorithm
"""
__authors__ = ["Pierre Paleo"]
__license__ = "MIT"
__date__ = "19/09/2017"

from silx.opencl.projection import Projection
from silx.opencl.reconstruction import SIRT
from silx.image.phantomgenerator import getMRIBrainPhantom
import numpy as np


def main():
    # Generate a sinogram of width 512 from the MRI brain phantom
    phantom = getMRIBrainPhantom()
    n_angles = 40
    P = Projection(phantom.shape, n_angles)
    sino = P(phantom)

    # Instantiate SIRT
    sirt_algo = SIRT(sino.shape)

    # Run it on the current sinogram
    n_it = 150
    res_gpu = sirt_algo(sino, n_it)
    res = res_gpu.get()


if __name__ == "__main__":
    main()
