#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
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
Contains the default parameters for the SIFT algorithm
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/02/2018"
__status__ = "beta"


class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

par = Enum(OctaveMax=100000,
           DoubleImSize=0,
           order=3,
           InitSigma=1.6,
           BorderDist=5,
           Scales=3,
           PeakThresh=255.0 * 0.04 / 3.0,
           EdgeThresh=0.06,
           EdgeThresh1=0.08,
# To detect an edge response, we require the ratio of smallest
# to largest principle curvatures of the DOG function
# (eigenvalues of the Hessian) to be below a threshold.  For
# efficiency, we use Harris' idea of requiring the determinant to
# be above par.EdgeThresh times the squared trace, as for eigenvalues
# A and B, det = AB, trace = A+B.  So if A = 10B, then det = 10B**2,
# and trace**2 = (11B)**2 = 121B**2, so par.EdgeThresh = 10/121 =
# 0.08 to require ratio of eigenvalues less than 10.
           OriBins=36,
           OriSigma=1.5,
           OriHistThresh=0.8,
           MaxIndexVal=0.2,
           MagFactor=3,
           IndexSigma=1.0,
           IgnoreGradSign=0,
           MatchRatio=0.73,
           MatchXradius=1000000.0,
           MatchYradius=1000000.0,
           noncorrectlylocalized=0)


