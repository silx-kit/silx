# -*- coding: utf-8 -*-
#
#    Project: silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "12/01/2017"

import unittest
from . import test_gaussian
from . import test_preproc
from . import test_reductions
from . import test_convol
from . import test_algebra
from . import test_image
from . import test_keypoints
from . import test_matching
from . import test_align
from . import test_transform


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_algebra.suite())
    testSuite.addTest(test_gaussian.suite())
    testSuite.addTest(test_preproc.suite())
    testSuite.addTest(test_reductions.suite())
    testSuite.addTest(test_convol.suite())
    testSuite.addTest(test_image.suite())
    testSuite.addTest(test_keypoints.suite())
    testSuite.addTest(test_matching.suite())
    testSuite.addTests(test_align.suite())
    testSuite.addTests(test_transform.suite())

    return testSuite
