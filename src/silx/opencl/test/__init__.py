# -*- coding: utf-8 -*-
#
#    Project: silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2016  European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "11/01/2019"

import os
import unittest
from . import test_addition
from . import test_medfilt
from . import test_backprojection
from . import test_projection
from . import test_linalg
from . import test_array_utils
from ..codec import test as test_codec
from . import test_image
from . import test_kahan
from . import test_stats
from . import test_convolution
from . import test_sparse


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(test_addition.suite())
    test_suite.addTests(test_medfilt.suite())
    test_suite.addTests(test_backprojection.suite())
    test_suite.addTests(test_projection.suite())
    test_suite.addTests(test_linalg.suite())
    test_suite.addTests(test_array_utils.suite())
    test_suite.addTests(test_codec.suite())
    test_suite.addTests(test_image.suite())
    test_suite.addTests(test_kahan.suite())
    test_suite.addTests(test_stats.suite())
    test_suite.addTests(test_convolution.suite())
    test_suite.addTests(test_sparse.suite())
    # Allow to remove sift from the project
    test_base_dir = os.path.dirname(__file__)
    sift_dir = os.path.join(test_base_dir, "..", "sift")
    if os.path.exists(sift_dir):
        from ..sift import test as test_sift
        test_suite.addTests(test_sift.suite())

    return test_suite
