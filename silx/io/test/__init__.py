# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "08/12/2017"

import unittest

from .test_specfile import suite as test_specfile_suite
from .test_specfilewrapper import suite as test_specfilewrapper_suite
from .test_dictdump import suite as test_dictdump_suite
from .test_spech5 import suite as test_spech5_suite
from .test_spectoh5 import suite as test_spectoh5_suite
from .test_octaveh5 import suite as test_octaveh5_suite
from .test_fabioh5 import suite as test_fabioh5_suite
from .test_utils import suite as test_utils_suite
from .test_nxdata import suite as test_nxdata_suite
from .test_commonh5 import suite as test_commonh5_suite
from .test_rawh5 import suite as test_rawh5_suite
from .test_url import suite as test_url_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_dictdump_suite())
    test_suite.addTest(test_specfile_suite())
    test_suite.addTest(test_specfilewrapper_suite())
    test_suite.addTest(test_spech5_suite())
    test_suite.addTest(test_spectoh5_suite())
    test_suite.addTest(test_octaveh5_suite())
    test_suite.addTest(test_utils_suite())
    test_suite.addTest(test_fabioh5_suite())
    test_suite.addTest(test_nxdata_suite())
    test_suite.addTest(test_commonh5_suite())
    test_suite.addTest(test_rawh5_suite())
    test_suite.addTest(test_url_suite())
    return test_suite
