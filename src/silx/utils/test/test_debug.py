# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Tests for debug module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "27/02/2018"


import unittest
from silx.utils import debug
from silx.utils import testutils


@debug.log_all_methods
class _Foobar(object):

    def a(self):
        return None

    def b(self):
        return self.a()

    def random_args(self, *args, **kwargs):
        return args, kwargs

    def named_args(self, a, b):
        return a + 1, b + 1


class TestDebug(unittest.TestCase):
    """Tests for debug module."""

    def logB(self):
        """
        Can be used to check the log output using:
        `./run_tests.py silx.utils.test.test_debug.TestDebug.logB -v`
        """
        print()
        test = _Foobar()
        test.b()

    @testutils.test_logging(debug.debug_logger.name, warning=2)
    def testMethod(self):
        test = _Foobar()
        test.a()

    @testutils.test_logging(debug.debug_logger.name, warning=4)
    def testInterleavedMethod(self):
        test = _Foobar()
        test.b()

    @testutils.test_logging(debug.debug_logger.name, warning=2)
    def testNamedArgument(self):
        # Arguments arre still provided to the patched method
        test = _Foobar()
        result = test.named_args(10, 11)
        self.assertEqual(result, (11, 12))

    @testutils.test_logging(debug.debug_logger.name, warning=2)
    def testRandomArguments(self):
        # Arguments arre still provided to the patched method
        test = _Foobar()
        result = test.random_args("foo", 50, a=10, b=100)
        self.assertEqual(result[0], ("foo", 50))
        self.assertEqual(result[1], {"a": 10, "b": 100})


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestDebug))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
