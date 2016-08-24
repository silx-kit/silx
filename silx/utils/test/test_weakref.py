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
"""Tests for weakref module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/08/2016"


import unittest
from .. import weakref


class Dummy(object):
    """Dummy class to use it as geanie pig"""
    def inc(self, a):
        return a + 1


def dummy_inc(a):
    """Dummy function to use it as geanie pig"""
    return a + 1


class TestWeakMethod(unittest.TestCase):
    """Tests for weakref.WeakMethod"""

    def testMethod(self):
        dummy = Dummy()
        callable = weakref.WeakMethod(dummy.inc)
        self.assertEquals(callable()(10), 11)

    def testMethodWithDeadObject(self):
        dummy = Dummy()
        callable = weakref.WeakMethod(dummy.inc)
        dummy = None
        self.assertIsNone(callable())

    def testMethodWithDeadFunction(self):
        dummy = Dummy()
        dummy.inc2 = lambda self, a: a + 1
        callable = weakref.WeakMethod(dummy.inc2)
        dummy.inc2 = None
        self.assertIsNone(callable())

    def testFunction(self):
        callable = weakref.WeakMethod(dummy_inc)
        self.assertEquals(callable()(10), 11)

    def testDeadFunction(self):
        def inc(a):
            return a + 1
        callable = weakref.WeakMethod(inc)
        inc = None
        self.assertIsNone(callable())

    def testLambda(self):
        store = lambda a: a + 1
        callable = weakref.WeakMethod(store)
        self.assertEquals(callable()(10), 11)

    def testDeadLambda(self):
        callable = weakref.WeakMethod(lambda a: a + 1)
        self.assertIsNone(callable())

    def testCallbackOnDeadObject(self):
        self.__count = 0
        def callback(ref):
            self.__count += 1
            self.assertIs(callable, ref)
        dummy = Dummy()
        callable = weakref.WeakMethod(dummy.inc, callback)
        dummy = None
        self.assertEquals(self.__count, 1)

    def testCallbackOnDeadMethod(self):
        self.__count = 0
        def callback(ref):
            self.__count += 1
            self.assertIs(callable, ref)
        dummy = Dummy()
        dummy.inc2 = lambda self, a: a + 1
        callable = weakref.WeakMethod(dummy.inc2, callback)
        dummy.inc2 = None
        self.assertEquals(self.__count, 1)

    def testCallbackOnDeadFunction(self):
        self.__count = 0
        def callback(ref):
            self.__count += 1
            self.assertIs(callable, ref)
        store = lambda a: a + 1
        callable = weakref.WeakMethod(lambda a: a + 1, callback)
        store = None
        self.assertEquals(self.__count, 1)

    def testEquals(self):
        dummy = Dummy()
        callable1 = weakref.WeakMethod(dummy.inc)
        callable2 = weakref.WeakMethod(dummy.inc)
        self.assertEquals(callable1, callable2)

    def testInSet(self):
        callable_set = set([])
        dummy = Dummy()
        callable_set.add(weakref.WeakMethod(dummy.inc))
        callable = weakref.WeakMethod(dummy.inc)
        self.assertIn(callable, callable_set)

    def testInDict(self):
        callable_dict = {}
        dummy = Dummy()
        callable_dict[weakref.WeakMethod(dummy.inc)] = 10
        callable = weakref.WeakMethod(dummy.inc)
        self.assertEquals(callable_dict.get(callable), 10)

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestWeakMethod))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
