# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
__date__ = "15/09/2016"


import unittest
from .. import weakref


class Dummy(object):
    """Dummy class to use it as geanie pig"""
    def inc(self, a):
        return a + 1

    def __lt__(self, other):
        return True


def dummy_inc(a):
    """Dummy function to use it as geanie pig"""
    return a + 1


class TestWeakMethod(unittest.TestCase):
    """Tests for weakref.WeakMethod"""

    def testMethod(self):
        dummy = Dummy()
        callable_ = weakref.WeakMethod(dummy.inc)
        self.assertEqual(callable_()(10), 11)

    def testMethodWithDeadObject(self):
        dummy = Dummy()
        callable_ = weakref.WeakMethod(dummy.inc)
        dummy = None
        self.assertIsNone(callable_())

    def testMethodWithDeadFunction(self):
        dummy = Dummy()
        dummy.inc2 = lambda self, a: a + 1
        callable_ = weakref.WeakMethod(dummy.inc2)
        dummy.inc2 = None
        self.assertIsNone(callable_())

    def testFunction(self):
        callable_ = weakref.WeakMethod(dummy_inc)
        self.assertEqual(callable_()(10), 11)

    def testDeadFunction(self):
        def inc(a):
            return a + 1
        callable_ = weakref.WeakMethod(inc)
        inc = None
        self.assertIsNone(callable_())

    def testLambda(self):
        store = lambda a: a + 1  # noqa: E731
        callable_ = weakref.WeakMethod(store)
        self.assertEqual(callable_()(10), 11)

    def testDeadLambda(self):
        callable_ = weakref.WeakMethod(lambda a: a + 1)
        self.assertIsNone(callable_())

    def testCallbackOnDeadObject(self):
        self.__count = 0

        def callback(ref):
            self.__count += 1
            self.assertIs(callable_, ref)
        dummy = Dummy()
        callable_ = weakref.WeakMethod(dummy.inc, callback)
        dummy = None
        self.assertEqual(self.__count, 1)

    def testCallbackOnDeadMethod(self):
        self.__count = 0

        def callback(ref):
            self.__count += 1
            self.assertIs(callable_, ref)
        dummy = Dummy()
        dummy.inc2 = lambda self, a: a + 1
        callable_ = weakref.WeakMethod(dummy.inc2, callback)
        dummy.inc2 = None
        self.assertEqual(self.__count, 1)

    def testCallbackOnDeadFunction(self):
        self.__count = 0

        def callback(ref):
            self.__count += 1
            self.assertIs(callable_, ref)
        store = lambda a: a + 1  # noqa: E731
        callable_ = weakref.WeakMethod(store, callback)
        store = None
        self.assertEqual(self.__count, 1)

    def testEquals(self):
        dummy = Dummy()
        callable1 = weakref.WeakMethod(dummy.inc)
        callable2 = weakref.WeakMethod(dummy.inc)
        self.assertEqual(callable1, callable2)

    def testInSet(self):
        callable_set = set([])
        dummy = Dummy()
        callable_set.add(weakref.WeakMethod(dummy.inc))
        callable_ = weakref.WeakMethod(dummy.inc)
        self.assertIn(callable_, callable_set)

    def testInDict(self):
        callable_dict = {}
        dummy = Dummy()
        callable_dict[weakref.WeakMethod(dummy.inc)] = 10
        callable_ = weakref.WeakMethod(dummy.inc)
        self.assertEqual(callable_dict.get(callable_), 10)


class TestWeakMethodProxy(unittest.TestCase):

    def testMethod(self):
        dummy = Dummy()
        callable_ = weakref.WeakMethodProxy(dummy.inc)
        self.assertEqual(callable_(10), 11)

    def testMethodWithDeadObject(self):
        dummy = Dummy()
        method = weakref.WeakMethodProxy(dummy.inc)
        dummy = None
        self.assertRaises(ReferenceError, method, 9)


class TestWeakList(unittest.TestCase):
    """Tests for weakref.WeakList"""

    def setUp(self):
        self.list = weakref.WeakList()
        self.object1 = Dummy()
        self.object2 = Dummy()
        self.list.append(self.object1)
        self.list.append(self.object2)

    def testAppend(self):
        obj = Dummy()
        self.list.append(obj)
        self.assertEqual(len(self.list), 3)
        obj = None
        self.assertEqual(len(self.list), 2)

    def testRemove(self):
        self.list.remove(self.object1)
        self.assertEqual(len(self.list), 1)

    def testPop(self):
        obj = self.list.pop(0)
        self.assertIs(obj, self.object1)
        self.assertEqual(len(self.list), 1)

    def testGetItem(self):
        self.assertIs(self.object1, self.list[0])

    def testGetItemSlice(self):
        objects = self.list[:]
        self.assertEqual(len(objects), 2)
        self.assertIs(self.object1, objects[0])
        self.assertIs(self.object2, objects[1])

    def testIter(self):
        obj_list = list(self.list)
        self.assertEqual(len(obj_list), 2)
        self.assertIs(self.object1, obj_list[0])

    def testLen(self):
        self.assertEqual(len(self.list), 2)

    def testSetItem(self):
        obj = Dummy()
        self.list[0] = obj
        self.assertIsNot(self.object1, self.list[0])
        obj = None
        self.assertEqual(len(self.list), 1)

    def testSetItemSlice(self):
        obj = Dummy()
        self.list[:] = [obj, obj]
        self.assertEqual(len(self.list), 2)
        self.assertIs(obj, self.list[0])
        self.assertIs(obj, self.list[1])
        obj = None
        self.assertEqual(len(self.list), 0)

    def testDelItem(self):
        del self.list[0]
        self.assertEqual(len(self.list), 1)
        self.assertIs(self.object2, self.list[0])

    def testDelItemSlice(self):
        del self.list[:]
        self.assertEqual(len(self.list), 0)

    def testContains(self):
        self.assertIn(self.object1, self.list)

    def testAdd(self):
        others = [Dummy()]
        l = self.list + others
        self.assertIs(l[0], self.object1)
        self.assertEqual(len(l), 3)
        others = None
        self.assertEqual(len(l), 2)

    def testExtend(self):
        others = [Dummy()]
        self.list.extend(others)
        self.assertIs(self.list[0], self.object1)
        self.assertEqual(len(self.list), 3)
        others = None
        self.assertEqual(len(self.list), 2)

    def testIadd(self):
        others = [Dummy()]
        self.list += others
        self.assertIs(self.list[0], self.object1)
        self.assertEqual(len(self.list), 3)
        others = None
        self.assertEqual(len(self.list), 2)

    def testMul(self):
        l = self.list * 2
        self.assertIs(l[0], self.object1)
        self.assertEqual(len(l), 4)
        self.object1 = None
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], self.object2)
        self.assertIs(l[1], self.object2)

    def testImul(self):
        self.list *= 2
        self.assertIs(self.list[0], self.object1)
        self.assertEqual(len(self.list), 4)
        self.object1 = None
        self.assertEqual(len(self.list), 2)
        self.assertIs(self.list[0], self.object2)
        self.assertIs(self.list[1], self.object2)

    def testCount(self):
        self.list.append(self.object2)
        self.assertEqual(self.list.count(self.object1), 1)
        self.assertEqual(self.list.count(self.object2), 2)

    def testIndex(self):
        self.assertEqual(self.list.index(self.object1), 0)
        self.assertEqual(self.list.index(self.object2), 1)

    def testInsert(self):
        obj = Dummy()
        self.list.insert(1, obj)
        self.assertEqual(len(self.list), 3)
        self.assertIs(self.list[1], obj)
        obj = None
        self.assertEqual(len(self.list), 2)

    def testReverse(self):
        self.list.reverse()
        self.assertEqual(len(self.list), 2)
        self.assertIs(self.list[0], self.object2)
        self.assertIs(self.list[1], self.object1)

    def testReverted(self):
        new_list = reversed(self.list)
        self.assertEqual(len(new_list), 2)
        self.assertIs(self.list[1], self.object2)
        self.assertIs(self.list[0], self.object1)
        self.assertIs(new_list[0], self.object2)
        self.assertIs(new_list[1], self.object1)
        self.object1 = None
        self.assertEqual(len(new_list), 1)

    def testStr(self):
        self.assertNotEqual(self.list.__str__(), "[]")

    def testRepr(self):
        self.assertNotEqual(self.list.__repr__(), "[]")

    def testSort(self):
        # only a coverage
        self.list.sort()
        self.assertEqual(len(self.list), 2)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestWeakMethod))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestWeakMethodProxy))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestWeakList))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
