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
__date__ = "02/10/2017"


import unittest
import pickle
import numpy
from silx.utils.proxy import Proxy, docstring


class Thing(object):

    def __init__(self, value):
        self.value = value

    def __getitem__(self, selection):
        return selection + 1

    def method(self, value):
        return value + 2


class InheritedProxy(Proxy):
    """Inheriting the proxy allow to specialisze methods"""

    def __init__(self, obj, value):
        Proxy.__init__(self, obj)
        self.value = value + 2

    def __getitem__(self, selection):
        return selection + 3

    def method(self, value):
        return value + 4


class TestProxy(unittest.TestCase):
    """Test that the proxy behave as expected"""

    def text_init(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertTrue(isinstance(p, Thing))
        self.assertTrue(isinstance(p, Proxy))

    # methods and properties

    def test_has_special_method(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertTrue(hasattr(p, "__getitem__"))

    def test_missing_special_method(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertFalse(hasattr(p, "__and__"))

    def test_method(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertEqual(p.method(10), obj.method(10))

    def test_property(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertEqual(p.value, obj.value)

    # special functions

    def test_getitem(self):
        obj = Thing(10)
        p = Proxy(obj)
        self.assertEqual(p[10], obj[10])

    def test_setitem(self):
        obj = numpy.array([10, 20, 30])
        p = Proxy(obj)
        p[0] = 20
        self.assertEqual(obj[0], 20)

    def test_slice(self):
        obj = numpy.arange(20)
        p = Proxy(obj)
        expected = obj[0:10:2]
        result = p[0:10:2]
        self.assertEqual(list(result), list(expected))

    # binary comparator methods

    def test_lt(self):
        obj = numpy.array([20])
        p = Proxy(obj)
        expected = obj < obj
        result = p < p
        self.assertEqual(result, expected)

    # binary numeric methods

    def test_add(self):
        obj = numpy.array([20])
        proxy = Proxy(obj)
        expected = obj + obj
        result = proxy + proxy
        self.assertEqual(result, expected)

    def test_iadd(self):
        expected = numpy.array([20])
        expected += 10
        obj = numpy.array([20])
        result = Proxy(obj)
        result += 10
        self.assertEqual(result, expected)

    def test_radd(self):
        obj = numpy.array([20])
        p = Proxy(obj)
        expected = 10 + obj
        result = 10 + p
        self.assertEqual(result, expected)

    # binary logical methods

    def test_and(self):
        obj = numpy.array([20])
        p = Proxy(obj)
        expected = obj & obj
        result = p & p
        self.assertEqual(result, expected)

    def test_iand(self):
        expected = numpy.array([20])
        expected &= 10
        obj = numpy.array([20])
        result = Proxy(obj)
        result &= 10
        self.assertEqual(result, expected)

    def test_rand(self):
        obj = numpy.array([20])
        p = Proxy(obj)
        expected = 10 & obj
        result = 10 & p
        self.assertEqual(result, expected)

    # unary methods

    def test_neg(self):
        obj = numpy.array([20])
        p = Proxy(obj)
        expected = -obj
        result = -p
        self.assertEqual(result, expected)

    def test_round(self):
        obj = 20.5
        p = Proxy(obj)
        expected = round(obj)
        result = round(p)
        self.assertEqual(result, expected)

    # cast

    def test_bool(self):
        obj = True
        p = Proxy(obj)
        if p:
            pass
        else:
            self.fail()

    def test_str(self):
        obj = Thing(10)
        p = Proxy(obj)
        expected = str(obj)
        result = str(p)
        self.assertEqual(result, expected)

    def test_repr(self):
        obj = Thing(10)
        p = Proxy(obj)
        expected = repr(obj)
        result = repr(p)
        self.assertEqual(result, expected)

    def test_text_bool(self):
        obj = ""
        p = Proxy(obj)
        if p:
            self.fail()
        else:
            pass

    def test_text_str(self):
        obj = "a"
        p = Proxy(obj)
        expected = str(obj)
        result = str(p)
        self.assertEqual(result, expected)

    def test_text_repr(self):
        obj = "a"
        p = Proxy(obj)
        expected = repr(obj)
        result = repr(p)
        self.assertEqual(result, expected)

    def test_hash(self):
        obj = [0, 1, 2]
        p = Proxy(obj)
        with self.assertRaises(TypeError):
            hash(p)
        obj = (0, 1, 2)
        p = Proxy(obj)
        hash(p)


class TestInheritedProxy(unittest.TestCase):
    """Test that inheriting the Proxy class behave as expected"""

    # methods and properties

    def test_method(self):
        obj = Thing(10)
        p = InheritedProxy(obj, 11)
        self.assertEqual(p.method(10), 11 + 3)

    def test_property(self):
        obj = Thing(10)
        p = InheritedProxy(obj, 11)
        self.assertEqual(p.value, 11 + 2)

    # special functions

    def test_getitem(self):
        obj = Thing(10)
        p = InheritedProxy(obj, 11)
        self.assertEqual(p[12], 12 + 3)


class TestPickle(unittest.TestCase):

    def test_dumps(self):
        obj = Thing(10)
        p = Proxy(obj)
        expected = pickle.dumps(obj)
        result = pickle.dumps(p)
        self.assertEqual(result, expected)

    def test_loads(self):
        obj = Thing(10)
        p = Proxy(obj)
        obj2 = pickle.loads(pickle.dumps(p))
        self.assertTrue(isinstance(obj2, Thing))
        self.assertFalse(isinstance(obj2, Proxy))
        self.assertEqual(obj.value, obj2.value)


class TestDocstring(unittest.TestCase):
    """Test docstring decorator"""

    class Base(object):
        def method(self):
            """Docstring"""
            pass

    def test_inheritance(self):
        class Derived(TestDocstring.Base):
            @docstring(TestDocstring.Base)
            def method(self):
                pass

        self.assertEqual(Derived.method.__doc__,
                         TestDocstring.Base.method.__doc__)

    def test_composition(self):
        class Composed(object):
            def __init__(self):
                self._base = TestDocstring.Base()

            @docstring(TestDocstring.Base)
            def method(self):
                return self._base.method()

            @docstring(TestDocstring.Base.method)
            def renamed(self):
                return self._base.method()

        self.assertEqual(Composed.method.__doc__,
                         TestDocstring.Base.method.__doc__)

        self.assertEqual(Composed.renamed.__doc__,
                         TestDocstring.Base.method.__doc__)

    def test_function(self):
        def f():
            """Docstring"""
            pass

        @docstring(f)
        def g():
            pass

        self.assertEqual(f.__doc__, g.__doc__)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestProxy))
    test_suite.addTest(loadTests(TestPickle))
    test_suite.addTest(loadTests(TestInheritedProxy))
    test_suite.addTest(loadTests(TestDocstring))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
