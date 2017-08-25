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
"""Tests for commonh5 wrapper"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/08/2017"

import logging
import numpy
import unittest

_logger = logging.getLogger(__name__)

import silx.io

try:
    import h5py
except ImportError:
    h5py = None

try:
    from .. import commonh5
except ImportError:
    commonh5 = None


class TestCommonH5(unittest.TestCase):

    def setUp(self):
        if h5py is None:
            self.skipTest("h5py is needed")
        if commonh5 is None:
            self.skipTest("silx.io.commonh5 is needed")

    def test_file(self):
        node = commonh5.File("foo")
        self.assertTrue(silx.io.is_file(node))
        self.assertTrue(silx.io.is_group(node))
        self.assertFalse(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)

    def test_group(self):
        node = commonh5.Group("foo")
        self.assertFalse(silx.io.is_file(node))
        self.assertTrue(silx.io.is_group(node))
        self.assertFalse(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)

    def test_dataset(self):
        node = commonh5.Dataset("foo", data=numpy.array([1]))
        self.assertFalse(silx.io.is_file(node))
        self.assertFalse(silx.io.is_group(node))
        self.assertTrue(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)

    def test_node_attrs(self):
        node = commonh5.Node("Foo", attrs={"a": 1})
        self.assertEqual(node.attrs["a"], 1)
        node.attrs["b"] = 8
        self.assertEqual(node.attrs["b"], 8)
        node.attrs["b"] = 2
        self.assertEqual(node.attrs["b"], 2)

    def test_node_readonly_attrs(self):
        f = commonh5.File(name="Foo", mode="r")
        node = commonh5.Node("Foo", attrs={"a": 1})
        node.attrs["b"] = 8
        f.add_node(node)
        self.assertEqual(node.attrs["b"], 8)
        try:
            node.attrs["b"] = 1
            self.fail()
        except RuntimeError:
            pass

    def test_create_dataset(self):
        f = commonh5.File(name="Foo", mode="w")
        node = f.create_dataset("foo", data=numpy.array([1]))
        self.assertIs(node.parent, f)
        self.assertIs(f["foo"], node)

    def test_create_group(self):
        f = commonh5.File(name="Foo", mode="w")
        node = f.create_group("foo")
        self.assertIs(node.parent, f)
        self.assertIs(f["foo"], node)

    def test_readonly_create_dataset(self):
        f = commonh5.File(name="Foo", mode="r")
        try:
            f.create_dataset("foo", data=numpy.array([1]))
            self.fail()
        except RuntimeError:
            pass

    def test_readonly_create_group(self):
        f = commonh5.File(name="Foo", mode="r")
        try:
            f.create_group("foo")
            self.fail()
        except RuntimeError:
            pass

    def test_create_unicode_dataset(self):
        f = commonh5.File(name="Foo", mode="w")
        try:
            f.create_dataset("foo", data=numpy.array(u"aaaa"))
            self.fail()
        except TypeError:
            pass


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestCommonH5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
