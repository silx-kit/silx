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
__date__ = "21/09/2017"

import logging
import numpy
import unittest
import tempfile
import shutil

_logger = logging.getLogger(__name__)

import silx.io
import silx.io.utils
import h5py

try:
    from .. import commonh5
except ImportError:
    commonh5 = None


class TestCommonFeatures(unittest.TestCase):
    """Test common features supported by h5py and our implementation."""

    @classmethod
    def createFile(cls):
        return None

    @classmethod
    def setUpClass(cls):
        # Set to None cause create_resource can raise an excpetion
        cls.h5 = None
        cls.h5 = cls.create_resource()
        if cls.h5 is None:
            raise unittest.SkipTest("File not created")

    @classmethod
    def create_resource(cls):
        """Must be implemented"""
        return None

    @classmethod
    def tearDownClass(cls):
        cls.h5 = None

    def test_file(self):
        node = self.h5
        self.assertTrue(silx.io.is_file(node))
        self.assertTrue(silx.io.is_group(node))
        self.assertFalse(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)

    def test_group(self):
        node = self.h5["group"]
        self.assertFalse(silx.io.is_file(node))
        self.assertTrue(silx.io.is_group(node))
        self.assertFalse(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)
        class_ = self.h5.get("group", getclass=True)
        classlink = self.h5.get("group", getlink=True, getclass=True)
        self.assertEqual(class_, h5py.Group)
        self.assertEqual(classlink, h5py.HardLink)

    def test_dataset(self):
        node = self.h5["group/dataset"]
        self.assertFalse(silx.io.is_file(node))
        self.assertFalse(silx.io.is_group(node))
        self.assertTrue(silx.io.is_dataset(node))
        self.assertEqual(len(node.attrs), 0)
        class_ = self.h5.get("group/dataset", getclass=True)
        classlink = self.h5.get("group/dataset", getlink=True, getclass=True)
        self.assertEqual(class_, h5py.Dataset)
        self.assertEqual(classlink, h5py.HardLink)

    def test_soft_link(self):
        node = self.h5["link/soft_link"]
        self.assertEqual(node.name, "/link/soft_link")
        class_ = self.h5.get("link/soft_link", getclass=True)
        link = self.h5.get("link/soft_link", getlink=True)
        classlink = self.h5.get("link/soft_link", getlink=True, getclass=True)
        self.assertEqual(class_, h5py.Dataset)
        self.assertTrue(isinstance(link, (h5py.SoftLink, commonh5.SoftLink)))
        self.assertTrue(silx.io.utils.is_softlink(link))
        self.assertEqual(classlink, h5py.SoftLink)
 
    def test_external_link(self):
        node = self.h5["link/external_link"]
        self.assertEqual(node.name, "/target/dataset")
        class_ = self.h5.get("link/external_link", getclass=True)
        classlink = self.h5.get("link/external_link", getlink=True, getclass=True)
        self.assertEqual(class_, h5py.Dataset)
        self.assertEqual(classlink, h5py.ExternalLink)

    def test_external_link_to_link(self):
        node = self.h5["link/external_link_to_link"]
        self.assertEqual(node.name, "/target/link")
        class_ = self.h5.get("link/external_link_to_link", getclass=True)
        classlink = self.h5.get("link/external_link_to_link", getlink=True, getclass=True)
        self.assertEqual(class_, h5py.Dataset)
        self.assertEqual(classlink, h5py.ExternalLink)

    def test_create_groups(self):
        c = self.h5.create_group(self.id() + "/a/b/c")
        d = c.create_group("/" + self.id() + "/a/b/d")

        self.assertRaises(ValueError, self.h5.create_group, self.id() + "/a/b/d")
        self.assertEqual(c.name, "/" + self.id() + "/a/b/c")
        self.assertEqual(d.name, "/" + self.id() + "/a/b/d")

    def test_setitem_python_object_dataset(self):
        group = self.h5.create_group(self.id())
        group["a"] = 10
        self.assertEqual(group["a"].dtype.kind, "i")

    def test_setitem_numpy_dataset(self):
        group = self.h5.create_group(self.id())
        group["a"] = numpy.array([10, 20, 30])
        self.assertEqual(group["a"].dtype.kind, "i")
        self.assertEqual(group["a"].shape, (3,))

    def test_setitem_link(self):
        group = self.h5.create_group(self.id())
        group["a"] = 10
        group["b"] = group["a"]
        self.assertEqual(group["b"].dtype.kind, "i")

    def test_setitem_dataset_is_sub_group(self):
        self.h5[self.id() + "/a"] = 10


class TestCommonFeatures_h5py(TestCommonFeatures):
    """Check if h5py is compliant with what we expect."""

    @classmethod
    def create_resource(cls):
        cls.tmp_dir = tempfile.mkdtemp()

        externalh5 = h5py.File(cls.tmp_dir + "/external.h5", mode="w")
        externalh5["target/dataset"] = 50
        externalh5["target/link"] = h5py.SoftLink("/target/dataset")
        externalh5.close()

        h5 = h5py.File(cls.tmp_dir + "/base.h5", mode="w")
        h5["group/dataset"] = 50
        h5["link/soft_link"] = h5py.SoftLink("/group/dataset")
        h5["link/external_link"] = h5py.ExternalLink("external.h5", "/target/dataset")
        h5["link/external_link_to_link"] = h5py.ExternalLink("external.h5", "/target/link")

        return h5

    @classmethod
    def tearDownClass(cls):
        super(TestCommonFeatures_h5py, cls).tearDownClass()
        if hasattr(cls, "tmp_dir") and cls.tmp_dir is not None:
            shutil.rmtree(cls.tmp_dir)


class TestCommonFeatures_commonH5(TestCommonFeatures):
    """Check if commonh5 is compliant with h5py."""

    @classmethod
    def create_resource(cls):
        h5 = commonh5.File("base.h5", "w")
        h5.create_group("group").create_dataset("dataset", data=numpy.int32(50))

        link = h5.create_group("link")
        link.add_node(commonh5.SoftLink("soft_link", "/group/dataset"))

        return h5

    def test_external_link(self):
        # not applicable
        pass

    def test_external_link_to_link(self):
        # not applicable
        pass


class TestSpecificCommonH5(unittest.TestCase):
    """Test specific features from commonh5.

    Test of shared features should be done by TestCommonFeatures."""

    def setUp(self):
        if commonh5 is None:
            self.skipTest("silx.io.commonh5 is needed")

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

    def test_setitem_dataset(self):
        self.h5 = commonh5.File(name="Foo", mode="w")
        group = self.h5.create_group(self.id())
        group["a"] = commonh5.Dataset(None, data=numpy.array(10))
        self.assertEqual(group["a"].dtype.kind, "i")

    def test_setitem_explicit_link(self):
        self.h5 = commonh5.File(name="Foo", mode="w")
        group = self.h5.create_group(self.id())
        group["a"] = 10
        group["b"] = commonh5.SoftLink(None, path="/" + self.id() + "/a")
        self.assertEqual(group["b"].dtype.kind, "i")


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestCommonFeatures_h5py))
    test_suite.addTest(loadTests(TestCommonFeatures_commonH5))
    test_suite.addTest(loadTests(TestSpecificCommonH5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
