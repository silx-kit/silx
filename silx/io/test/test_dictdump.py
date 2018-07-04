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
"""Tests for dicttoh5 module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/01/2018"

from collections import OrderedDict
import numpy
import os
import tempfile
import unittest

try:
    import h5py
    h5py_missing = False
except ImportError:
    h5py_missing = True

from collections import defaultdict

from silx.utils.testutils import TestLogging

from ..configdict import ConfigDict
from ..dictdump import dicttoh5, dicttojson, dump
from ..dictdump import h5todict, load
from ..dictdump import logger as dictdump_logger


def tree():
    """Tree data structure as a recursive nested dictionary"""
    return defaultdict(tree)


city_attrs = tree()
city_attrs["Europe"]["France"]["Grenoble"]["area"] = "18.44 km2"
city_attrs["Europe"]["France"]["Grenoble"]["inhabitants"] = 160215
city_attrs["Europe"]["France"]["Grenoble"]["coordinates"] = [45.1830, 5.7196]
city_attrs["Europe"]["France"]["Tourcoing"]["area"]


@unittest.skipIf(h5py_missing, "Could not import h5py")
class TestDictToH5(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "cityattrs.h5")

    def tearDown(self):
        os.unlink(self.h5_fname)
        os.rmdir(self.tempdir)

    def testH5CityAttrs(self):
        filters = {'shuffle': True,
                   'fletcher32': True}
        dicttoh5(city_attrs, self.h5_fname, h5path='/city attributes',
                 mode="w", create_dataset_args=filters)

        h5f = h5py.File(self.h5_fname)

        self.assertIn("Tourcoing/area", h5f["/city attributes/Europe/France"])
        ds = h5f["/city attributes/Europe/France/Grenoble/inhabitants"]
        self.assertEqual(ds[...], 160215)

        # filters only apply to datasets that are not scalars (shape != () )
        ds = h5f["/city attributes/Europe/France/Grenoble/coordinates"]
        #self.assertEqual(ds.compression, "gzip")
        self.assertTrue(ds.fletcher32)
        self.assertTrue(ds.shuffle)

        h5f.close()

        ddict = load(self.h5_fname, fmat="hdf5")
        self.assertAlmostEqual(
                min(ddict["city attributes"]["Europe"]["France"]["Grenoble"]["coordinates"]),
                5.7196)

    def testH5Overwrite(self):
        dd = ConfigDict({'t': True})

        dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a')
        dd = ConfigDict({'t': False})
        with TestLogging(dictdump_logger, warning=1):
            dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a',
                     overwrite_data=False)

        res = h5todict(self.h5_fname)
        assert(res['t'] == True)

        dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a',
                 overwrite_data=True)

        res = h5todict(self.h5_fname)
        assert(res['t'] == False)


@unittest.skipIf(h5py_missing, "Could not import h5py")
class TestH5ToDict(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "cityattrs.h5")
        dicttoh5(city_attrs, self.h5_fname)

    def tearDown(self):
        os.unlink(self.h5_fname)
        os.rmdir(self.tempdir)

    def testExcludeNames(self):
        ddict = h5todict(self.h5_fname, path="/Europe/France",
                         exclude_names=["ourcoing", "inhab", "toto"])
        self.assertNotIn("Tourcoing", ddict)
        self.assertIn("Grenoble", ddict)

        self.assertNotIn("inhabitants", ddict["Grenoble"])
        self.assertIn("coordinates", ddict["Grenoble"])
        self.assertIn("area", ddict["Grenoble"])


class TestDictToJson(unittest.TestCase):
    def setUp(self):
        self.dir_path = tempfile.mkdtemp()
        self.json_fname = os.path.join(self.dir_path, "cityattrs.json")

    def tearDown(self):
        os.unlink(self.json_fname)
        os.rmdir(self.dir_path)

    def testJsonCityAttrs(self):
        self.json_fname = os.path.join(self.dir_path, "cityattrs.json")
        dicttojson(city_attrs, self.json_fname, indent=3)

        with open(self.json_fname, "r") as f:
            json_content = f.read()
            self.assertIn('"inhabitants": 160215', json_content)


class TestDictToIni(unittest.TestCase):
    def setUp(self):
        self.dir_path = tempfile.mkdtemp()
        self.ini_fname = os.path.join(self.dir_path, "test.ini")

    def tearDown(self):
        os.unlink(self.ini_fname)
        os.rmdir(self.dir_path)

    def testConfigDictIO(self):
        """Ensure values and types of data is preserved when dictionary is
        written to file and read back."""
        testdict = {
            'simple_types': {
                'float': 1.0,
                'int': 1,
                'percent string': '5 % is too much',
                'backslash string': 'i can use \\',
                'empty_string': '',
                'nonestring': 'None',
                'nonetype': None,
                'interpstring': 'interpolation: %(percent string)s',
            },
            'containers': {
                'list': [-1, 'string', 3.0, False, None],
                'array': numpy.array([1.0, 2.0, 3.0]),
                'dict': {
                    'key1': 'Hello World',
                    'key2': 2.0,
                }
            }
        }

        dump(testdict, self.ini_fname)

        #read the data back
        readdict = load(self.ini_fname)

        testdictkeys = list(testdict.keys())
        readkeys = list(readdict.keys())

        self.assertTrue(len(readkeys) == len(testdictkeys),
                        "Number of read keys not equal")

        self.assertEqual(readdict['simple_types']["interpstring"],
                         "interpolation: 5 % is too much")

        testdict['simple_types']["interpstring"] = "interpolation: 5 % is too much"

        for key in testdict["simple_types"]:
            original = testdict['simple_types'][key]
            read = readdict['simple_types'][key]
            self.assertEqual(read, original,
                             "Read <%s> instead of <%s>" % (read, original))

        for key in testdict["containers"]:
            original = testdict["containers"][key]
            read = readdict["containers"][key]
            if key == 'array':
                self.assertEqual(read.all(), original.all(),
                            "Read <%s> instead of <%s>" % (read, original))
            else:
                self.assertEqual(read, original,
                            "Read <%s> instead of <%s>" % (read, original))

    def testConfigDictOrder(self):
        """Ensure order is preserved when dictionary is
        written to file and read back."""
        test_dict = {'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2}
        # sort by key
        test_ordered_dict1 = OrderedDict(sorted(test_dict.items(),
                                                key=lambda t: t[0]))
        # sort by value
        test_ordered_dict2 = OrderedDict(sorted(test_dict.items(),
                                                key=lambda t: t[1]))
        # add the two ordered dict as sections of a third ordered dict
        test_ordered_dict3 = OrderedDict()
        test_ordered_dict3["section1"] = test_ordered_dict1
        test_ordered_dict3["section2"] = test_ordered_dict2

        # write to ini and read back as a ConfigDict (inherits OrderedDict)
        dump(test_ordered_dict3,
             self.ini_fname, fmat="ini")
        read_instance = ConfigDict()
        read_instance.read(self.ini_fname)

        # loop through original and read-back dictionaries,
        # test identical order for key/value pairs
        for orig_key, section in zip(test_ordered_dict3.keys(),
                                     read_instance.keys()):
            self.assertEqual(orig_key, section)
            for orig_key2, read_key in zip(test_ordered_dict3[section].keys(),
                                           read_instance[section].keys()):
                self.assertEqual(orig_key2, read_key)
                self.assertEqual(test_ordered_dict3[section][orig_key2],
                                 read_instance[section][read_key])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDictToIni))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDictToH5))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDictToJson))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestH5ToDict))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
