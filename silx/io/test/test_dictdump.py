# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""Tests for dicttoh5 module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/04/2016"

import h5py
import os
import tempfile
import unittest

from ..dictdump import dicttoh5, dicttojson

from collections import defaultdict


def tree():
    """Tree data structure as a recursive nested dictionary"""
    return defaultdict(tree)


city_attrs = tree()
city_attrs["Europe"]["France"]["Grenoble"]["area"] = "18.44 km2"
city_attrs["Europe"]["France"]["Grenoble"]["inhabitants"] = 160215
city_attrs["Europe"]["France"]["Grenoble"]["coordinates"] = [45.1830, 5.7196]
city_attrs["Europe"]["France"]["Tourcoing"]["area"]


class TestDictToH5(unittest.TestCase):
    def setUp(self):
        fd, self.h5_fname = tempfile.mkstemp(text=False)
        # Close and delete (we just want the name)
        os.close(fd)
        os.unlink(self.h5_fname)

    def tearDown(self):
        os.unlink(self.h5_fname)

    def testH5CityAttrs(self):
        filters = {'compression': "gzip", 'shuffle': True,
                   'fletcher32': True}
        dicttoh5(city_attrs, self.h5_fname, h5path='/city attributes',
                 mode="w", create_dataset_args=filters)

        h5f = h5py.File(self.h5_fname)

        self.assertIn("Tourcoing/area", h5f["/city attributes/Europe/France"])
        ds = h5f["/city attributes/Europe/France/Grenoble/inhabitants"]
        self.assertEqual(ds[...], 160215)

        # filters only apply to datasets that are not scalars (shape != () )
        ds = h5f["/city attributes/Europe/France/Grenoble/coordinates"]
        self.assertEqual(ds.compression, "gzip")
        self.assertTrue(ds.fletcher32)
        self.assertTrue(ds.shuffle)

        h5f.close()

# We can't compare strings because the defaultdict is not ordered
# expected_json_content = """{
#    "Europe": {
#       "France": {
#          "Grenoble": {
#             "inhabitants": 160215, 
#             "coordinates": [
#                45.183, 
#                5.7196
#             ], 
#             "area": "18.44 km2"
#          }, 
#          "Tourcoing": {
#             "area": {}
#          }
#       }
#    }
# }"""


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
            self.assertIn('"inhabitants": 160215,', json_content)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestDictToH5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")