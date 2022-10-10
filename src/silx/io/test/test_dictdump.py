# /*##########################################################################
# Copyright (C) 2016-2022 European Synchrotron Radiation Facility
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


from collections import defaultdict, OrderedDict
from copy import deepcopy
from io import BytesIO
import os
import tempfile
import unittest

import h5py
import numpy
try:
    import pint
except ImportError:
    pint = None
import pytest

from silx.utils.testutils import LoggingValidator

from ..configdict import ConfigDict
from .. import dictdump
from ..dictdump import dicttoh5, dicttojson, dump
from ..dictdump import h5todict, load
from ..dictdump import logger as dictdump_logger
from ..utils import is_link
from ..utils import h5py_read_dataset


@pytest.fixture
def tmp_h5py_file():
    with BytesIO() as buffer:
        with h5py.File(buffer, mode="w") as h5file:
            yield h5file


def tree():
    """Tree data structure as a recursive nested dictionary"""
    return defaultdict(tree)


inhabitants = 160215

city_attrs = tree()
city_attrs["Europe"]["France"]["Grenoble"]["area"] = "18.44 km2"
city_attrs["Europe"]["France"]["Grenoble"]["inhabitants"] = inhabitants
city_attrs["Europe"]["France"]["Grenoble"]["coordinates"] = [45.1830, 5.7196]
city_attrs["Europe"]["France"]["Tourcoing"]["area"]

ext_attrs = tree()
ext_attrs["ext_group"]["dataset"] = 10
ext_filename = "ext.h5"

link_attrs = tree()
link_attrs["links"]["group"]["dataset"] = 10
link_attrs["links"]["group"]["relative_softlink"] = h5py.SoftLink("dataset")
link_attrs["links"]["relative_softlink"] = h5py.SoftLink("group/dataset")
link_attrs["links"]["absolute_softlink"] = h5py.SoftLink("/links/group/dataset")
link_attrs["links"]["external_link"] = h5py.ExternalLink(ext_filename, "/ext_group/dataset")


class DictTestCase(unittest.TestCase):

    def assertRecursiveEqual(self, expected, actual, nodes=tuple()):
        err_msg = "\n\n Tree nodes: {}".format(nodes)
        if isinstance(expected, dict):
            self.assertTrue(isinstance(actual, dict), msg=err_msg)
            self.assertEqual(
                set(expected.keys()),
                set(actual.keys()),
                msg=err_msg
            )
            for k in actual:
                self.assertRecursiveEqual(
                    expected[k],
                    actual[k],
                    nodes=nodes + (k,),
                )
            return
        if isinstance(actual, numpy.ndarray):
            actual = actual.tolist()
        if isinstance(expected, numpy.ndarray):
            expected = expected.tolist()

        self.assertEqual(expected, actual, msg=err_msg)


class H5DictTestCase(DictTestCase):

    def _dictRoundTripNormalize(self, treedict):
        """Convert the dictionary as expected from a round-trip
        treedict -> dicttoh5 -> h5todict -> newtreedict
        """
        for key, value in list(treedict.items()):
            if isinstance(value, dict):
                self._dictRoundTripNormalize(value)

        # Expand treedict[("group", "attr_name")]
        #     to treedict["group"]["attr_name"]
        for key, value in list(treedict.items()):
            if not isinstance(key, tuple):
                continue
            # Put the attribute inside the group
            grpname, attr = key
            if not grpname:
                continue
            group = treedict.setdefault(grpname, dict())
            if isinstance(group, dict):
                del treedict[key]
                group[("", attr)] = value

    def dictRoundTripNormalize(self, treedict):
        treedict2 = deepcopy(treedict)
        self._dictRoundTripNormalize(treedict2)
        return treedict2


class TestDictToH5(H5DictTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "cityattrs.h5")
        self.h5_ext_fname = os.path.join(self.tempdir, ext_filename)

    def tearDown(self):
        if os.path.exists(self.h5_fname):
            os.unlink(self.h5_fname)
        if os.path.exists(self.h5_ext_fname):
            os.unlink(self.h5_ext_fname)
        os.rmdir(self.tempdir)

    def testH5CityAttrs(self):
        filters = {'shuffle': True,
                   'fletcher32': True}
        dicttoh5(city_attrs, self.h5_fname, h5path='/city attributes',
                 mode="w", create_dataset_args=filters)

        h5f = h5py.File(self.h5_fname, mode='r')

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

    def testH5OverwriteDeprecatedApi(self):
        dd = ConfigDict({'t': True})

        dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a')
        dd = ConfigDict({'t': False})
        dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a',
                 overwrite_data=False)

        res = h5todict(self.h5_fname)
        assert(res['t'] == True)

        dicttoh5(h5file=self.h5_fname, treedict=dd, mode='a',
                 overwrite_data=True)

        res = h5todict(self.h5_fname)
        assert(res['t'] == False)

    def testAttributes(self):
        """Any kind of attribute can be described"""
        ddict = {
            "group": {"datatset": "hmmm", ("", "group_attr"): 10},
            "dataset": "aaaaaaaaaaaaaaa",
            ("", "root_attr"): 11,
            ("dataset", "dataset_attr"): 12,
            ("group", "group_attr2"): 13,
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict, h5file)
            self.assertEqual(h5file["group"].attrs['group_attr'], 10)
            self.assertEqual(h5file.attrs['root_attr'], 11)
            self.assertEqual(h5file["dataset"].attrs['dataset_attr'], 12)
            self.assertEqual(h5file["group"].attrs['group_attr2'], 13)

    def testPathAttributes(self):
        """A group is requested at a path"""
        ddict = {
            ("", "NX_class"): 'NXcollection',
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            # This should not warn
            with LoggingValidator(dictdump_logger, warning=0):
                dictdump.dicttoh5(ddict, h5file, h5path="foo/bar")

    def testKeyOrder(self):
        ddict1 = {
            "d": "plow",
            ("d", "a"): "ox",
        }
        ddict2 = {
            ("d", "a"): "ox",
            "d": "plow",
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict1, h5file, h5path="g1")
            dictdump.dicttoh5(ddict2, h5file, h5path="g2")
            self.assertEqual(h5file["g1/d"].attrs['a'], "ox")
            self.assertEqual(h5file["g2/d"].attrs['a'], "ox")

    def testAttributeValues(self):
        """Any NX data types can be used"""
        ddict = {
            ("", "bool"): True,
            ("", "int"): 11,
            ("", "float"): 1.1,
            ("", "str"): "a",
            ("", "boollist"): [True, False, True],
            ("", "intlist"): [11, 22, 33],
            ("", "floatlist"): [1.1, 2.2, 3.3],
            ("", "strlist"): ["a", "bb", "ccc"],
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict, h5file)
            for k, expected in ddict.items():
                result = h5file.attrs[k[1]]
                if isinstance(expected, list):
                    if isinstance(expected[0], str):
                        numpy.testing.assert_array_equal(result, expected)
                    else:
                        numpy.testing.assert_array_almost_equal(result, expected)
                else:
                    self.assertEqual(result, expected)

    def testAttributeAlreadyExists(self):
        """A duplicated attribute warns if overwriting is not enabled"""
        ddict = {
            "group": {"dataset": "hmmm", ("", "attr"): 10},
            ("group", "attr"): 10,
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict, h5file)
            self.assertEqual(h5file["group"].attrs['attr'], 10)

    def testFlatDict(self):
        """Description of a tree with a single level of keys"""
        ddict = {
            "group/group/dataset": 10,
            ("group/group/dataset", "attr"): 11,
            ("group/group", "attr"): 12,
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict, h5file)
            self.assertEqual(h5file["group/group/dataset"][()], 10)
            self.assertEqual(h5file["group/group/dataset"].attrs['attr'], 11)
            self.assertEqual(h5file["group/group"].attrs['attr'], 12)

    def testLinks(self):
        with h5py.File(self.h5_ext_fname, "w") as h5file:
            dictdump.dicttoh5(ext_attrs, h5file)
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(link_attrs, h5file)
        with h5py.File(self.h5_fname, "r") as h5file:
            self.assertEqual(h5file["links/group/dataset"][()], 10)
            self.assertEqual(h5file["links/group/relative_softlink"][()], 10)
            self.assertEqual(h5file["links/relative_softlink"][()], 10)
            self.assertEqual(h5file["links/absolute_softlink"][()], 10)
            self.assertEqual(h5file["links/external_link"][()], 10)

    def testDumpNumpyArray(self):
        ddict = {
            'darks': {
                '0': numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.uint16)
            }
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttoh5(ddict, h5file)
        with h5py.File(self.h5_fname, "r") as h5file:
            numpy.testing.assert_array_equal(h5py_read_dataset(h5file["darks"]["0"]),
                                             ddict['darks']['0'])

    def testOverwrite(self):
        # Tree structure that will be tested
        group1 = {
            ("", "attr2"): "original2",
            "dset1": 0,
            "dset2": [0, 1],
            ("dset1", "attr1"): "original1",
            ("dset1", "attr2"): "original2",
            ("dset2", "attr1"): "original1",
            ("dset2", "attr2"): "original2",
        }
        group2 = {
            "subgroup1": group1.copy(),
            "subgroup2": group1.copy(),
            ("subgroup1", "attr1"): "original1",
            ("subgroup2", "attr1"): "original1"
        }
        group2.update(group1)
        # initial HDF5 tree
        otreedict = {
            ('', 'attr1'): "original1",
            ('', 'attr2'): "original2",
            'group1': group1,
            'group2': group2,
            ('group1', 'attr1'): "original1",
            ('group2', 'attr1'): "original1"
        }
        wtreedict = None  # dumped dictionary
        etreedict = None  # expected HDF5 tree after dump

        def reset_file():
            dicttoh5(
                otreedict,
                h5file=self.h5_fname,
                mode="w",
            )

        def append_file(update_mode):
            dicttoh5(
                wtreedict,
                h5file=self.h5_fname,
                mode="a",
                update_mode=update_mode
            )

        def assert_file():
            rtreedict = h5todict(
                self.h5_fname,
                include_attributes=True,
                asarray=False
            )
            netreedict = self.dictRoundTripNormalize(etreedict)
            try:
                self.assertRecursiveEqual(netreedict, rtreedict)
            except AssertionError:
                from pprint import pprint
                print("\nDUMP:")
                pprint(wtreedict)
                print("\nEXPECTED:")
                pprint(netreedict)
                print("\nHDF5:")
                pprint(rtreedict)
                raise

        def assert_append(update_mode):
            append_file(update_mode)
            assert_file()

        # Test wrong arguments
        with self.assertRaises(ValueError):
            dicttoh5(
                otreedict,
                h5file=self.h5_fname,
                mode="w",
                update_mode="wrong-value"
            )

        # No writing
        reset_file()
        etreedict = deepcopy(otreedict)
        assert_file()

        # Write identical dictionary
        wtreedict = deepcopy(otreedict)

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add", "modify", "replace"]:
            assert_append(update_mode)

        # Write empty dictionary
        wtreedict = dict()

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add", "modify", "replace"]:
            assert_append(update_mode)

        # Modified dataset
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = dict()
        wtreedict["group2"]["subgroup2"]["dset1"] = {"dset3": [10, 20]}
        wtreedict["group2"]["subgroup2"]["dset2"] = [10, 20]

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add"]:
            assert_append(update_mode)

        etreedict["group2"]["subgroup2"]["dset2"] = [10, 20]
        assert_append("modify")

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        etreedict["group2"]["subgroup2"] = dict()
        etreedict["group2"]["subgroup2"]["dset1"] = {"dset3": [10, 20]}
        etreedict["group2"]["subgroup2"]["dset2"] = [10, 20]
        assert_append("replace")

        # Modified group
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = [0, 1]

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add", "modify"]:
            assert_append(update_mode)

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        etreedict["group2"]["subgroup2"] = [0, 1]
        assert_append("replace")

        # Modified attribute
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = dict()
        wtreedict["group2"]["subgroup2"][("dset1", "attr1")] = "modified"

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add"]:
            assert_append(update_mode)

        etreedict["group2"]["subgroup2"][("dset1", "attr1")] = "modified"
        assert_append("modify")

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        etreedict["group2"]["subgroup2"] = dict()
        etreedict["group2"]["subgroup2"]["dset1"] = dict()
        etreedict["group2"]["subgroup2"]["dset1"][("", "attr1")] = "modified"
        assert_append("replace")

        # Delete group
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = None

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add"]:
            assert_append(update_mode)

        del etreedict["group2"]["subgroup2"]
        del etreedict["group2"][("subgroup2", "attr1")]
        assert_append("modify")

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        assert_append("replace")

        # Delete dataset
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = dict()
        wtreedict["group2"]["subgroup2"]["dset2"] = None

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add"]:
            assert_append(update_mode)

        del etreedict["group2"]["subgroup2"]["dset2"]
        del etreedict["group2"]["subgroup2"][("dset2", "attr1")]
        del etreedict["group2"]["subgroup2"][("dset2", "attr2")]
        assert_append("modify")

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        etreedict["group2"]["subgroup2"] = dict()
        assert_append("replace")

        # Delete attribute
        wtreedict = dict()
        wtreedict["group2"] = dict()
        wtreedict["group2"]["subgroup2"] = dict()
        wtreedict["group2"]["subgroup2"][("dset2", "attr1")] = None

        reset_file()
        etreedict = deepcopy(otreedict)
        for update_mode in [None, "add"]:
            assert_append(update_mode)

        del etreedict["group2"]["subgroup2"][("dset2", "attr1")]
        assert_append("modify")

        etreedict["group2"] = dict()
        del etreedict[("group2", "attr1")]
        etreedict["group2"]["subgroup2"] = dict()
        etreedict["group2"]["subgroup2"]["dset2"] = dict()
        assert_append("replace")


@pytest.mark.skipif(pint is None, reason="Require pint")
def test_dicttoh5_pint(tmp_h5py_file):
    ureg = pint.UnitRegistry()
    treedict = {
        "array_mm": pint.Quantity([1, 2, 3], ureg.mm),
        "value_kg": 3 * ureg.kg,
    }

    dicttoh5(treedict, tmp_h5py_file)

    result = h5todict(tmp_h5py_file)
    assert set(treedict.keys()) == set(result.keys())
    for key, value in treedict.items():
        assert numpy.array_equal(result[key], value.magnitude)


class TestH5ToDict(H5DictTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "cityattrs.h5")
        self.h5_ext_fname = os.path.join(self.tempdir, ext_filename)
        dicttoh5(city_attrs, self.h5_fname)
        dicttoh5(link_attrs, self.h5_fname, mode="a")
        dicttoh5(ext_attrs, self.h5_ext_fname)

    def tearDown(self):
        if os.path.exists(self.h5_fname):
            os.unlink(self.h5_fname)
        if os.path.exists(self.h5_ext_fname):
            os.unlink(self.h5_ext_fname)
        os.rmdir(self.tempdir)

    def testExcludeNames(self):
        ddict = h5todict(self.h5_fname, path="/Europe/France",
                         exclude_names=["ourcoing", "inhab", "toto"])
        self.assertNotIn("Tourcoing", ddict)
        self.assertIn("Grenoble", ddict)

        self.assertNotIn("inhabitants", ddict["Grenoble"])
        self.assertIn("coordinates", ddict["Grenoble"])
        self.assertIn("area", ddict["Grenoble"])

    def testAsArrayTrue(self):
        """Test with asarray=True, the default"""
        ddict = h5todict(self.h5_fname, path="/Europe/France/Grenoble")
        self.assertTrue(numpy.array_equal(ddict["inhabitants"], numpy.array(inhabitants)))

    def testAsArrayFalse(self):
        """Test with asarray=False"""
        ddict = h5todict(self.h5_fname, path="/Europe/France/Grenoble", asarray=False)
        self.assertEqual(ddict["inhabitants"], inhabitants)

    def testDereferenceLinks(self):
        ddict = h5todict(self.h5_fname, path="links", dereference_links=True)
        self.assertTrue(ddict["absolute_softlink"], 10)
        self.assertTrue(ddict["relative_softlink"], 10)
        self.assertTrue(ddict["external_link"], 10)
        self.assertTrue(ddict["group"]["relative_softlink"], 10)

    def testPreserveLinks(self):
        ddict = h5todict(self.h5_fname, path="links", dereference_links=False)
        self.assertTrue(is_link(ddict["absolute_softlink"]))
        self.assertTrue(is_link(ddict["relative_softlink"]))
        self.assertTrue(is_link(ddict["external_link"]))
        self.assertTrue(is_link(ddict["group"]["relative_softlink"]))

    def testStrings(self):
        ddict = {"dset_bytes": b"bytes",
                 "dset_utf8": "utf8",
                 "dset_2bytes": [b"bytes", b"bytes"],
                 "dset_2utf8": ["utf8", "utf8"],
                 ("", "attr_bytes"): b"bytes",
                 ("", "attr_utf8"): "utf8",
                 ("", "attr_2bytes"): [b"bytes", b"bytes"],
                 ("", "attr_2utf8"): ["utf8", "utf8"]}
        dicttoh5(ddict, self.h5_fname, mode="w")
        adict = h5todict(self.h5_fname, include_attributes=True, asarray=False)
        self.assertEqual(ddict["dset_bytes"], adict["dset_bytes"])
        self.assertEqual(ddict["dset_utf8"], adict["dset_utf8"])
        self.assertEqual(ddict[("", "attr_bytes")], adict[("", "attr_bytes")])
        self.assertEqual(ddict[("", "attr_utf8")], adict[("", "attr_utf8")])
        numpy.testing.assert_array_equal(ddict["dset_2bytes"], adict["dset_2bytes"])
        numpy.testing.assert_array_equal(ddict["dset_2utf8"], adict["dset_2utf8"])
        numpy.testing.assert_array_equal(ddict[("", "attr_2bytes")], adict[("", "attr_2bytes")])
        numpy.testing.assert_array_equal(ddict[("", "attr_2utf8")], adict[("", "attr_2utf8")])


class TestDictToNx(H5DictTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "nx.h5")
        self.h5_ext_fname = os.path.join(self.tempdir, "nx_ext.h5")

    def tearDown(self):
        if os.path.exists(self.h5_fname):
            os.unlink(self.h5_fname)
        if os.path.exists(self.h5_ext_fname):
            os.unlink(self.h5_ext_fname)
        os.rmdir(self.tempdir)

    def testAttributes(self):
        """Any kind of attribute can be described"""
        ddict = {
            "group": {"dataset": 100, "@group_attr1": 10},
            "dataset": 200,
            "@root_attr": 11,
            "dataset@dataset_attr": "12",
            "group@group_attr2": 13,
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttonx(ddict, h5file)
            self.assertEqual(h5file["group"].attrs['group_attr1'], 10)
            self.assertEqual(h5file.attrs['root_attr'], 11)
            self.assertEqual(h5file["dataset"].attrs['dataset_attr'], "12")
            self.assertEqual(h5file["group"].attrs['group_attr2'], 13)

    def testKeyOrder(self):
        ddict1 = {
            "d": "plow",
            "d@a": "ox",
        }
        ddict2 = {
            "d@a": "ox",
            "d": "plow",
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttonx(ddict1, h5file, h5path="g1")
            dictdump.dicttonx(ddict2, h5file, h5path="g2")
            self.assertEqual(h5file["g1/d"].attrs['a'], "ox")
            self.assertEqual(h5file["g2/d"].attrs['a'], "ox")

    def testAttributeValues(self):
        """Any NX data types can be used"""
        ddict = {
            "@bool": True,
            "@int": 11,
            "@float": 1.1,
            "@str": "a",
            "@boollist": [True, False, True],
            "@intlist": [11, 22, 33],
            "@floatlist": [1.1, 2.2, 3.3],
            "@strlist": ["a", "bb", "ccc"],
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttonx(ddict, h5file)
            for k, expected in ddict.items():
                result = h5file.attrs[k[1:]]
                if isinstance(expected, list):
                    if isinstance(expected[0], str):
                        numpy.testing.assert_array_equal(result, expected)
                    else:
                        numpy.testing.assert_array_almost_equal(result, expected)
                else:
                    self.assertEqual(result, expected)

    def testFlatDict(self):
        """Description of a tree with a single level of keys"""
        ddict = {
            "group/group/dataset": 10,
            "group/group/dataset@attr": 11,
            "group/group@attr": 12,
        }
        with h5py.File(self.h5_fname, "w") as h5file:
            dictdump.dicttonx(ddict, h5file)
            self.assertEqual(h5file["group/group/dataset"][()], 10)
            self.assertEqual(h5file["group/group/dataset"].attrs['attr'], 11)
            self.assertEqual(h5file["group/group"].attrs['attr'], 12)

    def testLinks(self):
        ddict = {"ext_group": {"dataset": 10}}
        dictdump.dicttonx(ddict, self.h5_ext_fname)
        ddict = {"links": {"group": {"dataset": 10, ">relative_softlink": "dataset"},
                           ">relative_softlink": "group/dataset",
                           ">absolute_softlink": "/links/group/dataset",
                           ">external_link": "nx_ext.h5::/ext_group/dataset"}}
        dictdump.dicttonx(ddict, self.h5_fname)
        with h5py.File(self.h5_fname, "r") as h5file:
            self.assertEqual(h5file["links/group/dataset"][()], 10)
            self.assertEqual(h5file["links/group/relative_softlink"][()], 10)
            self.assertEqual(h5file["links/relative_softlink"][()], 10)
            self.assertEqual(h5file["links/absolute_softlink"][()], 10)
            self.assertEqual(h5file["links/external_link"][()], 10)

    def testUpLinks(self):
        ddict = {"data": {"group": {"dataset": 10, ">relative_softlink": "dataset"}},
                 "links": {"group": {"subgroup": {">relative_softlink": "../../../data/group/dataset"}}}}
        dictdump.dicttonx(ddict, self.h5_fname)
        with h5py.File(self.h5_fname, "r") as h5file:
            self.assertEqual(h5file["/links/group/subgroup/relative_softlink"][()], 10)

    def testOverwrite(self):
        entry_name = "entry"
        wtreedict = {
            "group1": {"a": 1, "b": 2},
            "group2@attr3": "attr3",
            "group2@attr4": "attr4",
            "group2": {
                "@attr1": "attr1",
                "@attr2": "attr2",
                "c": 3,
                "d": 4,
                "dataset4": 8,
                "dataset4@units": "keV",
            },
            "group3": {"subgroup": {"e": 9, "f": 10}},
            "dataset1": 5,
            "dataset2": 6,
            "dataset3": 7,
            "dataset3@units": "mm",
        }
        esubtree = {
            "@NX_class": "NXentry",
            "group1": {"@NX_class": "NXcollection", "a": 1, "b": 2},
            "group2": {
                "@NX_class": "NXcollection",
                "@attr1": "attr1",
                "@attr2": "attr2",
                "@attr3": "attr3",
                "@attr4": "attr4",
                "c": 3,
                "d": 4,
                "dataset4": 8,
                "dataset4@units": "keV",
            },
            "group3": {
                "@NX_class": "NXcollection",
                "subgroup": {"@NX_class": "NXcollection", "e": 9, "f": 10},
            },
            "dataset1": 5,
            "dataset2": 6,
            "dataset3": 7,
            "dataset3@units": "mm",
        }
        etreedict = {entry_name: esubtree}

        def append_file(update_mode, add_nx_class):
            dictdump.dicttonx(
                wtreedict,
                h5file=self.h5_fname,
                mode="a",
                h5path=entry_name,
                update_mode=update_mode,
                add_nx_class=add_nx_class
            )

        def assert_file():
            rtreedict = dictdump.nxtodict(
                self.h5_fname,
                include_attributes=True,
                asarray=False,
            )
            netreedict = self.dictRoundTripNormalize(etreedict)
            try:
                self.assertRecursiveEqual(netreedict, rtreedict)
            except AssertionError:
                from pprint import pprint
                print("\nDUMP:")
                pprint(wtreedict)
                print("\nEXPECTED:")
                pprint(netreedict)
                print("\nHDF5:")
                pprint(rtreedict)
                raise

        def assert_append(update_mode, add_nx_class=None):
            append_file(update_mode, add_nx_class=add_nx_class)
            assert_file()

        # First to an empty file
        assert_append(None)

        # Add non-existing attributes/datasets/groups
        wtreedict["group1"].pop("a")
        wtreedict["group2"].pop("@attr1")
        wtreedict["group2"]["@attr2"] = "attr3"  # only for update
        wtreedict["group2"]["@type"] = "test"
        wtreedict["group2"]["dataset4"] = 9  # only for update
        del wtreedict["group2"]["dataset4@units"]
        wtreedict["group3"] = {}
        esubtree["group2"]["@type"] = "test"
        assert_append("add")

        # Add update existing attributes and datasets
        esubtree["group2"]["@attr2"] = "attr3"
        esubtree["group2"]["dataset4"] = 9
        assert_append("modify")

        # Do not add missing NX_class by default when updating
        wtreedict["group2"]["@NX_class"] = "NXprocess"
        esubtree["group2"]["@NX_class"] = "NXprocess"
        assert_append("modify")
        del wtreedict["group2"]["@NX_class"]
        assert_append("modify")

        # Overwrite existing groups/datasets/attributes
        esubtree["group1"].pop("a")
        esubtree["group2"].pop("@attr1")
        esubtree["group2"]["@NX_class"] = "NXcollection"
        esubtree["group2"]["dataset4"] = 9
        del esubtree["group2"]["dataset4@units"]
        esubtree["group3"] = {"@NX_class": "NXcollection"}
        assert_append("replace", add_nx_class=True)


@pytest.mark.skipif(pint is None, reason="Require pint")
def test_dicttonx_pint(tmp_h5py_file):
    ureg = pint.UnitRegistry()
    treedict = {
        "array_mm": pint.Quantity([1, 2, 3], ureg.mm),
        "value_kg": 3 * ureg.kg,
    }

    dictdump.dicttonx(treedict, tmp_h5py_file)

    result = dictdump.nxtodict(tmp_h5py_file)
    for key, value in treedict.items():
        assert numpy.array_equal(result[key], value.magnitude)
        assert result[f"{key}@units"] == f"{value.units:~C}"


class TestNxToDict(H5DictTestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "nx.h5")
        self.h5_ext_fname = os.path.join(self.tempdir, "nx_ext.h5")

    def tearDown(self):
        if os.path.exists(self.h5_fname):
            os.unlink(self.h5_fname)
        if os.path.exists(self.h5_ext_fname):
            os.unlink(self.h5_ext_fname)
        os.rmdir(self.tempdir)

    def testAttributes(self):
        """Any kind of attribute can be described"""
        ddict = {
            "group": {"dataset": 100, "@group_attr1": 10},
            "dataset": 200,
            "@root_attr": 11,
            "dataset@dataset_attr": "12",
            "group@group_attr2": 13,
        }
        dictdump.dicttonx(ddict, self.h5_fname)
        ddict = dictdump.nxtodict(self.h5_fname, include_attributes=True)
        self.assertEqual(ddict["group"]["@group_attr1"], 10)
        self.assertEqual(ddict["@root_attr"], 11)
        self.assertEqual(ddict["dataset@dataset_attr"], "12")
        self.assertEqual(ddict["group"]["@group_attr2"], 13)

    def testDereferenceLinks(self):
        """Write links and dereference on read"""
        ddict = {"ext_group": {"dataset": 10}}
        dictdump.dicttonx(ddict, self.h5_ext_fname)
        ddict = {"links": {"group": {"dataset": 10, ">relative_softlink": "dataset"},
                           ">relative_softlink": "group/dataset",
                           ">absolute_softlink": "/links/group/dataset",
                           ">external_link": "nx_ext.h5::/ext_group/dataset"}}
        dictdump.dicttonx(ddict, self.h5_fname)

        ddict = dictdump.h5todict(self.h5_fname, dereference_links=True)
        self.assertTrue(ddict["links"]["absolute_softlink"], 10)
        self.assertTrue(ddict["links"]["relative_softlink"], 10)
        self.assertTrue(ddict["links"]["external_link"], 10)
        self.assertTrue(ddict["links"]["group"]["relative_softlink"], 10)

    def testPreserveLinks(self):
        """Write/read links"""
        ddict = {"ext_group": {"dataset": 10}}
        dictdump.dicttonx(ddict, self.h5_ext_fname)
        ddict = {"links": {"group": {"dataset": 10, ">relative_softlink": "dataset"},
                           ">relative_softlink": "group/dataset",
                           ">absolute_softlink": "/links/group/dataset",
                           ">external_link": "nx_ext.h5::/ext_group/dataset"}}
        dictdump.dicttonx(ddict, self.h5_fname)

        ddict = dictdump.nxtodict(self.h5_fname, dereference_links=False)
        self.assertTrue(ddict["links"][">absolute_softlink"], "dataset")
        self.assertTrue(ddict["links"][">relative_softlink"], "group/dataset")
        self.assertTrue(ddict["links"][">external_link"], "/links/group/dataset")
        self.assertTrue(ddict["links"]["group"][">relative_softlink"], "nx_ext.h5::/ext_group/datase")

    def testNotExistingPath(self):
        """Test converting not existing path"""
        with h5py.File(self.h5_fname, 'a') as f:
            f['data'] = 1

        ddict = h5todict(self.h5_fname, path="/I/am/not/a/path", errors='ignore')
        self.assertFalse(ddict)

        with LoggingValidator(dictdump_logger, error=1):
            ddict = h5todict(self.h5_fname, path="/I/am/not/a/path", errors='log')
            self.assertFalse(ddict)

        with self.assertRaises(KeyError):
            h5todict(self.h5_fname, path="/I/am/not/a/path", errors='raise')

    def testBrokenLinks(self):
        """Test with broken links"""
        with h5py.File(self.h5_fname, 'a') as f:
            f["/Mars/BrokenSoftLink"] = h5py.SoftLink("/Idontexists")
            f["/Mars/BrokenExternalLink"] = h5py.ExternalLink("notexistingfile.h5", "/Idontexists")

        ddict = h5todict(self.h5_fname, path="/Mars", errors='ignore')
        self.assertFalse(ddict)

        with LoggingValidator(dictdump_logger, error=2):
            ddict = h5todict(self.h5_fname, path="/Mars", errors='log')
            self.assertFalse(ddict)

        with self.assertRaises(KeyError):
            h5todict(self.h5_fname, path="/Mars", errors='raise')


class TestDictToJson(DictTestCase):
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


class TestDictToIni(DictTestCase):
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
