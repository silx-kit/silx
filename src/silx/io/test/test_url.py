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
"""Tests for url module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/01/2018"


import unittest
from ..url import DataUrl


class TestDataUrl(unittest.TestCase):

    def assertUrl(self, url, expected):
        self.assertEqual(url.is_valid(), expected[0])
        self.assertEqual(url.is_absolute(), expected[1])
        self.assertEqual(url.scheme(), expected[2])
        self.assertEqual(url.file_path(), expected[3])
        self.assertEqual(url.data_path(), expected[4])
        self.assertEqual(url.data_slice(), expected[5])

    def test_fabio_absolute(self):
        url = DataUrl("fabio:///data/image.edf?slice=2")
        expected = [True, True, "fabio", "/data/image.edf", None, (2, )]
        self.assertUrl(url, expected)

    def test_fabio_absolute_windows(self):
        url = DataUrl("fabio:///C:/data/image.edf?slice=2")
        expected = [True, True, "fabio", "C:/data/image.edf", None, (2, )]
        self.assertUrl(url, expected)

    def test_silx_absolute(self):
        url = DataUrl("silx:///data/image.h5?path=/data/dataset&slice=1,5")
        expected = [True, True, "silx", "/data/image.h5", "/data/dataset", (1, 5)]
        self.assertUrl(url, expected)

    def test_commandline_shell_separator(self):
        url = DataUrl("silx:///data/image.h5::path=/data/dataset&slice=1,5")
        expected = [True, True, "silx", "/data/image.h5", "/data/dataset", (1, 5)]
        self.assertUrl(url, expected)

    def test_silx_absolute2(self):
        url = DataUrl("silx:///data/image.edf?/scan_0/detector/data")
        expected = [True, True, "silx", "/data/image.edf", "/scan_0/detector/data", None]
        self.assertUrl(url, expected)

    def test_silx_absolute_windows(self):
        url = DataUrl("silx:///C:/data/image.h5?/scan_0/detector/data")
        expected = [True, True, "silx", "C:/data/image.h5", "/scan_0/detector/data", None]
        self.assertUrl(url, expected)

    def test_silx_relative(self):
        url = DataUrl("silx:./image.h5")
        expected = [True, False, "silx", "./image.h5", None, None]
        self.assertUrl(url, expected)

    def test_fabio_relative(self):
        url = DataUrl("fabio:./image.edf")
        expected = [True, False, "fabio", "./image.edf", None, None]
        self.assertUrl(url, expected)

    def test_silx_relative2(self):
        url = DataUrl("silx:image.h5")
        expected = [True, False, "silx", "image.h5", None, None]
        self.assertUrl(url, expected)

    def test_fabio_relative2(self):
        url = DataUrl("fabio:image.edf")
        expected = [True, False, "fabio", "image.edf", None, None]
        self.assertUrl(url, expected)

    def test_file_relative(self):
        url = DataUrl("image.edf")
        expected = [True, False, None, "image.edf", None, None]
        self.assertUrl(url, expected)

    def test_file_relative2(self):
        url = DataUrl("./foo/bar/image.edf")
        expected = [True, False, None, "./foo/bar/image.edf", None, None]
        self.assertUrl(url, expected)

    def test_file_relative3(self):
        url = DataUrl("foo/bar/image.edf")
        expected = [True, False, None, "foo/bar/image.edf", None, None]
        self.assertUrl(url, expected)

    def test_file_absolute(self):
        url = DataUrl("/data/image.edf")
        expected = [True, True, None, "/data/image.edf", None, None]
        self.assertUrl(url, expected)

    def test_file_absolute_windows(self):
        url = DataUrl("C:/data/image.edf")
        expected = [True, True, None, "C:/data/image.edf", None, None]
        self.assertUrl(url, expected)

    def test_absolute_with_path(self):
        url = DataUrl("/foo/foobar.h5?/foo/bar")
        expected = [True, True, None, "/foo/foobar.h5", "/foo/bar", None]
        self.assertUrl(url, expected)

    def test_windows_file_data_slice(self):
        url = DataUrl("C:/foo/foobar.h5?path=/foo/bar&slice=5,1")
        expected = [True, True, None, "C:/foo/foobar.h5", "/foo/bar", (5, 1)]
        self.assertUrl(url, expected)

    def test_scheme_file_data_slice(self):
        url = DataUrl("silx:/foo/foobar.h5?path=/foo/bar&slice=5,1")
        expected = [True, True, "silx", "/foo/foobar.h5", "/foo/bar", (5, 1)]
        self.assertUrl(url, expected)

    def test_scheme_windows_file_data_slice(self):
        url = DataUrl("silx:C:/foo/foobar.h5?path=/foo/bar&slice=5,1")
        expected = [True, True, "silx", "C:/foo/foobar.h5", "/foo/bar", (5, 1)]
        self.assertUrl(url, expected)

    def test_empty(self):
        url = DataUrl("")
        expected = [False, False, None, "", None, None]
        self.assertUrl(url, expected)

    def test_unknown_scheme(self):
        url = DataUrl("foo:/foo/foobar.h5?path=/foo/bar&slice=5,1")
        expected = [False, True, "foo", "/foo/foobar.h5", "/foo/bar", (5, 1)]
        self.assertUrl(url, expected)

    def test_slice(self):
        url = DataUrl("/a.h5?path=/b&slice=5,1")
        expected = [True, True, None, "/a.h5", "/b", (5, 1)]
        self.assertUrl(url, expected)

    def test_slice2(self):
        url = DataUrl("/a.h5?path=/b&slice=2:5")
        expected = [True, True, None, "/a.h5", "/b", (slice(2, 5),)]
        self.assertUrl(url, expected)

    def test_slice3(self):
        url = DataUrl("/a.h5?path=/b&slice=::2")
        expected = [True, True, None, "/a.h5", "/b", (slice(None, None, 2),)]
        self.assertUrl(url, expected)

    def test_slice_ellipsis(self):
        url = DataUrl("/a.h5?path=/b&slice=...")
        expected = [True, True, None, "/a.h5", "/b", (Ellipsis, )]
        self.assertUrl(url, expected)

    def test_slice_slicing(self):
        url = DataUrl("/a.h5?path=/b&slice=:")
        expected = [True, True, None, "/a.h5", "/b", (slice(None), )]
        self.assertUrl(url, expected)

    def test_slice_missing_element(self):
        url = DataUrl("/a.h5?path=/b&slice=5,,1")
        expected = [False, True, None, "/a.h5", "/b", None]
        self.assertUrl(url, expected)

    def test_slice_no_elements(self):
        url = DataUrl("/a.h5?path=/b&slice=")
        expected = [False, True, None, "/a.h5", "/b", None]
        self.assertUrl(url, expected)

    def test_create_relative_url(self):
        url = DataUrl(scheme="silx", file_path="./foo.h5", data_path="/", data_slice=(5, 1))
        self.assertFalse(url.is_absolute())
        url2 = DataUrl(url.path())
        self.assertEqual(url, url2)

    def test_create_absolute_url(self):
        url = DataUrl(scheme="silx", file_path="/foo.h5", data_path="/", data_slice=(5, 1))
        url2 = DataUrl(url.path())
        self.assertEqual(url, url2)

    def test_create_absolute_windows_url(self):
        url = DataUrl(scheme="silx", file_path="C:/foo.h5", data_path="/", data_slice=(5, 1))
        url2 = DataUrl(url.path())
        self.assertEqual(url, url2)

    def test_create_slice_url(self):
        url = DataUrl(scheme="silx", file_path="/foo.h5", data_path="/", data_slice=(5, 1, Ellipsis, slice(None)))
        url2 = DataUrl(url.path())
        self.assertEqual(url, url2)

    def test_wrong_url(self):
        url = DataUrl(scheme="silx", file_path="/foo.h5", data_slice=(5, 1))
        self.assertFalse(url.is_valid())

    def test_path_creation(self):
        """make sure the construction of path succeed and that we can
        recreate a DataUrl from a path"""
        for data_slice in (1, (1,)):
            with self.subTest(data_slice=data_slice):
                url = DataUrl(scheme="silx", file_path="/foo.h5", data_slice=data_slice)
                path = url.path()
                DataUrl(path=path)
