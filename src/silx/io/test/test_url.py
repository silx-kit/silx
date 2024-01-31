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


import pytest
from ..url import DataUrl


def assert_url(url, expected):
    assert url.is_valid() == expected[0]
    assert url.is_absolute() == expected[1]
    assert url.scheme() == expected[2]
    assert url.file_path() == expected[3]
    assert url.data_path() == expected[4]
    assert url.data_slice() == expected[5]


def test_fabio_absolute():
    url = DataUrl("fabio:///data/image.edf?slice=2")
    expected = [True, True, "fabio", "/data/image.edf", None, (2,)]
    assert_url(url, expected)


def test_fabio_absolute_windows():
    url = DataUrl("fabio:///C:/data/image.edf?slice=2")
    expected = [True, True, "fabio", "C:/data/image.edf", None, (2,)]
    assert_url(url, expected)


def test_silx_absolute():
    url = DataUrl("silx:///data/image.h5?path=/data/dataset&slice=1,5")
    expected = [True, True, "silx", "/data/image.h5", "/data/dataset", (1, 5)]
    assert_url(url, expected)


def test_commandline_shell_separator():
    url = DataUrl("silx:///data/image.h5::path=/data/dataset&slice=1,5")
    expected = [True, True, "silx", "/data/image.h5", "/data/dataset", (1, 5)]
    assert_url(url, expected)


def test_silx_absolute2():
    url = DataUrl("silx:///data/image.edf?/scan_0/detector/data")
    expected = [True, True, "silx", "/data/image.edf", "/scan_0/detector/data", None]
    assert_url(url, expected)


def test_silx_absolute_windows():
    url = DataUrl("silx:///C:/data/image.h5?/scan_0/detector/data")
    expected = [True, True, "silx", "C:/data/image.h5", "/scan_0/detector/data", None]
    assert_url(url, expected)


def test_silx_relative():
    url = DataUrl("silx:./image.h5")
    expected = [True, False, "silx", "./image.h5", None, None]
    assert_url(url, expected)


def test_fabio_relative():
    url = DataUrl("fabio:./image.edf")
    expected = [True, False, "fabio", "./image.edf", None, None]
    assert_url(url, expected)


def test_silx_relative2():
    url = DataUrl("silx:image.h5")
    expected = [True, False, "silx", "image.h5", None, None]
    assert_url(url, expected)


def test_fabio_relative2():
    url = DataUrl("fabio:image.edf")
    expected = [True, False, "fabio", "image.edf", None, None]
    assert_url(url, expected)


def test_file_relative():
    url = DataUrl("image.edf")
    expected = [True, False, None, "image.edf", None, None]
    assert_url(url, expected)


def test_file_relative2():
    url = DataUrl("./foo/bar/image.edf")
    expected = [True, False, None, "./foo/bar/image.edf", None, None]
    assert_url(url, expected)


def test_file_relative3():
    url = DataUrl("foo/bar/image.edf")
    expected = [True, False, None, "foo/bar/image.edf", None, None]
    assert_url(url, expected)


def test_file_absolute():
    url = DataUrl("/data/image.edf")
    expected = [True, True, None, "/data/image.edf", None, None]
    assert_url(url, expected)


def test_file_absolute_windows():
    url = DataUrl("C:/data/image.edf")
    expected = [True, True, None, "C:/data/image.edf", None, None]
    assert_url(url, expected)


def test_absolute_with_path():
    url = DataUrl("/foo/foobar.h5?/foo/bar")
    expected = [True, True, None, "/foo/foobar.h5", "/foo/bar", None]
    assert_url(url, expected)


def test_windows_file_data_slice():
    url = DataUrl("C:/foo/foobar.h5?path=/foo/bar&slice=5,1")
    expected = [True, True, None, "C:/foo/foobar.h5", "/foo/bar", (5, 1)]
    assert_url(url, expected)


def test_scheme_file_data_slice():
    url = DataUrl("silx:/foo/foobar.h5?path=/foo/bar&slice=5,1")
    expected = [True, True, "silx", "/foo/foobar.h5", "/foo/bar", (5, 1)]
    assert_url(url, expected)


def test_scheme_windows_file_data_slice():
    url = DataUrl("silx:C:/foo/foobar.h5?path=/foo/bar&slice=5,1")
    expected = [True, True, "silx", "C:/foo/foobar.h5", "/foo/bar", (5, 1)]
    assert_url(url, expected)


def test_empty():
    url = DataUrl("")
    expected = [False, False, None, "", None, None]
    assert_url(url, expected)


def test_unknown_scheme():
    url = DataUrl("foo:/foo/foobar.h5?path=/foo/bar&slice=5,1")
    expected = [False, True, "foo", "/foo/foobar.h5", "/foo/bar", (5, 1)]
    assert_url(url, expected)


def test_slice():
    url = DataUrl("/a.h5?path=/b&slice=5,1")
    expected = [True, True, None, "/a.h5", "/b", (5, 1)]
    assert_url(url, expected)


def test_slice2():
    url = DataUrl("/a.h5?path=/b&slice=2:5")
    expected = [True, True, None, "/a.h5", "/b", (slice(2, 5),)]
    assert_url(url, expected)


def test_slice3():
    url = DataUrl("/a.h5?path=/b&slice=::2")
    expected = [True, True, None, "/a.h5", "/b", (slice(None, None, 2),)]
    assert_url(url, expected)


def test_slice_ellipsis():
    url = DataUrl("/a.h5?path=/b&slice=...")
    expected = [True, True, None, "/a.h5", "/b", (Ellipsis,)]
    assert_url(url, expected)


def test_slice_slicing():
    url = DataUrl("/a.h5?path=/b&slice=:")
    expected = [True, True, None, "/a.h5", "/b", (slice(None),)]
    assert_url(url, expected)


def test_slice_missing_element():
    url = DataUrl("/a.h5?path=/b&slice=5,,1")
    expected = [False, True, None, "/a.h5", "/b", None]
    assert_url(url, expected)


def test_slice_no_elements():
    url = DataUrl("/a.h5?path=/b&slice=")
    expected = [False, True, None, "/a.h5", "/b", None]
    assert_url(url, expected)


def test_create_relative_url():
    url = DataUrl(scheme="silx", file_path="./foo.h5", data_path="/", data_slice=(5, 1))
    assert not url.is_absolute()
    url2 = DataUrl(url.path())
    assert url == url2


def test_create_absolute_url():
    url = DataUrl(scheme="silx", file_path="/foo.h5", data_path="/", data_slice=(5, 1))
    url2 = DataUrl(url.path())
    assert url == url2


def test_create_absolute_windows_url():
    url = DataUrl(
        scheme="silx", file_path="C:/foo.h5", data_path="/", data_slice=(5, 1)
    )
    url2 = DataUrl(url.path())
    assert url == url2


def test_create_slice_url():
    url = DataUrl(
        scheme="silx",
        file_path="/foo.h5",
        data_path="/",
        data_slice=(5, 1, Ellipsis, slice(None)),
    )
    url2 = DataUrl(url.path())
    assert url == url2


def test_wrong_url():
    url = DataUrl(scheme="silx", file_path="/foo.h5", data_slice=(5, 1))
    assert not url.is_valid()


@pytest.mark.parametrize(
    "data",
    [
        (1, "silx:///foo.h5?slice=1"),
        ((1,), "silx:///foo.h5?slice=1"),
        (slice(None), "silx:///foo.h5?slice=:"),
        (slice(1, None), "silx:///foo.h5?slice=1:"),
        (slice(None, -2), "silx:///foo.h5?slice=:-2"),
        (slice(1, None, 3), "silx:///foo.h5?slice=1::3"),
        (slice(None, 2, 3), "silx:///foo.h5?slice=:2:3"),
        (slice(None, None, 3), "silx:///foo.h5?slice=::3"),
        (slice(1, 2, 3), "silx:///foo.h5?slice=1:2:3"),
        ((1, slice(1, 2)), "silx:///foo.h5?slice=1,1:2"),
    ],
)
def test_path_creation(data):
    """make sure the construction of path succeed and that we can
    recreate a DataUrl from a path"""
    data_slice, expected_path = data
    url = DataUrl(scheme="silx", file_path="/foo.h5", data_slice=data_slice)
    path = url.path()
    DataUrl(path=path)
    assert path == expected_path


def test_file_path_none():
    """
    make sure a file path can be None
    """
    url = DataUrl(scheme="silx", file_path=None, data_path="/path/to/data")
    assert url.file_path() is None
    assert url.scheme() == "silx"
    assert url.data_path() == "/path/to/data"


def test_data_path_none():
    """
    make sure a data path can be None
    """
    url = DataUrl(scheme="silx", file_path="my_file.hdf5", data_path=None)
    assert url.file_path() == "my_file.hdf5"
    assert url.scheme() == "silx"
    assert url.data_path() is None


def test_scheme_none():
    """
    make sure a scheme can be None
    """
    url = DataUrl(scheme=None, file_path="my_file.hdf5", data_path="/path/to/data")
    assert url.file_path() == "my_file.hdf5"
    assert url.scheme() is None
    assert url.data_path() == "/path/to/data"
