# /*##########################################################################
# Copyright (C) 2018-2023 European Synchrotron Radiation Facility
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

import pytest
import h5py
from ..parseutils import filenames_to_dataurls


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("silx_app_utils")
    with h5py.File(tmp_path / "test1.h5", "w") as h5:
        h5["g1/sub1/data1"] = 1
        h5["g1/sub1/data2"] = 2
        h5["g1/sub2/data1"] = 3
    return tmp_path


def test_h5__datapath(data_path):
    urls = filenames_to_dataurls([data_path / "test1.h5::g1/sub1/data1"])
    urls = list(urls)
    assert len(urls) == 1
    assert urls[0].data_path().replace("\\", "/") == "g1/sub1/data1"


def test_h5__datapath_not_existing(data_path):
    urls = filenames_to_dataurls([data_path / "test1.h5::g1/sub0/data1"])
    urls = list(urls)
    assert len(urls) == 1
    assert urls[0].data_path().replace("\\", "/") == "g1/sub0/data1"


def test_h5__datapath_with_magic(data_path):
    urls = filenames_to_dataurls([data_path / "test1.h5::g1/sub*/data*"])
    urls = list(urls)
    assert len(urls) == 3


def test_h5__datapath_with_magic_not_existing(data_path):
    urls = filenames_to_dataurls([data_path / "test1.h5::g1/sub0/data*"])
    urls = list(urls)
    assert len(urls) == 0


def test_h5__datapath_with_recursive_magic(data_path):
    urls = filenames_to_dataurls([data_path / "test1.h5::**/data1"])
    urls = list(urls)
    assert len(urls) == 2
