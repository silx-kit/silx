# /*##########################################################################
# Copyright (C) 2021 European Synchrotron Radiation Facility
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
"""Test silx.io.convert.write_to_h5"""


import h5py
import numpy
from silx.io import spech5

from silx.io.convert import write_to_h5
from silx.io.dictdump import h5todict
from silx.io import commonh5
from silx.io.spech5 import SpecH5


def test_with_commonh5(tmp_path):
    """Test write_to_h5 with commonh5 input"""
    fobj = commonh5.File("filename.txt", mode="w")
    group = fobj.create_group("group")
    dataset = group.create_dataset("dataset", data=numpy.array(50))
    group["soft_link"] = dataset # Create softlink

    output_filepath = tmp_path / "output.h5"
    write_to_h5(fobj, str(output_filepath))

    assert h5todict(str(output_filepath)) == {
        'group': {'dataset': numpy.array(50), 'soft_link': numpy.array(50)},
    }
    with h5py.File(output_filepath, mode="r") as h5file:
        soft_link = h5file.get("/group/soft_link", getlink=True)
        assert isinstance(soft_link, h5py.SoftLink)
        assert soft_link.path == "/group/dataset"


def test_with_hdf5(tmp_path):
    """Test write_to_h5 with HDF5 file input"""
    filepath = tmp_path / "base.h5"
    with h5py.File(filepath, mode="w") as h5file:
        h5file["group/dataset"] = 50
        h5file["group/soft_link"] = h5py.SoftLink("/group/dataset")
        h5file["group/external_link"] = h5py.ExternalLink("base.h5", "/group/dataset")

    output_filepath = tmp_path / "output.h5"
    write_to_h5(str(filepath), str(output_filepath))
    assert h5todict(str(output_filepath)) == {
        'group': {'dataset': 50, 'soft_link': 50},
    }
    with h5py.File(output_filepath, mode="r") as h5file:
        soft_link = h5file.get("group/soft_link", getlink=True)
        assert isinstance(soft_link, h5py.SoftLink)
        assert soft_link.path == "/group/dataset"


def test_with_spech5(tmp_path):
    """Test write_to_h5 with SpecH5 input"""
    filepath = tmp_path / "file.spec"
    filepath.write_bytes(
        bytes(
"""#F /tmp/sf.dat

#S 1 cmd
#L a  b
1 2
""",
        encoding='ascii')
    )

    output_filepath = tmp_path / "output.h5"
    with spech5.SpecH5(str(filepath)) as spech5file:
        write_to_h5(spech5file, str(output_filepath))
    print(h5todict(str(output_filepath)))

    def assert_equal(item1, item2):
        if isinstance(item1, dict):
            assert tuple(item1.keys()) == tuple(item2.keys())
            for key in item1.keys():
                assert_equal(item1[key], item2[key])
        else:
            numpy.array_equal(item1, item2)

    assert_equal(h5todict(str(output_filepath)), {
        '1.1': {
            'instrument': {
                'positioners': {},
                'specfile': {
                    'file_header': ['#F /tmp/sf.dat'],
                    'scan_header': ['#S 1 cmd', '#L a  b'],
                },
            },
            'measurement': {
                'a': [1.],
                'b': [2.],
            },
            'start_time': '',
            'title': 'cmd',
        },
    })
