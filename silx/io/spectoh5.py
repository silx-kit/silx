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
"""Deprecated module. Use :mod:`convert` instead."""

from .convert import Hdf5Writer
from .convert import write_to_h5
from .convert import convert as other_convert

from silx.utils import deprecation

deprecation.deprecated_warning(type_="Module",
                               name="silx.io.spectoh5",
                               since_version="0.6",
                               replacement="silx.io.convert")


class SpecToHdf5Writer(Hdf5Writer):
    def __init__(self, h5path='/', overwrite_data=False,
                 link_type="hard", create_dataset_args=None):
        deprecation.deprecated_warning(
            type_="Class",
            name="SpecToHdf5Writer",
            since_version="0.6",
            replacement="silx.io.convert.Hdf5Writer")
        Hdf5Writer.__init__(self, h5path, overwrite_data,
                            link_type, create_dataset_args)

    # methods whose signatures changed
    def write(self, sfh5, h5f):
        Hdf5Writer.write(self, infile=sfh5, h5f=h5f)

    def append_spec_member_to_h5(self, spec_h5_name, obj):
        Hdf5Writer.append_member_to_h5(self,
                                       h5like_name=spec_h5_name,
                                       obj=obj)


@deprecation.deprecated(replacement="silx.io.convert.write_to_h5",
                        since_version="0.6")
def write_spec_to_h5(specfile, h5file, h5path='/',
                     mode="a", overwrite_data=False,
                     link_type="hard", create_dataset_args=None):

    write_to_h5(infile=specfile,
                h5file=h5file,
                h5path=h5path,
                mode=mode,
                overwrite_data=overwrite_data,
                link_type=link_type,
                create_dataset_args=create_dataset_args)


@deprecation.deprecated(replacement="silx.io.convert.convert",
                        since_version="0.6")
def convert(specfile, h5file, mode="w-",
            create_dataset_args=None):
    other_convert(infile=specfile,
                  h5file=h5file,
                  mode=mode,
                  create_dataset_args=create_dataset_args)
