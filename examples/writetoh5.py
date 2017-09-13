#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""This script converts a supported data file (SPEC, EDF,...) to a HDF5 file.

By default, it creates a new output file or fails if the output file given
on the command line already exist, but the user can choose to overwrite
existing files, or append SPEC data to existing HDF5 files.

In case of appending data to HDF5 files, the user can choose between ignoring
input data if a corresponding dataset already exists in the output file, or
overwriting existing datasets.

By default, new scans are written to the root (/) of the HDF5 file, but it is
possible to specify a different target path.
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/09/2016"

import argparse
from silx.io.convert import write_to_h5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_path',
                    help='Path to input data file')
parser.add_argument('h5_path',
                    help='Path to output HDF5 file')
parser.add_argument('-t', '--target-path', default="/",
                    help='Name of the group in which to save the scans ' +
                         'in the output file')

mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument('-o', '--overwrite', action="store_true",
                        help='Overwrite output file if it exists, ' +
                             'else create new file.')
mode_group.add_argument('-a', '--append', action="store_true",
                        help='Append data to existing file if it exists, ' +
                             'else create new file.')

parser.add_argument('--overwrite-data', action="store_true",
                    help='In append mode, overwrite existing groups and ' +
                         'datasets in the output file, if they exist with ' +
                         'the same name as input data. By default, existing' +
                         ' data is not touched, corresponding input data is' +
                         ' ignored.')

args = parser.parse_args()

if args.overwrite_data and not args.append:
    print("Option --overwrite-data ignored " +
          "(only relevant combined with option -a)")

if args.overwrite:
    mode = "w"
elif args.append:
    mode = "a"
else:
    # by default, use "write" mode and fail if file already exists
    mode = "w-"

write_to_h5(args.input_path, args.h5_path,
            h5path=args.target_path,
            mode=mode,
            overwrite_data=args.overwrite_data)
