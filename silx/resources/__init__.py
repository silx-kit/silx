# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Access project's data files.
"""

__authors__ = ["Thomas Vincent"]
__license__ = "MIT"
__date__ = "12/05/2016"


import os


data_dir = os.path.abspath(os.path.dirname(__file__))
"""Path to the directory containing data files.

This path can be overloaded by setting the SILX_DATA_DIR environment variable
to an alternative directory path.
This is usefull for run_tests.py to run tests in-place in the build directory.
"""

if 'SILX_DATA_DIR' in os.environ:
    if not os.path.isdir(os.environ['SILX_DATA_DIR']):
        raise RuntimeError(
            "SILX_DATA_DIR environment variable set to %s\n"
            "which is not a directory." % os.environ['SILX_DATA_DIR'])
    data_dir = os.path.abspath(os.environ['SILX_DATA_DIR'])


def resource_filename(resource):
    """Return filename corresponding to resource.

    :param str resource: Resource file path relative to resource directory
                         using '/' path separator.
    :return: Name of the resource file in the file system
    """
    filename = os.path.join(data_dir, *resource.split('/'))
    if not os.path.isfile(filename):
        raise ValueError('File does not exist: %s' % filename)
    return filename
