# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "07/11/2017"

import os.path
import glob

from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs


def configuration(parent_package='', top_path=None):
    config = Configuration('scipy_spatial', parent_package, top_path)

    qhull_src = list(glob.glob(os.path.join(os.path.dirname(__file__), 'qhull',
                                    'src', '*.c')))
    config.add_extension('qhull',
                         sources=['qhull.pyx'] + qhull_src,
                         include_dirs=[get_numpy_include_dirs()])
    config.add_data_files('qhull/COPYING.txt')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
