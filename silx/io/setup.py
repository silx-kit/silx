# coding: ascii
#
# JK: Numpy.distutils which imports this does not handle utf-8 in version<1.12
#
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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

__authors__ = ["P. Knobel", "V.A. Sole"]
__license__ = "MIT"
__date__ = "03/10/2016"

import os
import sys

from numpy.distutils.misc_util import Configuration


# Locale and platform management
SPECFILE_USE_GNU_SOURCE = os.getenv("SPECFILE_USE_GNU_SOURCE")
if SPECFILE_USE_GNU_SOURCE is None:
    SPECFILE_USE_GNU_SOURCE = 0
    if sys.platform.lower().startswith("linux"):
        warn = ("silx.io.specfile WARNING:",
                "A cleaner locale independent implementation",
                "may be achieved setting SPECFILE_USE_GNU_SOURCE to 1",
                "For instance running this script as:",
                "SPECFILE_USE_GNU_SOURCE=1 python setup.py build")
        print(os.linesep.join(warn))
else:
    SPECFILE_USE_GNU_SOURCE = int(SPECFILE_USE_GNU_SOURCE)

if sys.platform == "win32":
    define_macros = [('WIN32', None)]
elif os.name.lower().startswith('posix'):
    define_macros = [('SPECFILE_POSIX', None)]
    # the best choice is to have _GNU_SOURCE defined
    # as a compilation flag because that allows the
    # use of strtod_l
    if SPECFILE_USE_GNU_SOURCE:
        define_macros = [('_GNU_SOURCE', 1)]
else:
    define_macros = []


def configuration(parent_package='', top_path=None):
    config = Configuration('io', parent_package, top_path)
    config.add_subpackage('test')
    config.add_subpackage('nxdata')

    srcfiles = ['sfheader', 'sfinit', 'sflists', 'sfdata', 'sfindex',
                'sflabel', 'sfmca', 'sftools', 'locale_management']
    sources = [os.path.join('specfile', 'src', ffile + '.c') for ffile in srcfiles]
    sources.append('specfile.pyx')

    config.add_extension('specfile',
                         sources=sources,
                         define_macros=define_macros,
                         include_dirs=[os.path.join('specfile', 'include')],
                         language='c')
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
