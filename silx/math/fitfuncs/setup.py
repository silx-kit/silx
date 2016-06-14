# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "14/06/2016"

"""Setup script for the SPECFITFUNS module distribution.

Temporary script for early development. To be merged with silx/math/setup.py"""

import os, sys, glob
try:
    import numpy
except ImportError:
    text  = "You must have numpy installed.\n"
    text += "See http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103\n"
    raise ImportError(text)

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

if sys.platform == "win32":
    define_macros = [('WIN32',None)]
else:
    define_macros = []


#sources = glob.glob('*.c')
sources = glob.glob(os.path.join('src', '*.c'))
sources.append("fitfunctions.pyx")
inc_dir = ['include', numpy.get_include()]

setup(name="fitfuns",
      description="fit functions module",

      # Description of the modules and packages in the distribution
      ext_modules=cythonize([Extension(name          = 'fitfunctions',
                                       sources       = sources,
                                       define_macros = define_macros,
                                       include_dirs  = inc_dir,
                                       language='c')]))
