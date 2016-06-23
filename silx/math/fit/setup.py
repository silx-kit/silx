# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/06/2016"

import os.path

import numpy

from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('fit', parent_package, top_path)
    config.add_subpackage('test')

    # =====================================
    # fit functions
    # =====================================
    fit_dir = 'functions'
    fit_src = [os.path.join(fit_dir, "src", srcf)
               for srcf in ["funs.c"]]
    fit_src.append(os.path.join(fit_dir, "functions.pyx"))
    fit_inc = [os.path.join(fit_dir, 'include'), numpy.get_include()]

    config.add_extension('functions',
                         sources=fit_src,
                         include_dirs=fit_inc,
                         language='c')

    # =====================================
    # fit filters
    # =====================================
    fit_dir = 'filters'
    fit_src = [os.path.join(fit_dir, "src", srcf)
               for srcf in ["smoothnd.c", "snip1d.c",
                            "snip2d.c", "snip3d.c", "strip.c"]]
    fit_src.append(os.path.join(fit_dir, "filters.pyx"))
    fit_inc = [os.path.join(fit_dir, 'include'), numpy.get_include()]

    config.add_extension('filters',
                         sources=fit_src,
                         include_dirs=fit_inc,
                         language='c')

    # =====================================
    # peaks
    # =====================================
    fit_dir = 'peaks'
    fit_src = [os.path.join(fit_dir, "src", srcf)
               for srcf in ["peaks.c"]]
    fit_src.append(os.path.join(fit_dir, "peaks.pyx"))
    fit_inc = [os.path.join(fit_dir, 'include'), numpy.get_include()]

    config.add_extension('peaks',
                         sources=fit_src,
                         include_dirs=fit_inc,
                         language='c')
    # =====================================
    # =====================================
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
