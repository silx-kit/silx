# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""The silx package contains the following main sub-packages:

- silx.gui: Qt widgets for data visualization and data file browsing
- silx.image: Some processing functions for 2D images
- silx.io: Reading and writing data files (HDF5/NeXus, SPEC, ...)
- silx.math: Some processing functions for 1D, 2D, 3D, nD arrays
- silx.opencl: OpenCL-based data processing
- silx.sx: High-level silx functions suited for (I)Python console.
- silx.utils: Miscellaneous convenient functions

See silx documentation: http://www.silx.org/doc/silx/latest/
"""

from __future__ import absolute_import, print_function, division

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "26/04/2018"

import os as _os
import logging as _logging
from ._config import Config as _Config

config = _Config()
"""Global configuration shared with the whole library"""

# Attach a do nothing logging handler for silx
_logging.getLogger(__name__).addHandler(_logging.NullHandler())


project = _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))

try:
    from ._version import __date__ as date  # noqa
    from ._version import version, version_info, hexversion, strictversion  # noqa
except ImportError:
    raise RuntimeError("Do NOT use %s from its sources: build it and use the built version" % project)
