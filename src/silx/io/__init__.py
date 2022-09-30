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
"""This package provides functionalities to read and write data files.

It is geared towards support of and conversion to HDF5/NeXus.

See silx documentation: http://www.silx.org/doc/silx/latest/
"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "11/12/2017"


from .utils import open  # pylint:disable=redefined-builtin
from .utils import save1D

from .utils import is_dataset
from .utils import is_file
from .utils import is_group
from .utils import is_softlink
from .utils import supported_extensions
from .utils import get_data

# avoid to import open with "import *"
__all = locals().keys()
__all = filter(lambda x: not x.startswith("_"), __all)
__all = filter(lambda x: x != "open", __all)
__all__ = list(__all)
