# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""This package provides a set of Qt widgets for displaying content relative to
HDF5 format.

.. note::

    This package depends on *h5py*.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "23/09/2016"


from .Hdf5TreeView import Hdf5TreeView  # noqa
from ._utils import H5Node
from ._utils import Hdf5ContextMenuEvent  # noqa
from .NexusSortFilterProxyModel import NexusSortFilterProxyModel  # noqa
from .Hdf5TreeModel import Hdf5TreeModel  # noqa

__all__ = ['Hdf5TreeView', 'H5Node', 'Hdf5ContextMenuEvent', 'NexusSortFilterProxyModel', 'Hdf5TreeModel']
