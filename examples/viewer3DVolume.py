# coding: utf-8
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
"""This script illustrates the use of :class:`silx.gui.plot3d.ScalarFieldView`.

It loads a 3D scalar data set from a file and displays iso-surfaces and
an interactive cutting plane.
It can also be started without providing a file.
"""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/01/2017"


import argparse
import logging
import os.path
import sys

import numpy

from silx.gui import qt

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d import SFViewParamTree

logging.basicConfig()

_logger = logging.getLogger(__name__)


try:
    import h5py
except ImportError:
    _logger.warning('h5py is not installed: HDF5 not supported')
    h5py = None


def load(filename):
    """Load 3D scalar field from file.

    It supports 3D stack HDF5 files and numpy files.

    :param str filename: Name of the file to open
                         and path in file for hdf5 file
    :return: numpy.ndarray with 3 dimensions.
    """
    if not os.path.isfile(filename.split('::')[0]):
        raise IOError('No input file: %s' % filename)

    if h5py is not None and h5py.is_hdf5(filename.split('::')[0]):
        if '::' not in filename:
            raise ValueError(
                'HDF5 path not provided: Use <filename>::<path> format')

        filename, path = filename.split('::')
        path, indices = path.split('#')[0], path.split('#')[1:]

        with h5py.File(filename) as f:
            data = f[path]

            # Loop through indices along first dimensions
            for index in indices:
                data = data[int(index)]

            data = numpy.array(data, order='C', dtype='float32')

    else:  # Try with numpy
        try:
            data = numpy.load(filename)
        except IOError:
            raise IOError('Unsupported file format: %s' % filename)

    if data.ndim != 3:
        raise RuntimeError(
            'Unsupported data set dimensions, only supports 3D datasets')

    return data


def default_isolevel(data):
    """Compute a default isosurface level: mean + 1 std

    :param numpy.ndarray data: The data to process
    :rtype: float
    """
    data = data[numpy.isfinite(data)]
    if len(data) == 0:
        return 0
    else:
        return numpy.mean(data) + numpy.std(data)


# Parse input arguments
parser = argparse.ArgumentParser(
    description=__doc__)
parser.add_argument(
    '-l', '--level', nargs='?', type=float, default=float('nan'),
    help="The value at which to generate the iso-surface")
parser.add_argument(
    '-sx', '--xscale', nargs='?', type=float, default=1.,
    help="The scale of the data on the X axis")
parser.add_argument(
    '-sy', '--yscale', nargs='?', type=float, default=1.,
    help="The scale of the data on the Y axis")
parser.add_argument(
    '-sz', '--zscale', nargs='?', type=float, default=1.,
    help="The scale of the data on the Z axis")
parser.add_argument(
    '-ox', '--xoffset', nargs='?', type=float, default=0.,
    help="The offset of the data on the X axis")
parser.add_argument(
    '-oy', '--yoffset', nargs='?', type=float, default=0.,
    help="The offset of the data on the Y axis")
parser.add_argument(
    '-oz', '--zoffset', nargs='?', type=float, default=0.,
    help="The offset of the data on the Z axis")
parser.add_argument(
    'filename',
    nargs='?',
    default=None,
    help="""Filename to open.

    It supports 3D volume saved as .npy or in .h5 files.

    It also support nD data set (n>=3) stored in a HDF5 file.
    For HDF5, provide the filename and path as: <filename>::<path_in_file>.
    If the data set has more than 3 dimensions, it is possible to choose a
    3D data set as a subset by providing the indices along the first n-3
    dimensions with '#':
    <filename>::<path_in_file>#<1st_dim_index>...#<n-3th_dim_index>

    E.g.: data.h5::/data_5D#1#1
    """)
args = parser.parse_args(args=sys.argv[1:])

# Start GUI
app = qt.QApplication([])

# Create the viewer main window
window = ScalarFieldView()

# Create a parameter tree for the scalar field view
treeView = SFViewParamTree.TreeView(window)
treeView.setSfView(window)  # Attach the parameter tree to the view

# Add the parameter tree to the main window in a dock widget
dock = qt.QDockWidget()
dock.setWindowTitle('Parameters')
dock.setWidget(treeView)
window.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

# Load data from file
if args.filename is not None:
    data = load(args.filename)
    _logger.info('Data:\n\tShape: %s\n\tRange: [%f, %f]',
                 str(data.shape), data.min(), data.max())
else:
    # Create dummy data
    _logger.warning('Not data file provided, creating dummy data')
    coords = numpy.linspace(-10, 10, 64)
    z = coords.reshape(-1, 1, 1)
    y = coords.reshape(1, -1, 1)
    x = coords.reshape(1, 1, -1)
    data = numpy.sin(x * y * z) / (x * y * z)

# Set ScalarFieldView data
window.setData(data)

# Set scale of the data
window.setScale(args.xscale, args.yscale, args.zscale)

# Set offset of the data
window.setTranslation(args.xoffset, args.yoffset, args.zoffset)

# Set axes labels
window.setAxesLabels('X', 'Y', 'Z')

# Add an iso-surface
if not numpy.isnan(args.level):
    # Add an iso-surface at the given iso-level
    window.addIsosurface(args.level, '#FF0000FF')
else:
    # Add an iso-surface from a function
    window.addIsosurface(default_isolevel, '#FF0000FF')

window.show()
app.exec_()
