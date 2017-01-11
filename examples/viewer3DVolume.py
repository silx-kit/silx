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
"""This script illustrates the use of silx.gui.plot3d.ScalarFieldView.

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

import numpy

from silx.gui import qt

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d import SFViewParamTree


logging.basicConfig()

_logger = logging.getLogger(__name__)


def load(filename):
    """Load 3D scalar field from file.

    It supports 3D stack HDF5 files and numpy files.

    :param str filename: Name of the file to open
                         and path in file for hdf5 file
    :return: numpy.ndarray with 3 dimensions.
    """
    if not os.path.isfile(filename.split('::')[0]):
        raise IOError('No input file: %s' % filename)

    if '.h5' in filename.lower():
        import h5py

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

    elif filename.lower().endswith('.npy'):
        data = numpy.load(filename)

    else:
        raise IOError('Unsupported file format: %s' % filename)

    if data.ndim != 3:
        raise RuntimeError(
            'Unsupported data set dimensions, only supports 3D datasets')

    return data


def main(argv=None):
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
    args = parser.parse_args(args=argv)

    # Start GUI
    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])

    # Create the viewer main window
    window = ScalarFieldView()
    window.setAttribute(qt.Qt.WA_DeleteOnClose)

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
        size = 128
        z, y, x = numpy.mgrid[0:size, 0:size, 0:size]
        data = numpy.asarray(
            size**2 - ((x-size/2)**2 + (y-size/2)**2 + (z-size/2)**2),
            dtype='float32')

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
        window.addIsosurface(args.level)
    else:
        # Add an iso-surface from a function
        window.addIsosurface(
            lambda data: numpy.mean(data) + numpy.std(data),
            '#FF0000FF')

    window.show()
    return app.exec_()


if __name__ == '__main__':
    import sys

    sys.exit(main(argv=sys.argv[1:]))
