# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module adds convenient functions to use plot3d widgets from the console.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "07/02/2018"


from collections import Iterable
import logging
import numpy

from ..gui import qt
from ..gui.plot3d.ScalarFieldView import ScalarFieldView
from ..gui.plot3d import SFViewParamTree
from ..gui.plot.Colormap import Colormap
from ..gui.plot.Colors import rgba


_logger = logging.getLogger(__name__)


def contour3d(scalars,
              contours=1,
              copy=True,
              color=None,
              colormap='viridis',
              vmin=None,
              vmax=None,
              opacity=1.):
    """Plot isosurfaces of a 3D scalar field in a dedicated widget.

    It opens a silx plot3d.SceneWindow.

    Examples:

    First import :mod:`sx` functions:

    >>> from silx import sx

    Provided data, a 3D scalar field as a numpy array of float32:

    >>> plot3d_window = sx.contour3d(data)

    Alternatively you can provide the level of the isosurfaces:

    >>> plot3d_window = sx.contour3d(data, contours=[0.2, 0.4])

    This function is similar to mayavi.mlab.contour3d.

    :param scalars: The 3D scalar field to visualize
    :type: numpy.ndarray of float32 with 3 dimensions
    :param Union[int, float, List[float]] contours:
        Either the number of isosurfaces to draw (as an int) or
        the isosurface level (as a float) or a list of isosurface levels
        (as a list of float)
    :param bool copy:
        True (default) to make a copy of scalars.
        False to avoid this copy (do not modify provided data afterwards)
    :param color:
        Color.s to use for isosurfaces.
        Either a single color or a list of colors (one for each isosurface).
        A color can be defined by its name (as a str) or
        as RGB(A) as float or uint8.
    :param str colormap:
        If color is not provided, this colormap is used
        for coloring isosurfaces.
    :param Union[float, None] vmin:
        Minimum value of the colormap
    :param Union[float, None] vmax:
        Maximum value of the colormap
    :param float opacity:
        Transparency of the isosurfaces as a float in [0., 1.]
    :return: The widget used to visualize the data
    :rtype: ~silx.gui.plot3d.SceneWindow.SceneWindow
    """
    # Prepare isolevel values
    if isinstance(contours, int):
        # Compute contours number of isovalues
        mean = numpy.mean(scalars)
        std = numpy.std(scalars)

        start = mean - std * ((contours - 1) // 2)
        contours = [start + std * index for index in range(contours)]

    elif isinstance(contours, float):
        contours = [contours]

    assert isinstance(contours, Iterable)

    # Prepare colors
    if color is not None:
        if isinstance(color, str) or isinstance(color[0], (int, float)):
            # Single color provided, use it for all isosurfaces
            colors = [rgba(color)] * len(contours)
        else:
            # As many colors as contours
            colors = [rgba(c) for c in color]

        # convert colors from float to uint8
        colors = (numpy.array(colors) * 255).astype(numpy.uint8)

    else:  # Use colormap
        colormap = Colormap(name=colormap, vmin=vmin, vmax=vmax)
        colors = colormap.applyToData(contours)

    assert len(colors) == len(contours)

    # Prepare and apply opacity
    assert isinstance(opacity, float)
    opacity = min(max(0., opacity), 1.)  # Clip opacity
    colors[:, -1] = (colors[:, -1] * opacity).astype(numpy.uint8)

    # Prepare widget
    scalarField = ScalarFieldView()

    scalarField.setBackgroundColor((0.9, 0.9, 0.9))
    scalarField.setForegroundColor((0.1, 0.1, 0.1))
    scalarField.setData(scalars, copy=copy)

    # Create a parameter tree for the scalar field view
    treeView = SFViewParamTree.TreeView(scalarField)
    treeView.setSfView(scalarField)  # Attach the parameter tree to the view

    # Add the parameter tree to the main window in a dock widget
    dock = qt.QDockWidget()
    dock.setWindowTitle('Parameters')
    dock.setWidget(treeView)
    scalarField.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

    for level, color in zip(contours, colors):
        scalarField.addIsosurface(level, color)

    scalarField.show()

    return scalarField
