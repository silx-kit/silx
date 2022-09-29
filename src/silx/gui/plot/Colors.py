# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""Color conversion function, color dictionary and colormap tools."""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "14/06/2018"

import silx.utils.deprecation

silx.utils.deprecation.deprecated_warning("Module",
                                          name="silx.gui.plot.Colors",
                                          reason="moved",
                                          replacement="silx.gui.colors",
                                          since_version="0.8.0",
                                          only_once=True,
                                          skip_backtrace_count=1)

from ..colors import *  # noqa


@silx.utils.deprecation.deprecated(replacement='silx.gui.colors.Colormap.applyColormap')
def applyColormapToData(data,
                        name='gray',
                        normalization='linear',
                        autoscale=True,
                        vmin=0.,
                        vmax=1.,
                        colors=None):
    """Apply a colormap to the data and returns the RGBA image

    This supports data of any dimensions (not only of dimension 2).
    The returned array will have one more dimension (with 4 entries)
    than the input data to store the RGBA channels
    corresponding to each bin in the array.

    :param numpy.ndarray data: The data to convert.
    :param str name: Name of the colormap (default: 'gray').
    :param str normalization: Colormap mapping: 'linear' or 'log'.
    :param bool autoscale: Whether to use data min/max (True, default)
                           or [vmin, vmax] range (False).
    :param float vmin: The minimum value of the range to use if
                       'autoscale' is False.
    :param float vmax: The maximum value of the range to use if
                       'autoscale' is False.
    :param numpy.ndarray colors: Only used if name is None.
        Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
    :return: The computed RGBA image
    :rtype: numpy.ndarray of uint8
    """
    colormap = Colormap(name=name,
                        normalization=normalization,
                        vmin=vmin,
                        vmax=vmax,
                        colors=colors)
    return colormap.applyToData(data)


@silx.utils.deprecation.deprecated(replacement='silx.gui.colors.Colormap.getSupportedColormaps')
def getSupportedColormaps():
    """Get the supported colormap names as a tuple of str.

    The list should at least contain and start by:
    ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
    """
    return Colormap.getSupportedColormaps()
