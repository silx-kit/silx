# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""
This module contains utilitaries that should be moved into silx.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/11/2017"

import fabio

_fabioFormats = set([])


def _fabioAvailableExtensions():
    global _fabioFormats
    if len(_fabioFormats) > 0:
        return _fabioFormats

    formats = fabio.fabioformats.get_classes(reader=True)
    allExtensions = set([])

    for reader in formats:
        if not hasattr(reader, "DESCRIPTION"):
            continue
        if not hasattr(reader, "DEFAULT_EXTENTIONS"):
            continue

        ext = reader.DEFAULT_EXTENTIONS
        ext = ["*.%s" % e for e in ext]
        allExtensions.update(ext)

    allExtensions = list(sorted(list(allExtensions)))
    _fabioFormats = set(allExtensions)
    return _fabioFormats


def supportedFileFormats(h5py=True, spec=True, fabio=True, numpy=True):
    """Returns the list of supported file extensions using silx.open.

    :returns: A dictionary indexed by file description and containg a set of
        extensions (an extension is a string like "*.ext").
    :rtype: Dict[str, List[str]]
    """
    formats = {}
    if h5py:
        formats["HDF5 files"] = set(["*.h5", "*.hdf"])
        formats["NeXus files"] = set(["*.nx", "*.nxs", "*.h5", "*.hdf"])
    if spec:
        formats["NeXus layout from spec files"] = set(["*.dat", "*.spec", "*.mca"])
    if fabio:
        formats["NeXus layout from fabio files"] = set(_fabioAvailableExtensions())
    if numpy:
        formats["Numpy binary files"] = set(["*.npz", "*.npy"])
    return formats
