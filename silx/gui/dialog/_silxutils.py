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
__date__ = "21/11/2017"

import fabio
from silx.gui import qt
from silx.gui.hdf5.Hdf5TreeModel import Hdf5TreeModel
from silx.gui.plot.Colormap import Colormap

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


def indexFromH5Object(model, h5Object):
    """This code should be inside silx"""
    if h5Object is None:
        return qt.QModelIndex()

    filename = h5Object.file.filename

    # Seach for the right roots
    rootIndices = []
    for index in range(model.rowCount(qt.QModelIndex())):
        index = model.index(index, 0, qt.QModelIndex())
        obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
        if obj.file.filename == filename:
            # We can have many roots with different subtree of the same
            # root
            rootIndices.append(index)

    if len(rootIndices) == 0:
        # No root found
        return qt.QModelIndex()

    path = h5Object.name + "/"
    path = path.replace("//", "/")

    # Search for the right node
    found = False
    foundIndices = []
    for _ in range(1000 * len(rootIndices)):
        # Avoid too much iterations, in case of recurssive links
        if len(foundIndices) == 0:
            if len(rootIndices) == 0:
                # Nothing found
                break
            # Start fron a new root
            foundIndices.append(rootIndices.pop(0))

            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                found = True
                break

        parentIndex = foundIndices[-1]
        for index in range(model.rowCount(parentIndex)):
            index = model.index(index, 0, parentIndex)
            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)

            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                foundIndices.append(index)
                found = True
                break
            elif path.startswith(p):
                foundIndices.append(index)
                break
        else:
            # Nothing found, start again with another root
            foundIndices = []

        if found:
            break

    if found:
        return foundIndices[-1]
    return qt.QModelIndex()


_colormapVersionSerial = 1


def readColormap(stream):
    """
    Read a colormap from a stream

    :param qt.QDataStream stream: Stream containing the state
    """
    className = stream.readString().decode("ascii")
    if className != Colormap.__name__:
        return None

    version = stream.readUInt32()
    if version != _colormapVersionSerial:
        return None

    isNone = stream.readBool()
    if isNone:
        return None
    name = stream.readString().decode("ascii")
    isNull = stream.readBool()
    if not isNull:
        vmin = stream.readQVariant()
    else:
        vmin = None
    isNull = stream.readBool()
    if not isNull:
        vmax = stream.readQVariant()
    else:
        vmax = None
    normalization = stream.readString().decode("ascii")
    return Colormap(name=name, normalization=normalization, vmin=vmin, vmax=vmax)


def writeColormap(stream, colormap):
    """
    Write a colormap to a stream

    :param qt.QDataStream stream: Stream to write the colormap
    :param silx.gui.plot.Colormap.Colormap colormap: The colormap
    """
    stream.writeString(Colormap.__name__.encode("ascii"))
    stream.writeUInt32(_colormapVersionSerial)
    stream.writeBool(colormap is None)
    if colormap is None:
        return
    stream.writeString(colormap.getName().encode("ascii"))
    stream.writeBool(colormap.getVMin() is None)
    if colormap.getVMin() is not None:
        stream.writeQVariant(colormap.getVMin())
    stream.writeBool(colormap.getVMax() is None)
    if colormap.getVMax() is not None:
        stream.writeQVariant(colormap.getVMax())
    stream.writeString(colormap.getNormalization().encode("ascii"))
