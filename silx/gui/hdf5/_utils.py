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
"""This package provides a set of helper class and function used by the
package `silx.gui.hdf5` package.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/10/2016"


import logging
from .. import qt

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e


class Hdf5ContextMenuEvent(object):
    """Hold information provided to context menu callbacks."""

    def __init__(self, source, menu, hoveredObject):
        """
        Constructor

        :param QWidget source: Widget source
        :param QMenu menu: Context menu which will be displayed
        :param H5Node hoveredObject: Hovered H5 node
        """
        self.__source = source
        self.__menu = menu
        self.__hoveredObject = hoveredObject

    def source(self):
        """Source of the event

        :rtype: Hdf5TreeView
        """
        return self.__source

    def menu(self):
        """Menu which will be displayed

        :rtype: qt.QMenu
        """
        return self.__menu

    def hoveredObject(self):
        """Item content hovered by the mouse when the context menu was
        requested

        :rtype: H5Node
        """
        return self.__menu


def htmlFromDict(dictionary):
    """Generate a readable HTML from a dictionary

    :param dict dictionary: A Dictionary
    :rtype: str
    """
    result = "<html><ul>"
    for key, value in dictionary.items():
        result += "<li><b>%s</b>: %s</li>" % (key, value)
    result += "</ul></html>"
    return result


class Hdf5NodeMimeData(qt.QMimeData):
    """Mimedata class to identify an internal drag and drop of a Hdf5Node."""

    MIME_TYPE = "application/x-internal-h5py-node"

    def __init__(self, node=None):
        qt.QMimeData.__init__(self)
        self.__node = node
        self.setData(self.MIME_TYPE, "".encode(encoding='utf-8'))

    def node(self):
        return self.__node


class H5Node(object):
    """Adapter over an h5py object to provide missing informations from h5py
    nodes, like internal node path and filename (which are not provided by
    :mod:`h5py` for soft and external links).

    It also provides an abstraction to reach node type for mimicked h5py
    objects.
    """

    def __init__(self, h5py_item=None):
        """Constructor

        :param Hdf5Item h5py_item: An Hdf5Item
        """
        self.__h5py_object = h5py_item.obj
        self.__h5py_item = h5py_item

    def __getattr__(self, name):
        return object.__getattribute__(self.__h5py_object, name)

    @property
    def h5py_object(self):
        """Returns the internal h5py node.

        :rtype: h5py.File or h5py.Group or h5py.Dataset
        """
        return self.__h5py_object

    @property
    def ntype(self):
        """Returns the node type, as an h5py class.

        :rtype:
            :class:`h5py.File`, :class:`h5py.Group` or :class:`h5py.Dataset`
        """
        if hasattr(self.__h5py_object, "h5py_class"):
            return self.__h5py_object.h5py_class
        else:
            return self.__h5py_object.__class__

    @property
    def basename(self):
        """Returns the basename of this h5py node. It is the last identifier of
        the path.

        :rtype: str
        """
        return self.__h5py_object.name.split("/")[-1]

    @property
    def local_name(self):
        """Returns the local path of this h5py node.

        For links, this path is not equal to the h5py one.

        :rtype: str
        """
        if self.__h5py_item is None:
            raise RuntimeError("h5py_item is not defined")

        result = []
        item = self.__h5py_item
        while item is not None:
            if issubclass(item.h5pyClass, h5py.File):
                break
            result.append(item.basename)
            item = item.parent
        if item is None:
            raise RuntimeError("The item does not have parent holding h5py.File")
        if result == []:
            return "/"
        result.append("")
        result.reverse()
        return "/".join(result)

    def __file_item(self):
        """Returns the parent item holding the :class:`h5py.File` object

        :rtype: h5py.File
        :raises RuntimeException: If no file are found
        """
        item = self.__h5py_item
        while item is not None:
            if issubclass(item.h5pyClass, h5py.File):
                return item
            item = item.parent
        raise RuntimeError("The item does not have parent holding h5py.File")

    @property
    def local_file(self):
        """Returns the local :class:`h5py.File` object.

        For path containing external links, this file is not equal to the h5py
        one.

        :rtype: h5py.File
        :raises RuntimeException: If no file are found
        """
        item = self.__file_item()
        return item.obj

    @property
    def local_filename(self):
        """Returns the local filename of the h5py node.

        For path containing external links, this path is not equal to the
        filename provided by h5py.

        :rtype: str
        :raises RuntimeException: If no file are found
        """
        return self.local_file.filename

    @property
    def local_basename(self):
        """Returns the local filename of the h5py node.

        For path containing links, this basename can be different than the
        basename provided by h5py.

        :rtype: str
        """
        if issubclass(self.__h5py_item.h5pyClass, h5py.File):
            return ""
        return self.__h5py_item.basename
