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
"""This package provides a set of helper class and function used by the
package `silx.gui.hdf5` package.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/01/2019"


import logging
import os.path

import silx.io.utils
import silx.io.url
from .. import qt
from silx.utils.html import escape

_logger = logging.getLogger(__name__)


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
        return self.__hoveredObject


def htmlFromDict(dictionary, title=None):
    """Generate a readable HTML from a dictionary

    :param dict dictionary: A Dictionary
    :rtype: str
    """
    result = """<html>
        <head>
        <style type="text/css">
        ul { -qt-list-indent: 0; list-style: none; }
        li > b {display: inline-block; min-width: 4em; font-weight: bold; }
        </style>
        </head>
        <body>
        """
    if title is not None:
        result += "<b>%s</b>" % escape(title)
    result += "<ul>"
    for key, value in dictionary.items():
        result += "<li><b>%s</b>: %s</li>" % (escape(key), escape(value))
    result += "</ul>"
    result += "</body></html>"
    return result


class Hdf5DatasetMimeData(qt.QMimeData):
    """Mimedata class to identify an internal drag and drop of a Hdf5Node."""

    MIME_TYPE = "application/x-internal-h5py-dataset"

    SILX_URI_TYPE = "application/x-silx-uri"

    def __init__(self, node=None, dataset=None, isRoot=False):
        qt.QMimeData.__init__(self)
        self.__dataset = dataset
        self.__node = node
        self.__isRoot = isRoot
        self.setData(self.MIME_TYPE, "".encode(encoding='utf-8'))
        if node is not None:
            h5Node = H5Node(node)
            silxUrl = h5Node.url
            self.setText(silxUrl)
            self.setData(self.SILX_URI_TYPE, silxUrl.encode(encoding='utf-8'))

    def isRoot(self):
        return self.__isRoot

    def node(self):
        return self.__node

    def dataset(self):
        if self.__node is not None:
            return self.__node.obj
        return self.__dataset


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
        self.__h5py_target = None
        self.__h5py_item = h5py_item

    def __getattr__(self, name):
        if hasattr(self.__h5py_object, name):
            attr = getattr(self.__h5py_object, name)
            return attr
        raise AttributeError("H5Node has no attribute %s" % name)

    def __get_target(self, obj):
        """
        Return the actual physical target of the provided object.

        Objects can contains links in the middle of the path, this function
        check each groups and remove this prefix in case of the link by the
        link of the path.

        :param obj: A valid h5py object (File, group or dataset)
        :type obj: h5py.Dataset or h5py.Group or h5py.File
        :rtype: h5py.Dataset or h5py.Group or h5py.File
        """
        elements = obj.name.split("/")
        if obj.name == "/":
            return obj
        elif obj.name.startswith("/"):
            elements.pop(0)
        path = ""
        subpath = ""
        while len(elements) > 0:
            e = elements.pop(0)
            subpath = path + "/" + e
            link = obj.parent.get(subpath, getlink=True)
            classlink = silx.io.utils.get_h5_class(link)

            if classlink == silx.io.utils.H5Type.EXTERNAL_LINK:
                subpath = "/".join(elements)
                external_obj = obj.parent.get(self.basename + "/" + subpath)
                return self.__get_target(external_obj)
            elif classlink == silx.io.utils.H5Type.SOFT_LINK:
                # Restart from this stat
                root_elements = link.path.split("/")
                if link.path == "/":
                    path = ""
                    root_elements = []
                elif link.path.startswith("/"):
                    path = ""
                    root_elements.pop(0)

                for name in reversed(root_elements):
                    elements.insert(0, name)
            else:
                path = subpath

        return obj.file[path]

    @property
    def h5py_target(self):
        if self.__h5py_target is not None:
            return self.__h5py_target
        self.__h5py_target = self.__get_target(self.__h5py_object)
        return self.__h5py_target

    @property
    def h5py_object(self):
        """Returns the internal h5py node.

        :rtype: h5py.File or h5py.Group or h5py.Dataset
        """
        return self.__h5py_object

    @property
    def h5type(self):
        """Returns the node type, as an H5Type.

        :rtype: H5Node
        """
        return silx.io.utils.get_h5_class(self.__h5py_object)

    @property
    def ntype(self):
        """Returns the node type, as an h5py class.

        :rtype:
            :class:`h5py.File`, :class:`h5py.Group` or :class:`h5py.Dataset`
        """
        type_ = self.h5type
        return silx.io.utils.h5type_to_h5py_class(type_)

    @property
    def basename(self):
        """Returns the basename of this h5py node. It is the last identifier of
        the path.

        :rtype: str
        """
        return self.__h5py_object.name.split("/")[-1]

    @property
    def is_broken(self):
        """Returns true if the node is a broken link.

        :rtype: bool
        """
        if self.__h5py_item is None:
            raise RuntimeError("h5py_item is not defined")
        return self.__h5py_item.isBrokenObj()

    @property
    def local_name(self):
        """Returns the path from the master file root to this node.

        For links, this path is not equal to the h5py one.

        :rtype: str
        """
        if self.__h5py_item is None:
            raise RuntimeError("h5py_item is not defined")

        result = []
        item = self.__h5py_item
        while item is not None:
            # stop before the root item (item without parent)
            if item.parent.parent is None:
                name = item.obj.name
                if name != "/":
                    result.append(item.obj.name)
                break
            else:
                result.append(item.basename)
            item = item.parent
        if item is None:
            raise RuntimeError("The item does not have parent holding h5py.File")
        if result == []:
            return "/"
        if not result[-1].startswith("/"):
            result.append("")
        result.reverse()
        name = "/".join(result)
        return name

    def __get_local_file(self):
        """Returns the file of the root of this tree

        :rtype: h5py.File
        """
        item = self.__h5py_item
        while item.parent.parent is not None:
            class_ = silx.io.utils.get_h5_class(class_=item.h5pyClass)
            if class_ == silx.io.utils.H5Type.FILE:
                break
            item = item.parent

        class_ = silx.io.utils.get_h5_class(class_=item.h5pyClass)
        if class_ == silx.io.utils.H5Type.FILE:
            return item.obj
        else:
            return item.obj.file

    @property
    def local_file(self):
        """Returns the master file in which is this node.

        For path containing external links, this file is not equal to the h5py
        one.

        :rtype: h5py.File
        :raises RuntimeException: If no file are found
        """
        return self.__get_local_file()

    @property
    def local_filename(self):
        """Returns the filename from the master file of this node.

        For path containing external links, this path is not equal to the
        filename provided by h5py.

        :rtype: str
        :raises RuntimeException: If no file are found
        """
        return self.local_file.filename

    @property
    def local_basename(self):
        """Returns the basename from the master file root to this node.

        For path containing links, this basename can be different than the
        basename provided by h5py.

        :rtype: str
        """
        class_ = self.__h5py_item.h5Class
        if class_ is not None and class_ == silx.io.utils.H5Type.FILE:
            return ""
        return self.__h5py_item.basename

    @property
    def physical_file(self):
        """Returns the physical file in which is this node.

        .. versionadded:: 0.6

        :rtype: h5py.File
        :raises RuntimeError: If no file are found
        """
        class_ = silx.io.utils.get_h5_class(self.__h5py_object)
        if class_ == silx.io.utils.H5Type.EXTERNAL_LINK:
            # It means the link is broken
            raise RuntimeError("No file node found")
        if class_ == silx.io.utils.H5Type.SOFT_LINK:
            # It means the link is broken
            return self.local_file

        physical_obj = self.h5py_target
        return physical_obj.file

    @property
    def physical_name(self):
        """Returns the path from the location this h5py node is physically
        stored.

        For broken links, this filename can be different from the
        filename provided by h5py.

        :rtype: str
        """
        class_ = silx.io.utils.get_h5_class(self.__h5py_object)
        if class_ == silx.io.utils.H5Type.EXTERNAL_LINK:
            # It means the link is broken
            return self.__h5py_object.path
        if class_ == silx.io.utils.H5Type.SOFT_LINK:
            # It means the link is broken
            return self.__h5py_object.path

        physical_obj = self.h5py_target
        return physical_obj.name

    @property
    def physical_filename(self):
        """Returns the filename from the location this h5py node is physically
        stored.

        For broken links, this filename can be different from the
        filename provided by h5py.

        :rtype: str
        """
        class_ = silx.io.utils.get_h5_class(self.__h5py_object)
        if class_ == silx.io.utils.H5Type.EXTERNAL_LINK:
            # It means the link is broken
            return self.__h5py_object.filename
        if class_ == silx.io.utils.H5Type.SOFT_LINK:
            # It means the link is broken
            return self.local_file.filename

        return self.physical_file.filename

    @property
    def physical_basename(self):
        """Returns the basename from the location this h5py node is physically
        stored.

        For broken links, this basename can be different from the
        basename provided by h5py.

        :rtype: str
        """
        return self.physical_name.split("/")[-1]

    @property
    def data_url(self):
        """Returns a :class:`silx.io.url.DataUrl` object identify this node in the file
        system.

        :rtype: ~silx.io.url.DataUrl
        """
        absolute_filename = os.path.abspath(self.local_filename)
        return silx.io.url.DataUrl(scheme="silx",
                                   file_path=absolute_filename,
                                   data_path=self.local_name)

    @property
    def url(self):
        """Returns an URL object identifying this node in the file
        system.

        This URL can be used in different ways.

        .. code-block:: python

            # Parsing the URL
            import silx.io.url
            dataurl = silx.io.url.DataUrl(item.url)
            # dataurl provides access to URL fields

            # Open a numpy array
            import silx.io
            dataset = silx.io.get_data(item.url)

            # Open an hdf5 object (URL targetting a file or a group)
            import silx.io
            with silx.io.open(item.url) as h5:
                ...your stuff...

        :rtype: str
        """
        data_url = self.data_url
        return data_url.path()
