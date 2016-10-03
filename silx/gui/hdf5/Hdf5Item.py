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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/10/2016"


import numpy
import logging
from .. import qt
from .. import icons
from . import _utils
from .Hdf5Node import Hdf5Node

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e


class Hdf5Item(Hdf5Node):
    """Subclass of :class:`qt.QStandardItem` to represent an HDF5-like
    item (dataset, file, group or link) as an element of a HDF5-like
    tree structure.
    """

    def __init__(self, text, obj, parent, key=None, h5pyClass=None, isBroken=False, populateAll=False):
        """
        :param str text: text displayed
        :param object obj: Pointer to h5py data. See the `obj` attribute.
        """
        self.__obj = obj
        self.__key = key
        self.__h5pyClass = h5pyClass
        self.__isBroken = isBroken
        self.__error = None
        self.__text = text
        Hdf5Node.__init__(self, parent, populateAll=populateAll)

    @property
    def obj(self):
        if self.__key:
            self.__initH5pyObject()
        return self.__obj

    @property
    def basename(self):
        return self.__text

    @property
    def h5pyClass(self):
        """Returns the class of the stored object.

        When the object is in lazy loading, this method should be able to
        return the type of the futrue loaded object. It allows to delay the
        real load of the object.

        :rtype: h5py.File or h5py.Dataset or h5py.Group
        """
        if self.__h5pyClass is None:
            if hasattr(self.obj, "h5py_class"):
                self.__h5pyClass = self.obj.h5py_class
            else:
                self.__h5pyClass = self.obj.__class__
        return self.__h5pyClass

    def isGroupObj(self):
        """Returns true if the stored HDF5 object is a group (contains sub
        groups or datasets).

        :rtype: bool
        """
        return issubclass(self.h5pyClass, h5py.Group)

    def isBrokenObj(self):
        """Returns true if the stored HDF5 object is broken.

        The stored object is then an h5py link (external or not) which point
        to nowhere (tbhe external file is not here, the expected dataset is
        still not on the file...)

        :rtype: bool
        """
        return self.__isBroken

    def _expectedChildCount(self):
        if self.isGroupObj():
            return len(self.obj)
        return 0

    def __initH5pyObject(self):
        """Lazy load of the HDF5 node. It is reached from the parent node
        with the key of the node."""
        parent_obj = self.parent.obj

        try:
            obj = parent_obj.get(self.__key)
        except Exception as e:
            _logger.debug("Internal h5py error", exc_info=True)
            try:
                self.__obj = parent_obj.get(self.__key, getlink=True)
            except Exception:
                self.__obj = None
            self.__error = e.args[0]
            self.__isBroken = True
        else:
            if obj is None:
                # that's a broken link
                self.__obj = parent_obj.get(self.__key, getlink=True)

                # TODO monkeypatch file (ask that in h5py for consistancy)
                if not hasattr(self.__obj, "name"):
                    parent_name = parent_obj.name
                    if parent_name == "/":
                        self.__obj.name = "/" + self.__key
                    else:
                        self.__obj.name = parent_name + "/" + self.__key
                # TODO monkeypatch file (ask that in h5py for consistancy)
                if not hasattr(self.__obj, "file"):
                    self.__obj.file = parent_obj.file

                if isinstance(self.__obj, h5py.ExternalLink):
                    message = "External link broken. Path %s::%s does not exist" % (self.__obj.filename, self.__obj.path)
                elif isinstance(self.__obj, h5py.SoftLink):
                    message = "Soft link broken. Path %s does not exist" % (self.__obj.path)
                else:
                    name = self.obj.__class__.__name__.split(".")[-1].capitalize()
                    message = "%s broken" % (name)
                self.__error = message
                self.__isBroken = True
            else:
                self.__obj = obj

        self.__key = None

    def _populateChild(self, populateAll=False):
        if self.isGroupObj():
            for name in self.obj:
                try:
                    class_ = self.obj.get(name, getclass=True)
                    has_error = False
                except Exception as e:
                    _logger.error("Internal h5py error", exc_info=True)
                    try:
                        class_ = self.obj.get(name, getclass=True, getlink=True)
                    except Exception as e:
                        class_ = h5py.HardLink
                    has_error = True
                item = Hdf5Item(text=name, obj=None, parent=self, key=name, h5pyClass=class_, isBroken=has_error)
                self.appendChild(item)

    def hasChildren(self):
        """Retuens true of this node have chrild.

        :rtype: bool
        """
        if not self.isGroupObj():
            return False
        return Hdf5Node.hasChildren(self)

    def _getDefaultIcon(self):
        """Returns the icon displayed by the main column.

        :rtype: qt.QIcon
        """
        style = qt.QApplication.style()
        if self.__isBroken:
            icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
            return icon
        class_ = self.h5pyClass
        if issubclass(class_, h5py.File):
            return style.standardIcon(qt.QStyle.SP_FileIcon)
        elif issubclass(class_, h5py.Group):
            return style.standardIcon(qt.QStyle.SP_DirIcon)
        elif issubclass(class_, h5py.SoftLink):
            return style.standardIcon(qt.QStyle.SP_DirLinkIcon)
        elif issubclass(class_, h5py.ExternalLink):
            return style.standardIcon(qt.QStyle.SP_FileLinkIcon)
        elif issubclass(class_, h5py.Dataset):
            if len(self.obj.shape) < 4:
                name = "item-%ddim" % len(self.obj.shape)
            else:
                name = "item-ndim"
            if str(self.obj.dtype) == "object":
                name = "item-object"
            icon = icons.getQIcon(name)
            return icon
        return None

    def _getDefaultTooltip(self):
        """Returns the default tooltip

        :rtype: str
        """
        if self.__error is not None:
            self.obj  # lazy loading of the object
            return self.__error
        class_ = self.h5pyClass

        attrs = {}
        if issubclass(class_, h5py.Dataset):
            attrs = dict(self.obj.attrs)
            if self.obj.shape == ():
                attrs["shape"] = "scalar"
            else:
                attrs["shape"] = self.obj.shape
            attrs["dtype"] = self.obj.dtype
            if self.obj.shape == ():
                attrs["value"] = self.obj.value
            else:
                attrs["value"] = "..."
        elif isinstance(self.obj, h5py.ExternalLink):
            attrs["linked path"] = self.obj.path
            attrs["linked file"] = self.obj.filename
        elif isinstance(self.obj, h5py.SoftLink):
            attrs["linked path"] = self.obj.path

        if len(attrs) > 0:
            tooltip = _utils.htmlFromDict(attrs)
        else:
            tooltip = ""

        return tooltip

    def dataName(self, role):
        """Data for the name column"""
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            return self.__text
        if role == qt.Qt.DecorationRole:
            return self._getDefaultIcon()
        if role == qt.Qt.ToolTipRole:
            return self._getDefaultTooltip()
        return None

    def dataType(self, role):
        """Data for the type column"""
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            if self.__error is not None:
                return ""
            class_ = self.h5pyClass
            if issubclass(class_, h5py.Dataset):
                if self.obj.dtype.type == numpy.string_:
                    text = "string"
                else:
                    text = str(self.obj.dtype)
            else:
                text = ""
            return text

        return None

    def dataShape(self, role):
        """Data for the shape column"""
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            if self.__error is not None:
                return ""
            class_ = self.h5pyClass
            if not issubclass(class_, h5py.Dataset):
                return ""
            shape = [str(i) for i in self.obj.shape]
            text = u" \u00D7 ".join(shape)
            return text
        return None

    def dataValue(self, role):
        """Data for the value column"""
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            if self.__error is not None:
                return ""
            class_ = self.h5pyClass
            if not issubclass(class_, h5py.Dataset):
                return ""

            numpy_object = self.obj.value

            if self.obj.dtype.type == numpy.object_:
                text = str(numpy_object)
            elif self.obj.dtype.type == numpy.string_:
                text = str(numpy_object)
            else:
                size = 1
                for dim in numpy_object.shape:
                    size = size * dim

                if size > 5:
                    text = "..."
                else:
                    text = str(numpy_object)
            return text
        return None

    def dataDescription(self, role):
        """Data for the description column"""
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            if self.__isBroken:
                self.obj  # lazy loading of the object
                return self.__error
            if "desc" in self.obj.attrs:
                text = self.obj.attrs["desc"]
            else:
                return ""
            return text
        if role == qt.Qt.ToolTipRole:
            if self.__error is not None:
                self.obj  # lazy loading of the object
                self.__initH5pyObject()
                return self.__error
            if "desc" in self.obj.attrs:
                text = self.obj.attrs["desc"]
            else:
                return ""
            return "Description: %s" % text
        return None

    def dataNode(self, role):
        """Data for the node column"""
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            class_ = self.h5pyClass
            text = class_.__name__.split(".")[-1]
            return text
        if role == qt.Qt.ToolTipRole:
            class_ = self.h5pyClass
            return "Class name: %s" % self.__class__
        return None
