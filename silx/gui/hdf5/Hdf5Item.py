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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "26/09/2017"


import logging
import collections
from .. import qt
from .. import icons
from . import _utils
from .Hdf5Node import Hdf5Node
import silx.io.utils
from silx.gui.data.TextFormatter import TextFormatter
from ..hdf5.Hdf5Formatter import Hdf5Formatter

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e

_formatter = TextFormatter()
_hdf5Formatter = Hdf5Formatter(textFormatter=_formatter)
# FIXME: The formatter should be an attribute of the Hdf5Model


class Hdf5Item(Hdf5Node):
    """Subclass of :class:`qt.QStandardItem` to represent an HDF5-like
    item (dataset, file, group or link) as an element of a HDF5-like
    tree structure.
    """

    def __init__(self, text, obj, parent, key=None, h5pyClass=None, linkClass=None, populateAll=False):
        """
        :param str text: text displayed
        :param object obj: Pointer to h5py data. See the `obj` attribute.
        """
        self.__obj = obj
        self.__key = key
        self.__h5pyClass = h5pyClass
        self.__isBroken = obj is None and h5pyClass is None
        self.__error = None
        self.__text = text
        self.__linkClass = linkClass
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
        if self.__h5pyClass is None and self.obj is not None:
            self.__h5pyClass = silx.io.utils.get_h5py_class(self.obj)
        return self.__h5pyClass

    @property
    def linkClass(self):
        """Returns the link class object of this node

        :type: h5py.SoftLink or h5py.HardLink or h5py.ExternalLink or None
        """
        return self.__linkClass

    def isGroupObj(self):
        """Returns true if the stored HDF5 object is a group (contains sub
        groups or datasets).

        :rtype: bool
        """
        if self.h5pyClass is None:
            return False
        return issubclass(self.h5pyClass, h5py.Group)

    def isBrokenObj(self):
        """Returns true if the stored HDF5 object is broken.

        The stored object is then an h5py link (external or not) which point
        to nowhere (tbhe external file is not here, the expected dataset is
        still not on the file...)

        :rtype: bool
        """
        return self.__isBroken

    def _getFormatter(self):
        """
        Returns an Hdf5Formatter

        :rtype: Hdf5Formatter
        """
        return _hdf5Formatter

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

                # TODO monkey-patch file (ask that in h5py for consistency)
                if not hasattr(self.__obj, "name"):
                    parent_name = parent_obj.name
                    if parent_name == "/":
                        self.__obj.name = "/" + self.__key
                    else:
                        self.__obj.name = parent_name + "/" + self.__key
                # TODO monkey-patch file (ask that in h5py for consistency)
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
                if not self.isGroupObj():
                    try:
                        # pre-fetch of the data
                        if obj.shape is None:
                            pass
                        elif obj.shape == tuple():
                            obj[()]
                        else:
                            if obj.compression is None and obj.size > 0:
                                key = tuple([0] * len(obj.shape))
                                obj[key]
                    except Exception as e:
                        _logger.debug(e, exc_info=True)
                        message = "%s broken. %s" % (self.__obj.name, e.args[0])
                        self.__error = message
                        self.__isBroken = True

        self.__key = None

    def _populateChild(self, populateAll=False):
        if self.isGroupObj():
            for name in self.obj:
                try:
                    class_ = self.obj.get(name, getclass=True)
                    link = self.obj.get(name, getclass=True, getlink=True)
                except Exception as e:
                    _logger.warn("Internal h5py error", exc_info=True)
                    class_ = None
                    try:
                        link = self.obj.get(name, getclass=True, getlink=True)
                    except Exception as e:
                        link = h5py.HardLink
                item = Hdf5Item(text=name, obj=None, parent=self, key=name, h5pyClass=class_, linkClass=link)
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
        # Pre-fetch the object, in case it is broken
        obj = self.obj
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
            if obj.shape is None:
                name = "item-none"
            elif len(obj.shape) < 4:
                name = "item-%ddim" % len(obj.shape)
            else:
                name = "item-ndim"
            icon = icons.getQIcon(name)
            return icon
        return None

    def _createTooltipAttributes(self):
        """
        Add key/value attributes that will be displayed in the item tooltip

        :param Dict[str,str] attributeDict: Key/value attributes
        """
        attributeDict = collections.OrderedDict()

        if issubclass(self.h5pyClass, h5py.Dataset):
            attributeDict["#Title"] = "HDF5 Dataset"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
            attributeDict["Shape"] = self._getFormatter().humanReadableShape(self.obj)
            attributeDict["Value"] = self._getFormatter().humanReadableValue(self.obj)
            attributeDict["Data type"] = self._getFormatter().humanReadableType(self.obj, full=True)
        elif issubclass(self.h5pyClass, h5py.Group):
            attributeDict["#Title"] = "HDF5 Group"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
        elif issubclass(self.h5pyClass, h5py.File):
            attributeDict["#Title"] = "HDF5 File"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = "/"
        elif isinstance(self.obj, h5py.ExternalLink):
            attributeDict["#Title"] = "HDF5 External Link"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
            attributeDict["Linked path"] = self.obj.path
            attributeDict["Linked file"] = self.obj.filename
        elif isinstance(self.obj, h5py.SoftLink):
            attributeDict["#Title"] = "HDF5 Soft Link"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
            attributeDict["Linked path"] = self.obj.path
        else:
            pass
        return attributeDict

    def _getDefaultTooltip(self):
        """Returns the default tooltip

        :rtype: str
        """
        if self.__error is not None:
            self.obj  # lazy loading of the object
            return self.__error

        attrs = self._createTooltipAttributes()
        title = attrs.pop("#Title", None)
        if len(attrs) > 0:
            tooltip = _utils.htmlFromDict(attrs, title=title)
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
                text = self._getFormatter().humanReadableType(self.obj)
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
            return self._getFormatter().humanReadableShape(self.obj)
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
            if not issubclass(self.h5pyClass, h5py.Dataset):
                return ""
            return self._getFormatter().humanReadableValue(self.obj)
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
            if self.isBrokenObj():
                return ""
            class_ = self.h5pyClass
            text = class_.__name__.split(".")[-1]
            return text
        if role == qt.Qt.ToolTipRole:
            class_ = self.h5pyClass
            if class_ is None:
                return ""
            return "Class name: %s" % self.__class__
        return None

    def dataLink(self, role):
        """Data for the link column

        Overwrite it to implement the content of the 'link' column.

        :rtype: qt.QVariant
        """
        if role == qt.Qt.DecorationRole:
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            link = self.linkClass
            if link is None:
                return ""
            elif link is h5py.ExternalLink:
                return "External"
            elif link is h5py.SoftLink:
                return "Soft"
            elif link is h5py.HardLink:
                return ""
            else:
                return link.__name__
        if role == qt.Qt.ToolTipRole:
            return None
        return None
