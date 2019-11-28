# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
__date__ = "17/01/2019"


import logging
import collections
import enum

from .. import qt
from .. import icons
from . import _utils
from .Hdf5Node import Hdf5Node
import silx.io.utils
from silx.gui.data.TextFormatter import TextFormatter
from ..hdf5.Hdf5Formatter import Hdf5Formatter
_logger = logging.getLogger(__name__)
_formatter = TextFormatter()
_hdf5Formatter = Hdf5Formatter(textFormatter=_formatter)
# FIXME: The formatter should be an attribute of the Hdf5Model


class DescriptionType(enum.Enum):
    """List of available kind of description.
    """
    ERROR = "error"
    DESCRIPTION = "description"
    TITLE = "title"
    PROGRAM = "program"
    NAME = "name"
    VALUE = "value"


class Hdf5Item(Hdf5Node):
    """Subclass of :class:`qt.QStandardItem` to represent an HDF5-like
    item (dataset, file, group or link) as an element of a HDF5-like
    tree structure.
    """

    def __init__(self, text, obj, parent, key=None, h5Class=None, linkClass=None, populateAll=False):
        """
        :param str text: text displayed
        :param object obj: Pointer to a h5py-link object. See the `obj` attribute.
        """
        self.__obj = obj
        self.__key = key
        self.__h5Class = h5Class
        self.__isBroken = obj is None and h5Class is None
        self.__error = None
        self.__text = text
        self.__linkClass = linkClass
        self.__description = None
        self.__nx_class = None
        Hdf5Node.__init__(self, parent, populateAll=populateAll)

    def _getCanonicalName(self):
        parent = self.parent
        if parent is None:
            return self.__text
        else:
            return "%s/%s" % (parent._getCanonicalName(), self.__text)

    @property
    def obj(self):
        if self.__key:
            self.__initH5Object()
        return self.__obj

    @property
    def basename(self):
        return self.__text

    @property
    def h5Class(self):
        """Returns the class of the stored object.

        When the object is in lazy loading, this method should be able to
        return the type of the futrue loaded object. It allows to delay the
        real load of the object.

        :rtype: silx.io.utils.H5Type
        """
        if self.__h5Class is None and self.obj is not None:
            self.__h5Class = silx.io.utils.get_h5_class(self.obj)
        return self.__h5Class

    @property
    def h5pyClass(self):
        """Returns the class of the stored object.

        When the object is in lazy loading, this method should be able to
        return the type of the futrue loaded object. It allows to delay the
        real load of the object.

        :rtype: h5py.File or h5py.Dataset or h5py.Group
        """
        type_ = self.h5Class
        return silx.io.utils.h5type_to_h5py_class(type_)

    @property
    def linkClass(self):
        """Returns the link class object of this node

        :rtype: H5Type
        """
        return self.__linkClass

    def isGroupObj(self):
        """Returns true if the stored HDF5 object is a group (contains sub
        groups or datasets).

        :rtype: bool
        """
        if self.h5Class is None:
            return False
        return self.h5Class in [silx.io.utils.H5Type.GROUP, silx.io.utils.H5Type.FILE]

    def isBrokenObj(self):
        """Returns true if the stored HDF5 object is broken.

        The stored object is then an h5py-like link (external or not) which
        point  to nowhere (tbhe external file is not here, the expected
        dataset is still not on the file...)

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

    def __initH5Object(self):
        """Lazy load of the HDF5 node. It is reached from the parent node
        with the key of the node."""
        parent_obj = self.parent.obj

        try:
            obj = parent_obj.get(self.__key)
        except Exception as e:
            _logger.error("Internal error while reaching HDF5 object: %s", str(e))
            _logger.debug("Backtrace", exc_info=True)
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

                class_ = silx.io.utils.get_h5_class(self.__obj)

                if class_ == silx.io.utils.H5Type.EXTERNAL_LINK:
                    message = "External link broken. Path %s::%s does not exist" % (self.__obj.filename, self.__obj.path)
                elif class_ == silx.io.utils.H5Type.SOFT_LINK:
                    message = "Soft link broken. Path %s does not exist" % (self.__obj.path)
                else:
                    name = self.__obj.__class__.__name__.split(".")[-1].capitalize()
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
            keys = []
            try:
                for name in self.obj:
                    keys.append(name)
            except Exception:
                lib_name = self.obj.__class__.__module__.split(".")[0]
                _logger.error("Internal %s error. The file is corrupted.", lib_name)
                _logger.debug("Backtrace", exc_info=True)
                if keys == []:
                    # If the file was open in READ_ONLY we still can reach something
                    # https://github.com/silx-kit/silx/issues/2262
                    try:
                        for name in self.obj:
                            keys.append(name)
                    except Exception:
                        lib_name = self.obj.__class__.__module__.split(".")[0]
                        _logger.error("Internal %s error (second time). The file is corrupted.", lib_name)
                        _logger.debug("Backtrace", exc_info=True)
            for name in keys:
                try:
                    class_ = self.obj.get(name, getclass=True)
                    link = self.obj.get(name, getclass=True, getlink=True)
                    link = silx.io.utils.get_h5_class(class_=link)
                except Exception:
                    lib_name = self.obj.__class__.__module__.split(".")[0]
                    _logger.error("Internal %s error", lib_name)
                    _logger.debug("Backtrace", exc_info=True)
                    class_ = None
                    try:
                        link = self.obj.get(name, getclass=True, getlink=True)
                        link = silx.io.utils.get_h5_class(class_=link)
                    except Exception:
                        _logger.debug("Backtrace", exc_info=True)
                        link = silx.io.utils.H5Type.HARD_LINK

                h5class = None
                if class_ is not None:
                    h5class = silx.io.utils.get_h5_class(class_=class_)
                    if h5class is None:
                        _logger.error("Class %s unsupported", class_)
                item = Hdf5Item(text=name, obj=None, parent=self, key=name, h5Class=h5class, linkClass=link)
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
        class_ = self.h5Class
        if class_ == silx.io.utils.H5Type.FILE:
            return style.standardIcon(qt.QStyle.SP_FileIcon)
        elif class_ == silx.io.utils.H5Type.GROUP:
            return style.standardIcon(qt.QStyle.SP_DirIcon)
        elif class_ == silx.io.utils.H5Type.SOFT_LINK:
            return style.standardIcon(qt.QStyle.SP_DirLinkIcon)
        elif class_ == silx.io.utils.H5Type.EXTERNAL_LINK:
            return style.standardIcon(qt.QStyle.SP_FileLinkIcon)
        elif class_ == silx.io.utils.H5Type.DATASET:
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

        if self.h5Class == silx.io.utils.H5Type.DATASET:
            attributeDict["#Title"] = "HDF5 Dataset"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
            attributeDict["Shape"] = self._getFormatter().humanReadableShape(self.obj)
            attributeDict["Value"] = self._getFormatter().humanReadableValue(self.obj)
            attributeDict["Data type"] = self._getFormatter().humanReadableType(self.obj, full=True)
        elif self.h5Class == silx.io.utils.H5Type.GROUP:
            attributeDict["#Title"] = "HDF5 Group"
            if self.nexusClassName:
                attributeDict["NX_class"] = self.nexusClassName
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
        elif self.h5Class == silx.io.utils.H5Type.FILE:
            attributeDict["#Title"] = "HDF5 File"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = "/"
        elif self.h5Class == silx.io.utils.H5Type.EXTERNAL_LINK:
            attributeDict["#Title"] = "HDF5 External Link"
            attributeDict["Name"] = self.basename
            attributeDict["Path"] = self.obj.name
            attributeDict["Linked path"] = self.obj.path
            attributeDict["Linked file"] = self.obj.filename
        elif self.h5Class == silx.io.utils.H5Type.SOFT_LINK:
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

    @property
    def nexusClassName(self):
        """Returns the Nexus class name"""
        if self.__nx_class is None:
            obj = self.obj.attrs.get("NX_class", None)
            if obj is None:
                text = ""
            else:
                text = self._getFormatter().textFormatter().toString(obj)
                text = text.strip('"')
                # Check NX_class formatting
                lower = text.lower()
                if lower.startswith('nx'):
                    formatedNX_class = 'NX' + lower[2:]
                if lower == 'nxcansas':
                    formatedNX_class = 'NXcanSAS'  # That's the only class with capital letters...
                if text != formatedNX_class:
                    _logger.error("NX_class: %s is malformed (should be %s)",
                                  text,
                                  formatedNX_class)
                text = formatedNX_class

            self.__nx_class = text
        return self.__nx_class

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
            class_ = self.h5Class
            if self.isGroupObj():
                text = self.nexusClassName
            elif class_ == silx.io.utils.H5Type.DATASET:
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
            class_ = self.h5Class
            if class_ != silx.io.utils.H5Type.DATASET:
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
            if self.h5Class != silx.io.utils.H5Type.DATASET:
                return ""
            return self._getFormatter().humanReadableValue(self.obj)
        return None

    _NEXUS_CLASS_TO_VALUE_CHILDREN = {
        'NXaperture': (
            (DescriptionType.DESCRIPTION, 'description'),
        ),
        'NXbeam_stop': (
            (DescriptionType.DESCRIPTION, 'description'),
        ),
        'NXdetector': (
            (DescriptionType.NAME, 'local_name'),
            (DescriptionType.DESCRIPTION, 'description')
        ),
        'NXentry': (
            (DescriptionType.TITLE, 'title'),
        ),
        'NXenvironment': (
            (DescriptionType.NAME, 'short_name'),
            (DescriptionType.NAME, 'name'),
            (DescriptionType.DESCRIPTION, 'description')
        ),
        'NXinstrument': (
            (DescriptionType.NAME, 'name'),
        ),
        'NXlog': (
            (DescriptionType.DESCRIPTION, 'description'),
        ),
        'NXmirror': (
            (DescriptionType.DESCRIPTION, 'description'),
        ),
        'NXpositioner': (
            (DescriptionType.NAME, 'name'),
        ),
        'NXprocess': (
            (DescriptionType.PROGRAM, 'program'),
        ),
        'NXsample': (
            (DescriptionType.TITLE, 'short_title'),
            (DescriptionType.NAME, 'name'),
            (DescriptionType.DESCRIPTION, 'description')
        ),
        'NXsample_component': (
            (DescriptionType.NAME, 'name'),
            (DescriptionType.DESCRIPTION, 'description')
        ),
        'NXsensor': (
            (DescriptionType.NAME, 'short_name'),
            (DescriptionType.NAME, 'name')
        ),
        'NXsource': (
            (DescriptionType.NAME, 'name'),
        ),  # or its 'short_name' attribute... This is not supported
        'NXsubentry': (
            (DescriptionType.DESCRIPTION, 'definition'),
            (DescriptionType.PROGRAM, 'program_name'),
            (DescriptionType.TITLE, 'title'),
        ),
    }
    """Mapping from NeXus class to child names containing data to use as value"""

    def __computeDataDescription(self):
        """Compute the data description of this item

        :rtype: Tuple[kind, str]
        """
        if self.__isBroken or self.__error is not None:
            self.obj  # lazy loading of the object
            return DescriptionType.ERROR, self.__error

        if self.h5Class == silx.io.utils.H5Type.DATASET:
            return DescriptionType.VALUE, self._getFormatter().humanReadableValue(self.obj)

        elif self.isGroupObj() and self.nexusClassName:
            # For NeXus groups, try to find a title or name
            # By default, look for a title (most application definitions should have one)
            defaultSequence = ((DescriptionType.TITLE, 'title'),)
            sequence = self._NEXUS_CLASS_TO_VALUE_CHILDREN.get(self.nexusClassName, defaultSequence)
            for kind, child_name in sequence:
                for index in range(self.childCount()):
                    child = self.child(index)
                    if (isinstance(child, Hdf5Item) and
                            child.h5Class == silx.io.utils.H5Type.DATASET and
                            child.basename == child_name):
                        return kind, self._getFormatter().humanReadableValue(child.obj)

        description = self.obj.attrs.get("desc", None)
        if description is not None:
            return DescriptionType.DESCRIPTION, description
        else:
            return None, None

    def __getDataDescription(self):
        """Returns a cached version of the data description

        As the data description have to reach inside the HDF5 tree, the result
        is cached. A better implementation could be to use a MRU cache, to avoid
        to allocate too much data.

        :rtype: Tuple[kind, str]
        """
        if self.__description is None:
            self.__description = self.__computeDataDescription()
        return self.__description

    def dataDescription(self, role):
        """Data for the description column"""
        if role == qt.Qt.DecorationRole:
            kind, _label = self.__getDataDescription()
            if kind is not None:
                icon = icons.getQIcon("description-%s" % kind.value)
                return icon
            return None
        if role == qt.Qt.TextAlignmentRole:
            return qt.Qt.AlignTop | qt.Qt.AlignLeft
        if role == qt.Qt.DisplayRole:
            _kind, label = self.__getDataDescription()
            return label
        if role == qt.Qt.ToolTipRole:
            if self.__error is not None:
                self.obj  # lazy loading of the object
                self.__initH5Object()
                return self.__error
            kind, label = self.__getDataDescription()
            if label is not None:
                return "<b>%s</b><br/>%s" % (kind.value.capitalize(), label)
            else:
                return ""
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
            class_ = self.obj.__class__
            text = class_.__name__.split(".")[-1]
            return text
        if role == qt.Qt.ToolTipRole:
            class_ = self.obj.__class__
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
            elif link == silx.io.utils.H5Type.EXTERNAL_LINK:
                return "External"
            elif link == silx.io.utils.H5Type.SOFT_LINK:
                return "Soft"
            elif link == silx.io.utils.H5Type.HARD_LINK:
                return ""
            else:
                return link.__name__
        if role == qt.Qt.ToolTipRole:
            return None
        return None
