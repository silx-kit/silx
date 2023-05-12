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
__date__ = "24/07/2018"

import weakref
from typing import Optional


class Hdf5Node(object):
    """Abstract tree node

    It provides link to the childs and to the parents, and a link to an
    external object.
    """
    def __init__(
        self,
        parent=None,
        populateAll=False,
        openedPath: Optional[str]=None,
    ):
        """
        Constructor

        :param Hdf5Node parent: Parent of the node, if exists, else None
        :param bool populateAll: If true, populate all the tree node. Else
            everything is lazy loaded.
        :param openedPath:
            The url or filename the node was created from, None if not directly created
        """
        self.__child = None
        self.__parent = None
        self.__openedPath = openedPath
        if parent is not None:
            self.__parent = weakref.ref(parent)
        if populateAll:
            self.__child = []
            self._populateChild(populateAll=True)

    def _getCanonicalName(self):
        parent = self.parent
        if parent is None:
            return "root"
        else:
            return "%s/?" % (parent._getCanonicalName())

    @property
    def _openedPath(self) -> Optional[str]:
        """url or filename the node was created from, None if not directly created"""
        return self.__openedPath

    @property
    def parent(self):
        """Parent of the node, or None if the node is a root

        :rtype: Hdf5Node
        """
        if self.__parent is None:
            return None
        parent = self.__parent()
        if parent is None:
            self.__parent = parent
        return parent

    def setParent(self, parent):
        """Redefine the parent of the node.

        It does not set the node as the children of the new parent.

        :param Hdf5Node parent: The new parent
        """
        if parent is None:
            self.__parent = None
        else:
            self.__parent = weakref.ref(parent)

    def appendChild(self, child):
        """Append a child to the node.

        It does not update the parent of the child.

        :param Hdf5Node child: Child to append to the node.
        """
        self.__initChild()
        self.__child.append(child)

    def removeChildAtIndex(self, index):
        """Remove a child at an index of the children list.

        The child is removed and returned.

        :param int index: Index in the child list.
        :rtype: Hdf5Node
        :raises: IndexError if list is empty or index is out of range.
        """
        self.__initChild()
        return self.__child.pop(index)

    def insertChild(self, index, child):
        """
        Insert a child at a specific index of the child list.

        It does not update the parent of the child.

        :param int index: Index in the child list.
        :param Hdf5Node child: Child to insert in the child list.
        """
        self.__initChild()
        self.__child.insert(index, child)

    def indexOfChild(self, child):
        """
        Returns the index of the child in the child list of this node.

        :param Hdf5Node child: Child to find
        :raises: ValueError if the value is not present.
        """
        self.__initChild()
        return self.__child.index(child)

    def hasChildren(self):
        """Returns true if the node contains children.

        :rtype: bool
        """
        return self.childCount() > 0

    def childCount(self):
        """Returns the number of child in this node.

        :rtype: int
        """
        if self.__child is not None:
            return len(self.__child)
        return self._expectedChildCount()

    def child(self, index):
        """Return the child at an expected index.

        :param int index: Index of the child in the child list of the node
        :rtype: Hdf5Node
        """
        self.__initChild()
        return self.__child[index]

    def __initChild(self):
        """Init the child of the node in case the list was lazy loaded."""
        if self.__child is None:
            self.__child = []
            self._populateChild()

    def _expectedChildCount(self):
        """Returns the expected count of children

        :rtype: int
        """
        return 0

    def _populateChild(self, populateAll=False):
        """Recurse through an HDF5 structure to append groups an datasets
        into the tree model.

        Overwrite it to implement the initialisation of child of the node.
        """
        pass

    def dataName(self, role):
        """Data for the name column

        Overwrite it to implement the content of the 'name' column.

        :rtype: qt.QVariant
        """
        return None

    def dataType(self, role):
        """Data for the type column

        Overwrite it to implement the content of the 'type' column.

        :rtype: qt.QVariant
        """
        return None

    def dataShape(self, role):
        """Data for the shape column

        Overwrite it to implement the content of the 'shape' column.

        :rtype: qt.QVariant
        """
        return None

    def dataValue(self, role):
        """Data for the value column

        Overwrite it to implement the content of the 'value' column.

        :rtype: qt.QVariant
        """
        return None

    def dataDescription(self, role):
        """Data for the description column

        Overwrite it to implement the content of the 'description' column.

        :rtype: qt.QVariant
        """
        return None

    def dataNode(self, role):
        """Data for the node column

        Overwrite it to implement the content of the 'node' column.

        :rtype: qt.QVariant
        """
        return None

    def dataLink(self, role):
        """Data for the link column

        Overwrite it to implement the content of the 'link' column.

        :rtype: qt.QVariant
        """
        return None
