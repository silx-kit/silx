# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides the base scene structure.

This module provides the classes for describing a tree structure with
rendering and picking API.
All nodes inherit from :class:`Base`.
Nodes with children are provided with :class:`PrivateGroup` and
:class:`Group` classes.
Leaf rendering nodes should inherit from :class:`Elem`.
"""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import itertools
import weakref

import numpy

from . import event
from . import transform

from .viewport import Viewport


# Nodes #######################################################################

class Base(event.Notifier):
    """A scene node with common features."""

    def __init__(self):
        super(Base, self).__init__()
        self._visible = True
        self._pickable = False

        self._parentRef = None

        self._transforms = transform.TransformList()
        self._transforms.addListener(self._transformChanged)

    # notifying properties

    visible = event.notifyProperty('_visible',
                                   doc="Visibility flag of the node")
    pickable = event.notifyProperty('_pickable',
                                    doc="True to make node pickable")

    # Access to tree path

    @property
    def parent(self):
        """Parent or None if no parent"""
        return None if self._parentRef is None else self._parentRef()

    def _setParent(self, parent):
        """Set the parent of this node.

        For internal use.

        :param Base parent: The parent.
        """
        if parent is not None and self._parentRef is not None:
            raise RuntimeError('Trying to add a node at two places.')
            # Alternative: remove it from previous children list
        self._parentRef = None if parent is None else weakref.ref(parent)

    @property
    def path(self):
        """Tuple of scene nodes, from the tip of the tree down to this node.

        If this tree is attached to a :class:`Viewport`,
        then the :class:`Viewport` is the first element of path.
        """
        if self.parent is None:
            return self,
        elif isinstance(self.parent, Viewport):
            return self.parent, self
        else:
            return self.parent.path + (self, )

    @property
    def viewport(self):
        """The viewport this node is attached to or None."""
        root = self.path[0]
        return root if isinstance(root, Viewport) else None

    @property
    def objectToNDCTransform(self):
        """Transform from object to normalized device coordinates.

        Do not forget perspective divide.
        """
        # Using the Viewport's transforms property to proxy the camera
        path = self.path
        assert isinstance(path[0], Viewport)
        return transform.StaticTransformList(elem.transforms for elem in path)

    @property
    def objectToSceneTransform(self):
        """Transform from object to scene.

        Combine transforms up to the Viewport (not including it).
        """
        path = self.path
        if isinstance(path[0], Viewport):
            path = path[1:]  # Remove viewport to remove camera transforms
        return transform.StaticTransformList(elem.transforms for elem in path)

    # transform

    @property
    def transforms(self):
        """List of transforms defining the frame of this node relative
        to its parent."""
        return self._transforms

    @transforms.setter
    def transforms(self, iterable):
        self._transforms.removeListener(self._transformChanged)
        if isinstance(iterable, transform.TransformList):
            # If it is a TransformList, do not create one to enable sharing.
            self._transforms = iterable
        else:
            assert hasattr(iterable, '__iter__')
            self._transforms = transform.TransformList(iterable)
        self._transforms.addListener(self._transformChanged)

    def _transformChanged(self, source):
        self.notify()  # Broadcast transform notification

    # Bounds

    _CUBE_CORNERS = numpy.array(list(itertools.product((0., 1.), repeat=3)),
                                dtype=numpy.float32)
    """Unit cube corners used to transform bounds"""

    def _bounds(self, dataBounds=False):
        """Override in subclass to provide bounds in object coordinates"""
        return None

    def bounds(self, transformed=False, dataBounds=False):
        """Returns the bounds of this node aligned with the axis,
        with or without transform applied.

        :param bool transformed: False to give bounds in object coordinates
                                 (the default), True to apply this object's
                                 transforms.
        :param bool dataBounds: False to give bounds of vertices (the default),
                                True to give bounds of the represented data.
        :return: The bounds: ((xMin, yMin, zMin), (xMax, yMax, zMax)) or None
                 if no bounds.
        :rtype: numpy.ndarray of float
        """
        bounds = self._bounds(dataBounds)

        if transformed and bounds is not None:
            bounds = self.transforms.transformBounds(bounds)

        return bounds

    # Rendering

    def prepareGL2(self, ctx):
        """Called before the rendering to prepare OpenGL resources.

        Override in subclass.
        """
        pass

    def renderGL2(self, ctx):
        """Called to perform the OpenGL rendering.

        Override in subclass.
        """
        pass

    def render(self, ctx):
        """Called internally to perform rendering."""
        if self.visible:
            ctx.pushTransform(self.transforms)
            self.prepareGL2(ctx)
            self.renderGL2(ctx)
            ctx.popTransform()

    def postRender(self, ctx):
        """Hook called when parent's node render is finished.

        Called in the reverse of rendering order (i.e., last child first).

        Meant for nodes that modify the :class:`RenderContext` ctx to
        reset their modifications.
        """
        pass

    def pick(self, ctx, x, y, depth=None):
        """True/False picking, should be fast"""
        if self.pickable:
            pass

    def pickRay(self, ctx, ray):
        """Picking returning list of ray intersections."""
        if self.pickable:
            pass


class Elem(Base):
    """A scene node that does some rendering."""

    def __init__(self):
        super(Elem, self).__init__()
        # self.showBBox = False  # Here or outside scene graph?
        # self.clipPlane = None  # This needs to be handled in the shader


class PrivateGroup(Base):
    """A scene node that renders its (private) childern.

    :param iterable children: :class:`Base` nodes to add as children
    """

    class ChildrenList(event.NotifierList):
        """List of children with notification and children's parent update."""

        def _listWillChangeHook(self, methodName, *args, **kwargs):
            super(PrivateGroup.ChildrenList, self)._listWillChangeHook(
                methodName, *args, **kwargs)
            for item in self:
                item._setParent(None)

        def _listWasChangedHook(self, methodName, *args, **kwargs):
            for item in self:
                item._setParent(self._parentRef())
            super(PrivateGroup.ChildrenList, self)._listWasChangedHook(
                methodName, *args, **kwargs)

        def __init__(self, parent, children):
            self._parentRef = weakref.ref(parent)
            super(PrivateGroup.ChildrenList, self).__init__(children)

    def __init__(self, children=()):
        super(PrivateGroup, self).__init__()
        self.__children = PrivateGroup.ChildrenList(self, children)
        self.__children.addListener(self._updated)

    @property
    def _children(self):
        """List of children to be rendered.

        This private attribute is meant to be used by subclass.
        """
        return self.__children

    @_children.setter
    def _children(self, iterable):
        self.__children.removeListener(self._updated)
        for item in self.__children:
            item._setParent(None)
        del self.__children  # This is needed
        self.__children = PrivateGroup.ChildrenList(self, iterable)
        self.__children.addListener(self._updated)
        self.notify()

    def _updated(self, source, *args, **kwargs):
        """Listen for updates"""
        if source is not self:  # Avoid infinite recursion
            self.notify(*args, **kwargs)

    def _bounds(self, dataBounds=False):
        """Compute the bounds from transformed children bounds"""
        bounds = []
        for child in self._children:
            if child.visible:
                childBounds = child.bounds(
                    transformed=True, dataBounds=dataBounds)
                if childBounds is not None:
                    bounds.append(childBounds)

        if len(bounds) == 0:
            return None
        else:
            bounds = numpy.array(bounds, dtype=numpy.float32)
            return numpy.array((bounds[:, 0, :].min(axis=0),
                                bounds[:, 1, :].max(axis=0)),
                               dtype=numpy.float32)

    def prepareGL2(self, ctx):
        pass

    def renderGL2(self, ctx):
        """Render all children"""
        for child in self._children:
            child.render(ctx)
        for child in reversed(self._children):
            child.postRender(ctx)


class Group(PrivateGroup):
    """A scene node that renders its (public) children."""

    @property
    def children(self):
        """List of children to be rendered."""
        return self._children

    @children.setter
    def children(self, iterable):
        self._children = iterable
