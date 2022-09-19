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
"""This module provides a simple generic notification system."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/07/2018"


import logging

from silx.utils.weakref import WeakList

_logger = logging.getLogger(__name__)


# Notifier ####################################################################

class Notifier(object):
    """Base class for object with notification mechanism."""

    def __init__(self):
        self._listeners = WeakList()

    def addListener(self, listener):
        """Register a listener.

        Adding an already registered listener has no effect.

        :param callable listener: The function or method to register.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)
        else:
            _logger.warning('Ignoring addition of an already registered listener')

    def removeListener(self, listener):
        """Remove a previously registered listener.

        :param callable listener: The function or method to unregister.
        """
        try:
            self._listeners.remove(listener)
        except ValueError:
            _logger.warning('Trying to remove a listener that is not registered')

    def notify(self, *args, **kwargs):
        """Notify all registered listeners with the given parameters.

        Listeners are called directly in this method.
        Listeners are called in the order they were registered.
        """
        for listener in self._listeners:
            listener(self, *args, **kwargs)


def notifyProperty(attrName, copy=False, converter=None, doc=None):
    """Create a property that adds notification to an attribute.

    :param str attrName: The name of the attribute to wrap.
    :param bool copy: Whether to return a copy of the attribute
                      or not (the default).
    :param converter: Function converting input value to appropriate type
                      This function takes a single argument and return the
                      converted value.
                      It can be used to perform some asserts.
    :param str doc: The docstring of the property
    :return: A property with getter and setter
    """
    if copy:
        def getter(self):
            return getattr(self, attrName).copy()
    else:
        def getter(self):
            return getattr(self, attrName)

    if converter is None:
        def setter(self, value):
            if getattr(self, attrName) != value:
                setattr(self, attrName, value)
                self.notify()

    else:
        def setter(self, value):
            value = converter(value)
            if getattr(self, attrName) != value:
                setattr(self, attrName, value)
                self.notify()

    return property(getter, setter, doc=doc)


class HookList(list):
    """List with hooks before and after modification."""

    def __init__(self, iterable):
        super(HookList, self).__init__(iterable)

        self._listWasChangedHook('__init__', iterable)

    def _listWillChangeHook(self, methodName, *args, **kwargs):
        """To override. Called before modifying the list.

        This method is called with the name of the method called to
        modify the list and its parameters.
        """
        pass

    def _listWasChangedHook(self, methodName, *args, **kwargs):
        """To override. Called after modifying the list.

        This method is called with the name of the method called to
        modify the list and its parameters.
        """
        pass

    # Wrapping methods that modify the list

    def _wrapper(self, methodName, *args, **kwargs):
        """Generic wrapper of list methods calling the hooks."""
        self._listWillChangeHook(methodName, *args, **kwargs)
        result = getattr(super(HookList, self),
                         methodName)(*args, **kwargs)
        self._listWasChangedHook(methodName, *args, **kwargs)
        return result

    # Add methods

    def __iadd__(self, *args, **kwargs):
        return self._wrapper('__iadd__', *args, **kwargs)

    def __imul__(self, *args, **kwargs):
        return self._wrapper('__imul__', *args, **kwargs)

    def append(self, *args, **kwargs):
        return self._wrapper('append', *args, **kwargs)

    def extend(self, *args, **kwargs):
        return self._wrapper('extend', *args, **kwargs)

    def insert(self, *args, **kwargs):
        return self._wrapper('insert', *args, **kwargs)

    # Remove methods

    def __delitem__(self, *args, **kwargs):
        return self._wrapper('__delitem__', *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        return self._wrapper('__delslice__', *args, **kwargs)

    def remove(self, *args, **kwargs):
        return self._wrapper('remove', *args, **kwargs)

    def pop(self, *args, **kwargs):
        return self._wrapper('pop', *args, **kwargs)

    # Set methods

    def __setitem__(self, *args, **kwargs):
        return self._wrapper('__setitem__', *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        return self._wrapper('__setslice__', *args, **kwargs)

    # In place methods

    def sort(self, *args, **kwargs):
        return self._wrapper('sort', *args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self._wrapper('reverse', *args, **kwargs)


class NotifierList(HookList, Notifier):
    """List of Notifiers with notification mechanism.

    This class registers itself as a listener of the list items.

    The default listener method forward notification from list items
    to the listeners of the list.
    """

    def __init__(self, iterable=()):
        Notifier.__init__(self)
        HookList.__init__(self, iterable)

    def _listWillChangeHook(self, methodName, *args, **kwargs):
        for item in self:
            item.removeListener(self._notified)

    def _listWasChangedHook(self, methodName, *args, **kwargs):
        for item in self:
            item.addListener(self._notified)
        self.notify()

    def _notified(self, source, *args, **kwargs):
        """Default listener forwarding list item changes to its listeners."""
        # Avoid infinite recursion if the list is listening itself
        if source is not self:
            self.notify(*args, **kwargs)
