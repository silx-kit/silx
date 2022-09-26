
# Taken from: https://gist.github.com/cpbotha/1b42a20c8f3eb9bb7cb8
# Plus: https://github.com/spyder-ide/qtpy/commit/001a862c401d757feb63025f88dbb4601d353c84

# Copyright (c) 2011 Sebastian Wiesner <lunaryorn@gmail.com>
# Modifications by Charl Botha <cpbotha@vxlabs.com>
# * customWidgets support (registerCustomWidget() causes segfault in
#   pyside 1.1.2 on Ubuntu 12.04 x86_64)
# * workingDirectory support in loadUi

# found this here:
# https://github.com/lunaryorn/snippets/blob/master/qt4/designer/pyside_dynamic.py

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
    How to load a user interface dynamically with PySide.

    .. moduleauthor::  Sebastian Wiesner  <lunaryorn@gmail.com>
"""

import logging

from ._qt import BINDING
if BINDING == 'PySide2':
    from PySide2.QtCore import QMetaObject, Property, Qt
    from PySide2.QtWidgets import QFrame
    from PySide2.QtUiTools import QUiLoader
elif BINDING == 'PySide6':
    from PySide6.QtCore import QMetaObject, Property, Qt
    from PySide6.QtWidgets import QFrame
    from PySide6.QtUiTools import QUiLoader
else:
    raise RuntimeError("Unsupported Qt binding: %s", BINDING)


_logger = logging.getLogger(__name__)


class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.

    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.

    This mimics the behaviour of :func:`PyQt*.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.

        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.

        ``customWidgets`` is a dictionary mapping from class name to class
        object for widgets that you've promoted in the Qt Designer
        interface. Usually, this should be done by calling
        registerCustomWidget on the QUiLoader, but
        with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = {}
        self.uifile = None
        self.customWidgets.update(customWidgets)

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets,
                # must be a custom widget
                # this will raise KeyError if the user has not supplied the
                # relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                if class_name not in self.customWidgets:
                    raise Exception('No custom widget ' + class_name +
                                    ' found in customWidgets param of' +
                                    'UiFile %s.' % self.uifile)
                try:
                    widget = self.customWidgets[class_name](parent)
                except Exception:
                    _logger.error("Fail to instanciate widget %s from file %s", class_name, self.uifile)
                    raise

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt*.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.
                # print(name)

            return widget

    def _parse_custom_widgets(self, ui_file):
        """
        This function is used to parse a ui file and look for the <customwidgets>
        section, then automatically load all the custom widget classes.
        """
        import importlib
        from xml.etree.ElementTree import ElementTree

        # Parse the UI file
        etree = ElementTree()
        ui = etree.parse(ui_file)

        # Get the customwidgets section
        custom_widgets = ui.find('customwidgets')

        if custom_widgets is None:
            return

        custom_widget_classes = {}

        for custom_widget in custom_widgets.getchildren():

            cw_class = custom_widget.find('class').text
            cw_header = custom_widget.find('header').text

            module = importlib.import_module(cw_header)

            custom_widget_classes[cw_class] = getattr(module, cw_class)

        self.customWidgets.update(custom_widget_classes)

    def load(self, uifile):
        self._parse_custom_widgets(uifile)
        self.uifile = uifile
        return QUiLoader.load(self, uifile)


class _Line(QFrame):
    """Widget to use as 'Line' Qt designer"""
    def __init__(self, parent=None):
        super(_Line, self).__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

    def getOrientation(self):
        shape = self.frameShape()
        if shape == QFrame.HLine:
            return Qt.Horizontal
        elif shape == QFrame.VLine:
            return Qt.Vertical
        else:
            raise RuntimeError("Wrong shape: %d", shape)

    def setOrientation(self, orientation):
        if orientation == Qt.Horizontal:
            self.setFrameShape(QFrame.HLine)
        elif orientation == Qt.Vertical:
            self.setFrameShape(QFrame.VLine)
        else:
            raise ValueError("Unsupported orientation %s" % str(orientation))

    orientation = Property("Qt::Orientation", getOrientation, setOrientation)


CUSTOM_WIDGETS = {"Line": _Line}
"""Default custom widgets for `loadUi`"""


def loadUi(uifile, baseinstance=None, package=None, resource_suffix=None):
    """
    Dynamically load a user interface from the given ``uifile``.

    ``uifile`` is a string containing a file name of the UI file to load.

    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.

    Return ``baseinstance``, if ``baseinstance`` is not ``None``. Otherwise
    return the newly created instance of the user interface.
    """
    if package is not None:
        _logger.warning(
            "loadUi package parameter not implemented with PySide")
    if resource_suffix is not None:
        _logger.warning(
            "loadUi resource_suffix parameter not implemented with PySide")

    loader = UiLoader(baseinstance, customWidgets=CUSTOM_WIDGETS)
    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget
