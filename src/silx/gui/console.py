# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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
"""This module provides an IPython console widget.

You can push variables - any python object - to the
console's interactive namespace. This provides users with an advanced way
of interacting with your program. For instance, if your program has a
:class:`PlotWidget` or a :class:`PlotWindow`, you can push a reference to
these widgets to allow your users to add curves, save data to files… by using
the widgets' methods from the console.

.. note::

    This module has a dependency on
    `qtconsole <https://pypi.org/project/qtconsole/>`_.
    An ``ImportError`` will be raised if it is
    imported while the dependencies are not satisfied.

Basic usage example::

    from silx.gui import qt
    from silx.gui.console import IPythonWidget

    app = qt.QApplication([])

    hello_button = qt.QPushButton("Hello World!", None)
    hello_button.show()

    console = IPythonWidget()
    console.show()
    console.pushVariables({"the_button": hello_button})

    app.exec()

This program will display a console widget and a push button in two separate
windows. You will be able to interact with the button from the console,
for example change its text::

    >>> the_button.setText("Spam spam")

An IPython interactive console is a powerful tool that enables you to work
with data and plot it.
See `this tutorial <https://plot.ly/python/ipython-notebook-tutorial/>`_
for more information on some of the rich features of IPython.
"""
__authors__ = ["Tim Rae", "V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/05/2016"

import logging

from . import qt

_logger = logging.getLogger(__name__)


# This widget cannot be used inside an interactive IPython shell.
# It would raise MultipleInstanceError("Multiple incompatible subclass
# instances of InProcessInteractiveShell are being created").
try:
    __IPYTHON__
except NameError:
    pass  # Not in IPython
else:
    msg = "Module " + __name__ + " cannot be used within an IPython shell"
    raise ImportError(msg)

try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as \
        _RichJupyterWidget
except ImportError:
    try:
        from qtconsole.rich_ipython_widget import RichJupyterWidget as \
            _RichJupyterWidget
    except ImportError:
        from qtconsole.rich_ipython_widget import RichIPythonWidget as \
            _RichJupyterWidget

from qtconsole.inprocess import QtInProcessKernelManager

try:
    from ipykernel import version_info as _ipykernel_version_info
except ImportError:
    _ipykernel_version_info = None


class IPythonWidget(_RichJupyterWidget):
    """Live IPython console widget.

    .. image:: img/IPythonWidget.png

    :param custom_banner: Custom welcome message to be printed at the top of
       the console.
    """

    def __init__(self, parent=None, custom_banner=None, *args, **kwargs):
        if parent is not None:
            kwargs["parent"] = parent
        super(IPythonWidget, self).__init__(*args, **kwargs)
        if custom_banner is not None:
            self.banner = custom_banner
        self.setWindowTitle(self.banner)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()

        # Monkey-patch to workaround issue:
        # https://github.com/ipython/ipykernel/issues/370
        if (_ipykernel_version_info is not None and
                _ipykernel_version_info[0] > 4 and
                _ipykernel_version_info[:3] <= (5, 1, 0)):
            def _abort_queues(*args, **kwargs):
                pass
            kernel_manager.kernel._abort_queues = _abort_queues

        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
        self.exit_requested.connect(stop)

    def sizeHint(self):
        """Return a reasonable default size for usage in :class:`PlotWindow`"""
        return qt.QSize(500, 300)

    def pushVariables(self, variable_dict):
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget.

        :param variable_dict: Dictionary of variables to be pushed to the
            console's interactive namespace (```{variable_name: object, …}```)
        """
        self.kernel_manager.kernel.shell.push(variable_dict)


class IPythonDockWidget(qt.QDockWidget):
    """Dock Widget including a :class:`IPythonWidget` inside
    a vertical layout.

    .. image:: img/IPythonDockWidget.png

    :param available_vars: Dictionary of variables to be pushed to the
        console's interactive namespace: ``{"variable_name": object, …}``
    :param custom_banner: Custom welcome message to be printed at the top of
        the console
    :param title: Dock widget title
    :param parent: Parent :class:`qt.QMainWindow` containing this
        :class:`qt.QDockWidget`
    """
    def __init__(self, parent=None, available_vars=None, custom_banner=None,
                 title="Console"):
        super(IPythonDockWidget, self).__init__(title, parent)

        self.ipyconsole = IPythonWidget(custom_banner=custom_banner)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(self.ipyconsole)

        if available_vars is not None:
            self.ipyconsole.pushVariables(available_vars)


def main():
    """Run a Qt app with an IPython console"""
    app = qt.QApplication([])
    widget = IPythonDockWidget()
    widget.show()
    app.exec()


if __name__ == '__main__':
    main()
