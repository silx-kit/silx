# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Widgets offering an IPython console.

This widget is meant to work with :class:`PlotWindow`. The console keeps a
reference to the :class:`PlotWindow` to allow interacting with it (adding
curves, saving data, ...)
"""
__authors__ = ["Tim Rae", "V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "28/04/2016"

import logging
import weakref

from .. import qt

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import IPython

# TODO: assess dependencies (pip install ipython, qtconsole)
# TODO: helpful error message in case ipython or qtconsole are not available

# qtconsole is a separate module in recent versions of IPython/Jupyter
# http://blog.jupyter.org/2015/04/15/the-big-split/
if IPython.__version__.startswith("2"):
    QTCONSOLE = False
else:
    try:
        import qtconsole
        QTCONSOLE = True
    except ImportError:
        QTCONSOLE = False


if QTCONSOLE:
    try:
        from qtconsole.rich_ipython_widget import RichJupyterWidget as RichIPythonWidget
    except:
        from qtconsole.rich_ipython_widget import RichIPythonWidget
    from qtconsole.inprocess import QtInProcessKernelManager
else:
    # Import the console machinery from ipython

    # TODO: check if this is necessary for silx
    # # Check if we using a frozen version because
    # # the test of IPython does not find the Qt bindings
    # executables = ["PyMcaMain.exe", "QStackWidget.exe", "PyMcaPostBatch.exe"]
    # if os.path.basename(sys.executable) in executables:
    #     import IPython.external.qt_loaders
    #     def has_binding(*var, **kw):
    #         return True
    #     IPython.external.qt_loaders.has_binding = has_binding 

    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager

from IPython.lib import guisupport

try:
    import sip
    sip.setapi("QString", 2)
    sip.setapi("QVariant", 2)
    _logger.debug("sip imported")
except:
    _logger.debug("sip not imported")
    pass

class IPythonWidget(RichIPythonWidget):
    """ Convenience class for a live IPython console widget.
    We can replace the standard banner using the customBanner argument"""

    def __init__(self,customBanner=None,*args,**kwargs):
        super(IPythonWidget, self).__init__(*args, **kwargs)
        if customBanner != None:
            self.banner = customBanner
        self.setWindowTitle(self.banner)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt4'  # TODO: should "qt4" be hardcoded?
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        """ Given a dictionary containing name / value pairs, push those
        variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)

    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()

    # Does not seem to work
    # def printText(self, text):
    #     """ Prints some plain text to the console """
    #     self._append_plain_text(text)

    def executeCommand(self, command):
        """ Execute a command in the frame of the console widget """
        self._execute(command, False)


class IPythonDockWidget(qt.QDockWidget):
    """Dock Widget including an IPython Console widget inside
    a vertical layout """
    def __init__(self, plot, parent=None):
        assert plot is not None
        # self._plotRef = weakref.ref(plot) # TODO: is it OK to give a hard reference to plot?

        super(IPythonDockWidget, self).__init__(parent)

        banner = "Welcome to the embedded ipython console\n"
        banner += "The variable 'plt' is available to interact with the plot."
        banner += " Type 'whos' and 'help(plt)' for more information.\n\n"

        ipyConsole = IPythonWidget(customBanner=banner)
        ipyConsole.clearTerminal()

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(ipyConsole)

        ipyConsole.pushVariables({"plt": plot})


def main():
    class MyFakePlot(object):
        msg = "Dummy plt var to test pushing variables to tho console"
        def print_msg(self):
            print(self.msg)
    app = qt.QApplication([])
    plt = MyFakePlot()
    widget = IPythonDockWidget(plot=plt)
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()
