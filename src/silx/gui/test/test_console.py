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
"""Basic tests for IPython console widget"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"


import sys

import pytest
from silx.gui import qt

# ipykernel 7 warns about its own synchronous IPythonKernel.do_history (called when
# qtconsole requests history at startup), do_complete, do_inspect and do_shutdown
pytestmark = pytest.mark.filterwarnings(
    "ignore:For consistency across implementations:PendingDeprecationWarning"
)


# dummy objects to test pushing variables to the interactive namespace
_a = 1


def _f():
    print("Hello World!")


@pytest.fixture
def console(qapp_utils):
    """Create a console widget"""
    try:
        from silx.gui.console import IPythonDockWidget
    except ImportError:
        pytest.skip("IPythonDockWidget is not available")

    console = IPythonDockWidget(
        available_vars={"a": _a, "f": _f}, custom_banner="Welcome!\n"
    )
    console.show()
    qapp_utils.qWaitForWindowExposed(console)
    yield console
    console.ipyconsole.shutdownKernel()
    console.setAttribute(qt.Qt.WA_DeleteOnClose)
    console.close()
    console = None


def testShow(console):
    pass


def testInteract(console, qapp_utils):
    qapp_utils.mouseClick(console, qt.Qt.LeftButton)
    qapp_utils.keyClicks(console, "import silx")
    qapp_utils.keyClick(console, qt.Qt.Key_Enter)
    qapp_utils.qapp.processEvents()


def test_host_interpreter_unchanged(qWidgetFactory):
    """Creating an IPythonWidget must not alter the host interpreter (#538)"""
    try:
        from silx.gui.console import IPythonWidget
    except ImportError:
        pytest.skip("IPythonWidget is not available")

    main_module = sys.modules["__main__"]
    prompts = {name: getattr(sys, name, None) for name in ("ps1", "ps2", "ps3")}

    widget = qWidgetFactory(IPythonWidget)
    assert sys.modules["__main__"] is main_module
    assert {name: getattr(sys, name, None) for name in ("ps1", "ps2", "ps3")} == prompts
    widget.shutdownKernel()
