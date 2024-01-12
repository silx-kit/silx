import pytest

from silx.gui import qt
from silx.gui.qt.inspect import isValid


@pytest.fixture(autouse=True)
def auto_qapp(qapp):
    pass


@pytest.fixture
def qWidgetFactory(qapp, qapp_utils):
    """QWidget factory as fixture

    This fixture provides a function taking a QWidget subclass as argument
    which returns an instance of this QWidget making sure it is shown first
    and destroyed once the test is done.
    """
    widgets = []

    def createWidget(cls, *args, **kwargs):
        widget = cls(*args, **kwargs)
        widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        widget.show()
        qapp_utils.qWaitForWindowExposed(widget)
        widgets.append(widget)

        return widget

    yield createWidget

    qapp.processEvents()

    for widget in widgets:
        if isValid(widget):
            widget.close()
    qapp.processEvents()

    # Wait some time for all widgets to be deleted
    for _ in range(10):
        validWidgets = [widget for widget in widgets if isValid(widget)]
        if validWidgets:
            qapp_utils.qWait(10)

    validWidgets = [widget for widget in widgets if isValid(widget)]
    assert not validWidgets, f"Some widgets were not destroyed: {validWidgets}"
