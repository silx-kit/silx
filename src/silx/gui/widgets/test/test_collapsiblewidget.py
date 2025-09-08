from silx.gui import qt
from silx.gui.widgets.CollapsibleWidget import CollapsibleWidget


def test_collapse(qWidgetFactory):
    widget: CollapsibleWidget = qWidgetFactory(CollapsibleWidget)

    layout = qt.QHBoxLayout()
    innerWidget = qWidgetFactory(qt.QWidget)
    layout.addWidget(innerWidget)
    widget.setContentsLayout(layout)

    assert not widget.isCollapsed()
    widget.setCollapsed(True)
    assert widget.isCollapsed()
