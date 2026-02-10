import numpy
import pytest

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.LegendsWidget import LegendsWidget


@pytest.fixture
def plot_with_legend(qWidgetFactory):
    plot = qWidgetFactory(PlotWidget)
    widget = qWidgetFactory(LegendsWidget, plotWidget=plot)
    return plot, widget


def test_initial_state(plot_with_legend):
    _, legend_widget = plot_with_legend
    assert len(legend_widget._itemWidgets) == 0


def test_add_remove_items(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve(numpy.arange(10), numpy.arange(10), legend="curve1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 0
    item = plot.getCurve("curve1")
    legend_widget.addItem(item)
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1
    legend_widget.removeItem(item)
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 0


def test_visibility_toggle(qapp, qapp_utils, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="test_item")
    item = plot.getCurve("test_item")
    legend_widget.addItem(item)
    qapp.processEvents()
    item = plot.getCurve("test_item")
    item_widget = legend_widget._itemWidgets[item]
    assert item.isVisible() is True
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    assert item.isVisible() is False
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    assert item.isVisible() is True


def test_clear_items(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="c1")
    plot.addCurve([0, 1], [1, 0], legend="c2")
    legend_widget.addItem(plot.getCurve("c1"))
    legend_widget.addItem(plot.getCurve("c2"))
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 2
    legend_widget.clearItems()
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 0


def test_sync_on_plot_remove(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="c1")
    legend_widget.addItem(plot.getCurve("c1"))
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1
    plot.remove("c1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 0
