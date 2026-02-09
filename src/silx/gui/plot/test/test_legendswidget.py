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
    plot, legend_widget = plot_with_legend
    assert len(legend_widget._itemWidgets) == 0


def test_add_remove_items(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve(numpy.arange(10), numpy.arange(10), legend="curve1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1
    plot.remove(legend="curve1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 0

def test_visibility_toggle(qapp, qapp_utils, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="test_item")
    qapp.processEvents()
    item = plot.getCurve("test_item")
    item_widget = legend_widget._itemWidgets[item]
    assert item.isVisible() is True
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    assert item.isVisible() is False
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    assert item.isVisible() is True
