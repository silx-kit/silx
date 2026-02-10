import numpy
import pytest

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.LegendsWidget import LegendsWidget
from .. import items

@pytest.fixture
def plot_with_legend(qWidgetFactory):
    plot = qWidgetFactory(PlotWidget)
    widget = qWidgetFactory(LegendsWidget, plotWidget=plot)
    return plot, widget


def test_initial_state(qWidgetFactory):
    legend_widget = qWidgetFactory(LegendsWidget)
    assert len(legend_widget._itemWidgets) == 0


def test_auto_sync(qapp, plot_with_legend):
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

def test_manual_add_same_item(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="c1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1
    item = plot.getCurve("c1")
    legend_widget.addItem(item)
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1

def test_manual_remove_item_and_then_from_plot(qapp, plot_with_legend):
    plot, legend_widget = plot_with_legend
    plot.addCurve([0, 1], [0, 1], legend="c1")
    plot.addCurve([1, 1], [1, 1], legend="c2")
    qapp.processEvents()
    item = plot.getCurve("c1")
    legend_widget.removeItem(item)
    assert len(legend_widget._itemWidgets) == 1
    assert plot.getCurve("c1") is not None
    plot.remove("c1")
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 1

def test_setPlotWidget_switch_cases(qapp, qWidgetFactory):
    plot1 = qWidgetFactory(PlotWidget)
    plot2 = qWidgetFactory(PlotWidget)
    legend_widget = qWidgetFactory(LegendsWidget, plotWidget=plot1)
    plot1.addCurve([0, 1], [0, 1], legend="p1_item")
    orphan_item = items.Curve()
    orphan_item.setName("orphan_item")
    legend_widget.addItem(orphan_item)
    qapp.processEvents()
    assert len(legend_widget._itemWidgets) == 2
    legend_widget.setPlotWidget(plot2)
    qapp.processEvents()
    assert "p1_item" not in [it.getName() for it in legend_widget._itemWidgets]
    assert "orphan_item" in [it.getName() for it in legend_widget._itemWidgets]

def test_orphan_item_interaction(qapp, qapp_utils, qWidgetFactory):
    legend_widget = qWidgetFactory(LegendsWidget)
    orphan_item = items.Curve()
    orphan_item.setName("orphan")
    legend_widget.addItem(orphan_item)
    qapp.processEvents()
    item_widget = legend_widget._itemWidgets[orphan_item]
    assert orphan_item.isVisible() is True
    assert item_widget._label.isEnabled() is True
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    qapp.processEvents()
    assert orphan_item.isVisible() is False
    assert item_widget._label.isEnabled() is False
    qapp_utils.mouseClick(item_widget, qt.Qt.LeftButton)
    assert orphan_item.isVisible() is True
    assert item_widget._label.isEnabled() is True