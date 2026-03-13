import pytest

from silx.gui.widgets.IntEdit import IntEdit


@pytest.fixture
def intEdit(qWidgetFactory):
    widget = qWidgetFactory(IntEdit)
    yield widget


def test_value(intEdit: IntEdit):
    intEdit.setCurrentValue(150)
    assert intEdit.getCurrentValue() == 150
    assert intEdit.getValue() == 150

    intEdit.setCurrentValue(-50)
    assert intEdit.getCurrentValue() == -50
    assert intEdit.getValue() == -50

    intEdit.setCurrentValue(5.5)
    assert intEdit.getCurrentValue() is None
    assert intEdit.getValue() == -50


def test_default_value(intEdit: IntEdit):
    intEdit.setDefaultValue(145)
    assert intEdit.getCurrentValue() is None
    assert intEdit.getValue() == 145

    intEdit.setCurrentValue(50)
    assert intEdit.getCurrentValue() == 50
    assert intEdit.getValue() == 50

    intEdit.setCurrentValue(9.1)
    assert intEdit.getCurrentValue() is None
    assert intEdit.getValue() == 145


def test_range(intEdit: IntEdit):
    intEdit.setRange(-10, 20)
    assert intEdit.getRange() == (-10, 20)

    intEdit.setCurrentValue(25)
    assert intEdit.getCurrentValue() == 20
    assert intEdit.getValue() == 20

    intEdit.setCurrentValue(-50)
    assert intEdit.getCurrentValue() == -10
    assert intEdit.getValue() == -10

    intEdit.setCurrentValue(25, extend_range=True)
    assert intEdit.getCurrentValue() == 25
    assert intEdit.getValue() == 25
    assert intEdit.getRange() == (-10, 25)

    intEdit.setCurrentValue(-20, extend_range=True)
    assert intEdit.getCurrentValue() == -20
    assert intEdit.getValue() == -20
    assert intEdit.getRange() == (-20, 25)
