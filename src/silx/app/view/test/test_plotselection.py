import pytest
from silx.gui import qt
from silx.io.dictdump import dicttoh5
import silx.io
from silx.app.view.CustomPlotSelectionWindow import CustomPlotSelectionWindow


@pytest.fixture
def silxMimeData(tmpdir):
    # Create HDF5 file with 1D dataset in tmpdir
    filename = str(tmpdir.join("test.h5"))
    dicttoh5({"x": [1, 2, 3], "y1": [1, 2], "y2": [4, 3, 2, 1], "y3": []}, filename)

    # Create mime data
    mime_data = qt.QMimeData()
    url = silx.io.url.DataUrl(file_path=filename, data_path="/x")
    mime_data.setData("application/x-silx-uri", url.path().encode())

    return mime_data


@pytest.fixture
def invalidSilxMimeData(tmpdir):
    # Create HDF5 file with 2D dataset in tmpdir
    filename = str(tmpdir.join("test.h5"))
    dicttoh5({"x": [[1, 2], [3, 4]], "y": [[1, 2], [3, 4]]}, filename)

    # Create mime data
    mime_data = qt.QMimeData()
    url = silx.io.url.DataUrl(file_path=filename, data_path="/x")
    mime_data.setData("application/x-silx-uri", url.path().encode())

    return mime_data


def testDrop(qapp, silxMimeData, qWidgetFactory):
    window = qWidgetFactory(CustomPlotSelectionWindow)

    mime_data = silxMimeData

    # Create drag enter event
    dragEnterEvent = qt.QDragEnterEvent(
        qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier
    )
    window.getPlot1D().dragEnterEvent(dragEnterEvent)
    qapp.processEvents()

    # Create drag move event
    dragMoveEvent = qt.QDragMoveEvent(
        qt.QPoint(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dragMoveEvent(dragMoveEvent)
    qapp.processEvents()

    # Create drop event
    dropEvent = qt.QDropEvent(
        qt.QPointF(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dropEvent(dropEvent)
    qapp.processEvents()

    # Verify the drop by checking if the data is in the model
    model = window.getTreeView().model()
    assert model.rowCount() == 2

    x_item = model.getXParent()
    y_item = model.getYParent().child(0, 1)

    assert x_item is not None
    assert y_item is not None


def testRemoveDataset(silxMimeData, qapp, qWidgetFactory):
    window = qWidgetFactory(CustomPlotSelectionWindow)

    mime_data = silxMimeData

    # Create drop event for X dataset
    dragEnterEvent = qt.QDragEnterEvent(
        qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier
    )
    window.getPlot1D().dragEnterEvent(dragEnterEvent)
    qapp.processEvents()

    # Create drop event for Y1 dataset
    dropEvent = qt.QDropEvent(
        qt.QPointF(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dropEvent(dropEvent)
    qapp.processEvents()

    model = window.getTreeView().model()
    assert model.getYParent().rowCount() == 2

    # Remove the dataset
    window.getTreeView()._removeFile(model.getYParent().child(0, 2), model.getYParent())
    qapp.processEvents()

    assert model.getYParent().rowCount() == 1


def testResetModel(silxMimeData, qapp, qWidgetFactory):
    window = qWidgetFactory(CustomPlotSelectionWindow)

    mime_data = silxMimeData

    # Create drop event for X dataset
    dragEnterEvent = qt.QDragEnterEvent(
        qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier
    )
    window.getPlot1D().dragEnterEvent(dragEnterEvent)
    qapp.processEvents()

    # Create drop event for dataset
    dropEvent = qt.QDropEvent(
        qt.QPointF(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dropEvent(dropEvent)
    qapp.processEvents()

    model = window.getTreeView().model()
    assert model.getYParent().rowCount() == 2

    # Reset the model
    model.clearAll()
    qapp.processEvents()

    assert model.getXParent().rowCount() == 0
    assert model.getYParent().rowCount() == 1


def testMultipleDrop(qapp, silxMimeData, qWidgetFactory, qapp_utils):
    window = qWidgetFactory(CustomPlotSelectionWindow)

    mime_data = silxMimeData

    # Create drag enter event
    dragEnterEvent = qt.QDragEnterEvent(
        qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier
    )

    # Create drag move event
    dragMoveEvent = qt.QDragMoveEvent(
        qt.QPoint(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )

    # Create drop event
    dropEvent = qt.QDropEvent(
        qt.QPointF(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )

    for _ in range(2):
        window.getPlot1D().dragEnterEvent(dragEnterEvent)
        qapp.processEvents()
        window.getPlot1D().dragMoveEvent(dragMoveEvent)
        qapp.processEvents()
        window.getPlot1D().dropEvent(dropEvent)
        qapp.processEvents()

    # Verify the drop by checking if the data is in the model
    model = window.getTreeView().model()
    assert model.rowCount() == 2
    print(model.rowCount())

    x_item = model.getXParent()
    y_item1 = model.getYParent().child(0, 1)
    y_item2 = model.getYParent().child(1, 1)

    assert x_item is not None
    assert y_item1 is not None
    assert y_item2 is not None


def testDropInvalid(qapp, invalidSilxMimeData, qWidgetFactory, qapp_utils):
    window = qWidgetFactory(CustomPlotSelectionWindow)

    mime_data = invalidSilxMimeData

    # Create drag enter event
    dragEnterEvent = qt.QDragEnterEvent(
        qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier
    )
    window.getPlot1D().dragEnterEvent(dragEnterEvent)
    qapp.processEvents()

    # Create drag move event
    dragMoveEvent = qt.QDragMoveEvent(
        qt.QPoint(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dragMoveEvent(dragMoveEvent)
    qapp.processEvents()

    # Create drop event
    dropEvent = qt.QDropEvent(
        qt.QPointF(50, 50),
        qt.Qt.CopyAction,
        mime_data,
        qt.Qt.LeftButton,
        qt.Qt.NoModifier,
    )
    window.getPlot1D().dropEvent(dropEvent)
    qapp.processEvents()

    # Verify the drop by checking if the data is in the model
    model = window.getTreeView().model()
    assert model.rowCount() == 2
