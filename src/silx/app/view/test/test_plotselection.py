import pytest
from silx.gui import qt
from silx.io.dictdump import dicttoh5
import silx.io
from silx.app.view.CustomPlotSelectionWindow import CustomPlotSelectionWindow

@pytest.fixture
def setupWindow(qapp, tmpdir):
    # Create HDF5 file with 1D dataset in tmpdir
    filename = str(tmpdir.join("test.h5"))
    dicttoh5({"x": [1, 2, 3], "y1": [1, 2], "y2": [4, 3, 2, 1], "y3": []}, filename)

    # Initialize the window
    window = CustomPlotSelectionWindow()
    qapp.processEvents()
    window.show()
    qapp.processEvents()
    
    return window, filename

def testRemoveDataset(setupWindow, qapp):
    window, filename = setupWindow

    # Create mime data and drag enter event with the HDF5 file path
    mime_data = qt.QMimeData()
    url = silx.io.url.DataUrl(file_path=filename, data_path="/x")
    mime_data.setData("application/x-silx-uri", url.path().encode())
    
    # Create drop event for X dataset
    dragEnterEvent = qt.QDragEnterEvent(qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dragEnterEvent(dragEnterEvent)
    qapp.processEvents()
    
    # Create drop event for Y1 dataset
    dropEvent = qt.QDropEvent(qt.QPoint(50, 50), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dropEvent(dropEvent)
    qapp.processEvents()

    model = window.treeView.model()
    assert model.getYParent().rowCount() == 2

    # Remove the Y1 dataset
    window.treeView._removeFile(model.getYParent().child(0, 2), model.getYParent())
    qapp.processEvents()

    assert model.getYParent().rowCount() == 1

    window.close()

def testResetModel(setupWindow, qapp):
    window, filename = setupWindow

    # Create mime data and drag enter event with the HDF5 file path
    mime_data = qt.QMimeData()
    url = silx.io.url.DataUrl(file_path=filename, data_path="/x")
    mime_data.setData("application/x-silx-uri", url.path().encode())
    
    # Create drop event for X dataset
    dragEnterEvent = qt.QDragEnterEvent(qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dragEnterEvent(dragEnterEvent)
    qapp.processEvents()
    
    # Create drop event for Y1 dataset
    dropEvent = qt.QDropEvent(qt.QPoint(50, 50), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dropEvent(dropEvent)
    qapp.processEvents()

    model = window.treeView.model()
    assert model.getYParent().rowCount() == 2

    # Reset the model
    model.clearAll()
    qapp.processEvents()

    assert model.getXParent().rowCount() == 0
    assert model.getYParent().rowCount() == 1 

    window.close()

def testDrop(qapp, qapp_utils, tmpdir):
    # Create HDF5 file with 1D dataset in tmpdir
    filename = str(tmpdir.join("test.h5"))
    dicttoh5({"x": [1, 2, 3], "y1": [1, 2], "y2": [4, 3, 2, 1], "y3": []}, filename)

    # Initialize the window
    window = CustomPlotSelectionWindow()
    qapp_utils.qWaitForWindowExposed(window)
    window.show()

   # Create mime data and drag enter event with the HDF5 file path
    mime_data = qt.QMimeData()
    url = silx.io.url.DataUrl(file_path=filename, data_path="/x")
    mime_data.setData("application/x-silx-uri", url.path().encode())

    # Create drop event for X dataset
    dragEnterEvent = qt.QDragEnterEvent(qt.QPoint(0, 0), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dragEnterEvent(dragEnterEvent)
    qapp.processEvents()

    # Create drag move event
    dragMoveEvent = qt.QDragMoveEvent(qt.QPoint(50, 50), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dragMoveEvent(dragMoveEvent)
    qapp.processEvents()

    # Create drop event
    dropEvent = qt.QDropEvent(qt.QPoint(50, 50), qt.Qt.CopyAction, mime_data, qt.Qt.LeftButton, qt.Qt.NoModifier)
    window.plot1D.dropEvent(dropEvent)
    qapp.processEvents()

    # Verify the drop by checking if the data is in the model
    model = window._treeView.model()
    assert model.rowCount() == 2

    x_item = model.getXParent()
    y_item = model.getYParent().child(0, 1)
    
    assert x_item is not None
    assert y_item is not None

    assert x_item.text() == "x"
    assert y_item.text() in ["y1", "y2", "y3"]

    window.close()
