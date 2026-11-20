"""Small derived applications reuse the viewer's file ownership lifecycle."""

import h5py

from silx.app.view.DataPanel import DataPanel
from silx.app.view.Viewer import Viewer
from silx.gui import qt
from silx.gui.hdf5 import Hdf5DatasetMimeData, Hdf5TreeModel, Hdf5TreeView


def test_viewer_uses_the_provided_data_panel(qWidgetFactory, tmp_path):
    class Panel(DataPanel):
        pass

    window = qWidgetFactory(Viewer, dataPanelFactory=Panel)
    panel = window.getDataPanel()
    assert isinstance(panel, Panel)
    assert window.findChild(Panel) is panel
    model = window.getHdf5TreeView().findHdf5TreeModel()
    assert isinstance(model, Hdf5TreeModel)

    # The viewer still manages the files it opens
    filename = str(tmp_path / "data.h5")
    with h5py.File(filename, "w") as source:
        source["value"] = 7
    loaded = []
    window.sigFileLoaded.connect(lambda root, name: loaded.append(root))
    window.appendFile(filename)
    assert len(loaded) == 1 and model.rowCount() == 1
    assert window.getHdf5TreeView() is window.findChild(Hdf5TreeView)
    splitter = window.getBrowserSplitter()
    assert isinstance(splitter, qt.QSplitter)
    assert splitter.isAncestorOf(window.getHdf5TreeView())


def _recordFileSignals(window):
    events = []
    window.sigFileLoaded.connect(
        lambda root, name: events.append(("loaded", root, bool(root.id)))
    )
    window.sigFileAboutToBeClosed.connect(
        lambda root: events.append(("closing", root, bool(root.id)))
    )
    return events


def test_file_signals_are_emitted_while_the_file_is_open(qWidgetFactory, tmp_path):
    filename = str(tmp_path / "data.h5")
    with h5py.File(filename, "w") as source:
        source["value"] = 7

    window = qWidgetFactory(Viewer)
    events = _recordFileSignals(window)

    window.appendFile(filename)
    root = events[0][1]
    assert events == [("loaded", root, True)]

    window.refreshOpenedRoot(root)
    reloaded = events[2][1]
    assert events[1:] == [("closing", root, True), ("loaded", reloaded, True)]
    assert not bool(root.id) and bool(reloaded.id)

    window.close()
    assert events[3:] == [("closing", reloaded, True)]
    assert not bool(reloaded.id)


def test_file_signals_on_model_synchronization(qWidgetFactory, qapp_utils, tmp_path):
    filename = str(tmp_path / "data.h5")
    with h5py.File(filename, "w") as source:
        source["value"] = 7

    window = qWidgetFactory(Viewer)
    events = _recordFileSignals(window)
    window.appendFile(filename)
    root = events[0][1]

    window.getHdf5TreeView().findHdf5TreeModel().synchronizeH5pyObject(root)
    for _ in range(250):  # The model reloads the file asynchronously
        if len(events) >= 3:
            break
        qapp_utils.qWait(20)
    reloaded = events[2][1]
    assert reloaded is not root
    assert events[1:] == [("closing", root, True), ("loaded", reloaded, True)]
    assert not bool(root.id) and bool(reloaded.id)


def test_second_close_request_does_not_clean_up_again(qWidgetFactory, tmp_path):
    filename = str(tmp_path / "data.h5")
    with h5py.File(filename, "w") as source:
        source["value"] = 7
    window = qWidgetFactory(Viewer)
    window.appendFile(filename)
    closed = []
    window.sigFileAboutToBeClosed.connect(closed.append)

    assert window.close()
    assert len(closed) == 1
    assert window.close()
    assert len(closed) == 1


def test_dataset_drag_payload_is_available_from_the_public_package(qapp):
    payload = Hdf5DatasetMimeData()
    assert isinstance(payload, qt.QMimeData)
