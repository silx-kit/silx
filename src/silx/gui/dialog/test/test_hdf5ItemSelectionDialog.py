import h5py
import pytest
import numpy
from silx.io.url import DataUrl
from silx.gui import qt
from silx.gui.dialog.GroupDialog import GroupDialog
from silx.gui.dialog.DatasetDialog import DatasetDialog


@pytest.mark.parametrize(
    "cls_cst_data_path",
    (
        (GroupDialog, "/path/to", "/path/to_2"),
        (DatasetDialog, "/path/to/data", "/path/to_2/data"),
    ),
)
def test_setSelectedUrl(qapp, cls_cst_data_path, tmp_path):
    """Check coherence between setSelectedDataUrl and getSelectedDataUrl"""

    class_constructor, data_path_to_test, data_path_to_test_2 = cls_cst_data_path
    my_file = tmp_path / "file.hdf5"
    with h5py.File(my_file, mode="w") as h5f:
        h5f["path/to/data"] = numpy.ones((10, 10))
        h5f["path/to_2/data"] = numpy.ones((10, 10))

    # group dialog
    dialog = class_constructor()
    selected_url = DataUrl(
        file_path=my_file,
        data_path=data_path_to_test,
    )
    dialog.setSelectedDataUrl(url=selected_url)

    assert dialog.getSelectedDataUrl().path() == selected_url.path()

    with pytest.raises(ValueError):
        dialog.setSelectedDataUrl(
            url=DataUrl(file_path="not/existing.hdf5", data_path="data", scheme="self")
        )

    # test use case the file is already added but not set a new dataset
    new_selected_url = DataUrl(
        file_path=my_file,
        data_path=data_path_to_test_2,
    )
    dialog.setSelectedDataUrl(url=new_selected_url)
    assert len(dialog._model._get_files()) == 1
    assert dialog.getSelectedDataUrl().path() == new_selected_url.path()

    # test use case setting again the previous url
    dialog.setSelectedDataUrl(url=selected_url)
    assert len(dialog._model._get_files()) == 1
    assert dialog.getSelectedDataUrl().path() == selected_url.path()


def test_adding_empty_file(qapp, tmp_path):
    """Test adding a DataUrl pointing to an empty file"""
    widget = DatasetDialog()
    my_file = tmp_path / "file.hdf5"
    with h5py.File(my_file, mode="w"):
        pass

    selected_url = DataUrl(
        file_path=my_file,
        data_path="/path/to/data",
    )

    with pytest.raises(ValueError):
        widget.setSelectedDataUrl(url=selected_url)

    # make sure the file haven't been added
    assert len(widget._model._get_files()) == 0
