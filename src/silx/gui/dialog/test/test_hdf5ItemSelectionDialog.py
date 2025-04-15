import h5py
import pytest
import numpy
from silx.io.url import DataUrl
from silx.gui import qt
from silx.gui.dialog.GroupDialog import GroupDialog
from silx.gui.dialog.DatasetDialog import DatasetDialog


@pytest.mark.parametrize(
    "cls_cst_data_path", ((GroupDialog, "/path/to"), (DatasetDialog, "/path/to/data"))
)
def test_setSelectedUrl(qapp, cls_cst_data_path, tmp_path):
    """Check coherence between setSelectedDataUrl and getSelectedDataUrl"""

    class_constructor, data_path_to_test = cls_cst_data_path
    my_file = tmp_path / "file.hdf5"
    with h5py.File(my_file, mode="w") as h5f:
        h5f["path/to/data"] = numpy.ones((10, 10))

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
