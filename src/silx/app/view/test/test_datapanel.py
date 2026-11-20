import h5py
import numpy
import pytest

from silx.app.view.DataPanel import DataPanel


@pytest.fixture
def dataset_h5(tmp_path):
    filename = str(tmp_path / "base.h5")
    with h5py.File(filename, "w") as h5:
        h5["/group/dataset"] = numpy.arange(12).reshape(3, 4)
    with h5py.File(filename, "r") as h5:
        yield h5


def test_remove_datasets_from_hides_the_title(qapp, dataset_h5):
    """Closing the displayed file must clear the panel, header included."""
    panel = DataPanel()
    panel.setData(dataset_h5["/group/dataset"])
    title = panel._DataPanel__dataTitle
    assert title.isVisibleTo(panel)

    panel.removeDatasetsFrom(dataset_h5)

    assert panel.getData() is None
    assert not title.isVisibleTo(panel)
    panel.close()
