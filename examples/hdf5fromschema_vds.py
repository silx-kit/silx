import os
import sys
import tempfile

import h5py
import numpy

from silx.io.dictdump import dicttonx


def generate_example(tmpdir):
    ndatasets = 3
    nimages_per_dataset = 5
    border = 1
    shape = (nimages_per_dataset, 50, 60)

    # Link target to stack the 3D datasets along the first dimension.
    # Source files are not opened.
    target = {
        "dictdump_schema": "vds_v1",
        "dtype": numpy.dtype("uint16"),
        "shape": (ndatasets * nimages_per_dataset, 10, 10),
        "sources": [
            {
                "data_path": "/group/dataset",
                "dtype": numpy.dtype("uint16"),
                "file_path": "data0.h5",
                "shape": shape,
                "source_index": (
                    slice(None, None, None),
                    slice(20, 30, None),
                    slice(40, 50, None),
                ),
                "target_index": slice(0, 5, None),
            },
            {
                "data_path": "/group/dataset",
                "dtype": numpy.dtype("uint16"),
                "file_path": "data1.h5",
                "shape": shape,
                "source_index": (
                    slice(None, None, None),
                    slice(20, 30, None),
                    slice(40, 50, None),
                ),
                "target_index": slice(5, 10, None),
            },
            {
                "data_path": "/group/dataset",
                "dtype": numpy.dtype("uint16"),
                "file_path": "data2.h5",
                "shape": shape,
                "source_index": (
                    slice(None, None, None),
                    slice(20, 30, None),
                    slice(40, 50, None),
                ),
                "target_index": slice(10, 15, None),
            },
        ],
    }

    schema = {
        "@default": "plot2d",
        "plot2d": {
            ">images_roi": target,
            "@signal": "images_roi",
            "@NX_class": "NXdata",
            "title": "Image stack",
        },
    }

    processed_filename1 = os.path.join(tmpdir, "master1.h5")

    with h5py.File(processed_filename1, "a") as h5file:
        dicttonx(
            treedict=schema,
            h5file=h5file,
            h5path="/",
            update_mode="replace",
            add_nx_class=True,
        )

    # Create 3D dataset sources
    for i in range(ndatasets):
        data_file = os.path.join(tmpdir, f"data{i}.h5")
        data = numpy.zeros(shape, dtype=numpy.uint16)
        for j in range(nimages_per_dataset):
            v = i * nimages_per_dataset + j + 1
            data[j, 20 + border : 30 - border, 40 + border : 50 - border] = v
        with h5py.File(data_file, mode="w") as fh:
            fh["/group/dataset"] = data

    # Link target to stack the 3D datasets along the first dimension.
    # Source files are opened.
    target = [
        f"data{i}.h5?path=/group/dataset&slice=:,20:30,40:50" for i in range(ndatasets)
    ]

    schema = {
        "@default": "plot2d",
        "plot2d": {
            ">images_roi": target,
            "@signal": "images_roi",
            "@NX_class": "NXdata",
            "title": "Image stack",
        },
    }

    processed_filename2 = os.path.join(tmpdir, "master2.h5")

    with h5py.File(processed_filename2, "a") as h5file:
        dicttonx(
            treedict=schema,
            h5file=h5file,
            h5path="/",
            update_mode="replace",
            add_nx_class=True,
        )

    return [processed_filename1, processed_filename2]


def main(args):
    from silx.app.view import main as silx_view_main

    with tempfile.TemporaryDirectory(prefix="silxdocs_") as tmpdir:
        filenames = generate_example(tmpdir)
        silx_view_main.main((*args, *filenames))


if __name__ == "__main__":
    main(sys.argv)
