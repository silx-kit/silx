import os
import sys
import tempfile

import h5py
import numpy

from fabio.tifimage import TifImage
from silx.io.dictdump import dicttonx


def generate_example(tmpdir):
    nimages = 5
    shape = (50, 60)

    # # Link target to stack TIFF images along the first dimension of a 3D dataset.
    # # Source files are not opened.
    target = {
        "dictdump_schema": "external_binary_link_v1",
        "dtype": numpy.uint16,
        "shape": (5, 50, 60),
        "sources": [
            ("data0.tiff", 196, 6000),
            ("data1.tiff", 196, 6000),
            ("data2.tiff", 196, 6000),
            ("data3.tiff", 196, 6000),
            ("data4.tiff", 196, 6000),
        ],
    }

    schema = {
        "@default": "plot2d",
        "plot2d": {
            ">images": target,
            "@signal": "images",
            "@NX_class": "NXdata",
            "title": "TIFF image stack",
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

    # Create TIFF files
    y, x = numpy.ogrid[: shape[0], : shape[1]]
    for i in range(nimages):
        data_file = os.path.join(tmpdir, f"data{i}.tiff")
        data = (1000 * (numpy.sin(0.2 * x + i) + numpy.cos(0.2 * y + i)) + 2000).astype(
            numpy.uint16
        )

        with TifImage(data=data) as tifimage:
            tifimage.write(data_file)

    # Link target to stack TIFF images along the first dimension of a 3D dataset.
    # Source files are opened.
    target = [f"data{i}.tiff" for i in range(nimages)]

    schema = {
        "@default": "plot2d",
        "plot2d": {
            ">images": target,
            "@signal": "images",
            "@NX_class": "NXdata",
            "title": "TIFF image stack",
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

    return [processed_filename1, processed_filename2, *target]


def main(args):
    from silx.app.view import main as silx_view_main

    with tempfile.TemporaryDirectory(prefix="silxdocs_") as tmpdir:
        filenames = generate_example(tmpdir)
        silx_view_main.main((*args, *filenames))


if __name__ == "__main__":
    main(sys.argv)
