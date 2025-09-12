import os
import sys
import tempfile

import h5py
import numpy

from silx.io.dictdump import dicttonx


def generate_example(tmpdir):
    x = numpy.arange(110) / 50
    y = numpy.random.uniform(size=110)

    data = {
        "@NX_class": "NXroot",  # HDF5 attribute
        "@default": "entry",
        "entry": {
            "@NX_class": "NXentry",
            "@default": "process",
            "process": {
                "@NX_class": "NXprocess",
                "@default": "plot2d",
                "description": "Dark-current subtraction",
                "software_name": "MyReductionPipeline",
                "version": "1.0",
                "parameters": {
                    "@NX_class": "NXparameters",
                    "dark_current_level": 42.0,
                    "threshold": 100,
                },
                "data": {
                    "@NX_class": "NXcollection",
                    ">x": "./raw_data.h5::/1.1/instrument/positioners/samy",  # HDF5 external link
                    "y": y,
                },
                "plot1d": {
                    ">y": "../data/y",  # HDF5 soft link
                    ">x": "../data/x",
                    "@signal": "y",
                    "@axes": "x",
                    "@NX_class": "NXdata",
                    "title": "Dark-current subtracted",
                },
                "plot2d": {
                    ">y": {  # HDF5 virtual dataset
                        "dictdump_schema": "vds_v1",
                        "shape": (10, 11),
                        "dtype": float,
                        "sources": [
                            {"data_path": "../data/y", "shape": (110,), "dtype": float},
                        ],
                    },
                    "@signal": "y",
                    "@NX_class": "NXdata",
                    "title": "Dark-current subtracted",
                },
            },
        },
    }

    raw_filename = os.path.join(tmpdir, "raw_data.h5")
    processed_filename = os.path.join(tmpdir, "processed_data.h5")

    with h5py.File(processed_filename, "a") as h5file:
        dicttonx(
            treedict=data,
            h5file=h5file,
            h5path="/",
            update_mode="replace",
            add_nx_class=True,
        )

    with h5py.File(raw_filename, "w") as h5file:
        h5file["/1.1/instrument/positioners/samy"] = x

    return processed_filename


def main(args):
    from silx.app.view import main as silx_view_main

    with tempfile.TemporaryDirectory(prefix="silxdocs_") as tmpdir:
        filename = generate_example(tmpdir)

        silx_view_main.main((*args, filename))


if __name__ == "__main__":
    main(sys.argv)
