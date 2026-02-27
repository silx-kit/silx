import json

import h5py
import numpy as np
from silx.io.dictdump import dicttonx


def default_attribute_entry(path: str) -> dict:
    return {
        "@NX_class": "NXentry",
        "@default": "nxdata",
        "title": "Different valid/invalid @default",
        "nxdata": {
            "@NX_class": "NXdata",
            "@signal": "signal",
            "@axes": ["axis"],
            "signal": [1, 2, 1],
            "axis": [1, 2, 3],
        },
        "absolute_default": {
            "@NX_class": "NXprocess",
            "@default": f"{path}/nxdata",
        },
        "relative_default": {
            "@NX_class": "NXprocess",
            "@default": "../nxdata",
        },
        "sub_sub_group_default": {
            "@NX_class": "NXprocess",
            "@default": "sub_group/nxdata",
            "sub_group": {
                "nxdata": {
                    "@NX_class": "NXdata",
                    "@signal": "signal",
                    "@axes": ["axis"],
                    "signal": [1, 2, 1],
                    "axis": [1, 2, 3],
                },
            },
        },
        "dangling_default": {
            "@NX_class": "NXprocess",
            "@default": "unavailable_dataset",
        },
        "default_to_empty_group": {
            "@NX_class": "NXprocess",
            "@default": "empty_group",
            "empty_group": {},
        },
        "default_to_empty_nxdata": {
            "@NX_class": "NXsubentry",
            "@default": "empty_nxdata",
            "empty_nxdata": {
                "@NX_class": "Nxprocess",
            },
        },
    }


def nxdata_documentation_examples_entry() -> dict:
    entry = {
        "@NX_class": "NXentry",
        "title": "NXdata groups from examples in nexus documentation",
        "program_name": "nexus",
        "program_name@version": "2026.01",
        "reference": "https://manual.nexusformat.org/classes/base_classes/NXdata.html",
    }

    x = [0, 1, 8, 30, 35, 150, 200, 340, 500, 520]
    entry["plot_curve"] = {
        "@NX_class": "NXdata",
        "@auxiliary_signals": ["y2"],
        "@axes": ["x"],
        "@signal": "y1",
        "x": x,
        "y1": 4 + np.sin(2 * np.array(x)),
        "y2": 2 + np.sin(3 * np.array(x)),
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_curve.html",
    }

    rstate = np.random.RandomState(42)
    x = rstate.uniform(-3, 3, 500)
    y = rstate.uniform(-3, 3, 500)
    z = (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2) - y**2)
    entry["plot_scatter2d"] = {
        "@NX_class": "NXdata",
        "@x_indices": [0],
        "@y_indices": [0],
        "@signal": "z",
        "x": x,
        "y": y,
        "z": z,
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_scatter2d.html",
    }

    x = np.linspace(-3, 3, 16)
    y = np.linspace(-3, 3, 30)
    xx, yy = np.meshgrid(x, y)
    z = (1 - xx / 2 + xx**5 + yy**3) * np.exp(-(xx**2) - yy**2)
    entry["plot_image"] = {
        "@NX_class": "NXdata",
        "@axes": ["y", "x"],
        "@signal": "z",
        "x": x,
        "y": y,
        "z": z,
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_image.html",
    }

    entry["plot_hist1d"] = {
        "@NX_class": "NXdata",
        "@axes": ["x"],
        "@signal": "y",
        "x": [0.5, 1.5, 2.5, 4, 5, 6.5, 7, 8],
        "y": [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6],
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_hist1d.html",
    }

    x = [-3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0]
    y = [-3.0, -2.8, -1.3, -0.75, 0.0, 0.1, 1.5, 2.25, 3.0]
    xx = np.linspace(-3, 3, 200)
    yy = np.linspace(-3, 3, 200)
    xx, yy = np.meshgrid(xx, yy)
    zz = (1 - xx / 2 + xx**5 + yy**3) * np.exp(-(xx**2) - yy**2)
    z, _, _ = np.histogram2d(
        yy.flatten(), xx.flatten(), bins=[y, x], weights=zz.flatten()
    )
    entry["plot_hist2d"] = {
        "@NX_class": "NXdata",
        "@axes": ["y", "x"],
        "@signal": "z",
        "x": x,
        "y": y,
        "z": z,
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_hist2d.html",
    }

    x = [-3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0]
    y = np.linspace(-3, 3, 16)
    nx = len(x) - 1
    ny = len(y)
    z = np.zeros((ny, nx))
    xx = np.linspace(-3, 3, 200)
    for i in range(ny):
        zi = (1 - xx / 2 + xx**5 + y[i] ** 3) * np.exp(-(xx**2) - y[i] ** 2)
        z[i, :], _ = np.histogram(xx, bins=x, weights=zi)
    entry["plot_hist2dmix"] = {
        "@NX_class": "NXdata",
        "@axes": ["y", "x"],
        "@signal": "z",
        "x": x,
        "y": y,
        "z": z,
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_hist2dmix.html",
    }

    x_set = np.linspace(-3, 3, 6)  # TODO not 7
    y_set = np.linspace(-3, 3, 16)

    rstate = np.random.RandomState(42)
    noise_x = 0.1 * (x_set[1] - x_set[0])
    noise_y = 0.1 * (y_set[1] - y_set[0])
    x_encoder = x_set[np.newaxis, :] + rstate.normal(
        0, noise_x, (len(y_set), len(x_set))
    )
    y_encoder = y_set + rstate.normal(0, noise_y, len(y_set))

    nx = len(x_set) - 1
    ny = len(y_set)
    z = np.zeros((ny, nx))
    xx = np.linspace(-3, 3, 200)
    for i in range(ny):
        zi = (1 - xx / 2 + xx**5 + y_encoder[i] ** 3) * np.exp(
            -(xx**2) - y_encoder[i] ** 2
        )
        z[i, :], _ = np.histogram(xx, bins=x_encoder[i, :], weights=zi)

    entry["plot_fscan2d"] = {
        "@NX_class": "NXdata",
        "@axes": ["y_set", "x_set"],
        "@x_encoder_indices": [0, 1],
        "@y_encoder_indices": 0,
        "@signal": "z",
        "z": z,
        "x_encoder": x_encoder,
        "y_encoder": y_encoder,
        "x_set": x_set,
        "y_set": y_set,
        "reference": "https://manual.nexusformat.org/classes/base_classes/data/plot_fscan2d.html",
    }

    return entry


def nxdata_documentation_snippets_entry() -> dict:
    entry = {
        "@NX_class": "NXentry",
        "title": "NXdata groups from NeXus snippets in nexus documentation",
        "program_name": "nexus",
        "program_name@version": "2026.01",
        "reference": "https://manual.nexusformat.org/classes/base_classes/NXdata.html",
    }

    entry["simple_curve"] = {
        "title": "Example of a simple curve plot",
        "@NX_class": "NXdata",
        "@axes": ["x"],
        "@signal": "data",
        "data": np.arange(100.0),
        "x": np.arange(100.0),
    }

    entry["auxiliary_signals"] = {
        "title": "Example with three signals, one of which being the default",
        "@NX_class": "NXdata",
        "@signal": "data1",
        "@auxiliary_signals": ["data2", "data3"],
        "data1": np.arange(10.0 * 20 * 30).reshape(10, 20, 30),
        "data2": np.arange(10.0 * 20 * 30).reshape(10, 20, 30) + 1,
        "data3": np.arange(10.0 * 20 * 30).reshape(10, 20, 30) + 2,
    }

    entry["axes_features"] = {
        "title": "Example covering all axes features supported",
        "@NX_class": "NXdata",
        "@signal": "data",
        "@axes": ["x_set", "y_set", "."],
        "@x_encoder_indices": [0, 1],
        "@y_encoder_indices": 1,
        "data": np.arange(10.0 * 7 * 1024).reshape(10, 7, 1024),
        "x_encoder": [np.linspace(-3.1, 3.1, 11) for _ in range(7)],
        "y_encoder": np.linspace(-3, 3, 7),
        "x_set": np.linspace(-3, 3, 10),
        "y_set": np.linspace(-3, 3, 7),
    }

    entry["uncertainties"] = {
        "title": "Example of uncertainties on the signal, auxiliary signals and axis coordinates",
        "@NX_class": "NXdata",
        "@signal": "data1",
        "@auxiliary_signals": ["data2", "data3"],
        "@axes": ["x", ".", "z"],
        "data1": np.arange(10.0 * 20 * 30).reshape(10, 20, 30),
        "data2": np.arange(10.0 * 20 * 30).reshape(10, 20, 30) + 1,
        "data3": np.arange(10.0 * 20 * 30).reshape(10, 20, 30) + 2,
        "x": np.linspace(-3, 3, 10),
        "z": np.linspace(-3, 3, 30),
        "data1_errors": np.random.rand(10, 20, 30),
        "data2_errors": np.random.rand(10, 20, 30),
        "data3_errors": np.random.rand(10, 20, 30),
        "x_errors": np.random.rand(10),
        "z_errors": np.random.rand(30),
    }

    entry["default_slice_by_name"] = {
        "@NX_class": "NXdata",
        "@signal": "data",
        "@axes": ["image_id", "channel", ".", "."],
        "@image_id_indices": 0,
        "@channel_indices": 1,
        "@default_slice": [".", "difference", ".", "."],
        "image_id": np.arange(1, 10),
        "channel": ["threshold_1", "threshold_2", "difference"],
        "data": np.random.randint(0, 255, size=(10, 3, 100, 100)),
        "reference": "https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-default-slice-attribute",
    }

    entry["default_slice_by_index"] = {
        "@NX_class": "NXdata",
        "@signal": "data",
        "@axes": ["image_id", "channel", ".", "."],
        "@image_id_indices": 0,
        "@channel_indices": 1,
        "@default_slice": [".", "2", ".", "."],
        "image_id": np.arange(1, 10),
        "channel": ["threshold_1", "threshold_2", "difference"],
        "data": np.random.randint(0, 255, size=(10, 3, 100, 100)),
        "reference": "https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata-default-slice-attribute",
    }

    return entry


_JS_MAX_SAFE_INTEGER = 2**53 - 1


def h5web_mock_entry(path: str) -> dict:
    entry = {
        "@NX_class": "NXentry",
        "@default": "nexus_entry",
        "title": "NXdata groups from h5web mock provider",
        "program_name": "h5web",
        "program_name@version": "16.0.1",
        "reference": "https://github.com/silx-kit/h5web/blob/main/packages/app/src/providers/mock/mock-file.ts",
    }

    png_red_dot = np.frombuffer(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x05\x00\x00\x00\x05\x08\x06\x00\x00\x00\x8do&\xe5\x00\x00\x00\x1cIDAT\x08\xd7c\xf8\xff\xff?\xc3\x7f\x06 \x05\xc3 \x12\x84\xd01\xf1\x82X\xcd\x04\x00\x0e\xf55\xcb\xd1\x8e\x0e\x1f\x00\x00\x00\x00IEND\xaeB`\x82",
        dtype=np.uint8,
    )

    entry["entities"] = {
        "empty_group": {},
        # "empty_dataset": h5py.Empty(np.float32),  # TODO
        # "datatype": np.dtype([("int", np.int32)]),  # TODO
        "raw": 42,
        "raw_large": _JS_MAX_SAFE_INTEGER + 1,
        "raw_png": png_red_dot.astype(h5py.opaque_dtype(png_red_dot.dtype)),
        "scalar_num": 0,
        "scalar_num@attr": 0,
        "scalar_bigint": _JS_MAX_SAFE_INTEGER + 1,
        "scalar_bigint@attr": _JS_MAX_SAFE_INTEGER + 1,
        "scalar_str": "foo",
        "scalar_str@attr": "foo",
        "scalar_bool": True,
        "scalar_bool@attr": True,
        "scalar_cplx": 1 + 5j,
        "scalar_cplx@attr": 1 + 5j,
        "scalar_compound": np.array(
            [("foo", 2)], dtype=[("str", h5py.string_dtype()), ("int", np.int8)]
        ),
        "scalar_array": [1, 2],
        "scalar_array@attr": [1, 2],
        # "scalar_vlen": [1, 2, 3],  # TODO
        # "scalar_vlen@attr": [1, 2, 3],  # TODO
        # enum not supported by h5py
        # "scalar_enum": 2,
        # cannot write hard link with h5py
        # "unresolved_hard_link": h5py.HardLink(""),
        "unresolved_soft_link": h5py.SoftLink("/foo"),
        "unresolved_external_link": h5py.ExternalLink(
            "unavailable_file.h5", "entry_000/dataset"
        ),
    }

    coumpound_dtype = [
        ("string", h5py.string_dtype()),
        ("int", int),
        ("float", float),
        ("bool", bool),
        ("complex", complex),
    ]
    entry["nD_datasets"] = {
        "oneD_linear": np.arange(100),
        "oneD": np.random.rand(100),
        "oneD_bigint": [_JS_MAX_SAFE_INTEGER] * 10,
        "oneD_cplx": np.random.rand(100) + 1j * np.random.rand(100),
        "oneD_compound": np.array(
            [(f"{i}", i, float(i), bool(i % 2), i + 1j) for i in range(100)],
            dtype=coumpound_dtype,
        ),
        "oneD_bool": np.random.choice([False, True], 100),
        # enum not supported by h5py
        # "oneD_enum": np.random.randint(0, 3, 100).astype(np.int8),
        "twoD": np.random.rand(10, 20),
        "twoD_neg": -np.random.rand(10, 20),
        "twoD_bigint": [[_JS_MAX_SAFE_INTEGER] * 10] * 10,
        "twoD_cplx": np.random.rand(10, 20) + 1j * np.random.rand(10, 20),
        "twoD_compound": np.array(
            [
                (f"{j},{i}", i, float(j), bool((j + i) % 2), j + 1j * i)
                for i in range(10)
                for j in range(20)
            ],
            dtype=coumpound_dtype,
        ),
        "twoD_bool": np.random.choice([False, True], (10, 20)),
        # enum not supported by h5py
        # "twoD_enum": np.random.randint(0, 3, (10, 20)).astype(np.int8),
        "threeD": np.random.rand(10, 20, 30),
        "threeD_bool": np.random.choice([False, True], (10, 20, 30)),
        "threeD_cplx": np.random.rand(10, 20, 30) + 1j * np.random.rand(10, 20, 30),
        "threeD_rgb": np.random.rand(3, 3, 3),
        "threeD_rgb@CLASS": "IMAGE",
        "threeD_rgb@IMAGE_SUBCLASS": "IMAGE_TRUECOLOR",
        "fourD": np.random.rand(10, 20, 30, 40),
    }

    entry["typed_arrays"] = {
        "uint8": np.random.randint(0, 255, (10, 20), dtype=np.uint8),
        "int16": np.random.randint(-32768, 32767, (10, 20), dtype=np.int16),
        "int64": np.random.randint(-(2**63), 2**63 - 1, (10, 20), dtype=np.int64),
        "float32": np.random.rand(10, 20).astype(np.float32),
        "float64": np.random.rand(10, 20).astype(np.float64),
        "uint8_rgb": np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8),
        "uint8_rgb@CLASS": "IMAGE",
        "uint8_rgb@IMAGE_SUBCLASS": "IMAGE_TRUECOLOR",
        "int8_rgb": np.random.randint(-128, 127, (3, 3, 3), dtype=np.int8),
        "int8_rgb@CLASS": "IMAGE",
        "int8_rgb@IMAGE_SUBCLASS": "IMAGE_TRUECOLOR",
        "int32_rgb": np.random.randint(0, 255, (3, 3, 3), dtype=np.int32),
        "int32_rgb@CLASS": "IMAGE",
        "int32_rgb@IMAGE_SUBCLASS": "IMAGE_TRUECOLOR",
        "float32_rgb": np.random.rand(3, 3, 3).astype(np.float32),
        "float32_rgb@CLASS": "IMAGE",
        "float32_rgb@IMAGE_SUBCLASS": "IMAGE_TRUECOLOR",
    }

    nexus_entry = {
        "@NX_class": "NXentry",
        "@default": "nx_process/nx_data",
    }
    entry["nexus_entry"] = nexus_entry

    nexus_entry["nx_process"] = {
        "@NX_class": "NXprocess",
        "nx_data": {
            "@NX_class": "NXdata",
            "@signal": "twoD",
            "@SILX_style": json.dumps({"signal_scale_type": "symlog"}),
            "title": "NeXus 2D",
            "twoD": np.random.rand(10, 20),
        },
        "absolute_default_path": {
            "@NX_class": "NXentry",
            "@default": f"{path}/nexus_entry/nx_process/nx_data",
        },
    }
    nexus_entry["spectrum"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD",
        "@axes": [".", "X"],
        "twoD": np.random.rand(10, 20),
        "twoD@interpretation": "spectrum",
        "twoD@units": "arb. units",
        "errors": np.random.rand(10, 20),
        "X": np.linspace(0, 1, 20),
        "X@units": "nm",
    }
    nexus_entry["image"] = {
        "@NX_class": "NXdata",
        "@signal": "fourD",
        "@axes": [".", ".", "Y", "X"],
        "@SILX_style": json.dumps({"signal_scale_type": "log"}),
        "fourD": np.random.rand(10, 20, 30, 40),
        "fourD@long_name": "Interference fringes",
        "fourD@interpretation": "image",
        "X": np.linspace(0, 1, 40),
        "X@units": "nm",
        "Y": np.linspace(0, 1, 30),
        "Y@units": "deg",
        "Y@long_name": "Angle (degrees)",
    }
    nexus_entry["log_spectrum"] = {
        "@NX_class": "NXdata",
        "@signal": "oneD",
        "@axes": ["X_log"],
        "@SILX_style": json.dumps(
            {
                "signal_scale_type": "log",
                "axis_scale_types": ["log"],
            }
        ),
        "oneD": np.random.rand(100),
        "oneD_errors": np.random.rand(100),
        "X_log": np.linspace(0, 2, 100),
    }
    nexus_entry["spectrum_with_aux"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD",
        "@axes": [".", "X"],
        "@auxiliary_signals": ["secondary", "tertiary_cplx"],
        "twoD": np.random.rand(10, 20),
        "twoD@interpretation": "spectrum",
        "twoD@units": "arb. units",
        "twoD_errors": np.random.rand(10, 20),
        "X": np.linspace(0, 1, 20),
        "X@units": "nm",
        "secondary": np.random.rand(10, 20),
        "tertiary_cplx": np.random.rand(10, 20) + 1j * np.random.rand(10, 20),
        "secondary_errors": np.random.rand(10, 20),
    }
    nexus_entry["image_with_aux"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD",
        "@axes": ["Y", "X"],
        "twoD": np.random.rand(30, 40),
        "twoD@interpretation": "image",
        "X": np.linspace(0, 1, 40),
        "Y": np.linspace(0, 1, 30),
        "auxiliary_signals": ["secondary", "tertiary"],
        "secondary": np.random.rand(30, 40),
        "tertiary": np.random.rand(30, 40),
    }
    nexus_entry["complex_spectrum"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD_cplx",
        "twoD_cplx": np.random.rand(10, 20) + 1j * np.random.rand(10, 20),
        "twoD_cplx@interpretation": "spectrum",
        "auxiliary_signals": ["secondary_cplx", "tertiary_float"],
        "secondary_cplx": np.random.rand(10, 20) + 1j * np.random.rand(10, 20),
        "tertiary_float": np.random.rand(10, 20),
    }
    nexus_entry["complex_image"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD_cplx",
        "@auxiliary_signals": ["secondary_cplx", "tertiary_float"],
        "@axes": [".", "position"],
        "twoD_cplx": np.random.rand(30, 40) + 1j * np.random.rand(30, 40),
        "position": np.linspace(0, 1, 40),
        "secondary_cplx": np.random.rand(30, 40) + 1j * np.random.rand(30, 40),
        "tertiary_float": np.random.rand(30, 40),
    }
    nexus_entry["rgb-image"] = {
        "@NX_class": "NXdata",
        "@signal": "fourD_rgb",
        "@axes": [".", "Y_rgb", "X_rgb"],
        "fourD_rgb": np.random.rand(10, 3, 3, 3),
        "fourD_rgb@interpretation": "rgb-image",
        "fourD_rgb@long_name": "RGB CMY DGW",
        "X_rgb": np.linspace(0, 1, 3),
        "Y_rgb": np.linspace(0, 1, 3),
    }
    nexus_entry["descending-axes"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD",
        "@axes": ["Y_desc", "X_desc"],
        "twoD": np.random.rand(10, 20),
        "X_desc": np.linspace(1, 0, 20),
        "Y_desc": np.linspace(1, 0, 10),
    }
    nexus_entry["scatter"] = {
        "@NX_class": "NXdata",
        "@signal": "scatter_data",
        "@axes": ["X", "Y_scatter"],
        "scatter_data": np.random.rand(100),
        "X": np.random.rand(100),
        "Y_scatter": np.random.rand(100),
    }
    nexus_entry["bigint"] = {
        "@NX_class": "NXdata",
        "@signal": "twoD_bigint",
        "@auxiliary_signals": ["secondary_bigint"],
        "@axes": [".", "X_bigint"],
        "twoD_bigint": np.random.randint(10**15, 10**16, (10, 20)),
        "secondary_bigint": np.random.randint(10**15, 10**16, (10, 20)),
        "X_bigint": np.linspace(0, 1, 20),
    }
    nexus_entry["old-style"] = {
        "@NX_class": "NXdata",
        "twoD": np.random.rand(10, 20),
        "twoD@signal": 1,
        "twoD@axes": "Y:X",
        "X": np.linspace(0, 1, 20),
        "X@units": "nm",
        "Y": np.linspace(0, 1, 10),
        "Y@units": "deg",
        "Y@long_name": "Angle (degrees)",
    }
    nexus_entry["numeric-like"] = {
        "@NX_class": "NXprocess",
        "bool": {
            "@NX_class": "NXdata",
            "@signal": "twoD_bool",
            "@auxiliary_signals": ["secondary_bool"],
            "twoD_bool": np.random.choice([False, True], (10, 20)),
            "secondary_bool": np.random.choice([False, True], (10, 20)),
        },
        # enum not supported by h5py
        # "enum": {
        #     "@NX_class": "NXdata",
        #     "@signal": "twoD_enum",
        #     "@auxiliary_signals": ["secondary_enum"],
        #     "twoD_enum": np.random.randint(0, 3, (10, 20)),
        #     "secondary_enum": np.random.randint(0, 3, (10, 20)),
        # },
    }
    nexus_entry["default_slice"] = {
        "@NX_class": "NXdata",
        "@signal": "fourD",
        "@default_slice": ["1", ".", "2", "."],
        "fourD": np.random.rand(10, 20, 30, 40),
    }

    entry["nexus_note"] = {
        "@NX_class": "NXnote",
        "data": json.dumps(
            {
                "energy": 10.2,
                "geometry": {"dist": 0.1, "rot": 0.074},
            }
        ),
        "type": "application/json",
    }

    entry["nexus_no_default"] = {
        "@NX_class": "NXprocess",
        "ignore_me": {
            "@NX_class": "NXentry",
        },
        "spectrum": {
            "@NX_class": "NXdata",
            "@signal": "oneD",
            "oneD": np.random.rand(100),
        },
    }

    entry["nexus_malformed"] = {
        "default_not_found": {
            "@default": "/test",
        },
        "no_signal": {
            "@NX_class": "NXdata",
        },
        "signal_not_found": {
            "@NX_class": "NXdata",
            "@signal": "unknown",
        },
        "signal_not_dataset": {
            "@NX_class": "NXdata",
            "@signal": "some_group",
            "some_group": {},
        },
        "signal_old-style_not_dataset": {
            "@NX_class": "NXdata",
            "some_group": {},
            "some_group@signal": 1,
        },
        "signal_not_array": {
            "@NX_class": "NXdata",
            "@signal": "some_scalar",
            "some_scalar": 0,
        },
        "signal_not_numeric": {
            "@NX_class": "NXdata",
            "@signal": "oneD_str",
            "oneD_str": ["a", "b", "c"],
        },
        "interpretation_unknown": {
            "@NX_class": "NXdata",
            "@signal": "fourD",
            "fourD": np.random.rand(10, 20, 30, 40),
            "fourD@interpretation": "unknown",
        },
        "rgb-image_incompatible": {
            "@NX_class": "NXdata",
            "@signal": "oneD",
            "oneD": np.random.rand(100),
            "oneD@interpretation": "rgb-image",
        },
        "default_slice_wrong_length": {
            "@NX_class": "NXdata",
            "@signal": "fourD",
            "@default_slice": ["1", ".", "2"],
            "fourD": np.random.rand(10, 20, 30, 40),
        },
        "default_slice_out_of_bounds": {
            "@NX_class": "NXdata",
            "@signal": "fourD",
            "@default_slice": ["3", ".", "2", "."],
            "fourD": np.random.rand(2, 20, 30, 40),
        },
        "silx_style_unknown": {
            "@NX_class": "NXdata",
            "@signal": "oneD",
            "@axes": ["X"],
            "@SILX_style": json.dumps(
                {
                    "unknown": "log",
                    "signal_scale_type": "invalid",
                    "axes_scale_types": ["invalid"],
                }
            ),
            "oneD": np.random.rand(100),
            "X": np.linspace(0, 1, 100),
        },
        "silx_style_malformed": {
            "@NX_class": "NXdata",
            "@signal": "oneD",
            "@SILX_style": "{",
            "oneD": np.random.rand(100),
        },
        "note_invalid_json": {
            "@NX_class": "NXnote",
            "data": "{foo: 'bar'}",
            "type": "application/json",
        },
        "note_unknown_mime_type": {
            "@NX_class": "NXnote",
            "data": "foo: bar",
            "type": "application/yaml",
        },
    }
    return entry


def main():
    filename = "nexus_sample.h5"
    print("Creating file '%s'..." % filename)
    with h5py.File(filename, "w") as h5:
        dicttonx({"@NX_class": "NXroot", "@default": "default_attribute"}, h5, "/")
        dicttonx(
            default_attribute_entry("/default_attribute"), h5, "/default_attribute"
        )
        dicttonx(
            nxdata_documentation_examples_entry(), h5, "/nxdata_documentation_examples"
        )
        dicttonx(
            nxdata_documentation_snippets_entry(), h5, "/nxdata_documentation_snippets"
        )
        dicttonx(h5web_mock_entry("/h5web_mock"), h5, "/h5web_mock")


if __name__ == "__main__":
    main()
