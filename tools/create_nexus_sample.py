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


if __name__ == "__main__":
    main()
