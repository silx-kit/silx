
Writing HDF5 groups to produce specific plots in silx view
==========================================================

.. image:: img/nxcurve.png
    :width: 50%


In this tutorial, you will write HDF5 groups describing data relationships so that they can be displayed in ``silx view``.

To follow the tutorial, you need to be able to:

- understand the concepts of *group*, *dataset* and *attribute* in HDF5
- use the *h5py* library to write HDF5 data.

You can find some information on these topics at the beginning of the
:doc:`io` tutorial.

NeXus Data Format
-----------------

So that ``silx`` produces the specific plots, the relationships between the different datasets must be described according to the `NeXus standard <https://www.nexusformat.org/>`_.

NeXus is a common data format for neutron, x-ray, and muon science. The NeXus format can be applied to HDF5 groups by adding a ``NX_class`` attribute to implement `NeXus base classes <https://manual.nexusformat.org/classes/base_classes/index.html>`_.

NeXus files are often structured this way:

- The file root is defined as **NXroot** (``NX_class`` attribute set to ``NXroot``, often omitted)
- The file root contains **NXentry** groups (``NX_class`` attribute set to ``NXentry``)
- The NXentry groups contain **NXdata** groups (``NX_class`` attribute set to ``NXdata``).

The **NXdata** group is the group we will write. It describes the data to be plotted and the related dimension scales.

Creating a basic NXdata
-----------------------

.. note::

    From now on, attributes will be written with a leading ``@``.

To create a ``NXdata`` group, create an HDF5 group with an attribute ``@NX_class`` set to ``NXdata``.

The datasets needed for the plot will be added to the group while the attributes of the group define the relationship between the datasets.

Signal
++++++

The ``@signal`` attribute of the ``NXdata`` group contains the name of the dataset inside the group containing the data to be plotted.

A ``NXdata`` group needs at least a **signal** dataset to produce a plot.

Axes
++++

Even though this is optional, ``NXdata`` can also contain datasets to be used as *dimension scales*. The list of those dataset names should be added to the ``@axes`` attribute of the ``NXdata`` group.

The number of names should match the number of dimensions of the signal data. If there is no matching dataset for a particular dimension, a ``"."`` should be written in place of the dataset name.

The axis dataset should be a 1D dataset, whose length matches the size of the corresponding signal dimension.

Auxiliary signals
+++++++++++++++++

The ``NXdata`` group can optionally contain datasets that should be displayed along the main signal, called **auxiliary signals**.

The corresponding dataset names should be added to the ``@auxiliary_signals`` attribute of the ``NXdata`` group.

There can be any number of auxiliary signals (even none) but all must have the same shape as the main signal (whose name is contained in the ``@signal`` attribute).

Default plots
+++++++++++++

``silx view`` will produce a plot when opening a NXdata group. This plot can also be produced when opening the parent of the NXdata group (often ``NXentry``) by setting the ``NXdata`` group as the default group.

For this, the parent group should have a ``@default`` attribute that contains the relative path to the ``NXdata`` group (e.g. its name if the ``NXdata`` group is a direct child).

Default groups can be defined up to the file root to produce a plot when opening the file in ``silx view``.

See the section below for some examples.

Examples of NXdata with h5py
----------------------------

A curve
+++++++

The simplest NXdata example would be a 1D signal to be plotted as a curve.

.. code-block:: python

    import h5py
    import numpy

    with h5py.File("myfile.h5", "w") as h5file:
        # It is mandatory to have at least one NXentry
        # https://manual.nexusformat.org/classes/base_classes/NXentry.html#nxentry
        entry = h5file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        nxdata = entry.create_group("my_curve")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "y"
        ds = nxdata.create_dataset("y", data=numpy.array([0.1, 0.2, 0.15, 0.44]))

        # Add units (optional)
        ds.attrs["units"] = "mm"

        # Add an axis (optional)
        nxdata.attrs["axes"] = ["x"]
        ds = nxdata.create_dataset("x", data=numpy.array([101.1, 101.2, 101.3, 101.4]))
        ds.attrs["units"] = "deg"

        # Add additional curves with auxiliary signals (optional)
        nxdata.create_dataset("y2", data=numpy.array([0.2, 0.4, 0.3, 0.88]))
        nxdata.create_dataset("y3", data=numpy.array([1, 0.7, 0.4, 0.1]))
        nxdata.attrs["auxiliary_signals"] = ["y2", "y3"]


This will give the following plot in ``silx view``:

.. image:: img/nxcurve.png


A 2D scatter plot
+++++++++++++++++

A scatter plot is the only case for which we can have more axes than
there are signal dimensions. The signal is 1D, and there can be any
number of axes with the same number of values as the signal.

Silx supports 2D and 3D scatters.

To generate a 2D scatter plot:

.. code-block:: python

    import h5py
    import numpy

    with h5py.File("myfile.h5", "w") as h5file:
        entry = h5file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        nxdata = entry.create_group("my_scatter")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "values"
        nxdata.attrs["axes"] = ["x", "y"]
        nxdata.create_dataset("values", data=numpy.array([0.1, 0.2, 0.15, 0.44]))
        nxdata.create_dataset("x", data=numpy.array([101.1, 101.2, 101.3, 101.4]))
        nxdata.create_dataset("y", data=numpy.array([2, 4, 6, 8]))

Again, in ``silx view``:

.. image:: img/nxscatter.png

A 3D scatter plot
+++++++++++++++++

A 3D scatter plot can be generated likewise

.. code-block:: python

    import h5py
    import numpy

    with h5py.File("myfile.h5", "w") as h5file:
        entry = h5file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        nxdata = entry.create_group("my_3D_scatter")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "values"
        nxdata.attrs["axes"] = ["x", "y", "z"]
        nxdata.create_dataset("values", data=numpy.array([0.1, 0.2, 0.15, 0.44]))
        nxdata.create_dataset("x", data=numpy.array([101.1, 101.2, 101.3, 101.4]))
        nxdata.create_dataset("y", data=numpy.array([2, 4, 6, 8]))
        nxdata.create_dataset("z", data=numpy.array([-10, -8, -6, -4]))

.. image:: img/nxscatter3d.png

.. note::

    When producing the 3D scatter plot, ``silx view`` will scale the markers according to the first auxiliary signal.

    .. code-block:: python

        nxdata.create_dataset("sizes", data=numpy.array([2, 4, 4, 2]))
        nxdata.attrs["auxiliary_signals"] = ["sizes"]

    If there is no auxiliary signal, all markers will have the same size.

A stack of images
+++++++++++++++++

In case of a stack, the first axis corresponds to stack indices and may not be represented by a dataset.

In this case, we use ``.`` as a placeholder in ``axes`` for this dimension:

.. code-block:: python

    import h5py
    import numpy

    with h5py.File("myfile.h5", "w") as h5file:
        entry = h5file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        nxdata = entry.create_group("image")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "frames"
        nxdata.attrs["axes"] = [".", "y", "x"]
        # 2 frames of size 3 rows x 4 columns
        signal = nxdata.create_dataset(
            "frames",
            data=numpy.array(
                [
                    [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7], [1.8, 1.9, 2.0, 2.1]],
                    [[8.0, 8.1, 8.2, 8.3], [8.4, 8.5, 8.6, 8.7], [8.8, 8.9, 9.0, 9.1]],
                ]
            ),
        )
        x = nxdata.create_dataset("x", data=numpy.array([0.1, 0.2, 0.3, 0.4]))
        x.attrs["units"] = "mm"
        y = nxdata.create_dataset("y", data=numpy.array([2, 4, 6]))
        y.attrs["units"] = "s"

.. image:: img/nximage.png

.. note::

    If the image axes have the same ``units`` or both have no ``units``, the image aspect ratio will be kept.

    In the example above, the two axes ``x`` and ``y`` have different ``units`` so that the image aspect ratio is not conserved.
