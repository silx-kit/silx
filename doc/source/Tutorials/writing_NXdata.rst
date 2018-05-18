
Writing NXdata
==============

This tutorial explains how to write a *NXdata* group into a HDF5 file.

A basic knowledge of the HDF5 file format, including understanding
the concepts of *group*, *dataset* and *attribute*,
is a prerequisite for this tutorial. You should also be able to read
a python script using the *h5py* library to write HDF5 data.
You can find some information on these topics at the beginning of the
:doc:`io` tutorial.

Definitions
-----------

NeXus Data Format
+++++++++++++++++

NeXus is a common data format for neutron, x-ray, and muon science.
It is being developed as an international standard by scientists and programmers
representing major scientific facilities in order to facilitate greater
cooperation in the analysis and visualization of neutron, x-ray, and muon data.

It uses the HDF5 format, adding additional rules and structure to help
people and software understand how to read a data file.

The name of a group in a NeXus data file can be any string of characters,
but it must have a `NX_class` attribute defining a
`*class type* <http://download.nexusformat.org/doc/html/introduction.html#important-classes>`_.

Examples of such classes are:

 - *NXroot*: root group of the file (may be implicit, if the can be `NX_class` attribute is omitted)
 - *NXentry*: describes a measurement; it is mandatory that there is at least one
   group of this type in the NeXus file
 - *NXsample*: contains information pertaining to the sample, such as its chemical composition,
   mass, and environment variables (temperature, pressure, magnetic field, etc.)
 - *NXinstrument*: encapsulates all the instrumental information that might be relevant to a measurement
 - *NXdata*: describes the plottable data and related dimension scales

You can find all the specifications about the NeXus format on the
`nexusformat.org website <https://www.nexusformat.org/>`_. The rest of this tutorial will
focus exclusively on *NXdata*.

NXdata groups
+++++++++++++

NXdata describes the plottable data and related dimension scales.

It is mandatory that there is at least one NXdata group in each NXentry group.
Note that the variable and data can be defined with different names.
The `signal` and `axes` attributes of the group define which items
are plottable data and which are dimension scales, respectively.

In the case of a curve, for instance, you would have a 1D signal
dataset (*y* values) and optionally another 1D signal of identical
size as axis (*x* values). In the case of an image, you would have
a 2D dataset as signal and optionally two 1D datasets to scale
the X and Y axes.

A NXdata group should define all the information needed to
provide a sensible plot, including axis labels and a plot title.
It can also include additional metadata such as standard deviations
of data values, or errors an axes.

.. note::


    The NXdata specification evolved slightly over the course of time.
    The `complete documentation for the *NXdata* class
    <http://download.nexusformat.org/doc/html/classes/base_classes/NXdata.html>`_ mentions
    older rules that you will probably have to take into account
    if you intend to write a program that reads NeXus files.

    If you only need to write such files and only need to read back files
    you have yourself written, you should adhere to the most recent rules.
    We will only mention these most recent specifications in this tutorial.


NXdata examples
---------------

A simple curve
++++++++++++++




