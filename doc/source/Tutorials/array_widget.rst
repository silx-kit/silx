
.. currentmodule:: silx.gui.widgets.ArrayTableWidget

ArrayTableWidget
================

:class:`ArrayTableWidget` is a widget designed to visualize arrays
numpy arrays or h5py datasets).

3D example
----------

:class:`ArrayTableWidget` is a widget designed to visualize arrays
numpy arrays or h5py datasets).

Let's look at a simple usage example::
    >>> from silx.gui import qt
    >>> from silx.gui.widgets.ArrayTableWidget import ArrayTableWidget
    >>> import numpy
    >>> array = numpy.arange(1000)
    >>> array.shape = (5, 10, 20)
    >>> a = qt.QApplication([])
    >>> w = ArrayTableWidget()
    >>> w.setArrayData(array, labels=True)
    >>> w.show()

.. |imgArray0| image:: img/arraywidget3D_0.png
   :height: 300px
   :align: middle

|imgArray0|

We get a widget that allows us to see a *slice*, or a *frame*, of 2D data
with 10 lines and 20 columns, in a 3D array (5 x 10 x 20).
The column index corresponds to the last dimension of the array, and the
row index corresponds to the second to last dimension. The first index can be browsed
using a slider, icons or a text entry for acces to any given slice among the 5 available.

The parameter ``labels=True`` of :meth:`setArrayData` causes the browser to be labbeled
*Dimension 0*.

If we want to see slices in different perspective, we can use
:meth:`ArrayTableWidget.setPerspective`. The perspective is defined the list
of dimensions that are not represented in the frame, orthogonal to it.
For a 3D array, there 3 possible perspectives: *[0, ]* (the default perspective we are
currently using), *[1, ]* and *[2, ]*.

Lets change the perspective::
    >>> w.setPerspective([1])

.. |imgArray1| image:: img/arraywidget3D_1.png
   :height: 300px
   :align: middle

|imgArray1|

What we see now is a frame of *5 x 20* valuesarra, and the browser now browses the second dimension
to select one of 10 available frames. The label is updated accordingly to *Dimension 1*.

To select a different frame programatically, without using the browser, you can
use the :meth:`ArrayTableWidget.setIndex`. To select the 9-th frame, use::
    >>> w.setIndex([8])

More dimensions
---------------

This widget can be used for arrays of any numbers of dimensions. Let's create
a 5-dimensional array and display it::
    >>> array = numpy.arange(10000)
    >>> array.shape = (5, 2, 10, 5, 20)
    >>> w.setArrayData(array, labels=True)

.. |imgArray2| image:: img/arraywidget5D_0.png
   :height: 300px
   :align: middle

|imgArray2|

We now have 3 frames browsers, on for each of the orthogonal dimensions.

Let's look at a frame whose axes are along the second
and the fourth dimension, by setting the orthogonal axes to the first,
third and fifth dimensions::
   >>> w.setPerspective([0, 2, 4])

.. |imgArray3| image:: img/arraywidget5D_1.png
   :height: 300px
   :align: middle

|imgArray3|


Listing all the orthogonal dimensions might not feel very natural for arrays
with more than 3 or 4 dimensions.
Fortunately, you can use the opposite approach of defining the two axes
parallel to the frame, using :meth:`ArrayTableWidget.setFrameAxes`::
   >>> w.setFrameAxes(row_axis=1, col_axis=3)

This achieves the exact same result.

.. note::

    Currently you cannot switch the row and column axes. The row axis
    is always the lowest free dimension and the column axis is the
    highest one with the current implementation.
    So setting ``w.setFrameAxes(row_axis=3, col_axis=1)`` would not modify
    the table axes.

To select a frame programmaticaly, you can again use :meth:`setFrameIndex`.
This time you must provide 3 unique indices::
    >>> w.setIndex([2, 5, 14])

The 3 indices relate to the first, third and fifth dimensions.

The frame index must always be defined as indices on the orthogonal axes/dimensions,
as defined by the *perspective*.





