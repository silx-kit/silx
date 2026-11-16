.. _silx-view-plot-selection:

Custom plot selection
---------------------

.. grid :: 1 2 2 2

   .. grid-item-card::
      :columns: 12
      :text-align: center

      .. figure:: img/silx-view-plotSelection.png
         :align: center

      Plot custom curves from selected 1D datasets.

How to access
.............

.. grid :: 1 2 2 2

   .. grid-item-card::
      :columns: 12
      :text-align: center

      .. figure:: img/silx-view-plotMenu.png
         :align: center

      Open the 'Plot selection' window (View menu -> Plot selection)

      .. figure:: img/silx-view-dropPlot.png
         :align: center

      The "Plot selection" window opens: You can plot custom curves by dragging and dropping 1D datasets from "silx view" to "Plot selection" window's "X" and "Y" fields.

What it does
............

The *Plot selection* window plots 1D datasets against each other, whatever the
file or the HDF5 group they come from. This is useful to compare curves, or to
plot a counter against a motor position, without having to write a script or to
create an NXdata group.

The window is made of two parts:

* **The dataset list** (left):

  * **X**: the dataset used as the abscissa of all the curves (at most one).
    Its name is used as the X axis label.
  * **Y**: the datasets plotted as curves (as many as needed). The icon in
    front of each dataset shows the style and color of its curve in the plot.

  Hovering a dataset shows its name, its HDF5 path and its file name.

* **The plot** (right): a standard silx 1D plot, with its toolbar to zoom,
  change the axes scale (e.g. log scale), fit, save or print the curves...

Changing the X dataset updates all the curves. If no X dataset is selected,
the Y curves are plotted against their index.
If the X and Y datasets do not have the same length, the curve is truncated to
the shortest one.

.. warning::

   The selection is not saved: it is lost when *silx view* is closed.

Demo
....

.. raw:: html

   <video controls loop muted width="100%">
      <source src="../_static/silx-view-plotSelection-demo.webm" type="video/webm">
   </video>

Show / hide the window
......................

The window is hidden when *silx view* starts.

* **Show it** from the menu bar: *Views* → *Plot selection*.
  The menu entry is checkable: it is checked while the window is visible.
* **Hide it** by unchecking *Views* → *Plot selection* again, or by closing
  the window. The menu entry is kept in sync with the window.
* The window is also shown automatically when a dataset is sent to it from the
  tree context menu (see :ref:`plot-selection-context-menu`).

Hiding the window does not clear it: the selected datasets are kept and
displayed again the next time the window is shown.

Select the datasets
...................

Only **1D datasets** can be used. Other datasets are rejected on drop.

Drag and drop in the dataset list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drag a 1D dataset from the *silx view* HDF5 tree and drop it in the dataset
list of the *Plot selection* window:

* dropped on the **X** row: the dataset is used as X values (it replaces the
  previous X dataset, if any);
* dropped anywhere else (e.g. on the *Drop a 1D dataset* placeholder below
  **Y**): the dataset is added as a new Y curve.

The targeted row is highlighted while dragging.

Drag and drop on the plot
~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets can also be dropped directly on the plot:

* dropped **below the plot area** (on the X axis): sets the X values;
* dropped **inside the plot area**: adds a Y curve.

An overlay shows which area the dataset will be dropped in.

.. _plot-selection-context-menu:

From the context menu
~~~~~~~~~~~~~~~~~~~~~

Right-click on a 1D dataset in the *silx view* HDF5 tree:

* *Set X values of plot selection*
* *Add Y values to plot selection*

The *Plot selection* window is shown if it was hidden.

Remove datasets
...............

* Click the red cross next to a dataset to remove it from the selection.
* Use the *Clear All* button of the toolbar to remove all datasets.
