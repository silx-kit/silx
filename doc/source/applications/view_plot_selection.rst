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

The window is hidden when *silx view* starts. To show it, select *Views* → *Plot selection* from the menu bar.

Select the datasets
...................

Only **1D datasets** can be used. Other datasets are rejected on drop.

Drag and drop
~~~~~~~~~~~~~

Drag a 1D dataset from the *silx view* HDF5 tree and drop it either in the
dataset list or on the plot of the *Plot selection* window:

* to use it as **X** values (it replaces the previous X dataset, if any), drop it:

  * on the **X** row of the dataset list, or
  * **below the plot area** (on the X axis);

* to add it as a new **Y** curve, drop it:

  * on the **Y** row of the dataset list, or
  * **inside the plot area**.

While dragging, the targeted row of the dataset list is highlighted, and an
overlay shows the targeted area of the plot.

.. _plot-selection-context-menu:

From the context menu
~~~~~~~~~~~~~~~~~~~~~

Right-click on a 1D dataset in the *silx view* HDF5 tree:

* *Set X values of plot selection*
* *Add Y values to plot selection*

The *Plot selection* window is shown if it was hidden.
