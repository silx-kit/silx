.. currentmodule:: silx.gui.plot

:mod:`actions`: Actions for PlotWidget
===========================================

.. currentmodule:: silx.gui.plot.actions

The :class:`PlotAction` is a base class used to define plot actions.

Plot actions serve to add menu items or toolbar items that are associated with a method
that can interact with the associated :class:`.PlotWidget`.

For an introduction to creating custom plot actions, see :doc:`examples`.

actions API
-----------

Actions are divided into the following sub-modules:


.. toctree::
   :maxdepth: 1

   control.rst
   medfilt.rst
   fit.rst
   histogram.rst
   io.rst

.. autoclass:: silx.gui.plot.actions.PlotAction
