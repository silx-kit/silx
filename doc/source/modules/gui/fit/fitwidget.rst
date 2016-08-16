
.. currentmodule:: silx.gui.fit

:mod:`FitWidget`
================

.. automodule:: silx.gui.fit.FitWidget

Usage example
-------------

.. code-block:: python

    import numpy
    from silx.gui.fit import FitWidget
    from silx.math.fit.functions import sum_gauss
    from silx.gui import qt

    x = numpy.arange(2000).astype(numpy.float)
    constant_bg = 3.14

    # gaussian parameters: height, position, fwhm
    p = numpy.array([1000, 100., 30.0,
                     500, 300., 25.,
                     1700, 500., 35.,
                     750, 700., 30.0,
                     1234, 900., 29.5,
                     302, 1100., 30.5,
                     75, 1300., 210.])
    y = sum_gauss(x, *p) + constant_bg

    a = qt.QApplication([])
    a.lastWindowClosed.connect(a.quit)
    w = FitWidget(enableconfig=1, enablestatus=1, enablebuttons=1)
    w.setdata(x=x, y=y)
    w.show()
    a.exec_()

.. |imgFitWidget3| image:: img/fitwidget3.png
   :width: 400px
   :align: middle

Executing this code, then selecting a constant background, clicking
the estimate button, then the fit button, shows the following result:

    |imgFitWidget3|


The following example shows how to define a custom fit function.

.. code-block:: python

    from silx.math.fit import FitManager
    from silx.gui import qt
    from silx.gui.fit import FitWidget

    def linearfun(x, a, b):
        return a * x + b

    # create synthetic data for the example
    x = list(range(0, 100))
    y = [linearfun(x_, 2.0, 3.0) for x_ in x]

    # we need to create a custom fit manager and add our theory
    myfitmngr = FitManager()
    myfitmngr.setdata(x, y)
    myfitmngr.addtheory("my linear function",
                        function=linearfun,
                        parameters=["a", "b"])

    a = qt.QApplication([])

    # our fit widget can now use our custom fit manager
    fw = FitWidget(fitmngr=myfitmngr)
    fw.show()

    a.exec_()

API
---

.. currentmodule:: silx.gui.fit.FitWidget

.. autoclass:: FitWidget
   :members: __init__, setdata
