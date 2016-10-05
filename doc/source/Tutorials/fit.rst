

Fit tools
---------

.. _leastsq-tutorial:

Using :func:`leastsq`
+++++++++++++++++++++

Running an iterative fit with :func:`leastsq` involves the following steps:

    - designing a fit model function that has the signature ``f(x, ...)``,
      where ``x`` is an array of values of the independant variable and all
      remaining parameters are the parameters to be fitted
    - defining the sequence of initial values for all parameters to be fitted.
      You can usually start with ``[1., 1., ...]`` if you don't know a better
      estimate. The algorithm is robust enough to converge to a solution most
      of the time.
    - setting constraints (optional)

Let's demonstrate this process in a short example, using synthetic data.
We create an array of synthetic data using a polynomial of degree 4, and try
to use :func:`leastsq` to find back the same parameters used to create the
synthetic data.

.. code-block:: python

   import numpy
   from silx.math.fit import leastsq

   # create some synthetic polynomial data
   x = numpy.arange(1000)
   y = 2.4 * x**4 - 10 * x**3 + 15.2 * x**2 - 24.6 * x + 150

   # define our fit function: a generic polynomial of degree 4
   def poly4(x, a, b, c, d, e):
       return a * x**4 + b * x**3 + c * x**2 + d * x + e

   # The fit is an iterative process that requires an initial
   # estimation of the parameters. Let's just use 1s.
   initial_parameters = numpy.array([1., 1., 1., 1., 1.])

   # Run fit
   fitresult = leastsq(model=poly4,
                       xdata=x,
                       ydata=y,
                       p0=initial_parameters,
                       full_output=True)

   # leastsq with full_output=True returns 3 objets
   optimal_parameters, covariance, infodict = fitresult
   # the first object is an array with the fitted parameters
   a, b, c, d, e = optimal_parameters

   print("Fit took %d iterations" % infodict["niter"])
   print("Reduced chi-square: %f" % infodict["reduced_chisq"])
   print("Theoretical parameters:\n\t" +
         "a=2.4, b=-10, c=15.2, d=-24.6, e=150")
   print("Optimal parameters for y2 fitting:\n\t" +
         "a=%f, b=%f, c=%f, d=%f, e=%f" % (a, b, c, d, e))

The output of this program is::

   Fit took 35 iterations
   Reduced chi-square: 682592.670690
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y2 fitting:
       a=2.400000, b=-9.999665, c=14.970422, d=31.683448, e=-3216.131136

We can see that this fit result is poor. In particular, parameters ``d`` and ``e``
are very poorly fitted.
This is most likely due to numerical rounding errors, as we are dealing with
very large values in our ``y`` array. If you limit the ``x`` range to deal with
smaller ``y`` values, the fit result becomes perfect. In our example, replacing ``x``
with::

    x = numpy.arange(1000)

produces the following result::

   Fit took 9 iterations
   Reduced chi-square: 0.000000
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y2 fitting:
       a=2.400000, b=-10.000000, c=15.200000, d=-24.600000, e=150.000000

But let's revert back to our initial ``x`` range (0 -- 1000) and try to improve
the result using a different approach. The :func:`leastsq` functions provides
a way to set constraints on parameters. You can for instance assert that a given
parameter must remain equal to it's initial value, or define an acceptable range
for it to vary, or decide that a parameter must be equal to another parameter
multiplied by a certain factor. This is very useful in cases in which you have
enough knowledge to make reasonable assumptions on some parameters.

In our case, we will set constraints on ``d`` ann ``e``. We will quote ``d`` to
the range between -25 and -24, and fix ``e`` to 150.

Replace the call to :func:`leastsq` by following lines:

.. code-block:: python

   # Define constraints
   cons = [[0, 0, 0],          # a: no constraint
           [0, 0, 0],          # b: no constraint
           [0, 0, 0],          # c: no constraint
           [2, -25., -23.],    # -25 < d < -24
           [3, 0, 0]]          # e is fixed to initial value
   fitresult = leastsq(poly4, x, y,
                       # initial values must be consistent with constraints
                       p0=[1., 1., 1., -24., 150.],
                       constraints=cons,
                       full_output=True)
The output of this is::

   Fit took 100 iterations
   Reduced chi-square: 3.749280
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters:
       a=2.400000, b=-9.999999, c=15.199648, d=-24.533014, e=150.000000

The chi-square value is much improved and the results are much better, at the
cost of mose iterations.

.. _fitmanager-tutorial:

Using :class:`FitManager`
+++++++++++++++++++++++++

bar


.. _fitwidget-tutorial:

Using :class:`FitWidget`
++++++++++++++++++++++++

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
    w.setData(x=x, y=y)
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
    myfitmngr.setData(x, y)
    myfitmngr.addtheory("my linear function",
                        function=linearfun,
                        parameters=["a", "b"])

    a = qt.QApplication([])

    # our fit widget can now use our custom fit manager
    fw = FitWidget(fitmngr=myfitmngr)
    fw.show()

    a.exec_()
