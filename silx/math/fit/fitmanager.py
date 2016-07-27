# coding: utf-8
# /*#########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##########################################################################*/
"""
This module provides a tool to perform advanced fitting. The actual fit relies
on :func:`silx.math.fit.leastsq`.

This module deals with:

    - handling of the model functions (using a set of default functions or
      loading custom user functions)
    - handling of estimation function, that are used to determine the number
      of parameters to be fitted for functions with unknown number of
      parameters (such as the sum of a variable number of gaussian curves),
      and find reasonable initial parameters for input to the iterative
      fitting algorithm
    - handling of custom  derivative functions that can be passed as a
      parameter to  :func:`silx.math.fit.leastsq`
    - removal of constant and linear background signal prior to performing the
      actual fit

"""
from collections import OrderedDict
import logging
import numpy
import os
import sys

from .filters import strip
from .leastsq import leastsq
from .fittheory import FitTheory


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "26/07/2016"

_logger = logging.getLogger(__name__)


class FitManager(object):
    """
    Multi-peak fitting functions manager

    :param x: Abscissa data. If ``None``, :attr:`xdata` is set to
        ``numpy.array([0.0, 1.0, 2.0, ..., len(y)-1])``
    :type x: Sequence or numpy array or None
    :param y: The dependant data ``y = f(x)``. ``y`` must have the same
        shape as ``x`` if ``x`` is not ``None``.
    :type y: Sequence or numpy array or None
    :param sigmay: The uncertainties in the ``ydata`` array. These are
        used as weights in the least-squares problem.
        If ``None``, the uncertainties are assumed to be 1.
    :type sigmay: Sequence or numpy array or None

    :param auto_fwhm: Flag to enable or disable automatic estimation of
        the peaks' full width at half maximum.
    :param fwhm_points:
    :param auto_scaling: Enable on disable auto-scaling based on y data.
        If ``True``, init argument ``yscaling`` is ignored and
        :attr:`fitconfig` ``['Yscaling']`` is calculated based on data.
        If ``False``, ``yscaling`` is used.
    :param yscaling: Scaling parameter for ``y`` data.
    :param sensitivity: Sensitivity value used for by peak detection
        algorithm. To be detected, a peak must have an amplitude greater
        than ``σn * sensitivity`` (where ``σn`` is an estimated value of
        the standard deviation of the noise).
    """
    # TODO: document following attributes
    # Data attributes:
    #
    #  - :attr:`xdata0`, :attr:`ydata0` and :attr:`sigmay0` store the initial data
    #    and uncertainties. These attributes are not modified after
    #    initialization.
    #  - :attr:`xdata`, :attr:`ydata` and :attr:`sigmay` store the data after
    #    removing values where :attr:`xdata < xmin` or :attr:`xdata > xmax`.
    #    These attributes may be modified at a latter stage by filters.

    def __init__(self, x=None, y=None, sigmay=None, auto_fwhm=True, fwhm_points=8,
                 auto_scaling=False, yscaling=1.0, sensitivity=2.5):
        """
        """
        self.fitconfig = {}
        """Dictionary of fit configuration parameters.

        Keys are:

            - 'fittheory': name of the function used for fitting peaks
            - 'fitbkg': name of the function used for fitting a low frequency
              background signal
            - 'AutoFwhm': Flag to enable or disable automatic estimation of
              the peaks' full width at half maximum.
            - 'FwhmPoints': default full width at half maximum value for the
              peaks'. Ignored if ``AutoFwhm==True``.
            - 'AutoScaling': Flag to enable or disable automatic estimation of
              'Yscaling' (using an inverse chi-square value)
            - 'Yscaling': Scaling factor for the data
            - 'Sensitivity': Sensitivity parameter for the peak detection
              algorithm (:func:`silx.math.fit.peak_search`)
        """

        self.fitconfig['AutoFwhm'] = auto_fwhm
        self.fitconfig['FwhmPoints'] = fwhm_points
        self.fitconfig['AutoScaling'] = auto_scaling
        self.fitconfig['Yscaling'] = yscaling
        self.fitconfig['Sensitivity'] = sensitivity
        self.fitconfig['fitbkg'] = 'No Background'
        self.fitconfig['fittheory'] = None

        self.theories = OrderedDict()
        """Dictionary of fit theories, defining functions to be fitted
        to individual peaks.

        Keys are descriptive theory names (e.g "Gaussians" or "Step up").
        Values are :class:`silx.math.fit.fittheory.FitTheory` objects with
        the following attributes:

            - *"function"* is the fit function for an individual peak
            - *"parameters"* is a sequence of parameter names
            - *"estimate"* is the parameter estimation function
            - *"configure"* is the function returning the configuration dict
              for the theory in the format described in the :attr:` fitconfig`
              documentation
            - *"derivative"* (optional) is a custom derivative function, whose
              signature is described in the documentation of
              :func:`silx.math.fit.leastsq.leastsq`
              (``model_deriv(xdata, parameters, index)``).
            - *"description"* is a description string
        """

        self.bkgdict = OrderedDict((
             ('No Background', {
                 'description': "No background function",
                 'function': self.bkg_none,
                 'parameters': [],
                 'estimate': None}),
             ('Constant', {
                 'description': "Constant background",
                 'function': self.bkg_constant,
                 'parameters': ['Constant'],
                 'estimate': self.estimate_builtin_bkg}),
             ('Linear', {
                 'description': "Linear background, parameters 'Constant' and 'Slope'",
                 'function': self.bkg_linear,
                 'parameters': ['Constant', 'Slope'],
                 'estimate': self.estimate_builtin_bkg}),
             ('Internal', {
                 'description': "Background based on strip filter\n" +
                                "Parameters 'Curvature', 'Iterations' and 'Constant'",
                 'function': self.bkg_internal,
                 'parameters': ['Curvature', 'Iterations', 'Constant'],
                 'estimate': self.estimate_builtin_bkg})))
        """Dictionary of background functions.

        Keys are descriptive theory names (e.g "Constant" or "Linear").
        Values are dictionaries with the following items:

          - *description* is an optional description string, which can be used
            for instance as a tooltip message in a GUI.

          - *function* is a callable function with the signature ``function(x, params) -> y``
            where params is a sequence of parameters.

          - *parameters* is a sequence of parameter names (e.g. could be
            for a linear function ``["constant", "slope"]``).

          - *estimate* is a function to compute initial values for parameters.
            It should have the following signature:
            ``f(x, y, bg_data, xscaling=1.0, yscaling=None) -> (estimated_param, constraints, bg_data)``

                Parameters:

                - ``x`` is the independant variable, i.e. all the points where
                  the function is calculated
                - ``y`` is the data from which we want to extract the bg
                - ``bg_data`` is the background data, usually extracted from ``y``
                  using a strip filter.
                - ``xscaling`` is an optional scaling factor applied to the ``x``
                  array
                - ``yscaling`` is an optional scaling factor applied to the ``y``
                  array

                Return values:

                - ``estimated_param`` is a list of estimated values for each
                  background parameter.
                - ``constraints`` is a 2D sequence of dimension ``(n_parameters, 3)``

                  See explanation about 'constraints' in :attr:`fit_results`
                  documentation.
                - ``bg_data`` is the background data extracted from the signal
                  by the estimation function.
        """

        # TODO:  document following attributes
        self.bkg_internal_oldx = numpy.array([])
        self.bkg_internal_oldy = numpy.array([])
        self.bkg_internal_oldpars = [0, 0]
        self.bkg_internal_oldbkg = numpy.array([])

        self.setdata(x, y, sigmay)

        self.parameter_names = []
        """This list stores all fit parameter names: background function
        parameters and fit function parameters for every peak. It is filled
        in :meth:`estimate`.

        It is the responsibility of the estimate function defined in
        :attr:`theories` to determine how many parameters there will be,
        based on how many peaks it detects and how many parameters are needed
        to fit an individual peak.
        """

        self.fit_results = []
        """This list stores detailed information about all fit parameters.
        It is initialized in :meth:`estimate` and completed with final fit
        values in :meth:`startfit`.

        Each fit parameter is stored as a dictionary with following fields:

            - 'name': Parameter name.
            - 'estimation': Estimated value.
            - 'group': Group number. Group 0 corresponds to the background
              function parameters. Group ``n`` (for ``n>0``) corresponds to
              the fit function parameters for the n-th peak.
            - 'code': Constraint code

                - 0 - FREE
                - 1 - POSITIVE
                - 2 - QUOTED
                - 3 - FIXED
                - 4 - FACTOR
                - 5 - DELTA
                - 6 - SUM

            - 'cons1':

                - Ignored if 'code' is FREE, POSITIVE or FIXED.
                - Min value of the parameter if code is QUOTED
                - Index of fitted parameter to which 'cons2' is related
                  if code is FACTOR, DELTA or SUM.

            - 'cons2':

                - Ignored if 'code' is FREE, POSITIVE or FIXED.
                - Max value of the parameter if QUOTED
                - Factor to apply to related parameter with index 'cons1' if
                  'code' is FACTOR
                - Difference with parameter with index 'cons1' if
                  'code' is DELTA
                - Sum obtained when adding parameter with index 'cons1' if
                  'code' is SUM

            - 'fitresult': Fitted value.
            - 'sigma': Standard deviation for the parameter estimate
            - 'xmin': Lower limit of the ``x`` data range on which the fit
              was performed
            - 'xmax': Upeer limit of the ``x`` data range on which the fit
              was performed
        """

        self.selectedfunction = None
        """The model function, currently selected ``f(x, ...)``.
        It must take the independent variable as the first argument and the
        parameters to fit as separate remaining arguments.

        The function can be chosen from :attr:`theories` using
        :meth:`settheory`"""

        self.selectedparameters = None
        """List of parameters names for the currently selected theory.
        """

        self.selectedestimate = None
        """Estimation function for the currently selected theory. See
        :meth:`loadtheories` for more documentation on custom derivative
        functions.
        """

        self.selectedconfigure = None
        """Configuration function for the currently selected theory. See
        :meth:`loadtheories` for more documentation on custom configuration
        functions.
        """

        self.selectedderivative = None
        """None (default) or function providing the derivatives of the fitting
        function respect to the fitted parameters.
        It will be called as ``model_deriv(xdata, parameters, index)`` where
        ``parameters`` is a sequence with the current values of the fitting
        parameters, ``index`` is the fitting parameter index for which the
        derivative has to be calculated."""

    ##################
    # Public methods #
    ##################
    def addbackground(self, background, function, parameters, estimate=None,
                      description=None):
        """Add a new background function to dictionary :attr:`bkgdict`.

        :param background: String with the name describing the function
        :param function: Actual function
        :param parameters: Parameters names ['p1','p2','p3',...]
        :param estimate: The initial parameters estimation function if any
        """
        self.bkgdict[background] = {
            'description': description,
            'function': function,
            'parameters': parameters,
            'estimate': estimate}

    def addtheory(self, theory_name, fittheory):
        """Add a new theory to dictionary :attr:`theories`.

        See :meth:`loadtheories` for more information on estimation functions,
        configuration functions and custom derivative functions.

        :param theory_name: String with the name describing the function
        :param fittheory: :class:`FitTheory` object
        :type fittheory: :class:`silx.math.fit.fittheory.FitTheory`
        """
        self.theories[theory_name] = fittheory

    def configure(self, **kw):
        """Configure the current theory by filling or updating the
        :attr:`fitconfig` dictionary.
        Call the custom configuration function, if any. This allows the user
        to modify the behavior of the custom fit function or the custom
        estimate function.

        This methods accepts only named parameters. All ``**kw`` parameters
        are expected to be fields of :attr:`fitconfig` to be updated, unless
        they have a special meaning for the custom configuration function
        defined in ``fitconfig['fittheory']``.

        This method returns the modified config dictionary returned by the
        custom configuration function.
        """
        # inspect **kw to find known keys, update them in self.fitconfig
        for key in self.fitconfig:
            if key in kw:
                self.fitconfig[key] = kw[key]

        # initialize dict with existing config dict
        result = {}
        result.update(self.fitconfig)

        # Apply custom configuration function defined in self.theories.configure
        theory_name = self.fitconfig['fittheory']
        if theory_name in self.theories:
            custom_config_fun = self.selectedconfigure
            if custom_config_fun is not None:
                result.update(custom_config_fun(**kw))

                # Update self.fitconfig with custom config
                for key in self.fitconfig:
                    if key in result:
                        self.fitconfig[key] = result[key]

        # overwrite existing keys with values from **kw in fitconfig
        if "fitbkg" in self.fitconfig:
            self.setbackground(self.fitconfig["fitbkg"])
        if "fittheory" in self.fitconfig["fittheory"]:
            if self.fitconfig["fittheory"] is not None:
                self.settheory(self.fitconfig["fittheory"])

        result.update(self.fitconfig)
        return result

    def dataupdate(self):
        """This method can be updated with a user defined function to
        update data (for instance modify range fo :attr:`xdata`,
        :attr:`ydata` and :attr:`sigmay` when user zooms in or out in a GUI
        plot).

        It is called at the beginning of :meth:`estimate` and
        :meth:`startfit`.

        By default, it does nothing.
        """
        pass

    def estimate(self, callback=None):
        """
        Fill :attr:`fit_results` with an estimation of the fit parameters.

        At first, the background parameters are estimated, if a background
        model has been specified.
        Then, a custom estimation function related to the model function is
        called.

        This process determines the number of needed fit parameters and
        provides an initial estimation for them, to serve as an input for the
        actual iterative fitting performed in :meth:`startfit`.

        :param callback: Optional callback function, conforming to the
            signature ``callback(data)`` with ``data`` being a dictionary.
            This callback function is called before and after the estimation
            process, and is given a dictionary containing the values of
            :attr:`state` (``'Estimate in progress'`` or ``'Ready to Fit'``)
            and :attr:`chisq`.
            This is used for instance in :mod:`silx.gui.fit.FitWidget` to
            update a widget displaying a status message.
        :return: Estimated parameters
        """
        self.state = 'Estimate in progress'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        CONS = {0: 'FREE',
                1: 'POSITIVE',
                2: 'QUOTED',
                3: 'FIXED',
                4: 'FACTOR',
                5: 'DELTA',
                6: 'SUM',
                7: 'IGNORE'}

        # Update data using user defined method
        self.dataupdate()

        xwork = self.xdata
        ywork = self.ydata

        # estimate the background
        bg_params, bg_constraints, bg_data = self.estimate_bkg(xwork, ywork)

        # scaling
        if self.fitconfig['AutoScaling']:
            yscaling = self.guess_yscaling(y=ywork)
        elif self.fitconfig['Yscaling'] is not None:
            yscaling = self.fitconfig['Yscaling']

        # estimate the function
        esti_fun = self.estimate_fun(xwork, ywork, bg_data, yscaling=yscaling)

        fun_esti_parameters = esti_fun[0]
        fun_esti_constraints = esti_fun[1]

        # build the names
        self.parameter_names = []

        fitbgname = self.fitconfig['fitbkg']
        for bg_param_name in self.bkgdict[fitbgname]["parameters"]:
            self.parameter_names.append(bg_param_name)

        param_index, peak_index = 0, 0
        while param_index < len(fun_esti_parameters):
            peak_index += 1
            for fun_param_name in self.selectedparameters:
                self.parameter_names.append(fun_param_name + "%d" % peak_index)
                param_index += 1

        self.fit_results = []
        nb_fun_params_per_group = len(self.selectedparameters)
        group_number = 0
        xmin = min(xwork)
        xmax = max(xwork)
        nb_bg_params = len(bg_params)
        for (pindex, pname) in enumerate(self.parameter_names):
            # First come background parameters
            if pindex < nb_bg_params:
                estimation_value = bg_params[pindex]
                constraint_code = CONS[int(bg_constraints[pindex][0])]
                cons1 = bg_constraints[pindex][1]
                cons2 = bg_constraints[pindex][2]
            # then come peak function parameters
            else:
                fun_param_index = pindex - nb_bg_params

                # increment group_number for each new fitted peak
                if (fun_param_index % nb_fun_params_per_group) == 0:
                    group_number += 1

                estimation_value = fun_esti_parameters[fun_param_index]
                constraint_code = CONS[int(fun_esti_constraints[fun_param_index][0])]
                # cons1 is the index of another fit parameter. In the global
                # fit_results, we must adjust the index to account for the bg
                # params added to the start of the list.
                cons1 = fun_esti_constraints[fun_param_index][1]
                if constraint_code in ["FACTOR", "DELTA", "SUM"]:
                    cons1 += nb_bg_params
                cons2 = fun_esti_constraints[fun_param_index][2]

            self.fit_results.append({'name': pname,
                                     'estimation': estimation_value,
                                     'group': group_number,
                                     'code': constraint_code,
                                     'cons1': cons1,
                                     'cons2': cons2,
                                     'fitresult': 0.0,
                                     'sigma': 0.0,
                                     'xmin': xmin,
                                     'xmax': xmax})

        self.state = 'Ready to Fit'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})
        return numpy.append(bg_params, fun_esti_parameters)

    def gendata(self, x=None, paramlist=None):
        """Return a data array using the currently selected fit function
        and the fitted parameters.

        :param x: Independent variable where the function is calculated.
            If ``None``, use :attr:`xdata`.
        :param paramlist: List of dictionaries, each dictionary item being a
            fit parameter. The dictionary's format is documented in
            :attr:`fit_results`.
            If ``None`` (default), use parameters from :attr:`fit_results`.
        :return: :meth:`fitfunction` calculated for parameters whose code is
            not set to ``"IGNORE"``.

        This calculates :meth:`fitfunction` on `x` data using fit parameters
        from a list of parameter dictionaries, if field ``code`` is not set
        to ``"IGNORE"``.
        """
        if x is None:
            x = self.xdata
        if paramlist is None:
            paramlist = self.fit_results
        active_params = []
        for param in paramlist:
            if param['code'] not in ['IGNORE', 0, 0.]:
                active_params.append(param['fitresult'])

        newdata = self.fitfunction(numpy.array(x), *active_params)
        return newdata

    def get_estimation(self):
        """Return the list of fit parameter names."""
        if self.state not in ["Ready to fit", "Fit in progress", "Ready"]:
            _logger.warning("get_estimation() called before estimate() completed")
        return [param["estimation"] for param in self.fit_results]

    def get_names(self):
        """Return the list of fit parameter estimations."""
        if self.state not in ["Ready to fit", "Fit in progress", "Ready"]:
            msg = "get_names() called before estimate() completed, "
            msg += "names are not populated at this stage"
            _logger.warning(msg)
        return [param["name"] for param in self.fit_results]

    def get_fit_result(self):
        """Return the list of fit parameter results."""
        if self.state not in ["Ready"]:
            msg = "get_fit_result() called before startfit() completed, "
            msg += "results are not available a this stage"
            _logger.warning(msg)
        return [param["fitresult"] for param in self.fit_results]

    def loadtheories(self, theories):
        """Import user defined fit functions defined in an external Python
        source file, and save them in :attr:`theories`.

        An example of such a file can be found in the sources of
        :mod:`silx.math.fit.fittheories`. It must contain a
        dictionary named ``THEORY`` with the following structure::

            THEORY = {
                'theory_name_1':
                    FitTheory(description='Description of theory 1',
                              function=fitfunction1,
                              parameters=('param name 1', 'param name 2', …),
                              estimate=estimation_function1,
                              configure=configuration_function1,
                              derivative=derivative_function1),
                'theory_name_2':
                    FitTheory(…),
            }

        See documentation of :mod:`silx.math.fit.fittheories` and
        :mod:`silx.math.fit.fittheory` for more
        information on designing your fit functions file.

        This method can also load user defined functions in the legacy
        format used in *PyMca*.

        :param theories: Name of python source file, or module containing the
            definition of fit functions.
        :raise: ImportError if theories cannot be imported
        """
        from types import ModuleType
        if isinstance(theories, ModuleType):
            theories_module = theories
        else:
            # if theories is not a module, it must be a string
            string_types = (basestring,) if sys.version_info[0] == 2 else (str,)  # noqa
            if not isinstance(theories, string_types):
                raise ImportError("theory must be a python module, a module" +
                                  "name or a python filename")
            # if theories is a filename
            if os.path.isfile(theories):
                sys.path.append(os.path.dirname(theories))
                f = os.path.basename(os.path.splitext(theories)[0])
                theories_module = __import__(f)
            # if theories is a module name
            else:
                theories_module = __import__(theories)

        if hasattr(theories_module, "INIT"):
            theories.INIT()

        if not hasattr(theories_module, "THEORY"):
            msg = "File %s does not contain a THEORY dictionary" % theories
            raise ImportError(msg)

        elif isinstance(theories_module.THEORY, dict):
            # silx format for theory definition
            for theory_name, fittheory in list(theories_module.THEORY.items()):
                self.addtheory(theory_name, fittheory)
        else:
            self._load_legacy_theories(theories_module)

    def setbackground(self, theory):
        """Choose a background type from within :attr:`bkgdict`.

        This updates the following attributes:

            - :attr:`fitconfig` ``['fitbkg']``
            - :attr:`bkgfun`

        :param theory: The name of the background to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`bkgdict``.
        """
        if theory in self.bkgdict:
            self.fitconfig['fitbkg'] = theory
            self.bkgfun = self.bkgdict[theory]["function"]
        else:
            msg = "No theory with name %s in bkgdict.\n" % theory
            msg += "Available theories: %s\n" % self.bkgdict.keys()
            raise KeyError(msg)

    def setdata(self, x, y, sigmay=None, xmin=None, xmax=None):
        """Set data attributes:

            - ``xdata0``, ``ydata0`` and ``sigmay0`` store the initial data
              and uncertainties. These attributes are not modified after
              initialization.
            - ``xdata``, ``ydata`` and ``sigmay`` store the data after
              removing values where ``xdata < xmin`` or ``xdata > xmax``.
              These attributes may be modified at a latter stage by filters.


        :param x: Abscissa data. If ``None``, :attr:`xdata`` is set to
            ``numpy.array([0.0, 1.0, 2.0, ..., len(y)-1])``
        :type x: Sequence or numpy array or None
        :param y: The dependant data ``y = f(x)``. ``y`` must have the same
            shape as ``x`` if ``x`` is not ``None``.
        :type y: Sequence or numpy array or None
        :param sigmay: The uncertainties in the ``ydata`` array. These are
            used as weights in the least-squares problem.
            If ``None``, the uncertainties are assumed to be 1.
        :type sigmay: Sequence or numpy array or None
        :param xmin: Lower value of x values to use for fitting
        :param xmax: Upper value of x values to use for fitting
        """
        if y is None:
            self.xdata0 = numpy.array([], numpy.float)
            self.ydata0 = numpy.array([], numpy.float)
            # self.sigmay0 = numpy.array([], numpy.float)
            self.xdata = numpy.array([], numpy.float)
            self.ydata = numpy.array([], numpy.float)
            # self.sigmay = numpy.array([], numpy.float)

        else:
            self.ydata0 = numpy.array(y)
            self.ydata = numpy.array(y)
            if x is None:
                self.xdata0 = numpy.arange(len(self.ydata0))
                self.xdata = numpy.arange(len(self.ydata0))
            else:
                self.xdata0 = numpy.array(x)
                self.xdata = numpy.array(x)

            # default weight in the least-square problem is 1.
            if sigmay is None:
                self.sigmay0 = None
                self.sigmay = None
            else:
                self.sigmay0 = numpy.array(sigmay)
                self.sigmay = numpy.array(sigmay)

            # take the data between limits, using boolean array indexing
            if (xmin is not None or xmax is not None) and len(self.xdata):
                xmin = xmin if xmin is not None else min(self.xdata)
                xmax = xmax if xmax is not None else max(self.xdata)
                bool_array = (self.xdata >= xmin) & (self.xdata <= xmax)
                self.xdata = self.xdata[bool_array]
                self.ydata = self.ydata[bool_array]
                self.sigmay = self.sigmay[bool_array] if sigmay is not None else None

    def settheory(self, theory):
        """Pick a theory from :attr:`theories`.

        This updates the following attributes:

            - :attr:`fitconfig` ``['fittheory']``
            - :attr:`selectedfunction`
            - :attr:`selectedderivative`

        :param theory: Name of the theory to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`theories`.
        """
        if theory is None:
            self.fitconfig['fittheory'] = None
        elif theory in self.theories:
            self.fitconfig['fittheory'] = theory
            self.selectedfunction = self.theories[theory].function
            self.selectedparameters = self.theories[theory].parameters
            self.selectedestimate = self.theories[theory].estimate
            self.selectedconfigure = self.theories[theory].configure
            self.selectedderivative = self.theories[theory].derivative
        else:
            msg = "No theory with name %s in theories.\n" % theory
            msg += "Available theories: %s\n" % self.theories.keys()
            raise KeyError(msg)

    def startfit(self, callback=None):
        """Run the actual fitting and fill :attr:`fit_results` with fit results.

        Before running this method, :attr:`fit_results` must already be
        populated with a list of all parameters and their estimated values.
        For this, run :meth:`estimate` beforehand.

        :param callback: Optional callback function, conforming to the
            signature ``callback(data)`` with ``data`` being a dictionary.
            This callback function is called before and after the estimation
            process, and is given a dictionary containing the values of
            :attr:`state` (``'Fit in progress'`` or ``'Ready'``)
            and :attr:`chisq`.
            This is used for instance in :mod:`silx.gui.fit.FitWidget` to
            update a widget displaying a status message.
        :return: Fitted parameters
        """
        self.dataupdate()

        self.state = 'Fit in progress'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        param_val = []
        param_constraints = []
        # Initial values are set to the ones computed in estimate()
        for param in self.fit_results:
            param_val.append(param['estimation'])
            param_constraints.append([param['code'], param['cons1'], param['cons2']])

        ywork = self.ydata

        if self.fitconfig['fitbkg'] == "Square Filter":
            ywork = self.squarefilter(
                    self.ydata, self.fit_results[0]['estimation'])

        params, covariance_matrix, infodict = leastsq(
                self.fitfunction,  # bg + actual model function
                self.xdata, ywork, param_val,
                sigma=self.sigmay,
                constraints=param_constraints,
                model_deriv=self.selectedderivative,
                full_output=True)
        # if covariance_matrix is not None:
        #     sigmas = numpy.sqrt(numpy.diag(covariance_matrix))
        # else:
        #     # sometimes leastsq returns None and logs:
        #     # "Error calculating covariance matrix after successful fit"
        #     sigmas = numpy.zeros(shape=(len(params),))
        sigmas = infodict['uncertainties']

        for i, param in enumerate(self.fit_results):
            if param['code'] != 'IGNORE':
                param['fitresult'] = params[i]
                param['sigma'] = sigmas[i]

        self.chisq = infodict["chisq"]
        self.state = 'Ready'

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        return params, sigmas, infodict

    ###################
    # Private methods #
    ###################
    def fitfunction(self, x, *pars):
        """Function to be fitted.

        This is the sum of the selected background function plus
        a number of peak functions.

        :param x: Independent variable where the function is calculated.
        :param pars: Sequence of all fit parameters. The first few parameters
            are background parameters, then come the peak function parameters.
            The total number of fit parameters in ``pars`` will
            be `nb_bg_pars + nb_peak_pars * nb_peaks`.
        :return: Output of the fit function with ``x`` as input and ``pars``
            as fit parameters.
        """
        fitbkg_key = self.fitconfig['fitbkg']
        bg_pars_list = self.bkgdict[fitbkg_key]["parameters"]
        nb_bg_pars = len(bg_pars_list)

        peak_pars_list = self.selectedparameters
        nb_peak_pars = len(peak_pars_list)

        nb_peaks = int((len(pars) - nb_bg_pars) / nb_peak_pars)

        result = numpy.zeros(numpy.shape(x), numpy.float)

        # Compute one peak function per peak, and sum the output numpy arrays
        for i in range(nb_peaks):
            start_par_index = nb_bg_pars + i * nb_peak_pars
            end_par_index = nb_bg_pars + (i + 1) * nb_peak_pars
            result += self.selectedfunction(x, *pars[start_par_index:end_par_index])

        if nb_bg_pars > 0:
            result += self.bkgfun(x, *pars[0:nb_bg_pars])
        # TODO: understand and document this Square Filter
        if self.fitconfig['fitbkg'] == "Square Filter":
            result = result - pars[1]
            return pars[1] + self.squarefilter(result, pars[0])
        else:
            return result

    def estimate_bkg(self, x, y):
        """Estimate background parameters using the function defined in
        the current fit configuration.

        To change the selected background model, :attr:`fitconfig`['fitbkg']
        must be changed. The actual background function to be used is
        referenced in :attr:`bkgdict`

        :param x: Sequence of x data
        :param y: sequence of y data
        :return: Tuple of two sequences and one data array
            ``(estimated_param, constraints, bg_data)``:

            - ``estimated_param`` is a list of estimated values for each
              background parameter.
            - ``constraints`` is a 2D sequence of dimension ``(n_parameters, 3)``

                - ``constraints[i][0]``: Constraint code.
                  See explanation about codes in :attr:`fit_results`

                - ``constraints[i][1]``
                  See explanation about 'cons1' in :attr:`fit_results`
                  documentation.

                - ``constraints[i][2]``
                  See explanation about 'cons2' in :attr:`fit_results`
                  documentation.
            - ``bg_data`` is the background data computed by the function.
        """
        fitbkg = self.fitconfig['fitbkg']
        background_estimate_function = self.bkgdict[fitbkg]["estimate"]
        if background_estimate_function is not None:
            return background_estimate_function(x, y)
        else:
            return [], [], numpy.zeros_like(y)

    def estimate_fun(self, x, y, bg, xscaling=1.0, yscaling=None):
        """Estimate fit parameters using the function defined in
        the current fit configuration.

        :param x: Sequence of x data
        :param y: sequence of y data
        :param bg: Background signal, to be subtracted from ``y`` before fitting.
        :param xscaling: Scaling factor for ``x`` data. Default ``1.0`` (no scaling)
        :param yscaling: Scaling factor for ``y`` data. Default ``None``,
            meaning use value from configuration dictionary
            :attr:`fitconfig` (``'Yscaling'`` field).
        :return: Tuple of two sequences ``(estimated_param, constraints)``:

            - ``estimated_param`` is a list of estimated values for each
              background parameter.
            - ``constraints`` is a 2D sequence of dimension (n_parameters, 3)

                - ``constraints[i][0]``: Constraint code.
                  See explanation about codes in :attr:`fit_results`

                - ``constraints[i][1]``
                  See explanation about 'cons1' in :attr:`fit_results`
                  documentation.

                - ``constraints[i][2]``
                  See explanation about 'cons2' in :attr:`fit_results`
                  documentation.
        :raise: ``TypeError`` if estimation function is not callable

        """
        fittheory = self.fitconfig['fittheory']
        estimatefunction = self.selectedestimate
        if hasattr(estimatefunction, '__call__'):
            return estimatefunction(x, y, bg,
                                    yscaling=yscaling)
        else:
            # return [], []

            # fit requires at least one parameter
            raise TypeError("Estimation function in attribute " +
                            "theories[%s]" % fittheory +
                            " must be callable.")

    def bkg_constant(self, x, *pars):
        """Constant background function ``y = constant``

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters: ``(constant, )``
        :return: Array of the same shape as ``x`` filled with constant value
        """
        return pars[0] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_linear(self, x, *pars):
        """Linear background function ``y = constant + slope * x``

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters: ``(constant, slope)``
        :return: Array ``y = constant + slope * x``
        """
        return pars[0] + pars[1] * x

    def bkg_internal(self, x, *pars):
        """
        Internal Background based on strip filter
        (:meth:`silx.math.fit.filters.strip`)
        """
        # TODO: document

        if self.bkg_internal_oldpars[0] == pars[0]:
            if self.bkg_internal_oldpars[1] == pars[1]:
                if (len(x) == len(self.bkg_internal_oldx)) & \
                   (len(self.ydata) == len(self.bkg_internal_oldy)):
                    # same parameters
                    if numpy.sum(self.bkg_internal_oldx == x) == len(x):
                        if numpy.sum(self.bkg_internal_oldy == self.ydata) == len(self.ydata):
                            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)
        self.bkg_internal_oldy = self.ydata
        self.bkg_internal_oldx = x
        self.bkg_internal_oldpars = pars
        idx = numpy.nonzero((self.xdata >= x[0]) & (self.xdata <= x[-1]))[0]
        yy = numpy.take(self.ydata, idx)
        nrx = numpy.shape(x)[0]
        nry = numpy.shape(yy)[0]
        if nrx == nry:
            self.bkg_internal_oldbkg = strip(yy, pars[0], pars[1])
            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)

        else:
            self.bkg_internal_oldbkg = strip(numpy.take(yy, numpy.arange(0, nry, 2)),
                                                         pars[0], pars[1])
            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_squarefilter(self, x, *pars):
        """
        Square filter Background
        """
        # TODO: docstring
        # why is this different than bkg_constant?
        # what is pars[0]?
        # what defines the (xmin, xmax) limits of the square function?
        return pars[1] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_none(self, x, *pars):
        """Null background function.

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters. Ignored, only present
            because other methods expect this signature for all background
            functions
        :return: Array of 0 values of the same shape as ``x``
        """
        return numpy.zeros(x.shape, numpy.float)

    def _load_legacy_theories(self, theories_module):
        """Load theories from a custom module in the old PyMca format.

        See PyMca5.PyMcaMath.fitting.SpecfitFunctions for an example.
        """
        mandatory_attributes = ["THEORY", "PARAMETERS",
                                "FUNCTION", "ESTIMATE"]
        err_msg = "Custom fit function file must define: "
        err_msg += ", ".join(mandatory_attributes)
        for attr in mandatory_attributes:
            if not hasattr(theories_module, attr):
                raise ImportError(err_msg)

        derivative = theories_module.DERIVATIVE if hasattr(theories_module, "DERIVATIVE") else None
        configure = theories_module.CONFIGURE if hasattr(theories_module, "CONFIGURE") else None

        if isinstance(theories_module.THEORY, (list, tuple)):
            # multiple fit functions
            for i in range(len(theories_module.THEORY)):
                deriv = derivative[i] if derivative is not None else None
                config = configure[i] if configure is not None else None
                self.addtheory(theories_module.THEORY[i],
                               FitTheory(
                                   theories_module.FUNCTION[i],
                                   theories_module.PARAMETERS[i],
                                   theories_module.ESTIMATE[i],  # FIXME: should we handle no ESTIMATE?
                                   config,
                                   deriv))
        else:
            # single fit function
            self.addtheory(theories_module.THEORY,
                           FitTheory(
                               theories_module.FUNCTION,
                               theories_module.PARAMETERS,
                               theories_module.ESTIMATE,
                               configure,
                               derivative))

    def estimate_builtin_bkg(self, x, y):
        """Compute the initial parameters for the background function before
        starting the iterative fit.

        The return parameters and constraints depends on the selected theory:

            - ``'Constant'``: [min(background)], constraint FREE
            - ``'Internal'``: [1.000, 10000, 0.0], constraint FIXED
            - ``'No Background'``: empty array []
            - ``'Square Filter'``:
            - ``'Linear'``: [constant, slope], constraint FREE

        :param x: Array of values for the independant variable
        :param y: Array of data values for the dependant data
        :return: Tuple ``(fitted_param, constraints, background)`` where:

            - ``fitted_param`` is a list of the estimated background
              parameters. The length of the list depends on the theory used
            - ``constraints`` is a numpy array of shape
              *(len(fitted_param), 3)* containing constraint information for
              each parameter *code, cons1, cons2* (see documentation of
              :attr:`fit_results`)
            - ``background`` is the background signal extracted from the data
              using a :func:`strip` filter
        """
        # TODO: document square filter
        background = strip(y, w=1, niterations=10000, factor=1.0)
        npoints = len(background)
        if self.fitconfig['fitbkg'] == 'Constant':
            # Constant background
            Sy = min(background)
            fittedpar = [Sy]
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
        elif self.fitconfig['fitbkg'] == 'Internal':
            # Internal
            fittedpar = [1.000, 10000, 0.0]
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
            # code = 3: FIXED
            cons[0][0] = 3
            cons[1][0] = 3
            cons[2][0] = 3
        elif self.fitconfig['fitbkg'] == 'No Background':
            # None
            fittedpar = []
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
        elif self.fitconfig['fitbkg'] == 'Square Filter':
            fwhm = self.fitconfig['FwhmPoints']

            # set an odd number
            if fwhm % 2 == 1:
                fittedpar = [fwhm, 0.0]
            else:
                fittedpar = [fwhm + 1, 0.0]
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
            # code = 3: FIXED
            cons[0][0] = 3
            cons[1][0] = 3
        else:  # Linear regression
            n = float(npoints)
            Sy = numpy.sum(background)
            Sx = float(numpy.sum(x))
            Sxx = float(numpy.sum(x * x))
            Sxy = float(numpy.sum(x * background))

            deno = n * Sxx - (Sx * Sx)
            if (deno != 0):
                bg = (Sxx * Sy - Sx * Sxy) / deno
                slop = (n * Sxy - Sx * Sy) / deno
            else:
                bg = 0.0
                slop = 0.0
            fittedpar = [bg / 1.0, slop / 1.0]
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
        # Fixme: should we return a bg computed using our estimated params, instead of stripped bg?
        return fittedpar, cons, background

    def guess_yscaling(self, y=None):
        """Return the inverse chi-squared value"""
        if y is None:
            y = self.ydata

        # Apply basic smoothing
        # (the convolution adds one extra sample to each side of the array)
        yfit = numpy.convolve(y, [1., 1., 1.])[1:-1] / 3.0

        # Find indices of non-zero samples
        # (numpy.nonzero returns one array per dimension.
        # We are dealing with 1D data, hence the [0])
        idx = numpy.nonzero(y)[0]

        # Reject zero values
        y = numpy.take(y, idx)
        yfit = numpy.take(yfit, idx)

        chisq = numpy.sum(((y - yfit) * (y - yfit)) /
                          (numpy.fabs(y) * len(y)))
        try:
            scaling = 1. / chisq
        except ZeroDivisionError:
            scaling = 1.0
        return scaling

    def squarefilter(self, y, width):
        """

        :param y:
        :param width:
        :return:
        """ # TODO: document
        if len(y) == 0:
            if isinstance(y, list):
                return []
            else:
                return numpy.array([])

        # make width an odd number of samples and calculate half width
        width = int(width) + ((int(width) + 1) % 2)
        half_width = int(width / 2)

        len_coef = 2 * half_width + width

        if len(y) < len_coef:
            return y

        coef = numpy.zeros((len_coef,), numpy.float)

        coef[0:half_width] = -0.5 / float(half_width)
        coef[half_width:(half_width + width)] = 1.0 / float(width)
        coef[(half_width + width):len(coef)] = -0.5 / float(half_width)

        result = numpy.zeros(len(y), numpy.float)
        result[(width - 1):-(width - 1)] = numpy.convolve(y, coef, 0)
        result[0:width - 1] = result[width - 1]
        result[-(width - 1):] = result[-(width + 1)]
        return result


def test():
    from .functions import sum_gauss
    from . import fittheories

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)

    p = [1000, 100., 250,
         255, 700., 45,
         1500, 800.5, 95]
    y = 2.65 * x + 13 + sum_gauss(x, *p)

    # Fitting
    fit = FitManager()
    fit.setdata(x=x, y=y)
    fit.loadtheories(fittheories)
    fit.settheory('gauss')
    fit.setbackground('Linear')
    fit.estimate()
    fit.startfit()

    print("Searched parameters = ", p)
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.fit_results:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)

    # Plot
    constant, slope = dummy_list[:2]
    p1 = dummy_list[2:]
    y2 = slope * x + constant + sum_gauss(x, *p1)

    from silx.gui import qt
    from silx.gui.plot.PlotWindow import PlotWindow
    app = qt.QApplication([])
    pw = PlotWindow(control=True)
    pw.addCurve(x, y, "Original")
    pw.addCurve(x, y2, "Fit result")
    pw.legendsDockWidget.show()
    pw.show()
    app.exec_()


if __name__ == "__main__":
    test()
