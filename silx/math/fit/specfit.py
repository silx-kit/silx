# coding: utf-8
#  /*#########################################################################
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
"""Multi-peak fitting



"""
from collections import OrderedDict
import logging
import numpy
import os
import sys

from .filters import strip
from .peaks import peak_search
from .leastsq import leastsq


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "09/06/2016"

_logger = logging.getLogger(__name__)


class Specfit():
    """
    Multi-peak fitting functions manager

    Data attributes:

     - :attr:`xdata0`, :attr:`ydata0` and :attr:`sigmay0` store the initial data
       and uncertainties. These attributes are not modified after
       initialization.
     - :attr:`xdata`, :attr:`ydata` and :attr:`sigmay` store the data after
       removing values where :attr:`xdata < xmin` or :attr:`xdata > xmax`.
       These attributes may be modified at a latter stage by filters.
    """

    def __init__(self, x=None, y=None, sigmay=None, auto_fwhm=False, fwhm_points=8,
                 auto_scaling=False, yscaling=1.0, sensitivity=2.5,
                 residuals_flag=0, mca_mode=0):
        """
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
        :param residuals_flag:
        :param mca_mode:
        :param event_handler:
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

            - 'ResidualsFlag': ?
        """
        # TODO: ResidualsFlag

        self.fitconfig['AutoFwhm'] = auto_fwhm
        self.fitconfig['FwhmPoints'] = fwhm_points
        self.fitconfig['AutoScaling'] = auto_scaling
        self.fitconfig['Yscaling'] = yscaling
        self.fitconfig['Sensitivity'] = sensitivity
        self.fitconfig['ResidualsFlag'] = residuals_flag
        self.fitconfig['McaMode'] = mca_mode
        self.fitconfig['fitbkg'] = 'No Background'
        self.fitconfig['fittheory'] = None

        self.theorydict = OrderedDict()
        """Dictionary of functions to be fitted to individual peaks.

        Keys are descriptive theory names (e.g "Gaussians" or "Step up").
        Values are lists:
        ``[function, parameters, estimate, configure, ????, derivative]``

            - ``function`` is the fit function for an individual peak
            - ``parameters`` is a sequence of parameter names
            - ``estimate`` is the parameter estimation function
            - ``configure`` is the function returning the configuration dict
              for the theory in the format described in the :attr:` fitconfig`
              documentation
            - ``derivative`` (optional) is a custom derivative function, whose
              signature is described in the documentation of
              :func:`silx.math.fit.leastsq.leastsq`
              (``model_deriv(xdata, parameters, index)``).
        """

        self.dataupdate = None
        """This attribute can be updated with a user defined function to
        update data (for instance modify range fo :attr:`xdata`,
        :attr:`ydata` and :attr:`sigmay` when user zooms in or out in a GUI
        plot).
        """

        self.bkgdict = OrderedDict(
            [('No Background', [self.bkg_none, [], None]),
             ('Constant', [self.bkg_constant, ['Constant'],
                           self.estimate_builtin_bkg]),
             ('Linear', [self.bkg_linear, ['Constant', 'Slope'],
                         self.estimate_builtin_bkg]),
             ('Internal', [self.bkg_internal,
                           ['Curvature', 'Iterations', 'Constant'],
                           self.estimate_builtin_bkg])])
        """Dictionary of background functions.

        Keys are descriptive theory names (e.g "Constant" or "Linear").
        Values are list: ``[function, parameters, estimate]``

        ``function`` is a callable function with the signature ``function(x, params) -> y``
        where params is a sequence of parameters.

        ``parameters`` is a sequence of parameter names (e.g. could be
        for a linear function ``["constant", "slope"]``).

        ``estimate`` is a function to compute initial values for parameters.
        It should have the following signature:
        ``f(xx, yy, zzz, xscaling=1.0, yscaling=None)``
        """  # FIXME estimate signature, meaning of zz

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
        :attr:`theorydict` to determine how many parameters there will be,
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
            - 'sigma': ???
            - 'xmin': ???
            - 'xmax': ???
        """

        self.modelderiv = None
        """"""
        self.theoryfun = None
        """"""

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
            self.sigmay0 = numpy.array([], numpy.float)
            self.xdata = numpy.array([], numpy.float)
            self.ydata = numpy.array([], numpy.float)
            self.sigmay = numpy.array([], numpy.float)

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
                self.sigmay0 = numpy.ones(self.ydata.shape, dtype=numpy.float)
                self.sigmay = numpy.ones(self.ydata.shape, dtype=numpy.float)
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
                self.sigmay = self.sigmay[bool_array]

    def addtheory(self, theory, function, parameters, estimate=None,
                  configure=None, derivative=None):
        """Add a new theory to dictionary :attr:`theorydict`.

        :param theory: String with the name describing the function
        :param function: Actual peak function
        :param parameters: Parameters names for function ``['p1','p2',…]``
        :param estimate: Initial parameters estimation
        :param configure: Optional function to be called to initialize
            parameters prior to fit
        :param derivative: Optional analytical derivative function.
            Its signature should be ``f(xdata, parameters, index)``

        """
        self.theorydict[theory] = [function, parameters,
                                   estimate, configure, derivative]

    def addbackground(self, background, function, parameters, estimate=None):
        """Add a new background function to dictionary :attr:`bkgdict`.

        :param background: String with the name describing the function
        :param function: Actual function
        :param parameters: Parameters names ['p1','p2','p3',...]
        :param estimate: The initial parameters estimation function if any
        """
        self.bkgdict[background] = [function, parameters, estimate]

    def settheory(self, theory):
        """Pick a theory from :attr:`theorydict`.

        This updates the following attributes:

            - :attr:`fitconfig` ``['fittheory']``
            - :attr:`theoryfun`
            - :attr:`modelderiv`

        :param theory: Name of the theory to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`theorydict`.
        """
        if theory in self.theorydict:
            self.fitconfig['fittheory'] = theory
            self.theoryfun = self.theorydict[theory][0]
            self.modelderiv = None

        else:
            msg = "No theory with name %s in theorydict.\n" % theory
            msg += "Available theories: %s\n" % self.theorydict.keys()
            raise KeyError(msg)

    def setbackground(self, theory):
        """Choose a background type from within :attr:`bkgdict``.

        This updates the following attributes:

            - :attr:`fitconfig` ``['fitbkg']``
            - :attr:`bkgfun`

        :param theory: The name of the background to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`bkgdict``.
        """
        if theory in self.bkgdict:
            self.fitconfig['fitbkg'] = theory
            self.bkgfun = self.bkgdict[theory][0]
        else:
            msg = "No theory with name %s in bkgdict.\n" % theory
            msg += "Available theories: %s\n" % self.bkgdict.keys()
            raise KeyError(msg)

    def fitfunction(self, x, *pars):
        """Fit function.

        This is the sum of the background function plus
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
        bg_pars_list = self.bkgdict[fitbkg_key][1]
        nb_bg_pars = len(bg_pars_list)

        fittheory_key = self.fitconfig['fittheory']
        peak_pars_list = self.theorydict[fittheory_key][1]
        nb_peak_pars = len(peak_pars_list)

        nb_peaks = int((len(pars) - nb_bg_pars) / nb_peak_pars)

        result = numpy.zeros(numpy.shape(x), numpy.float)

        # Compute one peak function per peak, and sum the output numpy arrays
        for i in range(nb_peaks):
            start_par_index = nb_bg_pars + i * nb_peak_pars
            end_par_index = nb_bg_pars + (i + 1) * nb_peak_pars
            result += self.theoryfun(x, *pars[start_par_index:end_par_index])

        if nb_bg_pars > 0:
            result += self.bkgfun(x, *pars[0:nb_bg_pars])
        # TODO: understand and document this Square Filter business
        if self.fitconfig['fitbkg'] == "Square Filter":
            result = result - pars[1]
            return pars[1] + self.squarefilter(result, pars[0])
        else:
            return result

    def estimate(self, mcafit=False, callback=None):
        """
        Fill :attr:`fit_results` with an estimation made on the given data.

        This method registers and sends a ``'FitStatusChanged'`` event, before
        starting the estimation and after completing. This event sends a
        *status* (``Estimate in progress"`` or ``"Ready to Fit"``).
        """
        # TODO: explain estimation process
        self.state = 'Estimate in progress'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        CONS = ['FREE',
                'POSITIVE',
                'QUOTED',
                'FIXED',
                'FACTOR',
                'DELTA',
                'SUM',
                'IGNORE']

        # Update data (actual data or just range) using user defined method
        if self.dataupdate is not None:
            if not mcafit:
                self.dataupdate()

        xwork = self.xdata
        ywork = self.ydata

        # estimate the background
        esti_bkg = self.estimate_bkg(xwork, ywork)
        bkg_esti_parameters = esti_bkg[0]
        bkg_esti_constraints = esti_bkg[1]
        try:
            zz = numpy.array(esti_bkg[2])
        except IndexError:
            zz = numpy.zeros(numpy.shape(ywork), numpy.float)

        # scaling
        if self.fitconfig['AutoScaling']:
            yscaling = self.guess_yscaling(y=ywork)
        elif self.fitconfig['Yscaling'] is not None:
            yscaling = self.fitconfig['Yscaling']

        # estimate the function
        esti_fun = self.estimate_fun(xwork, ywork, zz, yscaling=yscaling)

        fun_esti_parameters = esti_fun[0]
        fun_esti_constraints = esti_fun[1]

        # build the names
        self.parameter_names = []

        fitbkg = self.fitconfig['fitbkg']
        for bg_param_name in self.bkgdict[fitbkg][1]:
            self.parameter_names.append(bg_param_name)

        fittheory = self.fitconfig['fittheory']
        param_index, peak_index = 0, 0
        while param_index < len(fun_esti_parameters):
            peak_index += 1
            for fun_param_name in self.theorydict[fittheory][1]:
                self.parameter_names.append(fun_param_name + "%d" % peak_index)
                param_index += 1

        self.fit_results = []
        nb_fun_params_per_group = len(self.theorydict[fittheory][1])
        group_number = 0 #k
        xmin = min(xwork)
        xmax = max(xwork)
        nb_bg_params = len(bkg_esti_parameters)
        for (pindex, pname) in enumerate(self.parameter_names):
            # First come background parameters
            if pindex < nb_bg_params:
                estimation_value = bkg_esti_parameters[pindex]
                constraint_code = CONS[int(bkg_esti_constraints[pindex][0])]
                cons1 = bkg_esti_constraints[pindex][1]
                cons2 = bkg_esti_constraints[pindex][2]
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
        return self.fit_results

    def estimate_bkg(self, x, y):
        """Estimate background parameters using the function defined in
        the current fit configuration.

        :param x: Sequence of x data
        :param y: sequence of y data
        :return: Tuple of two sequences ``(estimated_param, constraints)``:

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
        """
        fitbkg = self.fitconfig['fitbkg']
        background_estimate_function = self.bkgdict[fitbkg][2]
        if background_estimate_function is not None:
            return background_estimate_function(x, y)
        else:
            return [], []

    def estimate_fun(self, x, y, z, xscaling=1.0, yscaling=None):
        """Estimate fit parameters using the function defined in
        the current fit configuration.

        :param x: Sequence of x data
        :param y: sequence of y data
        :param z: *undocumented* (possibly the bg estimated with :func:`strip`)
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

        """
        # Fixme document z
        fittheory = self.fitconfig['fittheory']
        estimatefunction = self.theorydict[fittheory][2]
        if estimatefunction is not None:
            return estimatefunction(x, y, z,
                                    xscaling=xscaling, yscaling=yscaling)
        else:
            return [], []

    def importfun(self, file):
        """Import user defined fit functions defined in an external Python
        source file.

        :param file: Name of python source file containing the definition
            of fit functions.

        An example of such a file can be found at
        `https://github.com/vasole/pymca/blob/master/PyMca5/PyMcaMath/fitting/SpecfitFunctions.py`_

        Imported functions are saved in :attr:`theorydict`.
        """
        sys.path.append(os.path.dirname(file))
        f = os.path.basename(os.path.splitext(file)[0])
        newfun = __import__(f)
        if hasattr(newfun, "INIT"):
            newfun.INIT()

        theory = newfun.THEORY if hasattr(newfun, "THEORY") else \
            "%s" % file
        parameters = newfun.PARAMETERS
        function = newfun.FUNCTION
        estimate = newfun.ESTIMATE if hasattr(newfun, "ESTIMATE") else None
        derivative = newfun.DERIVATIVE if hasattr(newfun, "DERIVATIVE")\
            else None
        configure = newfun.CONFIGURE if hasattr(newfun, "CONFIGURE") else None

        # if theory is a list, we assume all other fit parameters to be lists
        # of the same length
        if isinstance(theory, list):

            for i in range(len(theory)):
                deriv = derivative[i] if derivative is not None else None
                esti = estimate[i] if estimate is not None else None
                conf = configure[i] if configure is not None else None
                self.addtheory(
                    theory[i], function[i], parameters[i], esti, conf, deriv)
        else:
            self.addtheory(
                theory, function, parameters, estimate, configure, derivative)

    def startfit(self, mcafit=0, callback=None):
        """Run the actual fitting and fill :attr:`fit_results` with fit results.

        Before running this method, :attr:`fit_results` must already be
        populated with a list of all parameters and their estimated values.
        For this, run :meth:`estimate` beforehand.

        This method registers and sends a *FitStatusChanged* event, before
        starting the fit and after completing. This event sends a
        *status* (`"Fit in progress"` or "Ready") and a *chisq* value.
        """
        if self.dataupdate is not None:
            if not mcafit:
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

        # constraints = None if param['code'] in ['FREE', 0, 0.0] else \
        #     param_constraints

        found = leastsq(self.fitfunction, self.xdata, ywork, param_val,
                        constraints=param_constraints,
                        model_deriv=self.modelderiv, full_output=True)

        for i, param in enumerate(self.fit_results):
            if param['code'] != 'IGNORE':
                param['fitresult'] = found[0][i]
                #param['sigma'] = found[1][i]
                param['sigma'] = found[2]["chisq"]   # FIXME: where can old sigma be found, after switching leastsq()?

        self.chisq = found[2]["chisq"]
        self.state = 'Ready'

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

    # leastsq() already provides a default derivative
    #
    # def myderiv(self, param0, index, x):
    #     """Apply derivative function to compute partial derivative of
    #     :meth:`fitfunction` with respect to fit parameter indexed by ``index``
    #     at value ``param0[index]``
    #
    #     If no derivative function is provided for the chosen theory, use a
    #     default symmetric derivative :meth:`num_deriv`
    #
    #     :param param0: List of all fit parameter values (estimated)
    #     :param index: Index of fit parameter which varies for this partial
    #         derivative.
    #     :param x: Independent variable where :meth:`fitfunction` is computed.
    #     :return: Numpy array of the same shape as ``x`` containing the result
    #        of the partial derivative.
    #
    #     """
    #     fitbkg = self.fitconfig['fitbkg']
    #     fittheory = self.fitconfig['fittheory']
    #     nb_bg_params = len(self.bkgdict[fitbkg][1])
    #
    #     # for custom derivative function, it seems the parameter numbering
    #     # must ignore background parameters
    #     if index >= nb_bg_params:
    #         if len(self.theorydict[fittheory]) > 5:
    #             derivative = self.theorydict[fittheory][5]
    #             if derivative is not None:
    #                 return derivative(param0, index - nb_bg_params, x)
    #
    #     # if no derivative function is provided, or we are dealing with a
    #     # background parameter, use the default one
    #     return self.num_deriv(param0, index, x)
    #
    # def num_deriv(self, param0, index, x):
    #     """Symmetric partial derivative of :meth:`fitfunction` with respect
    #     to fit parameter indexed by ``index`` at value ``param0[index]``
    #
    #     :param param0: List of all fit parameter values (estimated)
    #     :param index: Index of fit parameter which varies for this partial
    #         derivative.
    #     :param x: Independent variable where :meth:`fitfunction` is computed.
    #     :return: Numpy array of the same shape as ``x`` containing the result
    #         of the partial derivative::
    #
    #             param1 = [param0[i] * 1.00001 if i==index else param0[i]]
    #             param2 = [param0[i] * 0.99999 if i==index else param0[i]]
    #             return (f(param1, x) - f(param2, x)) / (2 * 0.00001)
    #     """
    #     # numerical derivative
    #     x = numpy.array(x)
    #     delta = (param0[index] + numpy.equal(param0[index], 0.0)) * 0.00001
    #     newpar = param0.__copy__()
    #     newpar[index] = param0[index] + delta
    #     f1 = self.fitfunction(x, *newpar)
    #     newpar[index] = param0[index] - delta
    #     f2 = self.fitfunction(x, *newpar)
    #     return (f1 - f2) / (2.0 * delta)

    def gendata(self, x=None, paramlist=None):
        """Calculate :meth:`fitfunction` on `x` data using fit parameters from
        a list of parameter dictionaries, if field ``code`` is not set
        to ``"IGNORE"``.

        :param x: Independent variable where the function is calculated.
        :param paramlist: List of dictionaries, each dictionary item being a
            fit parameter. The dictionary's format is documented in
            :attr:`fit_results`.
            If ``None``, use parameters from :attr:`fit_results`.
        :return: :meth:`fitfunction` calculated for parameters whose code is
            not ``"IGNORE"``.
        """
        if x is None:
            x = self.xdata
        if paramlist is None:
            paramlist = self.fit_results
        noigno = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                noigno.append(param['fitresult'])

        newdata = self.fitfunction(numpy.array(x), *noigno)
        return newdata

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
        Internal Background
        """
        # TODO: understand and document

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

    def estimate_builtin_bkg(self, x, y):
        """Compute the initial parameters for the background function before
        starting the iterative fit.

        The return values depends on the selected theory:

            - ``'Constant'``: [min(background)], constraint FREE
            - ``'Internal'``: [1.000, 10000, 0.0], constraint FIXED
            - ``'No Background'``: empty array []
            - ``'Square Filter'``:
            - ``'Linear'``

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
        background = strip(y, 1, 1000, 1.0001)
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
        else:  # Linear
            S = float(npoints)
            Sy = numpy.sum(background)
            Sx = float(numpy.sum(x))
            Sxx = float(numpy.sum(x * x))
            Sxy = float(numpy.sum(x * background))

            deno = S * Sxx - (Sx * Sx)
            if (deno != 0):
                bg = (Sxx * Sy - Sx * Sxy) / deno
                slop = (S * Sxy - Sx * Sy) / deno
            else:
                bg = 0.0
                slop = 0.0
            fittedpar = [bg / 1.0, slop / 1.0]
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
        return fittedpar, cons, background

    def configure(self, **kw):
        """Configure the current theory by filling or updating the
        :attr:`fitconfig` dictionary. Return config dictionary modified by
        custom configuration function defined in :attr:`theorydict`.

        This methods accepts only named parameters. All ``**kw`` parameters
        are expected to be fields of :attr:`fitconfig` to be updated, unless
        they have a special meaning for the custom configuration function
        defined in ``fitconfig['fittheory']``.

        """
        # FIXME: - why overwrite result from custom config() with **kw?
        #
        # # inspect **kw to find known keys, update them in self.fitconfig
        # for key in self.fitconfig.keys():
        #     if key in kw:
        #         self.fitconfig[key] = kw[key]

        # initialize dict with existing config dict
        result = {}
        result.update(self.fitconfig)

        # Apply custom configuration function defined in self.theorydict[][3]
        theory_name = self.fitconfig['fittheory']
        if theory_name is not None:
            if theory_name in self.theorydict.keys():
                custom_config_fun = self.theorydict[theory_name][3]
                if custom_config_fun is not None:
                    result.update(custom_config_fun(**kw))

                    # Update self.fitconfig with custom config
                    for key in self.fitconfig.keys():
                        if key in result:
                            self.fitconfig[key] = result[key]

        # overwrite existing keys with values from **kw in fitconfig
        for key in self.fitconfig.keys():
            if key in kw:
                self.fitconfig[key] = kw[key]
            if key == "fitbkg":
                self.setbackground(self.fitconfig[key])
            if key == "fittheory":
                if theory_name is not None:
                    self.settheory(self.fitconfig[key])

        result.update(self.fitconfig)
        return result

    def mcafit(self, x=None, y=None, sigmay=None, yscaling=None,
               sensitivity=None, fwhm_points=None, **kw):
        """Run a multi-peak fitting in several steps:

            - detect peaks
            - for each detected peak:

                - run a fit on a limited subset of the
                  data around the peak
                - if the fit is not good enough (large chi-squared value),
                  try to iteratively fit more peaks in the range until
                  the fit is good enough
            - return a list of parameters for all fitted peaks

        :param x: Independent variable where the data is measured
        :param y: Measured data
        :param sigmay: The uncertainties in the ``y`` array. These are
            used as weights in the least-squares problem.
        :param yscaling: Scaling factor for ``y`` data. If ``None``, defaults
            to value defined in ``"Yscaling"`` field of dictionary
            :attr:`fitconfig`, unless  ``"AutoScaling"`` is ``True`` in
            :attr:`fitconfig`, in which case the value returned by
            :meth:`guess_yscaling` is used.
        :param sensitivity: Sensitivity value used for by peak detection
            algorithm. To be detected, a peak must have an amplitude greater
            than ``σn * sensitivity`` (where ``σn`` is an estimated value of
            the standard deviation of the noise).
        :param fwhm_points: Full-width at half-maximum of the peak function,
            expressed in number of data points. If ``None`` and
            ``"AutoFwhm"``is ``True`` in :attr:`fitconfig`, use value returned
            by :meth:`guess_fwhm`.

        :return: MCA fit result for each peak, as a list of dictionaries with the
            following fields:

                - ``xbegin``: minimum of :attr:`xdata` on which the fit was
                  performed
                - ``xend``: maximum of :attr:`xdata` on which fit was
                  performed
                - ``fitstate``: *Estimate in progress*, *Ready to fit*,
                  *Fit in progress*, *Ready* or *Unknown*
                - ``fitconfig``: :attr:`fitconfig`
                - ``config``: Return value of :meth:`configure` (dictionary of
                  configuration parameters)
                - ``paramlist``: :attr:`fit_results`
                - ``chisq``: :attr:`chisq`
                - ``mca_areas``: :attr:`chisq`
        """
        if y is None:
            y = self.ydata0

        if x is None:
            x = numpy.arange(len(y)).astype(numpy.float)

        self.setdata(x, y, sigmay)

        if yscaling is None:
            if self.fitconfig['AutoScaling']:
                yscaling = self.guess_yscaling()
            else:
                yscaling = self.fitconfig['Yscaling']

        if sensitivity is None:
            sensitivity = self.fitconfig['Sensitivity']

        if fwhm_points is None:
            if self.fitconfig['AutoFwhm']:
                fwhm_points = self.guess_fwhm(y=y)
            else:
                fwhm_points = self.fitconfig['FwhmPoints']

        fwhm_points = int(fwhm_points)

        # needed to make sure same peaks are found
        self.configure(Yscaling=yscaling,
                       autoscaling=False,
                       FwhmPoints=fwhm_points,
                       Sensitivity=sensitivity)
        ysearch = self.ydata * yscaling
        npoints = len(ysearch)

        # Detect peaks
        peaks = []
        if npoints > (1.5) * fwhm_points:
            peaksidx = peak_search(ysearch, fwhm_points, sensitivity)
            for idx in peaksidx:
                peaks.append(self.xdata[int(idx)])
            _logger.debug("MCA Found peaks = " + str(peaks))

        # Define regions of interest around each peak
        if len(peaks):
            regions = self.mcaregions(peaks, self.xdata[fwhm_points] - self.xdata[0])
        else:
            regions = []
        _logger.debug(" regions = " + str(regions))

        # Run fit on each individual ROI around the peak
        mcaresult = []
        xmin0 = self.xdata[0]
        xmax0 = self.xdata[-1]
        for region in regions:
            # Limit data range to +- 3*fwhm around the peak
            self.setdata(self.xdata0, self.ydata0, self.sigmay0,
                         xmin=region[0], xmax=region[1])

            # Estimate and fit
            self.estimate(mcafit=1)
            if self.state == 'Ready to Fit':
                self.startfit(mcafit=1)
                # If fit is not good enough, try fitting more peaks in the ROI
                if self.chisq is not None and self.fitconfig['ResidualsFlag']:
                    while(self.chisq > 2.5):
                        newpar, newcons = self.mcaresidualssearch()
                        # If a new peak was fitted, add a group of parameters
                        if newpar != []:
                            newg = 1
                            for param in self.fit_results:
                                newg = max(
                                    newg, int(float(param['group']) + 1))
                                param['estimation'] = param['fitresult']
                            i = -1
                            for pname in self.theorydict[self.fitconfig['fittheory']][1]:
                                i += 1
                                name = pname + "%d" % newg
                                self.fit_results.append({'name': name,
                                                         'estimation': newpar[i],
                                                         'group': newg,
                                                         'code': newcons[0][i],
                                                         'cons1': newcons[1][i],
                                                         'cons2': newcons[2][i],
                                                         'fitresult': 0.0,
                                                         'sigma': 0.0})
                            self.startfit()
                        else:
                            break
            mcaresult.append(self.mcagetresult())  # FIXME: does this store pointers to the same attributes that are overwritten for each region?

            # Restore ydata and xdata to full length
            self.setdata(self.xdata0, self.ydata0, xmin=xmin0, xmax=xmax0)
        return mcaresult

    def mcaregions(self, peaks, fwhm):
        """Return list of ``x`` regions around peaks (plus and minus 3 times
        ``fwhm``).

        :param peaks: List of ``x`` coordinates of peaks
        :param fwhm: Full width at half maximum, in the same unit as ``x``.
        :return: List of ``x`` ranges, as length-2 lists ``[x0, x1]``
        """
        mindelta = 3.0 * fwhm
        plusdelta = 3.0 * fwhm
        regions = []
        min_x = min(self.xdata[0], self.xdata[-1])
        max_x = max(self.xdata[0], self.xdata[-1])
        for peak in peaks:
            x0 = max(peak - mindelta, min_x)
            x1 = min(peak + plusdelta, max_x)
            if regions == []:
                regions.append([x0, x1])
            else:
                if x0 < regions[-1][0]:
                    regions[-1][0] = x0
                elif x0 < (regions[-1][1]):
                    regions[-1][1] = x1
                else:
                    regions.append([x0, x1])
        return regions

    def mcagetresult(self):
        """Return result of MCA fit as a dictionary with the following fields:

            - ``xbegin``: minimum of :attr:`xdata` on which fit was performed
            - ``xend``: maximum of :attr:`xdata` on which fit was performed
            - ``fitstate``: *Estimate in progress*, *Ready to fit*,
              *Fit in progress*, *Ready* or *Unknown*
            - ``fitconfig``: :attr:`fitconfig`
            - ``config``: Return value of :meth:`configure` (dictionary of
              configuration parameters)
            - ``paramlist``: :attr:`fit_results`
            - ``chisq``: :attr:`chisq`
            - ``mca_areas``: :attr:`chisq`

        :return: Dictionary of fit results and fit configuration parameters
        """
        # FIXME: the returned dict contains pointers to attributes, instead of copies. Is this good?
        result = {}
        result['xbegin'] = min(self.xdata[0], self.xdata[-1])
        result['xend'] = max(self.xdata[0], self.xdata[-1])
        try:
            result['fitstate'] = self.state
        except AttributeError:
            result['fitstate'] = 'Unknown'
        result['fitconfig'] = self.fitconfig
        result['config'] = self.configure()
        result['paramlist'] = self.fit_results
        result['chisq'] = self.chisq
        result['mca_areas'] = self.mcagetareas()

        return result

    def mcagetareas(self, x=None, y=None, sigmay=None, paramlist=None):
        # TODO document
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if sigmay is None:
            sigmay = self.sigmay
        if paramlist is None:
            paramlist = self.fit_results
        groups = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                if int(float(param['group'])) != 0:
                    if param['group'] not in groups:
                        groups.append(param['group'])

        result = []
        for group in groups:
            noigno = []
            pos = 0
            fwhm = 0
            for param in paramlist:
                if param['group'] != group:
                    if param['code'] != 'IGNORE':
                        noigno.append(param['fitresult'])
                else:
                    if 'Position' in param['name']:
                        pos = param['fitresult']
                    if 'Fwhm' in param['name']:
                        fwhm = param['fitresult']
            # now I add everything around +/- 4 sigma
            # around the peak position
            sigma = fwhm / 2.354
            xmin = max(pos - 3.99 * sigma, min(x))
            xmax = min(pos + 3.99 * sigma, max(x))
            idx = numpy.nonzero((x >= xmin) & (x <= xmax))[0]
            x_around = numpy.take(x, idx)
            y_around = numpy.take(y, idx)
            ybkg_around = numpy.take(self.fitfunction(x, *noigno), idx)
            neto = y_around - ybkg_around
            deltax = x_around[1:] - x_around[0:-1]
            area = numpy.sum(neto[0:-1] * deltax)
            sigma_area = (numpy.sqrt(numpy.sum(y_around)))
            result.append([pos, area, sigma_area, fwhm])

        return result

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

    def guess_fwhm(self, y=None):
        """Return a best guess of the full-width at half maximum,
        in number of samples.

        The algorithm removes the background, then finds a global maximum
        and its corresponding full width at half maximum.

        :param y: Data to be used for guessing the fwhm. If ``None``,
            use :attr:`self.ydata`.
        :return: Estimation of full-width at half maximum, based on fwhm of
            the global maximum.
        """
        if y is None:
            y = self.ydata
        # set at least a default value for the fwhm
        fwhm = 4

        # remove data background (computed with a strip filter)
        background = strip(y, w=1, niterations=1000)
        yfit = y - background

        # basic peak search: find the global maximum
        maximum = max(yfit)
        # find indices of all values == maximum
        idx = numpy.nonzero(yfit == maximum)[0]
        # take the last one
        posindex = idx[-1]
        height = yfit[posindex]

        # now find the width of the peak at half maximum
        imin = posindex
        while yfit[imin] > 0.5 * height and imin > 0:
            imin -= 1
        imax = posindex
        while yfit[imax] > 0.5 * height and imax < len(yfit) - 1:
            imax += 1

        fwhm = max(imax - imin - 1, fwhm)

        return fwhm

    def mcaresidualssearch(self, x=None, y=None, sigmay=None, paramlist=None):
        # Todo: document or remove
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if sigmay is None:
            sigmay = self.sigmay
        if paramlist is None:
            paramlist = self.fit_results

        groups = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                if int(float(param['group'])) != 0:
                    if param['group'] not in groups:
                        groups.append(param['group'])

        newpar = []
        newcodes = []
        if self.fitconfig['fitbkg'] == 'Square Filter':
            return newpar, newcodes
        areanotdone = 1

        # estimate the fwhm
        fwhm = 10
        fwhmcode = 'POSITIVE'
        fwhmcons1 = 0
        fwhmcons2 = 0
        i = -1
        peaks = []
        for param in paramlist:
            i += 1
            pname = param['name']
            if 'Fwhm' in param['name']:
                fwhm = param['fitresult']
                if param['code'] in ['FREE', 'FIXED', 'QUOTED', 'POSITIVE',
                                     0, 1, 2, 3]:
                    fwhmcode = 'FACTOR'
                    fwhmcons1 = i
                    fwhmcons2 = 1.0
            if pname.find('Position') != -1:
                peaks.append(param['fitresult'])

        # calculate the residuals
        yfit = self.gendata(x=x, paramlist=paramlist)

        residuals = (y - yfit) / (sigmay + numpy.equal(sigmay, 0.0))

        # set to zero all the residuals around peaks
        for peak in peaks:
            idx = numpy.less(x, peak - 0.8 * fwhm) + \
                numpy.greater(x, peak + 0.8 * fwhm)
            yfit *= idx
            y *= idx
            residuals *= idx

        # estimate the position
        maxres = max(residuals)
        idx = numpy.nonzero(residuals == maxres)[0]
        pos = numpy.take(x, idx)[-1]

        # estimate the height!
        height = numpy.take(y - yfit, idx)[-1]
        if (height <= 0):
            return newpar, newcodes

        for pname in self.theorydict[self.fitconfig['fittheory']][1]:
            if pname.find('Position') != -1:
                estimation = pos
                code = 'QUOTED'
                cons1 = pos - 0.5 * fwhm
                cons2 = pos + 0.5 * fwhm
            elif pname.find('Area') != -1:
                if areanotdone:
                    areanotdone = 0
                    area = (height * fwhm / (2.0 * numpy.sqrt(2 * numpy.log(2)))) * \
                        numpy.sqrt(2 * numpy.pi)
                    if area <= 0:
                        return [], [[], [], []]
                    estimation = area
                    code = 'POSITIVE'
                    cons1 = 0.0
                    cons2 = 0.0
                else:
                    estimation = 0.0
                    code = 'FIXED'
                    cons1 = 0.0
                    cons2 = 0.0
            elif 'Fwhm' in pname:
                estimation = fwhm
                code = fwhmcode
                cons1 = fwhmcons1
                cons2 = fwhmcons2
            else:
                estimation = 0.0
                code = 'FIXED'
                cons1 = 0.0
                cons2 = 0.0
            newpar.append(estimation)
            newcodes.append([code, cons1, cons2])
        return newpar, newcodes

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
    from . import SpecfitFunctions

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)
    constant_background = 3.14
    p = [1500, 100., 50,
         1000, 700., 30.5,
         314, 800.5, 15]
    y = constant_background + sum_gauss(x, *p)

    # Fitting
    fit = Specfit()
    fit.setdata(x=x, y=y)
    fit.importfun(SpecfitFunctions.__file__)
    fit.settheory('Gaussians')
    fit.setbackground('Constant')
    fit.estimate()
    fit.startfit()

    print("Searched parameters = ", [3.14, 1500, 100., 50.0, 1000, 700., 30.5, 314, 800.5, 15])
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.fit_results:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)

    # Plot
    constant_background = dummy_list[0]
    p1 = dummy_list[1:]
    y2 = constant_background + sum_gauss(x, *p1)

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
