# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This module implements a Levenberg-Marquardt algorithm with constraints on the
fitted parameters without introducing any other dependendency than numpy.

If scipy dependency is not an issue, and no constraints are applied to the fitting
parameters, there is no real gain compared to the use of scipy.optimize.curve_fit
other than a more conservative calculation of uncertainties on fitted parameters.

This module is a refactored version of PyMca Gefit.py module.
"""
__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "15/05/2017"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
from numpy.linalg import inv
from numpy.linalg.linalg import LinAlgError
import time
import logging
import copy

_logger = logging.getLogger(__name__)

# codes understood by the routine
CFREE       = 0
CPOSITIVE   = 1
CQUOTED     = 2
CFIXED      = 3
CFACTOR     = 4
CDELTA      = 5
CSUM        = 6
CIGNORED    = 7

def leastsq(model, xdata, ydata, p0, sigma=None,
              constraints=None, model_deriv=None, epsfcn=None,
              deltachi=None, full_output=None,
              check_finite=True,
              left_derivative=False,
              max_iter=100):
    """
    Use non-linear least squares Levenberg-Marquardt algorithm to fit a function, f, to
    data with optional constraints on the fitted parameters.

    Assumes ``ydata = f(xdata, *params) + eps``

    :param model: callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
        The returned value is a one dimensional array of floats.

    :param xdata: An M-length sequence.
        The independent variable where the data is measured.

    :param ydata: An M-length sequence
        The dependent data --- nominally f(xdata, ...)

    :param p0: N-length sequence
        Initial guess for the parameters.

    :param sigma: None or M-length sequence, optional
        If not None, the uncertainties in the ydata array. These are used as
        weights in the least-squares problem
        i.e. minimising ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )``
        If None, the uncertainties are assumed to be 1

    :param constraints:
        If provided, it is a 2D sequence of dimension (n_parameters, 3) where,
        for each parameter denoted by the index i, the meaning is

                     - constraints[i][0]

                        - 0 - Free (CFREE)
                        - 1 - Positive (CPOSITIVE)
                        - 2 - Quoted (CQUOTED)
                        - 3 - Fixed (CFIXED)
                        - 4 - Factor (CFACTOR)
                        - 5 - Delta (CDELTA)
                        - 6 - Sum (CSUM)


                     - constraints[i][1]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Min value of the parameter if constraints[i][0] is CQUOTED
                        - Index of fitted parameter to which it is related

                     - constraints[i][2]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Max value of the parameter if constraints[i][0] is CQUOTED
                        - Factor to apply to related parameter with index constraints[i][1]
                        - Difference with parameter with index constraints[i][1]
                        - Sum obtained when adding parameter with index constraints[i][1]
    :type constraints: *optional*, None or 2D sequence

    :param model_deriv:
        None (default) or function providing the derivatives of the fitting function respect to the fitted parameters.
        It will be called as model_deriv(xdata, parameters, index) where parameters is a sequence with the current
        values of the fitting parameters, index is the fitting parameter index for which the the derivative has
        to be provided in the supplied array of xdata points.
    :type model_deriv: *optional*, None or callable


    :param epsfcn: float
        A variable used in determining a suitable parameter variation when
        calculating the numerical derivatives (for model_deriv=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        Original Gefit module was using epsfcn 1.0e-5 while default value
        is now numpy.finfo(numpy.float).eps as in scipy
    :type epsfcn: *optional*, float

    :param deltachi: float
        A variable used to control the minimum change in chisq to consider the
        fitting process not worth to be continued. Default is 0.1 %.
    :type deltachi: *optional*, float

    :param full_output: bool, optional
        non-zero to return all optional outputs. The default is None what will give a warning in case
        of a constrained fit without having set this kweyword.

    :param check_finite: bool, optional
            If True, check that the input arrays do not contain nans of infs,
            and raise a ValueError if they do. Setting this parameter to
            False will ignore input arrays values containing nans.
            Default is True.

    :param left_derivative:
            This parameter only has an influence if no derivative function
            is provided. When True the left and right derivatives of the
            model will be calculated for each fitted parameters thus leading to
            the double number of function evaluations. Default is False.
            Original Gefit module was always using left_derivative as True.
    :type left_derivative: *optional*, bool

    :param max_iter: Maximum number of iterations (default is 100)

    :return: Returns a tuple of length 2 (or 3 if full_ouput is True) with the content:

         ``popt``: array
           Optimal values for the parameters so that the sum of the squared error
           of ``f(xdata, *popt) - ydata`` is minimized
         ``pcov``: 2d array
           If no constraints are applied, this array contains the estimated covariance
           of popt. The diagonal provides the variance of the parameter estimate.
           To compute one standard deviation errors use ``perr = np.sqrt(np.diag(pcov))``.
           If constraints are applied, this array does not contain the estimated covariance of
           the parameters actually used during the fitting process but the uncertainties after
           recalculating the covariance if all the parameters were free.
           To get the actual uncertainties following error propagation of the actually fitted
           parameters one should set full_output to True and access the uncertainties key.
         ``infodict``: dict
           a dictionary of optional outputs with the keys:

            ``uncertainties``
                The actual uncertainty on the optimized parameters.
            ``nfev``
                The number of function calls
            ``fvec``
                The function evaluated at the output
            ``niter``
                The number of iterations performed
            ``chisq``
                The chi square ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )``
            ``reduced_chisq``
                The chi square ``np.sum( ((f(xdata, *popt) - ydata) / sigma)**2 )`` divided
                by the number of degrees of freedom ``(M - number_of_free_parameters)``
    """
    function_call_counter = 0
    if numpy.isscalar(p0):
        p0 = [p0]
    parameters = numpy.array(p0, dtype=numpy.float64, copy=False)
    if deltachi is None:
        deltachi = 0.001

    # NaNs can not be handled
    if check_finite:
        xdata = numpy.asarray_chkfinite(xdata)
        ydata = numpy.asarray_chkfinite(ydata)
        if sigma is not None:
            sigma = numpy.asarray_chkfinite(sigma)
        else:
            sigma = numpy.ones((ydata.shape), dtype=numpy.float)
        ydata.shape = -1
        sigma.shape = -1
    else:
        ydata = numpy.asarray(ydata)
        xdata = numpy.asarray(xdata)
        ydata.shape = -1
        if sigma is not None:
            sigma = numpy.asarray(sigma)
        else:
            sigma = numpy.ones((ydata.shape), dtype=numpy.float)
        sigma.shape = -1
        # get rid of NaN in input data
        idx = numpy.isfinite(ydata)
        if False in idx:
            # xdata must have a shape able to be understood by the user function
            # in principle, one should not need to change it, however, if there are
            # points to be excluded, one has to be able to exclude them.
            # We can only hope that the sequence is properly arranged
            if xdata.size == ydata.size:
                if len(xdata.shape) != 1:
                    msg = "Need to reshape input xdata."
                    _logger.warning(msg)
                xdata.shape = -1
            else:
                raise ValueError("Cannot reshape xdata to deal with NaN in ydata")
            ydata = ydata[idx]
            xdata = xdata[idx]
            sigma = sigma[idx]
        idx = numpy.isfinite(sigma)
        if False in idx:
            # xdata must have a shape able to be understood by the user function
            # in principle, one should not need to change it, however, if there are
            # points to be excluded, one has to be able to exclude them.
            # We can only hope that the sequence is properly arranged
            ydata = ydata[idx]
            xdata = xdata[idx]
            sigma = sigma[idx]
        idx = numpy.isfinite(xdata)
        filter_xdata = False
        if False in idx:
            # What to do?
            try:
                # Let's see if the function is able to deal with non-finite data
                msg = "Checking if function can deal with non-finite data"
                _logger.debug(msg)
                evaluation = model(xdata, *parameters)
                function_call_counter += 1
                if evaluation.shape != ydata.shape:
                    if evaluation.size == ydata.size:
                        msg = "Supplied function does not return a proper array of floats."
                        msg += "\nFunction should be rewritten to return a 1D array of floats."
                        msg += "\nTrying to reshape output."
                        _logger.warning(msg)
                        evaluation.shape = ydata.shape
                if False in numpy.isfinite(evaluation):
                    msg = "Supplied function unable to handle non-finite x data"
                    msg += "\nAttempting to filter out those x data values."
                    _logger.warning(msg)
                    filter_xdata = True
                else:
                    filter_xdata = False
                evaluation = None
            except:
                # function cannot handle input data
                filter_xdata = True
        if filter_xdata:
            if xdata.size != ydata.size:
                raise ValueError("xdata contains non-finite data that cannot be filtered")
            else:
                # we leave the xdata as they where
                old_shape = xdata.shape
                xdata.shape = ydata.shape
                idx0 = numpy.isfinite(xdata)
                xdata.shape = old_shape
            ydata = ydata[idx0]
            xdata = xdata[idx]
            sigma = sigma[idx0]
    weight = 1.0 / (sigma + numpy.equal(sigma, 0))
    weight0 = weight * weight

    nparameters = len(parameters)

    if epsfcn is None:
        epsfcn = numpy.finfo(numpy.float).eps
    else:
        epsfcn = max(epsfcn, numpy.finfo(numpy.float).eps)

    # check if constraints have been passed as text
    constrained_fit = False
    if constraints is not None:
        # make sure we work with a list of lists
        input_constraints = constraints
        tmp_constraints = [None] * len(input_constraints)
        for i in range(nparameters):
            tmp_constraints[i] = list(input_constraints[i])
        constraints = tmp_constraints
        for i in range(nparameters):
            if hasattr(constraints[i][0], "upper"):
                txt = constraints[i][0].upper()
                if txt == "FREE":
                    constraints[i][0] = CFREE
                elif txt == "POSITIVE":
                    constraints[i][0] = CPOSITIVE
                elif txt == "QUOTED":
                    constraints[i][0] = CQUOTED
                elif txt == "FIXED":
                    constraints[i][0] = CFIXED
                elif txt == "FACTOR":
                    constraints[i][0] = CFACTOR
                    constraints[i][1] = int(constraints[i][1])
                elif txt == "DELTA":
                    constraints[i][0] = CDELTA
                    constraints[i][1] = int(constraints[i][1])
                elif txt == "SUM":
                    constraints[i][0] = CSUM
                    constraints[i][1] = int(constraints[i][1])
                elif txt in ["IGNORED", "IGNORE"]:
                    constraints[i][0] = CIGNORED
                else:
                    #I should raise an exception
                    raise ValueError("Unknown constraint %s" % constraints[i][0])
            if constraints[i][0] > 0:
                constrained_fit = True
    if constrained_fit:
        if full_output is None:
            _logger.info("Recommended to set full_output to True when using constraints")

    # Levenberg-Marquardt algorithm
    fittedpar = parameters.__copy__()
    flambda = 0.001
    iiter = max_iter
    #niter = 0
    last_evaluation=None
    x = xdata
    y = ydata
    chisq0 = -1
    iteration_counter = 0
    while (iiter > 0):
        weight = weight0
        """
        I cannot evaluate the initial chisq here because I do not know
        if some parameters are to be ignored, otherways I could do it as follows:
        if last_evaluation is None:
            yfit = model(x, *fittedpar)
            last_evaluation = yfit
            chisq0 = (weight * pow(y-yfit, 2)).sum()
        and chisq would not need to be recalculated.
        Passing the last_evaluation assumes that there are no parameters being
        ignored or not between calls.
        """
        iteration_counter += 1
        chisq0, alpha0, beta, internal_output = chisq_alpha_beta(
                                                 model, fittedpar,
                                                 x, y, weight, constraints=constraints,
                                                 model_deriv=model_deriv,
                                                 epsfcn=epsfcn,
                                                 left_derivative=left_derivative,
                                                 last_evaluation=last_evaluation,
                                                 full_output=True)
        n_free = internal_output["n_free"]
        free_index = internal_output["free_index"]
        noigno = internal_output["noigno"]
        fitparam = internal_output["fitparam"]
        function_calls = internal_output["function_calls"]
        function_call_counter += function_calls
        #print("chisq0 = ", chisq0, n_free, fittedpar)
        #raise
        nr, nc = alpha0.shape
        flag = 0
        #lastdeltachi = chisq0
        while flag == 0:
            alpha = alpha0 * (1.0 + flambda * numpy.identity(nr))
            deltapar = numpy.dot(beta, inv(alpha))
            if constraints is None:
                newpar = fitparam + deltapar [0]
            else:
                newpar = parameters.__copy__()
                pwork = numpy.zeros(deltapar.shape, numpy.float)
                for i in range(n_free):
                    if constraints is None:
                        pwork [0] [i] = fitparam [i] + deltapar [0] [i]
                    elif constraints [free_index[i]][0] == CFREE:
                        pwork [0] [i] = fitparam [i] + deltapar [0] [i]
                    elif constraints [free_index[i]][0] == CPOSITIVE:
                        #abs method
                        pwork [0] [i] = fitparam [i] + deltapar [0] [i]
                        #square method
                        #pwork [0] [i] = (numpy.sqrt(fitparam [i]) + deltapar [0] [i]) * \
                        #                (numpy.sqrt(fitparam [i]) + deltapar [0] [i])
                    elif constraints[free_index[i]][0] == CQUOTED:
                        pmax = max(constraints[free_index[i]][1],
                                   constraints[free_index[i]][2])
                        pmin = min(constraints[free_index[i]][1],
                                   constraints[free_index[i]][2])
                        A = 0.5 * (pmax + pmin)
                        B = 0.5 * (pmax - pmin)
                        if B != 0:
                            pwork [0] [i] = A + \
                                        B * numpy.sin(numpy.arcsin((fitparam[i] - A)/B)+ \
                                        deltapar [0] [i])
                        else:
                            txt = "Error processing constrained fit\n"
                            txt += "Parameter limits are %g and %g\n" % (pmin, pmax)
                            txt += "A = %g B = %g"  % (A, B)
                            raise ValueError("Invalid parameter limits")
                    newpar[free_index[i]] = pwork [0] [i]
                newpar = numpy.array(_get_parameters(newpar, constraints))
            workpar = numpy.take(newpar, noigno)
            yfit = model(x, *workpar)
            if last_evaluation is None:
                if len(yfit.shape) > 1:
                    msg = "Supplied function does not return a 1D array of floats."
                    msg += "\nFunction should be rewritten."
                    msg += "\nTrying to reshape output."
                    _logger.warning(msg)
            yfit.shape = -1
            function_call_counter += 1
            chisq = (weight * pow(y-yfit, 2)).sum()
            absdeltachi = chisq0 - chisq
            if absdeltachi < 0:
                flambda *= 10.0
                if flambda > 1000:
                    flag = 1
                    iiter = 0
            else:
                flag = 1
                fittedpar = newpar.__copy__()
                lastdeltachi = 100 * (absdeltachi / (chisq + (chisq == 0)))
                if iteration_counter < 2:
                    # ignore any limit, the fit *has* to be improved
                    pass
                elif (lastdeltachi) < deltachi:
                    iiter = 0
                elif absdeltachi < numpy.sqrt(epsfcn):
                    iiter = 0
                    _logger.info("Iteration finished due to too small absolute chi decrement")
                chisq0 = chisq
                flambda = flambda / 10.0
                last_evaluation = yfit
            iiter = iiter - 1
    # this is the covariance matrix of the actually fitted parameters
    cov0 = inv(alpha0)
    if constraints is None:
        cov = cov0
    else:
        # yet another call needed with all the parameters being free except those
        # that are FIXED and that will be assigned a 100 % uncertainty.
        new_constraints = copy.deepcopy(constraints)
        flag_special = [0] * len(fittedpar)
        for idx, constraint in enumerate(constraints):
            if constraints[idx][0] in [CFIXED, CIGNORED]:
                flag_special[idx] = constraints[idx][0]
            else:
                new_constraints[idx][0] = CFREE
                new_constraints[idx][1] = 0
                new_constraints[idx][2] = 0
        chisq, alpha, beta, internal_output = chisq_alpha_beta(
                                                 model, fittedpar,
                                                 x, y, weight, constraints=new_constraints,
                                                 model_deriv=model_deriv,
                                                 epsfcn=epsfcn,
                                                 left_derivative=left_derivative,
                                                 last_evaluation=last_evaluation,
                                                 full_output=True)
        # obtained chisq should be identical to chisq0
        try:
            cov = inv(alpha)
        except LinAlgError:
            _logger.critical("Error calculating covariance matrix after successful fit")
            cov = None
        if cov is not None:
            for idx, value in enumerate(flag_special):
                if value in [CFIXED, CIGNORED]:
                    cov = numpy.insert(numpy.insert(cov, idx, 0, axis=1), idx, 0, axis=0)
                    cov[idx, idx] = fittedpar[idx] * fittedpar[idx]

    if not full_output:
        return fittedpar, cov
    else:
        sigma0 = numpy.sqrt(abs(numpy.diag(cov0)))
        sigmapar = _get_sigma_parameters(fittedpar, sigma0, constraints)
        ddict = {}
        ddict["chisq"] = chisq0
        ddict["reduced_chisq"] = chisq0 / (len(yfit)-n_free)
        ddict["covariance"] = cov0
        ddict["uncertainties"] = sigmapar
        ddict["fvec"] = last_evaluation
        ddict["nfev"] = function_call_counter
        ddict["niter"] = iteration_counter
        return fittedpar, cov, ddict #, chisq/(len(yfit)-len(sigma0)), sigmapar,niter,lastdeltachi

def chisq_alpha_beta(model, parameters, x, y, weight, constraints=None,
                   model_deriv=None, epsfcn=None, left_derivative=False,
                   last_evaluation=None, full_output=False):

    """
    Get chi square, the curvature matrix alpha and the matrix beta according to the input parameters.
    If all the parameters are unconstrained, the covariance matrix is the inverse of the alpha matrix.

    :param model: callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
        The returned value is a one dimensional array of floats.

    :param parameters: N-length sequence
        Values of parameters at which function and derivatives are to be calculated.

    :param x: An M-length sequence.
        The independent variable where the data is measured.

    :param y: An M-length sequence
        The dependent data --- nominally f(xdata, ...)

    :param weight: M-length sequence
        Weights to be applied in the calculation of chi square
        As a reminder ``chisq = np.sum(weigth * (model(x, *parameters) - y)**2)``

    :param constraints:
        If provided, it is a 2D sequence of dimension (n_parameters, 3) where,
        for each parameter denoted by the index i, the meaning is

                     - constraints[i][0]

                        - 0 - Free (CFREE)
                        - 1 - Positive (CPOSITIVE)
                        - 2 - Quoted (CQUOTED)
                        - 3 - Fixed (CFIXED)
                        - 4 - Factor (CFACTOR)
                        - 5 - Delta (CDELTA)
                        - 6 - Sum (CSUM)


                     - constraints[i][1]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Min value of the parameter if constraints[i][0] is CQUOTED
                        - Index of fitted parameter to which it is related

                     - constraints[i][2]

                        - Ignored if constraints[i][0] is 0, 1, 3
                        - Max value of the parameter if constraints[i][0] is CQUOTED
                        - Factor to apply to related parameter with index constraints[i][1]
                        - Difference with parameter with index constraints[i][1]
                        - Sum obtained when adding parameter with index constraints[i][1]
    :type constraints: *optional*, None or 2D sequence

    :param model_deriv:
        None (default) or function providing the derivatives of the fitting function respect to the fitted parameters.
        It will be called as model_deriv(xdata, parameters, index) where parameters is a sequence with the current
        values of the fitting parameters, index is the fitting parameter index for which the the derivative has
        to be provided in the supplied array of xdata points.
    :type model_deriv: *optional*, None or callable


    :param epsfcn: float
        A variable used in determining a suitable parameter variation when
        calculating the numerical derivatives (for model_deriv=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        Original Gefit module was using epsfcn 1.0e-10 while default value
        is now numpy.finfo(numpy.float).eps as in scipy
    :type epsfcn: *optional*, float

    :param left_derivative:
            This parameter only has an influence if no derivative function
            is provided. When True the left and right derivatives of the
            model will be calculated for each fitted parameters thus leading to
            the double number of function evaluations. Default is False.
            Original Gefit module was always using left_derivative as True.
    :type left_derivative: *optional*, bool

    :param last_evaluation: An M-length array
            Used for optimization purposes. If supplied, this array will be taken as the result of
            evaluating the function, that is as the result of ``model(x, *parameters)`` thus avoiding
            the evaluation call.

    :param full_output: bool, optional
            Additional output used for internal purposes with the keys:
        ``function_calls``
            The number of model function calls performed.
        ``fitparam``
            A sequence with the actual free parameters
        ``free_index``
            Sequence with the indices of the free parameters in input parameters sequence.
        ``noigno``
            Sequence with the indices of the original parameters considered in the calculations.
    """
    if epsfcn is None:
        epsfcn = numpy.finfo(numpy.float).eps
    else:
        epsfcn = max(epsfcn, numpy.finfo(numpy.float).eps)
    #nr0, nc = data.shape
    n_param = len(parameters)
    if constraints is None:
        derivfactor = numpy.ones((n_param, ))
        n_free = n_param
        noigno = numpy.arange(n_param)
        free_index = noigno * 1
        fitparam = parameters * 1
    else:
        n_free = 0
        fitparam = []
        free_index = []
        noigno = []
        derivfactor = []
        for i in range(n_param):
            if constraints[i][0] != CIGNORED:
                noigno.append(i)
            if constraints[i][0] == CFREE:
                fitparam.append(parameters [i])
                derivfactor.append(1.0)
                free_index.append(i)
                n_free += 1
            elif constraints[i][0] == CPOSITIVE:
                fitparam.append(abs(parameters[i]))
                derivfactor.append(1.0)
                #fitparam.append(numpy.sqrt(abs(parameters[i])))
                #derivfactor.append(2.0*numpy.sqrt(abs(parameters[i])))
                free_index.append(i)
                n_free += 1
            elif constraints[i][0] == CQUOTED:
                pmax = max(constraints[i][1], constraints[i][2])
                pmin  =min(constraints[i][1], constraints[i][2])
                if ((pmax-pmin) > 0) & \
                   (parameters[i] <= pmax) & \
                   (parameters[i] >= pmin):
                    A = 0.5 * (pmax + pmin)
                    B = 0.5 * (pmax - pmin)
                    fitparam.append(parameters[i])
                    derivfactor.append(B*numpy.cos(numpy.arcsin((parameters[i] - A)/B)))
                    free_index.append(i)
                    n_free += 1
                elif (pmax-pmin) > 0:
                    print("WARNING: Quoted parameter outside boundaries")
                    print("Initial value = %f" % parameters[i])
                    print("Limits are %f and %f" % (pmin, pmax))
                    print("Parameter will be kept at its starting value")
    fitparam = numpy.array(fitparam, numpy.float)
    alpha = numpy.zeros((n_free, n_free), numpy.float)
    beta = numpy.zeros((1, n_free), numpy.float)
    #delta = (fitparam + numpy.equal(fitparam, 0.0)) * 0.00001
    delta = (fitparam + numpy.equal(fitparam, 0.0)) * numpy.sqrt(epsfcn)
    nr  = y.size
    ##############
    # Prior to each call to the function one has to re-calculate the
    # parameters
    pwork = parameters.__copy__()
    for i in range(n_free):
        pwork [free_index[i]] = fitparam [i]
    if n_free == 0:
        raise ValueError("No free parameters to fit")
    function_calls = 0
    if not left_derivative:
        if last_evaluation is not None:
            f2 = last_evaluation
        else:
            f2 = model(x, *parameters)
            f2.shape = -1
            function_calls += 1
    for i in range(n_free):
        if model_deriv is None:
            #pwork = parameters.__copy__()
            pwork[free_index[i]] = fitparam [i] + delta [i]
            newpar = _get_parameters(pwork.tolist(), constraints)
            newpar = numpy.take(newpar, noigno)
            f1 = model(x, *newpar)
            f1.shape = -1
            function_calls += 1
            if left_derivative:
                pwork[free_index[i]] = fitparam [i] - delta [i]
                newpar = _get_parameters(pwork.tolist(), constraints)
                newpar=numpy.take(newpar, noigno)
                f2 = model(x, *newpar)
                function_calls += 1
                help0 = (f1 - f2) / (2.0 * delta[i])
            else:
                help0 = (f1 - f2) / (delta[i])
            help0 = help0 * derivfactor[i]
            pwork[free_index[i]] = fitparam [i]
            #removed I resize outside the loop:
            #help0 = numpy.resize(help0, (1, nr))
        else:
            help0 = model_deriv(x, pwork, free_index[i])
            help0 = help0 * derivfactor[i]

        if i == 0:
            deriv = help0
        else:
            deriv = numpy.concatenate((deriv, help0), 0)

    #line added to resize outside the loop
    deriv = numpy.resize(deriv, (n_free, nr))
    if last_evaluation is None:
        if constraints is None:
            yfit = model(x, *fitparam)
            yfit.shape = -1
        else:
            newpar = _get_parameters(pwork.tolist(), constraints)
            newpar = numpy.take(newpar, noigno)
            yfit = model(x, *newpar)
            yfit.shape = -1
        function_calls += 1
    else:
        yfit = last_evaluation
    deltay = y - yfit
    help0 = weight * deltay
    for i in range(n_free):
        derivi = numpy.resize(deriv[i, :], (1, nr))
        help1 = numpy.resize(numpy.sum((help0 * derivi), 1), (1, 1))
        if i == 0:
            beta = help1
        else:
            beta = numpy.concatenate((beta, help1), 1)
        help1 = numpy.inner(deriv, weight*derivi)
        if i == 0:
            alpha = help1
        else:
            alpha = numpy.concatenate((alpha, help1), 1)
    chisq = (help0 * deltay).sum()
    if full_output:
        ddict = {}
        ddict["n_free"] = n_free
        ddict["free_index"] = free_index
        ddict["noigno"] = noigno
        ddict["fitparam"] = fitparam
        ddict["derivfactor"] = derivfactor
        ddict["function_calls"] = function_calls
        return chisq, alpha, beta, ddict
    else:
        return chisq, alpha, beta


def _get_parameters(parameters, constraints):
    """
    Apply constraints to input parameters.

    Parameters not depending on other parameters, they are returned as the input.

    Parameters depending on other parameters, return the value after applying the
    relation to the parameter wo which they are related.
    """
    # 0 = Free       1 = Positive     2 = Quoted
    # 3 = Fixed      4 = Factor       5 = Delta
    if constraints is None:
        return parameters * 1
    newparam = []
    #first I make the free parameters
    #because the quoted ones put troubles
    for i in range(len(constraints)):
        if constraints[i][0] == CFREE:
            newparam.append(parameters[i])
        elif constraints[i][0] == CPOSITIVE:
            #newparam.append(parameters[i] * parameters[i])
            newparam.append(abs(parameters[i]))
        elif constraints[i][0] == CQUOTED:
            newparam.append(parameters[i])
        elif abs(constraints[i][0]) == CFIXED:
            newparam.append(parameters[i])
        else:
            newparam.append(parameters[i])
    for i in range(len(constraints)):
        if constraints[i][0] == CFACTOR:
            newparam[i] = constraints[i][2] * newparam[int(constraints[i][1])]
        elif constraints[i][0] == CDELTA:
            newparam[i] = constraints[i][2] + newparam[int(constraints[i][1])]
        elif constraints[i][0] == CIGNORED:
            # The whole ignored stuff should not be documented because setting
            # a parameter to 0 is not the same as being ignored.
            # Being ignored should imply the parameter is simply not accounted for
            # and should be stripped out of the list of parameters by the program
            # using this module
            newparam[i] = 0
        elif constraints[i][0] == CSUM:
            newparam[i] = constraints[i][2]-newparam[int(constraints[i][1])]
    return newparam


def _get_sigma_parameters(parameters, sigma0, constraints):
    """
    Internal function propagating the uncertainty on the actually fitted parameters and related parameters to the
    final parameters considering the applied constraints.

    Parameters
    ----------
        parameters : 1D sequence of length equal to the number of free parameters N
            The parameters actually used in the fitting process.
        sigma0 : 1D sequence of length N
            Uncertainties calculated as the square-root of the diagonal of
            the covariance matrix
        constraints : The set of constraints applied in the fitting process
    """
    # 0 = Free       1 = Positive     2 = Quoted
    # 3 = Fixed      4 = Factor       5 = Delta
    if constraints is None:
        return sigma0
    n_free = 0
    sigma_par = numpy.zeros(parameters.shape, numpy.float)
    for i in range(len(constraints)):
        if constraints[i][0] == CFREE:
            sigma_par [i] = sigma0[n_free]
            n_free += 1
        elif constraints[i][0] == CPOSITIVE:
            #sigma_par [i] = 2.0 * sigma0[n_free]
            sigma_par [i] = sigma0[n_free]
            n_free += 1
        elif constraints[i][0] == CQUOTED:
            pmax = max(constraints [i][1], constraints [i][2])
            pmin = min(constraints [i][1], constraints [i][2])
            # A = 0.5 * (pmax + pmin)
            B = 0.5 * (pmax - pmin)
            if (B > 0) & (parameters [i] < pmax) & (parameters [i] > pmin):
                sigma_par [i] = abs(B * numpy.cos(parameters[i]) * sigma0[n_free])
                n_free += 1
            else:
                sigma_par [i] = parameters[i]
        elif abs(constraints[i][0]) == CFIXED:
            sigma_par[i] = parameters[i]
    for i in range(len(constraints)):
        if constraints[i][0] == CFACTOR:
            sigma_par [i] = constraints[i][2]*sigma_par[int(constraints[i][1])]
        elif constraints[i][0] == CDELTA:
            sigma_par [i] = sigma_par[int(constraints[i][1])]
        elif constraints[i][0] == CSUM:
            sigma_par [i] = sigma_par[int(constraints[i][1])]
    return sigma_par


def main(argv=None):
    if argv is None:
        npoints = 10000
    elif hasattr(argv, "__len__"):
        if len(argv) > 1:
            npoints = int(argv[1])
        else:
            print("Usage:")
            print("fit [npoints]")
    else:
        # expected a number
        npoints = argv

    def gauss(t0, *param0):
        param = numpy.array(param0)
        t = numpy.array(t0)
        dummy = 2.3548200450309493 * (t - param[3]) / param[4]
        return param[0] + param[1] * t + param[2] * myexp(-0.5 * dummy * dummy)


    def myexp(x):
        # put a (bad) filter to avoid over/underflows
        # with no python looping
        return numpy.exp(x * numpy.less(abs(x), 250)) -\
               1.0 * numpy.greater_equal(abs(x), 250)

    xx = numpy.arange(npoints, dtype=numpy.float)
    yy = gauss(xx, *[10.5, 2, 1000.0, 20., 15])
    sy = numpy.sqrt(abs(yy))
    parameters = [0.0, 1.0, 900.0, 25., 10]
    stime = time.time()

    fittedpar, cov, ddict = leastsq(gauss, xx, yy, parameters,
                                                 sigma=sy,
                                                 left_derivative=False,
                                                 full_output=True,
                                                 check_finite=True)
    etime = time.time()
    sigmapars = numpy.sqrt(numpy.diag(cov))
    print("Took ", etime - stime, "seconds")
    print("Function calls  = ", ddict["nfev"])
    print("chi square  = ", ddict["chisq"])
    print("Fitted pars = ", fittedpar)
    print("Sigma pars  = ", sigmapars)
    try:
        from scipy.optimize import curve_fit as cfit
        SCIPY = True
    except ImportError:
        SCIPY = False
    if SCIPY:
        counter = 0
        stime = time.time()
        scipy_fittedpar, scipy_cov = cfit(gauss,
                                      xx,
                                      yy,
                                      parameters,
                                      sigma=sy)
        etime = time.time()
        print("Scipy Took ", etime - stime, "seconds")
        print("Counter = ", counter)
        print("scipy = ", scipy_fittedpar)
        print("Sigma = ", numpy.sqrt(numpy.diag(scipy_cov)))

if __name__ == "__main__":
    main()
