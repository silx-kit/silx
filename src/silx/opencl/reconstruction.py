#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Module for tomographic reconstruction algorithms"""

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "01/08/2019"

import logging
import numpy as np

from .common import pyopencl
from .processing import OpenclProcessing
from .backprojection import Backprojection
from .projection import Projection
from .linalg import LinAlg

import pyopencl.array as parray
from pyopencl.elementwise import ElementwiseKernel
logger = logging.getLogger(__name__)

cl = pyopencl


class ReconstructionAlgorithm(OpenclProcessing):
    """
    A parent class for all iterative tomographic reconstruction algorithms

    :param sino_shape: shape of the sinogram. The sinogram is in the format
                       (n_b, n_a) where n_b is the number of detector bins and
                       n_a is the number of angles.
    :param slice_shape: Optional, shape of the reconstructed slice.
                        By default, it is a square slice where the dimension
                        is the "x dimension" of the sinogram (number of bins).
    :param axis_position: Optional, axis position. Default is `(shape[1]-1)/2.0`.
    :param angles: Optional, a list of custom angles in radian.
    :param ctx: actual working context, left to None for automatic
                initialization from device type or platformid/deviceid
    :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
    :param platformid: integer with the platform_identifier, as given by clinfo
    :param deviceid: Integer with the device identifier, as given by clinfo
    :param profile: switch on profiling to be able to profile at the kernel level,
                    store profiling elements (makes code slightly slower)
    """

    def __init__(self, sino_shape, slice_shape=None, axis_position=None, angles=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        # Create a backprojector
        self.backprojector = Backprojection(
            sino_shape,
            slice_shape=slice_shape,
            axis_position=axis_position,
            angles=angles,
            ctx=self.ctx,
            profile=profile
        )
        # Create a projector
        self.projector = Projection(
            self.backprojector.slice_shape,
            self.backprojector.angles,
            axis_position=axis_position,
            detector_width=self.backprojector.num_bins,
            normalize=False,
            ctx=self.ctx,
            profile=profile
        )
        self.sino_shape = sino_shape
        self.is_cpu = self.backprojector.is_cpu
        # Arrays
        self.d_data = parray.empty(self.queue, sino_shape, dtype=np.float32)
        self.d_data.fill(0.0)
        self.d_sino = parray.empty_like(self.d_data)
        self.d_sino.fill(0.0)
        self.d_x = parray.empty(self.queue,
                                self.backprojector.slice_shape,
                                dtype=np.float32)
        self.d_x.fill(0.0)
        self.d_x_old = parray.empty_like(self.d_x)
        self.d_x_old.fill(0.0)

        self.add_to_cl_mem({
                            "d_data": self.d_data,
                            "d_sino": self.d_sino,
                            "d_x": self.d_x,
                            "d_x_old": self.d_x_old,
                            })

    def proj(self, d_slice, d_sino):
        """
        Project d_slice to d_sino
        """
        self.projector.transfer_device_to_texture(d_slice.data)  #.wait()
        self.projector.projection(dst=d_sino)

    def backproj(self, d_sino, d_slice):
        """
        Backproject d_sino to d_slice
        """
        self.backprojector.transfer_device_to_texture(d_sino.data)  #.wait()
        self.backprojector.backprojection(dst=d_slice)


class SIRT(ReconstructionAlgorithm):
    """
    A class for the SIRT algorithm

    :param sino_shape: shape of the sinogram. The sinogram is in the format
                       (n_b, n_a) where n_b is the number of detector bins and
                       n_a is the number of angles.
    :param slice_shape: Optional, shape of the reconstructed slice.
                        By default, it is a square slice where the dimension is
                        the "x dimension" of the sinogram (number of bins).
    :param axis_position: Optional, axis position. Default is `(shape[1]-1)/2.0`.
    :param angles: Optional, a list of custom angles in radian.
    :param ctx: actual working context, left to None for automatic
                initialization from device type or platformid/deviceid
    :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
    :param platformid: integer with the platform_identifier, as given by clinfo
    :param deviceid: Integer with the device identifier, as given by clinfo
    :param profile: switch on profiling to be able to profile at the kernel level,
                    store profiling elements (makes code slightly slower)

    .. warning:: This is a beta version of the SIRT algorithm. Reconstruction
            fails for at least on CPU (Xeon E3-1245 v5) using the AMD opencl
            implementation.
    """

    def __init__(self, sino_shape, slice_shape=None, axis_position=None, angles=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):

        ReconstructionAlgorithm.__init__(self, sino_shape, slice_shape=slice_shape,
                                         axis_position=axis_position, angles=angles,
                                         ctx=ctx, devicetype=devicetype, platformid=platformid,
                                         deviceid=deviceid, profile=profile)
        self.compute_preconditioners()

    def compute_preconditioners(self):
        """
        Create a diagonal preconditioner for the projection and backprojection
        operator.
        Each term of the diagonal is the sum of the projector/backprojector
        along rows [1], i.e the projection/backprojection of an array of ones.

        [1] Jens Gregor and Thomas Benson,
            Computational Analysis and Improvement of SIRT,
            IEEE transactions on medical imaging, vol. 27, no. 7,  2008
        """

        # r_{i,i} = 1/(sum_j a_{i,j})
        slice_ones = np.ones(self.backprojector.slice_shape, dtype=np.float32)
        R = 1./self.projector.projection(slice_ones)  # could be all done on GPU, but I want extra checks
        R[np.logical_not(np.isfinite(R))] = 1.  # In the case where the rotation axis is excentred
        self.d_R = parray.to_device(self.queue, R)
        # c_{j,j} = 1/(sum_i a_{i,j})
        sino_ones = np.ones(self.sino_shape, dtype=np.float32)
        C = 1./self.backprojector.backprojection(sino_ones)
        C[np.logical_not(np.isfinite(C))] = 1.  # In the case where the rotation axis is excentred
        self.d_C = parray.to_device(self.queue, C)

        self.add_to_cl_mem({
            "d_R": self.d_R,
            "d_C": self.d_C
        })

    # TODO: compute and possibly return the residual
    def run(self, data, n_it):
        """
        Run n_it iterations of the SIRT algorithm.
        """
        cl.enqueue_copy(self.queue, self.d_data.data, np.ascontiguousarray(data.astype(np.float32)))

        d_x_old = self.d_x_old
        d_x = self.d_x
        d_R = self.d_R
        d_C = self.d_C
        d_sino = self.d_sino
        d_x *= 0

        for k in range(n_it):
            d_x_old[:] = d_x[:]
            # x{k+1} = x{k} - C A^T R (A x{k} - b)
            self.proj(d_x, d_sino)
            d_sino -= self.d_data
            d_sino *= d_R
            if self.is_cpu:
                # This sync is necessary when using CPU, while it is not for GPU
                d_sino.finish()
            self.backproj(d_sino, d_x)
            d_x *= -d_C
            d_x += d_x_old
            if self.is_cpu:
                # This sync is necessary when using CPU, while it is not for GPU
                d_x.finish()

        return d_x

    __call__ = run


class TV(ReconstructionAlgorithm):
    """
    A class for reconstruction with Total Variation regularization using the
    Chambolle-Pock TV reconstruction algorithm.

    :param sino_shape: shape of the sinogram. The sinogram is in the format
                       (n_b, n_a) where n_b is the number of detector bins and
                       n_a is the number of angles.
    :param slice_shape: Optional, shape of the reconstructed slice. By default,
                        it is a square slice where the dimension is the
                        "x dimension" of the sinogram (number of bins).
    :param axis_position: Optional, axis position. Default is
                          `(shape[1]-1)/2.0`.
    :param angles: Optional, a list of custom angles in radian.
    :param ctx: actual working context, left to None for automatic
                initialization from device type or platformid/deviceid
    :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
    :param platformid: integer with the platform_identifier, as given by clinfo
    :param deviceid: Integer with the device identifier, as given by clinfo
    :param profile: switch on profiling to be able to profile at the kernel
                    level, store profiling elements (makes code slightly slower)

    .. warning:: This is a beta version of the Chambolle-Pock TV algorithm.
            Reconstruction fails for at least on CPU (Xeon E3-1245 v5) using
            the AMD opencl implementation.
    """

    def __init__(self, sino_shape, slice_shape=None, axis_position=None, angles=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):
        ReconstructionAlgorithm.__init__(self, sino_shape, slice_shape=slice_shape,
                                         axis_position=axis_position, angles=angles,
                                         ctx=ctx, devicetype=devicetype, platformid=platformid,
                                         deviceid=deviceid, profile=profile)
        self.compute_preconditioners()

        # Create a LinAlg instance
        self.linalg = LinAlg(self.backprojector.slice_shape, ctx=self.ctx)
        # Positivity constraint
        self.elwise_clamp = ElementwiseKernel(self.ctx, "float *a", "a[i] = max(a[i], 0.0f);")
        # Projection onto the L-infinity ball of radius Lambda
        self.elwise_proj_linf = ElementwiseKernel(
            self.ctx,
            "float2* a, float Lambda",
            "a[i].x = copysign(min(fabs(a[i].x), Lambda), a[i].x); a[i].y = copysign(min(fabs(a[i].y), Lambda), a[i].y);",
            "elwise_proj_linf"
        )
        # Additional arrays
        self.linalg.gradient(self.d_x)
        self.d_p = parray.empty_like(self.linalg.cl_mem["d_gradient"])
        self.d_q = parray.empty_like(self.d_data)
        self.d_g = self.linalg.d_image
        self.d_tmp = parray.empty_like(self.d_x)
        self.d_p.fill(0)
        self.d_q.fill(0)
        self.d_tmp.fill(0)
        self.add_to_cl_mem({
            "d_p": self.d_p,
            "d_q": self.d_q,
            "d_tmp": self.d_tmp,
        })

        self.theta = 1.0

    def compute_preconditioners(self):
        """
        Create a diagonal preconditioner for the projection and backprojection
        operator.
        Each term of the diagonal is the sum of the projector/backprojector
        along rows [2],
        i.e the projection/backprojection of an array of ones.

        [2] T. Pock, A. Chambolle,
            Diagonal preconditioning for first order primal-dual algorithms in
            convex optimization,
            International Conference on Computer Vision, 2011
        """

        # Compute the diagonal preconditioner "Sigma"
        slice_ones = np.ones(self.backprojector.slice_shape, dtype=np.float32)
        Sigma_k = 1./self.projector.projection(slice_ones)
        Sigma_k[np.logical_not(np.isfinite(Sigma_k))] = 1.
        self.d_Sigma_k = parray.to_device(self.queue, Sigma_k)
        self.d_Sigma_kp1 = self.d_Sigma_k + 1  # TODO: memory vs computation
        self.Sigma_grad = 1/2.0  # For discrete gradient, sum|D_i,j| = 2 along lines or cols

        # Compute the diagonal preconditioner "Tau"
        sino_ones = np.ones(self.sino_shape, dtype=np.float32)
        C = self.backprojector.backprojection(sino_ones)
        Tau = 1./(C + 2.)
        self.d_Tau = parray.to_device(self.queue, Tau)

        self.add_to_cl_mem({
            "d_Sigma_k": self.d_Sigma_k,
            "d_Sigma_kp1": self.d_Sigma_kp1,
            "d_Tau": self.d_Tau
        })

    def run(self, data, n_it, Lambda, pos_constraint=False):
        """
        Run n_it iterations of the TV-regularized reconstruction,
        with the regularization parameter Lambda.
        """
        cl.enqueue_copy(self.queue, self.d_data.data, np.ascontiguousarray(data.astype(np.float32)))

        d_x = self.d_x
        d_x_old = self.d_x_old
        d_tmp = self.d_tmp
        d_sino = self.d_sino
        d_p = self.d_p
        d_q = self.d_q
        d_g = self.d_g

        d_x *= 0
        d_p *= 0
        d_q *= 0

        for k in range(0, n_it):
            # Update primal variables
            d_x_old[:] = d_x[:]
            #~ x = x + Tau*div(p) - Tau*Kadj(q)
            self.backproj(d_q, d_tmp)
            self.linalg.divergence(d_p)
            # TODO: this in less than three ops (one kernel ?)
            d_g -= d_tmp  # d_g -> L.d_image
            d_g *= self.d_Tau
            d_x += d_g

            if pos_constraint:
                self.elwise_clamp(d_x)

            # Update dual variables
            #~ p = proj_linf(p + Sigma_grad*gradient(x + theta*(x - x_old)), Lambda)
            d_tmp[:] = d_x[:]
            # FIXME: mul_add is out of place, put an equivalent thing in linalg...
            #~ d_tmp.mul_add(1 + theta, d_x_old, -theta)
            d_tmp *= 1+self.theta
            d_tmp -= self.theta*d_x_old
            self.linalg.gradient(d_tmp)
            # TODO: out of place mul_add
            #~ d_p.mul_add(1, L.cl_mem["d_gradient"], Sigma_grad)
            self.linalg.cl_mem["d_gradient"] *= self.Sigma_grad
            d_p += self.linalg.cl_mem["d_gradient"]
            self.elwise_proj_linf(d_p, Lambda)

            #~ q = (q + Sigma_k*K(x + theta*(x - x_old)) - Sigma_k*data)/(1.0 + Sigma_k)
            self.proj(d_tmp, d_sino)
            # TODO: this in less instructions
            d_sino -= self.d_data
            d_sino *= self.d_Sigma_k
            d_q += d_sino
            d_q /= self.d_Sigma_kp1
        return d_x

    __call__ = run
