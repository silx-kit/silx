#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""
Simple usage examples of the tomographic reconstruction algorithms
"""
__authors__ = ["Pierre Paleo"]
__license__ = "MIT"
__date__ = "05/10/2017"

from silx.opencl.projection import Projection
from silx.opencl.backprojection import Backprojection
from silx.opencl.reconstruction import SIRT, TV
from silx.image.phantomgenerator import PhantomGenerator
import numpy as np

from six.moves import input

from silx.gui import qt
from silx.gui.plot import Plot2D

def main():
    print("In this example, we show the silx tomography utilities.")
    # ------------------------------------------------------------
    # How to project (Radon transform) an image to get a sinogram
    # ------------------------------------------------------------
    print("1. Projection")
    print("We first generate a synthetic sinogram by projecting an image, i.e computing its Radon Transform")
    phantom = PhantomGenerator.get2DPhantomSheppLogan(256)
    show_image(phantom, "phantom", "MRI brain phantom")

    # Number of projection angles. You can also provide an array of custom angles
    n_angles = 600
    # Create the projection geometry. See documentation for more available parameters
    P = Projection(phantom.shape, n_angles)
    # Now project the image to get a sinogram
    sino = P(phantom)
    show_image(sino, "sino", "Sinogram")

    # -----------------------------------------------------------------
    # How to reconstruct a sinogram with Filtered Backprojection (FBP)
    # -----------------------------------------------------------------
    print("2. Filtered Backprojection (FBP)")
    print("We show how to reconstruct a sinogram with FBP")
    # Create a Backprojection geometry. See documentation for more available parameters
    B = Backprojection(sino.shape)
    # Now reconstruct the current sinogram
    rec_fbp = B.filtered_backprojection(sino) # alternatively: B(sino)
    show_image(rec_fbp, "FBP", str("Reconstruction with FBP using %d projections" % P.nprojs))

    # --------------------------------------------------
    # How to reconstruct a sinogram with SIRT algorithm
    # --------------------------------------------------
    print("3. SIRT reconstruction")
    print("In practice, sinogram data is often noisy, and sometimes undersampled.")
    print("Iterative methods aim at addressing these issues.")
    # Assume that the data is subsampled
    sino_subsampled = np.ascontiguousarray(sino[::15])
    # Add noise to the sinogram
    sino_subsampled += np.random.randn(*sino_subsampled.shape) * sino.max()*1.0/100
    S = SIRT(sino_subsampled.shape)
    # Reconstruct the current singoram
    n_it = 150
    rec_sirt = S(sino_subsampled, n_it).get()
    show_image(S.backprojector(sino_subsampled), "SIRT", str("Reconstruction with FBP using %d projections" % S.projector.nprojs))
    show_image(rec_sirt, "SIRT", str("Reconstruction with SIRT using %d iterations" % n_it))

    # -----------------------------------------------------
    # How to reconstruct a sinogram with TV regularization
    # -----------------------------------------------------
    print("4. TV reconstruction")
    print("We show how to reconstruct a sinogram with Total Variation (TV) regularization.")
    print("TV regularization deals with the undersampling by promoting piecewise-constant images")
    T = TV(sino_subsampled.shape)
    # Reconstruct the current sinogram
    n_it = 400
    rec_tv = T(sino_subsampled, n_it, 8e2, pos_constraint=True).get()
    show_image(rec_tv, "TV", str("Reconstruction with TV using %d iterations" % n_it))

    plot = Plot2D()
    plot.setAttribute(qt.Qt.WA_DeleteOnClose)
    plot.getYAxis().setInverted(True)
    plot.setKeepDataAspectRatio(True)

    plot.addImage(S.backprojector(sino_subsampled), legend="FBP")
    plot.addImage(rec_sirt, legend="SIRT", replace=False, origin=(512, 0))
    plot.addImage(rec_tv, legend="TV", replace=False, origin=(1024, 0))
    plot.setGraphTitle("Comparison of reconstruction methods for undersampled and noisy data")
    plot.show()
    qt.qApp.processEvents()
    input("Press Enter to continue")


def show_image(image, legend, title):
    plot = Plot2D()
    plot.setAttribute(qt.Qt.WA_DeleteOnClose)
    plot.getYAxis().setInverted(True)
    plot.setKeepDataAspectRatio(True)

    plot.addImage(image, legend=legend)
    plot.setGraphTitle(title)
    plot.show()
    qt.qApp.processEvents()
    input("Press Enter to continue")


if __name__ == "__main__":
    qapp = qt.QApplication([])
    main()




