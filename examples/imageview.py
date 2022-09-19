#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
Example to show the use of :class:`~silx.gui.plot.ImageView.ImageView` widget.

It can be used to open an EDF or TIFF file from the shell command line.

To view an image file with the current installed silx library:
``python examples/imageview.py <file to open>``
To get help:
``python examples/imageview.py -h``
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/11/2018"

import logging
import numpy
import time
import threading

from silx.gui.utils import concurrent
from silx.gui.plot.ImageView import ImageViewMainWindow
from silx.gui import qt

logging.basicConfig()
logger = logging.getLogger(__name__)


Nx = 150
Ny = 50


class UpdateThread(threading.Thread):
    """Thread updating the image of a :class:`~sil.gui.plot.Plot2D`

    :param plot2d: The Plot2D to update."""

    def __init__(self, imageview):
        self.imageview = imageview
        self.running = False
        self.future_result = None
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self, pos={'x0': 0, 'y0': 0}):
        """Method implementing thread loop that updates the plot

        It produces an image every 10 ms or so, and
        either updates the plot or skip the image
        """
        while self.running:
            time.sleep(0.01)

            # Create image
            # width of peak
            sigma_x = 0.15
            sigma_y = 0.25
            # x and y positions
            x = numpy.linspace(-1.5, 1.5, Nx)
            y = numpy.linspace(-1.0, 1.0, Ny)
            xv, yv = numpy.meshgrid(x, y)
            signal = numpy.exp(- ((xv - pos['x0']) ** 2 / sigma_x ** 2
                                  + (yv - pos['y0']) ** 2 / sigma_y ** 2))
            # add noise
            signal += 0.3 * numpy.random.random(size=signal.shape)
            # random walk of center of peak ('drift')
            pos['x0'] += 0.05 * (numpy.random.random() - 0.5)
            pos['y0'] += 0.05 * (numpy.random.random() - 0.5)

            # If previous frame was not added to the plot yet, skip this one
            if self.future_result is None or self.future_result.done():
                # plot the data asynchronously, and
                # keep a reference to the `future` object
                self.future_result = concurrent.submitToQtMainThread(
                    self.imageview.setImage, signal, resetzoom=False)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main(argv=None):
    """Display an image from a file in an :class:`ImageView` widget.

    :param argv: list of command line arguments or None (the default)
                 to use sys.argv.
    :type argv: list of str
    :return: Exit status code
    :rtype: int
    :raises IOError: if no image can be loaded from the file
    """
    import argparse
    import os.path

    from silx.third_party.EdfFile import EdfFile

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description='Browse the images of an EDF file.')
    parser.add_argument(
        '-o', '--origin', nargs=2,
        type=float, default=(0., 0.),
        help="""Coordinates of the origin of the image: (x, y).
        Default: 0., 0.""")
    parser.add_argument(
        '-s', '--scale', nargs=2,
        type=float, default=(1., 1.),
        help="""Scale factors applied to the image: (sx, sy).
        Default: 1., 1.""")
    parser.add_argument(
        '-l', '--log', action="store_true",
        help="Use logarithm normalization for colormap, default: Linear.")
    parser.add_argument(
        'filename', nargs='?',
        help='EDF filename of the image to open')
    parser.add_argument(
        '--live', action='store_true',
        help='Live update of a generated image')
    args = parser.parse_args(args=argv)

    # Open the input file
    edfFile = None
    if args.live:
        data = None
    elif not args.filename:
        logger.warning('No image file provided, displaying dummy data')
        size = 512
        xx, yy = numpy.ogrid[-size:size, -size:size]
        data = numpy.cos(xx / (size//5)) + numpy.cos(yy / (size//5))
        data = numpy.random.poisson(numpy.abs(data))
        nbFrames = 1

    else:
        if not os.path.isfile(args.filename):
            raise IOError('No input file: %s' % args.filename)

        else:
            edfFile = EdfFile(args.filename)
            data = edfFile.GetData(0)

            nbFrames = edfFile.GetNumImages()
            if nbFrames == 0:
                raise IOError(
                    'Cannot read image(s) from file: %s' % args.filename)

    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler

    mainWindow = ImageViewMainWindow()
    mainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)

    if args.log:  # Use log normalization by default
        colormap = mainWindow.getDefaultColormap()
        colormap.setNormalization(colormap.LOGARITHM)

    if data is not None:
        mainWindow.setImage(data,
                            origin=args.origin,
                            scale=args.scale)

    if edfFile is not None and nbFrames > 1:
        # Add a toolbar for multi-frame EDF support
        multiFrameToolbar = qt.QToolBar('Multi-frame')
        multiFrameToolbar.addWidget(qt.QLabel(
            'Frame [0-%d]:' % (nbFrames - 1)))

        spinBox = qt.QSpinBox()
        spinBox.setRange(0, nbFrames - 1)

        def updateImage(index):
            mainWindow.setImage(edfFile.GetData(index),
                                origin=args.origin,
                                scale=args.scale,
                                reset=False)
        spinBox.valueChanged[int].connect(updateImage)
        multiFrameToolbar.addWidget(spinBox)

        mainWindow.addToolBar(multiFrameToolbar)

    mainWindow.show()
    mainWindow.setFocus(qt.Qt.OtherFocusReason)

    if args.live:
        # Start updating the plot
        updateThread = UpdateThread(mainWindow)
        updateThread.start()

    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main(argv=sys.argv[1:]))
