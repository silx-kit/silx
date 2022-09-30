# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""This script illustrates the update of a
:class:`~silx.gui.plot3d.SceneWindow.SceneWindow` widget from a thread.

The problem is that GUI methods should be called from the main thread.
To safely update the scene from another thread, one need to execute the update
asynchronously in the main thread.
In this example, this is achieved with
:func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

In this example a thread calls submitToQtMainThread to append data to a 3D scatter.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/03/2019"


import threading
import time

import numpy

from silx.gui import qt
from silx.gui.utils import concurrent
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui.plot3d import items


MAX_NUMBER_OF_POINTS = 10**6


class UpdateScatterThread(threading.Thread):
    """Thread updating the scatter 3D item data

    :param ~silx.gui.plot3d.items.Scatter3D scatter3d: 3D scatter to update.
    """

    def __init__(self, scatter3d):
        self.scatter3d = scatter3d
        self.running = False
        self.future_result = None
        super(UpdateScatterThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateScatterThread, self).start()

    def _appendScatterData(self, x, y, z, value):
        """Add some data points to the Scatter3D item.

        This method MUST be called in the Qt main thread.

        :param numpy.ndarray x:
        :param numpy.ndarray y:
        :param numpy.ndarray z:
        :param numpy.ndarray value:
        """
        # use copy=False to avoid useless copy of numpy arrays
        curX, curY, curZ, curValue = self.scatter3d.getData(copy=False)

        x = numpy.append(curX, x)
        y = numpy.append(curY, y)
        z = numpy.append(curZ, z)
        value = numpy.append(curValue, value)

        # Update data
        self.scatter3d.setData(x, y, z, value, copy=False)

    def run(self):
        """Method implementing thread loop that updates the scatter data

        It produces adds scatter points every 10 ms or so, up to 1 million.
        """
        count = 0  # Number of data points currently rendered

        # Init arrays that accumulate scatter points
        x = numpy.array((), dtype=numpy.float32)
        y = numpy.array((), dtype=numpy.float32)
        z = numpy.array((), dtype=numpy.float32)
        value = numpy.array((), dtype=numpy.float32)

        while self.running:
            time.sleep(0.01)

            # Generate new data points
            inclination = numpy.random.random(1000).astype(numpy.float32) * numpy.pi
            azimuth = numpy.random.random(1000).astype(numpy.float32) * 2. * numpy.pi
            radius = numpy.random.normal(loc=10., scale=.5, size=1000)
            newX = radius * numpy.sin(inclination) * numpy.cos(azimuth)
            newY = radius * numpy.sin(inclination) * numpy.sin(azimuth)
            newZ = radius * numpy.cos(inclination)
            newValue = numpy.random.random(1000).astype(numpy.float32)

            # Accumulate data points
            x = numpy.append(x, newX)
            y = numpy.append(y, newY)
            z = numpy.append(z, newZ)
            value = numpy.append(value, newValue)

            # Only append data if the previous one has been added
            if self.future_result is None or self.future_result.done():
                if count > MAX_NUMBER_OF_POINTS:
                    # Restart a new scatter plot asyn
                    self.future_result = concurrent.submitToQtMainThread(
                        self.scatter3d.setData, x, y, z, value)

                    count = len(x)
                else:
                    # Append data asynchronously
                    self.future_result = concurrent.submitToQtMainThread(
                        self._appendScatterData, x, y, z, value)

                    count += len(x)

                # Reset accumulators
                x = numpy.array((), dtype=numpy.float32)
                y = numpy.array((), dtype=numpy.float32)
                z = numpy.array((), dtype=numpy.float32)
                value = numpy.array((), dtype=numpy.float32)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a SceneWindow
    window = SceneWindow()
    window.show()

    sceneWidget = window.getSceneWidget()
    scatter = items.Scatter3D()
    scatter.setSymbol(',')
    scatter.getColormap().setName('magma')
    sceneWidget.addItem(scatter)

    # Create the thread that calls submitToQtMainThread
    updateThread = UpdateScatterThread(scatter)
    updateThread.start()  # Start updating the plot

    app.exec()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
