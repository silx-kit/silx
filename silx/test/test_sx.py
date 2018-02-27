# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "27/02/2018"


import logging
import os
import sys
import unittest
import numpy

from silx.test.utils import test_options


_logger = logging.getLogger(__name__)


if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
    # On linux and no DISPLAY available (e.g., ssh without -X)
    _logger.warning('silx.sx tests disabled (DISPLAY env. variable not set)')

    class SkipSXTest(unittest.TestCase):
        def runTest(self):
            self.skipTest(
                'silx.sx tests disabled (DISPLAY env. variable not set)')

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(SkipSXTest())
        return suite

elif not test_options.WITH_QT_TEST:
    # Explicitly disabled tests
    msg = "silx.sx tests disabled %s" % test_options.WITH_QT_TEST_REASON
    _logger.warning(msg)

    class SkipSXTest(unittest.TestCase):
        def runTest(self):
            self.skipTest(test_options.WITH_QT_TEST_REASON)

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(SkipSXTest())
        return suite

else:
    # Import here to avoid loading QT if tests are disabled

    from silx.gui import qt
    # load TestCaseQt before sx
    from silx.gui.test.utils import TestCaseQt
    from silx.gui.plot.Colors import rgba
    from silx import sx

    class SXTest(TestCaseQt):
        """Test the sx module"""

        def _expose_and_close(self, plot):
            self.qWaitForWindowExposed(plot)
            self.qapp.processEvents()
            plot.setAttribute(qt.Qt.WA_DeleteOnClose)
            plot.close()

        def test_plot(self):
            """Test plot function"""
            y = numpy.random.random(100)
            x = numpy.arange(len(y)) * 0.5

            # Nothing
            plt = sx.plot()
            self._expose_and_close(plt)

            # y
            plt = sx.plot(y, title='y')
            self._expose_and_close(plt)

            # y, style
            plt = sx.plot(y, 'blued ', title='y, "blued "')
            self._expose_and_close(plt)

            # x, y
            plt = sx.plot(x, y, title='x, y')
            self._expose_and_close(plt)

            # x, y, style
            plt = sx.plot(x, y, 'ro-', xlabel='x', title='x, y, "ro-"')
            self._expose_and_close(plt)

            # x, y, style, y
            plt = sx.plot(x, y, 'ro-', y ** 2, xlabel='x', ylabel='y',
                          title='x, y, "ro-", y ** 2')
            self._expose_and_close(plt)

            # x, y, style, y, style
            plt = sx.plot(x, y, 'ro-', y ** 2, 'b--',
                          title='x, y, "ro-", y ** 2, "b--"')
            self._expose_and_close(plt)

            # x, y, style, x, y, style
            plt = sx.plot(x, y, 'ro-', x, y ** 2, 'b--',
                          title='x, y, "ro-", x, y ** 2, "b--"')
            self._expose_and_close(plt)

            # x, y, x, y
            plt = sx.plot(x, y, x, y ** 2, title='x, y, x, y ** 2')
            self._expose_and_close(plt)

        def test_imshow(self):
            """Test imshow function"""
            img = numpy.arange(100.).reshape(10, 10) + 1

            # Nothing
            plt = sx.imshow()
            self._expose_and_close(plt)

            # image
            plt = sx.imshow(img)
            self._expose_and_close(plt)

            # image, gray cmap
            plt = sx.imshow(img, cmap='jet', title='jet cmap')
            self._expose_and_close(plt)

            # image, log cmap
            plt = sx.imshow(img, norm='log', title='log cmap')
            self._expose_and_close(plt)

            # image, fixed range
            plt = sx.imshow(img, vmin=10, vmax=20,
                            title='[10,20] cmap')
            self._expose_and_close(plt)

            # image, keep ratio
            plt = sx.imshow(img, aspect=True,
                            title='keep ratio')
            self._expose_and_close(plt)

            # image, change origin and scale
            plt = sx.imshow(img, origin=(10, 10), scale=(2, 2),
                            title='origin=(10, 10), scale=(2, 2)')
            self._expose_and_close(plt)

        def test_ginput(self):
            """Test ginput function

            This does NOT perform interactive tests
            """

            plt = sx.plot()
            self.qWaitForWindowExposed(plt)
            self.qapp.processEvents()

            result = sx.ginput(1, timeout=0.1)
            self.assertEqual(len(result), 0)

            plt.setAttribute(qt.Qt.WA_DeleteOnClose)
            plt.close()

        @unittest.skipUnless(test_options.WITH_GL_TEST,
                             test_options.WITH_GL_TEST_REASON)
        def test_contour3d(self):
            """Test contour3d function"""
            coords = numpy.linspace(-10, 10, 64)
            z = coords.reshape(-1, 1, 1)
            y = coords.reshape(1, -1, 1)
            x = coords.reshape(1, 1, -1)
            data = numpy.sin(x * y * z) / (x * y * z)

            # Just data
            window = sx.contour3d(data)

            isosurfaces = window.getIsosurfaces()
            self.assertEqual(len(isosurfaces), 1)

            self._expose_and_close(window)
            if not window.getPlot3DWidget().isValid():
                self.skipTest("OpenGL context is not valid")

            # N contours + color
            colors = ['red', 'green', 'blue']
            window = sx.contour3d(data, copy=False, contours=len(colors),
                                  color=colors)

            isosurfaces = window.getIsosurfaces()
            self.assertEqual(len(isosurfaces), len(colors))
            for iso, color in zip(isosurfaces, colors):
                self.assertEqual(rgba(iso.getColor()), rgba(color))

            self._expose_and_close(window)

            # by isolevel, single color
            contours = 0.2, 0.5
            window = sx.contour3d(data, copy=False, contours=contours,
                                  color='yellow')

            isosurfaces = window.getIsosurfaces()
            self.assertEqual(len(isosurfaces), len(contours))
            for iso, level in zip(isosurfaces, contours):
                self.assertEqual(iso.getLevel(), level)
                self.assertEqual(rgba(iso.getColor()),
                                 rgba('yellow'))

            self._expose_and_close(window)

            # Single isolevel, colormap
            window = sx.contour3d(data, copy=False, contours=0.5,
                                  colormap='gray', vmin=0.6, opacity=0.4)

            isosurfaces = window.getIsosurfaces()
            self.assertEqual(len(isosurfaces), 1)
            self.assertEqual(isosurfaces[0].getLevel(), 0.5)
            self.assertEqual(rgba(isosurfaces[0].getColor()),
                             (0., 0., 0., 0.4))

            self._expose_and_close(window)

        @unittest.skipUnless(test_options.WITH_GL_TEST,
                             test_options.WITH_GL_TEST_REASON)
        def test_points3d(self):
            """Test points3d function"""
            x = numpy.random.random(1024)
            y = numpy.random.random(1024)
            z = numpy.random.random(1024)
            values = numpy.random.random(1024)

            # 3D positions, no value
            window = sx.points3d(x, y, z)

            self._expose_and_close(window)
            if not window.getSceneWidget().isValid():
                self.skipTest("OpenGL context is not valid")

            # 3D positions, values
            window = sx.points3d(x, y, z, values, mode='2dsquare',
                                 colormap='magma', vmin=0.4, vmax=0.5)
            self._expose_and_close(window)

            # 2D positions, no value
            window = sx.points3d(x, y)
            self._expose_and_close(window)

            # 2D positions, values
            window = sx.points3d(x, y, values=values, mode=',',
                                 colormap='magma', vmin=0.4, vmax=0.5)
            self._expose_and_close(window)

    def suite():
        test_suite = unittest.TestSuite()
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(SXTest))
        return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
