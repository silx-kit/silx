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
__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "06/11/2018"


import numpy
import pytest

from silx.gui import qt
from silx.gui.colors import rgba
from silx.gui.colors import Colormap


@pytest.fixture(scope="module")
def sx(qapp):
    """Lazy loading to avoid it to create QApplication before qapp fixture"""
    from silx import sx
    if sx._IS_NOTEBOOK:
        pytest.skip("notebook context")
    if sx._NO_DISPLAY:
        pytest.skip("no DISPLAY specified")
    yield sx


def test_plot(sx, qapp_utils):
    """Test plot function"""
    y = numpy.random.random(100)
    x = numpy.arange(len(y)) * 0.5

    # Nothing
    plt = sx.plot()
    qapp_utils.exposeAndClose(plt)

    # y
    plt = sx.plot(y, title='y')
    qapp_utils.exposeAndClose(plt)

    # y, style
    plt = sx.plot(y, 'blued ', title='y, "blued "')
    qapp_utils.exposeAndClose(plt)

    # x, y
    plt = sx.plot(x, y, title='x, y')
    qapp_utils.exposeAndClose(plt)

    # x, y, style
    plt = sx.plot(x, y, 'ro-', xlabel='x', title='x, y, "ro-"')
    qapp_utils.exposeAndClose(plt)

    # x, y, style, y
    plt = sx.plot(x, y, 'ro-', y ** 2, xlabel='x', ylabel='y',
                  title='x, y, "ro-", y ** 2')
    qapp_utils.exposeAndClose(plt)

    # x, y, style, y, style
    plt = sx.plot(x, y, 'ro-', y ** 2, 'b--',
                  title='x, y, "ro-", y ** 2, "b--"')
    qapp_utils.exposeAndClose(plt)

    # x, y, style, x, y, style
    plt = sx.plot(x, y, 'ro-', x, y ** 2, 'b--',
                  title='x, y, "ro-", x, y ** 2, "b--"')
    qapp_utils.exposeAndClose(plt)

    # x, y, x, y
    plt = sx.plot(x, y, x, y ** 2, title='x, y, x, y ** 2')
    qapp_utils.exposeAndClose(plt)


def test_imshow(sx, qapp_utils):
    """Test imshow function"""
    img = numpy.arange(100.).reshape(10, 10) + 1

    # Nothing
    plt = sx.imshow()
    qapp_utils.exposeAndClose(plt)

    # image
    plt = sx.imshow(img)
    qapp_utils.exposeAndClose(plt)

    # image, named cmap
    plt = sx.imshow(img, cmap='jet', title='jet cmap')
    qapp_utils.exposeAndClose(plt)

    # image, custom colormap
    plt = sx.imshow(img, cmap=Colormap(), title='custom colormap')
    qapp_utils.exposeAndClose(plt)

    # image, log cmap
    plt = sx.imshow(img, norm='log', title='log cmap')
    qapp_utils.exposeAndClose(plt)

    # image, fixed range
    plt = sx.imshow(img, vmin=10, vmax=20,
                    title='[10,20] cmap')
    qapp_utils.exposeAndClose(plt)

    # image, keep ratio
    plt = sx.imshow(img, aspect=True,
                    title='keep ratio')
    qapp_utils.exposeAndClose(plt)

    # image, change origin and scale
    plt = sx.imshow(img, origin=(10, 10), scale=(2, 2),
                    title='origin=(10, 10), scale=(2, 2)')
    qapp_utils.exposeAndClose(plt)

    # image, origin='lower'
    plt = sx.imshow(img, origin='upper', title='origin="lower"')
    qapp_utils.exposeAndClose(plt)


def test_scatter(sx, qapp_utils):
    """Test scatter function"""
    x = numpy.arange(100)
    y = numpy.arange(100)
    values = numpy.arange(100)

    # simple scatter
    plt = sx.scatter(x, y, values)
    qapp_utils.exposeAndClose(plt)

    # No value
    plt = sx.scatter(x, y, values)
    qapp_utils.exposeAndClose(plt)

    # single value
    plt = sx.scatter(x, y, 10.)
    qapp_utils.exposeAndClose(plt)

    # set size
    plt = sx.scatter(x, y, values, size=20)
    qapp_utils.exposeAndClose(plt)

    # set colormap
    plt = sx.scatter(x, y, values, cmap='jet')
    qapp_utils.exposeAndClose(plt)

    # set colormap range
    plt = sx.scatter(x, y, values, vmin=2, vmax=50)
    qapp_utils.exposeAndClose(plt)

    # set colormap normalisation
    plt = sx.scatter(x, y, values, norm='log')
    qapp_utils.exposeAndClose(plt)


@pytest.mark.parametrize("plot_kind", ["plot", "imshow", "scatter"])
def test_ginput(sx, qapp, qapp_utils, plot_kind):
    """Test ginput function

    This does NOT perform interactive tests
    """
    create_plot = getattr(sx, plot_kind)
    plt = create_plot()
    qapp_utils.qWaitForWindowExposed(plt)
    qapp.processEvents()

    result = sx.ginput(1, timeout=0.1)
    assert len(result) == 0

    plt.setAttribute(qt.Qt.WA_DeleteOnClose)
    plt.close()


@pytest.mark.usefixtures("use_opengl")
def test_contour3d(sx, qapp_utils):
    """Test contour3d function"""
    coords = numpy.linspace(-10, 10, 64)
    z = coords.reshape(-1, 1, 1)
    y = coords.reshape(1, -1, 1)
    x = coords.reshape(1, 1, -1)
    data = numpy.sin(x * y * z) / (x * y * z)

    # Just data
    window = sx.contour3d(data)

    isosurfaces = window.getIsosurfaces()
    assert len(isosurfaces) == 1

    if not window.getPlot3DWidget().isValid():
        del window, isosurfaces  # Release widget reference
        pytest.skip("OpenGL context is not valid")

    # N contours + color
    colors = ['red', 'green', 'blue']
    window = sx.contour3d(data, copy=False, contours=len(colors),
                          color=colors)

    isosurfaces = window.getIsosurfaces()
    assert len(isosurfaces) == len(colors)
    for iso, color in zip(isosurfaces, colors):
        assert rgba(iso.getColor()) == rgba(color)

    # by isolevel, single color
    contours = 0.2, 0.5
    window = sx.contour3d(data, copy=False, contours=contours,
                          color='yellow')

    isosurfaces = window.getIsosurfaces()
    assert len(isosurfaces) == len(contours)
    for iso, level in zip(isosurfaces, contours):
        assert iso.getLevel() == level
        assert rgba(iso.getColor()) == rgba('yellow')

    # Single isolevel, colormap
    window = sx.contour3d(data, copy=False, contours=0.5,
                          colormap='gray', vmin=0.6, opacity=0.4)

    isosurfaces = window.getIsosurfaces()
    assert len(isosurfaces) == 1
    assert isosurfaces[0].getLevel() == 0.5
    assert rgba(isosurfaces[0].getColor()) == (0., 0., 0., 0.4)


@pytest.mark.usefixtures("use_opengl")
def test_points3d(sx, qapp_utils):
    """Test points3d function"""
    x = numpy.random.random(1024)
    y = numpy.random.random(1024)
    z = numpy.random.random(1024)
    values = numpy.random.random(1024)

    # 3D positions, no value
    window = sx.points3d(x, y, z)

    if not window.getSceneWidget().isValid():
        del window  # Release widget reference
        pytest.skip("OpenGL context is not valid")

    # 3D positions, values
    window = sx.points3d(x, y, z, values, mode='2dsquare',
                         colormap='magma', vmin=0.4, vmax=0.5)

    # 2D positions, no value
    window = sx.points3d(x, y)

    # 2D positions, values
    window = sx.points3d(x, y, values=values, mode=',',
                         colormap='magma', vmin=0.4, vmax=0.5)
