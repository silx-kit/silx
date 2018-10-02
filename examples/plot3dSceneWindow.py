# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
This script displays the different items of :class:`~silx.gui.plot3d.SceneWindow`.

It shows the different visualizations of :class:`~silx.gui.plot3d.SceneWindow`
and :class:`~silx.gui.plot3d.SceneWidget`.
It illustrates the API to set those items.

It features:

- 2D images: data and RGBA images
- 2D scatter data, displayed either as markers, wireframe or surface.
- 3D scatter plot
- 3D scalar field with iso-surface and cutting plane.
- A clipping plane.

"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/11/2017"


import sys
import numpy

from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items
from silx.gui.plot3d.tools.PositionInfoWidget import PositionInfoWidget
from silx.gui.widgets.BoxLayoutDockWidget import BoxLayoutDockWidget

SIZE = 1024

# Create QApplication
qapp = qt.QApplication([])

# Create a SceneWindow widget
window = SceneWindow()

# Get the SceneWidget contained in the window and set its colors
sceneWidget = window.getSceneWidget()
sceneWidget.setBackgroundColor((0.8, 0.8, 0.8, 1.))
sceneWidget.setForegroundColor((1., 1., 1., 1.))
sceneWidget.setTextColor((0.1, 0.1, 0.1, 1.))


# Add PositionInfoWidget to display picking info
positionInfo = PositionInfoWidget()
positionInfo.setSceneWidget(sceneWidget)
dock = BoxLayoutDockWidget()
dock.setWindowTitle("Selection Info")
dock.setWidget(positionInfo)
window.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)

# 2D Image ###

# Add a dummy RGBA image
img = numpy.random.random(3 * SIZE ** 2).reshape(SIZE, SIZE, 3)  # Dummy image

imageRgba = sceneWidget.addImage(img)  # Add ImageRgba item to the scene

# Set imageRgba transform
imageRgba.setTranslation(SIZE*.15, SIZE*.15, 0.)  # Translate the image
# Rotate the image by 45 degrees around its center
imageRgba.setRotationCenter('center', 'center', 0.)
imageRgba.setRotation(45., axis=(0., 0., 1.))
imageRgba.setScale(0.7, 0.7, 0.7)  # Scale down image


# Add a data image
data = numpy.arange(SIZE ** 2).reshape(SIZE, SIZE)  # Dummy data
imageData = sceneWidget.addImage(data)  # Add ImageData item to the scene

# Set imageData transform
imageData.setTranslation(0., SIZE, 0.)  # Translate the image

# Set imageData properties
imageData.setInterpolation('linear')  # 'linear' or 'nearest' interpolation
imageData.getColormap().setName('magma')  # Use magma colormap


# 2D scatter data ###

# Create 2D scatter dummy data
x = numpy.random.random(10 ** 3)
y = numpy.random.random(len(x))
values = numpy.exp(- 11. * ((x - .5) ** 2 + (y - .5) ** 2))

# Add 2D scatter data with 6 different visualisations
for row, heightMap in enumerate((False, True)):
    for col, mode in enumerate(('points', 'lines', 'solid')):
        # Add a new scatter
        item = sceneWidget.add2DScatter(x, y, values)

        # Set 2D scatter item tranform
        item.setTranslation(SIZE + col * SIZE, row * SIZE, 0.)
        item.setScale(SIZE, SIZE, SIZE)

        # Set 2D scatter item properties
        item.setHeightMap(heightMap)
        item.setVisualization(mode)
        item.getColormap().setName('viridis')
        item.setLineWidth(2.)


# Group  ###

# Create a group item and add it to the scene
# The group children share the group transform
group = items.GroupItem()  # Create a new group item
group.setTranslation(SIZE * 4, 0., 0.)  # Translate the group


# Clipping plane ###

# Add a clipping plane to the group (and thus to the scene)
# This item hides part of other items in the half space defined by the plane.
# Clipped items are those belonging to the same group (i.e., brothers) that
# comes after the clipping plane.
clipPlane = items.ClipPlane()  # Create a new clipping plane item
clipPlane.setNormal((1., -0.35, 0.))  # Set its normal
clipPlane.setPoint((0., 0., 0.))  # Set a point on the plane
group.addItem(clipPlane)  # Add clipping plane to the group


# 3D scatter data ###

# Create dummy data
x = numpy.random.random(10**3)
y = numpy.random.random(len(x))
z = numpy.random.random(len(x))
values = numpy.random.random(len(x))

# Create a 3D scatter item and set its data
scatter3d = items.Scatter3D()
scatter3d.setData(x, y, z, values)

# Set scatter3d transform
scatter3d.setScale(SIZE, SIZE, SIZE)

# Set scatter3d properties
scatter3d.getColormap().setName('magma')  # Use 'magma' colormap
scatter3d.setSymbol('d')  # Use diamond markers
scatter3d.setSymbolSize(11)  # Set the size of the markers

# Add scatter3d to the group (and thus to the scene)
group.addItem(scatter3d)


# 3D scalar volume ###

# Create dummy 3D array data
x, y, z = numpy.meshgrid(numpy.linspace(-10, 10, 64),
                         numpy.linspace(-10, 10, 64),
                         numpy.linspace(-10, 10, 64))
data = numpy.sin(x * y * z) / (x * y * z)

# Create a 3D scalar field item and set its data
volume = items.ScalarField3D()  # Create a new 3D volume item
volume.setData(data)  # Set its data
group.addItem(volume)  # Add it to the group (and thus to the scene)

# Set volume tranform
volume.setTranslation(0., SIZE, 0.)
volume.setScale(SIZE/data.shape[2], SIZE/data.shape[1], SIZE/data.shape[0])

# Add isosurfaces to the volume item given isolevel and color
volume.addIsosurface(0.2, '#FF000080')
volume.addIsosurface(0.5, '#0000FFFF')

# Set the volume cut plane
cutPlane = volume.getCutPlanes()[0]  # Get the volume's cut plane
cutPlane.setVisible(True)  # Set it to be visible
cutPlane.getColormap().setName('jet')  # Set cut plane's colormap
cutPlane.setNormal((0., 0., 1.))  # Set cut plane's normal
cutPlane.moveToCenter()  # Place the cut plane at the center of the volume

sceneWidget.addItem(group)  # Add the group as an item of the scene

# Show the SceneWidget widget
window.show()

# Display exception in a pop-up message box
sys.excepthook = qt.exceptionHandler

# Run Qt event loop
qapp.exec_()
