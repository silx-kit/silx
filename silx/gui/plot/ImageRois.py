# coding: utf-8
# /*##########################################################################
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
# ###########################################################################*/
"""Functions to prepare events to be sent to Plot callback."""

__author__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/09/2016"


import numpy as np
from silx.gui import qt, icons


class RoiItemBase(qt.QObject):
    sigRoiFinished = qt.Signal(str)

    shape = None

    plot = property(lambda self: self._plot)
    name = property(lambda self: self._name)

    def __init__(self, plot, parent, name=None):
        super(RoiItemBase, self).__init__(parent=parent)

        self._plot = plot
        self._anchors = []
        self._points = {}
        self._kwargs = []

        self._finished = False
        self._connected = False
        self._editing = False

        if not name:
            uuid = str(id(self))
            name = '{0}_{1}'.format(self.__class__.__name__, uuid)

        self._name = name

    def _interactiveModeChanged(self, source):
        if source is not self or source is not self.parent():
            if not self._finished:
                self._cancel()
            else:
                self.edit(False)

    def _plotSignal(self, event):
        evType = event['event']
        if evType == 'drawingFinished':
            params = event['parameters']
            if params['label'] == self._name:
                self._finish(event)
        elif evType == 'markerMoving':
            label = event['label']
            try:
                idx = self._anchors.index(label)
            except ValueError:
                idx = None
            else:
                x = event['x']
                y = event['y']
                self.setAnchorData(label, (x, y))
                self.anchorMoved(label, x, y, idx)
                self._draw()

    def registerAnchor(self, anchor, point, idx=-1):
        if anchor in self._anchors:
            return
        if idx is not None and idx >= 0 and idx < len(self._anchors):
            self._anchors.insert(anchor, idx)
        else:
            self._anchors.append(anchor)
            idx = len(self._anchors)
        self._points[anchor] = point
        return idx

    def unregisterAnchor(self, label):
        try:
            self._anchors.remove(label)
        except ValueError:
            pass

    def _connect(self):
        if self._connected:
            return

        self._plot.sigPlotSignal.connect(self._plotSignal)
        self._plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)
        self._connected = True

    def _disconnect(self):
        if not self._connected:
            return

        self._plot.sigPlotSignal.disconnect(self._plotSignal)
        self._plot.sigInteractiveModeChanged.disconnect(
            self._interactiveModeChanged)
        self._connected = False

    def _draw(self, drawAnchors=True, excludes=()):

        if drawAnchors:
            if excludes is not None and len(excludes) > 0:
                draw_legends = set(self._anchors) - set(excludes)
            else:
                draw_legends = self._anchors

            for i_anchor, anchor in enumerate(draw_legends):
                item = self._plot.addMarker(self._points[anchor][0],
                                            self._points[anchor][1],
                                            legend=anchor,
                                            draggable=True,
                                            symbol='x')
                assert item == anchor

        item = self._plot.addItem(self.xData(),
                                  self.yData(),
                                  shape=self.shape,
                                  legend=self._name,
                                  overlay=True)
        assert item == self._name

    def setAnchorData(self, name, point):
        self._points[name] = point

    def start(self):
        self.edit(False)
        self._finished = False
        self._plot.setInteractiveMode('draw',
                                      shape=self.shape,
                                      source=self,
                                      label=self._name)
        self._connect()
        self.drawStarted()

    def edit(self, enable):
        if not self._finished:
            return

        if self._editing == enable:
            return

        if enable:
            self._connect()
            self.editStarted()
            self._draw()
        else:
            self._disconnect()
            {self._plot.removeMarker(item) for item in self._anchors}
            self.editStopped()
            self._draw(drawAnchors=False)

        self._editing = enable

    def _finish(self, event):
        self.drawFinished(event)
        self._draw(drawAnchors=False)
        self._finished = True
        self.sigRoiFinished.emit(self.name)
        self._disconnect()

    def _cancel(self):
        self._disconnect()
        self.drawCanceled()

    def drawStarted(self):
        pass

    def drawFinished(self, event):
        pass

    def drawCanceled(self):
        pass

    def editStarted(self):
        pass

    def editStopped(self):
        pass

    def anchorMoved(self, label, x, y, idx):
        pass

    def xData(self):
        return []

    def yData(self):
        return []


class PolygonRoiItem(RoiItemBase):
    shape = 'polygon'

    def drawFinished(self, event):
        self._xData = event['xdata'].reshape(-1)
        self._yData = event['ydata'].reshape(-1)
        points = event['points']
        uuid = str(id(self))

        # len(points) - 1 because the first and last points are the same!
        vertices = ['V{0}_{1}'.format(idx, uuid)
                    for idx in range(len(points - 1))]
        map(self.registerAnchor, vertices, points)

    def anchorMoved(self, label, x, y, idx):
        self._xData[idx] = x
        self._yData[idx] = y
        self._xData[-1] = self._xData[0]
        self._yData[-1] = self._yData[0]

    def xData(self):
        return self._xData[:]

    def yData(self):
        return self._yData[:]


class RectRoiItem(RoiItemBase):
    shape = 'rectangle'

    def drawFinished(self, event):
        self._left = event['x']
        self._bottom = event['y']
        self._right = self._left + event['width']
        self._top = self._bottom + event['height']

        # initial coordinates of the rect, in that order :
        # bottom left, top left, top right, bottom right
        # which means the edges will always be (and the code assumes this) :
        #   vertical edges : [0, 1] and [2, 3]
        #   horizontal edges : [0, 3] and [1, 2]
        uuid = str(id(self))
        corners = ['C{0}_{1}'.format(idx, uuid) for idx in range(4)]
        xcoords = np.array([self._left, self._left, self._right, self._right])
        ycoords = np.array([self._bottom, self._top, self._top, self._bottom])
        opposites = [[3, 1], [2, 0], [1, 3], [0, 2]]
        rubber = 'RUBBER_{0}'.format(uuid)
        center = 'CENTER_{0}'.format(uuid)

        self._corners = corners
        self._rubber = rubber
        self._xcoords = xcoords
        self._ycoords = ycoords
        self._opposites = opposites
        self._center = center

        # note that the order of the anchors is preserved
        # (this is the index sent to the anchorMoved method)
        # since we re registering the corners first,
        # we will be able to use the index given to the anchorMoved
        # function to get data in the _xcoords, ycoords, ... arrays.
        # this only works because we re not adding or removing vertices
        # when editing
        {self.registerAnchor(corner, (xcoords[i], ycoords[i]))
         for i, corner in enumerate(corners)}
        self.registerAnchor(center, self.center)

    left = property(lambda self: self._left)
    right = property(lambda self: self._right)
    bottom = property(lambda self: self._bottom)
    top = property(lambda self: self._top)

    @property
    def center(self):
        xcoord = self.left + (self.right - self.left) / 2.
        ycoord = self.bottom + (self.top - self.bottom) / 2.
        return [xcoord, ycoord]

    def xData(self):
        return self._xcoords[:]

    def yData(self):
        return self._ycoords[:]

    def anchorMoved(self, name, x, y, index):
        if name == self._center:
            # center moved
            c_x, c_y = self.center
            self._xcoords += x - c_x
            self._ycoords += y - c_y
            {self.setAnchorData(corner,
                                (self._xcoords[i], self._ycoords[i]))
             for i, corner in enumerate(self._corners)}
        else:
            # see the comment about the index value
            # (in the finished method)
            h_op, v_op = self._opposites[index]

            v_op_x = self._xcoords[v_op]
            h_op_y = self._xcoords[h_op]

            newLeft = min(x, v_op_x)
            newRight = max(x, v_op_x)
            newBottom = min(y, h_op_y)
            newTop = max(y, h_op_y)

            if newLeft != v_op_x:
                self._xcoords[v_op] = newLeft
            if newRight != v_op_x:
                self._xcoords[v_op] = newRight
            if newBottom != h_op_y:
                self._ycoords[h_op] = newBottom
            if newTop != h_op_y:
                self._ycoords[h_op] = newTop

            self._xcoords[index] = x
            self._ycoords[index] = y

            self.setAnchorData(self._center, self.center)
            {self.setAnchorData(self._corners[i],
                                (self._xcoords[i], self._ycoords[i]))
             for i in (v_op, h_op)}

        # caching positions
        self._left = min(self._xcoords[1:3])
        self._right = max(self._xcoords[1:3])
        self._bottom = min(self._ycoords[0:2])
        self._top = max(self._ycoords[0:2])


class ImageRoiManager(qt.QObject):
    sigRoiAdded = qt.Signal(str)
    sigRoiMoved = qt.Signal(str, object)
    sigRoiDeleted = qt.Signal(str)

    shapes = ('zoom', 'edit', 'rectangle', 'polygon',)
    icons = ('normal', 'crosshair', 'shape-rectangle', 'shape-polygon',)
    classes = (None, None, RectRoiItem, PolygonRoiItem,)

    def __init__(self, plot, parent=None):
        super(ImageRoiManager, self).__init__(parent=parent)

        self._plot = plot

        self._actionGroup = actionGroup = qt.QActionGroup(self)

        # TODO : set the initial state of the actions
        # depending on the drawing mode?
        self._actions = {}
        for shape, icon, klass in zip(self.shapes, self.icons, self.classes):
            action = qt.QAction(icons.getQIcon(icon), shape, None)
            action.setCheckable(True)
            action.setChecked(False)
            self._actions[shape] = (action, klass)
            actionGroup.addAction(action)

        actionGroup.triggered.connect(self._actionTriggered,
                                      qt.Qt.QueuedConnection)

        self._plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged, qt.Qt.QueuedConnection)

        self._currentShape = None

        self._rois = {}

    def _actionTriggered(self, action):

        if not action.isChecked():
            return

        shape, klass = [(shape, klass) for shape, (act, klass)
                        in self._actions.items() if act == action][0]

        # zoom mode, no need to listen to the plot signals anymore
        # because for now the rois can only be toggled through this
        # class (but later... who knows)
        if shape == 'zoom':
            self._currentShape = None
            self._plot.setInteractiveMode('zoom', source=self)
        elif shape == 'edit':
            self._plot.setInteractiveMode('zoom', source=self)
            {item.edit(True) for item in self._rois.values()}
        else:
            {item.edit(False) for item in self._rois.values()}
            self._currentShape = shape
            item = klass(self._plot, self)
            item.sigRoiFinished.connect(self._roiDrawingFinished,
                                        qt.Qt.QueuedConnection)
            self._rois[item.name] = item
            item.start()

    def _roiDrawingFinished(self, name):
        item = self._rois[name]
        item.sigRoiFinished.disconnect(self._roiDrawingFinished)
        self._actions['edit'][0].setChecked(True)
        self._plot.setInteractiveMode('zoom', source=self)
        {item.edit(True) for item in self._rois.values()}

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:
        If changed from elsewhere, disable tool
        """
        if source in self._rois.values():
            pass
        elif source is not self:
            if self._currentShape is not None:
                action = self._actions[self._currentShape][0]
                action.setChecked(False)
                self._currentShape = None
            mode = self._plot.getInteractiveMode()
            if mode['mode'] == 'zoom':
                self._actions['zoom'][0].setChecked(True)
            else:
                self._actions['zoom'][0].setChecked(False)

    roiActions = property(lambda self: (self._actions[name][0]
                                        for name in self.shapes))


if __name__ == '__main__':
    from silx.gui.plot import ImageView, PlotWindow

    data = np.ones((100, 200))
    data[0::5, 0::5] = 0

    app = qt.QApplication([])
    if 0:
        plotWin = PlotWindow()
        plotWin.addImage(data)
    else:
        plotWin = ImageView.ImageView()
        plotWin.setImage(data)

    plotWin.setKeepDataAspectRatio(True)
    plotWin.setAttribute(qt.Qt.WA_DeleteOnClose)
    plotWin.setActiveCurveHandling(False)

    roiManager = ImageRoiManager(plotWin)
    toolBar = qt.QToolBar('Selection')
    toolBar.addWidget(qt.QLabel('Roi'))
    {toolBar.addAction(action) for action in roiManager.roiActions}
    plotWin.addToolBar(toolBar)
    plotWin.show()
    app.exec_()
