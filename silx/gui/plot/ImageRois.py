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
"""Roi items."""

import numpy as np
from silx.gui import qt, icons

__author__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/09/2016"


class RoiItemBase(qt.QObject):
    sigRoiDrawingStarted = qt.Signal(str)
    sigRoiDrawingFinished = qt.Signal(str)
    sigRoiDrawingCanceled = qt.Signal(str)
    sigRoiMoved = qt.Signal(str)

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
        self._startNotified = False
        self._connected = False
        self._editing = False
        self._visible = True

        if not name:
            uuid = str(id(self))
            name = '{0}_{1}'.format(self.__class__.__name__, uuid)

        self._name = name

    def setVisible(self, visible):
        self._visible = visible
        if not visible:
            self._disconnect()
            self._remove()
        else:
            self._connect()
            self._draw(drawAnchors=self._editing)

    def _remove(self, anchors=True, shape=True):
        if anchors:
            {self._plot.removeMarker(item) for item in self._anchors}
        if shape:
            self._plot.removeItem(self._name)

    def _interactiveModeChanged(self, source):
        if source is not self or source is not self.parent():
            self.stop()

    def _plotSignal(self, event):
        evType = event['event']
        if (evType == 'drawingFinished' and
                event['parameters']['label'] == self._name):
            self._finish(event)
        elif (evType == 'drawingProgress' and
                event['parameters']['label'] == self._name):
            if not self._startNotified:
                self.drawStarted()
                # TODO : this is a temporary workaround
                # until we can get a mouse click event
                self.sigRoiDrawingStarted.emit(self.name)
                self._startNotified = True
            self.sigRoiMoved.emit(self._name)
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
                self.sigRoiMoved.emit(self._name)
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
        self._startNotified = False
        self._visible = True
        self._plot.setInteractiveMode('draw',
                                      shape=self.shape,
                                      source=self,
                                      label=self._name)
        self._connect()

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
            self._remove(shape=False)
            self.editStopped()
            self._draw(drawAnchors=False)

        self._editing = enable

    def _finish(self, event):
        self.drawFinished(event)
        self._draw(drawAnchors=False)
        self._finished = True
        self.sigRoiDrawingFinished.emit(self.name)
        self._disconnect()

    def stop(self):
        """
        Stops whatever state the ROI is in.
        draw state : cancel the drawning, emit sigRoiDrawningCanceled
        edit state : ends the editing, emit sigRoiEditingFinished
        """
        if not self._finished:
            self._disconnect()
            self.drawCanceled()
            if self._startNotified:
                self.sigRoiDrawingCanceled.emit(self.name)
            return
        if self._editing:
            self.edit(False)

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
    roiShapes = ('rectangle', 'polygon',)
    roiIcons = ('shape-rectangle', 'shape-polygon',)
    roiClasses = (RectRoiItem, PolygonRoiItem,)

    sigRoiDrawingStarted = qt.Signal(str)
    sigRoiDrawingFinished = qt.Signal(str)
    sigRoiDrawingCanceled = qt.Signal(str)
    sigRoiMoved = qt.Signal(str)
    sigRoiRemoved = qt.Signal(str)

    def __init__(self, plot, parent=None):
        super(ImageRoiManager, self).__init__(parent=parent)

        self._plot = plot

        self._shapes = ('zoom', 'edit') + self.roiShapes
        self._icons = ('normal', 'crosshair') + self.roiIcons
        self._classes = (None, None) + self.roiClasses

        self._multipleSelection = False
        self._roiVisible = True
        self._roiInProgress = None

        self._roiActions = None
        self._optionActions = None
        self._roiActionGroup = None

        self._currentShape = None

        self._rois = {}

        self._plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged, qt.Qt.QueuedConnection)

    def _createRoiActions(self):

        if self._roiActions:
            return self._roiActions

        # roi shapes
        self._roiActionGroup = roiActionGroup = qt.QActionGroup(self)

        self._roiActions = roiActions = {}

        for shape, icon, klass in zip(self._shapes,
                                      self._icons,
                                      self._classes):

            action = qt.QAction(icons.getQIcon(icon), shape, None)
            action.setCheckable(True)
            roiActions[shape] = (action, klass)
            roiActionGroup.addAction(action)

            if shape == self._currentShape:
                action.setChecked(True)
            else:
                action.setChecked(False)

        roiActionGroup.triggered.connect(self._roiActionTriggered,
                                         qt.Qt.QueuedConnection)

        return roiActions

    def _createOptionActions(self):

        if self._optionActions:
            return self._optionActions

        # options
        self._optionActions = optionActions = {}

        action = qt.QAction(u'\u2685', None)
        action.setCheckable(True)
        action.setChecked(self._multipleSelection)
        action.setToolTip('Single/Multiple ROI selection')
        action.setText(u'\u2682' if self._multipleSelection else u'\u2680')
        action.triggered.connect(self.allowMultipleSelections)
        optionActions['multiple'] = action

        action = qt.QAction(u'\U0001F440', None)
        action.setCheckable(True)
        action.setChecked(self._roiVisible)
        action.setToolTip('Show/Hide ROI(s)')
        action.triggered.connect(self.showRois)
        optionActions['show'] = action

        return optionActions

    def clear(self, name=None):
        if name is None:
            for roi in self._rois.values():
                roi.setVisible(False)
                try:
                    roi.sigRoiMoved.disconnect(self._roiMoved)
                except:
                    pass
                self.sigRoiRemoved.emit(roi.name)
            self._rois = {}
        else:
            try:
                roi = self._rois.pop(name)
                roi.setVisible(False)
                try:
                    roi.sigRoiMoved.disconnect(self._roiMoved)
                except:
                    pass
                self.sigRoiRemoved.emit(roi.name)
            except KeyError:
                # TODO : to raise or not to raise?
                pass

    rois = property(lambda self: self._rois.keys())

    def showRois(self, show):
        # TODO : name param to tell that we only want to toggle
        # one specific ROI
        # TODO : exclusive param to tell that we want to show only
        # one given roi (w/ name param)
        self._roiVisible = show

        if self._optionActions:
            action = self._optionActions['show']
            action.setText(u'\U0001F440' if show else u'\u2012')
            if self.sender() != action:
                action.setChecked(show)

        {roi.setVisible(show) for roi in self._rois.values()}

    def allowMultipleSelections(self, allow):

        self._multipleSelection = allow
        if self._optionActions:
            action = self._optionActions['multiple']
            action.setText(u'\u2682' if allow else u'\u2680')
            if self.sender() != action:
                action.setChecked(allow)

    def _roiActionTriggered(self, action):

        if not action.isChecked():
            return

        shape, klass = [(shape, klass) for shape, (act, klass)
                        in self._roiActions.items() if act == action][0]

        self._currentShape = shape

        if shape == 'zoom':
            self._plot.setInteractiveMode('zoom', source=self)
        elif shape == 'edit':
            self._plot.setInteractiveMode('zoom', source=self)
            {item.edit(True) for item in self._rois.values()}
        else:
            self._startRoi()

    def _startRoi(self):
        """
        Initialize a new roi, ready to be drawn.
        """
        if self._currentShape not in self.roiShapes:
            return

        self._stopRoi()

        {item.edit(False) for item in self._rois.values()}
        self.showRois(True)

        i_shape = self.roiShapes.index(self._currentShape)
        klass = self.roiClasses[i_shape]
        item = klass(self._plot, self)

        self._roiInProgress = item

        item.sigRoiDrawingFinished.connect(self._roiDrawingFinished,
                                           qt.Qt.QueuedConnection)
        item.sigRoiDrawingStarted.connect(self._roiDrawingStarted,
                                          qt.Qt.QueuedConnection)
        item.sigRoiDrawingCanceled.connect(self._roiDrawingCanceled,
                                          qt.Qt.QueuedConnection)
        item.sigRoiMoved.connect(self.sigRoiMoved,
                                 qt.Qt.QueuedConnection)
        item.start()

    def _stopRoi(self):
        """
        Stops the roi that was ready to be drawn, if any.
        """
        if self._roiInProgress is None:
            return
        self._roiInProgress.stop()
        self._roiInProgress = None

    def _roiDrawingStarted(self, name):
        if not self._multipleSelection:
            self.clear()
        self.sigRoiDrawingStarted.emit(name)

    def _roiDrawingFinished(self, name):
        # TODO : check if the sender is the same as the roiInProgress
        item = self._roiInProgress

        self._roiInProgress = None

        item.sigRoiDrawingFinished.disconnect(self._roiDrawingFinished)
        item.sigRoiDrawingStarted.disconnect(self._roiDrawingStarted)
        item.sigRoiDrawingCanceled.disconnect(self._roiDrawingCanceled)
        item.sigRoiMoved.disconnect(self.sigRoiMoved)

        self._rois[item.name] = item

        self.sigRoiDrawingFinished.emit(item.name)

        self._startRoi()

    def _roiDrawingCanceled(self, name):
        self.sigRoiDrawingCanceled.emit(name)

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:
        If changed from elsewhere, disable tool.
        """
        if source in self._rois.values() or source == self._roiInProgress:
            pass
        elif source is not self:
            mode = self._plot.getInteractiveMode()
            if mode['mode'] == 'zoom':
                self._currentShape = 'zoom'
                if self._roiActions:
                    self._roiActions['zoom'][0].setChecked(True)
            elif self._currentShape is not None and self._roiActions:
                self._roiActions[self._currentShape][0].setChecked(False)
                self._currentShape = None

    roiActions = property(lambda self: (self._createRoiActions()[name][0]
                                        for name in self._shapes))

    @property
    def optionActions(self):
        actions = self._createOptionActions()
        return [actions[k] for k in sorted(actions.keys())]

    def toolBar(self,
                options=None,
                shapes=None):
        """
        shapes : list
        options : list
        """
        if shapes is not None:
            # this wont work if shape is a string and not an array
            diff = set(shapes) - set(self.roiShapes)
            if len(diff) > 0:
                raise ValueError('Unknown shape(s) {0}.'.format(diff))
            shapes = set(self._shapes) - (set(self.roiShapes) - set(shapes))
        else:
            shapes = self._shapes

        optionActions = self._createOptionActions()
        # TODO : find a better way to ensure that the order of
        # actions returned is always the same
        optionNames = sorted(optionActions.keys())
        if options is not None:
            # this wont work if shape is a string and not an array
            diff = set(options) - set(optionNames)
            if len(diff) > 0:
                raise ValueError('Unknown options(s) {0}.'.format(diff))
            options = [option for option in optionNames
                       if option in options]
        else:
            options = optionNames

        roiActions = [self._createRoiActions()[name][0]
                      for name in shapes]
        optionActions = [optionActions[option] for option in options]

        toolBar = qt.QToolBar('Roi')
        toolBar.addWidget(qt.QLabel('Roi'))
        {toolBar.addAction(action) for action in roiActions}

        toolBar.addSeparator()

        {toolBar.addAction(action) for action in optionActions}

        return toolBar

    def roiCoords(self, name):
        try:
            item = self._rois[name]
        except KeyError:
            raise ValueError('Unknown roi {0}.'.format(name))

        return (item.xData(), item.yData())


if __name__ == '__main__':
    pass
