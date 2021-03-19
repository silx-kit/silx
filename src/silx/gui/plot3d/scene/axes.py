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
"""Primitive displaying a text field in the scene."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/10/2016"


import logging
import numpy

from ...plot._utils import ticklayout

from . import core, primitives, text, transform


_logger = logging.getLogger(__name__)


class LabelledAxes(primitives.GroupBBox):
    """A group displaying a bounding box with axes labels around its children.
    """

    def __init__(self):
        super(LabelledAxes, self).__init__()
        self._ticksForBounds = None

        self._font = text.Font()

        self._boxVisibility = True

        # TODO offset labels from anchor in pixels

        self._xlabel = text.Text2D(font=self._font)
        self._xlabel.align = 'center'
        self._xlabel.transforms = [self._boxTransforms,
                                   transform.Translate(tx=0.5)]
        self._children.insert(-1, self._xlabel)

        self._ylabel = text.Text2D(font=self._font)
        self._ylabel.align = 'center'
        self._ylabel.transforms = [self._boxTransforms,
                                   transform.Translate(ty=0.5)]
        self._children.insert(-1, self._ylabel)

        self._zlabel = text.Text2D(font=self._font)
        self._zlabel.align = 'center'
        self._zlabel.transforms = [self._boxTransforms,
                                   transform.Translate(tz=0.5)]
        self._children.insert(-1, self._zlabel)

        # Init tick lines with dummy pos
        self._tickLines = primitives.DashedLines(
            positions=((0., 0., 0.), (0., 0., 0.)))
        self._tickLines.dash = 5, 10
        self._tickLines.visible = False
        self._children.insert(-1, self._tickLines)

        self._tickLabels = core.Group()
        self._children.insert(-1, self._tickLabels)

        # Sync color
        self.tickColor = 1., 1., 1., 1.

    def _updateBoxAndAxes(self):
        """Update bbox and axes position and size according to children.

        Overridden from GroupBBox
        """
        super(LabelledAxes, self)._updateBoxAndAxes()

        bounds = self._group.bounds(dataBounds=True)
        if bounds is not None:
            tx, ty, tz = (bounds[1] - bounds[0]) / 2.
        else:
            tx, ty, tz = 0.5, 0.5, 0.5

        self._xlabel.transforms[-1].tx = tx
        self._ylabel.transforms[-1].ty = ty
        self._zlabel.transforms[-1].tz = tz

    @property
    def tickColor(self):
        """Color of ticks and text labels.

        This does NOT set bounding box color.
        Use :attr:`color` for the bounding box.
        """
        return self._xlabel.foreground

    @tickColor.setter
    def tickColor(self, color):
        self._xlabel.foreground = color
        self._ylabel.foreground = color
        self._zlabel.foreground = color
        transparentColor = color[0], color[1], color[2], color[3] * 0.6
        self._tickLines.setAttribute('color', transparentColor)
        for label in self._tickLabels.children:
            label.foreground = color

    @property
    def font(self):
        """Font of axes text labels (Font)"""
        return self._font

    @font.setter
    def font(self, font):
        self._font = font
        self._xlabel.font = font
        self._ylabel.font = font
        self._zlabel.font = font
        for label in self._tickLabels.children:
            label.font = font

    @property
    def xlabel(self):
        """Text label of the X axis (str)"""
        return self._xlabel.text

    @xlabel.setter
    def xlabel(self, text):
        self._xlabel.text = text

    @property
    def ylabel(self):
        """Text label of the Y axis (str)"""
        return self._ylabel.text

    @ylabel.setter
    def ylabel(self, text):
        self._ylabel.text = text

    @property
    def zlabel(self):
        """Text label of the Z axis (str)"""
        return self._zlabel.text

    @zlabel.setter
    def zlabel(self, text):
        self._zlabel.text = text

    @property
    def boxVisible(self):
        """Returns bounding box, axes labels and grid visibility."""
        return self._boxVisibility

    @boxVisible.setter
    def boxVisible(self, visible):
        self._boxVisibility = bool(visible)
        for child in self._children:
            if child == self._tickLines:
                if self._ticksForBounds is not None:
                    child.visible = self._boxVisibility
            elif child != self._group:
                child.visible = self._boxVisibility

    def _updateTicks(self):
        """Check if ticks need update and update them if needed."""
        bounds = self._group.bounds(transformed=False, dataBounds=True)
        if bounds is None:  # No content
            if self._ticksForBounds is not None:
                self._ticksForBounds = None
                self._tickLines.visible = False
                self._tickLabels.children = []  # Reset previous labels

        elif (self._ticksForBounds is None or
                not numpy.all(numpy.equal(bounds, self._ticksForBounds))):
            self._ticksForBounds = bounds

            # Update ticks
            ticklength = numpy.abs(bounds[1] - bounds[0])

            xticks, xlabels = ticklayout.ticks(*bounds[:, 0])
            yticks, ylabels = ticklayout.ticks(*bounds[:, 1])
            zticks, zlabels = ticklayout.ticks(*bounds[:, 2])

            # Update tick lines
            coords = numpy.empty(
                ((len(xticks) + len(yticks) + len(zticks)), 4, 3),
                dtype=numpy.float32)
            coords[:, :, :] = bounds[0, :]  # account for offset from origin

            xcoords = coords[:len(xticks)]
            xcoords[:, :, 0] = numpy.asarray(xticks)[:, numpy.newaxis]
            xcoords[:, 1, 1] += ticklength[1]  # X ticks on XY plane
            xcoords[:, 3, 2] += ticklength[2]  # X ticks on XZ plane

            ycoords = coords[len(xticks):len(xticks) + len(yticks)]
            ycoords[:, :, 1] = numpy.asarray(yticks)[:, numpy.newaxis]
            ycoords[:, 1, 0] += ticklength[0]  # Y ticks on XY plane
            ycoords[:, 3, 2] += ticklength[2]  # Y ticks on YZ plane

            zcoords = coords[len(xticks) + len(yticks):]
            zcoords[:, :, 2] = numpy.asarray(zticks)[:, numpy.newaxis]
            zcoords[:, 1, 0] += ticklength[0]  # Z ticks on XZ plane
            zcoords[:, 3, 1] += ticklength[1]  # Z ticks on YZ plane

            self._tickLines.setPositions(coords.reshape(-1, 3))
            self._tickLines.visible = self._boxVisibility

            # Update labels
            color = self.tickColor
            offsets = bounds[0] - ticklength / 20.
            labels = []
            for tick, label in zip(xticks, xlabels):
                text2d = text.Text2D(text=label, font=self.font)
                text2d.align = 'center'
                text2d.foreground = color
                text2d.transforms = [transform.Translate(
                    tx=tick, ty=offsets[1], tz=offsets[2])]
                labels.append(text2d)

            for tick, label in zip(yticks, ylabels):
                text2d = text.Text2D(text=label, font=self.font)
                text2d.align = 'center'
                text2d.foreground = color
                text2d.transforms = [transform.Translate(
                    tx=offsets[0], ty=tick, tz=offsets[2])]
                labels.append(text2d)

            for tick, label in zip(zticks, zlabels):
                text2d = text.Text2D(text=label, font=self.font)
                text2d.align = 'center'
                text2d.foreground = color
                text2d.transforms = [transform.Translate(
                    tx=offsets[0], ty=offsets[1], tz=tick)]
                labels.append(text2d)

            self._tickLabels.children = labels  # Reset previous labels

    def prepareGL2(self, context):
        self._updateTicks()
        super(LabelledAxes, self).prepareGL2(context)
