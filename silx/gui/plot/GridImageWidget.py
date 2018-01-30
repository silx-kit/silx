# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Widget displaying a 3D volume or a list of 2D images as a grid of
plotted images. Each plot has a slider to allow selecting any one of the
images. All plots have their axes synchronized.

:meth:`GridImageWidget.setImages`

"""

from .. import qt
from . import Plot2D

from silx.third_party import six


class GridImageWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self._nrows = 2
        self._ncols = 2
        self._maxNPlots = 100
        self._plots = {}
        """:class:`Plot2D` indexed by 2-tuples (row, col)"""

        self.gridLayout = qt.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.gridLayout)

        self._updateGrid()

    def setNRows(self, nrows):
        assert isinstance(nrows, six.integer_types)
        if nrows * self._ncols > self._maxNPlots:
            max_nrows = self._maxNPlots // self._ncols
            raise ValueError(
                "Cannot increase nrows to more than %d." % max_nrows +
                "You need to decrease the number of cols first, or explicitly"
                " increase the limit (probably a bad idea).")
        if nrows == self._nrows:
            return
        self._nrows = nrows
        self._updateGrid()

    def setNCols(self, ncols):
        assert isinstance(ncols, six.integer_types)
        if ncols * self._nrows > self._maxNPlots:
            max_ncols = self._maxNPlots // self._nrows
            raise ValueError(
                "Cannot increase ncols to more than %d." % max_ncols +
                "You need to decrease the number of rows first, or explicitly"
                " increase the limit (probably a bad idea).")
        if ncols == self._ncols:
            return
        self._ncols = ncols
        self._updateGrid()

    def setMaxNPlots(self, nplots):
        """Update the max number of plots allowed.
        This is probably a bad idea.

        The maximum number of plots constrains the max number of rows
        and columns.

        :param int nplots: Maximum number of plots allowed.
        """
        assert isinstance(nplots, six.integer_types)
        self._maxNPlots = nplots

    def _updateGrid(self):
        """Show or hide the plots according to the current grid shape.
        Instantiate new plots if necessary."""
        # instantiate new plots as needed
        for r in range(self._nrows):
            for c in range(self._ncols):
                if (r, c) not in self._plots:
                    self._plots[(r, c)] = self._instantiateNewPlot()

        # show or hide plots as needed
        for idx in self._plots:
            r, c = idx
            if r < self._nrows and c < self._ncols:
                self._plots[idx].show()
            else:
                self._plots[idx].hide()

    def _instantiateNewPlot(self):
        return Plot2D(self)    # Fixme: composite widget with plot, slider, axis synchronized...
