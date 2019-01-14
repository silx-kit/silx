# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""Test the plot's save action (consistency of output)"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "28/11/2017"


import unittest
import tempfile
import os

from silx.gui.plot.test.utils import PlotWidgetTestCase

from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.io import SaveAction


class TestSaveActionSaveCurvesAsSpec(unittest.TestCase):

    def setUp(self):
        self.plot = PlotWidget(backend='none')
        self.saveAction = SaveAction(plot=self.plot)

        self.tempdir = tempfile.mkdtemp()
        self.out_fname = os.path.join(self.tempdir, "out.dat")

    def tearDown(self):
        os.unlink(self.out_fname)
        os.rmdir(self.tempdir)

    def testSaveMultipleCurvesAsSpec(self):
        """Test that labels are properly used."""
        self.plot.setGraphXLabel("graph x label")
        self.plot.setGraphYLabel("graph y label")

        self.plot.addCurve([0, 1], [1, 2], "curve with labels",
                           xlabel="curve0 X", ylabel="curve0 Y")
        self.plot.addCurve([-1, 3], [-6, 2], "curve with X label",
                           xlabel="curve1 X")
        self.plot.addCurve([-2, 0], [8, 12], "curve with Y label",
                           ylabel="curve2 Y")
        self.plot.addCurve([3, 1], [7, 6], "curve with no labels")

        self.saveAction._saveCurves(self.plot,
                                    self.out_fname,
                                    SaveAction.DEFAULT_ALL_CURVES_FILTERS[0])  # "All curves as SpecFile (*.dat)"

        with open(self.out_fname, "rb") as f:
            file_content = f.read()
            if hasattr(file_content, "decode"):
                file_content = file_content.decode()

            # case with all curve labels specified
            self.assertIn("#S 1 curve0 Y", file_content)
            self.assertIn("#L curve0 X  curve0 Y", file_content)

            # graph X&Y labels are used when no curve label is specified
            self.assertIn("#S 2 graph y label", file_content)
            self.assertIn("#L curve1 X  graph y label", file_content)

            self.assertIn("#S 3 curve2 Y", file_content)
            self.assertIn("#L graph x label  curve2 Y", file_content)

            self.assertIn("#S 4 graph y label", file_content)
            self.assertIn("#L graph x label  graph y label", file_content)


class TestSaveActionExtension(PlotWidgetTestCase):
    """Test SaveAction file filter API"""

    def _dummySaveFunction(self, plot, filename, nameFilter):
        pass

    def testFileFilterAPI(self):
        """Test addition/update of a file filter"""
        saveAction = SaveAction(plot=self.plot, parent=self.plot)

        # Add a new file filter
        nameFilter = 'Dummy file (*.dummy)'
        saveAction.setFileFilter('all', nameFilter, self._dummySaveFunction)
        self.assertTrue(nameFilter in saveAction.getFileFilters('all'))
        self.assertEqual(saveAction.getFileFilters('all')[nameFilter],
                         self._dummySaveFunction)

        # Add a new file filter at a particular position
        nameFilter = 'Dummy file2 (*.dummy)'
        saveAction.setFileFilter('all', nameFilter,
                                 self._dummySaveFunction, index=3)
        self.assertTrue(nameFilter in saveAction.getFileFilters('all'))
        filters = saveAction.getFileFilters('all')
        self.assertEqual(filters[nameFilter], self._dummySaveFunction)
        self.assertEqual(list(filters.keys()).index(nameFilter),3)

        # Update an existing file filter
        nameFilter = SaveAction.IMAGE_FILTER_EDF
        saveAction.setFileFilter('image', nameFilter, self._dummySaveFunction)
        self.assertEqual(saveAction.getFileFilters('image')[nameFilter],
                         self._dummySaveFunction)

        # Change the position of an existing file filter
        nameFilter = 'Dummy file2 (*.dummy)'
        oldIndex = list(saveAction.getFileFilters('all')).index(nameFilter)
        newIndex = oldIndex - 1
        saveAction.setFileFilter('all', nameFilter,
                                 self._dummySaveFunction, index=newIndex)
        filters = saveAction.getFileFilters('all')
        self.assertEqual(filters[nameFilter], self._dummySaveFunction)
        self.assertEqual(list(filters.keys()).index(nameFilter), newIndex)

def suite():
    test_suite = unittest.TestSuite()
    for cls in (TestSaveActionSaveCurvesAsSpec, TestSaveActionExtension):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
