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
# ###########################################################################*/
"""Test silx.gui.plot.StatsWidget with SceneWidget and ScalarFieldView"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/01/2019"


import pytest

import numpy

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot.stats.stats import Stats
from silx.gui import qt

from silx.gui.plot.StatsWidget import BasicStatsWidget

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d.SceneWidget import SceneWidget, items


@pytest.mark.usefixtures("use_opengl")
class TestSceneWidget(TestCaseQt, ParametricTestCase):
    """Tests StatsWidget combined with SceneWidget"""

    def setUp(self):
        super(TestSceneWidget, self).setUp()
        self.sceneWidget = SceneWidget()
        self.sceneWidget.resize(300, 300)
        self.sceneWidget.show()
        self.statsWidget = BasicStatsWidget()
        self.statsWidget.setPlot(self.sceneWidget)
        # self.qWaitForWindowExposed(self.sceneWidget)

    def tearDown(self):
        Stats._getContext.cache_clear()
        self.qapp.processEvents()
        self.sceneWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.sceneWidget.close()
        del self.sceneWidget
        self.statsWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.statsWidget.close()
        del self.statsWidget
        super(TestSceneWidget, self).tearDown()

    def test(self):
        """Test StatsWidget with SceneWidget"""
        # Prepare scene

        # Data image
        image = self.sceneWidget.addImage(numpy.arange(100).reshape(10, 10))
        image.setLabel('Image')
        # RGB image
        imageRGB = self.sceneWidget.addImage(
            numpy.arange(300, dtype=numpy.uint8).reshape(10, 10, 3))
        imageRGB.setLabel('RGB Image')
        # 2D scatter
        data = numpy.arange(100)
        scatter2D = self.sceneWidget.add2DScatter(x=data, y=data, value=data)
        scatter2D.setLabel('2D Scatter')
        # 3D scatter
        scatter3D = self.sceneWidget.add3DScatter(x=data, y=data, z=data, value=data)
        scatter3D.setLabel('3D Scatter')
        # Add a group
        group = items.GroupItem()
        self.sceneWidget.addItem(group)
        # 3D scalar field
        data = numpy.arange(64**3).reshape(64, 64, 64)
        scalarField = items.ScalarField3D()
        scalarField.setData(data, copy=False)
        scalarField.setLabel('3D Scalar field')
        group.addItem(scalarField)

        statsTable = self.statsWidget._getStatsTable()

        # Test selection only
        self.statsWidget.setDisplayOnlyActiveItem(True)
        self.assertEqual(statsTable.rowCount(), 0)

        self.sceneWidget.selection().setCurrentItem(group)
        self.assertEqual(statsTable.rowCount(), 0)

        for item in (image, scatter2D, scatter3D, scalarField):
            with self.subTest('selection only', item=item.getLabel()):
                self.sceneWidget.selection().setCurrentItem(item)
                self.assertEqual(statsTable.rowCount(), 1)
                self._checkItem(item)

        # Test all data
        self.statsWidget.setDisplayOnlyActiveItem(False)
        self.assertEqual(statsTable.rowCount(), 4)

        for item in (image, scatter2D, scatter3D, scalarField):
            with self.subTest('all items', item=item.getLabel()):
                self._checkItem(item)

    def _checkItem(self, item):
        """Check that item is in StatsTable and that stats are OK

        :param silx.gui.plot3d.items.Item3D item:
        """
        if isinstance(item, (items.Scatter2D, items.Scatter3D)):
            data = item.getValueData(copy=False)
        else:
            data = item.getData(copy=False)

        statsTable = self.statsWidget._getStatsTable()
        tableItems = statsTable._itemToTableItems(item)
        self.assertTrue(len(tableItems) > 0)
        self.assertEqual(tableItems['legend'].text(), item.getLabel())
        self.assertEqual(float(tableItems['min'].text()), numpy.min(data))
        self.assertEqual(float(tableItems['max'].text()), numpy.max(data))
        # TODO


class TestScalarFieldView(TestCaseQt):
    """Tests StatsWidget combined with ScalarFieldView"""

    def setUp(self):
        super(TestScalarFieldView, self).setUp()
        self.scalarFieldView = ScalarFieldView()
        self.scalarFieldView.resize(300, 300)
        self.scalarFieldView.show()
        self.statsWidget = BasicStatsWidget()
        self.statsWidget.setPlot(self.scalarFieldView)
        # self.qWaitForWindowExposed(self.sceneWidget)

    def tearDown(self):
        Stats._getContext.cache_clear()
        self.qapp.processEvents()
        self.scalarFieldView.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.scalarFieldView.close()
        del self.scalarFieldView
        self.statsWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.statsWidget.close()
        del self.statsWidget
        super(TestScalarFieldView, self).tearDown()

    def _getTextFor(self, row, name):
        """Returns text in table at given row for column name

        :param int row: Row number in the table
        :param str name: Column id
        :rtype: Union[str,None]
        """
        statsTable = self.statsWidget._getStatsTable()

        for column in range(statsTable.columnCount()):
            headerItem = statsTable.horizontalHeaderItem(column)
            if headerItem.data(qt.Qt.UserRole) == name:
                tableItem = statsTable.item(row, column)
                return tableItem.text()

        return None

    def test(self):
        """Test StatsWidget with ScalarFieldView"""
        data = numpy.arange(64**3, dtype=numpy.float64).reshape(64, 64, 64)
        self.scalarFieldView.setData(data)

        statsTable = self.statsWidget._getStatsTable()

        # Test selection only
        self.statsWidget.setDisplayOnlyActiveItem(True)
        self.assertEqual(statsTable.rowCount(), 1)

        # Test all data
        self.statsWidget.setDisplayOnlyActiveItem(False)
        self.assertEqual(statsTable.rowCount(), 1)

        for column in range(statsTable.columnCount()):
            self.assertEqual(float(self._getTextFor(0, 'min')), numpy.min(data))
            self.assertEqual(float(self._getTextFor(0, 'max')), numpy.max(data))
            sum_ = numpy.sum(data)
            comz = numpy.sum(numpy.arange(data.shape[0]) * numpy.sum(data, axis=(1, 2))) / sum_
            comy = numpy.sum(numpy.arange(data.shape[1]) * numpy.sum(data, axis=(0, 2))) / sum_
            comx = numpy.sum(numpy.arange(data.shape[2]) * numpy.sum(data, axis=(0, 1))) / sum_
            self.assertEqual(self._getTextFor(0, 'COM'), str((comx, comy, comz)))
