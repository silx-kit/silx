# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides the Colormap object
"""

from __future__ import absolute_import

__authors__ = ["H.Payno"]
__license__ = "MIT"
__date__ = "05/10/2018"

import unittest
import numpy
from silx.utils.testutils import ParametricTestCase
from silx.gui import colors
from silx.gui.colors import Colormap
from silx.gui.colors import preferredColormaps, setPreferredColormaps
from silx.utils.exceptions import NotEditableError


class TestRGBA(ParametricTestCase):
    """Basic tests of rgba function"""

    def testRGBA(self):
        """"Test rgba function with accepted values"""
        tests = {  # name: (colors, expected values)
            'blue': ('blue', (0., 0., 1., 1.)),
            '#010203': ('#010203', (1. / 255., 2. / 255., 3. / 255., 1.)),
            '#01020304': ('#01020304', (1. / 255., 2. / 255., 3. / 255., 4. / 255.)),
            '3 x uint8': (numpy.array((1, 255, 0), dtype=numpy.uint8),
                          (1 / 255., 1., 0., 1.)),
            '4 x uint8': (numpy.array((1, 255, 0, 1), dtype=numpy.uint8),
                          (1 / 255., 1., 0., 1 / 255.)),
            '3 x float overflow': ((3., 0.5, 1.), (1., 0.5, 1., 1.)),
        }

        for name, test in tests.items():
            color, expected = test
            with self.subTest(msg=name):
                result = colors.rgba(color)
                self.assertEqual(result, expected)


class TestApplyColormapToData(ParametricTestCase):
    """Tests of applyColormapToData function"""

    def testApplyColormapToData(self):
        """Simple test of applyColormapToData function"""
        colormap = Colormap(name='gray', normalization='linear',
                            vmin=0, vmax=255)

        size = 10
        expected = numpy.empty((size, 4), dtype='uint8')
        expected[:, 0] = numpy.arange(size, dtype='uint8')
        expected[:, 1] = expected[:, 0]
        expected[:, 2] = expected[:, 0]
        expected[:, 3] = 255

        for dtype in ('uint8', 'int32', 'float32', 'float64'):
            with self.subTest(dtype=dtype):
                array = numpy.arange(size, dtype=dtype)
                result = colormap.applyToData(data=array)
                self.assertTrue(numpy.all(numpy.equal(result, expected)))


class TestDictAPI(unittest.TestCase):
    """Make sure the old dictionary API is working
    """

    def setUp(self):
        self.vmin = -1.0
        self.vmax = 12

    def testGetItem(self):
        """test the item getter API ([xxx])"""
        colormap = Colormap(name='viridis',
                            normalization=Colormap.LINEAR,
                            vmin=self.vmin,
                            vmax=self.vmax)
        self.assertTrue(colormap['name'] == 'viridis')
        self.assertTrue(colormap['normalization'] == Colormap.LINEAR)
        self.assertTrue(colormap['vmin'] == self.vmin)
        self.assertTrue(colormap['vmax'] == self.vmax)
        with self.assertRaises(KeyError):
            colormap['toto']

    def testGetDict(self):
        """Test the getDict function API"""
        clmObject = Colormap(name='viridis',
                             normalization=Colormap.LINEAR,
                             vmin=self.vmin,
                             vmax=self.vmax)
        clmDict = clmObject._toDict()
        self.assertTrue(clmDict['name'] == 'viridis')
        self.assertTrue(clmDict['autoscale'] is False)
        self.assertTrue(clmDict['vmin'] == self.vmin)
        self.assertTrue(clmDict['vmax'] == self.vmax)
        self.assertTrue(clmDict['normalization'] == Colormap.LINEAR)

        clmObject.setVRange(None, None)
        self.assertTrue(clmObject._toDict()['autoscale'] is True)

    def testSetValidDict(self):
        """Test that if a colormap is created from a dict then it is correctly
        created and the values are copied (so if some values from the dict
        is changing, this won't affect the Colormap object"""
        clm_dict = {
            'name': 'temperature',
            'vmin': 1.0,
            'vmax': 2.0,
            'normalization': 'linear',
            'colors': None,
            'autoscale': False
        }

        # Test that the colormap is correctly created
        colormapObject = Colormap._fromDict(clm_dict)
        self.assertTrue(colormapObject.getName() == clm_dict['name'])
        self.assertTrue(colormapObject.getColormapLUT() == clm_dict['colors'])
        self.assertTrue(colormapObject.getVMin() == clm_dict['vmin'])
        self.assertTrue(colormapObject.getVMax() == clm_dict['vmax'])
        self.assertTrue(colormapObject.isAutoscale() == clm_dict['autoscale'])

        # Check that the colormap has copied the values
        clm_dict['vmin'] = None
        clm_dict['vmax'] = None
        clm_dict['colors'] = [1.0, 2.0]
        clm_dict['autoscale'] = True
        clm_dict['normalization'] = Colormap.LOGARITHM
        clm_dict['name'] = 'viridis'

        self.assertFalse(colormapObject.getName() == clm_dict['name'])
        self.assertFalse(colormapObject.getColormapLUT() == clm_dict['colors'])
        self.assertFalse(colormapObject.getVMin() == clm_dict['vmin'])
        self.assertFalse(colormapObject.getVMax() == clm_dict['vmax'])
        self.assertFalse(colormapObject.isAutoscale() == clm_dict['autoscale'])

    def testMissingKeysFromDict(self):
        """Make sure we can create a Colormap object from a dictionnary even if
        there is missing keys excepts if those keys are 'colors' or 'name'
        """
        colormap = Colormap._fromDict({'name': 'toto'})
        self.assertTrue(colormap.getVMin() is None)
        colormap = Colormap._fromDict({'colors': numpy.zeros(10)})
        self.assertTrue(colormap.getName() is None)

        with self.assertRaises(ValueError):
            Colormap._fromDict({})

    def testUnknowNorm(self):
        """Make sure an error is raised if the given normalization is not
        knowed
        """
        clm_dict = {
            'name': 'temperature',
            'vmin': 1.0,
            'vmax': 2.0,
            'normalization': 'toto',
            'colors': None,
            'autoscale': False
        }
        with self.assertRaises(ValueError):
            Colormap._fromDict(clm_dict)

    def testNumericalColors(self):
        """Make sure the old API using colors=int was supported"""
        clm_dict = {
            'name': 'temperature',
            'vmin': 1.0,
            'vmax': 2.0,
            'colors': 256,
            'autoscale': False
        }
        Colormap._fromDict(clm_dict)


class TestObjectAPI(ParametricTestCase):
    """Test the new Object API of the colormap"""
    def testVMinVMax(self):
        """Test getter and setter associated to vmin and vmax values"""
        vmin = 1.0
        vmax = 2.0

        colormapObject = Colormap(name='viridis',
                                  vmin=vmin,
                                  vmax=vmax,
                                  normalization=Colormap.LINEAR)

        with self.assertRaises(ValueError):
            colormapObject.setVMin(3)

        with self.assertRaises(ValueError):
            colormapObject.setVMax(-2)

        with self.assertRaises(ValueError):
            colormapObject.setVRange(3, -2)

        self.assertTrue(colormapObject.getColormapRange() == (1.0, 2.0))
        self.assertTrue(colormapObject.isAutoscale() is False)
        colormapObject.setVRange(None, None)
        self.assertTrue(colormapObject.getVMin() is None)
        self.assertTrue(colormapObject.getVMax() is None)
        self.assertTrue(colormapObject.isAutoscale() is True)

    def testCopy(self):
        """Make sure the copy function is correctly processing
        """
        colormapObject = Colormap(name='toto',
                                  colors=numpy.array([12, 13, 14]),
                                  vmin=None,
                                  vmax=None,
                                  normalization=Colormap.LOGARITHM)

        colormapObject2 = colormapObject.copy()
        self.assertTrue(colormapObject == colormapObject2)
        colormapObject.setColormapLUT(numpy.array([0, 1]))
        self.assertFalse(colormapObject == colormapObject2)

        colormapObject2 = colormapObject.copy()
        self.assertTrue(colormapObject == colormapObject2)
        colormapObject.setNormalization(Colormap.LINEAR)
        self.assertFalse(colormapObject == colormapObject2)

    def testGetColorMapRange(self):
        """Make sure the getColormapRange function of colormap is correctly
        applying
        """
        # test linear scale
        data = numpy.array([-1, 1, 2, 3, float('nan')])
        cl1 = Colormap(name='gray',
                       normalization=Colormap.LINEAR,
                       vmin=0,
                       vmax=2)
        cl2 = Colormap(name='gray',
                       normalization=Colormap.LINEAR,
                       vmin=None,
                       vmax=2)
        cl3 = Colormap(name='gray',
                       normalization=Colormap.LINEAR,
                       vmin=0,
                       vmax=None)
        cl4 = Colormap(name='gray',
                       normalization=Colormap.LINEAR,
                       vmin=None,
                       vmax=None)

        self.assertTrue(cl1.getColormapRange(data) == (0, 2))
        self.assertTrue(cl2.getColormapRange(data) == (-1, 2))
        self.assertTrue(cl3.getColormapRange(data) == (0, 3))
        self.assertTrue(cl4.getColormapRange(data) == (-1, 3))

        # test linear with annoying cases
        self.assertEqual(cl3.getColormapRange((-1, -2)), (0, 0))
        self.assertEqual(cl4.getColormapRange(()), (0., 1.))
        self.assertEqual(cl4.getColormapRange(
            (float('nan'), float('inf'), 1., -float('inf'), 2)), (1., 2.))
        self.assertEqual(cl4.getColormapRange(
            (float('nan'), float('inf'))), (0., 1.))

        # test log scale
        data = numpy.array([float('nan'), -1, 1, 10, 100, 1000])
        cl1 = Colormap(name='gray',
                       normalization=Colormap.LOGARITHM,
                       vmin=1,
                       vmax=100)
        cl2 = Colormap(name='gray',
                       normalization=Colormap.LOGARITHM,
                       vmin=None,
                       vmax=100)
        cl3 = Colormap(name='gray',
                       normalization=Colormap.LOGARITHM,
                       vmin=1,
                       vmax=None)
        cl4 = Colormap(name='gray',
                       normalization=Colormap.LOGARITHM,
                       vmin=None,
                       vmax=None)

        self.assertTrue(cl1.getColormapRange(data) == (1, 100))
        self.assertTrue(cl2.getColormapRange(data) == (1, 100))
        self.assertTrue(cl3.getColormapRange(data) == (1, 1000))
        self.assertTrue(cl4.getColormapRange(data) == (1, 1000))

        # test log with annoying cases
        self.assertEqual(cl3.getColormapRange((0.1, 0.2)), (1, 1))
        self.assertEqual(cl4.getColormapRange((-2., -1.)), (1., 1.))
        self.assertEqual(cl4.getColormapRange(()), (1., 10.))
        self.assertEqual(cl4.getColormapRange(
            (float('nan'), float('inf'), 1., -float('inf'), 2)), (1., 2.))
        self.assertEqual(cl4.getColormapRange(
            (float('nan'), float('inf'))), (1., 10.))

    def testApplyToData(self):
        """Test applyToData on different datasets"""
        datasets = [
            numpy.zeros((0, 0)),  # Empty array
            numpy.array((numpy.nan, numpy.inf)),  # All non-finite
            numpy.array((-numpy.inf, numpy.inf, 1.0, 2.0)),  # Some infinite
        ]

        for normalization in ('linear', 'log'):
            colormap = Colormap(name='gray',
                                normalization=normalization,
                                vmin=None,
                                vmax=None)

            for data in datasets:
                with self.subTest(data=data):
                    image = colormap.applyToData(data)
                    self.assertEqual(image.dtype, numpy.uint8)
                    self.assertEqual(image.shape[-1], 4)
                    self.assertEqual(image.shape[:-1], data.shape)

    def testGetNColors(self):
        """Test getNColors method"""
        # specific LUT
        colormap = Colormap(name=None,
                            colors=((0., 0., 0.), (1., 1., 1.)),
                            vmin=1000,
                            vmax=2000)
        colors = colormap.getNColors()
        self.assertTrue(numpy.all(numpy.equal(
            colors,
            ((0, 0, 0, 255), (255, 255, 255, 255)))))

    def testEditableMode(self):
        """Make sure the colormap will raise NotEditableError when try to
        change a colormap not editable"""
        colormap = Colormap()
        colormap.setEditable(False)
        with self.assertRaises(NotEditableError):
            colormap.setVRange(0., 1.)
        with self.assertRaises(NotEditableError):
            colormap.setVMin(1.)
        with self.assertRaises(NotEditableError):
            colormap.setVMax(1.)
        with self.assertRaises(NotEditableError):
            colormap.setNormalization(Colormap.LOGARITHM)
        with self.assertRaises(NotEditableError):
            colormap.setName('magma')
        with self.assertRaises(NotEditableError):
            colormap.setColormapLUT(numpy.array([0, 1]))
        with self.assertRaises(NotEditableError):
            colormap._setFromDict(colormap._toDict())
        state = colormap.saveState()
        with self.assertRaises(NotEditableError):
            colormap.restoreState(state)

    def testBadColorsType(self):
        """Make sure colors can't be something else than an array"""
        with self.assertRaises(TypeError):
            Colormap(name='temperature', colors=256)


class TestPreferredColormaps(unittest.TestCase):
    """Test get|setPreferredColormaps functions"""

    def setUp(self):
        # Save preferred colormaps
        self._colormaps = preferredColormaps()

    def tearDown(self):
        # Restore saved preferred colormaps
        setPreferredColormaps(self._colormaps)

    def test(self):
        colormaps = 'viridis', 'magma'

        setPreferredColormaps(colormaps)
        self.assertEqual(preferredColormaps(), colormaps)

        with self.assertRaises(ValueError):
            setPreferredColormaps(())

        with self.assertRaises(ValueError):
            setPreferredColormaps(('This is not a colormap',))

        colormaps = 'red', 'green'
        setPreferredColormaps(('This is not a colormap',) + colormaps)
        self.assertEqual(preferredColormaps(), colormaps)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestApplyColormapToData))
    test_suite.addTest(loadTests(TestRGBA))
    test_suite.addTest(loadTests(TestDictAPI))
    test_suite.addTest(loadTests(TestObjectAPI))
    test_suite.addTest(loadTests(TestPreferredColormaps))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
