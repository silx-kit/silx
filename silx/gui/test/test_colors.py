# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2020 European Synchrotron Radiation Facility
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
__date__ = "09/11/2018"

import unittest
import numpy
from silx.utils.testutils import ParametricTestCase
from silx.gui import qt
from silx.gui import colors
from silx.gui.colors import Colormap
from silx.gui.plot import items
from silx.utils.exceptions import NotEditableError


class TestColor(ParametricTestCase):
    """Basic tests of rgba function"""

    TEST_COLORS = {  # name: (colors, expected values)
        'blue': ('blue', (0., 0., 1., 1.)),
        '#010203': ('#010203', (1. / 255., 2. / 255., 3. / 255., 1.)),
        '#01020304': ('#01020304', (1. / 255., 2. / 255., 3. / 255., 4. / 255.)),
        '3 x uint8': (numpy.array((1, 255, 0), dtype=numpy.uint8),
                      (1 / 255., 1., 0., 1.)),
        '4 x uint8': (numpy.array((1, 255, 0, 1), dtype=numpy.uint8),
                      (1 / 255., 1., 0., 1 / 255.)),
        '3 x float overflow': ((3., 0.5, 1.), (1., 0.5, 1., 1.)),
    }

    def testRGBA(self):
        """"Test rgba function with accepted values"""
        for name, test in self.TEST_COLORS.items():
            color, expected = test
            with self.subTest(msg=name):
                result = colors.rgba(color)
                self.assertEqual(result, expected)

    def testQColor(self):
        """"Test getQColor function with accepted values"""
        for name, test in self.TEST_COLORS.items():
            color, expected = test
            with self.subTest(msg=name):
                result = colors.asQColor(color)
                self.assertAlmostEqual(result.redF(), expected[0], places=4)
                self.assertAlmostEqual(result.greenF(), expected[1], places=4)
                self.assertAlmostEqual(result.blueF(), expected[2], places=4)
                self.assertAlmostEqual(result.alphaF(), expected[3], places=4)


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

    def testAutoscaleFromDataReference(self):
        colormap = Colormap(name='gray', normalization='linear')
        data = numpy.array([50])
        reference = numpy.array([0, 100])
        value = colormap.applyToData(data, reference)
        self.assertEqual(len(value), 1)
        self.assertEqual(value[0, 0], 128)

    def testAutoscaleFromItemReference(self):
        colormap = Colormap(name='gray', normalization='linear')
        data = numpy.array([50])
        image = items.ImageData()
        image.setData(numpy.array([[0, 100]]))
        value = colormap.applyToData(data, reference=image)
        self.assertEqual(len(value), 1)
        self.assertEqual(value[0, 0], 128)


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
        """Make sure we can create a Colormap object from a dictionary even if
        there is missing keys except if those keys are 'colors' or 'name'
        """
        colormap = Colormap._fromDict({'name': 'blue'})
        self.assertTrue(colormap.getVMin() is None)
        colormap = Colormap._fromDict({'colors': numpy.zeros((5, 3))})
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
        colormapObject = Colormap(name=None,
                                  colors=numpy.array([[1., 0., 0.],
                                                      [0., 1., 0.],
                                                      [0., 0., 1.]]),
                                  vmin=None,
                                  vmax=None,
                                  normalization=Colormap.LOGARITHM)

        colormapObject2 = colormapObject.copy()
        self.assertTrue(colormapObject == colormapObject2)
        colormapObject.setColormapLUT([[0, 0, 0], [255, 255, 255]])
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
            colormap.setColormapLUT([[0., 0., 0.], [1., 1., 1.]])
        with self.assertRaises(NotEditableError):
            colormap._setFromDict(colormap._toDict())
        state = colormap.saveState()
        with self.assertRaises(NotEditableError):
            colormap.restoreState(state)

    def testBadColorsType(self):
        """Make sure colors can't be something else than an array"""
        with self.assertRaises(TypeError):
            Colormap(colors=256)

    def testEqual(self):
        colormap1 = Colormap()
        colormap2 = Colormap()
        self.assertEqual(colormap1, colormap2)

    def testCompareString(self):
        colormap = Colormap()
        self.assertNotEqual(colormap, "a")

    def testCompareNone(self):
        colormap = Colormap()
        self.assertNotEqual(colormap, None)

    def testSet(self):
        colormap = Colormap()
        other = Colormap(name="viridis", vmin=1, vmax=2, normalization=Colormap.LOGARITHM)
        self.assertNotEqual(colormap, other)
        colormap.setFromColormap(other)
        self.assertIsNot(colormap, other)
        self.assertEqual(colormap, other)

    def testAutoscaleMode(self):
        colormap = Colormap(autoscaleMode=Colormap.STDDEV3)
        self.assertEqual(colormap.getAutoscaleMode(), Colormap.STDDEV3)
        colormap.setAutoscaleMode(Colormap.MINMAX)
        self.assertEqual(colormap.getAutoscaleMode(), Colormap.MINMAX)

    def testStoreRestore(self):
        colormaps = [
            Colormap(name="viridis"),
            Colormap(normalization=Colormap.SQRT)
        ]
        gamma = Colormap(normalization=Colormap.GAMMA)
        gamma.setGammaNormalizationParameter(1.2)
        colormaps.append(gamma)
        for expected in colormaps:
            with self.subTest(colormap=expected):
                state = expected.saveState()
                result = Colormap()
                result.restoreState(state)
                self.assertEqual(expected, result)

    def testStorageV1(self):
        state = b'\x00\x00\x00\x10\x00C\x00o\x00l\x00o\x00r\x00m\x00a\x00p\x00\x00'\
                b'\x00\x01\x00\x00\x00\x0E\x00v\x00i\x00r\x00i\x00d\x00i\x00s\x00'\
                b'\x00\x00\x00\x06\x00?\xF0\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
                b'\x00\x06\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00'\
                b'l\x00o\x00g'
        state = qt.QByteArray(state)
        colormap = Colormap()
        colormap.restoreState(state)

        expected = Colormap(name="viridis", vmin=1, vmax=2, normalization=Colormap.LOGARITHM)
        self.assertEqual(colormap, expected)

class TestPreferredColormaps(unittest.TestCase):
    """Test get|setPreferredColormaps functions"""

    def setUp(self):
        # Save preferred colormaps
        self._colormaps = colors.preferredColormaps()

    def tearDown(self):
        # Restore saved preferred colormaps
        colors.setPreferredColormaps(self._colormaps)

    def test(self):
        colormaps = 'viridis', 'magma'

        colors.setPreferredColormaps(colormaps)
        self.assertEqual(colors.preferredColormaps(), colormaps)

        with self.assertRaises(ValueError):
            colors.setPreferredColormaps(())

        with self.assertRaises(ValueError):
            colors.setPreferredColormaps(('This is not a colormap',))

        colormaps = 'red', 'green'
        colors.setPreferredColormaps(('This is not a colormap',) + colormaps)
        self.assertEqual(colors.preferredColormaps(), colormaps)


class TestRegisteredLut(unittest.TestCase):
    """Test get|setPreferredColormaps functions"""

    def setUp(self):
        # Save preferred colormaps
        lut = numpy.arange(8 * 3)
        lut.shape = -1, 3
        lut = lut / (8.0 * 3)
        colors.registerLUT("test_8", colors=lut, cursor_color='blue')

    def testColormap(self):
        colormap = Colormap("test_8")
        self.assertIsNotNone(colormap)

    def testCursor(self):
        color = colors.cursorColorForColormap("test_8")
        self.assertEqual(color, 'blue')

    def testLut(self):
        colormap = Colormap("test_8")
        colors = colormap.getNColors(8)
        self.assertEqual(len(colors), 8)

    def testUint8(self):
        lut = numpy.array([[255, 0, 0], [200, 0, 0], [150, 0, 0]], dtype="uint")
        colors.registerLUT("test_type", lut)
        colormap = colors.Colormap(name="test_type")
        lut = colormap.getNColors(3)
        self.assertEqual(lut.shape, (3, 4))
        self.assertEqual(lut[0, 0], 255)

    def testFloatRGB(self):
        lut = numpy.array([[1.0, 0, 0], [0.5, 0, 0], [0, 0, 0]], dtype="float")
        colors.registerLUT("test_type", lut)
        colormap = colors.Colormap(name="test_type")
        lut = colormap.getNColors(3)
        self.assertEqual(lut.shape, (3, 4))
        self.assertEqual(lut[0, 0], 255)

    def testFloatRGBA(self):
        lut = numpy.array([[1.0, 0, 0, 128 / 256.0], [0.5, 0, 0, 1.0], [0.0, 0, 0, 1.0]], dtype="float")
        colors.registerLUT("test_type", lut)
        colormap = colors.Colormap(name="test_type")
        lut = colormap.getNColors(3)
        self.assertEqual(lut.shape, (3, 4))
        self.assertEqual(lut[0, 0], 255)
        self.assertEqual(lut[0, 3], 128)


class TestAutoscaleRange(ParametricTestCase):

    def testAutoscaleRange(self):
        nan = numpy.nan
        data = [
            # Positive values
            (Colormap.LINEAR, Colormap.MINMAX, numpy.array([10, 20, 50]), (10, 50)),
            (Colormap.LOGARITHM, Colormap.MINMAX, numpy.array([10, 50, 100]), (10, 100)),
            (Colormap.LINEAR, Colormap.STDDEV3, numpy.array([10, 100]), (-80, 190)),
            (Colormap.LOGARITHM, Colormap.STDDEV3, numpy.array([10, 100]), (1, 1000)),
            # With nan
            (Colormap.LINEAR, Colormap.MINMAX, numpy.array([10, 20, 50, nan]), (10, 50)),
            (Colormap.LOGARITHM, Colormap.MINMAX, numpy.array([10, 50, 100, nan]), (10, 100)),
            (Colormap.LINEAR, Colormap.STDDEV3, numpy.array([10, 100, nan]), (-80, 190)),
            (Colormap.LOGARITHM, Colormap.STDDEV3, numpy.array([10, 100, nan]), (1, 1000)),
            # With negative
            (Colormap.LOGARITHM, Colormap.MINMAX, numpy.array([10, 50, 100, -50]), (10, 100)),
            (Colormap.LOGARITHM, Colormap.STDDEV3, numpy.array([10, 100, -10]), (1, 1000)),
        ]
        for norm, mode, array, expectedRange in data:
            with self.subTest(norm=norm, mode=mode, array=array):
                colormap = Colormap()
                colormap.setNormalization(norm)
                colormap.setAutoscaleMode(mode)
                vRange = colormap._computeAutoscaleRange(array)
                if vRange is None:
                    self.assertIsNone(expectedRange)
                else:
                    self.assertAlmostEqual(vRange[0], expectedRange[0])
                    self.assertAlmostEqual(vRange[1], expectedRange[1])

def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestApplyColormapToData))
    test_suite.addTest(loadTests(TestColor))
    test_suite.addTest(loadTests(TestDictAPI))
    test_suite.addTest(loadTests(TestObjectAPI))
    test_suite.addTest(loadTests(TestPreferredColormaps))
    test_suite.addTest(loadTests(TestRegisteredLut))
    test_suite.addTest(loadTests(TestAutoscaleRange))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
