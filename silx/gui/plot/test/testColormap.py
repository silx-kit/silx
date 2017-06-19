# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
__date__ = "05/12/2016"

import unittest
from silx.gui.plot.Colormap import Colormap


class TestDictAPI(unittest.TestCase):
    """Make sure the old dictionnart API is working
    """

    def setUp(self):
        self.vmin = -1.0
        self.vmax = 12

    def testGetItem(self):
        """test the item getter API ([xxx])"""
        colormap = Colormap(name='viridis',
                            norm='linear',
                            vmin=self.vmin,
                            vmax=self.vmax)
        self.assertTrue(colormap['name'] == 'viridis')
        self.assertTrue(colormap['norm'] == 'linear')
        self.assertTrue(colormap['vmin'] == self.vmin)
        self.assertTrue(colormap['vmax'] == self.vmax)
        with self.assertRaises(KeyError):
            colormap['toto']

    def testGetDict(self):
        """Test the getDict function API"""
        clmObject = Colormap(name='viridis',
                            norm='linear',
                            vmin=self.vmin,
                            vmax=self.vmax)
        clmDict = clmObject.getDict()
        self.assertTrue(clmDict['name'] == 'viridis')
        self.assertTrue(clmDict['autoscale'] is False)
        self.assertTrue(clmDict['vmin'] == self.vmin)
        self.assertTrue(clmDict['vmax'] == self.vmax)
        self.assertTrue(clmDict['normalization'] == 'linear')

        clmObject.setColorMapRange(None, None)
        self.assertTrue(clmObject.getDict()['autoscale'] is True)

    def testSetValidDict(self):
        """Test that if a colormap is created fron a dict then it is correctly
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
        colormapObject = Colormap.getFromDict(clm_dict)
        self.assertTrue(colormapObject.getName() == clm_dict['name'])
        self.assertTrue(colormapObject.getColorMapLUT() == clm_dict['colors'])
        self.assertTrue(colormapObject.getVMin() == clm_dict['vmin'])
        self.assertTrue(colormapObject.getVMax() == clm_dict['vmax'])
        self.assertTrue(colormapObject.isAutoscale() == clm_dict['autoscale'])

        # Check that the colormap has copied the values
        clm_dict['vmin'] = None
        clm_dict['vmax'] = None
        clm_dict['colors'] = [1.0, 2.0]
        clm_dict['autoscale'] = True
        clm_dict['normalization'] = 'log'
        clm_dict['name'] = 'viridis'

        self.assertFalse(colormapObject.getName() == clm_dict['name'])
        self.assertFalse(colormapObject.getColorMapLUT() == clm_dict['colors'])
        self.assertFalse(colormapObject.getVMin() == clm_dict['vmin'])
        self.assertFalse(colormapObject.getVMax() == clm_dict['vmax'])
        self.assertFalse(colormapObject.isAutoscale() == clm_dict['autoscale'])

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
            colormapObject = Colormap.getFromDict(clm_dict)

    def testUnknowName(self):
        """Make sure an error is raised if the given name is not
        knowed
        """
        clm_dict = {
            'name': 'temperaturesTOTO',
            'vmin': 1.0,
            'vmax': 2.0,
            'normalization': 'linear',
            'colors': None,
            'autoscale': False
        }
        with self.assertRaises(ValueError):
            colormapObject = Colormap.getFromDict(clm_dict)

    def testIncoherentAutoscale(self):
        """Make sure an error is raised if the values given for vmin, vmax and
        autoscale are incoherent 
        """
        clm_dict = {
            'name': 'temperature',
            'vmin': None,
            'vmax': None,
            'normalization': 'linear',
            'colors': None,
            'autoscale': False
        }
        with self.assertRaises(ValueError):
            colormapObject = Colormap.getFromDict(clm_dict)

        clm_dict = {
            'name': 'temperature',
            'vmin': 1.0,
            'vmax': 2.0,
            'normalization': 'linear',
            'colors': None,
            'autoscale': True
        }
        with self.assertRaises(ValueError):
            colormapObject = Colormap.getFromDict(clm_dict)

    # TODO : missing code and test for dealing with negative vmin/vmax on log norm...


class TestObjectAPI(unittest.TestCase):
    """Test the new Object API of the colormap"""
    def setUp(self):
        signalHasBeenEmitting = False

    def testVminVMax(self):
        """Test getter and setter associated to vmin and vmax values"""
        vmin = 1.0
        vmax = 2.0

        colormapObject = Colormap(name='viridis',
                                  vmin=vmin,
                                  vmax=vmax,
                                  norm='linear')

        self.assertTrue(colormapObject.getColorMapRange() == (1.0, 2.0))
        self.assertTrue(colormapObject.isAutoscale() is False)
        colormapObject.setColorMapRange(None, None)
        self.assertTrue(colormapObject.getVMin() == None)
        self.assertTrue(colormapObject.getVMax() == None)
        self.assertTrue(colormapObject.isAutoscale() is True)

    def testCopy(self):
        colormapObject = Colormap(name='toto',
                                  colors=[12, 13, 14],
                                  vmin=None,
                                  vmax=None,
                                  norm='log')

        colormapObject2 = colormapObject.copy()
        self.assertTrue(colormapObject.getDict() == colormapObject2.getDict())
        colormapObject.setColorMapLUT([0, 1])
        self.assertFalse(colormapObject.getDict() == colormapObject2.getDict())

        colormapObject2 = colormapObject.copy()
        self.assertTrue(colormapObject.getDict() == colormapObject2.getDict())
        colormapObject.setNorm('linear')
        self.assertFalse(colormapObject.getDict() == colormapObject2.getDict())


def suite():
    test_suite = unittest.TestSuite()
    for ui in (TestDictAPI, TestObjectAPI):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(ui))

    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
