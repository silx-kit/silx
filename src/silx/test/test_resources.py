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
"""Test for resource files management."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/03/2019"


import os
import unittest
import shutil
import tempfile

import silx.resources


class TestResources(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestResources, cls).setUpClass()

        cls.tmpDirectory = tempfile.mkdtemp(prefix="resource_")
        os.mkdir(os.path.join(cls.tmpDirectory, "gui"))
        destination_dir = os.path.join(cls.tmpDirectory, "gui", "icons")
        os.mkdir(destination_dir)
        source = silx.resources.resource_filename("gui/icons/zoom-in.png")
        destination = os.path.join(destination_dir, "foo.png")
        shutil.copy(source, destination)
        source = silx.resources.resource_filename("gui/icons/zoom-out.svg")
        destination = os.path.join(destination_dir, "close.png")
        shutil.copy(source, destination)

    @classmethod
    def tearDownClass(cls):
        super(TestResources, cls).tearDownClass()
        shutil.rmtree(cls.tmpDirectory)

    def setUp(self):
        # Store the original configuration
        self._oldResources = dict(silx.resources._RESOURCE_DIRECTORIES)
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        # Restiture the original configuration
        silx.resources._RESOURCE_DIRECTORIES = self._oldResources

    def test_resource_dir(self):
        """Get a resource directory"""
        icons_dirname = silx.resources.resource_filename('gui/icons/')
        self.assertTrue(os.path.isdir(icons_dirname))

    def test_resource_file(self):
        """Get a resource file name"""
        filename = silx.resources.resource_filename('gui/icons/colormap.png')
        self.assertTrue(os.path.isfile(filename))

    def test_resource_nonexistent(self):
        """Get a non existent resource"""
        filename = silx.resources.resource_filename('non_existent_file.txt')
        self.assertFalse(os.path.exists(filename))

    def test_isdir(self):
        self.assertTrue(silx.resources.is_dir('gui/icons'))

    def test_not_isdir(self):
        self.assertFalse(silx.resources.is_dir('gui/icons/colormap.png'))

    def test_list_dir(self):
        result = silx.resources.list_dir('gui/icons')
        self.assertTrue(len(result) > 10)

    # With prefixed resources

    def test_resource_dir_with_prefix(self):
        """Get a resource directory"""
        icons_dirname = silx.resources.resource_filename('silx:gui/icons/')
        self.assertTrue(os.path.isdir(icons_dirname))

    def test_resource_file_with_prefix(self):
        """Get a resource file name"""
        filename = silx.resources.resource_filename('silx:gui/icons/colormap.png')
        self.assertTrue(os.path.isfile(filename))

    def test_resource_nonexistent_with_prefix(self):
        """Get a non existent resource"""
        filename = silx.resources.resource_filename('silx:non_existent_file.txt')
        self.assertFalse(os.path.exists(filename))

    def test_isdir_with_prefix(self):
        self.assertTrue(silx.resources.is_dir('silx:gui/icons'))

    def test_not_isdir_with_prefix(self):
        self.assertFalse(silx.resources.is_dir('silx:gui/icons/colormap.png'))

    def test_list_dir_with_prefix(self):
        result = silx.resources.list_dir('silx:gui/icons')
        self.assertTrue(len(result) > 10)

    # Test new repository

    def test_repository_not_exists(self):
        """The resource from 'test' is available"""
        self.assertRaises(ValueError, silx.resources.resource_filename, 'test:foo.png')

    def test_adding_test_directory(self):
        """The resource from 'test' is available"""
        silx.resources.register_resource_directory("test", "silx.test.resources", forced_path=self.tmpDirectory)
        path = silx.resources.resource_filename('test:gui/icons/foo.png')
        self.assertTrue(os.path.exists(path))

    def test_adding_test_directory_no_override(self):
        """The resource from 'silx' is still available"""
        silx.resources.register_resource_directory("test", "silx.test.resources", forced_path=self.tmpDirectory)
        filename1 = silx.resources.resource_filename('gui/icons/close.png')
        filename2 = silx.resources.resource_filename('silx:gui/icons/close.png')
        filename3 = silx.resources.resource_filename('test:gui/icons/close.png')
        self.assertTrue(os.path.isfile(filename1))
        self.assertTrue(os.path.isfile(filename2))
        self.assertTrue(os.path.isfile(filename3))
        self.assertEqual(filename1, filename2)
        self.assertNotEqual(filename1, filename3)

    def test_adding_test_directory_non_existing(self):
        """A resource while not exists in test is not available anyway it exists
        in silx"""
        silx.resources.register_resource_directory("test", "silx.test.resources", forced_path=self.tmpDirectory)
        resource_name = "gui/icons/colormap.png"
        path = silx.resources.resource_filename('test:' + resource_name)
        path2 = silx.resources.resource_filename('silx:' + resource_name)
        self.assertFalse(os.path.exists(path))
        self.assertTrue(os.path.exists(path2))


class TestResourcesWithoutPkgResources(TestResources):

    @classmethod
    def setUpClass(cls):
        super(TestResourcesWithoutPkgResources, cls).setUpClass()
        cls._old = silx.resources.pkg_resources
        silx.resources.pkg_resources = None

    @classmethod
    def tearDownClass(cls):
        silx.resources.pkg_resources = cls._old
        del cls._old
        super(TestResourcesWithoutPkgResources, cls).tearDownClass()


class TestResourcesWithCustomDirectory(TestResources):

    @classmethod
    def setUpClass(cls):
        super(TestResourcesWithCustomDirectory, cls).setUpClass()
        cls._old = silx.resources._RESOURCES_DIR
        base = os.path.dirname(silx.resources.__file__)
        silx.resources._RESOURCES_DIR = base

    @classmethod
    def tearDownClass(cls):
        silx.resources._RESOURCES_DIR = cls._old
        del cls._old
        super(TestResourcesWithCustomDirectory, cls).tearDownClass()
