import unittest

from .test_version import suite as test_version_suite
from ..gui.test import suite as test_gui_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_version_suite())
    test_suite.addTest(test_gui_suite())
    return test_suite
