import unittest

from .test_version import suite as test_version_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_version_suite())
    return test_suite
