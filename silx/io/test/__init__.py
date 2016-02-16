import unittest

from .test_specfile import suite as test_specfile_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_specfile_suite())
    return test_suite
