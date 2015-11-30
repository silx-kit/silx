import unittest

from .test_version import suite as test_version_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_version_suite())
    return test_suite


def run_tests():
    """Run the complete test suite."""
    test_suite = suite()
    runner = unittest.TextTestRunner()
    return runner.run(test_suite).wasSuccessful()
