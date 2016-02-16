import unittest

from .test_qt import suite as test_qt_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_qt_suite())
    return test_suite
